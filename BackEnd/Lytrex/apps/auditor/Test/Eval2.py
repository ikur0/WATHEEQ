import os
import time
import json
import glob
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi 

# LLM Imports for Judge System
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================
def get_int_input(prompt: str, default: int) -> int:
    val = input(prompt).strip()
    if not val: return default
    try:
        return int(val)
    except ValueError:
        print(f"  [!] Invalid input. Using default: {default}")
        return default

def get_bool_input(prompt: str) -> bool:
    val = input(prompt).strip().lower()
    return val in ['y', 'yes', 'true', '1']

def _tokenize(text: str) -> list:
    return text.lower().replace("\n", " ").split(" ")

# =========================================================================
# MODULE 1: INGESTION & PARENT MAPPING (SMALL-TO-BIG)
# =========================================================================
def load_and_chunk_framework(framework_name: str, chunk_size: int, chunk_overlap: int):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fw_dir = os.path.join(base_dir, "frameworks", framework_name.upper())
    
    json_files = glob.glob(os.path.join(fw_dir, "*.json"))
    if not json_files:
        print(f"[!] Error: Could not find any JSON files in {fw_dir}")
        return [], {}
        
    json_path = json_files[0]
    print(f"\n[*] Loading Data from {os.path.basename(json_path)}...")

    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    sections = raw_data.get("sections", raw_data) if isinstance(raw_data, dict) else raw_data
    docs = []
    parent_map = {} 
    
    for sec in sections:
        if not isinstance(sec, dict): continue
        text = sec.get("text", "")
        if not text.strip(): continue
        
        control_id = str(sec.get("section_id", sec.get("control_id", sec.get("section", "Unknown")))).upper()
        parent_map[control_id] = text 
        
        doc = Document(
            page_content=text,
            metadata={
                "control_id": control_id,
                "domain": str(sec.get("title", sec.get("domain", "Unknown"))),
                "section": str(sec.get("section", ""))
            }
        )
        docs.append(doc)

    print(f"[*] Splitting text (Size: {chunk_size}, Overlap: {chunk_overlap})...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"[+] Created {len(chunks)} total chunks.")
    
    return chunks, parent_map

# =========================================================================
# MODULE 2: HYBRID INDEXING & RERANKER
# =========================================================================
def build_hybrid_indexes(chunks: list, model_name: str):
    print(f"\n[*] Initializing Embeddings: {model_name}...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    print("[*] Building Dense Vector Space (FAISS)...")
    start_time = time.time()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"  [+] FAISS Index built in {time.time() - start_time:.2f} seconds.")
    
    print("[*] Building Sparse Keyword Space (BM25Okapi)...")
    start_time = time.time()
    all_docs = chunks
    tokenized_corpus = [_tokenize(doc.page_content) for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"  [+] BM25 Index built in {time.time() - start_time:.2f} seconds.")
    
    print("[*] Initializing Cross-Encoder (BAAI/bge-reranker-base)...")
    start_time = time.time()
    cross_encoder = CrossEncoder('BAAI/bge-reranker-base')
    print(f"  [+] Cross-Encoder initialized in {time.time() - start_time:.2f} seconds.")
    
    return vectorstore, bm25, all_docs, cross_encoder

# =========================================================================
# MODULE 3: AUTOMATED METRICS BENCHMARK WITH HYBRID PIPELINE
# =========================================================================
def run_automated_benchmark(vectorstore, bm25, all_docs, cross_encoder, framework_name, parent_map):
    print("\n" + "="*60)
    print(f" AUTOMATED METRICS BENCHMARK ({framework_name.upper()}) ")
    print("="*60)
    
    all_benchmarks = {
        "NCA": [
            ("All internal databases classified as ADVANCED strength must utilize the SHA2-512 algorithm for data hashing.", ["SEC_0004"], "The policy is compliant for ADVANCED strength, but SHA2-512 is only accepted for MODERATE. For ADVANCED, SHA3-512 must be used."),
            ("For IoT devices, the engineering team has approved the use of the PRESENT block cipher with an 80-bit key length.", ["SEC_0005"], "This is compliant. PRESENT with an 80-bit or 128-bit key length is accepted for lightweight crypto algorithms."),
            ("Network traffic encryption will utilize the AES algorithm running in Cipher Block Chaining (CBC) mode across advanced environments.", ["SEC_0006"], "This is a violation. Cipher Block Chaining (CBC) is only accepted for MODERATE environments, not ADVANCED."),
            ("Message authentication across the corporate API gateway is handled via CMAC, rotating keys after 2^50 messages.", ["SEC_0007"], "This is a violation. CMAC should be used for at most 2^48 messages, not 2^50.")
        ],
        "SAMA": [
            ("The Chief Information Security Officer will allocate the annual cybersecurity budget.", ["SEC_0041"], "This is a violation. The Board of Directors has the ultimate responsibility for ensuring sufficient budget is allocated, not the CISO."),
            ("We provide phishing simulations for internal employees, but do not extend threat awareness to our retail customers.", ["SEC_0043"], "This is a violation. Customer awareness should address retail and commercial customers, including suggested cybersecurity mechanisms to mitigate risks."),
            ("Upon termination of an employee, their physical access badge must be surrendered and network access revoked.", ["SEC_0054"], "This is compliant. Post-employment activities require revoking access rights and returning information assets assigned.")
        ],
        "ECC": [
            ("The internal cybersecurity team conducts an annual self-audit of all implemented security controls.", ["SEC_0006"], "This is a violation. The implementation of controls must be reviewed and audited independently by parties OTHER than the cybersecurity department."),
            ("All newly hired system administrators must sign a Non-Disclosure Agreement prior to onboarding.", ["SEC_0007"], "This is compliant. Cybersecurity responsibilities and non-disclosure clauses must be incorporated in employment contracts.")
        ]
    }
    
    benchmark_data = all_benchmarks.get(framework_name.upper(), [])
    if not benchmark_data:
        print(f"\n[!] No benchmark tests defined for {framework_name.upper()}.")
        return

    k_val = get_int_input("Enter number of Top-K candidates to rerank [Default 3]: ", 3)
    
    # --- NEW GRANULAR PROMPT SYSTEM ---
    run_llm_gen = get_bool_input("Do you want to generate LLM audit responses? (y/n): ")
    run_llm_judge = False
    
    if run_llm_gen:
        run_llm_judge = get_bool_input("Do you want to evaluate them using LLM-as-a-Judge? (y/n): ")
        
    llm, parser = None, None
    if run_llm_gen:
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            print("[!] GROQ_API_KEY missing from .env. Skipping LLM evaluation.")
            run_llm_gen = False
            run_llm_judge = False
        else:
            print("[*] Initializing Llama-3 (Groq) for generation...")
            llm = ChatGroq(temperature=0, groq_api_key=groq_key, model_name="llama-3.3-70b-versatile")
            
            generator_prompt = ChatPromptTemplate.from_template(
                "You are a strict compliance auditor. Based ONLY on the following regulatory context, evaluate the user's policy query.\n"
                "State clearly if it is compliant or a violation, and explain why.\n\nContext:\n{context}\n\nQuery:\n{query}"
            )
            
            if run_llm_judge:
                judge_prompt = ChatPromptTemplate.from_template(
                    "You are an impartial AI Judge evaluating a batch of compliance audits.\n"
                    "Review the following JSON array of test cases. Each case contains a 'query', a 'reference_answer', and the 'actual_answer' generated by the AI auditor.\n"
                    "For each case, score the 'actual_answer' from 0 to 10 based on how accurately its findings and reasoning align with the 'reference_answer'.\n\n"
                    "Batch Data:\n{batch_data}\n\n"
                    "Respond ONLY with a valid JSON array of objects, keeping the exact same order. Use this exact schema:\n"
                    "[{{\"query_index\": 1, \"reasoning\": \"brief explanation of the score\", \"score\": 0}}, ...]"
                )
                parser = JsonOutputParser()

    hits = 0
    mrr_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    queries_run = len(benchmark_data)
    final_report_data = []

    print(f"\n[*] Running {queries_run} isolated queries through the Lytrex Pipeline...")
    
    for i, (query, expected_ids, expected_answer) in enumerate(benchmark_data, 1):
        print(f"\n--- [{i}/{queries_run}] Testing Query ---")
        print(f"Query: '{query[:60]}...'")
        
        # 1. FAISS + BM25Okapi Retrieval
        fetch_k = k_val * 2
        chunk_faiss = vectorstore.similarity_search(query, k=fetch_k)
        
        tokenized_query = _tokenize(query)
        chunk_bm25 = bm25.get_top_n(tokenized_query, all_docs, n=fetch_k)
        
        combined_children = chunk_faiss + chunk_bm25
        
        # 2. Merge all candidates & Deduplicate by chunk text
        unique_children_map = {}
        for child in combined_children:
            if child.page_content not in unique_children_map:
                unique_children_map[child.page_content] = child
                
        unique_children = list(unique_children_map.values())
        
        # 3. Cross-Encoder (Query vs Candidate)
        pairs = [[query, child.page_content] for child in unique_children]
        scores = cross_encoder.predict(pairs)
        
        # 4. Top-K Results
        scored_children = sorted(zip(scores, unique_children), key=lambda x: x[0], reverse=True)
        top_k_children = [child for score, child in scored_children[:k_val]]
        
        # 5. Context Assembly (Parent chunks) & Final Deduplication
        unique_parents = []
        assembled_context = ""
        
        for child in top_k_children:
            parent_id = str(child.metadata.get("control_id", "Unknown")).upper()
            if parent_id not in unique_parents:
                unique_parents.append(parent_id)
                if parent_id in parent_map:
                    assembled_context += f"\n--- Section: {parent_id} ---\n{parent_map[parent_id]}\n"

        # 6. Metrics Calculation (Strictly based on assembled parents passed to LLM)
        hit_found = False
        reciprocal_rank = 0.0
        relevant_retrieved = 0
        found_targets = set()
        
        for rank, parent_id in enumerate(unique_parents, 1):
            is_relevant = False
            for target in expected_ids:
                if target.lower() in parent_id.lower():
                    is_relevant = True
                    found_targets.add(target)
                    break
                    
            if is_relevant:
                relevant_retrieved += 1
                if not hit_found:
                    hit_found = True
                    reciprocal_rank = 1.0 / rank
                    hits += 1

        mrr_sum += reciprocal_rank
        # Precision uses the length of unique parents actually assembled
        actual_k = len(unique_parents) if unique_parents else 1 
        precision_at_k = relevant_retrieved / actual_k
        precision_sum += precision_at_k
        
        recall_at_k = len(found_targets) / len(expected_ids)
        recall_sum += recall_at_k
        
        status = "✅ HIT" if hit_found else "❌ MISS"
        target_display = ", ".join(expected_ids)
        
        print(f"  Target(s): [{target_display}] | Status: {status}")
        print(f"  Final Assembled Parents : [{', '.join(unique_parents)}]")
        print(f"  Precision@Final: {precision_at_k:.2f} | Recall: {recall_at_k:.2f} | MRR: {reciprocal_rank:.4f}")

        # 7. LLM Micro-Evaluation
        actual_answer = "N/A (Skipped LLM Generation)"
        if run_llm_gen:
            if hit_found:
                print("  [*] Generating LLM audit response from assembled parents...")
                try:
                    gen_chain = generator_prompt | llm
                    actual_answer = gen_chain.invoke({"context": assembled_context, "query": query}).content
                except Exception as e:
                    actual_answer = f"[!] Generation Error: {e}"
            else:
                actual_answer = "[!] Generation Skipped: Target context was not retrieved."
                
            print(f"  LLM Response: {actual_answer}")

        # Store JSON Result for batch logic
        final_report_data.append({
            "query_index": i,
            "query": query,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "hit": hit_found
        })

    # =========================================================================
    # PHASE 8: BATCH JUDGE CALL OR FINAL DUMP
    # =========================================================================
    judge_scores = []
    
    if run_llm_judge:
        print("\n" + "="*80)
        print(" FINAL LLM JUDGE REPORT (1 BATCH CALL) ")
        print("="*80)
        
        batch_payload = []
        for data in final_report_data:
            if data['hit'] and "Skipped" not in data['actual_answer'] and "Error" not in data['actual_answer']:
                batch_payload.append({
                    "query_index": data["query_index"],
                    "query": data["query"],
                    "reference_answer": data["expected_answer"],
                    "actual_answer": data["actual_answer"]
                })
        
        judge_results_map = {}
        
        if batch_payload:
            print("[*] Sending all answers to LLM Judge in a single API call...")
            try:
                judge_chain = judge_prompt | llm | parser
                batch_json_string = json.dumps(batch_payload, indent=2)
                judgments = judge_chain.invoke({"batch_data": batch_json_string})
                
                if isinstance(judgments, list):
                    for j in judgments:
                        judge_results_map[j.get("query_index")] = j
            except Exception as e:
                print(f"[!] Batch Judge Evaluation Error: {e}")

        for data in final_report_data:
            idx = data['query_index']
            print(f"\n--- QUERY {idx} ---")
            print(f"Reference Answer : {data['expected_answer']}")
            print(f"Lytrex LLM Answer: {data['actual_answer']}")
            
            if idx in judge_results_map:
                res = judge_results_map[idx]
                score = res.get("score", 0)
                judge_scores.append(score)
                print(f"\n>> JUDGE SCORE: {score}/10")
                print(f">> REASONING  : {res.get('reasoning', 'N/A')}")
            else:
                print("\n>> JUDGE SCORE: 0/10 (Retrieval Failed / Generation Error)")
                judge_scores.append(0)
                
            print("-" * 80)
            
    elif run_llm_gen:
        # If generation was ON, but Judge was OFF, print a clean summary dump
        print("\n" + "="*80)
        print(" FINAL LLM AUDIT REPORT (GENERATION ONLY) ")
        print("="*80)
        for data in final_report_data:
            idx = data['query_index']
            print(f"\n--- QUERY {idx} ---")
            print(f"Reference Answer : {data['expected_answer']}")
            print(f"Lytrex LLM Answer: {data['actual_answer']}")
            print("-" * 80)

    # Calculate Global Averages
    avg_hit_rate = (hits / queries_run) * 100
    avg_recall = (recall_sum / queries_run) * 100
    avg_mrr = mrr_sum / queries_run
    avg_precision = (precision_sum / queries_run) * 100
    avg_judge = (sum(judge_scores) / len(judge_scores)) if judge_scores else 0

    print("\n" + "="*60)
    print(" SYSTEM AVERAGE METRICS OVERVIEW ")
    print("="*60)
    print(f"  Framework Tested    : {framework_name.upper()}")
    print(f"  Total Valid Queries : {queries_run}")
    print(f"  Avg Hit Rate (Binary)   : {avg_hit_rate:.1f}%")
    print(f"  Avg Precision           : {avg_precision:.1f}% (Based on Final Unique Parents)")
    print(f"  Avg Recall              : {avg_recall:.1f}%")
    print(f"  Mean Reciprocal Rank    : {avg_mrr:.4f}")
    if run_llm_judge:
        print(f"  Average LLM Score       : {avg_judge:.1f}/10")
    print("="*60)

# =========================================================================
# MODULE 4: INTERACTIVE RETRIEVAL SANDBOX (HYBRID)
# =========================================================================
def evaluate_retrieval(vectorstore, bm25, all_docs, cross_encoder, parent_map):
    print("\n" + "="*50)
    print(" INTERACTIVE RETRIEVAL SANDBOX (LYTREX PIPELINE) ")
    print("="*50)
    print("Type 'exit' or 'q' to return to the main menu.")
    
    while True:
        print("\n" + "-"*50)
        query = input("Enter your query: ").strip()
        
        if query.lower() in ['exit', 'q', 'quit', 'back']:
            break
        if not query:
            continue
            
        k_val = get_int_input("Enter number of Top-K candidates to rerank [Default 3]: ", 3)
        fetch_k = k_val * 2
        
        start_search = time.time()
        
        chunk_faiss = vectorstore.similarity_search(query, k=fetch_k)
        tokenized_query = _tokenize(query)
        chunk_bm25 = bm25.get_top_n(tokenized_query, all_docs, n=fetch_k)
        
        unique_children_map = {}
        for child in chunk_faiss + chunk_bm25:
            if child.page_content not in unique_children_map:
                unique_children_map[child.page_content] = child
                
        unique_children = list(unique_children_map.values())
        cross_inp = [[query, child.page_content] for child in unique_children]
        scores = cross_encoder.predict(cross_inp)
        
        scored_children = sorted(zip(scores, unique_children), key=lambda x: x[0], reverse=True)
        top_k_children = [child for score, child in scored_children[:k_val]]
        
        unique_parents = []
        for child in top_k_children:
            parent_id = str(child.metadata.get("control_id", "Unknown")).upper()
            if parent_id not in unique_parents:
                unique_parents.append(parent_id)

        search_time = time.time() - start_search
        
        print(f"\n[*] Hybrid Search completed in {search_time:.4f} seconds.")
        print(f"[*] Deduped Final Parents Triggered: {', '.join(unique_parents)}")
        
        for parent_id in unique_parents:
            if parent_id in parent_map:
                clean_text = parent_map[parent_id].replace('\n', ' ')
                preview = clean_text[:250] + "..." if len(clean_text) > 250 else clean_text
                print(f"\n  Source: [{parent_id}]")
                print(f"  Parent Text: {preview}")

# =========================================================================
# MODULE 5: T-SNE VISUALIZATION
# =========================================================================
def visualize_tsne(vectorstore, dimensions=3):
    print(f"\n[*] Extracting vectors for {dimensions}D visualization...")
    
    num_vectors = vectorstore.index.ntotal
    raw_vectors = vectorstore.index.reconstruct_n(0, num_vectors)
    docstore_dict = vectorstore.docstore._dict
    index_to_id = vectorstore.index_to_docstore_id
    
    hover_texts, parent_texts = [], []
    
    for i in range(num_vectors):
        doc = docstore_dict[index_to_id[i]]
        clean_text = doc.page_content.replace("\n", " ")
        hover_texts.append(clean_text[:100] + "..." if len(clean_text) > 100 else clean_text)
        
        control_id = doc.metadata.get("control_id", "Unknown")
        domain = doc.metadata.get("domain", "Framework")
        parent_label = f"[{control_id}] {domain}"
        parent_texts.append(parent_label[:35] + "..." if len(parent_label) > 35 else parent_label)

    X = np.array(raw_vectors)
    safe_perplexity = min(30, max(5, X.shape[0] - 1)) 
    
    print(f"[*] Compressing {X.shape[1]} dimensions down to {dimensions}D using t-SNE...")
    tsne = TSNE(n_components=dimensions, perplexity=safe_perplexity, random_state=42, init='pca', learning_rate='auto')
    X_reduced = tsne.fit_transform(X)

    df = pd.DataFrame({'Category': parent_texts, 'Hover Text': hover_texts})
    
    if dimensions == 3:
        df['X'], df['Y'], df['Z'] = X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2]
        fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='Category', hover_data={'X': False, 'Y': False, 'Z': False, 'Category': True, 'Hover Text': True}, template="plotly_dark")
        fig.update_layout(scene=dict(xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False), zaxis=dict(showticklabels=False)))
    else:
        df['X'], df['Y'] = X_reduced[:, 0], X_reduced[:, 1]
        fig = px.scatter(df, x='X', y='Y', color='Category', hover_data={'X': False, 'Y': False, 'Category': True, 'Hover Text': True}, template="plotly_dark")
        fig.update_layout(xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))

    fig.update_traces(marker=dict(size=5, opacity=0.8, line=dict(width=0)))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=False)
    
    print("[+] Opening interactive visualization in your default browser...")
    fig.show()

# =========================================================================
# MODULE 6: CLI INTERFACE
# =========================================================================
def main_menu():
    current_vectorstore = None
    current_bm25 = None
    current_all_docs = None
    current_cross_encoder = None
    current_parent_map = None
    
    while True:
        if current_vectorstore is None:
            print("\n" + "="*50)
            print(" LYTREX END-TO-END EVALUATION HARNESS ")
            print("="*50)
            
            framework = input("Enter Framework to load (e.g., NCA, ECC, SAMA, CCC): ").strip().upper() or "NCA"
            
            print("\n--- Document Slicing Setup ---")
            chunk_size = get_int_input("Enter chunk size [Default 500]: ", 500)
            chunk_overlap = get_int_input("Enter chunk overlap [Default 100]: ", 100)
            
            print("\n--- Embedding Model Setup ---")
            print("1. sentence-transformers/all-MiniLM-L6-v2 (Fast/Light)")
            print("2. BAAI/bge-large-en-v1.5 (Heavy/Accurate)")
            print("3. BAAI/bge-small-en-v1.5 (Balanced)")
            choice = input("Select model (1-3) [Default 1]: ").strip()
            
            if choice == "2": model_name = "BAAI/bge-large-en-v1.5"
            elif choice == "3": model_name = "BAAI/bge-small-en-v1.5"
            else: model_name = "sentence-transformers/all-MiniLM-L6-v2"
            
            chunks, parent_map = load_and_chunk_framework(framework, chunk_size, chunk_overlap)
            if chunks:
                current_vectorstore, current_bm25, current_all_docs, current_cross_encoder = build_hybrid_indexes(chunks, model_name)
                current_parent_map = parent_map
            else:
                continue

        print("\n" + "="*50)
        print(" ACTION MENU ")
        print("=" * 50)
        print("1. Run Automated End-to-End Benchmark (Hybrid RAGAS)")
        print("2. Run Custom Retrieval Tests (Hybrid Sandbox)")
        print("3. Display Vector Space (2D t-SNE Plot)")
        print("4. Display Vector Space (3D t-SNE Plot)")
        print("5. Reset & Load Different Framework / Setup")
        print("6. Exit")
        
        action = input("\nSelect action (1-6): ").strip()
        
        if action == "1":
            run_automated_benchmark(current_vectorstore, current_bm25, current_all_docs, current_cross_encoder, framework, current_parent_map)
        elif action == "2":
            evaluate_retrieval(current_vectorstore, current_bm25, current_all_docs, current_cross_encoder, current_parent_map)
        elif action == "3":
            visualize_tsne(current_vectorstore, dimensions=2)
        elif action == "4":
            visualize_tsne(current_vectorstore, dimensions=3)
        elif action == "5":
            print("[*] Flushing memory...")
            current_vectorstore = None
            current_bm25 = None
            current_all_docs = None
            current_cross_encoder = None
            current_parent_map = None
        elif action == "6":
            print("\nExiting Lytrex Harness. Goodbye!")
            break
        else:
            print("[!] Invalid selection. Please choose a number from 1 to 6.")

if __name__ == "__main__":
    main_menu()