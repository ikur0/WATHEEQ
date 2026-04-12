# Lytrex

Lytrex is an end-to-end intelligent system designed to automate and scale cybersecurity compliance auditing across enterprise documents.

It leverages advanced **Retrieval-Augmented Generation (RAG)**, combined with hierarchical document processing and a multi-stage reasoning pipeline, to evaluate organizational policies against regulatory frameworks such as **NCA**, **SAMA**, and **ECC**.

---

## System Overview

Lytrex is a structured AI auditing engine that:

- Processes large enterprise documents (50+ pages)
- Retrieves relevant regulatory controls with high precision
- Performs deep contextual analysis and reasoning
- Generates structured, traceable audit reports
- Produces a final compliance score with actionable recommendations

---

## Core System Capabilities

### 1. Intelligent Compliance Mapping

- Maps company policies to specific regulatory controls
- Uses semantic understanding rather than keyword matching

### 2. Scalable Document Processing

- Handles large documents using a **Map–Reduce** architecture
- Ensures full document coverage while preserving context

### 3. High-Precision Retrieval

Two-stage retrieval pipeline:

- **Dense retrieval** (FAISS)
- **Cross-encoder re-ranking** (BGE)

Achieves a strong balance between recall and precision.

### 4. Advanced Reasoning Engine

Performs:

- Policy validation
- Violation detection
- Recommendation generation

Ensures consistent and deterministic analysis.

### 5. Structured and Verifiable Outputs

- Produces strictly formatted **JSON outputs**
- Enforces traceability through explicit references:
  - Page number
  - Section number

---

## System Architecture (High-Level)

Lytrex operates through a layered architecture:

| Layer | Description |
|---|---|
| **1. Document Ingestion Layer** | Processes compliance frameworks and company documents |
| **2. Retrieval Layer** | Embedding generation, vector search (FAISS), and re-ranking |
| **3. Reasoning Layer** | Contextual evaluation and audit generation |
| **4. Aggregation Layer** | Combines page-level audit reports into a final consolidated result |

---

## Why Lytrex

Traditional compliance auditing is:

- Manual
- Time-consuming
- Inconsistent

Lytrex transforms this process into something that is:

- **Automated**
- **Scalable**
- **Consistent**
- **Explainable**

---

## Key Design Principle

> Accurate compliance auditing requires both **retrieval precision** and **reasoning depth**.

Lytrex achieves this through:

- **Dense retrieval** — for broad coverage
- **Re-ranking** — for contextual precision
- **Hierarchical chunking** — for context preservation
- **Map–Reduce architecture** — for scalability

# Transition to Technical Details
- The following sections describe the full technical architecture and implementation details of the Lytrex system.


# 1. Core AI Models

- **LLM (Reasoning Engine):** gpt-4o (OpenAI)  
  Configured with `temperature = 0` to ensure deterministic outputs and eliminate hallucinations, enabling precise, rule-based compliance analysis.

- **Embedding Model (Semantic Encoder):** text-embedding-3-large (OpenAI)  
  Transforms regulatory frameworks into high-dimensional (3,072-d) vector representations for semantic retrieval.

- **Re-ranking Model (Relevance Scorer):** BAAI/bge-reranker-base  
  A local Cross-Encoder that evaluates query–document pairs to refine retrieval based on deep contextual relevance.

---

# 2. Retrieval Pipeline (Two-Stage Architecture)

- **Vector Database:** FAISS (Facebook AI Similarity Search)  
  Fully local vector store enabling fast similarity search without external API calls.

- **Stage 1 — Dense Retrieval:**  
  Performs approximate nearest-neighbor search over embeddings to retrieve **Top 20 candidates (k × 5)** based on vector similarity.

- **Stage 2 — Cross-Encoder Re-ranking:**  
  Applies the BGE re-ranker to score retrieved candidates against the query context, selecting the **Top 4 (k)** most relevant results.

---

# 3. Hierarchical Document Processing Strategy

To mitigate the *“Lost in the Middle”* problem, the system uses **multi-level chunking** via RecursiveCharacterTextSplitter:

- **Child Chunks (~800 chars):**  
  Optimized for high-precision vector retrieval (used in FAISS).

- **Parent Chunks (~8,000 chars):**  
  Full-context framework sections passed to the LLM.  
  Attached via metadata (`doc.metadata["parent_content"]`) to preserve semantic completeness.

- **Map Chunks (~4,000 chars):**  
  Segments of the target company document (≈1 page each), enabling focused and thorough analysis.

---

# 4. Auditing Architecture (Map–Reduce Pipeline)

Designed to scale across large enterprise documents (50+ pages):

- **MAP Phase (Page-Level Analysis):**  
  - Iterates over document chunks (≈1 page each).  
  - Retrieves relevant framework context per chunk.  
  - gpt-4o generates structured **JSON-based mini audit reports** per page.

- **REDUCE Phase (Global Synthesis):**  
  - Aggregates all page-level reports.  
  - Reprocesses them using gpt-4o as a *Chief Auditor*.  
  - Deduplicates findings and produces a consolidated **final compliance score**.

---

# 5. Output Engineering & Reliability

- **Structured Output Enforcement:**  
  JsonOutputParser (LangChain) ensures strictly valid JSON responses, eliminating free-form text.

- **Traceability Mechanism:**  
  Prompts enforce explicit citation of evidence using `[Page X, Section Y]`, ensuring all findings are verifiable and grounded.

---

# 6. End-to-End Workflow (Highly Detailed Example)

**Input:**  
A 60-page company cybersecurity policy document (PDF) + NCA compliance framework.

---

## Step 1 — Preprocessing

- The framework PDF is split into:
  - 800-char child chunks → embedded using text-embedding-3-large  
  - Stored in FAISS  
- Each child chunk is linked to its 8,000-char parent via metadata.

---

## Step 2 — Company Document Chunking

- The 60-page document is split into ~15 chunks (≈4,000 chars each).  
- Each chunk represents ~1 page or logical section.

---

## Step 3 — MAP Phase (Per Chunk Execution)

**Example: Chunk #7 (Access Control Policy)**

1. **Query Construction:**  
   It takes the entire 4,000-character Map Chunk, embeds the whole thing as one giant dense vector, and uses that massive vector to search FAISS. This is a strength of text-embedding-3-large—it can hold the semantic weight of a whole page at once.

2. **Stage 1 Retrieval (FAISS):**  
   Retrieve Top 20 candidate controls (vector similarity)

3. **Stage 2 Re-ranking:**  
   BGE re-ranker scores each pair  
   Select Top 4 most relevant controls

4. **Context Expansion:**  
   Replace each retrieved child chunk with its full parent (~8,000 chars)

5. **LLM Evaluation:**  
   Pass:
   - Company chunk (Chunk #7)  
   - Retrieved framework controls  
   Into gpt-4o

6. **Output (Mini Report):**

{
    "internal_audit_reasoning": "Step-by-step logic. I checked [Page 7, Section 2.1] regarding user access. The retrieved framework control AC-03 requires the principle of least privilege. The company document explicitly states that all employees receive global admin rights by default, which is a direct contradiction. Found violation.",
    "compliance_score": 75,
    "executive_summary": "Section 2.1 outlines the company's access control policy. While basic authentication is required, the policy contains a critical flaw by granting excessive administrative rights to standard users, directly violating least-privilege mandates.",
    "compliant_areas": [
        "[Page 7, Section 2.1] Successfully requires multi-factor authentication for all initial system logins."
    ],
    "violations": [
        "[Page 7, Section 2.1] Contradicts framework Control AC-03 by failing to enforce the principle of least privilege for standard employees (-25 pts)."
    ],
    "recommendations": [
        "[Page 7, Section 2.1] Rewrite the policy to explicitly mandate Role-Based Access Control (RBAC) and ensure users are granted only the minimum permissions required for their roles."
    ]
}

---

## Step 4 — Iterate Across All Chunks

- Repeat Step 3 for all ~15 chunks  
- Generate 15 independent audit reports

---

## Step 5 — REDUCE Phase (Final Aggregation)

1. Combine all mini-reports into one array:

[report_1, report_2, ..., report_15]

2. Pass into gpt-4o with aggregation prompt:
   - Remove duplicate violations  
   - Merge similar findings  
   - Compute final compliance score  

---

## Step 6 — Final Output

{
  "final_compliance_score": 78,
  "master_executive_summary": "The organization demonstrates a baseline commitment to cybersecurity, particularly in mandatory multi-factor authentication and data encryption at rest. However, there are systemic vulnerabilities in the Access Control and Incident Response domains. Specifically, the failure to enforce the principle of least privilege across standard employee accounts presents a high-risk contradiction to NCA framework requirements.",
  "all_compliant_areas": [
    "[Page 7, Section 2.1] Successfully requires multi-factor authentication for all initial system logins.",
    "[Page 22, Section 4.3] Data at rest is encrypted using AES-256 standard.",
    "[Page 45, Section 8.1] Employee security awareness training is mandated annually."
  ],
  "all_unique_violations": [
    "[Page 7, Section 2.1] Contradicts framework Control AC-03 by failing to enforce the principle of least privilege for standard employees (-15 pts).",
    "[Page 31, Section 5.4] No defined maximum timeframe for reporting severe security incidents to the relevant authorities, violating IR-02 (-7 pts)."
  ],
  "master_recommendations": [
    "Overhaul the Access Control policy on Page 7 to explicitly mandate Role-Based Access Control (RBAC).",
    "Update the Incident Response playbook on Page 31 to include a strict 24-hour reporting window for critical breaches."
  ]
}

---

# Why This Works (Key Insight)

- **Dense retrieval = recall (broad search)**  
- **Cross-encoder = precision (deep reasoning)**  
- **Parent chunking = context preservation**  
- **Map–Reduce = scalability across large documents**  


# 7. Experimental Directions

- **Hybrid Retrieval (BM25 + Dense):**  
  Combine FAISS with BM25 to improve recall (semantic + keyword matching).
  - Add a sparse retriever like BM25 alongside FAISS.
  - BM25 searches for exact keyword matches, while FAISS searches for conceptual meaning.
  - Use LangChain’s EnsembleRetriever to merge the results of both before passing them to your BGE Re-ranker. This guarantees you never miss a rule just because the AI didn't map the numbers correctly.

- **Model Exploration:**  
  Test different embeddings (e.g., nomic, bge, e5) and LLMs (GPT, Qwen, LLaMA) for accuracy vs cost.

- **Chunk Tuning:**  
  Experiment with:
  - parent_chunk_size: 6000–10000  
  - child_chunk_size: 400–1200  
  - map_chunk_size: 3000–6000  
  Optimize for balance between context and precision.

- **Re-ranking Improvements:**  
  Try stronger models (bge-large, monoT5) and increase candidate pool (Top 20 → 30+).

- **Evaluation:**  
  Track Recall@k, precision, and consistency vs human audit.

- **Advanced Ideas:**  
  Query expansion, hierarchical retrieval, caching previous violations.