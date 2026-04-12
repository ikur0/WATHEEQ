import os
import glob
import json
from rag2 import ComplianceRAG 

def main():
    print("Initializing Lytrex Enterprise Map-Reduce RAG... Please wait.")
    rag = ComplianceRAG()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files_dir = os.path.join(base_dir, "files")
    os.makedirs(files_dir, exist_ok=True)

    while True:
        print("\n" + "="*60)
        print(" LYTREX COMPLIANCE AUDITOR - ENTERPRISE CONSOLE")
        print("="*60)
        print("(1) Audit short Text Snippet")
        print("(2) Audit full PDF Document (Map-Reduce)")
        print("(3) Exit System")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()

        if choice == '3':
            print("Exiting Lytrex Console. Goodbye!")
            break

        if choice not in ['1', '2']:
            print("Invalid choice. Please try again.")
            continue

        print("\nSelect Framework to assess against (NCA, ECC, SAMA, or ALL):")
        framework = input("Framework: ").strip().upper() or "ALL"

        # ---------------------------------------------
        # OPTION 1: TEXT INPUT (Standard Flow)
        # ---------------------------------------------
        if choice == '1':
            text_input = input("\nEnter the text you want to test:\n> ")
            if not text_input.strip(): continue
                
            # For short text, we just retrieve and ask if they want to run LLM
            # (Requires you to keep check_compliance_text from the previous iteration in rag.py if you want to preview)
            # Since we transitioned to Enterprise mode, let's just run it directly for simplicity.
            print("\nExecuting Standard Audit...")
            # Using the MAP prompt directly for short text
            vectorstore = rag._load_fw_vectorstore(framework)
            if not vectorstore: vectorstore = rag.ingest_single_framework(framework)
            
            results = vectorstore.similarity_search(text_input)
            unique = list(dict.fromkeys([r.metadata.get("parent_content", r.page_content) for r in results]))[:5]
            context = "\n\n---\n\n".join(unique)
            
            result = rag.evaluate_with_llm(context, text_input)
            print("\n" + "="*50)
            print("FINAL AUDIT RESPONSE:")
            print("="*50)
            print(json.dumps(result, indent=4))

        # ---------------------------------------------
        # OPTION 2: FULL PDF (Map-Reduce Flow)
        # ---------------------------------------------
        elif choice == '2':
            pdf_files = glob.glob(os.path.join(files_dir, "*.pdf"))
            if not pdf_files:
                print(f"\n[!] No PDF files found in '{files_dir}'.")
                continue
            
            print("\nAvailable files:")
            for i, file_path in enumerate(pdf_files, 1):
                print(f"({i}) {os.path.basename(file_path)}")

            try:
                file_index = int(input("\nChoose a file number: ").strip()) - 1
                selected_file = pdf_files[file_index]
            except (ValueError, IndexError):
                print("Invalid selection.")
                continue

            print(f"\nInitiating Enterprise Map-Reduce on {os.path.basename(selected_file)}...")
            print("This may take a minute as the LLM audits section-by-section.")
            
            # Run the Map-Reduce pipeline
            final_master_report = rag.audit_large_document(
                target_pdf_path=selected_file, 
                framework_name=framework, 
                
            )

            if "error" in final_master_report:
                print(f"\n[ERROR] {final_master_report['error']}")
            else:
                print("\n" + "="*50)
                print(" MASTER AUDIT REPORT (REDUCED):")
                print("="*50)
                print(json.dumps(final_master_report, indent=4))

if __name__ == "__main__":
    main()