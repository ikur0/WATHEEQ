import os
import glob
import json
from rag3 import ComplianceRAG 

def main():
    print("Initializing Lytrex Enterprise Map-Reduce RAG... Please wait.")
    
    # IMPORTANT: Ensure your OpenAI API key is exported in your environment or passed here directly
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

        print("\nSelect Output Mode:")
        print("(1) Detailed (Comprehensive lists of violations and compliant areas)")
        print("(2) Summary (Concise overview of key issues only)")
        mode_choice = input("Mode (1/2): ").strip()
        summary_mode = True if mode_choice == '2' else False

        # ---------------------------------------------
        # OPTION 1: TEXT INPUT (Standard Flow)
        # ---------------------------------------------
        if choice == '1':
            text_input = input("\nEnter the text you want to test:\n> ")
            if not text_input.strip(): continue
                
            print("\n[System] Retrieving relevant context from Vector Database...")
            
            # Step 1: Only retrieve the context (Bypass LLM initially)
            retrieval_result = rag.check_compliance_text(
                text=text_input,
                framework_name=framework,
                summary_mode=summary_mode,
                evaluate_llm=False
            )
            context = retrieval_result.get("retrieved_framework_context", "")

            # Question 1
            print("\nQ: want to see the reterived context ?")
            print("(1) yes")
            print("(2) No")
            show_ctx = input("> ").strip()
            
            if show_ctx == '1':
                print("\n" + "="*50)
                print(" RETRIEVED CONTEXT:")
                print("="*50)
                print(context)
                print("="*50)

            # Question 2
            print("\nQ: want to send it to LLM for evaluation ?")
            print("(1) yes")
            print("(2) No")
            send_llm = input("> ").strip()

            if send_llm == '1':
                print("\nExecuting LLM Audit...")
                # We can pass the already retrieved context directly to save API time
                result = rag.evaluate_with_llm(context, text_input, summary_mode)
                
                print("\n" + "="*50)
                print("FINAL RESPONSE:")
                print("="*50)
                print(json.dumps(result, indent=4))
            else:
                print("\n[System] Evaluation skipped. Returning to main menu.")

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

            print(f"\n[System] Retrieving framework context for {os.path.basename(selected_file)}...")
            
            # Step 1: Only retrieve the context for all sections (Bypass LLM initially)
            retrieval_result = rag.audit_large_document(
                target_pdf_path=selected_file, 
                framework_name=framework, 
                summary_mode=summary_mode,
                evaluate_llm=False
            )

            # Question 1
            print("\nQ: want to see the reterived context ?")
            print("(1) yes")
            print("(2) No")
            show_ctx = input("> ").strip()
            
            if show_ctx == '1':
                print("\n" + "="*50)
                print(" RETRIEVED CONTEXT PER SECTION:")
                print("="*50)
                print(json.dumps(retrieval_result.get("raw_retrieval_results", {}), indent=4))
                print("="*50)

            # Question 2
            print("\nQ: want to send it to LLM for evaluation ?")
            print("(1) yes")
            print("(2) No")
            send_llm = input("> ").strip()

            if send_llm == '1':
                print(f"\nInitiating Enterprise Map-Reduce on {os.path.basename(selected_file)}...")
                
                # Step 2: Run the full Map-Reduce pipeline
                final_master_report = rag.audit_large_document(
                    target_pdf_path=selected_file, 
                    framework_name=framework, 
                    summary_mode=summary_mode,
                    evaluate_llm=True
                )

                if "error" in final_master_report:
                    print(f"\n[ERROR] {final_master_report['error']}")
                else:
                    print("\n" + "="*50)
                    print(" MASTER AUDIT REPORT (REDUCED):")
                    print("="*50)
                    print(json.dumps(final_master_report, indent=4))
            else:
                print("\n[System] Evaluation skipped. Returning to main menu.")

if __name__ == "__main__":
    main()