import os
import glob
import json
from rag_improvment_groq import ComplianceRAG  # Imports your RAG class from rag.py

def main():
    print("Initializing RAG System... Please wait.")
    # Initialize RAG - all settings can be easily changed in rag.py constructor
    rag = ComplianceRAG()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files_dir = os.path.join(base_dir, "files")

    # Auto-create the files directory if it doesn't exist
    os.makedirs(files_dir, exist_ok=True)

    while True:
        print("\n" + "="*60)
        print(" LYTREX COMPLIANCE AUDITOR - DEV CONSOLE")
        print("="*60)
        print("(1) Type test text to RAG")
        print("(2) Input file to RAG")
        print("(3) Exit System")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()

        if choice == '3':
            print("Exiting Lytrex Console. Goodbye!")
            break

        if choice not in ['1', '2']:
            print("Invalid choice. Please try again.")
            continue

        if choice == '1':
            print("\n[!] Notice: The current ComplianceRAG class uses PyPDFLoader, which requires PDF files.")
            print("Please use Option 2 to process documents via file path.")
            continue

        # Get Framework
        print("\nSelect Framework to assess against:")
        print("Options: NCA, ECC, SAMA, or ALL (To search across everything)")
        framework = input("Framework: ").strip().upper()
        if not framework:
            framework = "ALL"

        result_data = None
        selected_file = None

        # ---------------------------------------------
        # OPTION 2: FILE INPUT
        # ---------------------------------------------
        if choice == '2':
            pdf_files = glob.glob(os.path.join(files_dir, "*.pdf"))
            if not pdf_files:
                print(f"\n[!] No PDF files found in the '{files_dir}' folder.")
                print("Please add some test PDFs to the folder and try again.")
                continue
            
            print("\nAvailable files in 'files/':")
            for i, file_path in enumerate(pdf_files, 1):
                file_name = os.path.basename(file_path)
                print(f"({i}) {file_name}")

            file_choice = input("\nChoose a file number: ").strip()
            
            try:
                file_index = int(file_choice) - 1
                if file_index < 0 or file_index >= len(pdf_files):
                    raise ValueError()
                selected_file = pdf_files[file_index]
            except ValueError:
                print("Invalid file selection. Restarting loop.")
                continue

            # ---------------------------------------------
            # PREFERENCES: MODE & CONTEXT VIEW
            # ---------------------------------------------
            print("\n" + "-"*50)
            print("Select Audit Mode:")
            print("(1) Detailed (Comprehensive section-by-section analysis)")
            print("(2) Summary (Strict top-level overview)")
            mode_choice = input("Choice (1/2): ").strip()
            is_detailed = (mode_choice != '2') # Defaults to detailed unless 2 is explicitly chosen

            print(f"\nRunning retrieval phase for {os.path.basename(selected_file)}...")
            
            # Run RAG but pause before LLM by directly calling audit_large_document with evaluate_llm=False
            result_data = rag.audit_large_document(
                target_pdf_path=selected_file, 
                framework_name=framework, 
                summary_mode=not is_detailed,
                evaluate_llm=False # Stops after fetching context
            )

        # ---------------------------------------------
        # ERROR HANDLING & RESULTS DISPLAY
        # ---------------------------------------------
        if result_data and "error" in result_data:
            print(f"\n[ERROR] {result_data['error']}")
            continue

        if not result_data:
            continue

        print("\n" + "="*50)
        print("Retrieval phase complete!")
        
        view_ctx = input("Do you want to view the retrieved contexts before deciding on LLM execution? (y/n): ").strip().lower()
        if view_ctx == 'y':
            raw_results = result_data.get("raw_retrieval_results", {})
            if not raw_results:
                print("\n[!] No contexts retrieved.")
            else:
                for section_name, data in raw_results.items():
                    print(f"\n\n{'='*20} {section_name.upper()} {'='*20}")
                    print("\n[COMPANY CHUNK]:")
                    print(data.get('query', 'N/A'))
                    print("\n[MATCHED FRAMEWORK CONTEXT]:")
                    print(data.get('context', 'N/A'))
                print("\n" + "="*50)

        # ---------------------------------------------
        # LLM EXECUTION PHASE
        # ---------------------------------------------
        print("\nWanna pass it to LLM for final audit?")
        print("(1) Yes")
        print("(2) No")
        llm_choice = input("Choice: ").strip()

        if llm_choice == '1':
            print(f"\nTriggering Full LLM Audit in {'Detailed' if is_detailed else 'Summary'} mode... This may take a moment.")
            # Run the primary API bridge method, which evaluates relevance and runs the LLM
            final_result = rag.check_compliance(
                target_pdf_path=selected_file,
                framework_name=framework,
                detailed=is_detailed
            )
            
            print("\n" + "="*50)
            print("FINAL LLM AUDIT RESPONSE:")
            print("="*50)
            print(json.dumps(final_result, indent=4))
        else:
            print("Skipping LLM execution. Returning to main menu.")

if __name__ == "__main__":
    main()