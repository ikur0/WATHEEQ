import os
import glob
import json
from rag import ComplianceRAG  # Imports your RAG class from rag.py

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

        # Get Framework
        print("\nSelect Framework to assess against:")
        print("Options: NCA, ECC, SAMA, or ALL (To search across everything)")
        framework = input("Framework: ").strip().upper()
        if not framework:
            framework = "ALL"

        result_data = None

        # ---------------------------------------------
        # OPTION 1: TEXT INPUT
        # ---------------------------------------------
        if choice == '1':
            text_input = input("\nEnter the text you want to test:\n> ")
            if not text_input.strip():
                print("Empty text provided. Restarting loop.")
                continue
                
            # Run RAG but pause before LLM (run_llm=False)
            result_data = rag.check_compliance_text(
                text=text_input, 
                framework_name=framework,
                k=5,
                run_llm=False # Stops after printing context
            )

        # ---------------------------------------------
        # OPTION 2: FILE INPUT
        # ---------------------------------------------
        elif choice == '2':
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

            print(f"\nProcessing {os.path.basename(selected_file)}...")
            
            # Run RAG but pause before LLM (run_llm=False)
            result_data = rag.check_compliance(
                target_pdf_path=selected_file, 
                framework_name=framework, 
                run_llm=False # Stops after printing context
            )

        # ---------------------------------------------
        # ERROR HANDLING & LLM EXECUTION
        # ---------------------------------------------
        if result_data and "error" in result_data:
            print(f"\n[ERROR] {result_data['error']}")
            continue

        if not result_data:
            continue

        # LLM Prompt Phase
        print("\n" + "="*50)
        print("Wanna pass it to LLM?")
        print("(1) Yes")
        print("(2) No")
        llm_choice = input("Choice: ").strip()

        if llm_choice == '1':
            formatted_ctx = result_data["formatted_context"]
            company_doc = result_data["company_doc"]
            
            # Manually trigger the LLM evaluation
            final_result = rag.evaluate_with_llm(formatted_ctx, company_doc, detailed=True)
            
            print("\n" + "="*50)
            print("FINAL LLM AUDIT RESPONSE:")
            print("="*50)
            print(json.dumps(final_result, indent=4))
        else:
            print("Skipping LLM execution. Returning to main menu.")


if __name__ == "__main__":
    main()