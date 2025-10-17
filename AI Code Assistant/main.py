import os
import argparse
import sys
from dotenv import load_dotenv

from utils.image_processor import ImageProcessor
from utils.code_generator import CodeGenerator
from utils.code_executor import CodeExecutor

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Groq Code Assistant CLI")
    parser.add_argument("--image", "-i", help="Path to image file")
    parser.add_argument("--text", "-t", help="Problem description text")
    parser.add_argument("--output", "-o", help="Output file for generated code")
    
    args = parser.parse_args()
    
    if not args.image and not args.text:
        print("❌ Please provide either --image or --text argument")
        parser.print_help()
        sys.exit(1)
    
    try:
        # Initialize components
        image_processor = ImageProcessor()
        code_generator = CodeGenerator()
        code_executor = CodeExecutor()
        
        # Extract problem description
        if args.image:
            if not os.path.exists(args.image):
                print(f"❌ Image file not found: {args.image}")
                sys.exit(1)
            
            extracted_text = image_processor.extract_text_from_image_file(args.image)
            if not extracted_text:
                print("❌ No text could be extracted from the image.")
                sys.exit(1)
        else:
            extracted_text = args.text
        
        print("=== Extracted Problem Description ===")
        print(extracted_text)
        print()
        
        # Generate code
        print("🔄 Generating code...")
        generated_code = code_generator.generate_python_code(extracted_text)
        
        if not generated_code:
            print("❌ Failed to generate code.")
            sys.exit(1)
        
        print("=== Generated Python Code ===")
        print(generated_code)
        print()
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(generated_code)
            print(f"💾 Code saved to: {args.output}")
        
        # Execute code
        print("🔄 Executing code...")
        execution_result = code_executor.execute_code(generated_code)
        
        print("=== Execution Result ===")
        print(execution_result)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
