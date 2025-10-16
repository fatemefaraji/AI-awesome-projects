import os
import logging
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from utils.image_processor import ImageProcessor
from utils.code_generator import CodeGenerator
from utils.code_executor import CodeExecutor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Initialize components
image_processor = ImageProcessor()
code_generator = CodeGenerator()
code_executor = CodeExecutor()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    
    try:
        # Handle form submission
        query = request.form.get("query", "").strip()
        image_file = request.files.get("image")
        
        # Validate input
        if not query and not image_file:
            return render_template("index.html", 
                                 error="⚠️ Please provide either a text query or an image.")
        
        if query and image_file:
            return render_template("index.html", 
                                 error="⚠️ Please provide only one input method (text OR image).")
        
        # Process image or text input
        if image_file:
            if not allowed_file(image_file.filename):
                return render_template("index.html", 
                                     error="⚠️ Invalid file type. Please upload an image.")
            
            # Extract text from image
            extracted_text = image_processor.extract_text_from_image(image_file)
            if not extracted_text:
                return render_template("index.html", 
                                     error="❌ No text could be extracted from the image.")
        else:
            extracted_text = query
        
        # Generate code
        generated_code = code_generator.generate_python_code(extracted_text)
        if not generated_code:
            return render_template("index.html", 
                                 error="❌ Failed to generate code from the problem description.")
        
        # Execute code
        execution_result = code_executor.execute_code(generated_code)
        
        return render_template("index.html",
                             extracted=extracted_text,
                             solution=generated_code,
                             result=execution_result)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return render_template("index.html", 
                             error=f"❌ An error occurred: {str(e)}")

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "Groq Code Assistant"})

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "False").lower() == "true", 
            host="0.0.0.0", 
            port=5000)
