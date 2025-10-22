import os
import logging
import time
from typing import Dict, Any, Optional, Tuple
from flask import Flask, request, render_template, jsonify, session
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from dotenv import load_dotenv
import uuid

from utils.image_processor import ImageProcessor
from utils.code_generator import CodeGenerator
from utils.code_executor import CodeExecutor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class AppConfig:
    """Application configuration class"""
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    SESSION_TIMEOUT = 3600  # 1 hour
    UPLOAD_FOLDER = 'uploads'
    MAX_CODE_LENGTH = 10000  # Maximum code length to prevent abuse

class GroqCodeAssistant:
    """Main application class"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config.from_object(AppConfig)
        self.app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
        
        # Ensure upload directory exists
        os.makedirs(AppConfig.UPLOAD_FOLDER, exist_ok=True)
        
        # Initialize components
        self.components = self._initialize_components()
        self._register_handlers()
        self._register_routes()
    
    def _initialize_components(self) -> Dict[str, Any]:
        """Initialize application components with error handling"""
        try:
            return {
                'image_processor': ImageProcessor(),
                'code_generator': CodeGenerator(),
                'code_executor': CodeExecutor()
            }
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    def _register_handlers(self):
        """Register error handlers"""
        
        @self.app.errorhandler(404)
        def not_found(error):
            return render_template('error.html', 
                                error_code=404,
                                error_message="Page not found"), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal server error: {str(error)}")
            return render_template('error.html',
                                error_code=500,
                                error_message="Internal server error"), 500
        
        @self.app.errorhandler(RequestEntityTooLarge)
        def too_large(error):
            return render_template('error.html',
                                error_code=413,
                                error_message="File too large. Maximum size is 16MB."), 413
        
        @self.app.errorhandler(Exception)
        def handle_exception(error):
            logger.error(f"Unhandled exception: {str(error)}")
            return render_template('error.html',
                                error_code=500,
                                error_message="An unexpected error occurred"), 500
    
    def _register_routes(self):
        """Register application routes"""
        
        @self.app.route("/", methods=["GET", "POST"])
        def index():
            return self._handle_index()
        
        @self.app.route("/health", methods=["GET"])
        def health_check():
            return self._handle_health_check()
        
        @self.app.route("/api/process", methods=["POST"])
        def api_process():
            return self._handle_api_process()
        
        @self.app.route("/history", methods=["GET"])
        def history():
            return self._handle_history()
        
        @self.app.route("/clear-session", methods=["POST"])
        def clear_session():
            return self._handle_clear_session()
    
    def _validate_input(self, query: str, image_file) -> Tuple[bool, Optional[str]]:
        """Validate user input"""
        if not query and not image_file:
            return False, "⚠️ Please provide either a text query or an image."
        
        if query and image_file:
            return False, "⚠️ Please provide only one input method (text OR image)."
        
        if len(query) > 5000:  # Reasonable limit for text input
            return False, "⚠️ Query too long. Maximum 5000 characters."
        
        return True, None
    
    def _process_image_input(self, image_file) -> Tuple[Optional[str], Optional[str]]:
        """Process image input and extract text"""
        if not self._allowed_file(image_file.filename):
            return None, "⚠️ Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, BMP, WEBP)."
        
        # Secure filename and save temporarily
        filename = secure_filename(image_file.filename)
        temp_path = os.path.join(AppConfig.UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
        
        try:
            image_file.save(temp_path)
            
            # Extract text from image
            extracted_text = self.components['image_processor'].extract_text_from_image_file(temp_path)
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except OSError as e:
                logger.warning(f"Failed to remove temporary file {temp_path}: {str(e)}")
            
            if not extracted_text:
                return None, "❌ No text could be extracted from the image. Please try with a clearer image."
            
            return extracted_text, None
        
        except Exception as e:
            # Clean up on error
            try:
                os.remove(temp_path)
            except OSError:
                pass
            logger.error(f"Error processing image: {str(e)}")
            return None, f"❌ Error processing image: {str(e)}"
    
    def _generate_and_execute_code(self, extracted_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Generate code and execute it"""
        try:
            # Generate code
            start_time = time.time()
            generated_code = self.components['code_generator'].generate_python_code(extracted_text)
            generation_time = time.time() - start_time
            
            if not generated_code:
                return None, None, "❌ Failed to generate code from the problem description."
            
            # Validate code length
            if len(generated_code) > AppConfig.MAX_CODE_LENGTH:
                return None, None, f"❌ Generated code too long ({len(generated_code)} characters). Please try a simpler problem."
            
            # Execute code
            start_time = time.time()
            execution_result = self.components['code_executor'].execute_code(generated_code)
            execution_time = time.time() - start_time
            
            logger.info(f"Code generation: {generation_time:.2f}s, execution: {execution_time:.2f}s")
            
            # Store in session history
            self._store_in_history(extracted_text, generated_code, execution_result)
            
            return generated_code, execution_result, None
        
        except Exception as e:
            logger.error(f"Error in code generation/execution: {str(e)}")
            return None, None, f"❌ Error processing your request: {str(e)}"
    
    def _store_in_history(self, problem: str, code: str, result: str):
        """Store request in session history"""
        if 'history' not in session:
            session['history'] = []
        
        history_entry = {
            'id': str(uuid.uuid4()),
            'timestamp': time.time(),
            'problem': problem[:500] + "..." if len(problem) > 500 else problem,
            'code_preview': code[:200] + "..." if len(code) > 200 else code,
            'result_preview': result[:300] + "..." if len(result) > 300 else result
        }
        
        session['history'].insert(0, history_entry)  # Add to beginning
        session['history'] = session['history'][:10]  # Keep only last 10 entries
        session.modified = True
    
    def _handle_index(self):
        """Handle main page requests"""
        if request.method == "GET":
            return render_template("index.html")
        
        try:
            # Handle form submission
            query = request.form.get("query", "").strip()
            image_file = request.files.get("image")
            
            # Validate input
            is_valid, error_message = self._validate_input(query, image_file)
            if not is_valid:
                return render_template("index.html", error=error_message)
            
            # Process input
            if image_file:
                extracted_text, error_message = self._process_image_input(image_file)
                if error_message:
                    return render_template("index.html", error=error_message)
            else:
                extracted_text = query
            
            # Generate and execute code
            generated_code, execution_result, error_message = self._generate_and_execute_code(extracted_text)
            if error_message:
                return render_template("index.html", error=error_message)
            
            return render_template("index.html",
                                 extracted=extracted_text,
                                 solution=generated_code,
                                 result=execution_result)
        
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return render_template("index.html", 
                                 error=f"❌ An unexpected error occurred: {str(e)}")
    
    def _handle_health_check(self):
        """Handle health check requests"""
        try:
            # Test component availability
            components_healthy = all(component is not None for component in self.components.values())
            
            status = {
                "status": "healthy" if components_healthy else "degraded",
                "service": "Groq Code Assistant",
                "timestamp": time.time(),
                "components": {
                    name: "healthy" if component else "unavailable"
                    for name, component in self.components.items()
                }
            }
            
            return jsonify(status), 200 if components_healthy else 503
        
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return jsonify({
                "status": "unhealthy",
                "service": "Groq Code Assistant",
                "error": str(e)
            }), 500
    
    def _handle_api_process(self):
        """Handle API processing requests"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            query = data.get('query', '').strip()
            # API doesn't support image upload for simplicity
            
            if not query:
                return jsonify({"error": "Query is required"}), 400
            
            generated_code, execution_result, error_message = self._generate_and_execute_code(query)
            
            if error_message:
                return jsonify({"error": error_message}), 400
            
            return jsonify({
                "problem": query,
                "generated_code": generated_code,
                "execution_result": execution_result,
                "success": True
            })
        
        except Exception as e:
            logger.error(f"API processing error: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    def _handle_history(self):
        """Display request history"""
        history = session.get('history', [])
        return render_template("history.html", history=history)
    
    def _handle_clear_session(self):
        """Clear session data"""
        session.clear()
        return jsonify({"success": True, "message": "Session cleared"})
    
    def _allowed_file(self, filename):
        """Check if file type is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in AppConfig.ALLOWED_EXTENSIONS
    
    def run(self):
        """Run the application"""
        debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
        host = os.getenv("FLASK_HOST", "0.0.0.0")
        port = int(os.getenv("FLASK_PORT", "5000"))
        
        logger.info(f"Starting Groq Code Assistant on {host}:{port} (debug: {debug_mode})")
        
        self.app.run(
            debug=debug_mode,
            host=host,
            port=port
        )

def create_app():
    """Application factory function"""
    return GroqCodeAssistant().app

if __name__ == "__main__":
    assistant = GroqCodeAssistant()
    assistant.run()