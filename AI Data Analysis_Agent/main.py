import os
import logging
import time
import uuid
from flask import Flask, request, render_template, jsonify, session
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from dotenv import load_dotenv

from utils.image_processor import ImageProcessor
from utils.code_generator import CodeGenerator
from utils.code_executor import CodeExecutor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app.log', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

class AppConfig:
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    UPLOAD_FOLDER = 'uploads'
    MAX_CODE_LENGTH = 10000

class GroqCodeAssistant:
    
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config.from_object(AppConfig)
        self.app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')
        
        os.makedirs(AppConfig.UPLOAD_FOLDER, exist_ok=True)
        
        self.components = self._initialize_components()
        self._register_handlers()
        self._register_routes()
    
    def _initialize_components(self):
        try:
            return {
                'image_processor': ImageProcessor(),
                'code_generator': CodeGenerator(),
                'code_executor': CodeExecutor()
            }
        except Exception as e:
            logger.error(f"Component initialization failed: {str(e)}")
            raise
    
    def _register_handlers(self):
        
        @self.app.errorhandler(404)
        def not_found(error):
            return render_template('error.html', error_code=404, error_message="Page not found"), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal server error: {str(error)}")
            return render_template('error.html', error_code=500, error_message="Internal server error"), 500
        
        @self.app.errorhandler(RequestEntityTooLarge)
        def too_large(error):
            return render_template('error.html', error_code=413, error_message="File too large"), 413
        
        @self.app.errorhandler(Exception)
        def handle_exception(error):
            logger.error(f"Unhandled exception: {str(error)}")
            return render_template('error.html', error_code=500, error_message="Unexpected error"), 500
    
    def _register_routes(self):
        
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
            session.clear()
            return jsonify({"success": True, "message": "Session cleared"})
    
    def _validate_input(self, query, image_file):
        if not query and not image_file:
            return False, "Provide text query or image"
        
        if query and image_file:
            return False, "Use only one input method"
        
        if len(query) > 5000:
            return False, "Query too long"
        
        return True, None
    
    def _process_image_input(self, image_file):
        if not self._allowed_file(image_file.filename):
            return None, "Invalid file type"
        
        filename = secure_filename(image_file.filename)
        temp_path = os.path.join(AppConfig.UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
        
        try:
            image_file.save(temp_path)
            extracted_text = self.components['image_processor'].extract_text_from_image_file(temp_path)
            
            try:
                os.remove(temp_path)
            except OSError:
                pass
            
            if not extracted_text:
                return None, "No text extracted from image"
            
            return extracted_text, None
        
        except Exception as e:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            logger.error(f"Image processing error: {str(e)}")
            return None, f"Image processing failed: {str(e)}"
    
    def _generate_and_execute_code(self, extracted_text):
        try:
            start_time = time.time()
            generated_code = self.components['code_generator'].generate_python_code(extracted_text)
            generation_time = time.time() - start_time
            
            if not generated_code:
                return None, None, "Code generation failed"
            
            if len(generated_code) > AppConfig.MAX_CODE_LENGTH:
                return None, None, "Generated code too long"
            
            start_time = time.time()
            execution_result = self.components['code_executor'].execute_code(generated_code)
            execution_time = time.time() - start_time
            
            logger.info(f"Generation: {generation_time:.2f}s, Execution: {execution_time:.2f}s")
            
            self._store_in_history(extracted_text, generated_code, execution_result)
            
            return generated_code, execution_result, None
        
        except Exception as e:
            logger.error(f"Code processing error: {str(e)}")
            return None, None, f"Processing failed: {str(e)}"
    
    def _store_in_history(self, problem, code, result):
        if 'history' not in session:
            session['history'] = []
        
        history_entry = {
            'id': str(uuid.uuid4()),
            'timestamp': time.time(),
            'problem': problem[:500] + "..." if len(problem) > 500 else problem,
            'code_preview': code[:200] + "..." if len(code) > 200 else code,
            'result_preview': result[:300] + "..." if len(result) > 300 else result
        }
        
        session['history'].insert(0, history_entry)
        session['history'] = session['history'][:10]
        session.modified = True
    
    def _handle_index(self):
        if request.method == "GET":
            return render_template("index.html")
        
        try:
            query = request.form.get("query", "").strip()
            image_file = request.files.get("image")
            
            is_valid, error_message = self._validate_input(query, image_file)
            if not is_valid:
                return render_template("index.html", error=error_message)
            
            if image_file:
                extracted_text, error_message = self._process_image_input(image_file)
                if error_message:
                    return render_template("index.html", error=error_message)
            else:
                extracted_text = query
            
            generated_code, execution_result, error_message = self._generate_and_execute_code(extracted_text)
            if error_message:
                return render_template("index.html", error=error_message)
            
            return render_template("index.html",
                                 extracted=extracted_text,
                                 solution=generated_code,
                                 result=execution_result)
        
        except Exception as e:
            logger.error(f"Request processing error: {str(e)}")
            return render_template("index.html", error=f"Processing error: {str(e)}")
    
    def _handle_health_check(self):
        try:
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
            logger.error(f"Health check error: {str(e)}")
            return jsonify({"status": "unhealthy", "error": str(e)}), 500
    
    def _handle_api_process(self):
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
            
            query = data.get('query', '').strip()
            
            if not query:
                return jsonify({"error": "Query required"}), 400
            
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
            logger.error(f"API error: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    def _handle_history(self):
        history = session.get('history', [])
        return render_template("history.html", history=history)
    
    def _allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in AppConfig.ALLOWED_EXTENSIONS
    
    def run(self):
        debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
        host = os.getenv("FLASK_HOST", "0.0.0.0")
        port = int(os.getenv("FLASK_PORT", "5000"))
        
        logger.info(f"Starting app on {host}:{port}")
        
        self.app.run(debug=debug_mode, host=host, port=port)

def create_app():
    return GroqCodeAssistant().app

if __name__ == "__main__":
    assistant = GroqCodeAssistant()
    assistant.run()