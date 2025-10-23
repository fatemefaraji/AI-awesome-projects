from flask import Flask, render_template, request, session, flash, redirect, url_for
import pandas as pd
from groq import Groq
import os
import logging
from typing import Tuple, Optional, Dict, Any
import io
from utils import preprocess_and_save

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-please-change")

# Configuration
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    SESSION_PERMANENT=False
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalysisApp:
    """Core application logic for data analysis functionality"""
    
    def __init__(self):
        self.supported_formats = {'.csv', '.xlsx', '.xls', '.json'}
    
    def validate_file(self, filename: str) -> Tuple[bool, str]:
        """Validate uploaded file format"""
        if not filename:
            return False, "No file selected"
        
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in self.supported_formats:
            return False, f"Unsupported file format. Supported formats: {', '.join(self.supported_formats)}"
        
        return True, ""
    
    def generate_analysis_code(self, query: str, api_key: str) -> Tuple[Optional[str], Optional[str]]:
        """Generate Python code for data analysis using Groq API"""
        try:
            prompt = f"""
You are a Python data analyst. Given a pandas DataFrame named `df`, write efficient and safe Python code using pandas to answer this question:

Question: {query}

Requirements:
1. Return ONLY the Python code without any explanations, markdown, or additional text
2. Use 'result' as the final output variable
3. Use pandas best practices and efficient operations
4. Handle potential errors gracefully
5. Do not use any operations that could modify the original DataFrame

Code:
"""
            client = Groq(api_key=api_key)
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.1,  # Lower temperature for more deterministic code
                max_tokens=1000
            )
            
            code = chat_completion.choices[0].message.content.strip()
            # Clean up common formatting issues
            code = code.removeprefix("```python").removesuffix("```").strip()
            
            return code, None
            
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            return None, f"API Error: {str(e)}"
    
    def execute_analysis_code(self, code: str, df: pd.DataFrame) -> Tuple[Any, Optional[str]]:
        """Safely execute the generated analysis code"""
        try:
            # Create a safe execution environment
            safe_globals = {
                'pd': pd,
                'df': df,
                '__builtins__': {**__builtins__}  # Copy of builtins for safety
            }
            
            # Remove dangerous builtins
            dangerous_builtins = ['open', 'file', 'exec', 'eval', 'compile', 
                                '__import__', 'exit', 'quit', 'globals', 'locals']
            for dangerous in dangerous_builtins:
                if dangerous in safe_globals['__builtins__']:
                    del safe_globals['__builtins__'][dangerous]
            
            # Execute the code
            exec(code, safe_globals)
            
            # Get the result
            result = safe_globals.get('result')
            if result is None:
                return None, "No result variable found in generated code"
                
            return result, None
            
        except Exception as e:
            logger.error(f"Code execution error: {str(e)}")
            return None, f"Execution Error: {str(e)}"

# Initialize the application logic
analysis_app = DataAnalysisApp()

@app.route("/", methods=["GET", "POST"])
def index():
    """Main route for the data analysis application"""
    if request.method == "POST":
        return handle_post_request()
    
    return render_template("index.html")

def handle_post_request():
    """Handle POST request data and process the analysis"""
    file = request.files.get("file")
    query = request.form.get("query", "").strip()
    groq_key = request.form.get("api_key", "").strip()
    
    # Validate inputs
    if not groq_key:
        flash("Please enter your Groq API key.", "error")
        return render_template("index.html")
    
    if not file or file.filename == "":
        flash("Please select a file to upload.", "error")
        return render_template("index.html")
    
    # Validate file format
    is_valid, error_msg = analysis_app.validate_file(file.filename)
    if not is_valid:
        flash(error_msg, "error")
        return render_template("index.html")
    
    try:
        # Process uploaded file
        df, cols, df_html, err = preprocess_and_save(file)
        if err:
            flash(f"Error processing file: {err}", "error")
            return render_template("index.html")
        
        # Store data in session for potential reuse
        session['df_columns'] = cols
        session['file_processed'] = True
        
        result_data = {
            'df_html': df_html,
            'df_preview_html': df.head().to_html(classes='table table-striped table-bordered', index=False) if df is not None else "",
            'code_generated': "",
            'result_html': "",
            'query_used': query
        }
        
        # Process query if provided
        if query:
            result_data.update(process_data_analysis(query, groq_key, df))
        
        return render_template("index.html", **result_data)
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        flash(f"An unexpected error occurred: {str(e)}", "error")
        return render_template("index.html")

def process_data_analysis(query: str, api_key: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Process data analysis query and return results"""
    # Generate analysis code
    code_generated, code_error = analysis_app.generate_analysis_code(query, api_key)
    if code_error:
        flash(f"Code generation failed: {code_error}", "error")
        return {}
    
    # Execute the generated code
    result, exec_error = analysis_app.execute_analysis_code(code_generated, df)
    if exec_error:
        flash(f"Code execution failed: {exec_error}", "error")
        return {'code_generated': code_generated}
    
    # Format the result
    result_html = format_result(result)
    
    return {
        'code_generated': code_generated,
        'result_html': result_html
    }

def format_result(result: Any) -> str:
    """Format the analysis result as HTML"""
    try:
        if isinstance(result, pd.DataFrame):
            return result.to_html(
                classes='table table-striped table-bordered', 
                index=False,
                escape=False
            )
        elif isinstance(result, pd.Series):
            return result.to_frame().to_html(
                classes='table table-striped table-bordered',
                index=True,
                escape=False
            )
        else:
            # For scalar results or other types
            return f"""
            <div class="alert alert-info">
                <h5>Result:</h5>
                <pre>{str(result)}</pre>
            </div>
            """
    except Exception as e:
        logger.error(f"Error formatting result: {str(e)}")
        return f"<div class='alert alert-danger'>Error formatting result: {str(e)}</div>"

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash("File too large. Maximum size is 16MB.", "error")
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    flash("An internal server error occurred. Please try again.", "error")
    return redirect(url_for('index'))

if __name__ == "__main__":
    # Use environment variable for port and debug mode
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    app.run(
        host="0.0.0.0", 
        port=port, 
        debug=debug
    )
