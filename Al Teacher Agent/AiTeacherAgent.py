
from flask import Flask, render_template, request
from groq import Groq
import asyncio
import os
import logging
from dotenv import load_dotenv
import markdown
from functools import wraps
import time
import html

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')

# Configuration
class Config:
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    REQUEST_TIMEOUT = 30  # seconds
    MAX_TOPIC_LENGTH = 200

# Async Groq client with connection pooling
class GroqClient:
    _instance = None
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required")
        self.client = Groq(api_key=api_key)
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            api_key = Config.GROQ_API_KEY
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            cls._instance = cls(api_key)
        return cls._instance

# Decorator for async route handling
def async_route(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapped

# Rate limiting decorator (simple in-memory implementation)
def rate_limit(max_requests=10, window=60):
    requests = []
    
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            now = time.time()
            # Clean old requests
            requests[:] = [req_time for req_time in requests if now - req_time < window]
            
            if len(requests) >= max_requests:
                return render_template(
                    "error.html", 
                    error="Rate limit exceeded. Please try again later."
                ), 429
            
            requests.append(now)
            return f(*args, **kwargs)
        return wrapped
    return decorator

# Input validation
def validate_topic(topic: str) -> tuple[bool, str]:
    if not topic or not topic.strip():
        return False, "Topic cannot be empty"
    
    topic = topic.strip()
    
    if len(topic) > Config.MAX_TOPIC_LENGTH:
        return False, f"Topic too long (max {Config.MAX_TOPIC_LENGTH} characters)"
    
    # Basic sanitization
    if any(char in topic for char in ['<', '>', 'script', '../']):
        return False, "Invalid characters in topic"
    
    return True, topic

# Async Groq chat completion with better error handling
async def ask_groq(prompt: str, model: str = None) -> str:
    if model is None:
        model = Config.DEFAULT_MODEL
    
    try:
        client = GroqClient.get_instance()
        
        response = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful AI teaching assistant. Reply in Markdown format with well-structured sections and simulated Google Doc links."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    model=model,
                    temperature=0.7,
                    max_tokens=4000
                )
            ),
            timeout=Config.REQUEST_TIMEOUT
        )
        
        return response.choices[0].message.content
        
    except asyncio.TimeoutError:
        logger.error(f"Groq API request timed out for model: {model}")
        return "Error: Request timed out. Please try again."
    except Exception as e:
        logger.error(f"Groq API error: {str(e)}")
        return f"Error: Unable to process request. Please try again later."

# Markdown to HTML conversion with safety
def markdown_to_html(markdown_text: str) -> str:
    try:
        # Escape any HTML that might be in the markdown first
        safe_text = html.escape(markdown_text)
        
        html_content = markdown.markdown(
            safe_text,
            extensions=[
                "fenced_code", 
                "nl2br", 
                "tables", 
                "sane_lists",
                "codehilite"
            ],
            extension_configs={
                'codehilite': {
                    'css_class': 'codehilite'
                }
            }
        )
        return html_content
    except Exception as e:
        logger.error(f"Markdown conversion error: {str(e)}")
        return f"<p>Error converting content to HTML</p>"

# Agent definitions with better prompts
AGENTS = {
    "professor": {
        "name": "Professor",
        "prompt_template": "Create a detailed knowledge base on '{topic}' including key concepts, applications, and fundamentals. Structure it with clear sections and include a simulated Google Doc link for further reading.",
        "icon": "🎓"
    },
    "advisor": {
        "name": "Learning Advisor", 
        "prompt_template": "Design a structured learning roadmap for '{topic}', broken into beginner to expert levels with clear milestones. Include realistic time estimates and a simulated Google Doc link for the full plan.",
        "icon": "📚"
    },
    "librarian": {
        "name": "Resource Librarian",
        "prompt_template": "Curate a comprehensive list of high-quality resources (videos, documentation, blogs, books) for learning '{topic}' with brief descriptions and difficulty levels. Include a simulated Google Doc link for the full resource collection.",
        "icon": "📖"
    },
    "assistant": {
        "name": "Practice Assistant",
        "prompt_template": "Create practical exercises, real-world projects, and assessment questions for mastering '{topic}', including solutions and implementation guidance. Include a simulated Google Doc link for the full exercise set.",
        "icon": "💡"
    }
}

# Run all agents concurrently
async def run_agents(topic: str) -> dict:
    tasks = []
    roles = []
    
    for role, config in AGENTS.items():
        prompt = config["prompt_template"].format(topic=topic)
        task = ask_groq(prompt)
        tasks.append(task)
        roles.append(role)
    
    try:
        raw_responses = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"Error gathering agent responses: {str(e)}")
        return {"error": "Failed to generate responses"}
    
    # Process responses
    responses = {}
    for role, raw_response in zip(roles, raw_responses):
        if isinstance(raw_response, Exception):
            logger.error(f"Agent {role} failed: {str(raw_response)}")
            responses[role] = {
                "name": AGENTS[role]["name"],
                "icon": AGENTS[role]["icon"],
                "content": "Error: Unable to generate response for this agent.",
                "error": True
            }
        else:
            html_content = markdown_to_html(raw_response)
            responses[role] = {
                "name": AGENTS[role]["name"],
                "icon": AGENTS[role]["icon"],
                "content": html_content,
                "error": False
            }
    
    return responses

@app.route("/", methods=["GET", "POST"])
@rate_limit(max_requests=5, window=60)  # 5 requests per minute
def index():
    if request.method == "POST":
        topic = request.form.get("topic", "").strip()
        
        # Validate input
        is_valid, validation_result = validate_topic(topic)
        if not is_valid:
            return render_template(
                "index.html", 
                error=validation_result,
                topic=topic
            )
        
        # Check API key
        if not Config.GROQ_API_KEY:
            return render_template(
                "index.html", 
                error="GROQ_API_KEY environment variable is not set.",
                topic=topic
            )
        
        try:
            # Run agents asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            responses = loop.run_until_complete(run_agents(validation_result))
            loop.close()
            
            return render_template(
                "index.html", 
                topic=validation_result,
                responses=responses
            )
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return render_template(
                "index.html", 
                error="An unexpected error occurred. Please try again.",
                topic=topic
            )
    
    return render_template("index.html")

@app.route("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.errorhandler(404)
def not_found(error):
    return render_template("error.html", error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return render_template("error.html", error="Internal server error"), 500

if __name__ == "__main__":
    # Validate required environment variables
    if not Config.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY environment variable is not set")
    
    # Run app
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(
        debug=debug_mode,
        host=os.getenv('FLASK_HOST', '127.0.0.1'),
        port=int(os.getenv('FLASK_PORT', 5000))
    )
