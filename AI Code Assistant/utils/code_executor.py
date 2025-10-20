import os
import logging
from e2b_code_interpreter import Sandbox
from agno.agent import Agent
from agno.models.groq import Groq

logger = logging.getLogger(__name__)

class CodeExecutor:
    def __init__(self):
        self.e2b_api_key = os.getenv("E2B_API_KEY")
        if not self.e2b_api_key:
            logger.error("E2B_API_KEY not found in environment variables")
            raise ValueError("E2B_API_KEY environment variable is required")
        
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if self.groq_api_key:
            self.error_agent = Agent(
                model=Groq(id="llama-3.3-70b-versatile", api_key=self.groq_api_key),
                markdown=True
            )
    
    def execute_code(self, code: str, timeout: int = 30) -> str:
        """Execute Python code in sandbox"""
        try:
            os.environ["E2B_API_KEY"] = self.e2b_api_key
            sandbox = Sandbox(timeout=timeout)
            
            execution = sandbox.run_code(code)
            
            if execution.error:
                error_explanation = self._explain_error(execution.error)
                return self._format_error_output(execution.error, error_explanation)
            else:
                return self._format_success_output(execution.logs)
        
        except Exception as e:
            logger.error(f"Error executing code: {str(e)}")
            return self._format_error_output(str(e), "Sandbox execution failed")
    
    def _explain_error(self, error_message: str) -> str:
        """Get explanation for execution error"""
        if not self.groq_api_key:
            return "Error explanation not available (GROQ_API_KEY missing)"
        
        try:
            prompt = f"""Explain this Python error in simple terms and suggest how to fix it:

Error:
{error_message}

Provide a concise explanation and 2-3 possible solutions."""
            
            response = self.error_agent.run(prompt)
            return response.content if response else "Could not generate error explanation"
        
        except Exception as e:
            logger.error(f"Error generating error explanation: {str(e)}")
            return "Error explanation unavailable"
    
    def _format_success_output(self, logs) -> str:
        """Format successful execution output"""
        try:
            if hasattr(logs, 'stdout') and logs.stdout:
                output_lines = [line.strip() for line in logs.stdout.splitlines() if line.strip()]
            else:
                output_lines = [str(logs).strip()] if str(logs).strip() else ["Code executed successfully (no output)"]
            
            formatted_output = "<br>".join(output_lines)
            
            return f"""
            <div class='bg-green-50 border border-green-200 rounded-lg p-4 mb-4'>
                <div class='flex items-center mb-2'>
                    <span class='text-green-600 font-semibold'>✅ Execution Successful</span>
                </div>
                <pre class='text-green-800 whitespace-pre-wrap'>{formatted_output}</pre>
            </div>
            """
        
        except Exception as e:
            return f"<div class='bg-yellow-50 p-3 rounded'>Output formatting error: {str(e)}</div>"
    
    def _format_error_output(self, error: str, explanation: str) -> str:
        """Format error output with explanation"""
        return f"""
        <div class='bg-red-50 border border-red-200 rounded-lg p-4 mb-4'>
            <div class='flex items-center mb-2'>
                <span class='text-red-600 font-semibold'>❌ Execution Error</span>
            </div>
            <div class='mb-3'>
                <pre class='text-red-700 whitespace-pre-wrap'>{error}</pre>
            </div>
            <div class='bg-orange-50 border-l-4 border-orange-400 p-3'>
                <strong class='text-orange-700'>💡 Explanation & Solutions:</strong>
                <p class='text-orange-800 mt-1'>{explanation}</p>
            </div>
        </div>
        """