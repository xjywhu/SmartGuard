import requests
import json
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with local Ollama server."""
    
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "llama3.2:latest"):
        self.base_url = base_url
        self.default_model = default_model
        
    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama server not available: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
        return []
    
    def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """Generate text using Ollama."""
        model = model or self.default_model
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return ""
    
    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, **kwargs) -> str:
        """Chat completion using Ollama."""
        model = model or self.default_model
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', '')
            else:
                logger.error(f"Ollama chat API error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error calling Ollama chat: {e}")
            return ""
    
    def format_openai_style(self, messages: List[Dict[str, str]], model: Optional[str] = None, **kwargs) -> str:
        """Format messages in OpenAI style and call Ollama."""
        # Convert OpenAI-style messages to a single prompt for generate API
        if len(messages) == 1 and messages[0].get('role') == 'user':
            return self.generate(messages[0]['content'], model, **kwargs)
        
        # For multi-message conversations, use chat API
        return self.chat(messages, model, **kwargs)