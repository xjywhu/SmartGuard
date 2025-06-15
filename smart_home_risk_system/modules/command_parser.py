#!/usr/bin/env python3
"""
Command Parser Module

LLM-based command parser that extracts device and action from natural language commands.
Uses OpenAI API to parse commands into structured JSON format.

Example:
    Input: "Turn off the smoke detector"
    Output: {"device": "smoke_detector", "action": "OFF"}
"""

from typing import Dict, Any, Optional
import json
import logging
from .ollama_client import OllamaClient

# Import templates - handle both relative and absolute imports
try:
    from ..prompts.llm_templates import COMMAND_PARSER_PROMPT, COMMAND_PARSER_EXAMPLES
except ImportError:
    try:
        from prompts.llm_templates import COMMAND_PARSER_PROMPT, COMMAND_PARSER_EXAMPLES
    except ImportError:
        # Fallback definitions if imports fail
        COMMAND_PARSER_PROMPT = """
You are a smart home command parser. Extract the device and intended action from natural language commands.
Return a JSON object with "device" and "action" fields.
"""
        COMMAND_PARSER_EXAMPLES = [
            {"input": "Turn off the smoke detector", "output": {"device": "smoke_detector", "action": "OFF"}}
        ]

class CommandParser:
    """Parses natural language commands into structured format."""
    
    def __init__(self, use_llm: bool = True, model: str = "llama3.2:latest"):
        self.use_llm = use_llm
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        if use_llm:
            self.ollama_client = OllamaClient(default_model=model)
            self.llm_available = self.ollama_client.is_available()
            if not self.llm_available:
                self.logger.warning("Ollama server not available. Using fallback parsing.")
        else:
            self.llm_available = False
    
    def parse(self, command_text: str) -> Dict[str, Any]:
        """
        Parse a natural language command into structured format.
        
        Args:
            command_text: Natural language command
            
        Returns:
            Parsed command with device and action
        """
        if self.use_llm and self.llm_available:
            return self._parse_with_llm(command_text)
        else:
            return self._fallback_parse(command_text)
    
    def _parse_with_llm(self, command: str) -> Dict[str, Any]:
        """Parse command using LLM."""
        try:
            prompt = COMMAND_PARSER_PROMPT.format(
                examples=COMMAND_PARSER_EXAMPLES,
                command=command
            )
            
            response = self.ollama_client.generate(
                prompt=prompt,
                model=self.model,
                temperature=0.1
            )
            
            # Try to parse JSON response
            try:
                result = json.loads(response.strip())
                return result
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse LLM response as JSON: {response}")
                return self._fallback_parse(command)
                
        except Exception as e:
            self.logger.error(f"LLM parsing failed: {e}")
            return self._fallback_parse(command)
    
    def _build_prompt(self, command_text: str) -> str:
        """
        Build the prompt with examples for the LLM.
        
        Args:
            command_text: Command to parse
            
        Returns:
            Formatted prompt string
        """
        examples_text = "\n".join([
            f"Input: {ex['input']}\nOutput: {json.dumps(ex['output'])}"
            for ex in COMMAND_PARSER_EXAMPLES
        ])
        
        return f"""{examples_text}

Input: {command_text}
Output:"""
    
    def _extract_json(self, response_text: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response text.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed JSON object
        """
        # Try to find JSON in the response
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        # Try to parse JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON-like structure
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                try:
                    return json.loads(response_text[start:end])
                except json.JSONDecodeError:
                    pass
            
            # If all else fails, return empty dict
            return {}
    
    def _validate_parsed_result(self, result: Dict[str, Any]) -> bool:
        """
        Validate that the parsed result has required fields.
        
        Args:
            result: Parsed result to validate
            
        Returns:
            True if valid, False otherwise
        """
        return (
            isinstance(result, dict) and
            "device" in result and
            "action" in result and
            isinstance(result["device"], str) and
            isinstance(result["action"], str)
        )
    
    def _fallback_parse(self, command_text: str) -> Dict[str, Any]:
        """
        Fallback rule-based parser when LLM is not available.
        
        Args:
            command_text: Command to parse
            
        Returns:
            Parsed command using simple rules
        """
        command_lower = command_text.lower()
        
        # Simple device detection
        device_mappings = {
            "smoke detector": "smoke_detector",
            "smoke alarm": "smoke_detector",
            "air conditioner": "air_conditioner",
            "ac": "air_conditioner",
            "heater": "heater",
            "thermostat": "thermostat",
            "light": "light",
            "lamp": "light",
            "fan": "fan",
            "window": "window",
            "door": "door",
            "camera": "camera",
            "tv": "tv",
            "television": "tv",
            "washing machine": "washing_machine",
            "dishwasher": "dishwasher",
            "oven": "oven",
            "stove": "stove",
            "refrigerator": "refrigerator",
            "fridge": "refrigerator"
        }
        
        # Find device
        device = "unknown_device"
        for device_name, device_key in device_mappings.items():
            if device_name in command_lower:
                device = device_key
                break
        
        # Simple action detection
        if any(word in command_lower for word in ["turn on", "switch on", "start", "activate", "open"]):
            action = "ON"
        elif any(word in command_lower for word in ["turn off", "switch off", "stop", "deactivate", "close"]):
            action = "OFF"
        elif any(word in command_lower for word in ["increase", "raise", "up", "higher"]):
            action = "INCREASE"
        elif any(word in command_lower for word in ["decrease", "lower", "down", "reduce"]):
            action = "DECREASE"
        elif "set" in command_lower:
            action = "SET"
        else:
            action = "UNKNOWN"
        
        return {
            "device": device,
            "action": action,
            "confidence": 0.6,  # Lower confidence for fallback
            "method": "fallback"
        }