#!/usr/bin/env python3
"""
State Forecaster Module

LLM-based state forecaster that predicts how the smart home context will change
after executing a command. Combines LLM prediction with context update logic.

Example:
    Input: command={"device": "smoke_detector", "action": "OFF"}, context={...}
    Output: Updated context with smoke_detector status changed to OFF
"""

from typing import Dict, Any, List, Optional
import json
import logging
from copy import deepcopy

try:
    from .ollama_client import OllamaClient
except ImportError:
    try:
        from ollama_client import OllamaClient
    except ImportError:
        print("Warning: ollama_client not available. Using fallback forecasting only.")
        OllamaClient = None

# Import templates - handle both relative and absolute imports
try:
    from ..prompts.llm_templates import STATE_FORECASTER_PROMPT, STATE_FORECASTER_EXAMPLES
except ImportError:
    try:
        from prompts.llm_templates import STATE_FORECASTER_PROMPT, STATE_FORECASTER_EXAMPLES
    except ImportError:
        # Fallback definitions if imports fail
        STATE_FORECASTER_PROMPT = """
You are a smart home state forecaster. Given a command and current context, predict how the context will change.
Return a JSON object with the updated context values.
"""
        STATE_FORECASTER_EXAMPLES = [
            {"input": "Turn off smoke detector", "output": {"devices": {"smoke_detector": "OFF"}}}
        ]

# Import utilities - handle both relative and absolute imports
try:
    from ..utils.context_utils import deep_merge_context
except ImportError:
    try:
        from utils.context_utils import deep_merge_context
    except ImportError:
        # Fallback implementations
        def deep_merge_context(base_context, updates):
            """Simple fallback merge function."""
            import copy
            result = copy.deepcopy(base_context)
            for key, value in updates.items():
                if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                    result[key].update(value)
                else:
                    result[key] = value
            return result

class StateForecaster:
    """Forecasts state changes based on commands and current context."""
    
    def __init__(self, use_llm: bool = True, model: str = "llama3.2:latest"):
        self.use_llm = use_llm
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        if use_llm and OllamaClient:
            self.ollama_client = OllamaClient(default_model=model)
            self.llm_available = self.ollama_client.is_available()
            if not self.llm_available:
                self.logger.warning("Ollama server not available. Using fallback forecasting.")
        else:
            self.llm_available = False
            if use_llm:
                self.logger.warning("OllamaClient not available. Using fallback forecasting.")
    
    def forecast(self, parsed_command: Dict[str, Any], current_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forecast the new context state after executing a command.
        
        Args:
            parsed_command: Parsed command with device and action
            current_context: Current smart home context
            
        Returns:
            Updated context after command execution
        """
        if self.use_llm and self.llm_available:
            return self._forecast_with_llm(parsed_command, current_context)
        else:
            return self._fallback_forecast(parsed_command, current_context)
    
    def _forecast_with_llm(self, parsed_command: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast state changes using LLM."""
        try:
            prompt = STATE_FORECASTER_PROMPT.format(
                examples=STATE_FORECASTER_EXAMPLES,
                command=json.dumps(parsed_command, indent=2),
                context=json.dumps(context, indent=2)
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
                return self._fallback_forecast(parsed_command, context)
                
        except Exception as e:
            self.logger.error(f"LLM forecasting failed: {e}")
            return self._fallback_forecast(parsed_command, context)
    
    def _build_forecast_prompt(self, parsed_command: Dict[str, Any], current_context: Dict[str, Any]) -> str:
        """
        Build the prompt for state forecasting.
        
        Args:
            parsed_command: Parsed command
            current_context: Current context
            
        Returns:
            Formatted prompt string
        """
        # Include examples
        examples_text = "\n\n".join([
            f"Command: {json.dumps(ex['command'])}\nCurrent Context: {json.dumps(ex['current_context'], indent=2)}\nChanges: {json.dumps(ex['changes'])}"
            for ex in STATE_FORECASTER_EXAMPLES
        ])
        
        return f"""{examples_text}

Command: {json.dumps(parsed_command)}
Current Context: {json.dumps(current_context, indent=2)}
Changes:"""
    
    def _extract_changes(self, response_text: str) -> Dict[str, Any]:
        """
        Extract predicted changes from LLM response.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Dictionary of predicted changes
        """
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
    
    def _apply_changes(self, current_context: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply predicted changes to the current context.
        
        Args:
            current_context: Current context
            changes: Predicted changes
            
        Returns:
            Updated context
        """
        # Create a deep copy of the current context
        new_context = deepcopy(current_context)
        
        # Apply changes using deep merge
        return deep_merge_context(new_context, changes)
    
    def _fallback_forecast(self, parsed_command: Dict[str, Any], current_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback rule-based forecasting when LLM is not available.
        
        Args:
            parsed_command: Parsed command
            current_context: Current context
            
        Returns:
            Updated context using simple rules
        """
        # Create a deep copy of the current context
        new_context = deepcopy(current_context)
        
        device = parsed_command.get("device", "")
        action = parsed_command.get("action", "")
        
        # Simple rule-based updates
        if "devices" in new_context:
            # Map device names to context keys
            device_mappings = {
                "smoke_detector": "smoke_detector",
                "air_conditioner": "ac_bedroom",  # Assume bedroom AC
                "ac": "ac_bedroom",
                "heater": "heater_living_room",  # Assume living room heater
                "light": "light_living_room",  # Assume living room light
                "fan": "fan_kitchen",  # Assume kitchen fan
                "tv": "tv_living_room",
                "washing_machine": "washing_machine",
                "dishwasher": "dishwasher",
                "oven": "oven",
                "stove": "stove",
                "refrigerator": "refrigerator"
            }
            
            # Find matching device in context
            context_device_key = None
            for context_key in new_context["devices"].keys():
                if device in context_key or any(d in context_key for d in device_mappings.keys() if d == device):
                    context_device_key = context_key
                    break
            
            # If exact match not found, try mapped names
            if not context_device_key and device in device_mappings:
                mapped_device = device_mappings[device]
                if mapped_device in new_context["devices"]:
                    context_device_key = mapped_device
            
            # Apply action to device
            if context_device_key:
                if action in ["ON", "ACTIVATE", "START"]:
                    new_context["devices"][context_device_key] = "ON"
                elif action in ["OFF", "DEACTIVATE", "STOP"]:
                    new_context["devices"][context_device_key] = "OFF"
        
        # Handle window/door actions
        if "sensors" in new_context:
            if "window" in device:
                # Find window sensors
                for sensor_key in new_context["sensors"].keys():
                    if "window" in sensor_key:
                        if action in ["OPEN", "ON"]:
                            new_context["sensors"][sensor_key] = "OPEN"
                        elif action in ["CLOSE", "OFF"]:
                            new_context["sensors"][sensor_key] = "CLOSED"
                        break
            
            elif "door" in device:
                # Find door sensors
                for sensor_key in new_context["sensors"].keys():
                    if "door" in sensor_key:
                        if action in ["OPEN", "UNLOCK"]:
                            new_context["sensors"][sensor_key] = "UNLOCKED"
                        elif action in ["CLOSE", "LOCK"]:
                            new_context["sensors"][sensor_key] = "LOCKED"
                        break
        
        return new_context