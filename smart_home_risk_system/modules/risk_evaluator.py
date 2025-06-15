#!/usr/bin/env python3
"""
Risk Evaluator Module

LLM-based risk evaluator that assesses the safety risk of executing a command
in a given smart home context. Returns risk level (LOW/MEDIUM/HIGH) and rationale.

Example:
    Input: command={"device": "smoke_detector", "action": "OFF"}, context={gas_sensor: "ON", ...}
    Output: {"risk_level": "HIGH", "rationale": "Turning off smoke detector while gas detected..."}
"""

import json
import os
from typing import Dict, Any, Optional
import logging

try:
    from .ollama_client import OllamaClient
except ImportError:
    try:
        from ollama_client import OllamaClient
    except ImportError:
        print("Warning: ollama_client not available. Using fallback evaluation only.")
        OllamaClient = None

# Import templates - handle both relative and absolute imports
try:
    from ..prompts.llm_templates import RISK_EVALUATOR_PROMPT, RISK_EVALUATOR_EXAMPLES
except ImportError:
    try:
        from prompts.llm_templates import RISK_EVALUATOR_PROMPT, RISK_EVALUATOR_EXAMPLES
    except ImportError:
        # Fallback definitions if imports fail
        RISK_EVALUATOR_PROMPT = """
You are a smart home risk evaluator. Assess the safety risk of commands in given contexts.
Return HIGH, MEDIUM, or LOW risk level with rationale.
"""
        RISK_EVALUATOR_EXAMPLES = [
            {"input": "Turn off smoke detector while user asleep", "output": {"risk_level": "HIGH", "rationale": "Disabling safety device while user is asleep"}}
        ]

class RiskEvaluator:
    """Evaluates risk level of commands based on context and predicted state changes."""
    
    def __init__(self, use_llm: bool = True, model: str = "mistral:latest"):
        """
        Initialize the risk evaluator.
        
        Args:
            use_llm: Whether to use LLM for evaluation
            model: Ollama model to use for risk evaluation
        """
        self.use_llm = use_llm
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        if use_llm and OllamaClient:
            self.ollama_client = OllamaClient(default_model=model)
            self.llm_available = self.ollama_client.is_available()
            if not self.llm_available:
                self.logger.warning("Ollama server not available. Using fallback risk evaluation.")
        else:
            self.llm_available = False
            self.ollama_client = None
    
    def evaluate(self, parsed_command: Dict[str, Any], context: Dict[str, Any], predicted_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate the risk of executing a command in the given context.
        
        Args:
            parsed_command: Parsed command with device and action
            context: Smart home context (current)
            predicted_context: Predicted context after command execution
            
        Returns:
            Risk assessment with level and rationale
        """
        if predicted_context is None:
            predicted_context = context
            
        if not self.llm_available or not self.use_llm:
            # Fallback to rule-based risk evaluation
            return self._fallback_evaluate(parsed_command, context, predicted_context)
        
        return self._evaluate_with_llm(parsed_command, context, predicted_context)
    
    def _evaluate_with_llm(self, parsed_command: Dict[str, Any], 
                          current_context: Dict[str, Any], 
                          predicted_context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate risk using LLM."""
        try:
            prompt = RISK_EVALUATOR_PROMPT.format(
                examples=RISK_EVALUATOR_EXAMPLES,
                command=json.dumps(parsed_command, indent=2),
                current_context=json.dumps(current_context, indent=2),
                predicted_context=json.dumps(predicted_context, indent=2)
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
                return self._fallback_evaluate(parsed_command, current_context, predicted_context)
                
        except Exception as e:
            self.logger.error(f"LLM risk evaluation failed: {e}")
            return self._fallback_evaluate(parsed_command, current_context, predicted_context)
    
    def _extract_risk_assessment(self, response_text: str) -> Dict[str, Any]:
        """
        Extract risk assessment from LLM response.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Risk assessment dictionary
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
            
            # Try to extract risk level and rationale from text
            return self._parse_text_response(response_text)
    
    def _parse_text_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse risk assessment from plain text response.
        
        Args:
            response_text: Plain text response
            
        Returns:
            Risk assessment dictionary
        """
        response_lower = response_text.lower()
        
        # Extract risk level
        risk_level = "MEDIUM"  # Default
        if "high" in response_lower:
            risk_level = "HIGH"
        elif "low" in response_lower:
            risk_level = "LOW"
        elif "medium" in response_lower:
            risk_level = "MEDIUM"
        
        # Use the response text as rationale
        rationale = response_text.strip()
        
        return {
            "risk_level": risk_level,
            "rationale": rationale,
            "confidence": 0.7,
            "method": "text_parsing"
        }
    
    def _validate_risk_assessment(self, assessment: Dict[str, Any]) -> bool:
        """
        Validate that the risk assessment has required fields.
        
        Args:
            assessment: Risk assessment to validate
            
        Returns:
            True if valid, False otherwise
        """
        return (
            isinstance(assessment, dict) and
            "risk_level" in assessment and
            "rationale" in assessment and
            assessment["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        )
    
    def _fallback_evaluate(self, parsed_command: Dict[str, Any], current_context: Dict[str, Any], predicted_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback rule-based risk evaluation when LLM is not available.
        
        Args:
            parsed_command: Parsed command
            current_context: Current smart home context
            predicted_context: Predicted context after command execution
            
        Returns:
            Risk assessment using simple rules
        """
        device = parsed_command.get("device", "").lower()
        action = parsed_command.get("action", "").lower()
        
        # High-risk scenarios
        high_risk_conditions = [
            # Turning off safety devices
            (device == "smoke_detector" and action == "off"),
            (device == "gas_sensor" and action == "off"),
            ("detector" in device and action == "off"),
            
            # Turning on heating devices when it's already hot
            (device in ["heater", "oven", "stove"] and action == "on" and 
             current_context.get("weather", {}).get("temperature_outdoor", 0) > 35),
            
            # Opening windows/doors when air quality is poor
            (device in ["window", "door"] and action in ["open", "unlock"] and
             current_context.get("aqi_outdoor", 0) > 200),
            
            # User not home but turning on appliances
            (not current_context.get("user_home", True) and 
             device in ["oven", "stove", "heater"] and action == "on"),
            
            # User asleep and turning on loud devices
            (current_context.get("user_asleep", False) and 
             device in ["tv", "music", "speaker"] and action == "on")
        ]
        
        # Medium-risk scenarios
        medium_risk_conditions = [
            # Turning on AC when windows are open
            (device in ["ac", "air_conditioner"] and action == "on" and
             any("OPEN" in str(v) for k, v in current_context.get("sensors", {}).items() if "window" in k)),
            
            # Turning on heating when AC is on
            (device == "heater" and action == "on" and
             any("ON" in str(v) for k, v in current_context.get("devices", {}).items() if "ac" in k)),
            
            # Late night activities
            (current_context.get("time", "").startswith(("22", "23", "00", "01", "02", "03")) and
             device in ["washing_machine", "dishwasher"] and action == "on")
        ]
        
        # Check conditions
        for condition in high_risk_conditions:
            if condition:
                return {
                    "risk_level": "HIGH",
                    "rationale": f"High risk: {device} {action} in current context poses safety concerns",
                    "confidence": 0.8,
                    "method": "rule_based"
                }
        
        for condition in medium_risk_conditions:
            if condition:
                return {
                    "risk_level": "MEDIUM",
                    "rationale": f"Medium risk: {device} {action} may cause inefficiency or minor issues",
                    "confidence": 0.7,
                    "method": "rule_based"
                }
        
        # Default to low risk
        return {
            "risk_level": "LOW",
            "rationale": f"Low risk: {device} {action} appears safe in current context",
            "confidence": 0.6,
            "method": "rule_based"
        }