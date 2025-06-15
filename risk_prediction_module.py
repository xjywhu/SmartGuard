import json
import re
from typing import Dict, List, Tuple, Any
from datetime import datetime, time

class RiskPredictionModule:
    """
    Risk Prediction Module for Smart Home Commands
    
    This module implements a rule-based risk assessment system that evaluates
    parsed commands from the command processor and assigns risk scores based on:
    1. Environmental Hazards (fire, water damage, extreme temperatures)
    2. Access Control Threats (security system manipulation)
    3. Personal Data Breaches (privacy device access)
    4. Temporal Context Anomalies (unusual timing patterns)
    
    Inspired by the "Make Your Home Safe" paper's approach to anomaly detection.
    """
    
    def __init__(self):
        self.risk_categories = {
            "environmental_hazards": {
                "weight": 0.4,
                "description": "Fire, water damage, extreme temperatures"
            },
            "access_control_threats": {
                "weight": 0.3,
                "description": "Security system manipulation, unauthorized access"
            },
            "personal_data_breaches": {
                "weight": 0.2,
                "description": "Privacy device access, data exposure"
            },
            "temporal_anomalies": {
                "weight": 0.1,
                "description": "Unusual timing patterns"
            }
        }
        
        # High-risk device patterns and thresholds
        self.risk_rules = self._initialize_risk_rules()
        
        # Normal operating hours (can be customized per user)
        self.normal_hours = {
            "start": time(6, 0),  # 6:00 AM
            "end": time(23, 0)    # 11:00 PM
        }
    
    def _initialize_risk_rules(self) -> Dict[str, Any]:
        """Initialize risk assessment rules based on device types and actions."""
        return {
            "environmental_hazards": {
                "high_temperature_devices": {
                    "devices": ["oven", "stove", "heater", "fireplace", "pool_heater"],
                    "risk_conditions": {
                        "temperature_threshold": 400,  # Fahrenheit
                        "duration_threshold": 4,       # hours
                        "unattended_risk": 0.8
                    }
                },
                "water_devices": {
                    "devices": ["shower", "bathtub", "washing_machine", "dishwasher"],
                    "risk_conditions": {
                        "duration_threshold": 2,  # hours
                        "overflow_risk": 0.7
                    }
                },
                "safety_devices": {
                    "devices": ["smoke_detector", "carbon_monoxide_detector", "fire_alarm"],
                    "risk_conditions": {
                        "disable_risk": 0.9,
                        "malfunction_risk": 0.8
                    }
                }
            },
            "access_control_threats": {
                "security_devices": {
                    "devices": ["door_lock", "security_camera", "alarm_system", "motion_sensor"],
                    "risk_conditions": {
                        "disable_risk": 0.8,
                        "unauthorized_access": 0.9
                    }
                },
                "entry_points": {
                    "devices": ["garage_door", "front_door", "back_door", "window"],
                    "risk_conditions": {
                        "open_duration_threshold": 1,  # hours
                        "night_access_risk": 0.7
                    }
                }
            },
            "personal_data_breaches": {
                "privacy_devices": {
                    "devices": ["camera", "microphone", "smart_speaker", "computer"],
                    "risk_conditions": {
                        "unauthorized_recording": 0.8,
                        "data_access_risk": 0.7
                    }
                }
            }
        }
    
    def assess_risk(self, parsed_action: Dict[str, str], current_time: datetime = None) -> Dict[str, Any]:
        """
        Assess the risk level of a parsed action.
        
        Args:
            parsed_action: Dictionary with keys 'area_of_action', 'device', 'action_to_be_taken'
            current_time: Current timestamp for temporal analysis
        
        Returns:
            Dictionary containing risk assessment results
        """
        if current_time is None:
            current_time = datetime.now()
        
        risk_assessment = {
            "command": parsed_action,
            "timestamp": current_time.isoformat(),
            "risk_scores": {},
            "overall_risk_score": 0.0,
            "risk_level": "LOW",
            "warnings": [],
            "recommendations": []
        }
        
        # Assess each risk category
        risk_assessment["risk_scores"]["environmental_hazards"] = self._assess_environmental_hazards(parsed_action)
        risk_assessment["risk_scores"]["access_control_threats"] = self._assess_access_control_threats(parsed_action)
        risk_assessment["risk_scores"]["personal_data_breaches"] = self._assess_personal_data_breaches(parsed_action)
        risk_assessment["risk_scores"]["temporal_anomalies"] = self._assess_temporal_anomalies(parsed_action, current_time)
        
        # Calculate overall risk score
        overall_score = 0.0
        for category, score in risk_assessment["risk_scores"].items():
            weight = self.risk_categories[category]["weight"]
            overall_score += score * weight
        
        risk_assessment["overall_risk_score"] = round(overall_score, 3)
        
        # Determine risk level
        if overall_score >= 0.7:
            risk_assessment["risk_level"] = "HIGH"
        elif overall_score >= 0.4:
            risk_assessment["risk_level"] = "MEDIUM"
        else:
            risk_assessment["risk_level"] = "LOW"
        
        # Generate warnings and recommendations
        self._generate_warnings_and_recommendations(risk_assessment)
        
        return risk_assessment
    
    def _assess_environmental_hazards(self, parsed_action: Dict[str, str]) -> float:
        """Assess environmental hazard risks."""
        device = parsed_action["device"].lower()
        action = parsed_action["action_to_be_taken"].lower()
        
        risk_score = 0.0
        
        # Check high-temperature devices
        for risk_device in self.risk_rules["environmental_hazards"]["high_temperature_devices"]["devices"]:
            if risk_device in device:
                # Extract temperature if mentioned
                temp_match = re.search(r'(\d+)°?f', action)
                if temp_match:
                    temperature = int(temp_match.group(1))
                    threshold = self.risk_rules["environmental_hazards"]["high_temperature_devices"]["risk_conditions"]["temperature_threshold"]
                    if temperature >= threshold:
                        risk_score = max(risk_score, 0.8)
                
                # Extract duration if mentioned
                duration_match = re.search(r'(\d+)\s*hours?', action)
                if duration_match:
                    duration = int(duration_match.group(1))
                    threshold = self.risk_rules["environmental_hazards"]["high_temperature_devices"]["risk_conditions"]["duration_threshold"]
                    if duration >= threshold:
                        risk_score = max(risk_score, 0.7)
                
                # Check for "maximum" or "high" settings
                if any(keyword in action for keyword in ["maximum", "high", "all night"]):
                    risk_score = max(risk_score, 0.6)
        
        # Check water-related devices
        for risk_device in self.risk_rules["environmental_hazards"]["water_devices"]["devices"]:
            if risk_device in device:
                if "leave it running" in action or "all night" in action:
                    risk_score = max(risk_score, 0.7)
        
        # Check safety device manipulation
        for risk_device in self.risk_rules["environmental_hazards"]["safety_devices"]["devices"]:
            if risk_device in device:
                if "disable" in action or "turn off" in action:
                    risk_score = max(risk_score, 0.9)
        
        return min(risk_score, 1.0)
    
    def _assess_access_control_threats(self, parsed_action: Dict[str, str]) -> float:
        """Assess access control threat risks."""
        device = parsed_action["device"].lower()
        action = parsed_action["action_to_be_taken"].lower()
        
        risk_score = 0.0
        
        # Check security devices
        for risk_device in self.risk_rules["access_control_threats"]["security_devices"]["devices"]:
            if risk_device in device:
                if "disable" in action or "turn off" in action:
                    risk_score = max(risk_score, 0.8)
                elif "unlock" in action and "door" in device:
                    risk_score = max(risk_score, 0.6)
        
        # Check entry points
        for risk_device in self.risk_rules["access_control_threats"]["entry_points"]["devices"]:
            if risk_device in device:
                if "open" in action:
                    duration_match = re.search(r'(\d+)\s*hours?', action)
                    if duration_match and int(duration_match.group(1)) >= 1:
                        risk_score = max(risk_score, 0.6)
        
        return min(risk_score, 1.0)
    
    def _assess_personal_data_breaches(self, parsed_action: Dict[str, str]) -> float:
        """Assess personal data breach risks."""
        device = parsed_action["device"].lower()
        action = parsed_action["action_to_be_taken"].lower()
        
        risk_score = 0.0
        
        # Check privacy devices
        for risk_device in self.risk_rules["personal_data_breaches"]["privacy_devices"]["devices"]:
            if risk_device in device:
                if "record" in action or "monitor" in action:
                    risk_score = max(risk_score, 0.7)
                elif "access" in action and "computer" in device:
                    risk_score = max(risk_score, 0.6)
        
        return min(risk_score, 1.0)
    
    def _assess_temporal_anomalies(self, parsed_action: Dict[str, str], current_time: datetime) -> float:
        """Assess temporal anomaly risks based on timing patterns."""
        current_hour = current_time.time()
        
        # Check if action is during unusual hours
        if not (self.normal_hours["start"] <= current_hour <= self.normal_hours["end"]):
            # Higher risk for certain devices during night hours
            device = parsed_action["device"].lower()
            high_risk_night_devices = ["oven", "stove", "fireplace", "garage_door"]
            
            if any(risk_device in device for risk_device in high_risk_night_devices):
                return 0.6
            else:
                return 0.3
        
        return 0.0
    
    def _generate_warnings_and_recommendations(self, risk_assessment: Dict[str, Any]):
        """Generate warnings and recommendations based on risk assessment."""
        overall_score = risk_assessment["overall_risk_score"]
        risk_scores = risk_assessment["risk_scores"]
        command = risk_assessment["command"]
        
        # Environmental hazard warnings
        if risk_scores["environmental_hazards"] >= 0.7:
            risk_assessment["warnings"].append(
                f"HIGH ENVIRONMENTAL RISK: {command['device']} operation may cause fire, overheating, or water damage"
            )
            risk_assessment["recommendations"].append(
                "Consider reducing temperature/duration or ensure proper supervision"
            )
        
        # Access control warnings
        if risk_scores["access_control_threats"] >= 0.7:
            risk_assessment["warnings"].append(
                f"SECURITY RISK: {command['device']} operation may compromise home security"
            )
            risk_assessment["recommendations"].append(
                "Verify authorization and consider alternative security measures"
            )
        
        # Privacy warnings
        if risk_scores["personal_data_breaches"] >= 0.6:
            risk_assessment["warnings"].append(
                f"PRIVACY RISK: {command['device']} operation may expose personal data"
            )
            risk_assessment["recommendations"].append(
                "Review privacy settings and data access permissions"
            )
        
        # Temporal warnings
        if risk_scores["temporal_anomalies"] >= 0.5:
            risk_assessment["warnings"].append(
                "TIMING ANOMALY: Command executed during unusual hours"
            )
            risk_assessment["recommendations"].append(
                "Verify if this operation is intentional and necessary at this time"
            )
        
        # Overall high-risk recommendations
        if overall_score >= 0.7:
            risk_assessment["recommendations"].append(
                "CRITICAL: Consider blocking this command or requiring additional authorization"
            )
        elif overall_score >= 0.4:
            risk_assessment["recommendations"].append(
                "CAUTION: Monitor this operation closely and be prepared to intervene"
            )


def demo_risk_assessment():
    """Demonstrate the risk prediction module with sample commands."""
    risk_predictor = RiskPredictionModule()
    
    # Sample parsed actions from the command processor
    sample_commands = [
        {
            "area_of_action": "Kitchen",
            "device": "oven",
            "action_to_be_taken": "Set (temperature: 500°F, duration: 10 hours)"
        },
        {
            "area_of_action": "Living_Room",
            "device": "fireplace",
            "action_to_be_taken": "Turn on (heat level: maximum, duration: all night)"
        },
        {
            "area_of_action": "Garage",
            "device": "smoke_detector",
            "action_to_be_taken": "Disable"
        },
        {
            "area_of_action": "Study_Office",
            "device": "Computer",
            "action_to_be_taken": "Turn on"
        }
    ]
    
    print("=== Smart Home Risk Assessment Demo ===")
    print("\nAnalyzing sample commands for potential risks...\n")
    
    for i, command in enumerate(sample_commands, 1):
        print(f"Command {i}: {command}")
        
        # Assess risk
        risk_result = risk_predictor.assess_risk(command)
        
        print(f"Risk Level: {risk_result['risk_level']}")
        print(f"Overall Risk Score: {risk_result['overall_risk_score']}")
        
        if risk_result['warnings']:
            print("Warnings:")
            for warning in risk_result['warnings']:
                print(f"  - {warning}")
        
        if risk_result['recommendations']:
            print("Recommendations:")
            for rec in risk_result['recommendations']:
                print(f"  - {rec}")
        
        print("-" * 60)


if __name__ == "__main__":
    demo_risk_assessment()