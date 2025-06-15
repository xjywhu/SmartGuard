import json
from datetime import datetime
from kg_builder_and_command_processor import load_home_system, load_commands
from risk_prediction_module import RiskPredictionModule

class IntegratedSmartHomeSystem:
    """
    Integrated Smart Home System that combines:
    1. Command Parsing (from kg_builder_and_command_processor.py)
    2. Risk Prediction (from risk_prediction_module.py)
    3. Decision Making based on risk assessment
    
    This system processes natural language commands, converts them to structured actions,
    assesses risks, and provides recommendations for safe smart home operation.
    """
    
    def __init__(self, home_system_path: str, commands_path: str):
        # Load knowledge graph data
        self.home_system_data = load_home_system(home_system_path)
        self.commands_data = load_commands(commands_path)
        
        # Initialize risk prediction module
        self.risk_predictor = RiskPredictionModule()
        
        # Decision thresholds
        self.decision_thresholds = {
            "auto_execute": 0.3,    # Commands below this score are auto-executed
            "require_confirmation": 0.7,  # Commands above this require confirmation
            "block_command": 0.8    # Commands above this are blocked
        }
        
        print(f"Integrated Smart Home System initialized")
        print(f"Loaded {len(self.home_system_data.get('Home_System', {}))} areas from knowledge graph")
        print(f"Loaded {len(self.commands_data)} sample commands")
    
    def process_command(self, command_text: str, device: str, location: str) -> dict:
        """
        Process a single command through the complete pipeline:
        1. Parse command to structured action
        2. Assess risk
        3. Make decision
        
        Args:
            command_text: Natural language command
            device: Target device
            location: Target location
        
        Returns:
            Complete processing result with parsed action, risk assessment, and decision
        """
        result = {
            "input": {
                "command_text": command_text,
                "device": device,
                "location": location,
                "timestamp": datetime.now().isoformat()
            },
            "parsed_action": None,
            "risk_assessment": None,
            "decision": None,
            "status": "processing"
        }
        
        try:
            # Step 1: Parse command to structured action
            parsed_action = self._parse_command_to_action(command_text, device, location)
            
            if not parsed_action:
                result["status"] = "failed"
                result["error"] = f"Device '{device}' not found in location '{location}'"
                return result
            
            result["parsed_action"] = parsed_action
            
            # Step 2: Assess risk
            risk_assessment = self.risk_predictor.assess_risk(parsed_action)
            result["risk_assessment"] = risk_assessment
            
            # Step 3: Make decision
            decision = self._make_decision(risk_assessment)
            result["decision"] = decision
            
            result["status"] = "completed"
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        
        return result
    
    def _parse_command_to_action(self, command_text: str, device: str, location: str) -> dict:
        """
        Parse command using the logic from kg_builder_and_command_processor.py
        """
        # Normalize location name
        normalized_location = location.replace(" ", "_").title()
        # Normalize device name to match KG conventions (e.g., 'computer' -> 'Computer', 'smoke detector' -> 'Smoke_Detector')
        normalized_device = device.replace(" ", "_").title()
        
        # Search in Home_System data
        home_system = self.home_system_data.get('Home_System', {})
        
        # First check Whole_House_Systems
        whole_house_systems = home_system.get('Whole_House_Systems', {})
        for system_name, system_devices in whole_house_systems.items():
            if isinstance(system_devices, dict):
                for device_name, device_details in system_devices.items():
                    if normalized_device == device_name or normalized_device.lower() in device_name.lower():
                        return {
                            "area_of_action": "Whole_House_Systems",
                            "device": device_name,
                            "action_to_be_taken": command_text
                        }
        
        # Then check specific rooms
        for room_name, room_data in home_system.items():
            if room_name == 'Whole_House_Systems':
                continue
            
            if room_name == normalized_location and isinstance(room_data, dict):
                for device_name, device_details in room_data.items():
                    # Direct match using normalized_device
                    if normalized_device == device_name:
                        return {
                            "area_of_action": room_name,
                            "device": device_name,
                            "action_to_be_taken": command_text
                        }
                    
                    # Substring match using normalized_device
                    if normalized_device.lower() in device_name.lower():
                        return {
                            "area_of_action": room_name,
                            "device": device_name,
                            "action_to_be_taken": command_text
                        }
        
        return None
    
    def _make_decision(self, risk_assessment: dict) -> dict:
        """
        Make a decision based on risk assessment
        """
        risk_score = risk_assessment["overall_risk_score"]
        risk_level = risk_assessment["risk_level"]
        
        decision = {
            "action": "unknown",
            "reason": "",
            "requires_user_confirmation": False,
            "alternative_suggestions": []
        }
        
        if risk_score >= self.decision_thresholds["block_command"]:
            decision["action"] = "block"
            decision["reason"] = f"Command blocked due to {risk_level} risk (score: {risk_score})"
            decision["alternative_suggestions"] = self._generate_alternatives(risk_assessment)
            
        elif risk_score >= self.decision_thresholds["require_confirmation"]:
            decision["action"] = "request_confirmation"
            decision["reason"] = f"Command requires confirmation due to {risk_level} risk (score: {risk_score})"
            decision["requires_user_confirmation"] = True
            
        elif risk_score >= self.decision_thresholds["auto_execute"]:
            decision["action"] = "execute_with_monitoring"
            decision["reason"] = f"Command will be executed with monitoring due to {risk_level} risk (score: {risk_score})"
            
        else:
            decision["action"] = "execute"
            decision["reason"] = f"Command approved for execution ({risk_level} risk, score: {risk_score})"
        
        return decision
    
    def _generate_alternatives(self, risk_assessment: dict) -> list:
        """
        Generate alternative suggestions for high-risk commands
        """
        alternatives = []
        command = risk_assessment["command"]
        device = command["device"].lower()
        action = command["action_to_be_taken"].lower()
        
        # Temperature-related alternatives
        if "temperature" in action or any(temp_device in device for temp_device in ["oven", "heater", "fireplace"]):
            alternatives.append("Consider reducing temperature to a safer level (below 400°F)")
            alternatives.append("Set a shorter duration timer")
            alternatives.append("Enable automatic shutoff after 2 hours")
        
        # Security-related alternatives
        if any(sec_device in device for sec_device in ["door", "lock", "alarm", "camera"]):
            alternatives.append("Use temporary access codes instead of disabling security")
            alternatives.append("Set automatic re-enable timer")
            alternatives.append("Enable additional monitoring during security system changes")
        
        # General safety alternatives
        alternatives.append("Schedule command for daytime hours")
        alternatives.append("Enable remote monitoring and alerts")
        alternatives.append("Require periodic check-ins during operation")
        
        return alternatives
    
    def process_batch_commands(self, commands: list) -> list:
        """
        Process multiple commands and return results
        """
        results = []
        
        for cmd in commands:
            if isinstance(cmd, dict) and all(key in cmd for key in ["device", "location", "action"]):
                result = self.process_command(cmd["action"], cmd["device"], cmd["location"])
                results.append(result)
            else:
                results.append({
                    "input": cmd,
                    "status": "error",
                    "error": "Invalid command format. Expected keys: device, location, action"
                })
        
        return results
    
    def get_system_status(self) -> dict:
        """
        Get current system status and statistics
        """
        return {
            "system_name": "Integrated Smart Home System",
            "components": {
                "command_parser": "Active",
                "risk_predictor": "Active",
                "decision_engine": "Active"
            },
            "knowledge_graph": {
                "total_areas": len(self.home_system_data.get('Home_System', {})),
                "sample_commands": len(self.commands_data)
            },
            "decision_thresholds": self.decision_thresholds,
            "timestamp": datetime.now().isoformat()
        }


def demo_integrated_system():
    """
    Demonstrate the integrated smart home system
    """
    print("=== Integrated Smart Home System Demo ===")
    print()
    
    # Initialize system
    system = IntegratedSmartHomeSystem(
        home_system_path="commandData/homeSystem.json",
        commands_path="commandData/120datasetmixJson.json"
    )
    
    print("\n" + "="*60)
    print("SYSTEM STATUS")
    print("="*60)
    status = system.get_system_status()
    print(json.dumps(status, indent=2))
    
    print("\n" + "="*60)
    print("PROCESSING SAMPLE COMMANDS")
    print("="*60)
    
    # Test commands with varying risk levels
    test_commands = [
        {
            "device": "computer",
            "location": "study",
            "action": "Turn on the computer"
        },
        {
            "device": "oven",
            "location": "kitchen",
            "action": "Set oven to 500°F for 8 hours"
        },
        {
            "device": "smoke_detector",
            "location": "living room",
            "action": "Disable smoke detector"
        },
        {
            "device": "door_lock",
            "location": "front door",
            "action": "Unlock front door for 3 hours"
        }
    ]
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\nCommand {i}: {cmd['action']} ({cmd['device']} in {cmd['location']})")
        print("-" * 50)
        
        result = system.process_command(cmd["action"], cmd["device"], cmd["location"])
        
        if result["status"] == "completed":
            print(f"✓ Status: {result['status']}")
            print(f"✓ Parsed Action: {result['parsed_action']}")
            print(f"✓ Risk Level: {result['risk_assessment']['risk_level']} (Score: {result['risk_assessment']['overall_risk_score']})")
            print(f"✓ Decision: {result['decision']['action']}")
            print(f"✓ Reason: {result['decision']['reason']}")
            
            if result['risk_assessment']['warnings']:
                print("⚠️  Warnings:")
                for warning in result['risk_assessment']['warnings']:
                    print(f"   - {warning}")
            
            if result['decision']['alternative_suggestions']:
                print("💡 Alternative Suggestions:")
                for suggestion in result['decision']['alternative_suggestions']:
                    print(f"   - {suggestion}")
        else:
            print(f"❌ Status: {result['status']}")
            if "error" in result:
                print(f"❌ Error: {result['error']}")


if __name__ == "__main__":
    demo_integrated_system()