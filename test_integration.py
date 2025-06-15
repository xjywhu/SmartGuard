import json
from integrated_smart_home_system import IntegratedSmartHomeSystem

def test_integration():
    """Test the integrated smart home system and save results to file"""
    
    print("Testing Integrated Smart Home System...")
    
    try:
        # Initialize system
        system = IntegratedSmartHomeSystem(
            home_system_path="commandData/homeSystem.json",
            commands_path="commandData/120datasetmixJson.json"
        )
        
        # Test commands with varying risk levels
        test_commands = [
            {
                "device": "computer",
                "location": "Study_Office",
                "action": "Turn on the computer"
            },
            {
                "device": "oven",
                "location": "kitchen",
                "action": "Set oven to 500°F for 8 hours"
            },
            {
                "device": "smoke_detector",
                "location": "Garage",
                "action": "Disable smoke detector"
            }
        ]
        
        results = []
        
        for i, cmd in enumerate(test_commands, 1):
            print(f"\nProcessing Command {i}: {cmd['action']}")
            
            result = system.process_command(cmd["action"], cmd["device"], cmd["location"])
            results.append({
                "command_number": i,
                "input_command": cmd,
                "result": result
            })
            
            if result["status"] == "completed":
                print(f"✓ Status: {result['status']}")
                print(f"✓ Risk Level: {result['risk_assessment']['risk_level']}")
                print(f"✓ Decision: {result['decision']['action']}")
            else:
                print(f"❌ Status: {result['status']}")
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
        
        # Save results to file
        with open("integration_test_results.json", "w") as f:
            json.dump({
                "test_summary": "Integrated Smart Home System Test Results",
                "system_status": system.get_system_status(),
                "test_results": results
            }, f, indent=2)
        
        print("\n✅ Integration test completed successfully!")
        print("📄 Results saved to integration_test_results.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_integration()