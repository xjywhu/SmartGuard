#!/usr/bin/env python3
"""
Test script for the Smart Home Risk Assessment System

This script tests the basic functionality of the system without requiring OpenAI API.
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from main import SmartHomeRiskSystem
from utils.context_utils import create_sample_context

def test_basic_functionality():
    """Test basic system functionality with fallback parsers."""
    print("Testing Smart Home Risk Assessment System...")
    print("=" * 50)
    
    # Initialize system
    system = SmartHomeRiskSystem()
    
    # Create sample context
    context = create_sample_context()
    print("\n1. Sample Context Created:")
    print(json.dumps(context, indent=2))
    
    # Test commands
    test_commands = [
        "Turn off the smoke detector",
        "Turn on the living room light",
        "Open the bedroom window",
        "Start the washing machine",
        "Turn off the camera"
    ]
    
    print("\n2. Testing Commands:")
    print("-" * 30)
    
    for i, command in enumerate(test_commands, 1):
        print(f"\nTest {i}: '{command}'")
        try:
            result = system.process_command(command, context)
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  Action: {result['action']}")
            print(f"  Rationale: {result['rationale'][:100]}..." if len(result['rationale']) > 100 else f"  Rationale: {result['rationale']}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n3. Testing Dataset Processing:")
    print("-" * 30)
    
    # Test with a small subset of the dataset
    dataset_path = Path(__file__).parent / "data" / "risky_smart_home_commands_dataset.json"
    if dataset_path.exists():
        try:
            results = system.test_dataset(str(dataset_path), max_samples=3)
            print(f"  Processed {len(results)} samples from dataset")
            
            for i, result in enumerate(results[:3], 1):
                print(f"\n  Sample {i}:")
                print(f"    Command: {result['command']}")
                print(f"    Predicted Risk: {result['predicted_risk']}")
                print(f"    Expected Risk: {result['expected_risk']}")
                print(f"    Match: {'✓' if result['predicted_risk'] == result['expected_risk'] else '✗'}")
        except Exception as e:
            print(f"  Error processing dataset: {e}")
    else:
        print(f"  Dataset not found at {dataset_path}")
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("\nNote: This test uses fallback parsers since OpenAI API is not configured.")
    print("To use LLM-based parsing, set the OPENAI_API_KEY environment variable.")

def test_individual_modules():
    """Test individual modules separately."""
    print("\nTesting Individual Modules:")
    print("=" * 30)
    
    # Test Command Parser
    print("\n1. Command Parser:")
    from modules.command_parser import CommandParser
    parser = CommandParser()
    
    test_command = "Turn off the smoke detector"
    parsed = parser.parse(test_command)
    print(f"  Input: '{test_command}'")
    print(f"  Output: {json.dumps(parsed, indent=4)}")
    
    # Test State Forecaster
    print("\n2. State Forecaster:")
    from modules.state_forecaster import StateForecaster
    forecaster = StateForecaster()
    
    context = create_sample_context()
    forecasted = forecaster.forecast(parsed, context)
    print(f"  Forecasted changes applied to context")
    print(f"  Smoke detector status: {forecasted.get('devices', {}).get('smoke_detector', 'Unknown')}")
    
    # Test Risk Evaluator
    print("\n3. Risk Evaluator:")
    from modules.risk_evaluator import RiskEvaluator
    evaluator = RiskEvaluator()
    
    risk_result = evaluator.evaluate(parsed, forecasted)
    print(f"  Risk Level: {risk_result['risk_level']}")
    print(f"  Rationale: {risk_result['rationale']}")

if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_individual_modules()
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()