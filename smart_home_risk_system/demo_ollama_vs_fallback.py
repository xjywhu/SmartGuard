#!/usr/bin/env python3
"""
Demo script comparing Ollama LLM performance vs fallback methods.

This script demonstrates:
1. How the system works with Ollama models
2. Comparison between LLM and fallback methods
3. Performance metrics for different approaches
"""

import json
import time
import logging
from typing import Dict, Any
from modules.command_parser import CommandParser
from modules.state_forecaster import StateForecaster
from modules.risk_evaluator import RiskEvaluator
from modules.ollama_client import OllamaClient
from utils.context_utils import load_context

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test commands with varying complexity
TEST_COMMANDS = [
    "Turn off the smoke detector",
    "Open the living room window", 
    "Set the thermostat to 72 degrees",
    "Lock all exterior doors",
    "Turn on the security system",
    "Dim the bedroom lights to 30%",
    "Start the washing machine",
    "Turn off all lights in the house",
    "Close the garage door",
    "Activate the sprinkler system"
]

def test_ollama_availability():
    """Test if Ollama is available and which models are ready."""
    print("🔍 Testing Ollama Availability...")
    print("=" * 50)
    
    client = OllamaClient()
    if not client.is_available():
        print("❌ Ollama server is not available!")
        print("   Please make sure Ollama is running: ollama serve")
        return False, []
    
    print("✅ Ollama server is available!")
    
    models = client.list_models()
    print(f"📋 Available models: {models}")
    
    # Test each model
    working_models = []
    for model in models:
        try:
            response = client.generate("Hello", model=model)
            if response.strip():
                working_models.append(model)
                print(f"✅ {model}: Working")
            else:
                print(f"⚠️  {model}: No response")
        except Exception as e:
            print(f"❌ {model}: Error - {e}")
    
    print(f"\n🎯 Working models: {working_models}")
    return True, working_models

def compare_parsing_methods(command: str, model: str = "llama3.2:latest"):
    """Compare LLM vs fallback parsing for a single command."""
    print(f"\n📝 Parsing Comparison for: '{command}'")
    print("-" * 60)
    
    # Test with Ollama LLM
    print("🤖 Testing with Ollama LLM...")
    llm_parser = CommandParser(use_llm=True, model=model)
    
    start_time = time.time()
    llm_result = llm_parser.parse(command)
    llm_time = time.time() - start_time
    
    print(f"   Result: {json.dumps(llm_result, indent=4)}")
    print(f"   Time: {llm_time:.2f}s")
    
    # Test with fallback
    print("\n🔧 Testing with Fallback Method...")
    fallback_parser = CommandParser(use_llm=False)
    
    start_time = time.time()
    fallback_result = fallback_parser.parse(command)
    fallback_time = time.time() - start_time
    
    print(f"   Result: {json.dumps(fallback_result, indent=4)}")
    print(f"   Time: {fallback_time:.2f}s")
    
    # Compare results
    print("\n📊 Comparison:")
    print(f"   LLM Method: {llm_result.get('method', 'unknown')}")
    print(f"   LLM Device: {llm_result.get('device', 'unknown')}")
    print(f"   LLM Action: {llm_result.get('action', 'unknown')}")
    print(f"   LLM Confidence: {llm_result.get('confidence', 0)}")
    print(f"   LLM Time: {llm_time:.2f}s")
    print()
    print(f"   Fallback Device: {fallback_result.get('device', 'unknown')}")
    print(f"   Fallback Action: {fallback_result.get('action', 'unknown')}")
    print(f"   Fallback Confidence: {fallback_result.get('confidence', 0)}")
    print(f"   Fallback Time: {fallback_time:.2f}s")
    
    return llm_result, fallback_result, llm_time, fallback_time

def test_full_pipeline(command: str, context: Dict[str, Any], models: Dict[str, str]):
    """Test the full pipeline with different models."""
    print(f"\n🔄 Full Pipeline Test for: '{command}'")
    print("=" * 70)
    
    results = {}
    
    # Test with Ollama models
    print("🤖 Testing with Ollama Models...")
    parser = CommandParser(use_llm=True, model=models['parser'])
    forecaster = StateForecaster(use_llm=True, model=models['forecaster'])
    evaluator = RiskEvaluator(use_llm=True, model=models['evaluator'])
    
    start_time = time.time()
    
    # Step 1: Parse command
    parsed_command = parser.parse(command)
    parse_time = time.time() - start_time
    
    # Step 2: Forecast state changes
    forecast_start = time.time()
    predicted_context = forecaster.forecast(parsed_command, context)
    forecast_time = time.time() - forecast_start
    
    # Step 3: Evaluate risk
    eval_start = time.time()
    risk_assessment = evaluator.evaluate(parsed_command, context, predicted_context)
    eval_time = time.time() - eval_start
    
    total_time = time.time() - start_time
    
    results['ollama'] = {
        'parsed_command': parsed_command,
        'predicted_context': predicted_context,
        'risk_assessment': risk_assessment,
        'times': {
            'parse': parse_time,
            'forecast': forecast_time,
            'evaluate': eval_time,
            'total': total_time
        }
    }
    
    print(f"   ✅ Completed in {total_time:.2f}s")
    print(f"   📝 Parse: {parse_time:.2f}s")
    print(f"   🔮 Forecast: {forecast_time:.2f}s")
    print(f"   ⚠️  Evaluate: {eval_time:.2f}s")
    print(f"   🎯 Risk Level: {risk_assessment.get('risk_level', 'UNKNOWN')}")
    
    # Test with fallback methods
    print("\n🔧 Testing with Fallback Methods...")
    fallback_parser = CommandParser(use_llm=False)
    fallback_forecaster = StateForecaster(use_llm=False)
    fallback_evaluator = RiskEvaluator(use_llm=False)
    
    start_time = time.time()
    
    # Step 1: Parse command
    fallback_parsed = fallback_parser.parse(command)
    fallback_parse_time = time.time() - start_time
    
    # Step 2: Forecast state changes
    fallback_forecast_start = time.time()
    fallback_predicted = fallback_forecaster.forecast(fallback_parsed, context)
    fallback_forecast_time = time.time() - fallback_forecast_start
    
    # Step 3: Evaluate risk
    fallback_eval_start = time.time()
    fallback_risk = fallback_evaluator.evaluate(fallback_parsed, context, fallback_predicted)
    fallback_eval_time = time.time() - fallback_eval_start
    
    fallback_total_time = time.time() - start_time
    
    results['fallback'] = {
        'parsed_command': fallback_parsed,
        'predicted_context': fallback_predicted,
        'risk_assessment': fallback_risk,
        'times': {
            'parse': fallback_parse_time,
            'forecast': fallback_forecast_time,
            'evaluate': fallback_eval_time,
            'total': fallback_total_time
        }
    }
    
    print(f"   ✅ Completed in {fallback_total_time:.2f}s")
    print(f"   📝 Parse: {fallback_parse_time:.2f}s")
    print(f"   🔮 Forecast: {fallback_forecast_time:.2f}s")
    print(f"   ⚠️  Evaluate: {fallback_eval_time:.2f}s")
    print(f"   🎯 Risk Level: {fallback_risk.get('risk_level', 'UNKNOWN')}")
    
    # Comparison
    print("\n📊 Performance Comparison:")
    print(f"   Speed: Ollama {total_time:.2f}s vs Fallback {fallback_total_time:.2f}s")
    
    if fallback_total_time > 0:
        speedup_ratio = total_time / fallback_total_time
        speedup_text = f"{speedup_ratio:.1f}x {'(Ollama slower)' if total_time > fallback_total_time else '(Ollama faster)'}"
    else:
        speedup_text = "Cannot calculate (fallback time is 0)"
    print(f"   Speedup: {speedup_text}")
    
    ollama_risk = risk_assessment.get('risk_level', 'UNKNOWN')
    fallback_risk_level = fallback_risk.get('risk_level', 'UNKNOWN')
    print(f"   Risk Assessment: Ollama={ollama_risk} vs Fallback={fallback_risk_level}")
    
    return results

def main():
    """Main demo function."""
    print("🏠 Smart Home Risk Assessment System - Ollama vs Fallback Demo")
    print("=" * 80)
    
    # Test Ollama availability
    ollama_available, working_models = test_ollama_availability()
    
    if not ollama_available or not working_models:
        print("\n❌ Cannot proceed without Ollama. Please ensure Ollama is running.")
        return
    
    # Select models for testing
    models = {
        'parser': working_models[0] if working_models else "llama3.2:latest",
        'forecaster': working_models[0] if working_models else "llama3.2:latest", 
        'evaluator': working_models[-1] if len(working_models) > 1 else working_models[0]
    }
    
    print(f"\n🎯 Selected Models:")
    print(f"   Parser: {models['parser']}")
    print(f"   Forecaster: {models['forecaster']}")
    print(f"   Evaluator: {models['evaluator']}")
    
    # Load sample context
    try:
        context = load_context("data/sample_context.json")
        print(f"\n📋 Loaded context from data/sample_context.json")
    except:
        context = {
            "user_home": True,
            "user_asleep": False,
            "time": "14:30",
            "devices": {
                "smoke_detector": "NORMAL",
                "security_system": "ARMED",
                "living_room_lights": "ON"
            },
            "sensors": {
                "temperature_indoor": 22,
                "air_quality": "GOOD"
            }
        }
        print(f"\n📋 Using default context")
    
    print(f"\n📄 Context: {json.dumps(context, indent=2)}")
    
    # Test individual parsing comparison
    print("\n" + "=" * 80)
    print("PART 1: COMMAND PARSING COMPARISON")
    print("=" * 80)
    
    test_command = "Turn off the smoke detector"
    compare_parsing_methods(test_command, models['parser'])
    
    # Test full pipeline
    print("\n" + "=" * 80)
    print("PART 2: FULL PIPELINE COMPARISON")
    print("=" * 80)
    
    pipeline_results = test_full_pipeline(test_command, context, models)
    
    # Test multiple commands
    print("\n" + "=" * 80)
    print("PART 3: MULTIPLE COMMANDS SUMMARY")
    print("=" * 80)
    
    summary_results = []
    
    for i, cmd in enumerate(TEST_COMMANDS[:5]):  # Test first 5 commands
        print(f"\n🔄 Testing {i+1}/5: {cmd}")
        
        # Quick test with Ollama
        parser = CommandParser(use_llm=True, model=models['parser'])
        evaluator = RiskEvaluator(use_llm=True, model=models['evaluator'])
        
        start_time = time.time()
        parsed = parser.parse(cmd)
        risk = evaluator.evaluate(parsed, context)
        ollama_time = time.time() - start_time
        
        # Quick test with fallback
        fallback_parser = CommandParser(use_llm=False)
        fallback_evaluator = RiskEvaluator(use_llm=False)
        
        start_time = time.time()
        fallback_parsed = fallback_parser.parse(cmd)
        fallback_risk = fallback_evaluator.evaluate(fallback_parsed, context)
        fallback_time = time.time() - start_time
        
        summary_results.append({
            'command': cmd,
            'ollama': {
                'time': ollama_time,
                'risk': risk.get('risk_level', 'UNKNOWN'),
                'method': parsed.get('method', 'unknown')
            },
            'fallback': {
                'time': fallback_time,
                'risk': fallback_risk.get('risk_level', 'UNKNOWN'),
                'method': fallback_parsed.get('method', 'fallback')
            }
        })
        
        print(f"   Ollama: {risk.get('risk_level', 'UNKNOWN')} ({ollama_time:.2f}s)")
        print(f"   Fallback: {fallback_risk.get('risk_level', 'UNKNOWN')} ({fallback_time:.2f}s)")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    ollama_times = [r['ollama']['time'] for r in summary_results]
    fallback_times = [r['fallback']['time'] for r in summary_results]
    
    print(f"📊 Performance Summary:")
    if ollama_times:
        print(f"   Average Ollama Time: {sum(ollama_times)/len(ollama_times):.2f}s")
    else:
        print(f"   Average Ollama Time: No data")
    
    if fallback_times:
        print(f"   Average Fallback Time: {sum(fallback_times)/len(fallback_times):.2f}s")
    else:
        print(f"   Average Fallback Time: No data")
    
    if sum(fallback_times) > 0:
        print(f"   Speed Ratio: {sum(ollama_times)/sum(fallback_times):.1f}x")
    else:
        print(f"   Speed Ratio: Cannot calculate (no fallback time data)")
    
    # Count LLM vs fallback usage
    llm_usage = sum(1 for r in summary_results if r['ollama']['method'] == 'llm')
    print(f"\n🤖 LLM Usage: {llm_usage}/{len(summary_results)} commands used actual LLM")
    print(f"🔧 Fallback Usage: {len(summary_results)-llm_usage}/{len(summary_results)} commands used fallback")
    
    # Risk level comparison
    risk_matches = sum(1 for r in summary_results if r['ollama']['risk'] == r['fallback']['risk'])
    print(f"\n⚖️  Risk Assessment Agreement: {risk_matches}/{len(summary_results)} commands had matching risk levels")
    
    print("\n✅ Demo completed! Check the detailed logs above for insights.")
    print("\n💡 Key Takeaways:")
    print("   - Ollama provides more nuanced parsing when working properly")
    print("   - Fallback methods are faster but less sophisticated")
    print("   - Both approaches can provide reasonable risk assessments")
    print("   - The system gracefully falls back when LLM is unavailable")

if __name__ == "__main__":
    main()