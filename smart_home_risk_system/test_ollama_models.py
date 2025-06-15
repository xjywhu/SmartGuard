#!/usr/bin/env python3
"""
Test script to compare different Ollama models for the Smart Home Risk Assessment System.

This script tests various Ollama models and compares their performance on:
1. Command parsing accuracy
2. State forecasting quality
3. Risk evaluation consistency
"""

import json
import logging
import time
from typing import Dict, List, Any
from modules.command_parser import CommandParser
from modules.state_forecaster import StateForecaster
from modules.risk_evaluator import RiskEvaluator
from modules.ollama_client import OllamaClient
from utils.context_utils import load_context

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test commands for evaluation
TEST_COMMANDS = [
    "Turn off the smoke detector",
    "Open all windows",
    "Turn on the heater",
    "Lock all doors",
    "Turn off the security system",
    "Start the dishwasher",
    "Turn on the air conditioning",
    "Dim the living room lights",
    "Turn off the water main valve",
    "Activate the sprinkler system"
]

# Available models to test
MODELS_TO_TEST = [
    "llama3.2:latest",
    "llama3:latest", 
    "llama2:latest",
    "mistral:latest"
]

def test_model_availability():
    """Test which models are available in Ollama."""
    logger.info("Testing Ollama model availability...")
    
    client = OllamaClient()
    if not client.is_available():
        logger.error("Ollama server is not available!")
        return []
    
    available_models = client.list_models()
    logger.info(f"Available models: {available_models}")
    
    # Filter to only test models that are available
    models_to_test = [model for model in MODELS_TO_TEST if model in available_models]
    logger.info(f"Models to test: {models_to_test}")
    
    return models_to_test

def test_command_parsing(models: List[str]) -> Dict[str, Dict]:
    """Test command parsing with different models."""
    logger.info("Testing command parsing with different models...")
    
    results = {}
    
    for model in models:
        logger.info(f"Testing command parsing with {model}...")
        parser = CommandParser(use_llm=True, model=model)
        
        model_results = {
            "successful_parses": 0,
            "failed_parses": 0,
            "total_time": 0,
            "results": []
        }
        
        for command in TEST_COMMANDS:
            start_time = time.time()
            try:
                result = parser.parse(command)
                end_time = time.time()
                
                # Check if parsing was successful (not fallback)
                if result.get('device') != 'UNKNOWN':
                    model_results["successful_parses"] += 1
                else:
                    model_results["failed_parses"] += 1
                
                model_results["results"].append({
                    "command": command,
                    "result": result,
                    "time": end_time - start_time
                })
                model_results["total_time"] += end_time - start_time
                
            except Exception as e:
                logger.error(f"Error parsing '{command}' with {model}: {e}")
                model_results["failed_parses"] += 1
        
        results[model] = model_results
        logger.info(f"{model}: {model_results['successful_parses']}/{len(TEST_COMMANDS)} successful parses")
    
    return results

def test_state_forecasting(models: List[str]) -> Dict[str, Dict]:
    """Test state forecasting with different models."""
    logger.info("Testing state forecasting with different models...")
    
    # Load sample context
    try:
        context = load_context("data/sample_context.json")
    except:
        # Fallback context if file not found
        context = {
            "user_home": True,
            "user_asleep": False,
            "time": "14:30",
            "devices": {
                "living_room_lights": "ON",
                "heater": "OFF",
                "security_system": "ARMED"
            },
            "sensors": {
                "temperature_indoor": 22,
                "smoke_detector": "NORMAL"
            }
        }
    
    # Sample parsed command
    sample_command = {
        "device": "heater",
        "action": "on",
        "location": "living_room",
        "confidence": 0.9
    }
    
    results = {}
    
    for model in models:
        logger.info(f"Testing state forecasting with {model}...")
        forecaster = StateForecaster(use_llm=True, model=model)
        
        start_time = time.time()
        try:
            result = forecaster.forecast(sample_command, context)
            end_time = time.time()
            
            results[model] = {
                "result": result,
                "time": end_time - start_time,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error forecasting with {model}: {e}")
            results[model] = {
                "error": str(e),
                "success": False
            }
    
    return results

def test_risk_evaluation(models: List[str]) -> Dict[str, Dict]:
    """Test risk evaluation with different models."""
    logger.info("Testing risk evaluation with different models...")
    
    # Load sample context
    try:
        context = load_context("data/sample_context.json")
    except:
        context = {
            "user_home": False,  # User not home - should increase risk
            "user_asleep": False,
            "time": "14:30",
            "devices": {
                "security_system": "ARMED",
                "smoke_detector": "NORMAL"
            }
        }
    
    # High-risk command: turning off smoke detector
    risky_command = {
        "device": "smoke_detector",
        "action": "off",
        "location": "house",
        "confidence": 0.95
    }
    
    results = {}
    
    for model in models:
        logger.info(f"Testing risk evaluation with {model}...")
        evaluator = RiskEvaluator(use_llm=True, model=model)
        
        start_time = time.time()
        try:
            result = evaluator.evaluate(risky_command, context, context)
            end_time = time.time()
            
            results[model] = {
                "result": result,
                "time": end_time - start_time,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error evaluating risk with {model}: {e}")
            results[model] = {
                "error": str(e),
                "success": False
            }
    
    return results

def generate_comparison_report(parsing_results: Dict, forecasting_results: Dict, 
                             evaluation_results: Dict) -> str:
    """Generate a comprehensive comparison report."""
    report = []
    report.append("=" * 80)
    report.append("OLLAMA MODELS COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Command Parsing Results
    report.append("📝 COMMAND PARSING RESULTS:")
    report.append("-" * 40)
    for model, results in parsing_results.items():
        success_rate = results['successful_parses'] / len(TEST_COMMANDS) * 100
        avg_time = results['total_time'] / len(TEST_COMMANDS)
        report.append(f"  {model}:")
        report.append(f"    Success Rate: {success_rate:.1f}% ({results['successful_parses']}/{len(TEST_COMMANDS)})")
        report.append(f"    Average Time: {avg_time:.2f}s")
        report.append("")
    
    # State Forecasting Results
    report.append("🔮 STATE FORECASTING RESULTS:")
    report.append("-" * 40)
    for model, results in forecasting_results.items():
        if results['success']:
            report.append(f"  {model}:")
            report.append(f"    Status: ✅ Success")
            report.append(f"    Time: {results['time']:.2f}s")
            report.append(f"    Result: {json.dumps(results['result'], indent=6)}")
        else:
            report.append(f"  {model}:")
            report.append(f"    Status: ❌ Failed")
            report.append(f"    Error: {results.get('error', 'Unknown error')}")
        report.append("")
    
    # Risk Evaluation Results
    report.append("⚠️  RISK EVALUATION RESULTS:")
    report.append("-" * 40)
    for model, results in evaluation_results.items():
        if results['success']:
            risk_level = results['result'].get('risk_level', 'UNKNOWN')
            report.append(f"  {model}:")
            report.append(f"    Status: ✅ Success")
            report.append(f"    Risk Level: {risk_level}")
            report.append(f"    Time: {results['time']:.2f}s")
            report.append(f"    Rationale: {results['result'].get('rationale', 'No rationale provided')}")
        else:
            report.append(f"  {model}:")
            report.append(f"    Status: ❌ Failed")
            report.append(f"    Error: {results.get('error', 'Unknown error')}")
        report.append("")
    
    # Recommendations
    report.append("💡 RECOMMENDATIONS:")
    report.append("-" * 40)
    
    # Find best parsing model
    best_parsing = max(parsing_results.items(), 
                      key=lambda x: x[1]['successful_parses'])
    report.append(f"  Best for Command Parsing: {best_parsing[0]}")
    
    # Find fastest model
    successful_models = [(model, results) for model, results in parsing_results.items() 
                        if results['successful_parses'] > 0]
    if successful_models:
        fastest_model = min(successful_models, 
                           key=lambda x: x[1]['total_time'] / len(TEST_COMMANDS))
        report.append(f"  Fastest Model: {fastest_model[0]}")
    
    # Risk evaluation consistency
    high_risk_models = [model for model, results in evaluation_results.items() 
                       if results.get('success') and 
                       results['result'].get('risk_level') in ['HIGH', 'CRITICAL']]
    if high_risk_models:
        report.append(f"  Models detecting high risk: {', '.join(high_risk_models)}")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    """Main function to run all tests and generate report."""
    logger.info("Starting Ollama models comparison test...")
    
    # Test model availability
    available_models = test_model_availability()
    if not available_models:
        logger.error("No models available for testing!")
        return
    
    # Run tests
    parsing_results = test_command_parsing(available_models)
    forecasting_results = test_state_forecasting(available_models)
    evaluation_results = test_risk_evaluation(available_models)
    
    # Generate and display report
    report = generate_comparison_report(parsing_results, forecasting_results, evaluation_results)
    print(report)
    
    # Save results to file
    results = {
        "parsing": parsing_results,
        "forecasting": forecasting_results,
        "evaluation": evaluation_results,
        "report": report
    }
    
    with open("ollama_models_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Results saved to ollama_models_comparison.json")
    logger.info("Test completed!")

if __name__ == "__main__":
    main()