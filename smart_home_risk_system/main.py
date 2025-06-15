#!/usr/bin/env python3
"""
Smart Home Risk Assessment System - Main Entry Point

This is the main CLI interface for the LLM-based smart home risk assessment system.
It processes commands through a modular pipeline:
1. Command Parser (LLM-based)
2. State Forecaster (LLM-based)
3. Risk Evaluator (LLM-based)
4. Decision Module

Usage:
    python main.py --command "Turn off the smoke detector" --context-file data/context.json
    python main.py --test-dataset data/risky_smart_home_commands_dataset.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Import modules - handle both relative and absolute imports
try:
    from .modules.command_parser import CommandParser
    from .modules.state_forecaster import StateForecaster
    from .modules.risk_evaluator import RiskEvaluator
    from .utils.context_utils import load_context, save_context, create_sample_context
except ImportError:
    try:
        from modules.command_parser import CommandParser
        from modules.state_forecaster import StateForecaster
        from modules.risk_evaluator import RiskEvaluator
        from utils.context_utils import load_context, save_context, create_sample_context
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all modules are properly installed and accessible.")
        raise

class SmartHomeRiskSystem:
    """Main system orchestrator for smart home risk assessment."""
    
    def __init__(self, use_llm=True, model="llama3.2:latest"):
        self.command_parser = CommandParser(use_llm=use_llm, model=model)
        self.state_forecaster = StateForecaster(use_llm=use_llm, model=model)
        self.risk_evaluator = RiskEvaluator(use_llm=use_llm, model="mistral:latest")
    
    def process_command(self, command_text: str, context: dict) -> dict:
        """
        Process a single command through the complete pipeline.
        
        Args:
            command_text: Natural language command
            context: Current smart home context
            
        Returns:
            Complete processing result with risk assessment and decision
        """
        result = {
            "command": command_text,
            "original_context": context.copy(),
            "parsed_command": None,
            "forecasted_context": None,
            "risk_assessment": None,
            "decision": None,
            "status": "processing"
        }
        
        try:
            # Step 1: Parse command
            print(f"\n🔍 Parsing command: '{command_text}'")
            parsed_command = self.command_parser.parse(command_text)
            result["parsed_command"] = parsed_command
            print(f"✅ Parsed: {parsed_command}")
            
            # Step 2: Forecast new state
            print(f"\n🔮 Forecasting state changes...")
            forecasted_context = self.state_forecaster.forecast(parsed_command, context)
            result["forecasted_context"] = forecasted_context
            print(f"✅ Forecasted context updated")
            
            # Step 3: Evaluate risk
            print(f"\n⚠️  Evaluating risk...")
            risk_assessment = self.risk_evaluator.evaluate(parsed_command, forecasted_context)
            result["risk_assessment"] = risk_assessment
            print(f"✅ Risk Level: {risk_assessment['risk_level']}")
            
            # Step 4: Make decision
            decision = self._make_decision(risk_assessment)
            result["decision"] = decision
            print(f"✅ Decision: {decision['action']}")
            
            result["status"] = "completed"
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"❌ Error: {e}")
        
        return result
    
    def _make_decision(self, risk_assessment: dict) -> dict:
        """
        Simple decision logic based on risk level.
        
        Args:
            risk_assessment: Risk evaluation result
            
        Returns:
            Decision with action and rationale
        """
        risk_level = risk_assessment.get("risk_level", "UNKNOWN")
        
        if risk_level == "HIGH":
            action = "BLOCKED"
            rationale = "Command blocked due to high risk level"
        elif risk_level == "MEDIUM":
            action = "REQUIRES_CONFIRMATION"
            rationale = "Command requires user confirmation due to medium risk"
        else:  # LOW or UNKNOWN
            action = "ALLOWED"
            rationale = "Command allowed - low risk detected"
        
        return {
            "action": action,
            "rationale": rationale,
            "risk_level": risk_level
        }
    
    def test_dataset(self, dataset_path: str) -> list:
        """
        Test the system against a dataset of commands.
        
        Args:
            dataset_path: Path to JSON dataset file
            
        Returns:
            List of test results
        """
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        results = []
        
        for i, entry in enumerate(dataset):
            print(f"\n{'='*60}")
            print(f"Testing entry {i+1}/{len(dataset)}: {entry['command']}")
            print(f"{'='*60}")
            
            result = self.process_command(entry['command'], entry['context'])
            
            # Add expected vs actual comparison
            result["expected_risk"] = entry.get("expected_risk")
            result["expected_rationale"] = entry.get("rationale")
            
            # Check if prediction matches expected
            predicted_risk = result.get("risk_assessment", {}).get("risk_level")
            result["prediction_correct"] = predicted_risk == entry.get("expected_risk")
            
            results.append(result)
            
            print(f"Expected: {entry.get('expected_risk')} | Predicted: {predicted_risk} | Correct: {result['prediction_correct']}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Smart Home Risk Assessment System")
    parser.add_argument("--command", help="Single command to process")
    parser.add_argument("--context-file", help="Path to context JSON file")
    parser.add_argument("--test-dataset", help="Path to test dataset JSON file")
    parser.add_argument("--output", help="Output file for results (optional)")
    parser.add_argument("--use-llm", action="store_true", default=True, help="Use LLM for processing")
    
    args = parser.parse_args()
    
    # Check if Ollama is available
    if args.use_llm:
        try:
            from modules.ollama_client import OllamaClient
            ollama_client = OllamaClient()
            if not ollama_client.is_available():
                print("Warning: Ollama server not available. Using fallback methods.")
        except ImportError:
            print("Warning: Ollama client not available. Using fallback methods.")
    
    system = SmartHomeRiskSystem(use_llm=args.use_llm)
    
    if args.test_dataset:
        # Test against dataset
        print(f"🧪 Testing against dataset: {args.test_dataset}")
        results = system.test_dataset(args.test_dataset)
        
        # Calculate accuracy
        correct_predictions = sum(1 for r in results if r.get("prediction_correct", False))
        accuracy = correct_predictions / len(results) if results else 0
        
        print(f"\n📊 Test Results Summary:")
        print(f"Total tests: {len(results)}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2%}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    "summary": {
                        "total_tests": len(results),
                        "correct_predictions": correct_predictions,
                        "accuracy": accuracy
                    },
                    "results": results
                }, f, indent=2)
            print(f"Results saved to: {args.output}")
    
    elif args.command:
        # Process single command
        if not args.context_file:
            print("Error: --context-file required when using --command")
            sys.exit(1)
        
        context = load_context(args.context_file)
        result = system.process_command(args.command, context)
        
        print(f"\n📋 Final Result:")
        print(json.dumps(result, indent=2))
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Result saved to: {args.output}")
    
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python main.py --test-dataset ../risky_smart_home_commands_dataset.json")
        print("  python main.py --command 'Turn off smoke detector' --context-file context.json")

if __name__ == "__main__":
    main()