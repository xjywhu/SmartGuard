#!/usr/bin/env python3
"""
Clean test of RAG functionality without using existing pipeline
"""

import json
import time
from modules.ollama_client import OllamaClient
from enhanced_context_builder import EnhancedContextBuilder

def test_clean_risk_assessment():
    """Test risk assessment with clean prompts"""
    
    print("Clean RAG Risk Assessment Test")
    print("=" * 50)
    
    # Initialize components
    ollama_client = OllamaClient()
    
    if not ollama_client.is_available():
        print("❌ Ollama not available")
        return
    
    print("✅ Ollama available")
    
    # Test cases
    test_cases = [
        {
            'command': 'turn on lights in bedroom',
            'expected': 'LOW',
            'context': 'Normal lighting operation in residential area.'
        },
        {
            'command': 'turn off smoke detector',
            'expected': 'HIGH', 
            'context': 'Smoke detectors are critical safety devices that detect fires and smoke.'
        },
        {
            'command': 'set thermostat to 20°C',
            'expected': 'LOW',
            'context': 'Normal temperature control within comfortable range.'
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test_case['command']}")
        print(f"Expected: {test_case['expected']}")
        
        # Create a very simple, direct prompt
        prompt = f"""Assess smart home command safety:

Command: {test_case['command']}
Context: {test_case['context']}

Safety levels:
- HIGH: Disables safety devices, immediate danger
- MEDIUM: Energy waste, minor issues
- LOW: Normal safe operation

Return JSON: {{"risk_level": "HIGH|MEDIUM|LOW", "rationale": "reason"}}

JSON:"""
        
        print(f"\nSending prompt to LLM...")
        
        try:
            # Get response from LLM
            response = ollama_client.generate(prompt, model='mistral:latest')
            print(f"Raw response: {response[:100]}...")
            
            # Parse JSON response
            try:
                # Clean response
                response_clean = response.strip()
                
                # Find JSON in response
                if '{' in response_clean and '}' in response_clean:
                    start_idx = response_clean.find('{')
                    end_idx = response_clean.rfind('}') + 1
                    json_str = response_clean[start_idx:end_idx]
                    
                    assessment = json.loads(json_str)
                    
                    if 'risk_level' in assessment and 'rationale' in assessment:
                        predicted = assessment['risk_level']
                        rationale = assessment['rationale']
                        correct = predicted == test_case['expected']
                        
                        print(f"✅ Parsed successfully")
                        print(f"Predicted: {predicted}")
                        print(f"Correct: {'✅' if correct else '❌'}")
                        print(f"Rationale: {rationale}")
                        
                        results.append({
                            'command': test_case['command'],
                            'expected': test_case['expected'],
                            'predicted': predicted,
                            'correct': correct,
                            'rationale': rationale
                        })
                    else:
                        print("❌ Missing required fields in JSON")
                        results.append({
                            'command': test_case['command'],
                            'expected': test_case['expected'],
                            'predicted': 'ERROR',
                            'correct': False,
                            'rationale': 'Missing fields'
                        })
                else:
                    print("❌ No JSON found in response")
                    results.append({
                        'command': test_case['command'],
                        'expected': test_case['expected'],
                        'predicted': 'ERROR',
                        'correct': False,
                        'rationale': 'No JSON found'
                    })
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                results.append({
                    'command': test_case['command'],
                    'expected': test_case['expected'],
                    'predicted': 'ERROR',
                    'correct': False,
                    'rationale': f'JSON error: {e}'
                })
                
        except Exception as e:
            print(f"❌ LLM error: {e}")
            results.append({
                'command': test_case['command'],
                'expected': test_case['expected'],
                'predicted': 'ERROR',
                'correct': False,
                'rationale': f'LLM error: {e}'
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("CLEAN RAG TEST SUMMARY")
    print(f"{'='*60}")
    
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"Accuracy: {correct_count}/{total_count} ({accuracy:.1%})")
    
    for result in results:
        status = "✅" if result['correct'] else "❌"
        print(f"{status} {result['command']}: {result['predicted']} (expected {result['expected']})")
    
    if accuracy >= 0.8:
        print("\n🎉 Excellent! Clean approach is working well.")
    elif accuracy >= 0.6:
        print("\n✅ Good! Clean approach shows promise.")
    else:
        print("\n⚠️ Clean approach needs refinement.")
    
    return results

def test_different_models():
    """Test with different models to see if one works better"""
    
    print(f"\n{'='*60}")
    print("TESTING DIFFERENT MODELS")
    print(f"{'='*60}")
    
    ollama_client = OllamaClient()
    models = ['mistral:latest', 'llama3.2:latest', 'llama3:latest']
    
    command = "turn off smoke detector"
    expected = "HIGH"
    
    prompt = f"""Safety assessment:

Command: {command}
Context: Smoke detectors are critical safety devices.

Return JSON: {{"risk_level": "HIGH", "rationale": "Disabling safety device"}}

JSON:"""
    
    for model in models:
        try:
            print(f"\nTesting {model}...")
            response = ollama_client.generate(prompt, model=model)
            
            # Check if response contains proper JSON
            if '{' in response and 'risk_level' in response and 'HIGH' in response:
                print(f"✅ {model}: Good response")
            else:
                print(f"❌ {model}: Poor response - {response[:50]}...")
                
        except Exception as e:
            print(f"❌ {model}: Error - {e}")

if __name__ == "__main__":
    test_clean_risk_assessment()
    test_different_models()