#!/usr/bin/env python3
"""
Test script for the improved RAG pipeline
Tests if the LLM now provides better risk classifications using the enhanced prompts
"""

import json
from rag_pipeline import RAGPipeline

def test_improved_rag():
    """Test the improved RAG pipeline with specific commands"""
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Test commands with expected risk levels
    test_cases = [
        {
            "command": "turn on lights in bedroom",
            "expected_risk": "LOW",
            "description": "Normal lighting operation"
        },
        {
            "command": "turn off smoke detector", 
            "expected_risk": "HIGH",
            "description": "Disabling safety device"
        },
        {
            "command": "set thermostat to 20°C",
            "expected_risk": "LOW", 
            "description": "Normal temperature setting"
        },
        {
            "command": "turn on heater when temperature is 35°C",
            "expected_risk": "HIGH",
            "description": "Heating during hot weather"
        },
        {
            "command": "lock front door",
            "expected_risk": "LOW",
            "description": "Normal security operation"
        }
    ]
    
    print("Testing Improved RAG Pipeline")
    print("=" * 50)
    
    results = []
    correct_predictions = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['command']}")
        print(f"Expected: {test_case['expected_risk']} - {test_case['description']}")
        
        try:
            # Test with improved RAG
            result = rag.assess_risk_with_rag(test_case['command'], model='llama3.2:latest')
            
            predicted_risk = result.get('risk_level', 'UNKNOWN')
            rationale = result.get('rationale', 'No rationale provided')
            
            print(f"Predicted: {predicted_risk}")
            
            # Check if prediction is correct
            is_correct = predicted_risk == test_case['expected_risk']
            if is_correct:
                correct_predictions += 1
                print("✓ CORRECT")
            else:
                print("✗ INCORRECT")
            
            # Show first 200 chars of rationale
            print(f"Rationale: {rationale[:200]}...")
            
            # Check if RAG context was used
            rag_metadata = result.get('rag_metadata', {})
            docs_count = rag_metadata.get('retrieved_docs_count', 0)
            avg_relevance = rag_metadata.get('avg_relevance_score', 0)
            
            print(f"RAG Info: {docs_count} docs retrieved, avg relevance: {avg_relevance:.3f}")
            
            results.append({
                'command': test_case['command'],
                'expected': test_case['expected_risk'],
                'predicted': predicted_risk,
                'correct': is_correct,
                'rag_docs': docs_count,
                'avg_relevance': avg_relevance
            })
            
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'command': test_case['command'],
                'expected': test_case['expected_risk'],
                'predicted': 'ERROR',
                'correct': False,
                'error': str(e)
            })
    
    # Summary
    accuracy = correct_predictions / len(test_cases)
    print(f"\n" + "=" * 50)
    print(f"SUMMARY:")
    print(f"Accuracy: {correct_predictions}/{len(test_cases)} ({accuracy:.1%})")
    print(f"Average RAG docs retrieved: {sum(r.get('rag_docs', 0) for r in results) / len(results):.1f}")
    print(f"Average relevance score: {sum(r.get('avg_relevance', 0) for r in results) / len(results):.3f}")
    
    # Save results
    with open('improved_rag_test_results.json', 'w') as f:
        json.dump({
            'test_summary': {
                'total_tests': len(test_cases),
                'correct_predictions': correct_predictions,
                'accuracy': accuracy,
                'timestamp': '2025-06-14 16:30:00'
            },
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\nResults saved to improved_rag_test_results.json")
    
    if accuracy >= 0.8:
        print("\n🎉 RAG pipeline is working well!")
    elif accuracy >= 0.6:
        print("\n⚠️  RAG pipeline shows improvement but needs refinement")
    else:
        print("\n❌ RAG pipeline still needs significant improvement")

if __name__ == "__main__":
    test_improved_rag()