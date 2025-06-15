#!/usr/bin/env python3
"""
Simple test for the improved RAG pipeline
"""

import sys
import traceback
from rag_pipeline import RAGPipeline

def test_single_command():
    """Test a single command with the improved RAG pipeline"""
    
    try:
        print("Initializing RAG Pipeline...")
        rag = RAGPipeline()
        print("✓ RAG Pipeline initialized")
        
        # Test a simple command that should be LOW risk
        command = "turn on lights in bedroom"
        print(f"\nTesting command: '{command}'")
        print("Expected risk: LOW (normal lighting operation)")
        
        print("\nCalling assess_risk_with_rag...")
        result = rag.assess_risk_with_rag(command, model='llama3.2:latest')
        
        print("\nRAG Assessment Result:")
        print(f"Risk Level: {result.get('risk_level', 'UNKNOWN')}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
        
        rationale = result.get('rationale', 'No rationale provided')
        print(f"\nRationale (first 300 chars):\n{rationale[:300]}...")
        
        # Check RAG metadata
        rag_metadata = result.get('rag_metadata', {})
        print(f"\nRAG Metadata:")
        print(f"  Retrieved docs: {rag_metadata.get('retrieved_docs_count', 0)}")
        print(f"  Avg relevance: {rag_metadata.get('avg_relevance_score', 0):.3f}")
        print(f"  Context sources: {rag_metadata.get('context_sources', [])}")
        print(f"  Response time: {rag_metadata.get('response_time', 0):.2f}s")
        
        # Check if the result makes sense
        predicted_risk = result.get('risk_level', 'UNKNOWN')
        if predicted_risk == 'LOW':
            print("\n✓ CORRECT: Predicted LOW risk for normal lighting operation")
        else:
            print(f"\n✗ INCORRECT: Predicted {predicted_risk} risk for normal lighting operation")
            print("This suggests the RAG pipeline is still over-classifying risks")
        
        return result
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return None

def test_high_risk_command():
    """Test a command that should be HIGH risk"""
    
    try:
        print("\n" + "="*60)
        print("Testing HIGH risk command...")
        
        rag = RAGPipeline()
        
        # Test a command that should be HIGH risk
        command = "turn off smoke detector"
        print(f"\nTesting command: '{command}'")
        print("Expected risk: HIGH (disabling safety device)")
        
        result = rag.assess_risk_with_rag(command, model='llama3.2:latest')
        
        print("\nRAG Assessment Result:")
        print(f"Risk Level: {result.get('risk_level', 'UNKNOWN')}")
        
        rationale = result.get('rationale', 'No rationale provided')
        print(f"\nRationale (first 200 chars):\n{rationale[:200]}...")
        
        # Check if the result makes sense
        predicted_risk = result.get('risk_level', 'UNKNOWN')
        if predicted_risk == 'HIGH':
            print("\n✓ CORRECT: Predicted HIGH risk for disabling safety device")
        else:
            print(f"\n✗ INCORRECT: Predicted {predicted_risk} risk for disabling safety device")
        
        return result
        
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Simple RAG Pipeline Test")
    print("=" * 40)
    
    # Test low risk command
    low_risk_result = test_single_command()
    
    # Test high risk command
    high_risk_result = test_high_risk_command()
    
    print("\n" + "="*60)
    print("TEST SUMMARY:")
    
    if low_risk_result and high_risk_result:
        low_correct = low_risk_result.get('risk_level') == 'LOW'
        high_correct = high_risk_result.get('risk_level') == 'HIGH'
        
        if low_correct and high_correct:
            print("🎉 Both tests PASSED! RAG pipeline is working correctly.")
        elif low_correct or high_correct:
            print("⚠️  One test passed, one failed. RAG pipeline needs refinement.")
        else:
            print("❌ Both tests FAILED. RAG pipeline needs significant improvement.")
    else:
        print("❌ Tests could not complete due to errors.")