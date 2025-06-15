#!/usr/bin/env python3
"""
Debug script to see what prompt is being sent to the LLM and what response we get
"""

import sys
from rag_pipeline import RAGPipeline
from modules.ollama_client import OllamaClient

def debug_rag_prompt():
    """Debug the RAG prompt generation and LLM response"""
    
    print("Debug RAG Prompt Generation")
    print("=" * 50)
    
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline()
        
        # Test command
        command = "turn on lights in bedroom"
        print(f"Command: {command}")
        
        # Get enhanced context
        context = rag.context_builder.build_enhanced_context(command)
        print(f"\nContext keys: {list(context.keys())}")
        
        # Retrieve relevant documents
        retrieved_docs = rag.retrieve_context(command)
        print(f"\nRetrieved {len(retrieved_docs)} documents")
        
        # Generate RAG-enhanced prompt
        rag_prompt = rag.generate_rag_prompt(command, context, retrieved_docs)
        
        print("\n" + "="*80)
        print("GENERATED PROMPT:")
        print("="*80)
        print(rag_prompt)
        print("="*80)
        
        # Test with Ollama directly
        print("\nTesting with Ollama directly...")
        ollama_client = OllamaClient()
        
        if not ollama_client.is_available():
            print("❌ Ollama server not available")
            return
        
        print("✅ Ollama server available")
        
        # Get response
        print("\nSending prompt to LLM...")
        response = ollama_client.generate(prompt=rag_prompt, model='llama3.2:latest')
        
        print("\n" + "="*80)
        print("LLM RESPONSE:")
        print("="*80)
        print(response)
        print("="*80)
        
        # Try to parse the response
        print("\nTrying to parse response...")
        from modules.risk_evaluator import RiskEvaluator
        risk_eval = RiskEvaluator()
        
        try:
            parsed = risk_eval._extract_risk_assessment(response)
            print(f"✅ Successfully parsed: {parsed}")
        except Exception as e:
            print(f"❌ Failed to parse: {e}")
            
            # Try to find JSON in the response
            import re
            json_pattern = r'\{[^{}]*"risk_level"[^{}]*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            if matches:
                print(f"Found potential JSON: {matches[0]}")
            else:
                print("No JSON pattern found in response")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_simple_prompt():
    """Test with a very simple prompt to see if the issue is with complexity"""
    
    print("\n" + "="*80)
    print("TESTING SIMPLE PROMPT")
    print("="*80)
    
    simple_prompt = '''Analyze this smart home command for safety risks:

Command: "turn on lights in bedroom"

Return only a JSON object with:
- "risk_level": "LOW", "MEDIUM", or "HIGH"
- "rationale": brief explanation

Example: {"risk_level": "LOW", "rationale": "Normal lighting operation"}

Response:'''
    
    print("Simple prompt:")
    print(simple_prompt)
    
    try:
        ollama_client = OllamaClient()
        response = ollama_client.generate(prompt=simple_prompt, model='llama3.2:latest')
        
        print("\nSimple prompt response:")
        print(response)
        
        # Try to parse
        import json
        try:
            parsed = json.loads(response.strip())
            print(f"✅ Simple prompt parsed successfully: {parsed}")
        except:
            print("❌ Simple prompt also failed to parse as JSON")
            
    except Exception as e:
        print(f"Error with simple prompt: {e}")

if __name__ == "__main__":
    debug_rag_prompt()
    test_simple_prompt()