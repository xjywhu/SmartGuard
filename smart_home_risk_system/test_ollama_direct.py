#!/usr/bin/env python3
"""
Direct test of Ollama to understand what's happening
"""

from modules.ollama_client import OllamaClient
import json

def test_ollama_direct():
    """Test Ollama directly with simple prompts"""
    
    client = OllamaClient()
    
    if not client.is_available():
        print("❌ Ollama not available")
        return
    
    print("✅ Ollama available")
    print(f"Available models: {client.list_models()}")
    
    # Test 1: Very simple prompt
    print("\n" + "="*60)
    print("TEST 1: Very simple prompt")
    print("="*60)
    
    simple_prompt = "What is 2+2? Answer with just the number."
    response1 = client.generate(simple_prompt, model='llama3.2:latest')
    print(f"Prompt: {simple_prompt}")
    print(f"Response: {response1}")
    
    # Test 2: JSON request
    print("\n" + "="*60)
    print("TEST 2: JSON request")
    print("="*60)
    
    json_prompt = '''Return a JSON object with two fields:
- "number": 42
- "text": "hello"

JSON:'''
    response2 = client.generate(json_prompt, model='llama3.2:latest')
    print(f"Prompt: {json_prompt}")
    print(f"Response: {response2}")
    
    # Test 3: Risk assessment
    print("\n" + "="*60)
    print("TEST 3: Risk assessment")
    print("="*60)
    
    risk_prompt = '''Is "turn on lights" safe? Answer with JSON:
{"risk_level": "LOW", "rationale": "Safe operation"}

JSON:'''
    response3 = client.generate(risk_prompt, model='llama3.2:latest')
    print(f"Prompt: {risk_prompt}")
    print(f"Response: {response3}")
    
    # Test 4: Try with different model
    print("\n" + "="*60)
    print("TEST 4: Different model (mistral)")
    print("="*60)
    
    try:
        response4 = client.generate(risk_prompt, model='mistral:latest')
        print(f"Mistral response: {response4}")
    except Exception as e:
        print(f"Mistral error: {e}")
    
    # Test 5: Check if it's a context issue
    print("\n" + "="*60)
    print("TEST 5: Check model behavior")
    print("="*60)
    
    behavior_prompt = "You are a helpful assistant. Say 'Hello' in JSON format: {\"message\": \"Hello\"}"
    response5 = client.generate(behavior_prompt, model='llama3.2:latest')
    print(f"Behavior test: {response5}")

if __name__ == "__main__":
    test_ollama_direct()