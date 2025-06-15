#!/usr/bin/env python3
"""
Final RAG test with working approach
"""

import json
import time
from modules.ollama_client import OllamaClient
from enhanced_context_builder import EnhancedContextBuilder
from rag_pipeline import VectorDatabase
from modules.command_parser import CommandParser

class WorkingRAGPipeline:
    """RAG Pipeline that actually works"""
    
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.context_builder = EnhancedContextBuilder()
        self.vector_db = VectorDatabase()
        self.command_parser = CommandParser()
        
        # Load existing vector database
        if not self.vector_db.load('vector_db.pkl'):
            print("Warning: Could not load vector database")
        
    def assess_risk_with_rag(self, command, context=None):
        """
        Assess risk using the working approach
        """
        try:
            # Get relevant context from vector database
            search_results = self.vector_db.search(command, top_k=3)
            
            # Build context from search results
            context_info = "Normal smart home operation."
            if search_results:
                # Use the most relevant result
                top_result = search_results[0]
                if hasattr(top_result, 'content'):
                    context_info = top_result.content[:150] + "..."
                elif isinstance(top_result, dict):
                    context_info = top_result.get('content', '')[:150] + "..."
            
            # Create the exact prompt that worked in clean test
            prompt = f"""Assess smart home command safety:

Command: {command}
Context: {context_info}

Safety levels:
- HIGH: Disables safety devices, immediate danger
- MEDIUM: Energy waste, minor issues
- LOW: Normal safe operation

Return JSON: {{"risk_level": "HIGH|MEDIUM|LOW", "rationale": "reason"}}

JSON:"""
            
            # Use the model that worked (llama3.2:latest)
            response = self.ollama_client.generate(prompt, model='llama3.2:latest')
            
            # Parse JSON response using the working method
            return self._parse_json_response(response)
            
        except Exception as e:
            print(f"RAG assessment failed: {e}")
            return {
                'risk_level': 'MEDIUM',
                'rationale': f'Error: {e}',
                'confidence': 0.3
            }
    
    def _parse_json_response(self, response):
        """
        Parse JSON response using the method that worked
        """
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
                    return {
                        'risk_level': assessment['risk_level'],
                        'rationale': assessment['rationale'],
                        'confidence': 0.9
                    }
            
            # Fallback to text parsing
            response_upper = response_clean.upper()
            if 'HIGH' in response_upper:
                risk_level = 'HIGH'
            elif 'MEDIUM' in response_upper:
                risk_level = 'MEDIUM'
            elif 'LOW' in response_upper:
                risk_level = 'LOW'
            else:
                risk_level = 'MEDIUM'
            
            return {
                'risk_level': risk_level,
                'rationale': 'Extracted from text',
                'confidence': 0.6
            }
            
        except Exception as e:
            print(f"JSON parsing failed: {e}")
            return {
                'risk_level': 'MEDIUM',
                'rationale': f'Parse error: {e}',
                'confidence': 0.3
            }

def test_working_rag():
    """Test the working RAG pipeline"""
    
    print("Working RAG Pipeline Test")
    print("=" * 50)
    
    # Initialize pipeline
    rag = WorkingRAGPipeline()
    
    if not rag.ollama_client.is_available():
        print("❌ Ollama not available")
        return
    
    print("✅ Working RAG Pipeline initialized")
    
    # Test cases
    test_cases = [
        {
            'command': 'turn on lights in bedroom',
            'expected': 'LOW',
            'description': 'normal lighting operation'
        },
        {
            'command': 'turn off smoke detector',
            'expected': 'HIGH',
            'description': 'disabling safety device'
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test_case['command']}")
        print(f"Expected: {test_case['expected']} ({test_case['description']})")
        
        # Assess risk
        result = rag.assess_risk_with_rag(test_case['command'])
        
        predicted = result['risk_level']
        rationale = result['rationale']
        confidence = result['confidence']
        correct = predicted == test_case['expected']
        
        print(f"\nRAG Assessment Result:")
        print(f"Risk Level: {predicted}")
        print(f"Confidence: {confidence:.1f}")
        print(f"Rationale: {rationale[:100]}...")
        
        status = "✅" if correct else "❌"
        print(f"\n{status} {'CORRECT' if correct else 'INCORRECT'}: Predicted {predicted} risk for {test_case['description']}")
        
        results.append({
            'command': test_case['command'],
            'expected': test_case['expected'],
            'predicted': predicted,
            'correct': correct,
            'confidence': confidence
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("WORKING RAG TEST SUMMARY")
    print(f"{'='*60}")
    
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"Accuracy: {correct_count}/{total_count} ({accuracy:.1%})")
    
    for result in results:
        status = "✅" if result['correct'] else "❌"
        print(f"{status} {result['command']}: {result['predicted']} (expected {result['expected']})")
    
    if accuracy == 1.0:
        print("\n🎉 Perfect! Working RAG pipeline is successful!")
    elif accuracy >= 0.8:
        print("\n✅ Excellent! Working RAG pipeline is very good.")
    elif accuracy >= 0.6:
        print("\n👍 Good! Working RAG pipeline shows promise.")
    else:
        print("\n⚠️ Working RAG pipeline needs more refinement.")
    
    return results

if __name__ == "__main__":
    test_working_rag()