#!/usr/bin/env python3
"""
Test with a fixed RAG implementation that properly handles JSON responses
"""

import json
import time
from typing import Dict, Any, List
from rag_pipeline import RAGPipeline

class FixedRAGPipeline(RAGPipeline):
    """Fixed RAG pipeline that properly handles JSON responses"""
    
    def assess_risk_with_rag(self, command: str, model: str = 'llama3.2:latest') -> Dict[str, Any]:
        """Perform risk assessment using RAG pipeline with proper JSON handling"""
        start_time = time.time()
        
        # Get enhanced context
        context = self.context_builder.build_enhanced_context(command)
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_context(command)
        
        # Generate RAG-enhanced prompt
        rag_prompt = self.generate_rag_prompt(command, context, retrieved_docs)
        
        # Get LLM response
        try:
            response = self.ollama_client.generate(prompt=rag_prompt, model=model)
            print(f"Raw LLM response: {response[:200]}...")  # Debug output
            
            # Try to parse as JSON directly
            try:
                # Clean the response
                response_clean = response.strip()
                
                # Remove any markdown code blocks
                if response_clean.startswith('```'):
                    lines = response_clean.split('\n')
                    # Find the actual JSON content
                    json_lines = []
                    in_json = False
                    for line in lines:
                        if line.strip().startswith('{') or in_json:
                            in_json = True
                            json_lines.append(line)
                            if line.strip().endswith('}'):
                                break
                    response_clean = '\n'.join(json_lines)
                
                # Try to find JSON in the response
                if '{' in response_clean and '}' in response_clean:
                    start_idx = response_clean.find('{')
                    end_idx = response_clean.rfind('}') + 1
                    json_str = response_clean[start_idx:end_idx]
                    
                    risk_assessment = json.loads(json_str)
                    
                    # Validate the response
                    if 'risk_level' not in risk_assessment or 'rationale' not in risk_assessment:
                        raise ValueError("Missing required fields")
                    
                    # Ensure risk_level is valid
                    if risk_assessment['risk_level'] not in ['LOW', 'MEDIUM', 'HIGH']:
                        raise ValueError(f"Invalid risk level: {risk_assessment['risk_level']}")
                    
                    # Add confidence if missing
                    if 'confidence' not in risk_assessment:
                        risk_assessment['confidence'] = 0.8
                    
                    print(f"✅ Successfully parsed JSON: {risk_assessment['risk_level']}")
                    
                else:
                    raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                print(f"❌ JSON parsing failed: {e}")
                print(f"Response was: {response}")
                
                # Fallback: extract risk level from text
                response_lower = response.lower()
                if 'high' in response_lower:
                    risk_level = 'HIGH'
                elif 'low' in response_lower:
                    risk_level = 'LOW'
                else:
                    risk_level = 'MEDIUM'
                
                risk_assessment = {
                    'risk_level': risk_level,
                    'rationale': response[:200] + '...' if len(response) > 200 else response,
                    'confidence': 0.6,
                    'method': 'text_fallback'
                }
                print(f"🔄 Using fallback parsing: {risk_level}")
            
            # Add RAG-specific metadata
            risk_assessment['rag_metadata'] = {
                'retrieved_docs_count': len(retrieved_docs),
                'avg_relevance_score': sum(doc['relevance_score'] for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0,
                'context_sources': [doc['metadata']['type'] for doc in retrieved_docs],
                'response_time': time.time() - start_time
            }
            
            return risk_assessment
            
        except Exception as e:
            print(f"RAG assessment failed: {e}")
            # Fallback to simple assessment
            return {
                'risk_level': 'MEDIUM',
                'rationale': f'Assessment failed: {str(e)}',
                'confidence': 0.3,
                'rag_metadata': {
                    'retrieved_docs_count': len(retrieved_docs),
                    'fallback_used': True,
                    'error': str(e),
                    'response_time': time.time() - start_time
                }
            }

def test_fixed_rag():
    """Test the fixed RAG pipeline"""
    
    print("Testing Fixed RAG Pipeline")
    print("=" * 50)
    
    try:
        # Initialize fixed RAG pipeline
        rag = FixedRAGPipeline()
        print("✅ Fixed RAG Pipeline initialized")
        
        # Test commands
        test_cases = [
            ("turn on lights in bedroom", "LOW"),
            ("turn off smoke detector", "HIGH"),
            ("set thermostat to 20°C", "LOW")
        ]
        
        results = []
        
        for command, expected in test_cases:
            print(f"\n{'='*60}")
            print(f"Testing: '{command}'")
            print(f"Expected: {expected}")
            
            result = rag.assess_risk_with_rag(command, model='llama3.2:latest')
            
            predicted = result.get('risk_level', 'UNKNOWN')
            correct = predicted == expected
            
            print(f"Predicted: {predicted}")
            print(f"Correct: {'✅' if correct else '❌'}")
            print(f"Rationale: {result.get('rationale', 'N/A')[:100]}...")
            
            results.append({
                'command': command,
                'expected': expected,
                'predicted': predicted,
                'correct': correct,
                'result': result
            })
        
        # Summary
        correct_count = sum(1 for r in results if r['correct'])
        total_count = len(results)
        accuracy = correct_count / total_count
        
        print(f"\n{'='*60}")
        print("FIXED RAG PIPELINE TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Accuracy: {correct_count}/{total_count} ({accuracy:.1%})")
        
        if accuracy >= 0.8:
            print("🎉 Excellent! RAG pipeline is working well.")
        elif accuracy >= 0.6:
            print("✅ Good! RAG pipeline shows improvement.")
        else:
            print("⚠️ RAG pipeline needs more work.")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_fixed_rag()