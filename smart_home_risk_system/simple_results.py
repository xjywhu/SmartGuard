import json
import os

def show_rag_results():
    """Display RAG test results clearly"""
    
    if not os.path.exists('rag_test_results.json'):
        print("❌ Results file not found!")
        return
    
    try:
        with open('rag_test_results.json', 'r') as f:
            data = json.load(f)
        
        print("🔍 RAG PIPELINE TEST RESULTS")
        print("=" * 50)
        print(f"📊 Total Commands: {data['test_summary']['total_commands']}")
        print(f"⏰ Test Time: {data['test_summary']['timestamp']}")
        print(f"🤖 RAG Status: {'✅ ENABLED' if data['test_summary']['rag_enabled'] else '❌ DISABLED'}")
        print()
        
        # Process each model's results
        for model_name, results in data['model_results'].items():
            print(f"🧠 MODEL: {model_name}")
            print("-" * 40)
            
            success_count = 0
            total_count = len(results)
            
            for i, test in enumerate(results, 1):
                print(f"\n{i}. Command: '{test['command']}'")
                
                # Check if we have both assessments
                if 'rag_assessment' in test and 'traditional_assessment' in test:
                    rag_risk = test['rag_assessment']['risk_level']
                    trad_risk = test['traditional_assessment']['risk_level']
                    confidence = test['rag_assessment'].get('confidence', 'N/A')
                    
                    # RAG metadata
                    if 'rag_metadata' in test['rag_assessment']:
                        meta = test['rag_assessment']['rag_metadata']
                        docs = meta.get('retrieved_docs_count', 0)
                        relevance = meta.get('avg_relevance_score', 0)
                        response_time = meta.get('response_time', 0)
                        
                        print(f"   🎯 RAG Risk: {rag_risk} (Confidence: {confidence})")
                        print(f"   🔄 Traditional Risk: {trad_risk}")
                        print(f"   📚 Retrieved Docs: {docs}")
                        print(f"   🎪 Relevance Score: {relevance:.3f}")
                        print(f"   ⏱️  Response Time: {response_time:.2f}s")
                        
                        # Check agreement
                        agreement = rag_risk == trad_risk
                        print(f"   {'✅' if agreement else '❌'} Risk Agreement: {agreement}")
                        
                        if agreement:
                            success_count += 1
                    else:
                        print(f"   🎯 RAG Risk: {rag_risk}")
                        print(f"   🔄 Traditional Risk: {trad_risk}")
                        print(f"   ⚠️  Missing metadata")
                else:
                    print("   ❌ Incomplete test data")
            
            # Model summary
            print(f"\n📈 SUMMARY for {model_name}:")
            print(f"   ✅ Agreement Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
            
            # Calculate averages
            valid_tests = [t for t in results if 'rag_assessment' in t and 'rag_metadata' in t['rag_assessment']]
            if valid_tests:
                avg_confidence = sum(t['rag_assessment'].get('confidence', 0) for t in valid_tests) / len(valid_tests)
                avg_docs = sum(t['rag_assessment']['rag_metadata'].get('retrieved_docs_count', 0) for t in valid_tests) / len(valid_tests)
                avg_relevance = sum(t['rag_assessment']['rag_metadata'].get('avg_relevance_score', 0) for t in valid_tests) / len(valid_tests)
                avg_time = sum(t['rag_assessment']['rag_metadata'].get('response_time', 0) for t in valid_tests) / len(valid_tests)
                
                print(f"   📈 Avg Confidence: {avg_confidence:.2f}")
                print(f"   📚 Avg Retrieved Docs: {avg_docs:.1f}")
                print(f"   🎪 Avg Relevance: {avg_relevance:.3f}")
                print(f"   ⏱️  Avg Response Time: {avg_time:.2f}s")
            
            print("\n" + "=" * 50)
        
        print("\n🎉 RAG SYSTEM BENEFITS:")
        print("   • Enhanced context awareness")
        print("   • Detailed risk rationales")
        print("   • Knowledge graph integration")
        print("   • Semantic document retrieval")
        print("   • Improved confidence scoring")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    show_rag_results()