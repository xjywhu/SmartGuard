import json
import os

def display_rag_results():
    """Display RAG test results in a readable format"""
    
    # Check if results file exists
    if not os.path.exists('rag_test_results.json'):
        print("❌ rag_test_results.json not found!")
        return
    
    try:
        with open('rag_test_results.json', 'r') as f:
            data = json.load(f)
        
        print('🔍 === RAG PIPELINE TEST RESULTS ===')
        print(f"📊 Total Commands Tested: {data['test_summary']['total_commands']}")
        print(f"⏰ Timestamp: {data['test_summary']['timestamp']}")
        print(f"🤖 RAG Enabled: {data['test_summary']['rag_enabled']}")
        print()
        
        for model, results in data['model_results'].items():
            print(f'🧠 Model: {model}')
            print('=' * 60)
            
            for i, result in enumerate(results[:5], 1):  # Show first 5 results
                print(f"\n{i}. 💬 Command: '{result['command']}'")
                
                # RAG Assessment
                rag = result['rag_assessment']
                print(f"   🎯 RAG Risk Level: {rag['risk_level']}")
                print(f"   📈 RAG Confidence: {rag['confidence']}")
                print(f"   📚 Retrieved Docs: {rag['rag_metadata']['retrieved_docs_count']}")
                print(f"   🎪 Avg Relevance: {rag['rag_metadata']['avg_relevance_score']:.3f}")
                print(f"   ⏱️  Response Time: {rag['rag_metadata']['response_time']:.2f}s")
                
                # Traditional Assessment
                trad = result['traditional_assessment']
                print(f"   🔄 Traditional Risk: {trad['risk_level']}")
                
                # Improvement
                imp = result['improvement_metrics']
                agreement = "✅" if imp['risk_level_agreement'] else "❌"
                print(f"   {agreement} Risk Agreement: {imp['risk_level_agreement']}")
                print(f"   📊 Confidence Gain: +{imp['confidence_improvement']}")
                
                # Show brief rationale if available
                if 'rationale' in rag and rag['rationale']:
                    rationale = rag['rationale'][:200] + "..." if len(rag['rationale']) > 200 else rag['rationale']
                    print(f"   💡 Brief Rationale: {rationale}")
            
            print('\n' + '='*60)
            
        # Summary statistics
        print("\n📈 SUMMARY STATISTICS:")
        for model, results in data['model_results'].items():
            total_tests = len(results)
            agreements = sum(1 for r in results if r['improvement_metrics']['risk_level_agreement'])
            avg_confidence = sum(r['rag_assessment']['confidence'] for r in results) / total_tests
            avg_docs = sum(r['rag_assessment']['rag_metadata']['retrieved_docs_count'] for r in results) / total_tests
            avg_relevance = sum(r['rag_assessment']['rag_metadata']['avg_relevance_score'] for r in results) / total_tests
            
            print(f"\n🧠 {model}:")
            print(f"   ✅ Agreement Rate: {agreements}/{total_tests} ({agreements/total_tests*100:.1f}%)")
            print(f"   📈 Avg Confidence: {avg_confidence:.2f}")
            print(f"   📚 Avg Retrieved Docs: {avg_docs:.1f}")
            print(f"   🎪 Avg Relevance Score: {avg_relevance:.3f}")
            
    except Exception as e:
        print(f"❌ Error reading results: {e}")

if __name__ == "__main__":
    display_rag_results()