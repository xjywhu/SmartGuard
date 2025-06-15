#!/usr/bin/env python3
"""
Enhanced Context Analysis and Visualization

This script analyzes the performance of the smart home risk assessment system
with enhanced context built using knowledge graphs and GNN-inspired features.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

class EnhancedContextAnalyzer:
    """Analyze and visualize enhanced context test results."""
    
    def __init__(self):
        self.results = None
        self.baseline_results = None
        self.load_results()
    
    def load_results(self):
        """Load test results from files."""
        # Load enhanced context results
        try:
            with open("enhanced_context_test_results.json", 'r') as f:
                self.results = json.load(f)
            print("✅ Loaded enhanced context test results")
        except FileNotFoundError:
            print("❌ Enhanced context test results not found")
            return
        
        # Load baseline results for comparison
        try:
            with open("test_results.json", 'r') as f:
                self.baseline_results = json.load(f)
            print("✅ Loaded baseline test results for comparison")
        except FileNotFoundError:
            print("⚠️ Baseline test results not found, will analyze enhanced context only")
    
    def analyze_knowledge_graph_impact(self):
        """Analyze the impact of knowledge graph features."""
        print("\n🕸️ KNOWLEDGE GRAPH IMPACT ANALYSIS")
        print("=" * 60)
        
        if not self.results:
            print("No results to analyze")
            return
        
        kg_stats = self.results["context_statistics"]["knowledge_graph"]
        
        print(f"Knowledge Graph Statistics:")
        print(f"  📊 Total Nodes: {kg_stats['total_nodes']}")
        print(f"  🔗 Total Edges: {kg_stats['total_edges']}")
        print(f"  🏠 Rooms: {kg_stats['room_count']}")
        print(f"  📱 Devices: {kg_stats['device_count']}")
        print(f"  🌐 Network Density: {kg_stats['density']:.3f}")
        print(f"  🔄 Connected Components: {kg_stats['connected_components']}")
        
        # Analyze context examples
        print("\nContext Building Examples:")
        for example in self.results["context_statistics"]["examples"]:
            print(f"\n  Command: '{example['command']}'")
            print(f"    🎯 Target Devices: {len(example['target_devices'])} identified")
            print(f"    ⚠️ Risk Indicators: {len(example['risk_indicators'])} found")
            print(f"    🏠 Spatial Context: {len(example['spatial_context'])} rooms involved")
            print(f"    ⏰ Temporal Risk: {'Yes' if example['temporal_risk'] else 'No'}")
        
        # Calculate context effectiveness
        total_devices_identified = sum(len(ex['target_devices']) for ex in self.results["context_statistics"]["examples"])
        total_risk_indicators = sum(len(ex['risk_indicators']) for ex in self.results["context_statistics"]["examples"])
        
        print(f"\nContext Effectiveness:")
        print(f"  🎯 Average Devices per Command: {total_devices_identified / len(self.results['context_statistics']['examples']):.1f}")
        print(f"  ⚠️ Average Risk Indicators per Command: {total_risk_indicators / len(self.results['context_statistics']['examples']):.1f}")
    
    def analyze_model_performance_with_context(self):
        """Analyze how models performed with enhanced context."""
        print("\n🤖 MODEL PERFORMANCE WITH ENHANCED CONTEXT")
        print("=" * 60)
        
        if not self.results:
            return
        
        # Analyze each model
        for model_name, model_results in self.results["model_results"].items():
            print(f"\n{model_name}:")
            print(f"  📊 Accuracy: {model_results['accuracy']:.1%}")
            print(f"  ⏱️ Avg Response Time: {model_results['avg_response_time']:.2f}s")
            print(f"  🧠 Context Usage Rate: {model_results['context_usage_rate']:.1%}")
            print(f"  ✅ Correct Predictions: {model_results['correct_predictions']}/{model_results['total_tests']}")
            
            # Analyze why context wasn't used
            if model_results['context_usage_rate'] == 0:
                print(f"  ⚠️ Context not utilized - likely falling back to rule-based methods")
            
            # Show detailed results for failed cases
            failed_cases = [r for r in model_results['detailed_results'] if not r.get('correct', False)]
            if failed_cases:
                print(f"  ❌ Failed Cases:")
                for case in failed_cases:
                    print(f"    - '{case['command']}': Expected {case.get('expected_risk', 'N/A')}, Got {case.get('predicted_risk', 'N/A')}")
        
        # Analyze fallback performance
        if "fallback_results" in self.results:
            fallback = self.results["fallback_results"]
            print(f"\nFallback Method:")
            print(f"  📊 Accuracy: {fallback['accuracy']:.1%}")
            print(f"  ⏱️ Avg Response Time: {fallback['avg_response_time']:.4f}s")
            print(f"  ✅ Correct Predictions: {fallback['correct_predictions']}/{fallback['total_tests']}")
    
    def compare_with_baseline(self):
        """Compare enhanced context results with baseline."""
        print("\n📈 ENHANCED CONTEXT vs BASELINE COMPARISON")
        print("=" * 60)
        
        if not self.baseline_results:
            print("No baseline results available for comparison")
            return
        
        # Extract baseline accuracy
        baseline_accuracy = self.baseline_results.get("summary", {}).get("accuracy", 0)
        
        # Compare with enhanced context results
        print(f"Baseline System (without enhanced context):")
        print(f"  📊 Accuracy: {baseline_accuracy:.1%}")
        
        print(f"\nEnhanced Context System:")
        
        # Get best performing method from enhanced context
        best_method = self.results["performance_comparison"]["method_rankings"][0]
        print(f"  🏆 Best Method: {best_method['method']}")
        print(f"  📊 Accuracy: {best_method['accuracy']:.1%}")
        print(f"  ⏱️ Response Time: {best_method['avg_response_time']:.4f}s")
        
        # Calculate improvement
        accuracy_improvement = best_method['accuracy'] - baseline_accuracy
        print(f"\nImprovement Analysis:")
        if accuracy_improvement > 0:
            print(f"  ✅ Accuracy improved by {accuracy_improvement:.1%}")
        elif accuracy_improvement < 0:
            print(f"  ⚠️ Accuracy decreased by {abs(accuracy_improvement):.1%}")
        else:
            print(f"  ➡️ Accuracy remained the same")
        
        # Analyze why enhanced context might not be helping
        if accuracy_improvement <= 0:
            print(f"\nPossible reasons for limited improvement:")
            print(f"  1. 🔄 LLM models falling back to rule-based methods")
            print(f"  2. 📝 Context format may need optimization for LLM consumption")
            print(f"  3. 🎯 Test dataset may be too simple for context to make a difference")
            print(f"  4. ⚙️ Model prompts may need adjustment to utilize context effectively")
    
    def analyze_context_utilization(self):
        """Analyze how well the enhanced context was utilized."""
        print("\n🧠 CONTEXT UTILIZATION ANALYSIS")
        print("=" * 60)
        
        if not self.results:
            return
        
        # Check context usage across all models
        total_context_usage = 0
        total_tests = 0
        
        for model_name, model_results in self.results["model_results"].items():
            usage_rate = model_results['context_usage_rate']
            tests = len(model_results['detailed_results'])
            
            total_context_usage += usage_rate * tests
            total_tests += tests
            
            print(f"{model_name}: {usage_rate:.1%} context usage")
        
        overall_usage = total_context_usage / total_tests if total_tests > 0 else 0
        print(f"\nOverall Context Usage: {overall_usage:.1%}")
        
        if overall_usage < 0.3:
            print("\n⚠️ LOW CONTEXT UTILIZATION DETECTED")
            print("Recommendations to improve context usage:")
            print("  1. 🔧 Modify LLM prompts to explicitly request context usage")
            print("  2. 📝 Restructure context format for better LLM parsing")
            print("  3. 🎯 Add context relevance scoring")
            print("  4. 🔄 Implement context injection at multiple pipeline stages")
        elif overall_usage > 0.7:
            print("\n✅ GOOD CONTEXT UTILIZATION")
            print("The enhanced context is being effectively used by the models")
        else:
            print("\n🔄 MODERATE CONTEXT UTILIZATION")
            print("There's room for improvement in context usage")
    
    def generate_improvement_recommendations(self):
        """Generate specific recommendations for improving the system."""
        print("\n💡 IMPROVEMENT RECOMMENDATIONS")
        print("=" * 60)
        
        recommendations = []
        
        # Analyze current performance
        if self.results:
            best_accuracy = self.results["performance_comparison"]["best_accuracy"]["score"]
            context_usage = sum(model["context_usage_rate"] for model in self.results["model_results"].values()) / len(self.results["model_results"]) if self.results["model_results"] else 0
            
            # Accuracy-based recommendations
            if best_accuracy < 0.7:
                recommendations.append("🎯 Improve accuracy by refining risk assessment rules")
                recommendations.append("📝 Enhance training data with more diverse scenarios")
            
            # Context usage recommendations
            if context_usage < 0.3:
                recommendations.append("🧠 Improve LLM prompt engineering to better utilize context")
                recommendations.append("🔧 Implement context-aware fine-tuning for models")
            
            # Performance recommendations
            fastest_time = self.results["performance_comparison"]["fastest_method"]["time"]
            if fastest_time > 1.0:
                recommendations.append("⚡ Optimize model inference for faster response times")
                recommendations.append("🔄 Consider model quantization or smaller models for speed")
        
        # Knowledge graph recommendations
        recommendations.extend([
            "🕸️ Expand knowledge graph with more device relationships",
            "📊 Add temporal patterns and user behavior modeling",
            "🔗 Implement graph neural network layers for better embeddings",
            "🎯 Add device dependency modeling for cascading risk assessment"
        ])
        
        # System architecture recommendations
        recommendations.extend([
            "🏗️ Implement hybrid LLM + rule-based approach with smart fallback",
            "📈 Add real-time learning from user feedback",
            "🛡️ Implement confidence scoring for risk predictions",
            "🔄 Add context caching for improved performance"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    def create_performance_visualization(self):
        """Create visualizations of performance metrics."""
        print("\n📊 CREATING PERFORMANCE VISUALIZATIONS")
        print("=" * 60)
        
        if not self.results:
            print("No results to visualize")
            return
        
        try:
            # Prepare data
            methods = []
            accuracies = []
            response_times = []
            
            # Add model results
            for model_name, model_results in self.results["model_results"].items():
                methods.append(model_name.replace(":latest", ""))
                accuracies.append(model_results["accuracy"])
                response_times.append(model_results["avg_response_time"])
            
            # Add fallback results
            if "fallback_results" in self.results:
                methods.append("Fallback")
                accuracies.append(self.results["fallback_results"]["accuracy"])
                response_times.append(self.results["fallback_results"]["avg_response_time"])
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Accuracy comparison
            bars1 = ax1.bar(methods, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            ax1.set_title('Model Accuracy Comparison\nwith Enhanced Context', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars1, accuracies):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
            
            # Response time comparison (log scale for better visualization)
            bars2 = ax2.bar(methods, response_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            ax2.set_title('Response Time Comparison\nwith Enhanced Context', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Response Time (seconds)')
            ax2.set_yscale('log')
            
            # Add value labels on bars
            for bar, time_val in zip(bars2, response_times):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                        f'{time_val:.3f}s', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('enhanced_context_performance.png', dpi=300, bbox_inches='tight')
            print("📊 Performance visualization saved as 'enhanced_context_performance.png'")
            
        except ImportError:
            print("⚠️ Matplotlib not available, skipping visualization")
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report."""
        print("\n📋 COMPREHENSIVE ENHANCED CONTEXT ANALYSIS REPORT")
        print("=" * 80)
        
        if not self.results:
            print("No results available for analysis")
            return
        
        # Header information
        timestamp = self.results["test_summary"]["timestamp"]
        print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test Timestamp: {timestamp}")
        print(f"Enhanced Context: {'✅ Enabled' if self.results['test_summary']['enhanced_context'] else '❌ Disabled'}")
        print(f"Knowledge Graph: {'✅ Enabled' if self.results['test_summary']['knowledge_graph_enabled'] else '❌ Disabled'}")
        print(f"GNN Features: {'✅ Enabled' if self.results['test_summary']['gnn_features_enabled'] else '❌ Disabled'}")
        
        # Run all analyses
        self.analyze_knowledge_graph_impact()
        self.analyze_model_performance_with_context()
        self.compare_with_baseline()
        self.analyze_context_utilization()
        self.generate_improvement_recommendations()
        self.create_performance_visualization()
        
        print("\n✅ COMPREHENSIVE ANALYSIS COMPLETED")
        print("=" * 80)

if __name__ == "__main__":
    analyzer = EnhancedContextAnalyzer()
    analyzer.generate_comprehensive_report()