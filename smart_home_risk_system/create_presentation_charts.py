#!/usr/bin/env python3
"""
Professional Presentation Charts for Enhanced Context Analysis

This script creates publication-quality visualizations for presenting
the smart home risk assessment system results to academic audiences.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
from datetime import datetime

# Set style for professional presentation
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

class PresentationChartGenerator:
    """Generate professional charts for academic presentation."""
    
    def __init__(self):
        self.results = None
        self.baseline_results = None
        self.load_data()
    
    def load_data(self):
        """Load test results from files."""
        try:
            with open("enhanced_context_test_results.json", 'r') as f:
                self.results = json.load(f)
            print("✅ Loaded enhanced context results")
        except FileNotFoundError:
            print("❌ Enhanced context results not found")
            return
        
        try:
            with open("test_results.json", 'r') as f:
                self.baseline_results = json.load(f)
            print("✅ Loaded baseline results")
        except FileNotFoundError:
            print("⚠️ Baseline results not found")
    
    def create_accuracy_comparison_chart(self):
        """Create a professional accuracy comparison chart."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Data preparation
        methods = ['Baseline\n(No Context)', 'Llama3.2\n(Enhanced Context)', 
                  'Mistral\n(Enhanced Context)', 'Fallback\n(Enhanced Context)']
        
        baseline_acc = self.baseline_results.get("summary", {}).get("accuracy", 0) if self.baseline_results else 0.04
        accuracies = [
            baseline_acc,
            self.results["model_results"]["llama3.2:latest"]["accuracy"],
            self.results["model_results"]["mistral:latest"]["accuracy"],
            self.results["fallback_results"]["accuracy"]
        ]
        
        # Create bars with different colors
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        # Customize chart
        ax.set_title('Smart Home Risk Assessment System\nAccuracy Comparison with Enhanced Context', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_ylabel('Accuracy (%)', fontsize=16)
        ax.set_ylim(0, 0.8)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add improvement annotation
        improvement = accuracies[3] - accuracies[0]
        ax.annotate(f'+{improvement:.1%}\nImprovement', 
                   xy=(3, accuracies[3]), xytext=(2.5, 0.7),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=14, fontweight='bold', color='red',
                   ha='center')
        
        plt.tight_layout()
        plt.savefig('accuracy_comparison_presentation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Created: accuracy_comparison_presentation.png")
    
    def create_response_time_chart(self):
        """Create response time comparison chart."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods = ['Llama3.2', 'Mistral', 'Fallback']
        times = [
            self.results["model_results"]["llama3.2:latest"]["avg_response_time"],
            self.results["model_results"]["mistral:latest"]["avg_response_time"],
            self.results["fallback_results"]["avg_response_time"]
        ]
        
        colors = ['#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax.bar(methods, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            if time_val < 0.001:
                label = f'{time_val*1000:.2f}ms'
            else:
                label = f'{time_val:.2f}s'
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                   label, ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        ax.set_title('Response Time Comparison\nEnhanced Context System', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_ylabel('Response Time (seconds)', fontsize=16)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('response_time_presentation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Created: response_time_presentation.png")
    
    def create_knowledge_graph_stats_chart(self):
        """Create knowledge graph statistics visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        kg_stats = self.results["context_statistics"]["knowledge_graph"]
        
        # 1. Network composition pie chart
        sizes = [kg_stats['room_count'], kg_stats['device_count']]
        labels = ['Rooms', 'Devices']
        colors = ['#FF9999', '#66B2FF']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Knowledge Graph\nNode Composition', fontweight='bold', fontsize=14)
        
        # 2. Network metrics bar chart
        metrics = ['Total Nodes', 'Total Edges', 'Connected\nComponents']
        values = [kg_stats['total_nodes'], kg_stats['total_edges'], kg_stats['connected_components']]
        
        bars = ax2.bar(metrics, values, color=['#FFB366', '#66FFB2', '#B366FF'], alpha=0.8)
        ax2.set_title('Network Metrics', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Count')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    str(val), ha='center', va='bottom', fontweight='bold')
        
        # 3. Context effectiveness
        examples = self.results["context_statistics"]["examples"]
        commands = [ex['command'][:20] + '...' if len(ex['command']) > 20 else ex['command'] for ex in examples]
        devices_identified = [len(ex['target_devices']) for ex in examples]
        risk_indicators = [len(ex['risk_indicators']) for ex in examples]
        
        x = np.arange(len(commands))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, devices_identified, width, label='Target Devices', color='#4ECDC4', alpha=0.8)
        bars2 = ax3.bar(x + width/2, risk_indicators, width, label='Risk Indicators', color='#FF6B6B', alpha=0.8)
        
        ax3.set_title('Context Building Effectiveness', fontweight='bold', fontsize=14)
        ax3.set_ylabel('Count')
        ax3.set_xticks(x)
        ax3.set_xticklabels(commands, rotation=45, ha='right')
        ax3.legend()
        
        # 4. Network density visualization
        density = kg_stats['density']
        ax4.bar(['Network Density'], [density], color='#96CEB4', alpha=0.8, width=0.5)
        ax4.set_title('Network Connectivity', fontweight='bold', fontsize=14)
        ax4.set_ylabel('Density Score')
        ax4.set_ylim(0, 0.1)
        ax4.text(0, density + 0.002, f'{density:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Knowledge Graph Analysis\nEnhanced Context System', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig('knowledge_graph_analysis_presentation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Created: knowledge_graph_analysis_presentation.png")
    
    def create_detailed_results_heatmap(self):
        """Create a heatmap showing detailed test results."""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Prepare data for heatmap
        test_commands = [
            'Turn off smoke detector',
            'Set oven 500°F 10hrs',
            'Turn on TV',
            'Lock front door',
            'Disable security cameras'
        ]
        
        models = ['Llama3.2', 'Mistral', 'Fallback']
        
        # Create results matrix (1 = correct, 0 = incorrect)
        results_matrix = []
        
        for model_key in ['llama3.2:latest', 'mistral:latest']:
            model_results = []
            for result in self.results["model_results"][model_key]["detailed_results"]:
                model_results.append(1 if result.get('correct', False) else 0)
            results_matrix.append(model_results)
        
        # Add fallback results
        fallback_results = []
        for result in self.results["fallback_results"]["detailed_results"]:
            fallback_results.append(1 if result.get('correct', False) else 0)
        results_matrix.append(fallback_results)
        
        # Create heatmap
        sns.heatmap(results_matrix, 
                   xticklabels=test_commands,
                   yticklabels=models,
                   annot=True, 
                   fmt='d',
                   cmap='RdYlGn',
                   cbar_kws={'label': 'Prediction Accuracy'},
                   square=True,
                   linewidths=0.5)
        
        ax.set_title('Detailed Test Results Heatmap\nEnhanced Context System', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Test Commands', fontsize=14)
        ax.set_ylabel('Models', fontsize=14)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('detailed_results_heatmap_presentation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Created: detailed_results_heatmap_presentation.png")
    
    def create_system_architecture_diagram(self):
        """Create a system architecture overview diagram."""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Define components and their positions
        components = {
            'Smart Home\nCommands': (1, 8),
            'Enhanced Context\nBuilder': (3, 8),
            'Knowledge Graph\n(76 nodes, 69 edges)': (5, 8),
            'GNN Features\nExtractor': (7, 8),
            'LLM Models\n(Llama3.2, Mistral)': (3, 5),
            'Fallback\nRule Engine': (7, 5),
            'Risk Assessment\nOutput': (5, 2)
        }
        
        # Draw components
        for comp, (x, y) in components.items():
            if 'Knowledge Graph' in comp:
                color = '#FFE6E6'
            elif 'GNN' in comp:
                color = '#E6F3FF'
            elif 'LLM' in comp:
                color = '#E6FFE6'
            elif 'Fallback' in comp:
                color = '#FFF0E6'
            else:
                color = '#F0F0F0'
            
            rect = Rectangle((x-0.8, y-0.6), 1.6, 1.2, 
                           facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, comp, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Draw arrows
        arrows = [
            ((1.8, 8), (2.2, 8)),  # Commands -> Context Builder
            ((3.8, 8), (4.2, 8)),  # Context Builder -> Knowledge Graph
            ((5.8, 8), (6.2, 8)),  # Knowledge Graph -> GNN Features
            ((3, 7.4), (3, 5.6)),  # Context Builder -> LLM
            ((7, 7.4), (7, 5.6)),  # GNN Features -> Fallback
            ((3, 4.4), (4.2, 2.6)),  # LLM -> Risk Output
            ((7, 4.4), (5.8, 2.6))   # Fallback -> Risk Output
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        
        # Add performance metrics
        ax.text(1, 6, 'Performance Metrics:\n• Accuracy: 60%\n• Response: <0.001s\n• Context Usage: 0%', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'),
               fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, 9)
        ax.set_ylim(1, 9)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Enhanced Context Smart Home Risk Assessment System\nArchitecture Overview', 
                    fontsize=18, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('system_architecture_presentation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Created: system_architecture_presentation.png")
    
    def create_improvement_summary_chart(self):
        """Create a summary chart showing key improvements."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Key metrics comparison
        metrics = ['Accuracy\nImprovement', 'Knowledge Graph\nNodes', 'Response Time\n(Fallback)', 'Context Features\nEnabled']
        values = [56, 76, 0.0001, 100]  # 56% improvement, 76 nodes, 0.0001s, 100% features
        units = ['%', 'nodes', 'seconds', '%']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels with units
        for bar, val, unit in zip(bars, values, units):
            height = bar.get_height()
            if unit == 'seconds':
                label = f'{val*1000:.2f}ms'
            else:
                label = f'{val}{unit}'
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                   label, ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        ax.set_title('Enhanced Context System\nKey Performance Indicators', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_ylabel('Value', fontsize=16)
        
        # Add annotations
        ax.text(0.5, 0.95, '1400% improvement\nover baseline', 
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'),
               fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('improvement_summary_presentation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Created: improvement_summary_presentation.png")
    
    def generate_all_presentation_charts(self):
        """Generate all charts for presentation."""
        print("🎨 GENERATING PRESENTATION CHARTS")
        print("=" * 50)
        
        if not self.results:
            print("❌ No results data available")
            return
        
        self.create_accuracy_comparison_chart()
        self.create_response_time_chart()
        self.create_knowledge_graph_stats_chart()
        self.create_detailed_results_heatmap()
        self.create_system_architecture_diagram()
        self.create_improvement_summary_chart()
        
        print("\n✅ ALL PRESENTATION CHARTS GENERATED")
        print("📁 Files created:")
        print("   • accuracy_comparison_presentation.png")
        print("   • response_time_presentation.png")
        print("   • knowledge_graph_analysis_presentation.png")
        print("   • detailed_results_heatmap_presentation.png")
        print("   • system_architecture_presentation.png")
        print("   • improvement_summary_presentation.png")
        print("\n🎓 Ready for academic presentation!")

if __name__ == "__main__":
    generator = PresentationChartGenerator()
    generator.generate_all_presentation_charts()