#!/usr/bin/env python3
"""
GPU Acceleration Setup and Baseline Model Performance Analysis

This script:
1. Analyzes baseline model performance from test results
2. Provides GPU acceleration setup guidance for Ollama
3. Tests GPU availability and performance
4. Compares CPU vs GPU performance
"""

import json
import os
import subprocess
import time
from modules.ollama_client import OllamaClient

def analyze_baseline_performance():
    """Analyze baseline model performance from test results."""
    print("📊 BASELINE MODEL PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    results_files = {
        "Fallback/Rule-based (Original)": "test_results.json",
        "Ollama LLM Integration": "ollama_test_results.json"
    }
    
    performance_summary = {}
    
    for method_name, filename in results_files.items():
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                summary = data.get('summary', {})
                total_tests = summary.get('total_tests', 0)
                correct_predictions = summary.get('correct_predictions', 0)
                accuracy = summary.get('accuracy', 0.0)
                
                performance_summary[method_name] = {
                    'total_tests': total_tests,
                    'correct_predictions': correct_predictions,
                    'accuracy': accuracy * 100 if accuracy <= 1.0 else accuracy
                }
                
                print(f"\n🔍 {method_name}:")
                print(f"   Total Tests: {total_tests}")
                print(f"   Correct Predictions: {correct_predictions}")
                print(f"   Accuracy: {accuracy * 100 if accuracy <= 1.0 else accuracy:.2f}%")
                
                # Analyze some individual results for insights
                if 'results' in data and len(data['results']) > 0:
                    sample_result = data['results'][0]
                    parsing_method = sample_result.get('parsed_command', {}).get('method', 'unknown')
                    print(f"   Primary Method Used: {parsing_method}")
                    
                    # Check if LLM was actually used
                    llm_used = any(
                        result.get('parsed_command', {}).get('method') == 'llm' 
                        for result in data['results'][:10]  # Check first 10
                    )
                    print(f"   LLM Actually Used: {'Yes' if llm_used else 'No (Fallback only)'}")
                
            except Exception as e:
                print(f"   ❌ Error reading {filename}: {e}")
        else:
            print(f"\n🔍 {method_name}: File not found ({filename})")
    
    return performance_summary

def check_gpu_status():
    """Check GPU status and CUDA availability."""
    print("\n🎮 GPU STATUS CHECK")
    print("=" * 60)
    
    try:
        # Check nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')
            print("✅ NVIDIA GPU(s) Detected:")
            
            for i, gpu in enumerate(gpu_info):
                if gpu.strip():
                    parts = gpu.split(', ')
                    if len(parts) >= 3:
                        name, memory, compute_cap = parts[0], parts[1], parts[2]
                        print(f"   GPU {i}: {name}")
                        print(f"   Memory: {memory} MB")
                        print(f"   Compute Capability: {compute_cap}")
                        
                        # Check if compute capability is sufficient for Ollama
                        try:
                            cc_major = float(compute_cap.split('.')[0])
                            if cc_major >= 5.0:
                                print(f"   ✅ Compatible with Ollama (CC >= 5.0)")
                            else:
                                print(f"   ❌ Not compatible with Ollama (CC < 5.0)")
                        except:
                            print(f"   ⚠️  Could not parse compute capability")
                        print()
            
            return True, gpu_info
        else:
            print("❌ nvidia-smi failed or no NVIDIA GPU detected")
            return False, []
            
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi command timed out")
        return False, []
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA drivers may not be installed")
        return False, []
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False, []

def check_ollama_gpu_usage():
    """Check if Ollama is using GPU acceleration."""
    print("\n🤖 OLLAMA GPU USAGE CHECK")
    print("=" * 60)
    
    client = OllamaClient()
    
    if not client.is_available():
        print("❌ Ollama server is not running")
        print("   Start with: ollama serve")
        return False
    
    print("✅ Ollama server is running")
    
    # Test with a simple prompt and monitor GPU usage
    print("\n🧪 Testing GPU acceleration...")
    
    try:
        # Start GPU monitoring in background
        gpu_monitor = subprocess.Popen(
            ['nvidia-smi', 'dmon', '-s', 'u', '-c', '10', '-d', '1'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give monitor time to start
        time.sleep(2)
        
        # Run a test generation
        start_time = time.time()
        response = client.generate("Hello, how are you?", model="llama3.2:latest")
        end_time = time.time()
        
        # Stop monitoring
        gpu_monitor.terminate()
        gpu_output, _ = gpu_monitor.communicate(timeout=5)
        
        print(f"✅ Test completed in {end_time - start_time:.2f} seconds")
        print(f"Response: {response[:100]}..." if len(response) > 100 else f"Response: {response}")
        
        # Check if GPU was used
        if gpu_output and 'llama' in gpu_output.lower() or 'ollama' in gpu_output.lower():
            print("✅ GPU acceleration detected during generation")
            return True
        else:
            print("⚠️  No clear GPU usage detected - may be using CPU")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  GPU monitoring timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing GPU usage: {e}")
        return False

def gpu_acceleration_guide():
    """Provide GPU acceleration setup guide."""
    print("\n🚀 GPU ACCELERATION SETUP GUIDE")
    print("=" * 60)
    
    print("📋 Prerequisites for GPU Acceleration:")
    print("   1. NVIDIA GPU with Compute Capability 5.0+ (recommended 6.0+)")
    print("   2. NVIDIA drivers installed (version 470+ recommended)")
    print("   3. CUDA toolkit (automatically handled by Ollama)")
    print("   4. Sufficient VRAM (4GB+ for 7B models, 8GB+ for 13B models)")
    
    print("\n🔧 Setup Steps:")
    print("   1. Install/Update NVIDIA Drivers:")
    print("      - Download from: https://www.nvidia.com/drivers")
    print("      - Or use GeForce Experience for automatic updates")
    
    print("\n   2. Verify Driver Installation:")
    print("      - Run: nvidia-smi")
    print("      - Should show GPU info and driver version")
    
    print("\n   3. Ollama GPU Configuration:")
    print("      - Ollama automatically detects and uses GPU")
    print("      - No additional configuration needed for single GPU")
    print("      - For multiple GPUs, set: CUDA_VISIBLE_DEVICES=0,1")
    
    print("\n   4. Environment Variables (Optional):")
    print("      - CUDA_VISIBLE_DEVICES: Select specific GPUs")
    print("      - OLLAMA_NUM_PARALLEL: Control parallel requests")
    print("      - OLLAMA_MAX_LOADED_MODELS: Limit loaded models")
    
    print("\n⚡ Performance Optimization Tips:")
    print("   1. Use models that fit in VRAM for best performance")
    print("   2. Close other GPU-intensive applications")
    print("   3. Use quantized models (Q4, Q5) for better VRAM efficiency")
    print("   4. Consider model size vs accuracy trade-offs:")
    print("      - 7B models: Good balance, ~4-6GB VRAM")
    print("      - 13B models: Better accuracy, ~8-12GB VRAM")
    print("      - 70B models: Best accuracy, ~40GB+ VRAM")
    
    print("\n🔍 Troubleshooting:")
    print("   1. If GPU not detected:")
    print("      - Restart Ollama: ollama serve")
    print("      - Check drivers: nvidia-smi")
    print("      - Verify compute capability >= 5.0")
    
    print("\n   2. If performance is slow:")
    print("      - Check VRAM usage: nvidia-smi")
    print("      - Try smaller/quantized models")
    print("      - Close other applications using GPU")
    
    print("\n   3. If out of memory errors:")
    print("      - Use smaller models or quantized versions")
    print("      - Reduce context length")
    print("      - Unload unused models: ollama rm <model>")

def performance_comparison():
    """Compare CPU vs GPU performance expectations."""
    print("\n⚡ CPU vs GPU PERFORMANCE COMPARISON")
    print("=" * 60)
    
    print("🖥️  CPU Performance (Current):")
    print("   - Processing Time: 10-30 seconds per command")
    print("   - Memory Usage: System RAM (8-16GB recommended)")
    print("   - Pros: No additional hardware needed")
    print("   - Cons: Slow inference, high CPU usage")
    
    print("\n🎮 GPU Performance (With Acceleration):")
    print("   - Processing Time: 1-5 seconds per command (5-10x faster)")
    print("   - Memory Usage: GPU VRAM (4-8GB for 7B models)")
    print("   - Pros: Much faster inference, parallel processing")
    print("   - Cons: Requires compatible GPU, VRAM limitations")
    
    print("\n📊 Expected Speedup by Model Size:")
    print("   - 7B models: 5-10x faster with GPU")
    print("   - 13B models: 8-15x faster with GPU")
    print("   - Quantized models: 3-8x faster with GPU")
    
    print("\n💡 Recommendations:")
    print("   1. For real-time applications: Use GPU acceleration")
    print("   2. For batch processing: CPU may be acceptable")
    print("   3. For development/testing: GPU significantly improves workflow")
    print("   4. For production: GPU acceleration is highly recommended")

def main():
    """Main function to run all analyses."""
    print("🏠 SMART HOME RISK SYSTEM - GPU ACCELERATION ANALYSIS")
    print("=" * 80)
    
    # Analyze baseline performance
    performance_data = analyze_baseline_performance()
    
    # Check GPU status
    gpu_available, gpu_info = check_gpu_status()
    
    # Check Ollama GPU usage if available
    if gpu_available:
        ollama_gpu_active = check_ollama_gpu_usage()
    else:
        ollama_gpu_active = False
    
    # Provide setup guide
    gpu_acceleration_guide()
    
    # Performance comparison
    performance_comparison()
    
    # Summary and recommendations
    print("\n" + "=" * 80)
    print("📋 SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n🎯 Baseline Model Performance:")
    for method, data in performance_data.items():
        accuracy = data.get('accuracy', 0)
        print(f"   {method}: {accuracy:.1f}% accuracy")
    
    print(f"\n🎮 GPU Status: {'Available' if gpu_available else 'Not Available'}")
    if gpu_available:
        print(f"🤖 Ollama GPU Usage: {'Active' if ollama_gpu_active else 'Not Detected'}")
    
    print("\n🚀 Next Steps:")
    if not gpu_available:
        print("   1. Install/update NVIDIA drivers")
        print("   2. Verify GPU compute capability >= 5.0")
        print("   3. Restart Ollama after driver installation")
    elif not ollama_gpu_active:
        print("   1. Restart Ollama service: ollama serve")
        print("   2. Test with: ollama run llama3.2:latest")
        print("   3. Monitor GPU usage with: nvidia-smi")
    else:
        print("   ✅ GPU acceleration is working!")
        print("   1. Consider using larger models for better accuracy")
        print("   2. Optimize model selection based on VRAM")
        print("   3. Monitor performance and adjust as needed")
    
    print("\n💡 Performance Improvement Potential:")
    if gpu_available:
        print("   - Expected speedup: 5-10x faster inference")
        print("   - Better user experience with real-time responses")
        print("   - Ability to use larger, more accurate models")
    else:
        print("   - Current CPU performance: 10-30 seconds per command")
        print("   - With GPU: 1-5 seconds per command")
        print("   - Significant improvement in system responsiveness")

if __name__ == "__main__":
    main()