#!/usr/bin/env python3
"""
Compare Whisper Models Performance

This script runs both faster-whisper and whisper-container benchmarks
and provides a detailed comparison of their performance.
"""

import os
import sys
import time
import json
import subprocess
import requests
from typing import Dict, List, Optional

class WhisperModelComparison:
    def __init__(self):
        self.results = {
            'faster_whisper': {},
            'whisper_container': {},
            'comparison': {},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def run_faster_whisper_benchmark(self) -> Dict:
        """Run faster-whisper benchmark"""
        print("🔬 Running faster-whisper benchmark...")
        
        try:
            # Run the quick benchmark script
            result = subprocess.run([
                sys.executable, 'scripts/quick_whisper_benchmark.py'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Load results from JSON file
                if os.path.exists('quick_whisper_benchmark.json'):
                    with open('quick_whisper_benchmark.json', 'r') as f:
                        return json.load(f)
                else:
                    return {'error': 'No results file generated'}
            else:
                return {'error': f'Benchmark failed: {result.stderr}'}
                
        except subprocess.TimeoutExpired:
            return {'error': 'Benchmark timed out'}
        except Exception as e:
            return {'error': f'Benchmark error: {str(e)}'}
    
    def run_whisper_container_benchmark(self) -> Dict:
        """Run whisper-container benchmark"""
        print("🔬 Running whisper-container benchmark...")
        
        try:
            # Run the container test script
            result = subprocess.run([
                sys.executable, 'scripts/test_whisper_container.py'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Load results from JSON file
                if os.path.exists('whisper_container_test.json'):
                    with open('whisper_container_test.json', 'r') as f:
                        return json.load(f)
                else:
                    return {'error': 'No results file generated'}
            else:
                return {'error': f'Container test failed: {result.stderr}'}
                
        except subprocess.TimeoutExpired:
            return {'error': 'Container test timed out'}
        except Exception as e:
            return {'error': f'Container test error: {str(e)}'}
    
    def check_container_availability(self) -> bool:
        """Check if whisper container is running"""
        try:
            response = requests.get("http://localhost:5000/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def compare_results(self) -> Dict:
        """Compare the results from both benchmarks"""
        comparison = {
            'faster_whisper_available': False,
            'whisper_container_available': False,
            'winner': None,
            'performance_difference': 0,
            'recommendations': []
        }
        
        # Check faster-whisper results
        fw_results = self.results.get('faster_whisper', {})
        if 'synthetic_audio' in fw_results and 'average_latency' in fw_results['synthetic_audio']:
            comparison['faster_whisper_available'] = True
            fw_latency = fw_results['synthetic_audio']['average_latency']
            comparison['faster_whisper_latency'] = fw_latency
        
        # Check whisper-container results
        wc_results = self.results.get('whisper_container', {})
        if 'container_test' in wc_results and 'average_latency' in wc_results['container_test']:
            comparison['whisper_container_available'] = True
            wc_latency = wc_results['container_test']['average_latency']
            comparison['whisper_container_latency'] = wc_latency
        
        # Compare if both are available
        if comparison['faster_whisper_available'] and comparison['whisper_container_available']:
            fw_latency = comparison['faster_whisper_latency']
            wc_latency = comparison['whisper_container_latency']
            
            if fw_latency < wc_latency:
                comparison['winner'] = 'faster-whisper'
                comparison['performance_difference'] = ((wc_latency - fw_latency) / wc_latency) * 100
                comparison['speed_ratio'] = wc_latency / fw_latency
            else:
                comparison['winner'] = 'whisper-container'
                comparison['performance_difference'] = ((fw_latency - wc_latency) / fw_latency) * 100
                comparison['speed_ratio'] = fw_latency / wc_latency
        
        # Generate recommendations
        if comparison['faster_whisper_available']:
            fw_efficiency = fw_results.get('synthetic_audio', {}).get('overall_efficiency', 0)
            if fw_efficiency > 2.0:
                comparison['recommendations'].append("faster-whisper shows excellent real-time performance")
            elif fw_efficiency > 1.0:
                comparison['recommendations'].append("faster-whisper performs faster than real-time")
            else:
                comparison['recommendations'].append("faster-whisper may be too slow for real-time use")
        
        if comparison['whisper_container_available']:
            comparison['recommendations'].append("whisper-container is available and running")
        else:
            comparison['recommendations'].append("whisper-container not available - start with 'docker compose up whisper'")
        
        return comparison
    
    def run_comparison(self) -> Dict:
        """Run complete comparison"""
        print("🚀 Starting Whisper Models Comparison...")
        print("="*60)
        
        # Check container availability first
        container_available = self.check_container_availability()
        print(f"🐳 Whisper container available: {container_available}")
        
        # Run faster-whisper benchmark
        print("\n1️⃣ Running faster-whisper benchmark...")
        self.results['faster_whisper'] = self.run_faster_whisper_benchmark()
        
        # Run whisper-container benchmark if available
        if container_available:
            print("\n2️⃣ Running whisper-container benchmark...")
            self.results['whisper_container'] = self.run_whisper_container_benchmark()
        else:
            print("\n⚠️ Skipping whisper-container benchmark (not available)")
            self.results['whisper_container'] = {'error': 'Container not available'}
        
        # Compare results
        print("\n3️⃣ Comparing results...")
        self.results['comparison'] = self.compare_results()
        
        # Save results
        with open('whisper_models_comparison.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print comparison summary"""
        print("\n" + "="*60)
        print("🎯 WHISPER MODELS COMPARISON SUMMARY")
        print("="*60)
        
        # Faster-whisper results
        fw = self.results.get('faster_whisper', {})
        if 'synthetic_audio' in fw:
            fw_synth = fw['synthetic_audio']
            print(f"\n🔬 faster-whisper (distill.small):")
            print(f"  ⏱️  Model loading: {fw_synth.get('model_loading_time', 0):.2f}s")
            print(f"  ⏱️  Average latency: {fw_synth.get('average_latency', 0):.2f}s")
            print(f"  ⚡ Efficiency: {fw_synth.get('overall_efficiency', 0):.2f}x real-time")
            if 'real_audio' in fw and 'average_real_latency' in fw['real_audio']:
                print(f"  🎵 Real audio latency: {fw['real_audio']['average_real_latency']:.2f}s")
        else:
            print(f"\n❌ faster-whisper benchmark failed: {fw.get('error', 'Unknown error')}")
        
        # Whisper-container results
        wc = self.results.get('whisper_container', {})
        if 'container_test' in wc:
            wc_test = wc['container_test']
            print(f"\n🐳 whisper-container (TensorRT base.en):")
            print(f"  🐳 Container available: {wc_test.get('container_available', False)}")
            print(f"  ⏱️  Average latency: {wc_test.get('average_latency', 0):.2f}s")
            if 'real_audio_test' in wc and 'average_real_latency' in wc['real_audio_test']:
                print(f"  🎵 Real audio latency: {wc['real_audio_test']['average_real_latency']:.2f}s")
        else:
            print(f"\n❌ whisper-container benchmark failed: {wc.get('error', 'Unknown error')}")
        
        # Comparison
        comp = self.results.get('comparison', {})
        if comp.get('faster_whisper_available') and comp.get('whisper_container_available'):
            print(f"\n🏆 WINNER: {comp.get('winner', 'Unknown')}")
            print(f"  📈 Performance difference: {comp.get('performance_difference', 0):.1f}%")
            print(f"  ⚡ Speed ratio: {comp.get('speed_ratio', 1):.2f}x")
        elif comp.get('faster_whisper_available'):
            print(f"\n🔬 Only faster-whisper results available")
        elif comp.get('whisper_container_available'):
            print(f"\n🐳 Only whisper-container results available")
        else:
            print(f"\n❌ No valid results for comparison")
        
        # Recommendations
        if comp.get('recommendations'):
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in comp['recommendations']:
                print(f"  • {rec}")
        
        print(f"\n💾 Full results saved to: whisper_models_comparison.json")
        print("="*60)

def main():
    """Main execution"""
    comparison = WhisperModelComparison()
    results = comparison.run_comparison()
    
    print("\n🎉 Comparison complete!")
    return results

if __name__ == "__main__":
    main()
