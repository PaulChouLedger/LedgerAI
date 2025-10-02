# Whisper Model Benchmarking

This directory contains comprehensive benchmarking tools to compare the performance of different Whisper implementations:

1. **faster-whisper** (distill.small) - Used in `transcription_tuner.py`
2. **whisper-container** (TensorRT base.en) - Used in the whisper container

## 🎯 Quick Start

### 1. Setup Environment
```bash
# Run setup script
./scripts/setup_benchmark.sh

# Activate environment
source benchmark-env/bin/activate
```

### 2. Run Quick Test (faster-whisper only)
```bash
python scripts/quick_whisper_benchmark.py
```

### 3. Run Full Comparison (both models)
```bash
# Start whisper container
docker compose up whisper

# Run comparison
python scripts/compare_whisper_models.py
```

## 📊 Benchmark Scripts

### `quick_whisper_benchmark.py`
- Tests faster-whisper with distill.small model
- Uses synthetic audio of varying durations (1s, 2s, 3s, 5s, 10s)
- Measures model loading time, transcription latency, and efficiency
- Tests with real audio files if available

**Metrics:**
- Model loading time
- Average transcription latency
- Real-time efficiency (x real-time)
- Memory usage
- Standard deviation

### `test_whisper_container.py`
- Tests whisper-container with TensorRT optimization
- Tests container availability and health
- Measures transcription latency via HTTP API
- Uses same test audio as faster-whisper

**Metrics:**
- Container availability
- Average transcription latency
- Real audio performance
- Error handling

### `compare_whisper_models.py`
- Runs both benchmarks and compares results
- Determines performance winner
- Provides recommendations
- Generates comprehensive report

## 🔬 Test Methodology

### Audio Samples
- **Synthetic**: Generated speech-like audio with multiple frequencies
- **Real Audio**: Uses existing `.wav`, `.mp3`, `.flac` files in project
- **Durations**: 1s, 2s, 3s, 5s, 10s for comprehensive testing

### Performance Metrics
- **Latency**: Time from audio input to transcription output
- **Efficiency**: Real-time performance (audio_duration / transcription_time)
- **Memory Usage**: RAM consumption during transcription
- **Model Loading**: Time to initialize the model

### Test Conditions
- **GPU**: CUDA acceleration enabled
- **Precision**: float16 for faster-whisper, TensorRT optimization for container
- **Language**: English only
- **Beam Size**: 5 for faster-whisper

## 📈 Expected Results

### faster-whisper (distill.small)
- **Pros**: 
  - Fast model loading (~2-3s)
  - Good real-time performance (2-3x real-time)
  - Direct Python integration
  - Smaller model size
- **Cons**: 
  - May be less accurate than larger models
  - Limited to distill.small architecture

### whisper-container (TensorRT base.en)
- **Pros**:
  - TensorRT optimization for Jetson
  - Better accuracy (base.en vs distill.small)
  - Containerized deployment
  - HTTP API interface
- **Cons**:
  - Slower model loading
  - Container overhead
  - More complex deployment

## 🎛️ Configuration

### Environment Variables
```bash
# For faster-whisper
export CUDA_VISIBLE_DEVICES=0  # GPU device
export WHISPER_MODEL=distil-small.en  # Model size

# For whisper-container
export WHISPER_CONTAINER_URL=http://localhost:5000
```

### Customization
Edit the benchmark scripts to:
- Change test audio durations
- Modify model parameters (beam_size, language)
- Add custom audio files
- Adjust performance thresholds

## 📊 Output Files

### `quick_whisper_benchmark.json`
```json
{
  "synthetic_audio": {
    "model_loading_time": 2.34,
    "average_latency": 1.23,
    "overall_efficiency": 2.45
  },
  "real_audio": {
    "average_real_latency": 1.18
  },
  "system_info": {
    "cuda_available": true,
    "gpu_name": "NVIDIA Jetson Orin NX"
  }
}
```

### `whisper_container_test.json`
```json
{
  "container_test": {
    "container_available": true,
    "average_latency": 1.45,
    "transcription_times": [1.2, 1.5, 1.6]
  },
  "real_audio_test": {
    "average_real_latency": 1.38
  }
}
```

### `whisper_models_comparison.json`
```json
{
  "comparison": {
    "winner": "faster-whisper",
    "performance_difference": 15.2,
    "speed_ratio": 1.18,
    "recommendations": [
      "faster-whisper shows excellent real-time performance",
      "whisper-container is available and running"
    ]
  }
}
```

## 🚀 Usage Examples

### Quick Performance Check
```bash
# Test faster-whisper only
python scripts/quick_whisper_benchmark.py

# Check results
cat quick_whisper_benchmark.json | jq '.synthetic_audio.average_latency'
```

### Full Comparison
```bash
# Start whisper container
docker compose up whisper &

# Wait for container to be ready
sleep 30

# Run comparison
python scripts/compare_whisper_models.py

# View results
cat whisper_models_comparison.json | jq '.comparison'
```

### Custom Testing
```python
# Test specific audio file
from scripts.quick_whisper_benchmark import QuickWhisperBenchmark

benchmark = QuickWhisperBenchmark()
results = benchmark.benchmark_with_real_audio(['path/to/audio.wav'])
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA not available**
   ```bash
   # Check CUDA installation
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Container not responding**
   ```bash
   # Check container status
   docker ps
   curl http://localhost:5000/health
   ```

3. **Memory issues**
   ```bash
   # Monitor memory usage
   htop
   # Reduce test durations in scripts
   ```

### Performance Optimization

1. **For faster-whisper**:
   - Use `compute_type="float16"` for speed
   - Reduce `beam_size` for faster inference
   - Use smaller models (distill.small vs base)

2. **For whisper-container**:
   - Ensure TensorRT optimization is enabled
   - Use appropriate batch sizes
   - Monitor container resource usage

## 📋 Recommendations

Based on benchmark results:

- **Real-time applications**: Use faster-whisper if latency < 1s is critical
- **Accuracy-critical**: Use whisper-container with larger models
- **Resource-constrained**: Use faster-whisper with distill.small
- **Production deployment**: Use whisper-container for scalability

## 🔄 Continuous Benchmarking

Set up automated benchmarking:

```bash
# Add to crontab for daily benchmarks
0 2 * * * cd /path/to/LedgerAI && source benchmark-env/bin/activate && python scripts/compare_whisper_models.py
```

This ensures performance monitoring and early detection of regressions.
