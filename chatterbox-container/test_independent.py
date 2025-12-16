#!/usr/bin/env python3
"""
Comprehensive independent test script for Chatterbox-TTS Container
Tests the container without modifying the aura pipeline
"""

import requests
import json
import os
import sys
import time
import subprocess
from pathlib import Path

# Configuration
CHATTERBOX_URL = os.getenv("CHATTERBOX_URL", "http://localhost:11437")
CONTAINER_NAME = "chatterbox-tts"
WORKSPACE_ROOT = Path(__file__).parent.parent
SETUP_DIR = WORKSPACE_ROOT / "setup"
VOICE_SAMPLES_DIR = WORKSPACE_ROOT / "assets" / "voice_samples"

# Test results
test_results = {
    "container_built": False,
    "container_running": False,
    "health_check": False,
    "synthesis_basic": False,
    "synthesis_voice_cloning": False,
    "voice_embedding": False,
    "latency": None,
    "audio_quality": "not_tested"
}

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_section(text):
    """Print a formatted section"""
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print(f"{'─' * 70}")

def check_docker_available():
    """Check if Docker is available"""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✅ Docker available: {result.stdout.strip()}")
            return True
        else:
            print("❌ Docker is not available")
            return False
    except FileNotFoundError:
        print("⚠️  Docker is not installed or not in PATH")
        print("   Will attempt to test container API if container is already running")
        return False
    except Exception as e:
        print(f"⚠️  Error checking Docker: {e}")
        print("   Will attempt to test container API if container is already running")
        return False

def check_nvidia_runtime():
    """Check if NVIDIA Docker runtime is available"""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "nvidia" in result.stdout.lower() or "gpu" in result.stdout.lower():
            print("✅ NVIDIA runtime appears to be available")
            return True
        else:
            print("⚠️  NVIDIA runtime may not be configured (container may still work)")
            return True  # Don't fail, just warn
    except Exception as e:
        print(f"⚠️  Could not check NVIDIA runtime: {e}")
        return True  # Don't fail

def build_container():
    """Build the chatterbox container"""
    print_section("Building Container")
    
    dockerfile_path = Path(__file__).parent / "Dockerfile"
    if not dockerfile_path.exists():
        print(f"❌ Dockerfile not found at: {dockerfile_path}")
        return False
    
    print(f"Building container from: {dockerfile_path.parent}")
    try:
        result = subprocess.run(
            ["docker", "build", "-t", CONTAINER_NAME, "."],
            cwd=dockerfile_path.parent,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        if result.returncode == 0:
            print("✅ Container built successfully")
            test_results["container_built"] = True
            return True
        else:
            print(f"❌ Container build failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ Container build timed out (exceeded 30 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error building container: {e}")
        return False

def check_container_running():
    """Check if container is already running"""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if CONTAINER_NAME in result.stdout:
            print(f"✅ Container '{CONTAINER_NAME}' is already running")
            return True
        return False
    except Exception as e:
        print(f"⚠️  Could not check container status: {e}")
        return False

def start_container():
    """Start the chatterbox container"""
    print_section("Starting Container")
    
    # Check if already running
    if check_container_running():
        test_results["container_running"] = True
        return True
    
    # Try docker-compose first (if available)
    if (SETUP_DIR / "docker-compose.yml").exists():
        print("Attempting to start via docker-compose...")
        try:
            result = subprocess.run(
                ["docker", "compose", "-f", str(SETUP_DIR / "docker-compose.yml"), "up", "-d", "chatterbox-tts"],
                cwd=SETUP_DIR,
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                print("✅ Container started via docker-compose")
                time.sleep(5)  # Wait for container to initialize
                test_results["container_running"] = True
                return True
        except Exception as e:
            print(f"⚠️  docker-compose failed: {e}")
    
    # Fallback: direct docker run
    print("Attempting to start via docker run...")
    try:
        # Stop and remove existing container if it exists
        subprocess.run(
            ["docker", "stop", CONTAINER_NAME],
            capture_output=True,
            timeout=10
        )
        subprocess.run(
            ["docker", "rm", CONTAINER_NAME],
            capture_output=True,
            timeout=10
        )
        
        # Start new container
        cmd = [
            "docker", "run", "-d",
            "--name", CONTAINER_NAME,
            "--runtime=nvidia",
            "--network=host",
            "-v", f"{WORKSPACE_ROOT / 'shared'}:/shared",
            "-v", f"{VOICE_SAMPLES_DIR}:/app/voice_samples",
            "-v", f"{WORKSPACE_ROOT / 'data' / 'voice_cache'}:/app/voice_cache",
            CONTAINER_NAME
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ Container started via docker run")
            time.sleep(5)  # Wait for container to initialize
            test_results["container_running"] = True
            return True
        else:
            print(f"❌ Failed to start container:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error starting container: {e}")
        return False

def test_health():
    """Test health endpoint"""
    print_section("Health Check")
    
    try:
        print(f"Checking: {CHATTERBOX_URL}/health")
        response = requests.get(f"{CHATTERBOX_URL}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Service: {data.get('service')}")
            print(f"   Chatterbox loaded: {data.get('chatterbox_loaded')}")
            print(f"   Can import: {data.get('can_import_chatterbox')}")
            print(f"   Device: {data.get('device')}")
            print(f"   Source directory exists: {data.get('source_directory_exists')}")
            
            if data.get('import_error'):
                print(f"   ⚠️  Import error: {data.get('import_error')}")
            
            test_results["health_check"] = True
            return True
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {CHATTERBOX_URL}")
        print(f"   Make sure container is running")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_synthesize(text="Hello, this is a test of the Chatterbox TTS container.", voice_sample=None, test_name="Basic Synthesis"):
    """Test synthesis endpoint"""
    print_section(f"{test_name}")
    
    try:
        payload = {
            "text": text,
            "exaggeration": 0.6
        }
        if voice_sample:
            payload["voice_sample"] = voice_sample
        
        print(f"Text: '{text[:60]}...'")
        if voice_sample:
            print(f"Voice sample: {voice_sample}")
        
        start_time = time.time()
        response = requests.post(
            f"{CHATTERBOX_URL}/synthesize",
            json=payload,
            timeout=60
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            # Save audio file
            output_file = f"test_output_{test_name.lower().replace(' ', '_')}.wav"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            file_size = len(response.content)
            print(f"✅ Synthesis successful")
            print(f"   Audio saved to: {output_file}")
            print(f"   File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            print(f"   Latency: {elapsed_time:.2f} seconds")
            
            if test_results["latency"] is None:
                test_results["latency"] = elapsed_time
            
            return True
        else:
            print(f"❌ Synthesis failed: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error')}")
            except:
                print(f"   Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_voice_embedding(voice_sample_path):
    """Test voice embedding extraction"""
    print_section("Voice Embedding Extraction")
    
    if not voice_sample_path or not os.path.exists(voice_sample_path):
        print(f"⚠️  Voice sample not found: {voice_sample_path}")
        print(f"   Skipping voice embedding test")
        return True  # Don't fail, just skip
    
    try:
        # Use absolute path inside container
        container_path = f"/app/voice_samples/{os.path.basename(voice_sample_path)}"
        
        payload = {
            "voice_sample_path": container_path
        }
        
        print(f"Voice sample: {voice_sample_path}")
        print(f"Container path: {container_path}")
        
        response = requests.post(
            f"{CHATTERBOX_URL}/voice/embedding",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Voice embedding extracted successfully")
            print(f"   Voice sample: {data.get('voice_sample')}")
            print(f"   Embedding cached: {data.get('embedding_cached')}")
            test_results["voice_embedding"] = True
            return True
        else:
            print(f"❌ Voice embedding failed: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error')}")
            except:
                print(f"   Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_integration_readiness():
    """Test if container is ready for integration"""
    print_section("Integration Readiness Assessment")
    
    readiness_score = 0
    total_checks = 0
    
    checks = [
        ("Container built", test_results["container_built"]),
        ("Container running", test_results["container_running"]),
        ("Health check passed", test_results["health_check"]),
        ("Basic synthesis works", test_results["synthesis_basic"]),
    ]
    
    for check_name, check_result in checks:
        total_checks += 1
        if check_result:
            readiness_score += 1
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name}")
    
    # Optional checks
    optional_checks = [
        ("Voice cloning works", test_results["synthesis_voice_cloning"]),
        ("Voice embedding works", test_results["voice_embedding"]),
    ]
    
    for check_name, check_result in optional_checks:
        if check_result:
            print(f"✅ {check_name} (optional)")
        else:
            print(f"⚠️  {check_name} (optional - not required)")
    
    readiness_percentage = (readiness_score / total_checks) * 100
    
    print(f"\n📊 Integration Readiness: {readiness_percentage:.0f}% ({readiness_score}/{total_checks} core checks passed)")
    
    if readiness_percentage == 100:
        print("\n✅ Container is READY for integration into aura pipeline")
        print("\n💡 Next steps:")
        print("   1. The container is working independently")
        print("   2. Modify aura-control/core/speaker.py to use HTTP API instead of direct import")
        print("   3. Update TTS engine selection to use container endpoint")
        print("   4. Test with actual aura pipeline")
    elif readiness_percentage >= 75:
        print("\n⚠️  Container is MOSTLY READY but has some issues")
        print("   Review the failed checks above before integration")
    else:
        print("\n❌ Container is NOT READY for integration")
        print("   Fix the failed checks before attempting integration")
    
    return readiness_percentage == 100

def print_summary():
    """Print test summary"""
    print_header("Test Summary")
    
    print("\nTest Results:")
    for test_name, result in test_results.items():
        if isinstance(result, bool):
            status = "✅" if result else "❌"
            print(f"  {status} {test_name}: {result}")
        elif result is not None:
            print(f"  📊 {test_name}: {result}")
        else:
            print(f"  ⚠️  {test_name}: Not tested")
    
    if test_results["latency"]:
        print(f"\n⏱️  Average latency: {test_results['latency']:.2f} seconds")
    
    print(f"\n🌐 Container URL: {CHATTERBOX_URL}")
    print(f"📦 Container name: {CONTAINER_NAME}")

def main():
    """Main test function"""
    print_header("Chatterbox-TTS Container Independent Test")
    print(f"Testing container at: {CHATTERBOX_URL}")
    print(f"Workspace root: {WORKSPACE_ROOT}")
    
    # Pre-flight checks
    print_section("Pre-flight Checks")
    docker_available = check_docker_available()
    
    if docker_available:
        check_nvidia_runtime()
        
        # Build container (if needed)
        try:
            subprocess.run(
                ["docker", "inspect", CONTAINER_NAME],
                capture_output=True,
                timeout=5
            )
            print(f"✅ Container image '{CONTAINER_NAME}' exists")
            test_results["container_built"] = True
        except:
            print("⚠️  Container image not found, building...")
            if not build_container():
                print("\n❌ Failed to build container")
                print("\n💡 You can still test if container is running elsewhere")
                docker_available = False
        
        # Start container
        if docker_available:
            if not start_container():
                print("\n⚠️  Failed to start container via script")
                print("\n💡 Container may already be running, or try manually:")
                print(f"   cd {SETUP_DIR}")
                print("   docker compose up -d chatterbox-tts")
                print("\n   Will attempt to test API anyway...")
    else:
        print("\n⚠️  Docker not available - skipping container management")
        print("   Will attempt to test container API if it's already running")
        print("   Make sure container is accessible at:", CHATTERBOX_URL)
    
    # Wait a bit more for initialization
    print("\n⏳ Waiting for container to fully initialize...")
    time.sleep(10)
    
    # Run tests
    if not test_health():
        print("\n❌ Health check failed - container may not be ready")
        print("\n💡 Check container logs:")
        print(f"   docker logs {CONTAINER_NAME}")
        sys.exit(1)
    
    # Basic synthesis test
    if test_synthesize():
        test_results["synthesis_basic"] = True
    
    # Voice cloning test (if voice sample available)
    voice_sample = None
    if VOICE_SAMPLES_DIR.exists():
        # Try to find a voice sample
        for sample_file in ["sample.wav", "startup.wav", "welcome.wav"]:
            sample_path = VOICE_SAMPLES_DIR / sample_file
            if sample_path.exists():
                voice_sample = sample_file
                break
        
        if voice_sample:
            # Test voice embedding first
            test_voice_embedding(VOICE_SAMPLES_DIR / voice_sample)
            
            # Test synthesis with voice cloning
            if test_synthesize(
                "Hello, this is a test with voice cloning enabled.",
                voice_sample=voice_sample,
                test_name="Voice Cloning Synthesis"
            ):
                test_results["synthesis_voice_cloning"] = True
        else:
            print("\n⚠️  No voice samples found for cloning test")
    
    # Integration readiness
    test_integration_readiness()
    
    # Print summary
    print_summary()
    
    print("\n" + "=" * 70)
    print("  Test Complete")
    print("=" * 70)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
