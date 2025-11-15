#!/bin/bash
# Quick script to build and install PortAudio from source
# Required for sounddevice on Jetson systems

set -e

echo "=========================================="
echo "  Building PortAudio from Source"
echo "=========================================="
echo ""

# Check if already installed
if ldconfig -p | grep -q libportaudio || [ -f "/usr/local/lib/libportaudio.so" ] || [ -f "/usr/lib/libportaudio.so" ]; then
    echo "✅ PortAudio library already exists"
    echo "   Testing Python import..."
    python3 -c "import sounddevice as sd; print('✅ PortAudio working!')" 2>/dev/null && {
        echo "✅ PortAudio is fully functional"
        exit 0
    } || echo "⚠️  Library exists but Python can't find it - rebuilding..."
fi

echo "📦 Installing build dependencies..."
sudo apt install -y build-essential wget autoconf automake libasound2-dev

echo ""
echo "📥 Downloading PortAudio source..."
cd /tmp
rm -rf portaudio portaudio.tgz
wget -q http://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz -O portaudio.tgz || {
    echo "❌ Failed to download PortAudio"
    exit 1
}

echo "📂 Extracting..."
tar -xzf portaudio.tgz
cd portaudio

echo "⚙️  Configuring..."
./configure || {
    echo "❌ Configuration failed"
    exit 1
}

echo "🔨 Building (this may take 2-5 minutes)..."
make -j$(nproc) || {
    echo "❌ Build failed"
    exit 1
}

echo "📦 Installing..."
sudo make install || {
    echo "❌ Installation failed"
    exit 1
}

echo "🔄 Updating library cache..."
sudo ldconfig

echo ""
echo "✅ Verifying installation..."
if ldconfig -p | grep -q portaudio; then
    echo "✅ PortAudio library registered successfully"
else
    echo "⚠️  Library installed but not in cache"
fi

# Test Python import
echo "🧪 Testing Python import..."
if python3 -c "import sounddevice as sd; print('✅ PortAudio is working!')" 2>/dev/null; then
    echo "✅ PortAudio is fully functional - sounddevice will work"
else
    echo "⚠️  Python import test failed"
    echo "   Try: source ~/aura-env/bin/activate && python3 -c 'import sounddevice as sd'"
fi

echo ""
echo "=========================================="
echo "  PortAudio Installation Complete"
echo "=========================================="
echo ""
echo "Cleaning up temporary files..."
cd /tmp
rm -rf portaudio portaudio.tgz

echo "✅ Done!"


