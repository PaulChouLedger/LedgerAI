# Modified Data Download Section with TTS Echo Sample Upload

Copy-paste this into your Colab notebook to replace the data download section:

```python
# @title  { display-mode: "form" }

# @markdown # 2. Download Data

# @markdown Training custom models requires downloading a wide variety of data

# @markdown that will help make the model perform well in real-world scenarios.

# @markdown This example notebook will download small samples of background noise,

# @markdown music, and Room Impulse Responses (to add echo). This will still produce

# @markdown a custom model that performs well, but if you are interested in adding even more,

# @markdown feel free to extend this notebook to download the full datasets and even add

# @markdown your own!

# @markdown

# @markdown **⚠️ CRITICAL: Echo Emphasis in Negative Samples**

# @markdown **Echo from TTS playback is a major source of false positives!** When your device

# @markdown plays TTS audio (like "hey aura"), the microphone picks up the echo, which can

# @markdown trigger false wake word detections. It is **ESSENTIAL** to include TTS echo samples

# @markdown in your negative training data to prevent this.

# @markdown

# @markdown **Why echo matters:**
# @markdown - TTS audio played through speakers creates echo/reverb
# @markdown - Microphone picks up this echo, which sounds similar to the wake word
# @markdown - Without echo samples, the model will trigger on TTS playback
# @markdown - Echo samples teach the model to distinguish real speech from TTS echo

# @markdown

# @markdown Downloading this example data will usually take about 15 minutes.

# @markdown **Important note!** The data downloaded here has a mixture of different

# @markdown licenses and usage restrictions. As such, any custom models trained with this

# @markdown data should be considered as appropriate for **non-commercial** personal use only.

# ## Install all dependencies

# !pip install datasets

# !pip install scipy

# !pip install tqdm

import locale

def getpreferredencoding(do_setlocale = True):

    return "UTF-8"

locale.getpreferredencoding = getpreferredencoding

# install openwakeword (full installation to support training)

!git clone https://github.com/dscripka/openwakeword

!pip install -e ./openwakeword --no-deps

# install other dependencies

!pip install mutagen==1.47.0

!pip install torchinfo==1.8.0

!pip install torchmetrics==1.2.0

!pip install speechbrain==0.5.14

!pip install audiomentations==0.33.0

!pip install torch-audiomentations==0.11.0

!pip install acoustics==0.2.6

!pip install onnxruntime==1.22.1 ai_edge_litert==1.4.0 onnxsim

!pip install onnx2tf

!pip install onnx

!pip install onnx_graphsurgeon

!pip install sng4onnx

!pip install pronouncing==0.2.0

!pip install datasets==2.14.6

!pip install deep-phonemizer==0.0.19

# Download required models (workaround for Colab)

import os

os.makedirs("./openwakeword/openwakeword/resources/models", exist_ok=True)

!wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx -O ./openwakeword/openwakeword/resources/models/embedding_model.onnx

!wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite -O ./openwakeword/openwakeword/resources/models/embedding_model.tflite

!wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx -O ./openwakeword/openwakeword/resources/models/melspectrogram.onnx

!wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite -O ./openwakeword/openwakeword/resources/models/melspectrogram.tflite

# Imports

import sys

if "piper-sample-generator/" not in sys.path:

    sys.path.append("piper-sample-generator/")

from generate_samples import generate_samples

import numpy as np

import torch

import sys

from pathlib import Path

import uuid

import yaml

import datasets

import scipy

from tqdm import tqdm

## Download all data

## Download MIR RIR data (takes about ~2 minutes)

output_dir = "./mit_rirs"

if not os.path.exists(output_dir):

    os.mkdir(output_dir)

    !git lfs install

    !git clone https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses

    rir_dataset = datasets.Dataset.from_dict({"audio": [str(i) for i in Path("./MIT_environmental_impulse_responses/16khz").glob("*.wav")]}).cast_column("audio", datasets.Audio())

    # Save clips to 16-bit PCM wav files

    for row in tqdm(rir_dataset):

        name = row['audio']['path'].split('/')[-1]

        scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))

## Download noise and background audio (takes about ~3 minutes)

# Audioset Dataset (https://research.google.com/audioset/dataset/index.html)

# Download one part of the audioset .tar files, extract, and convert to 16khz

# For full-scale training, it's recommended to download the entire dataset from

# https://huggingface.co/datasets/agkphysics/AudioSet, and

# even potentially combine it with other background noise datasets (e.g., FSD50k, Freesound, etc.)

if not os.path.exists("audioset"):

    os.mkdir("audioset")

    fname = "bal_train09.tar"

    out_dir = f"audioset/{fname}"

    link = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/" + fname

    !wget -O {out_dir} {link}

    !cd audioset && tar -xvf bal_train09.tar

    output_dir = "./audioset_16k"

    if not os.path.exists(output_dir):

        os.mkdir(output_dir)

    # Save clips to 16-bit PCM wav files

    audioset_dataset = datasets.Dataset.from_dict({"audio": [str(i) for i in Path("audioset/audio").glob("**/*.flac")]})

    audioset_dataset = audioset_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))

    for row in tqdm(audioset_dataset):

        name = row['audio']['path'].split('/')[-1].replace(".flac", ".wav")

        scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))

# Free Music Archive dataset

# https://github.com/mdeff/fma

output_dir = "./fma"

if not os.path.exists(output_dir):

    os.mkdir(output_dir)

    fma_dataset = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)

    fma_dataset = iter(fma_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000)))

    # Save clips to 16-bit PCM wav files

    n_hours = 1  # use only 1 hour of clips for this example notebook, recommend increasing for full-scale training

    for i in tqdm(range(n_hours*3600//30)):  # this works because the FMA dataset is all 30 second clips

        row = next(fma_dataset)

        name = row['audio']['path'].split('/')[-1].replace(".mp3", ".wav")

        scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))

        i += 1

        if i == n_hours*3600//30:

            break

# ============================================================================

# @markdown ## 🎤 Upload TTS Echo Samples (CRITICAL for False Positive Prevention)

# @markdown 

# @markdown **This section allows you to upload TTS echo samples recorded from your device.**

# @markdown These samples are ESSENTIAL for preventing false positives when TTS plays the wake word.

# @markdown 

# @markdown **How to record TTS echo samples:**

# @markdown 1. On your device, play TTS audio saying your wake word (e.g., "hey aura")

# @markdown 2. Record the audio using your microphone (the echo/reverb from speakers)

# @markdown 3. Save as 16kHz mono WAV files

# @markdown 4. Upload multiple samples (10-50 recommended) with different:

# @markdown    - Volume levels (quiet, medium, loud)

# @markdown    - Room conditions (different echo/reverb)

# @markdown    - Distances from speaker

# @markdown    - Background noise levels

# @markdown 

# @markdown **Why this matters:** Without TTS echo samples, your model will trigger on TTS playback,

# @markdown causing false wake word detections every time the device speaks.

# ============================================================================

from google.colab import files

import shutil

# Create directory for TTS echo samples

tts_echo_dir = "./tts_echo_samples"

os.makedirs(tts_echo_dir, exist_ok=True)

print("="*80)

print("📤 TTS ECHO SAMPLE UPLOAD")

print("="*80)

print("\nThis will allow you to upload TTS echo samples recorded from your device.")

print("These samples are used as NEGATIVE examples to prevent false positives.")

print("\nInstructions:")

print("1. Click 'Choose Files' below")

print("2. Select your recorded TTS echo WAV files (16kHz mono recommended)")

print("3. Upload multiple samples (10-50 recommended)")

print("4. Files will be saved to:", tts_echo_dir)

print("\n" + "="*80)

# Upload files

uploaded = files.upload()

# Process uploaded files

if uploaded:

    print(f"\n✅ Uploaded {len(uploaded)} file(s)")

    for filename, file_content in uploaded.items():

        # Save to TTS echo directory

        filepath = os.path.join(tts_echo_dir, filename)

        with open(filepath, 'wb') as f:

            f.write(file_content)

        print(f"  📁 Saved: {filename}")

        # Verify it's a valid audio file and convert to 16kHz if needed

        try:

            import soundfile as sf

            data, sr = sf.read(filepath)

            if sr != 16000:

                print(f"  ⚠️  Converting {filename} from {sr}Hz to 16000Hz...")

                # Resample to 16kHz

                from scipy import signal

                num_samples = int(len(data) * 16000 / sr)

                data_resampled = signal.resample(data, num_samples)

                # Convert to mono if stereo

                if len(data_resampled.shape) > 1:

                    data_resampled = np.mean(data_resampled, axis=1)

                # Save as 16-bit PCM WAV

                scipy.io.wavfile.write(filepath, 16000, (data_resampled * 32767).astype(np.int16))

                print(f"  ✅ Converted {filename} to 16kHz mono")

            else:

                # Ensure mono and 16-bit

                if len(data.shape) > 1:

                    data = np.mean(data, axis=1)

                scipy.io.wavfile.write(filepath, 16000, (data * 32767).astype(np.int16))

                print(f"  ✅ Verified {filename} (16kHz mono)")

        except Exception as e:

            print(f"  ⚠️  Warning: Could not verify/convert {filename}: {e}")

            print(f"     Please ensure it's a 16kHz mono WAV file")

    print(f"\n✅ All TTS echo samples saved to: {tts_echo_dir}")

    print(f"   Total files: {len(os.listdir(tts_echo_dir))}")

else:

    print("\n⚠️  No files uploaded. You can run this cell again to upload TTS echo samples.")

    print("   Note: TTS echo samples are highly recommended to prevent false positives!")

# Download pre-computed openWakeWord features for training and validation

# training set (~2,000 hours from the ACAV100M Dataset)

# See https://huggingface.co/datasets/davidscripka/openwakeword_features for more information

if not os.path.exists("./openwakeword_features_ACAV100M_2000_hrs_16bit.npy"):

    !wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy

# validation set for false positive rate estimation (~11 hours)

if not os.path.exists("validation_set_features.npy"):

    !wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy

print("\n" + "="*80)

print("✅ Data download complete!")

print("="*80)

print(f"\n📊 Summary:")

print(f"  - MIT RIR data: {len(os.listdir('./mit_rirs')) if os.path.exists('./mit_rirs') else 0} files")

print(f"  - AudioSet background: {len(os.listdir('./audioset_16k')) if os.path.exists('./audioset_16k') else 0} files")

print(f"  - FMA music: {len(os.listdir('./fma')) if os.path.exists('./fma') else 0} files")

print(f"  - TTS echo samples: {len(os.listdir(tts_echo_dir)) if os.path.exists(tts_echo_dir) else 0} files")

if os.path.exists(tts_echo_dir) and len(os.listdir(tts_echo_dir)) > 0:

    print(f"\n✅ TTS echo samples uploaded! These will be used as negative examples.")

    print(f"   This is critical for preventing false positives from TTS playback.")

else:

    print(f"\n⚠️  No TTS echo samples uploaded. Consider adding them to improve model robustness.")

print("="*80)
```

## Key Changes Made:

1. **Added emphasis on echo in negative samples** - Clear warnings about why TTS echo matters
2. **Added TTS echo sample upload section** - Interactive file upload with:
   - Clear instructions on how to record TTS echo samples
   - Automatic file validation and conversion to 16kHz mono
   - Summary showing how many samples were uploaded
3. **Enhanced documentation** - Explains why echo samples are critical
4. **File processing** - Automatically converts uploaded files to the correct format
5. **Summary statistics** - Shows count of all data types including TTS samples

This cell can be copy-pasted directly into your Colab notebook to replace the original data download section.

