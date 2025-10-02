#!/usr/bin/env bash
set -ex

# === Settings ===
CTRANSLATE_VERSION="4.6.0"
CTRANSLATE_BRANCH="v${CTRANSLATE_VERSION}"
CTRANSLATE_SOURCE="$HOME/CTranslate2"

echo "🔧 Building CTranslate2 ${CTRANSLATE_VERSION} from ${CTRANSLATE_BRANCH}"

# === Clone sources ===
git clone --branch=${CTRANSLATE_BRANCH} --recursive https://github.com/OpenNMT/CTranslate2.git ${CTRANSLATE_SOURCE} || \
git clone --recursive https://github.com/OpenNMT/CTranslate2.git ${CTRANSLATE_SOURCE}

# === Build directory ===
mkdir -p ${CTRANSLATE_SOURCE}/build
cd ${CTRANSLATE_SOURCE}/build

install_dir="${CTRANSLATE_SOURCE}/build/install"

# === Patch CMakeLists.txt to avoid libiomp5 error ===
sed -i '/Intel OpenMP runtime libiomp5 not found/d' ${CTRANSLATE_SOURCE}/CMakeLists.txt

# === Build C++ core ===
cmake .. \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=ON \
  -DWITH_MKL=OFF \
  -DOPENMP_RUNTIME=COMP \
  -DCMAKE_INSTALL_PREFIX=${install_dir}

make -j$(nproc)
make install

# === Copy to /usr/local ===
sudo cp -r ${install_dir}/* /usr/local/
sudo ldconfig

# === Build Python wheel ===
cd ${CTRANSLATE_SOURCE}/python
pip install -r install_requirements.txt
python3 setup.py bdist_wheel --dist-dir /tmp

# === Install locally ===
pip install --force-reinstall /tmp/ctranslate2*.whl
pip install "numpy==1.24.4"
