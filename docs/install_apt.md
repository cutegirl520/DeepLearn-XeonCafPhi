---
title: Installation: Ubuntu
---

# Ubuntu Installation

**General dependencies**

    sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev

**CUDA**: Install via the NVIDIA package instead of `apt-get` to be certain of the library and driver versions.
Install the library and latest driver separately; the driver bundled with the library is usually out-of-date.
This can be skipped for CPU-only installation.

**BLAS**: install ATLAS by `sudo apt-get install libatlas-base-dev` or install OpenBLAS or MKL for better CPU performance.

**Python** (optional): if you use the default Python you will need to `sudo apt-get install` the `python-dev` package to have the Pytho