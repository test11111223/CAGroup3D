# Running CAGroup3D in Winodws 10 #

- Testing environment: X299 + i7-7820X + Win10 22H2 + RTX 2080 Ti + 72GB DDR4 + 480GB SATA SSD
- ~~WSL2 is not installed~~

## Objective ##

- Train with **joint dataset** (ScanNetV2 + Sun RGB-D) and examine the result against both tasks.
- Not focused on reproduce the data (obviously different CUDA version will produce different results)
- ~~Some live demo~~

## Before cloning this repo ##

- Newest GPU driver. CUDA version in this repo will be 11.7. Use `nvidia-smi` to check.

- Prepare at least 50GB of disk space! 

- Install [CUDA Toolkit 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive)

- Install anaconda. [Miniconda](https://docs.conda.io/en/latest/miniconda.html) would be more flexable.

- Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/zh-hant/visual-cpp-build-tools/)

- [Ref CSDN](https://blog.csdn.net/m0_37890541/article/details/107723861) [Ref stackoverflow](https://stackoverflow.com/questions/70013/how-to-detect-if-im-compiling-code-with-a-particular-visual-studio-version) Modify `host_config.h`: `_MSC_VER >= 2000` 

- Prepare a python environment (python 3.10 + pytorch cu117 + spconv cu117) ~~copy manually in cmd~~:

- [If there is some strange SSL error](https://github.com/conda/conda/issues/11795#issuecomment-1340010125)

```sh
# python=3.11 will crash in application!
conda create -n cagroup3d-env -c conda-forge scikit-learn python=3.10
conda activate cagroup3d-env

# Gamble on cu117 (nvidia-smi shows GTX 2080Ti + CUDA 12.1), as pytorch has cu117 also
pip install spconv-cu117

# Yea, need torch.
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
```

- Now it is good to clone.

- (Optional) [VSCode](https://code.visualstudio.com/) has terminal which is not easily interrupted and [notepad++](https://notepad-plus-plus.org/downloads/) for non ascii display.

## After cloning this repo ##

- First install some build tools:

```sh
pip install ninja open3d
```

- ~~Endless CPP debugging~~:

```sh
cd CAGroup3D
python setup.py develop > logs/t.txt
```

## Rants ##

- [error C2131 on EPS](https://github.com/open-mmlab/OpenPCDet/pull/1040)
- [error C2131: expression did not evaluate to a constant](https://github.com/open-mmlab/OpenPCDet/issues/681#issuecomment-980000235)
- [Still C2131:](https://blog.csdn.net/qq_39027296/article/details/104936998)
- ['uint32_t' does not name a type](https://stackoverflow.com/questions/11069108/uint32-t-does-not-name-a-type): `#include <cstdint>`, and check `inline int check_rect_cross` in `iou3d_nms_kernel.cu`
- [THC/THC.h: No such file or directory](https://discuss.pytorch.org/t/question-about-thc-thc-h/147145/8). [Use ATen instead](https://github.com/sshaoshuai/Pointnet2.PyTorch/issues/34)
- **TODO** ["sys/mman.h": No such file or directory](https://github.com/open-mmlab/OpenPCDet/issues/1043) [Install gygwin](https://www.cs.odu.edu/~zeil/FAQs/Public/vscodeWithCygwin/) **with additional packages**: `gcc-core gcc-debuginfo gcc-objc gcc-g++ gdb make`