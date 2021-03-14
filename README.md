# Separating printing Chinese characters from overlapping handwriting using GAN

This is the official implementation of the "Separating printing Chinese characters from overlapping handwriting using GAN" paper.

## Dependencies and Installation

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download))
- [PyTorch >= 1.1](https://pytorch.org)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- [Deformable Convolution](https://arxiv.org/abs/1703.06211). We use [mmdetection](https://github.com/open-mmlab/mmdetection)'s dcn implementation. Please first compile it.
  
  ```
  cd ./codes/models/archs/dcn
  python setup.py develop
  ```
- Python packages: `pip install -r requirements.txt`

## Training and Testing
Modify the configuration files under `./codes/options/train` and `./codes/options/test`, then train or test model using:

```python
python ./codes/train.py -opt ./codes/options/train/train_DESRGAN.yml
python ./codes/test.py -opt ./codes/options/test/test_DESRGAN.yml
```

## License
This project is released under the Apache 2.0 license.

