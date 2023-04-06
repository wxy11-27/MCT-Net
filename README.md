# MCT-NET: Multi-hierarchical cross transformer for hyperspectral and multispectral image fusion

This is the official code of MCT-Net: Multi-hierarchical cross transformer for hyperspectral and multispectral image fusion. Knowledge-Based Systems, 2023, 264: 110362.

## Network Architecture
<div align=center><img src="/figure/CiT_Net.jpg" width="80%" height="80%">

## 1. Create Envirement:
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

## 2. Data Preparation:

- Download  Datasets([Baidu Disk](https://pan.baidu.com/s/19vNPVhGJD9_btrrUwZ1fAQ), code: `MCT1`)

- Place the Datasets to `/MCT-Net/data/`.


- Then this repo is collected as the following form:

  ```shell
  |--MCT-Net
      |--data  
              |--Pavia.mat
              |--PaviaU.mat
              |--Botswana.mat
              |--Urban.mat
              |--Washington_DC.mat	
  ```
  
 ## 3.Training 
 python main.py
## 4.Testing 
 (1). Download the pretrained model zoo from ([Baidu Disk](https://pan.baidu.com/s/1OzHaWp9K4wyLr_fgXcCDzw), code: `MCT1`) and place them to `/MCT/checkpoints/`. 
 
 (2). python test.py

## BibTeX
```
@article{wang2023mct,
  title={MCT-Net: Multi-hierarchical cross transformer for hyperspectral and multispectral image fusion},
  author={Wang, Xianghai and Wang, Xinying and Song, Ruoxi and Zhao, Xiaoyang and Zhao, Keyun},
  journal={Knowledge-Based Systems},
  volume={264},
  pages={110362},
  year={2023},
  publisher={Elsevier}
}
```

