<div align="center">
<h1> Adaptive Bidirectional Displacement for Semi-Supervised Medical Image Segmentation (CVPR 2024) </h1>
<a href='https://arxiv.org/pdf/2405.00378'><img src='https://img.shields.io/badge/Project-Paper-green'></a> 
<a href='https://github.com/chy-upc/ABD'><img src='https://img.shields.io/badge/Technique-Code-red'></a> 
</div>

<b>by Hanyang Chi, Jian Pang, Bingfeng Zhang, and Weifeng Liu.</b>
![image](framework.png)
Consistency learning is a central strategy to tackle unlabeled data in semi-supervised medical image segmentation (SSMIS), which enforces the model to produce consistent
predictions under the perturbation. However, most current approaches solely focus on utilizing a specific single perturbation, which can only cope with limited cases, while
employing multiple perturbations simultaneously is hard to guarantee the quality of consistency learning. In this paper, we propose an Adaptive Bidirectional Displacement (ABD) approach to solve the above challenge.

## 1. Installation
```bash
git clone https://github.com/chy-upc/ABD.git
```
This repository is based on PyTorch 1.11.0, CUDA 11.3 and Python 3.7.13. All experiments in our paper were conducted on NVIDIA GeForce RTX 3090 GPU with an identical experimental setting.
```
conda create -n ABD python=3.7.13
conda activate ABD
pip install -r requirements.txt
```
## 2. Dataset
Data could be got at [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC) and [promise12](https://promise12.grand-challenge.org/Download/).
For the PROMISE12 dataset, we also provide the pre-processed version at [Google Drive](https://drive.google.com/file/d/1KRzKemoFYxQN26d2eZf3wkm6_8Zf84JD/view?usp=drive_link).
```
├── ./data
    ├── [ACDC]
        ├── [data]
        ├── test.list
        ├── train_slices.list
        ├── train.list
        └── val.list
    └── [promise12]
        ├── CaseXX_segmentation.mhd
        ├── CaseXX_segmentation.raw
        ├── CaseXX.mhd
        ├── CaseXX.raw
        ├── test.list
        └── val.list
```
## 3. Pretrained Backbone
Download pre-trained [Swin-Unet](https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY) model to "./code/pretrained_ckpt" folder.
```
├── ./code/pretrained_ckpt
    └── swin_tiny_patch4_window7_224.pth
```
## 4. Usage
To train a model,
```
python ./code/train_ACDC_Cross_Teaching.py  # for ACDC training —— *Ours*-ABD (Cross Teaching) 
python ./code/train_ACDC_BCP.py  # for ACDC training —— *Ours*-ABD (BCP) 
python ./code/train_PROMISE12.py  # for PROMISE12 training
``` 
To test a model,
```
python ./code/test_ACDC.py  # for ACDC testing
python ./code/test_PROMISE12.py  # for PROMISE12 testing
```
## Citation
If you find these projects useful, please consider citing:
```bibtex
@inproceedings{chi2024adaptive,
  title={Adaptive Bidirectional Displacement for Semi-Supervised Medical Image Segmentation},
  author={Chi, Hanyang and Pang, Jian and Zhang, Bingfeng and Liu, Weifeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4070--4080},
  year={2024}
}
```
## Acknowledgements
Our code is largely based on [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [BCP](https://github.com/DeepMed-Lab-ECNU/BCP), and [SCP-Net](https://arxiv.org/pdf/2305.16214.pdf). Thanks for these authors for their valuable work, hope our work can also contribute to related research.
## Questions
If you have any questions, welcome contact me at 'chihanyang@s.upc.edu.cn'.
