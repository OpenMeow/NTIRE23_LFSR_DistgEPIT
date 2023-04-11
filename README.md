# DistgEPIT: Enhanced Disparity Learning for Light Field Image Super-Resolution

**Abstract:** Light Field (LF) cameras capture rich information in 4D LF images by recording both intensity and angular directions, making it crucial to learn the inherent spatial-angular correlation in low-resolution (LR) images for superior results. Despite impressive progress made by several CNN-based deep methods and pioneering Transformer-based methods for LF image super resolution (SR), most of them fail to fully leverage the LF spatial-angular correlation and tend to perform poorly in scenes with varying disparities. In this paper, we propose a hybrid method called DistgEPIT that implements an enhanced disparity learning mechanism with both convolution-based and transformer-based modules. It enables the capture of angular correlation, refinement of adjacent disparities, and extraction of essential spatial features. Additionally, we introduce a Position-Sensitive Windowing (PSW) strategy to maintain consistency of disparity between the training and inference stages, which yields an average PSNR gain of 0.2 dB by replacing the traditional padding and windowing method. Extensive experiments with ablation studies demonstrate the effectiveness of our proposed method, which ranked 1st place in the NITRE2023 LF image SR challenge.

## News

✅ **Mar 24, 2023:** This repository contains official pytorch implementation of "DistgEPIT: Enhanced Disparity Learning for Light Field Image Super-Resolution" in **1st solutions 👑** in [NTIRE2023 Light-Field Super Resolution Track](https://codalab.lisn.upsaclay.fr/competitions/9201) .

✅ **Apr 11, 2023:** This method has accepted in [NTIRE 2023 Workshop](https://cvlai.net/ntire/2023/), PDF click here 👋.

## Code

### Dependencies

It is recommended to use a **Python 3.8** or above version.

```
pip install einops opencv-python numpy scikit-image h5py imageio mat73 scipy
pip install torch torchvision
```

### Prepare Dataset

Download datasets and prepare file structure refers to [**BasicLFSR**](https://github.com/ZhengyuLiang24/BasicLFSR).
- **Central Selection** located in `Generate_Data_for_Training.py:67`
- **Interval Selection** located in `Generate_Data_for_Training.py:68`
- **Uneven Selection** located in `Generate_Data_for_Training.py:69`

```bash
# default use patch size 32 * 4 and stride 32
python Generate_Data_for_Training.py --angRes 5 --scale_factor 4
python Generate_Data_for_Test.py --angRes 5 --scale_factor 4
python Generate_Data_for_inference.py --angRes 5 --scale_factor 4
```

### Training & Test

✅ **Postion-Senstive Windowing (PSW)** opeartion located in `lfsr.py:inference_no_pad`

```bash
# training model
./lfsr train

# validation set
./lfsr val

# validation set with tta
./lfsr val_tta

# ntire2023 validation
./lfsr test_val

# ntire2023 test
./lfsr test
```

### Pretrained Model in NTIRE2023

Pretrained model could be download here [**click**](https://drive.google.com/file/d/1xTLmxR5RO_VtN8f_XerqEAmEsjuvNTY4/view?usp=share_link), download it and put them into `./checkpoints`

Below results by using command **`./lfsr.sh val_tta`**

```bash
MODEL=DistgEPITv6
MODEL_SOURCE=vanilla
TEST_PATCH=32
TEST_STRIDE=8
# mean_psnr: 33.2259, mean_ssim: 0.9494, real_psnr: 31.9943, synth_psnr: 35.0733
MODEL_PATH=checkpoints/Exp74.DistgEPITv6.B4.L1e-4.E100.P128_32mp.S5.EMA0.999.pth
# mean_psnr: 33.2172, mean_ssim: 0.9493, real_psnr: 31.9789, synth_psnr: 35.0746
MODEL_PATH=checkpoints/Exp80.DistgEPITv6.B4.L5e-5.E100.P128_32mp.S5.EMA0.999.FT.pth
# mean_psnr: 33.2334, mean_ssim: 0.9495, real_psnr: 31.9986, synth_psnr: 35.0856
MODEL_PATH=checkpoints/Exp90.DistgEPITv6.B4.L5e-5.E100.P128_32mp.S5.EMA0.999.FT2.pth
# mean_psnr: 33.2061, mean_ssim: 0.9492, real_psnr: 31.9707, synth_psnr: 35.0593
MODEL_PATH=checkpoints/Exp97.DistgEPITv6.B4.L1e-4.E100.P128_32mpp.S5.EMA0.999.FT.pth

MODEL=DistgEPIT_wider
MODEL_SOURCE=vanilla
TEST_PATCH=32
TEST_STRIDE=8
# mean_psnr: 33.2129, mean_ssim: 0.9492, real_psnr: 31.9880, synth_psnr: 35.0502
MODEL_PATH=checkpoints/Exp78.DistgEPIT_wider.B4.L1e-4.E100.P128_32mp.S6.EMA0.999.FT.pth
# mean_psnr: 33.2346, mean_ssim: 0.9493, real_psnr: 32.0087, synth_psnr: 35.0735
MODEL_PATH=checkpoints/Exp87.DistgEPIT_wider.B4.L1e-4.E100.P128_32mpp.S6.EMA0.999.FT.pth
# mean_psnr: 33.2287, mean_ssim: 0.9494, real_psnr: 32.0036, synth_psnr: 35.0664
MODEL_PATH=checkpoints/Exp91.DistgEPIT_wider.B4.L5e-5.E100.P128_32mp.S6.EMA0.999.FTT.pth
# 🌟🌟 mean_psnr: 33.2494, mean_ssim: 0.9495, real_psnr: 32.0270, synth_psnr: 35.0832
MODEL_PATH=checkpoints/Exp92.DistgEPIT_wider.B4.L5e-5.E100.P128_32mpp.S6.EMA0.999.FTT.pth

MODEL=DistgEPIT_deeper
MODEL_SOURCE=vanilla
TEST_PATCH=32
TEST_STRIDE=8
# mean_psnr: 33.2171, mean_ssim: 0.9492, real_psnr: 31.9998, synth_psnr: 35.0432
MODEL_PATH=checkpoints/Exp93.DistgEPIT_deeper.B4.L5e-5.E100.P128_32mp.S6.EMA.0.999.FT.pth

MODEL=DistgSSR
MODEL_SOURCE=wzq
TEST_PATCH=84
TEST_STRIDE42
MODEL_PATH=checkpoints/DistgSSR_32x8_6x6x128_finetune.pth
MODEL_PATH=checkpoints/DistgSSR_32x8_6x6x128.pth
MODEL_PATH=checkpoints/DistgSSR_32x8_6x6x64_finetune.pth
```

## Citation

If you find this work helpful, please consider citing the following papers:

```bibtex
@InProceedings{DistgEPIT,
    author    = {Jin, Kai and Yang, Angulia and Wei, Zeqiang and Guo, Sha and Gao, Mingzhi and Zhou, Xiuzhuang},
    title     = {DistgEPIT: Enhanced Disparity Learning for Light Field Image Super-Resolution},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
    year      = {2023},
}
```

```bibtex
@InProceedings{BasicLFSR,
  author    = {Wang, Yingqian and Wang, Longguang and Liang, Zhengyu and Yang, Jungang and Timofte, Radu and Guo, Yulan and Jin, Kai and Wei, Zeqiang and Yang, Angulia and Guo, Sha and Gao, Mingzhi and Zhou, Xiuzhuang and Duong, Vinh Van and Huu, Thuc Nguyen and Yim, Jonghoon and Jeon, Byeungwoo and Liu, Yutong and Cheng, Zhen and Xiao, Zeyu and Xu, Ruikang and Xiong, Zhiwei and Liu, Gaosheng and Jin, Manchang and Yue, Huanjing and Yang, Jingyu and Gao, Chen and Zhang, Shuo and Chang, Song and Lin, Youfang and Chao, Wentao and Wang, Xuechun and Wang, Guanghui and Duan, Fuqing and Xia, Wang and Wang, Yan and Xia, Peiqi and Wang, Shunzhou and Lu, Yao and Cong, Ruixuan and Sheng, Hao and Yang, Da and Chen, Rongshan and Wang, Sizhe and Cui, Zhenglong and Chen, Yilei and Lu, Yongjie and Cai, Dongjun and An, Ping and Salem, Ahmed and Ibrahem, Hatem and Yagoub, Bilel and Kang, Hyun-Soo and Zeng, Zekai and Wu, Heng},
  title     = {NTIRE 2023 Challenge on Light Field Image Super-Resolution: Dataset, Methods and Results},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year      = {2023},
}
```

## Contact

Welcome to send email to jinkai@bigo.sg, if you have any questions about this repository or other issues.
