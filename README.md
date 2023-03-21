# NTIRE23_LFSR_DistgEPIT

### requirements

```bash
pip install einops
pip install opencv-python
pip install numpy
pip install scikit-image
pip install h5py
pip install imageio
pip install mat73
pip install scipy
pip install torch
pip install einops
```

### command
```run
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

### configuration
```
MODEL=DistgEPITv6
MODEL_SOURCE=vanilla
TEST_PATCH=32
TEST_STRIDE=8
MODEL_PATH=checkpoints/Exp74.DistgEPITv6.B4.L1e-4.E100.P128_32mp.S5.EMA0.999.pth
MODEL_PATH=checkpoints/Exp80.DistgEPITv6.B4.L5e-5.E100.P128_32mp.S5.EMA0.999.FT.pth
MODEL_PATH=checkpoints/Exp90.DistgEPITv6.B4.L5e-5.E100.P128_32mp.S5.EMA0.999.FT2.pth
MODEL_PATH=checkpoints/Exp97.DistgEPITv6.B4.L1e-4.E100.P128_32mpp.S5.EMA0.999.FT.pth

MODEL=DistgEPIT_wider
MODEL_SOURCE=vanilla
TEST_PATCH=32
TEST_STRIDE=8
MODEL_PATH=checkpoints/Exp78.DistgEPIT_wider.B4.L1e-4.E100.P128_32mp.S6.EMA0.999.FT.pth
MODEL_PATH=checkpoints/Exp87.DistgEPIT_wider.B4.L1e-4.E100.P128_32mpp.S6.EMA0.999.FT.pth
MODEL_PATH=checkpoints/Exp91.DistgEPIT_wider.B4.L5e-5.E100.P128_32mp.S6.EMA0.999.FTT.pth
MODEL_PATH=checkpoints/Exp92.DistgEPIT_wider.B4.L5e-5.E100.P128_32mpp.S6.EMA0.999.FTT.pth

MODEL=DistgEPIT_deeper
MODEL_SOURCE=vanilla
TEST_PATCH=32
TEST_STRIDE=8
MODEL_PATH=checkpoints/Exp93.DistgEPIT_deeper.B4.L5e-5.E100.P128_32mp.S6.EMA.0.999.FT.pth

MODEL=DistgSSR
MODEL_SOURCE=wzq
TEST_PATCH=84
TEST_STRIDE42
MODEL_PATH=checkpoints/DistgSSR_32x8_6x6x128_finetune.pth
MODEL_PATH=checkpoints/DistgSSR_32x8_6x6x128.pth
MODEL_PATH=checkpoints/DistgSSR_32x8_6x6x64_finetune.pth
```

### training method

1. DistgSSR_32x8_6x6x64_fintune.pth
   1) batch size = 20, lr = 1e-3, 训练10周期
   2) batch size = 20, lr = 5e-4, 训练15周期
   3) batch size = 20, lr = 2.5e-4, 训练15周期
   4) 加入 rgb shuffle 和 13579 视图, batch size = 20, lr = 1.25e-4, 训练15周期
   在第 **iv** 步内选择验证集上最好的结果

2. DistgSSR_32x8_6x6x128.pth
   1) batch size = 10, lr = 5e-4, 训练15周期
   2) batch size = 10, lr = 2.5e-4, 训练15周期
   在第 **ii** 步内选择验证集上最好的结果

3. DistgSSR_32x8_6x6x128_finetune.pth
   1) 基于 DistgSSR_32x8_6x6x128.pth 模型进行finetune
   2) 加入 rgb shuffle 和 13579 视图, batch size = 10, lr = 1.25e-4, 训练15周期
   在第 **ii** 步内选择验证集上最好的结果

4. DistgEPIT series
   1) 80个epoch 2e-4, 20个epoch 1e-4
   2) 加入 13579 视图训练 20个epoch 1e-4
   在第 **ii** 步内选择验证集上最好的结果
