# BlendMask Training Guide / BlendMask 训练指南

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English

### Prerequisites

1. **Install Dependencies**

   Create a conda environment and install required packages:
   ```bash
   conda create -n adet_env python=3.9
   conda activate adet_env
   pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
   ```

   Install AdelaiDet:
   ```bash
   python setup.py build develop
   ```

2. **Prepare Dataset**

   The dataset should be organized as follows:
   ```
   datasetsss/coco/
   ├── annotations/
   │   ├── instances_train2017.json
   │   └── instances_val2017.json
   ├── train2017/        # Training images (.jpg format)
   ├── val2017/          # Validation images (can be symlink to train2017)
   └── thing_train2017/  # Semantic segmentation masks (.npz format)
   ```

### Training

Train BlendMask model with ResNet-50 backbone:
```bash
python tools/train_net.py \
    --config-file configs/BlendMask/R_50_1x.yaml \
    --num-gpus 1 \
    OUTPUT_DIR training_dir/blendmask_R_50_1x
```

**Training Parameters:**
- `--config-file`: Path to the configuration file
- `--num-gpus`: Number of GPUs to use (adjust based on your hardware)
- `OUTPUT_DIR`: Directory to save training outputs and checkpoints

**Available Configurations:**
- `R_50_1x.yaml`: ResNet-50, 1x training schedule (~12 epochs)
- `R_50_3x.yaml`: ResNet-50, 3x training schedule (~36 epochs)
- `R_101_3x.yaml`: ResNet-101, 3x training schedule
- `R_101_dcni3_5x.yaml`: ResNet-101 with deformable convolutions, 5x schedule

### Evaluation

Evaluate a trained model:
```bash
python tools/train_net.py \
    --config-file configs/BlendMask/R_50_1x.yaml \
    --eval-only \
    --num-gpus 1 \
    OUTPUT_DIR training_dir/blendmask_R_50_1x \
    MODEL.WEIGHTS training_dir/blendmask_R_50_1x/model_final.pth
```

### Inference

Run inference on a single image:
```bash
python demo/demo.py \
    --config-file configs/BlendMask/R_50_1x.yaml \
    --input input/your_image.jpg \
    --output output \
    --confidence-threshold 0.3 \
    --opts MODEL.WEIGHTS R_50_1x.pth
```

### Configuration Customization

The training script `tools/train_net.py` contains custom configurations for the blast detection task:
- **Classes**: `["background", "blast"]` (2 classes)
- **Dataset**: `coco_my_train` and `coco_my_val`
- **Pretrained weights**: `R_50_1x.pth`

To modify these, edit the `setup()` function in `tools/train_net.py` (lines 226-244).

---

<a name="chinese"></a>
## 中文

### 环境准备

1. **安装依赖**

   创建 conda 环境并安装所需包：
   ```bash
   conda create -n adet_env python=3.9
   conda activate adet_env
   pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
   ```

   安装 AdelaiDet：
   ```bash
   python setup.py build develop
   ```

2. **准备数据集**

   数据集应按以下结构组织：
   ```
   datasetsss/coco/
   ├── annotations/
   │   ├── instances_train2017.json
   │   └── instances_val2017.json
   ├── train2017/        # 训练图像（.jpg 格式）
   ├── val2017/          # 验证图像（可以是 train2017 的符号链接）
   └── thing_train2017/  # 语义分割掩码（.npz 格式）
   ```

### 训练

使用 ResNet-50 骨干网络训练 BlendMask 模型：
```bash
python tools/train_net.py \
    --config-file configs/BlendMask/R_50_1x.yaml \
    --num-gpus 1 \
    OUTPUT_DIR training_dir/blendmask_R_50_1x
```

**训练参数：**
- `--config-file`：配置文件路径
- `--num-gpus`：使用的 GPU 数量（根据您的硬件调整）
- `OUTPUT_DIR`：保存训练输出和检查点的目录

**可用配置：**
- `R_50_1x.yaml`：ResNet-50，1x 训练计划（约12个周期）
- `R_50_3x.yaml`：ResNet-50，3x 训练计划（约36个周期）
- `R_101_3x.yaml`：ResNet-101，3x 训练计划
- `R_101_dcni3_5x.yaml`：带可变形卷积的 ResNet-101，5x 计划

### 评估

评估训练好的模型：
```bash
python tools/train_net.py \
    --config-file configs/BlendMask/R_50_1x.yaml \
    --eval-only \
    --num-gpus 1 \
    OUTPUT_DIR training_dir/blendmask_R_50_1x \
    MODEL.WEIGHTS training_dir/blendmask_R_50_1x/model_final.pth
```

### 推理

在单张图像上运行推理：
```bash
python demo/demo.py \
    --config-file configs/BlendMask/R_50_1x.yaml \
    --input input/your_image.jpg \
    --output output \
    --confidence-threshold 0.3 \
    --opts MODEL.WEIGHTS R_50_1x.pth
```

### 配置自定义

训练脚本 `tools/train_net.py` 包含用于爆炸检测任务的自定义配置：
- **类别**：`["background", "blast"]`（2个类别）
- **数据集**：`coco_my_train` 和 `coco_my_val`
- **预训练权重**：`R_50_1x.pth`

要修改这些配置，请编辑 `tools/train_net.py` 中的 `setup()` 函数（第226-244行）。

---

## Troubleshooting / 故障排除

### Common Issues / 常见问题

1. **ModuleNotFoundError: No module named 'detectron2'**
   - Solution: Make sure you've activated the conda environment and installed detectron2
   - 解决方案：确保已激活 conda 环境并安装了 detectron2

2. **CUDA out of memory**
   - Solution: Reduce batch size in the config file or use fewer GPUs
   - 解决方案：在配置文件中减小批次大小或使用更少的 GPU

3. **FileNotFoundError: Image file not found**
   - Solution: Check that images are in .jpg format and annotation paths are correct
   - 解决方案：检查图像是否为 .jpg 格式，注释路径是否正确

## Model Zoo

Pre-trained weights are available:
- `R_50_1x.pth`: ResNet-50, 1x schedule
- `R_50_3x.pth`: ResNet-50, 3x schedule  
- `R_101_3x.pth`: ResNet-101, 3x schedule

These can be used as starting points for training or for inference.

可用的预训练权重：
- `R_50_1x.pth`：ResNet-50，1x 训练计划
- `R_50_3x.pth`：ResNet-50，3x 训练计划
- `R_101_3x.pth`：ResNet-101，3x 训练计划

这些权重可用作训练的起点或用于推理。
