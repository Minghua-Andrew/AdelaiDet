# AdelaiDet - BlendMask for Blast Detection

## Quick Start / 快速开始

### Training / 训练
```bash
python tools/train_net.py \
    --config-file configs/BlendMask/R_50_1x.yaml \
    --num-gpus 1 \
    OUTPUT_DIR training_dir/blendmask_R_50_1x
```

### Inference / 推理
```bash
python demo/demo.py \
    --config-file configs/BlendMask/R_50_1x.yaml \
    --input input/90.jpg \
    --output output \
    --confidence-threshold 0.3 \
    --opts MODEL.WEIGHTS R_50_1x.pth
```

**For detailed training instructions, see [TRAINING.md](TRAINING.md)**  
**详细的训练说明请参阅 [TRAINING.md](TRAINING.md)**