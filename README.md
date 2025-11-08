Detectron2系列: 
运行：python tools/train_net.py --config-file configs/BlendMask/R_50_1x.yaml --num-gpus 1 OUTPUT_DIR training_dir/blendmask_R_50_1x
测试：python demo/demo.py --config-file configs/BlendMask/R_50_1x.yaml --input input/90.jpg --output output --confidence-threshold 0.3 --opts MODEL.WEIGHTS R_50_1x.pth