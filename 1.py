import json
with open('/root/workspace/AdelaiDet/datasets/coco/annotations/instances_train2017.json', 'r') as f:
    data = json.load(f)
for ann in data['annotations']:
    if ann['id'] == 0:
        print(f"Annotation 0: {ann['segmentation']}")
        break