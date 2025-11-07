import copy
import os
import logging
import os.path as osp

import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image
from pycocotools import mask as maskUtils

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import SizeMismatchError
from detectron2.structures import BoxMode

from .augmentation import RandomCropWithInstance
from .detection_utils import (annotations_to_instances, build_augmentation,
                              transform_instance_annotations)

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithBasis"]

logger = logging.getLogger(__name__)


def segmToRLE(segm, img_size):
    h, w = img_size
    if type(segm) == list:
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm["counts"]) == list:
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        rle = segm
    return rle


def segmToMask(segm, img_size):
    rle = segmToRLE(segm, img_size)
    m = maskUtils.decode(rle)
    return m


class DatasetMapperWithBasis(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30,30],sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

    def __call__(self, dataset_dict):
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        dataset_dict = copy.deepcopy(dataset_dict)

        # -----------------------------
        # 🔧 修复图像路径问题
        # -----------------------------
        file_name = dataset_dict["file_name"].replace("\\", "/")  # Windows路径统一为/
        file_name = osp.normpath(file_name)
        try:
            image = utils.read_image(file_name, format=self.image_format)
        except Exception as e:
            print(f"[Error] Failed to read image: {dataset_dict['file_name']}")
            print(e)
            raise e

        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # -----------------------------
        # 🔧 修复 basis_sem 路径问题
        # -----------------------------
        if self.basis_loss_on and self.is_train:
            if self.ann_set == "coco":
                basis_sem_path = dataset_dict["file_name"].replace("train", "thing_train")
            else:
                basis_sem_path = dataset_dict["file_name"].replace("coco", "lvis").replace("train2017", "thing_train2017")

            # 修复路径分隔符并生成 npz 路径
            basis_sem_path = basis_sem_path.replace("\\", "/")
            basis_sem_path = osp.normpath(osp.splitext(basis_sem_path)[0] + ".npz")

            try:
                basis_sem_gt = np.load(basis_sem_path)["mask"]
            except Exception as e:
                print(f"[Error] Failed to load basis_sem file: {basis_sem_path}")
                print(e)
                raise e

            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt

        return dataset_dict
