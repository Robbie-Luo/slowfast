#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import numpy as np
import torch
import os
from collections import defaultdict
from . import vidor_helper as vidor_helper
from . import cv2_transform as cv2_transform
from . import transform as transform
from . import utils as utils
from .build import DATASET_REGISTRY
import csv
logger = logging.getLogger(__name__)
def print_log(str):
    logger.info(str)
    print(str)

@DATASET_REGISTRY.register()
class Vidor(torch.utils.data.Dataset):
    """
    Vidor Dataset
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        self._num_classes = cfg.MODEL.NUM_CLASSES
        # Augmentation params.
        if self._split == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.AVA.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.AVA.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.AVA.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.AVA.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE
            self._test_force_flip = cfg.AVA.TEST_FORCE_FLIP

        self.data = self._load_data(cfg)
        self.print_summary()

    def _load_data(self, cfg):
        """
        Load frame paths and annotations from files

        Args:
            cfg (CfgNode): config
        """
        path_to_file = cfg.VIDOR.TRAIN_FRAME_LIST 
        assert os.path.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )
        vidor_data = defaultdict(list)
        self.video_name_to_idx = {}
        self.video_idx_to_name = []
        self.clip_keys = []
        with open(path_to_file, "r") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                # The format of each row should follow:
                # frame_loc,box[0],box[1],box[2],box[3],label,clip_index]
                assert len(row) == 7
                frame_loc = row[0]
                boxes = row[1:5]
                label = row[5]
                clip_key = row[5:]
                if clip_key not in self.clip_keys:
                    self.clip_keys.append(clip_key)
                vidor_data[str(clip_key)].append(
                    (os.path.join(cfg.VIDOR.FRAME_PATH, row[0]),boxes,label)
                )
        return vidor_data

    def print_summary(self):
        print_log("=== VidOR dataset summary ===")
        print_log("Split: {}".format(self._split))
        print_log(f"Number of classes: {self._num_classes}")
        print_log(f"Number of clips: {len(self.clip_keys)}")
        total_frames = sum([len(self.data[str(clip)]) for clip in self.clip_keys])
        print_log(f"Number of frames:{total_frames}")

    def __len__(self):
        return len(self.clip_keys)

    def _images_and_boxes_preprocessing(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs = imgs / 255.0

        height, width = imgs.shape[2], imgs.shape[3]
        # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
        # range of [0, 1].
        boxes = transform.clip_boxes_to_image(boxes, height, width)

        if self._split == "train":
            # Train split
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = transform.random_crop(
                imgs, self._crop_size, boxes=boxes
            )

            # Random flip.
            imgs, boxes = transform.horizontal_flip(0.5, imgs, boxes=boxes)
        elif self._split == "val":
            # Val split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            # Apply center crop for val split
            imgs, boxes = transform.uniform_crop(
                imgs, size=self._crop_size, spatial_idx=1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        elif self._split == "test":
            # Test split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        else:
            raise NotImplementedError(
                "{} split not supported yet!".format(self._split)
            )

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = transform.color_jitter(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = transform.lighting_jitter(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = transform.color_normalization(
            imgs,
            np.array(self._data_mean, dtype=np.float32),
            np.array(self._data_std, dtype=np.float32),
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            # Note that Kinetics pre-training uses RGB!
            imgs = imgs[:, [2, 1, 0], ...]

        boxes = transform.clip_boxes_to_image(
            boxes, self._crop_size, self._crop_size
        )

        return imgs, boxes

    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        clip_data = self.data[str(self.clip_keys[idx])] 
        # Get boxes and labels for current clip.
        boxes = []
        labels = []
        image_paths = []
        for row in clip_data:
            image_path,box,label = row
            boxes.append(box)
            labels.append(label)
            image_paths.append(image_path)
        # Load images of current clip.
        imgs = utils.retry_load_images(
            image_paths, backend="pytorch"
        )
        print(np.shape(imgs))
        boxes = np.array(boxes)
        ori_boxes = boxes.copy()
        
        # T H W C -> T C H W.
        imgs = imgs.permute(0, 3, 1, 2)
        # Preprocess images and boxes.
        # imgs, boxes = self._images_and_boxes_preprocessing(
        #     imgs, boxes=boxes
        # )
        # T C H W -> C T H W.
        imgs = imgs.permute(1, 0, 2, 3)

        # Construct label arrays.
        print(np.shape(labels))
        label_arrs = np.zeros((len(labels), self._num_classes), dtype=np.int32)
        for i, label in enumerate(labels):
            label_arrs[i][label] = 1

        imgs = utils.pack_pathway_output(self.cfg, imgs)

        extra_data = {
            "boxes": boxes,
            "ori_boxes": ori_boxes,
        }

        return imgs, label_arrs, idx, {}
