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
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = cfg.MODEL.NUM_CLASSES
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
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
        if self._split == 'train':
            path_to_file = cfg.VIDOR.TRAIN_FRAME_LIST 
        else:
            path_to_file = cfg.VIDOR.VAL_FRAME_LIST
        assert os.path.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )
        vidor_data = []
        with open(path_to_file, "r") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                # The format of each row should follow:
                # frame_loc,box[0],box[1],box[2],box[3],label,clip_index]
                assert len(row) == 7
                frame_loc = os.path.join(cfg.VIDOR.FRAME_PATH, row[0])
                boxes = row[1:5]
                label = row[5]
                vidor_data.append((frame_loc ,boxes,label))
            sample_rate = self._sample_rate
            return [vidor_data[i:i + sample_rate] for i in range(0, len(vidor_data), sample_rate)]

    def print_summary(self):
        print_log("=== VidOR dataset summary ===")
        print_log("Split: {}".format(self._split))
        print_log(f"Number of classes: {self._num_classes}")
        print_log(f"Number of clips: {len(self.data)}")
        total_frames = sum([len(clip) for clip in self.data])
        print_log(f"Number of frames:{total_frames}")

    def __len__(self):
        return len(self.data)

    def _images_and_boxes_preprocessing_cv2(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """

        height, width, _ = imgs[0].shape
        # `transform.py` is list of np.array. However, for AVA, we only have
        # one np.array.
        boxes = [boxes[i,:].reshape(1,4) for i in range(boxes.shape[0])]

        # The image now is in HWC, BGR format.
        if self._split == "train":  # "train"
            imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = cv2_transform.random_crop_list(
                imgs, self._crop_size, order="HWC", boxes=boxes
            )

            # random flip
            imgs, boxes = cv2_transform.horizontal_flip_list(
                0.5, imgs, order="HWC", boxes=boxes
            )
        elif self._split == "val":
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(
                    self._crop_size, boxes[0], height, width
                )
            ]
            imgs, boxes = cv2_transform.spatial_shift_crop_list(
                self._crop_size, imgs, 1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "test":
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(
                    self._crop_size, boxes[0], height, width
                )
            ]
            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        else:
            raise NotImplementedError(
                "Unsupported split mode {}".format(self._split)
            )

        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        boxes = cv2_transform.clip_boxes_to_image(
            boxes[0], imgs[0].shape[1], imgs[0].shape[2]
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
        clip_data = self.data[idx] 
        # Get boxes and labels for current clip.
        boxes = []
        labels = []
        image_paths = []
        for row in clip_data:
            image_path,box,label = row
            labels.append(label)
            boxes.append(list(map(int, box)))
            image_paths.append(image_path)
        # Load images of current clip.
        imgs = utils.retry_load_images(
            image_paths, backend="cv2"
        )
        boxes = np.array(boxes)
        ori_boxes = boxes.copy()
        # Preprocess images and boxes
        imgs, boxes = self._images_and_boxes_preprocessing_cv2(
            imgs, boxes=boxes
        )

        # Construct label arrays.
        # label_arrs = np.zeros((len(labels), self._num_classes), dtype=np.int32)
        # for i, label in enumerate(labels):
        #     label_arrs[i][int(label)] = 1
        # label_arr = np.eye(self._num_classes)[values]

        imgs = utils.pack_pathway_output(self.cfg, imgs)

        extra_data = {
            "boxes": boxes,
            "ori_boxes": ori_boxes,
        }

        return imgs, int(labels[0]), idx, {}
