# OBSS SAHI Tool
# Code written by TUOLONG, 2023.
import unittest

import colorsys
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont, Image

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox, DecodeBoxNP

'''
训练自己的数据集必看注释！
'''


import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import check_requirements

class Yolov7DetectionModel(DetectionModel):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "model_path": '/home/zhewang/Project/yolov7-pytorch/logs/last_epoch_weights.pth',
        "classes_path": '/home/zhewang/Project/yolov7-pytorch/model_data/voc_classes.txt',
        # ---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        # ---------------------------------------------------------------------#
        "anchors_path": '/home/zhewang/Project/yolov7-pytorch/model_data/yolo_anchors.txt',
        "anchors_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        # ---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        # ---------------------------------------------------------------------#
        "input_shape": [640, 640],
        # ------------------------------------------------------#
        #   所使用到的yolov7的版本，本仓库一共提供两个：
        #   l : 对应yolov7
        #   x : 对应yolov7_x
        # ------------------------------------------------------#
        "phi": 'l',
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.001,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.5,
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": True,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,
    }
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

        # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)
    def generate(self, onnx=False):
        #---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        #---------------------------------------------------#
        self.net    = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()




    def check_dependencies(self) -> None:
        check_requirements(["ultralytics"])

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """
        try:
            self.net = YoloBody(self.anchors_mask, self.num_classes, self.phi)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.net.load_state_dict(torch.load(self.model_path, map_location=device))
            self.net = self.net.fuse().eval()

        except Exception as e:
            raise TypeError("model_path is not a valid yolov7 model path: ", e)

    def set_model(self, model: Any):
        """
        Sets the underlying YOLOv8 model.
        Args:
            model: Any
                A YOLOv8 model
        """

        self.model = model

        # set category_mapping
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping

    def perform_inference(self, image: np.ndarray):
        # ---------------------------------------------------#
        #   将np.ndarray 转换成 PIL
        # ---------------------------------------------------#
        image = Image.fromarray(image)
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        #   h, w, 3 => 3, h, w => 1, 3, h, w
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)


        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """
        # Confirm model is loaded
        if self.net is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)

        self._original_predictions = results

    @property
    def category_names(self):
        return self.class_names

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return self.num_classes

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        return False  # fix when yolov5 supports segmentation models

    def _create_object_prediction_list_from_original_predictions(
            self,
            shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
            full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
        object_prediction_list_per_image = []
        for image_ind, image_predictions_in_xyxy_format in enumerate(original_predictions):
            if image_predictions_in_xyxy_format is None:
                # Add your handling code here, e.g., print a message or skip processing for this image
                print(f"No predictions for image {image_ind}, skipping...")
                continue

            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            # process predictions
            for prediction in image_predictions_in_xyxy_format:
                x1 = prediction[1] #left
                y1 = prediction[0] #top
                x2 = prediction[3] #right
                y2 = prediction[2] #bottom
                bbox = [x1, y1, x2, y2]
                score = prediction[4] * prediction[5]
                category_id = int(prediction[6])
                category_name = self.class_names[category_id]

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image


