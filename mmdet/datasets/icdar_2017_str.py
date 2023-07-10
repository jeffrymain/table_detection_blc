from asyncio.windows_events import NULL
import os
from re import M
from unittest import result
from matplotlib import image
import numpy as np
import torch
from PIL import Image
import cv2
from .coco import CocoDataset
import tempfile
import os.path as osp
from .builder import DATASETS
import mmcv

from mmcv.utils import print_log


@DATASETS.register_module()
class ICDAR_2017_STR_Dataset(CocoDataset):
    CLASSES = ('cell')

