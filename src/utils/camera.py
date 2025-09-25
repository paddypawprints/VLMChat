import argparse
import multiprocessing
import queue
import sys
import threading
from functools import lru_cache
import os
from datetime import datetime
from PIL import Image

import numpy as np

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)


class Detection:
    def __init__(self, coords, category, conf, metadata, imx500, picam2):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


class IMX500ObjectDetection:
    def __init__(self, args=None):
        self.args = args if args else self.get_args()
        self.imx500 = IMX500(self.args.model)
        self.intrinsics = self.imx500.network_intrinsics
        if not self.intrinsics:
            self.intrinsics = NetworkIntrinsics()
            self.intrinsics.task = "object detection"
        elif self.intrinsics.task != "object detection":
            raise ValueError("Network is not an object detection task")

        # Override intrinsics from args
        for key, value in vars(self.args).items():
            if key == 'labels' and value is not None:
                with open(value, 'r') as f:
                    self.intrinsics.labels = f.read().splitlines()
            elif hasattr(self.intrinsics, key) and value is not None:
                setattr(self.intrinsics, key, value)

        # Defaults
        if self.intrinsics.labels is None:
            with open("assets/coco_labels.txt", "r") as f:
                self.intrinsics.labels = f.read().splitlines()
        self.intrinsics.update_with_defaults()

        self.picam2 = Picamera2(self.imx500.camera_num)
        main = {'format': 'RGB888'}
        self.config = self.picam2.create_preview_configuration(
            main,
            controls={"FrameRate": self.intrinsics.inference_rate},
            buffer_count=12
        )

        self.imx500.show_network_fw_progress_bar()
        self.picam2.start(self.config, show_preview=False)
        if self.intrinsics.preserve_aspect_ratio:
            self.imx500.set_auto_aspect_ratio()

        self.pool = multiprocessing.Pool(processes=4)
        self.jobs = queue.Queue()

        self.save_path = "captured_images"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, help="Path of the model",
                            default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
        parser.add_argument("--fps", type=int, help="Frames per second")
        parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
        parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
        parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
        parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
        parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
        parser.add_argument("--postprocess", choices=["", "nanodet"],
                            default=None, help="Run post process of type")
        parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                            help="preserve the pixel aspect ratio of the input tensor")
        parser.add_argument("--labels", type=str,
                            help="Path to the labels file")
        parser.add_argument("--print-intrinsics", action="store_true",
                            help="Print JSON network_intrinsics then exit")
        return parser.parse_args()

    @lru_cache
    def get_labels(self):
        labels = self.intrinsics.labels
        if self.intrinsics.ignore_dash_labels:
            labels = [label for label in labels if label and label != "-"]
        return labels

    def parse_detections(self, metadata: dict):
        """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
        bbox_normalization = self.intrinsics.bbox_normalization
        threshold = self.args.threshold
        iou = self.args.iou
        max_detections = self.args.max_detections

        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        input_w, input_h = self.imx500.get_input_size()
        if np_outputs is None:
            return None
        if self.intrinsics.postprocess == "nanodet":
            boxes, scores, classes = \
                postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                              max_out_dets=max_detections)[0]
            from picamera2.devices.imx500.postprocess import scale_boxes
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
        else:
            boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            if bbox_normalization:
                boxes = boxes / input_h

            boxes = np.array_split(boxes, 4, axis=1)
            boxes = zip(*boxes)

        detections = [
            Detection(box, category, score, metadata, self.imx500, self.picam2)
            for box, score, category in zip(boxes, scores, classes)
            if score > threshold
        ]
        return detections

    def run_detection_loop(self):
        while True:
            request = self.picam2.capture_request()
            metadata = request.get_metadata()
            if metadata:
                async_result = self.pool.apply_async(self.parse_detections, (metadata,))
                self.jobs.put((request, async_result))
            else:
                request.release()

    def capture_single_image(self) -> tuple[str, Image.Image]:
        """
        Capture a single image from the camera.

        Returns:
            tuple: (filepath, PIL.Image) - Path to saved image and the image object
        """
        # Capture array and convert to PIL Image
        array = self.picam2.capture_array()
        image = Image.fromarray(array)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(self.save_path, filename)

        # Save the image
        image.save(filepath)

        return filepath, image