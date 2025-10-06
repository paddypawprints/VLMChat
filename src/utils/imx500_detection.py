"""
IMX500 Camera with object detection capabilities.

This module provides the IMX500 camera implementation with neural processing
for object detection, inheriting from both BaseCamera and ObjectDetectionInterface.
"""

import argparse
import multiprocessing
import queue
import os
from functools import lru_cache
from typing import List, Optional

import numpy as np
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)

from .imx500_camera import IMX500Camera
from .detection_base import Detection, ObjectDetectionInterface
from .camera_base import Platform, Device


class IMX500ObjectDetection(IMX500Camera, ObjectDetectionInterface):
    """
    IMX500 camera with object detection capabilities.

    Extends the basic IMX500Camera with neural processing for object detection,
    implementing the ObjectDetectionInterface.
    """

    def __init__(self, args=None, platform: Platform = Platform.RPI,
                 device: Device = Device.CAMERA0):
        """
        Initialize IMX500 camera with object detection model.

        Args:
            args: Optional command-line arguments for configuration
            platform: Platform type (defaults to RPI)
            device: Device identifier (defaults to CAMERA0)

        Raises:
            ValueError: If the loaded model is not an object detection model
        """
        # Initialize base camera first (but override some settings)
        self._args = args if args else self.get_args()

        # Call BaseCamera.__init__ directly to avoid IMX500Camera setup
        from .camera_base import BaseCamera, CameraModel
        BaseCamera.__init__(self, CameraModel.IMX500, platform, device)

        # Setup IMX500 with neural processing
        self._imx500 = self._imx500 = self._args.model and self._args.model or "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
        from picamera2.devices import IMX500
        self._imx500 = IMX500(self._args.model)
        self._intrinsics = self._imx500.network_intrinsics

        # Initialize network intrinsics if not available
        if not self._intrinsics:
            self._intrinsics = NetworkIntrinsics()
            self._intrinsics.task = "object detection"
        elif self._intrinsics.task != "object detection":
            raise ValueError("Network is not an object detection task")

        # Apply configuration from command-line arguments
        for key, value in vars(self._args).items():
            if key == 'labels' and value is not None:
                with open(value, 'r') as f:
                    self._intrinsics.labels = f.read().splitlines()
            elif hasattr(self._intrinsics, key) and value is not None:
                setattr(self._intrinsics, key, value)

        # Load default COCO labels if none provided
        if self._intrinsics.labels is None:
            from config import get_config
            config = get_config()
            coco_labels_path = os.path.join(config.paths.project_root, config.paths.coco_labels_path)
            with open(coco_labels_path, "r") as f:
                self._intrinsics.labels = f.read().splitlines()
        self._intrinsics.update_with_defaults()

        # Initialize camera with RGB format for vision model compatibility
        from picamera2 import Picamera2
        self._picam2 = Picamera2(self._imx500.camera_num)
        main = {'format': 'RGB888'}
        self._config = self._picam2.create_preview_configuration(
            main,
            controls={"FrameRate": self._intrinsics.inference_rate},
            buffer_count=12
        )

        # Start camera with neural network loading progress
        self._imx500.show_network_fw_progress_bar()
        self._picam2.start(self._config, show_preview=False)
        if self._intrinsics.preserve_aspect_ratio:
            self._imx500.set_auto_aspect_ratio()

        # Initialize processing pipeline
        self._pool = multiprocessing.Pool(processes=4)
        self._jobs = queue.Queue()

        # Create directory for captured images using configuration
        from config import get_config
        config = get_config()
        self._save_path = config.paths.captured_images_dir
        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)

    @staticmethod
    def get_args():
        """
        Parse command-line arguments for camera configuration.

        Returns:
            argparse.Namespace: Parsed command-line arguments
        """
        parser = argparse.ArgumentParser(description="IMX500 Object Detection Camera")
        parser.add_argument("--model", type=str, help="Path to the neural network model",
                            default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
        parser.add_argument("--fps", type=int, help="Target frames per second for capture")
        parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction,
                            help="Enable bounding box coordinate normalization")
        parser.add_argument("--threshold", type=float, default=0.55,
                            help="Confidence threshold for object detection")
        parser.add_argument("--iou", type=float, default=0.65,
                            help="Intersection over Union threshold for NMS")
        parser.add_argument("--max-detections", type=int, default=10,
                            help="Maximum number of detections per frame")
        parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction,
                            help="Filter out empty or dash labels")
        parser.add_argument("--postprocess", choices=["", "nanodet"],
                            default=None, help="Post-processing method for detections")
        parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                            help="Preserve pixel aspect ratio of input tensor")
        parser.add_argument("--labels", type=str,
                            help="Path to custom labels file")
        parser.add_argument("--print-intrinsics", action="store_true",
                            help="Print model intrinsics and exit")
        return parser.parse_args()

    @property
    def args(self):
        """Get the command-line arguments."""
        return self._args

    @lru_cache
    def get_labels(self) -> List[str]:
        """
        Get filtered object detection labels.

        Returns:
            List[str]: Filtered list of object class labels
        """
        labels = self._intrinsics.labels
        # Filter out empty or dash labels if configured
        if self._intrinsics.ignore_dash_labels:
            labels = [label for label in labels if label and label != "-"]
        return labels

    def parse_detections(self, metadata: dict) -> Optional[List[Detection]]:
        """
        Parse neural network output into detected objects.

        Args:
            metadata: Frame metadata containing inference results

        Returns:
            List[Detection]: List of detected objects above confidence threshold,
                           or None if no outputs available
        """
        # Get detection parameters from configuration
        bbox_normalization = self._intrinsics.bbox_normalization
        threshold = self._args.threshold
        iou = self._args.iou
        max_detections = self._args.max_detections

        # Extract neural network outputs from metadata
        np_outputs = self._imx500.get_outputs(metadata, add_batch=True)
        input_w, input_h = self._imx500.get_input_size()
        if np_outputs is None:
            return None

        # Apply post-processing based on model type
        if self._intrinsics.postprocess == "nanodet":
            # Use NanoDet-specific post-processing
            boxes, scores, classes = \
                postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                              max_out_dets=max_detections)[0]
            from picamera2.devices.imx500.postprocess import scale_boxes
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
        else:
            # Standard detection post-processing
            boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            if bbox_normalization:
                boxes = boxes / input_h

            # Reshape box coordinates for processing
            boxes = np.array_split(boxes, 4, axis=1)
            boxes = zip(*boxes)

        # Create Detection objects for valid detections with coordinate conversion
        detections = []
        for box, score, category in zip(boxes, scores, classes):
            if score > threshold:
                # Convert coordinates using IMX500 method and convert to integers
                converted_box = self._imx500.convert_inference_coords(box, metadata, self._picam2)
                int_box = (int(converted_box[0]), int(converted_box[1]),
                          int(converted_box[2]), int(converted_box[3]))

                # Get category label
                labels = self.get_labels()
                category_label = labels[int(category)] if int(category) < len(labels) else f"class_{int(category)}"

                detections.append(Detection(int_box, category_label, float(score)))

        return detections

    def run_detection_loop(self):
        """
        Run continuous object detection loop.

        Continuously captures frames from the camera, processes them through
        the neural network for object detection, and queues results for
        asynchronous processing.
        """
        while True:
            # Capture frame with inference metadata
            request = self._picam2.capture_request()
            metadata = request.get_metadata()
            if metadata:
                # Process detections asynchronously
                async_result = self._pool.apply_async(self.parse_detections, (metadata,))
                self._jobs.put((request, async_result))
            else:
                # Release request if no metadata available
                request.release()