# Child class implementation
import .detection_base

def load_names_from_config(config_path: Optional[Path] = None) -> List[str]:
    """
    Load class names from config file. If config_path is None, look for
    coco_names.json then coco_names.yaml next to this file.
    """
    if config_path is None:
        base = Path(__file__).parent
        json_path = base / "coco_names.json"
        yaml_path = base / "coco_names.yaml"
    else:
        json_path = Path(config_path)
        yaml_path = None if json_path.suffix.lower() == ".json" else Path(config_path)

    # try JSON first
    if json_path and json_path.exists() and json_path.suffix.lower() == ".json":
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return list(data)
        except Exception:
            pass

    # try YAML if available
    if yaml_path is None:
        # maybe the user passed a .yaml explicitly as config_path
        if config_path and Path(config_path).suffix.lower() in (".yaml", ".yml"):
            yaml_path = Path(config_path)
    if yaml_path and yaml_path.exists():
        try:
            import yaml  # PyYAML
        except Exception:
            raise RuntimeError("PyYAML is required to load YAML config files. Install with 'pip install pyyaml'.")
        with yaml_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, dict) and "names" in data:
            return list(data["names"])
        if isinstance(data, list):
            return list(data)

    # fallback: numeric labels
    return [str(i) for i in range(80)]

class YoloObjectDetector(ObjectDetector):
    """
    Concrete implementation of ObjectDetector using a hypothetical YOLO model.
    """
    def __init__(
        self,
        engine_path: str = "yolov8n_fp16.engine",
        input_name: str = "images",
        output_name: str = "output0",
        model_size: int = 640,
        class_config: Optional[Path] = None,
    ):
        self.engine_path = engine_path
        self.input_name = input_name
        self.output_name = output_name
        self.model_size = model_size
        # If a specific config path is provided, reload names
        if class_config is not None:
            global CLASSES  # allow override
            CLASSES = load_names_from_config(class_config)

    def start(self) -> None:
        """Starts the detection process or initializes the model for use."""
        if not self._running:
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            with open(self.engine_path, "rb") as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError(f"Failed to deserialize engine from {self.engine_path}")
            self.ctx = self.engine.create_execution_context()
        else:
            print("YOLO Detector is already running.")

    def stop(self) -> None:
        """Stops any running threads or cleans up resources."""
        if self._running:
            print("YOLO Detector stopped.")
            self.ctx = None
            self.engine = None
            self._running = False

    def readiness(self) -> bool:
        """Checks if the detector is ready to perform detection."""
        # In a real scenario, this might check if the model is loaded/initialized
        return self._running

    # 1) Prepare image using tensor->blob (stretch resize on CPU)
    def prepare_image(self, image: Image) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        h, w = image.shape[:2]
        #arr = image[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB
        arr = image.astype(np.float32) / 255.0
        device = torch.device("cpu")
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        tensor_resized = F.interpolate(tensor, size=(self.model_size, self.model_size), mode="bilinear", align_corners=False)
        tensor_resized = tensor_resized.to(torch.float32).contiguous()
        blob = np.ascontiguousarray(tensor_resized.cpu().numpy())
        ratio_w = float(self.model_size) / float(w)
        ratio_h = float(self.model_size) / float(h)
        ratio = (ratio_w, ratio_h) if abs(ratio_w - ratio_h) > 1e-6 else float(ratio_w)
        scale = float(max(h, w)) / float(self.model_size)
        meta = {"mode": "stretch", "ratio": ratio, "pad_x": 0, "pad_y": 0}
        return blob, scale, meta

    def run_trt_inference(self, blob: np.ndarray) -> np.ndarray:
        in_shape = tuple(self.ctx.get_tensor_shape(self.input_name))
        out_shape = tuple(self.ctx.get_tensor_shape(self.output_name))
        in_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        out_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
        in_elems = int(np.prod(in_shape))
        out_elems = int(np.prod(out_shape))
        host_in = cuda.pagelocked_empty(in_elems, in_dtype)
        host_out = cuda.pagelocked_empty(out_elems, out_dtype)
        src = np.ascontiguousarray(blob).ravel().astype(in_dtype, copy=False)
        if src.size != in_elems:
            raise RuntimeError(f"Input size mismatch: engine expects {in_elems} elements, got {src.size}")
        import numpy as _np

        _np.copyto(_np.frombuffer(host_in, dtype=in_dtype, count=in_elems), src)
        d_in = cuda.mem_alloc(host_in.nbytes)
        d_out = cuda.mem_alloc(host_out.nbytes)
        self.ctx.set_tensor_address(self.input_name, int(d_in))
        self.ctx.set_tensor_address(self.output_name, int(d_out))
        stream = cuda.Stream()
        cuda.memcpy_htod_async(d_in, host_in, stream)
        self.ctx.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_out, d_out, stream)
        stream.synchronize()
        trt_out = np.array(host_out).reshape(out_shape)
        if trt_out.dtype == np.float16:
            trt_out = trt_out.astype(np.float32)
        out_np = np.asarray(trt_out)
        if out_np.ndim == 3:
            if out_np.shape[0] == 1:
                out_np = np.squeeze(out_np, axis=0)
            else:
                raise RuntimeError("TRT output batch > 1 not supported")
        if out_np.ndim != 2:
            raise RuntimeError(f"Unexpected TRT output rank {out_np.ndim}; expected 2 after squeeze.")
        return out_np

    def decode_trt_output(
        self,
        raw_out: np.ndarray,
        scale: float,
        meta: Dict[str, Any],
        conf_thresh: float = 0.25,
        max_detections: Optional[int] = None,
    ) -> List[Detection]:
        a = np.asarray(raw_out)
        if a.ndim == 1:
            for A in (84, 85):
                if a.size % A == 0:
                    preds = a.reshape(-1, A)
                    break
            else:
                raise RuntimeError("Cannot reshape flat TRT output into preds.")
        elif a.ndim == 2:
            if a.shape[0] in (84, 85):
                preds = a.T.copy()
            elif a.shape[1] in (84, 85):
                preds = a.copy()
            elif a.shape[0] < a.shape[1]:
                preds = a.copy()
            else:
                preds = a.T.copy()
        else:
            raise RuntimeError(f"Unsupported TRT output rank {a.ndim}")
        boxes = []
        scores = []
        class_ids = []
        P, A = preds.shape
        for i in range(P):
            row = preds[i]
            class_scores = row[4:]
            idx = int(np.argmax(class_scores))
            score = float(class_scores[idx])
            if score >= conf_thresh:
                cx, cy, w_box, h_box = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                left = cx - 0.5 * w_box
                top = cy - 0.5 * h_box
                boxes.append([left, top, w_box, h_box])
                scores.append(score)
                class_ids.append(idx)
        if len(boxes) == 0:
            return []
        keep = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, 0.45, 0.5)
        if isinstance(keep, (list, tuple, np.ndarray)):
            try:
                keep = np.array(keep).reshape(-1).tolist()
            except Exception:
                keep = [int(x[0]) if isinstance(x, (list, tuple, np.ndarray)) else int(x) for x in keep]
        else:
            keep = []
        detections: List[Dict[str, Any]] = []
        for idx in keep:
            idx = int(idx)
            cid = class_ids[idx]
            score = scores[idx]
            l, t, w_box, h_box = boxes[idx]
            if meta.get("mode") == "letterbox":
                ratio = float(meta["ratio"])
                pad_x = int(meta["pad_x"])
                pad_y = int(meta["pad_y"])
                x_orig = (l - pad_x) / ratio
                y_orig = (t - pad_y) / ratio
                w_orig = w_box / ratio
                h_orig = h_box / ratio
            elif meta.get("mode") == "stretch":
                ratio = meta["ratio"]
                if isinstance(ratio, tuple):
                    ratio_w, ratio_h = ratio
                else:
                    ratio_w = ratio_h = float(ratio)
                x_orig = l / ratio_w
                y_orig = t / ratio_h
                w_orig = w_box / ratio_w
                h_orig = h_box / ratio_h
            else:
                x_orig = l * scale
                y_orig = t * scale
                w_orig = w_box * scale
                h_orig = h_box * scale
            x1 = x_orig
            y1 = y_orig
            x2 = x_orig + w_orig
            y2 = y_orig + h_orig
            detections.append(
                Detection(
                    [x1, y1, x2, y2], 
                    CLASSES[cid] if cid < len(CLASSES) else str(cid),
                    score,
                )
            )
        if max_detections is not None and len(detections) > max_detections:
            detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)[:max_detections]
        return detections


    def detect(self, image: Image, detections: Optional[List[Detection]] = None) -> List[Detection]:
        """
        Performs object detection on the given image using the YOLO model.

        Args:
            image: The PIL image to process.
            detections: Optional list of existing detections (unused in this example).

        Returns:
            A list of Detection objects found in the image.
        """
        if not self.readiness():
            raise RuntimeError("Detector is not started. Call start() first.")

        blob, scale, meta = self.prepare_image(image)
        raw_out = self.run_trt_inference(blob)
        return self.decode_trt_output(raw_out, scale, meta, conf_thresh=conf_thresh, max_detections=max_detections)
