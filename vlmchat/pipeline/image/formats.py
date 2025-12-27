"""Image format enumeration and contracts."""
from enum import Enum


class ImageFormat(Enum):
    """Supported image data formats in pipeline."""
    PIL = "pil"                    # PIL.Image
    NUMPY = "numpy"                # np.ndarray (HWC, uint8)
    TORCH_CPU = "torch_cpu"        # torch.Tensor (CHW, float32, CPU)
    TORCH_GPU = "torch_gpu"        # torch.Tensor (CHW, float32, CUDA)
    OPENCV_GPU = "opencv_gpu"      # cv2.cuda_GpuMat
    DISPLAY_BUFFER = "display"     # Platform-specific display buffer


# Format contracts (used for validation)
FORMAT_CONTRACTS = {
    ImageFormat.PIL: "PIL.Image.Image",
    ImageFormat.NUMPY: "numpy.ndarray[H,W,C]",
    ImageFormat.TORCH_CPU: "torch.Tensor[C,H,W,device=cpu]",
    ImageFormat.TORCH_GPU: "torch.Tensor[C,H,W,device=cuda]",
    ImageFormat.OPENCV_GPU: "cv2.cuda_GpuMat",
    ImageFormat.DISPLAY_BUFFER: "platform_buffer"
}
