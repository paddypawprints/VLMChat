#!/usr/bin/env python3
"""
Demonstration script for ImageLibraryCamera.

This script demonstrates the usage of the ImageLibraryCamera class.
"""

import sys
import os
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from camera.image_library_camera import ImageLibraryCamera
from camera.camera_base import CameraModel, Platform, Device
from camera.camera_factory import CameraFactory
from camera.metrics_collector import Collector
from PIL import Image
import numpy as np


def create_sample_images(base_dir: Path, num_images: int = 5):
    """Create sample images for demonstration."""
    print(f"Creating {num_images} sample images in {base_dir}")
    
    # Create subdirectories
    dir_a = base_dir / "dir_a"
    dir_b = base_dir / "dir_b"
    dir_a_sub = dir_a / "subdir"
    
    for d in [dir_a, dir_b, dir_a_sub]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Create images with different colors in different directories
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    
    # Root directory images
    for i in range(2):
        img_array = np.full((100, 100, 3), colors[i], dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(base_dir / f"root_{i}.jpg")
    
    # dir_a images
    img_array = np.full((100, 100, 3), colors[2], dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(dir_a / "a_image.jpg")
    
    # dir_a/subdir images
    img_array = np.full((100, 100, 3), colors[3], dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(dir_a_sub / "sub_image.jpg")
    
    # dir_b images
    img_array = np.full((100, 100, 3), colors[4], dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(dir_b / "b_image.jpg")
    
    print(f"Created sample images")


def demo_basic_usage():
    """Demonstrate basic ImageLibraryCamera usage."""
    print("\n" + "="*60)
    print("DEMO: Basic ImageLibraryCamera Usage")
    print("="*60)
    
    # Create temporary directories
    image_dir = tempfile.mkdtemp(prefix="demo_images_")
    save_dir = tempfile.mkdtemp(prefix="demo_captures_")
    
    try:
        # Create sample images
        create_sample_images(Path(image_dir))
        
        # Create camera
        print(f"\nCreating ImageLibraryCamera:")
        print(f"  Image directory: {image_dir}")
        print(f"  Target resolution: 50x50")
        print(f"  Frame rate: 2 fps")
        
        camera = ImageLibraryCamera(
            image_base_dir=image_dir,
            width=50,
            height=50,
            framerate=2,
            save_dir=save_dir
        )
        
        print(f"\nFound {len(camera._image_paths)} images")
        print("\nImage traversal order:")
        for i, path in enumerate(camera._image_paths):
            rel_path = path.relative_to(image_dir)
            print(f"  {i+1}. {rel_path}")
        
        # Capture without starting thread
        print("\n--- Capturing image without thread ---")
        path, img = camera.capture_single_image()
        print(f"Captured: {Path(path).name if path != 'blank' else path}")
        print(f"Image size: {img.size}")
        
        # Start thread and capture multiple images
        print("\n--- Starting camera thread ---")
        camera.start()
        
        print("\nCapturing images over time:")
        for i in range(5):
            time.sleep(0.6)  # Wait for next frame (framerate is 2 fps)
            path, img = camera.capture_single_image()
            rel_path = Path(path).relative_to(image_dir) if path != 'blank' else path
            print(f"  {i+1}. Captured: {rel_path}, Size: {img.size}")
        
        camera.stop()
        print("\nCamera stopped")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(image_dir)
        shutil.rmtree(save_dir)


def demo_with_metrics():
    """Demonstrate ImageLibraryCamera with metrics collection."""
    print("\n" + "="*60)
    print("DEMO: ImageLibraryCamera with Metrics")
    print("="*60)
    
    # Create temporary directories
    image_dir = tempfile.mkdtemp(prefix="demo_images_")
    save_dir = tempfile.mkdtemp(prefix="demo_captures_")
    
    try:
        # Create sample images with different sizes
        print("\nCreating images with different sizes:")
        base = Path(image_dir)
        
        # Small image (will require resize and fill)
        img_array = np.full((20, 20, 3), (255, 0, 0), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(base / "small.jpg")
        print(f"  Created small.jpg (20x20)")
        
        # Large image (will require resize and crop)
        img_array = np.full((200, 200, 3), (0, 255, 0), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(base / "large.jpg")
        print(f"  Created large.jpg (200x200)")
        
        # Wide image (will require crop)
        img_array = np.full((50, 150, 3), (0, 0, 255), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(base / "wide.jpg")
        print(f"  Created wide.jpg (150x50)")
        
        # Create metrics collector
        collector = Collector()
        
        # Create camera with metrics
        camera = ImageLibraryCamera(
            image_base_dir=image_dir,
            width=100,
            height=100,
            framerate=5,
            save_dir=save_dir,
            metrics_collector=collector
        )
        
        print(f"\nProcessing images...")
        
        # Start camera thread
        camera.start()
        time.sleep(1.0)  # Let it process images
        camera.stop()
        
        # Check metrics
        print("\n--- Metrics Summary ---")
        ts_data = collector.snapshot_timeseries("image_library_operations")
        
        operation_counts = {}
        for dp in ts_data:
            op = dp.attributes.get('operation', 'unknown')
            operation_counts[op] = operation_counts.get(op, 0) + 1
        
        print(f"Operations performed:")
        for op, count in sorted(operation_counts.items()):
            print(f"  {op}: {count}")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(image_dir)
        shutil.rmtree(save_dir)


def demo_factory_creation():
    """Demonstrate creating ImageLibraryCamera through factory."""
    print("\n" + "="*60)
    print("DEMO: Creating ImageLibraryCamera via Factory")
    print("="*60)
    
    # Create temporary directories
    image_dir = tempfile.mkdtemp(prefix="demo_images_")
    save_dir = tempfile.mkdtemp(prefix="demo_captures_")
    
    try:
        # Create a simple 1x1 test image
        base = Path(image_dir)
        img_array = np.full((1, 1, 3), (128, 128, 128), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(base / "tiny.jpg")
        print(f"\nCreated 1x1 test image")
        
        # Create camera through factory
        print("\nCreating camera through CameraFactory:")
        args = {
            'image_base_dir': image_dir,
            'width': 10,
            'height': 10,
            'framerate': 1,
            'save_dir': save_dir,
        }
        
        camera = CameraFactory.create_camera(
            model=CameraModel.IMAGE_LIBRARY,
            platform=Platform.RPI,
            device=Device.CAMERA0,
            args=args
        )
        
        print(f"  Camera type: {type(camera).__name__}")
        print(f"  Camera model: {camera.model}")
        print(f"  Resolution: {camera.width}x{camera.height}")
        print(f"  Frame rate: {camera.framerate} fps")
        
        # Capture an image
        path, img = camera.capture_single_image()
        print(f"\nCaptured image:")
        print(f"  Size: {img.size}")
        print(f"  Path: {Path(path).name if path != 'blank' else path}")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(image_dir)
        shutil.rmtree(save_dir)


if __name__ == '__main__':
    print("\nImageLibraryCamera Demonstration")
    print("="*60)
    
    demo_basic_usage()
    demo_with_metrics()
    demo_factory_creation()
    
    print("\n" + "="*60)
    print("All demonstrations completed successfully!")
    print("="*60 + "\n")
