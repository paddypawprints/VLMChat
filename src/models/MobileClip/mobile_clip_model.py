"""
MobileCLIP Video Processing Pipeline
====================================
A complete implementation for real-time video analysis using MobileCLIP.
This separates the vision encoder (for real-time processing) from text embeddings (precomputed).
Perfect for edge deployment on Jetson, Hailo, and other embedded devices.

Key Features:
- Separate vision and text encoders
- Precomputed text embeddings for efficiency
- Real-time video frame processing
- ONNX export ready
- Configurable similarity thresholding

Author: Assistant
Date: 2024
License: MIT
"""

import cv2
import torch
import numpy as np
from PIL import Image
import mobileclip

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model configuration
MODEL_TYPE = 'mobileclip_s2'  # Options: 'mobileclip_s0', 'mobileclip_s1', 'mobileclip_s2', 'mobileclip_b'
DEVICE = 'cpu'  # Use 'cuda' if available, 'cpu' for compatibility

# Video processing configuration
CAMERA_INDEX = 0  # Default webcam index
FRAME_SKIP = 1    # Process every nth frame (1 = process all frames)
SIMILARITY_THRESHOLD = 0.25  # Minimum confidence to display detection

# Concepts to detect (customize these for your use case)
CONCEPTS = [
    "a person walking",
    "a car driving",
    "a dog running",
    "someone sitting",
    "outdoor scene",
    "indoor scene",
    "a bicycle",
    "a red object",
    "a tree",
    "a building"
]

# =============================================================================
# MODEL SETUP
# =============================================================================

def setup_model(model_type, device):
    """
    Initialize MobileCLIP model and preprocessors.

    Args:
        model_type: MobileCLIP variant (s0, s1, s2, b)
        device: 'cpu' or 'cuda'

    Returns:
        model: Complete MobileCLIP model
        preprocess: Image preprocessing function
        tokenizer: Text tokenizer
    """
    print(f"Loading MobileCLIP {model_type}...")
    model, preprocess = mobileclip.create_model(model_type, device=device)
    tokenizer = mobileclip.get_tokenizer(model_type)

    # Print model statistics
    vision_params = sum(p.numel() for p in model.visual.parameters())
    text_params = sum(p.numel() for p in model.textual.parameters())
    print(f"Vision encoder parameters: {vision_params:,}")
    print(f"Text encoder parameters: {text_params:,}")
    print(f"Total parameters: {vision_params + text_params:,}")

    return model, preprocess, tokenizer

# =============================================================================
# TEXT EMBEDDING PRECOMPUTATION
# =============================================================================

def precompute_text_embeddings(concepts, text_encoder, tokenizer, device):
    """
    Precompute text embeddings for all concepts once.
    This avoids running the text encoder during real-time processing.

    Args:
        concepts: List of text descriptions
        text_encoder: MobileCLIP text encoder
        tokenizer: Text tokenizer
        device: Computation device

    Returns:
        text_embeddings: Tensor of shape [num_concepts, embedding_dim]
    """
    print("Precomputing text embeddings...")
    text_embeddings = []

    for i, concept in enumerate(concepts):
        print(f"  Processing concept {i+1}/{len(concepts)}: {concept}")
        text_input = tokenizer(concept)

        # Move inputs to device
        text_input = {k: v.to(device) for k, v in text_input.items()}

        with torch.no_grad():
            text_embedding = text_encoder(**text_input)
        text_embeddings.append(text_embedding)

    # Stack all embeddings and remove extra dimension
    text_embeddings = torch.stack(text_embeddings).squeeze(1)
    print(f"Text embeddings shape: {text_embeddings.shape}")

    return text_embeddings

# =============================================================================
# VISION PROCESSING
# =============================================================================

def process_video_frame(frame, vision_encoder, preprocess, text_embeddings, concepts, threshold=0.25):
    """
    Process a single video frame using only the vision encoder.

    Args:
        frame: OpenCV frame (BGR format)
        vision_encoder: MobileCLIP vision encoder
        preprocess: Image preprocessing function
        text_embeddings: Precomputed text embeddings
        concepts: List of concept descriptions
        threshold: Minimum similarity threshold

    Returns:
        results: Dictionary containing detection results
    """
    # Convert OpenCV BGR to RGB and create PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Preprocess image for MobileCLIP
    processed_image = preprocess(pil_image).unsqueeze(0)  # Add batch dimension

    # Get image embedding using vision encoder only
    with torch.no_grad():
        image_embedding = vision_encoder(processed_image)

    # Normalize embeddings for cosine similarity
    image_embedding = torch.nn.functional.normalize(image_embedding, p=2, dim=1)
    text_embeddings_norm = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)

    # Compute cosine similarities between image and all text concepts
    similarities = torch.mm(image_embedding, text_embeddings_norm.T).squeeze(0)

    # Find best match
    best_match_idx = torch.argmax(similarities).item()
    best_similarity = similarities[best_match_idx].item()
    best_concept = concepts[best_match_idx]

    # Prepare results
    results = {
        'best_match': best_concept,
        'confidence': best_similarity,
        'above_threshold': best_similarity > threshold,
        'all_similarities': {concept: sim.item() for concept, sim in zip(concepts, similarities)}
    }

    return results

# =============================================================================
# VIDEO PROCESSING LOOP
# =============================================================================

def video_processing_loop(camera_index, vision_encoder, preprocess, text_embeddings, concepts,
                         frame_skip=1, threshold=0.25):
    """
    Main video processing loop.

    Args:
        camera_index: Webcam index
        vision_encoder: MobileCLIP vision encoder
        preprocess: Image preprocessing function
        text_embeddings: Precomputed text embeddings
        concepts: List of concept descriptions
        frame_skip: Process every nth frame
        threshold: Similarity threshold
    """
    print(f"Starting video processing (Camera: {camera_index}, Frame skip: {frame_skip})...")
    print("Press 'q' to quit, 's' to save current frame")

    cap = cv2.VideoCapture(camera_index)
    frame_count = 0

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frame_count += 1

            # Skip frames if needed for performance
            if frame_count % frame_skip != 0:
                continue

            # Process frame
            results = process_video_frame(frame, vision_encoder, preprocess,
                                        text_embeddings, concepts, threshold)

            # Display results on frame
            display_text = f"{results['best_match']} ({results['confidence']:.2f})"
            color = (0, 255, 0) if results['above_threshold'] else (0, 0, 255)

            cv2.putText(frame, display_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Show frame
            cv2.imshow('MobileCLIP Video Analysis', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")

    except KeyboardInterrupt:
        print("Video processing interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Video processing stopped")

# =============================================================================
# EXPORT FUNCTIONS (FOR EDGE DEPLOYMENT)
# =============================================================================

def export_vision_encoder(vision_encoder, output_path="mobileclip_vision.onnx"):
    """
    Export the vision encoder to ONNX format for edge deployment.

    Args:
        vision_encoder: MobileCLIP vision encoder
        output_path: Output ONNX file path
    """
    print(f"Exporting vision encoder to {output_path}...")
    vision_encoder.eval()

    # Create dummy input (adjust size based on your needs)
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        vision_encoder,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=13,
        export_params=True,
        do_constant_folding=True
    )
    print("Vision encoder exported successfully")

def save_text_embeddings(text_embeddings, concepts, output_path="text_embeddings.npz"):
    """
    Save precomputed text embeddings for later use.

    Args:
        text_embeddings: Precomputed text embeddings tensor
        concepts: List of concept descriptions
        output_path: Output file path
    """
    print(f"Saving text embeddings to {output_path}...")

    # Convert to numpy for portability
    embeddings_np = text_embeddings.cpu().numpy()

    # Save with concepts
    np.savez(output_path,
             embeddings=embeddings_np,
             concepts=np.array(concepts, dtype=object))

    print("Text embeddings saved successfully")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("MobileCLIP Video Processing Pipeline")
    print("=" * 50)

    # Setup device
    if torch.cuda.is_available() and DEVICE == 'cuda':
        print("CUDA is available - using GPU")
    else:
        print("Using CPU for processing")

    # Initialize model
    model, preprocess, tokenizer = setup_model(MODEL_TYPE, DEVICE)

    # Access separate encoders
    vision_encoder = model.visual
    text_encoder = model.textual

    # Precompute text embeddings
    text_embeddings = precompute_text_embeddings(
        CONCEPTS, text_encoder, tokenizer, DEVICE
    )

    # Start video processing
    video_processing_loop(
        camera_index=CAMERA_INDEX,
        vision_encoder=vision_encoder,
        preprocess=preprocess,
        text_embeddings=text_embeddings,
        concepts=CONCEPTS,
        frame_skip=FRAME_SKIP,
        threshold=SIMILARITY_THRESHOLD
    )

    # Uncomment to export models for edge deployment
    # export_vision_encoder(vision_encoder, "mobileclip_vision.onnx")
    # save_text_embeddings(text_embeddings, CONCEPTS, "text_embeddings.npz")

    print("Pipeline completed")

# =============================================================================
# USAGE NOTES
# =============================================================================
"""
Usage Instructions:

1. Install dependencies:
   pip install mobile-clip opencv-python torch torchvision Pillow

2. Run the script:
   python mobileclip_pipeline.py

3. Customize the CONCEPTS list for your specific use case

4. Adjust SIMILARITY_THRESHOLD based on your accuracy requirements

5. For better performance on edge devices:
   - Export to ONNX using export_vision_encoder()
   - Use the saved text embeddings with your ONNX model
   - Consider quantizing the model for faster inference

6. For multi-camera setups, change CAMERA_INDEX

Performance Tips:
- Increase FRAME_SKIP for better performance on low-power devices
- Use smaller model variants (s0, s1) for faster inference
- Precompute text embeddings once and reuse them
- Consider batch processing for multiple frames

Edge Deployment:
1. Export vision encoder to ONNX
2. Save text embeddings as NPZ file
3. Load ONNX model on target device (Jetson, Hailo, etc.)
4. Load precomputed text embeddings
5. Implement similar processing loop on target device
"""