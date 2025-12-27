#!/usr/bin/env python3
"""
Build TensorRT engine from ONNX with a defined workspace and API fallbacks.

Usage:
  python build_engine_fix.py

Edit the constants below if needed.
"""
import os
import sys
import numpy as np

try:
    import tensorrt as trt
except Exception as e:
    print("ERROR: import tensorrt failed:", e)
    sys.exit(1)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Configurable defaults
ONNX_PATH = "openclip_image_encoder.onnx"   # change if needed
ENGINE_PATH = "image_fp16_built.engine"
WORKSPACE_MB = 1024     # <-- set workspace here (MB)
USE_FP16 = True

def build_engine_from_onnx(onnx_path, engine_path, workspace_mb=1024, fp16=True):
    print(f"Building engine from {onnx_path} (workspace={workspace_mb} MB, fp16={fp16})")
    builder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags=explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        onnx_data = f.read()

    if not parser.parse(onnx_data):
        print("ONNX parse failed. Parser errors:")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    # set workspace in bytes
    #config.max_workspace_size = int(workspace_mb) * (1 << 20)
    if fp16:
        try:
            config.set_flag(trt.BuilderFlag.FP16)
        except Exception:
            # Older/newer APIs might differ; ignoreable if not supported
            pass

    # optional: add an optimization profile if input is dynamic and profile is needed.
    # profile = builder.create_optimization_profile()
    # profile.set_shape("image_input", (1,3,256,256), (1,3,256,256), (4,3,256,256))
    # config.add_optimization_profile(profile)

    runtime = trt.Runtime(TRT_LOGGER)
    engine = None

    # Try builder.build_engine (common in many TRT versions)
    if hasattr(builder, "build_engine"):
        try:
            engine = builder.build_engine(network, config)
            print("Used builder.build_engine()")
        except Exception as e:
            print("builder.build_engine failed:", e)
            engine = None

    # Try older API build_cuda_engine
    if engine is None and hasattr(builder, "build_cuda_engine"):
        try:
            engine = builder.build_cuda_engine(network)
            print("Used builder.build_cuda_engine()")
        except Exception as e:
            print("builder.build_cuda_engine failed:", e)
            engine = None

    # Try serialized network API (returns serialized bytes)
    if engine is None and hasattr(builder, "build_serialized_network"):
        try:
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError("build_serialized_network returned None")
            engine = runtime.deserialize_cuda_engine(serialized)
            print("Used builder.build_serialized_network() + runtime.deserialize_cuda_engine()")
        except Exception as e:
            print("build_serialized_network -> deserialize failed:", e)
            engine = None

    # Generic fallback: look for any "serialized" method on builder
    if engine is None:
        for name in dir(builder):
            if "serialized" in name.lower():
                fn = getattr(builder, name)
                if callable(fn):
                    try:
                        print("Trying fallback builder method:", name)
                        serialized = fn(network, config)
                        if serialized:
                            engine = runtime.deserialize_cuda_engine(serialized)
                            print("Fallback succeeded:", name)
                            break
                    except Exception:
                        pass

    if engine is None:
        raise RuntimeError(
            "Failed to build engine: no supported builder API succeeded. "
            "Tried build_engine, build_cuda_engine, build_serialized_network and fallbacks."
        )

    # serialize to disk if possible
    try:
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        print("Serialized engine saved to:", engine_path)
    except Exception as e:
        print("Warning: could not serialize engine to disk:", e)

    return engine

def main():
    if not os.path.exists(ONNX_PATH):
        print("ONNX file not found:", ONNX_PATH)
        sys.exit(1)
    engine = build_engine_from_onnx(ONNX_PATH, ENGINE_PATH, workspace_mb=WORKSPACE_MB, fp16=USE_FP16)
    print("Engine build complete. You can now run inference with this engine file.")

if __name__ == "__main__":
    main()
