#!/usr/bin/env python3
"""
Inspect a serialized TensorRT engine and print binding info.

Usage:
  python inspect_engine.py image_fp16.engine

Run with the same Python you use for your compare script (or try /usr/bin/python3).
"""
import sys
import os
import numpy as np

if len(sys.argv) != 2:
    print("Usage: python inspect_engine.py <engine_path>")
    sys.exit(1)

engine_path = sys.argv[1]
try:
    import tensorrt as trt
except Exception as e:
    print("Failed to import tensorrt:", e)
    sys.exit(2)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def get_num_bindings(engine):
    # robustly return binding count
    if hasattr(engine, "num_bindings"):
        return engine.num_bindings
    if hasattr(engine, "get_nb_bindings"):
        try:
            return engine.get_nb_bindings()
        except Exception:
            pass
    # fallback: probe until exception
    i = 0
    while True:
        try:
            engine.get_binding_name(i)
            i += 1
        except Exception:
            break
    return i

def main():
    if not os.path.exists(engine_path):
        print("Engine not found:", engine_path)
        sys.exit(3)

    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        buf = f.read()
    engine = runtime.deserialize_cuda_engine(buf)
    if engine is None:
        print("Failed to deserialize engine.")
        sys.exit(4)

    print("Engine loaded (via Python tensorrt).")
    nb = get_num_bindings(engine)
    print("num_bindings:", nb)

    ctx = engine.create_execution_context()
    print("Created execution context:", type(ctx))

    for i in range(nb):
        try:
            name = engine.get_binding_name(i)
        except Exception:
            name = f"binding_{i}"
        try:
            is_input = engine.binding_is_input(i)
        except Exception:
            is_input = None
        try:
            dtype = trt.nptype(engine.get_binding_dtype(i))
        except Exception:
            dtype = "<unknown>"
        try:
            shape = tuple(ctx.get_binding_shape(i))
        except Exception:
            shape = "<unknown>"
        kind = "INPUT" if is_input else "OUTPUT" if is_input is not None else "UNKNOWN"
        print(f" binding[{i}] name='{name}' kind={kind} dtype={dtype} shape={shape}")

if __name__ == "__main__":
    main()