#!/usr/bin/env python3
"""
Inspect the object returned by runtime.deserialize_cuda_engine(...).

Usage:
  python inspect_trt_engine.py path/to/engine.engine > engine_report.txt

This prints:
 - Python type information for the engine object
 - MRO / class name / module
 - dir(engine)
 - list of callable members (methods), with best-effort signatures and one-line docstrings
 - list of builtin/C-extension callables vs Python-level callables
 - a few safe example calls if the corresponding methods exist (get_binding_name(0), get_binding_index('name'))
"""
import sys
import os
import inspect
import textwrap

try:
    import tensorrt as trt
except Exception as e:
    print("ERROR: failed to import tensorrt:", e, file=sys.stderr)
    sys.exit(2)

# import pycuda.autoinit to ensure CUDA context created (only if needed later)
try:
    import pycuda.autoinit  # noqa: F401
except Exception:
    pass

ENGINE_PATH = sys.argv[1] if len(sys.argv) > 1 else "image_fp16.engine"

def short(s, maxlen=200):
    if not s:
        return ""
    s = s.strip().splitlines()[0]
    return s if len(s) <= maxlen else s[:maxlen] + "..."

def safe_sig(obj):
    try:
        return str(inspect.signature(obj))
    except Exception:
        return None

def main():
    if not os.path.exists(ENGINE_PATH):
        print("Engine file not found:", ENGINE_PATH, file=sys.stderr)
        sys.exit(3)

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    # deserialize
    with open(ENGINE_PATH, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    print("Engine object repr:", repr(engine))
    print("Type:", type(engine))
    cls = engine.__class__
    print("Class:", cls)
    print("Class module:", getattr(cls, "__module__", None))
    print("Class name:", getattr(cls, "__name__", None))
    print("MRO:", getattr(cls, "__mro__", None))
    # is instance of known TRT class?
    try:
        is_icuda = isinstance(engine, trt.ICudaEngine)
    except Exception:
        is_icuda = "<ICudaEngine not available>"
    print("isinstance(engine, trt.ICudaEngine):", is_icuda)
    print()

    # dir summary
    names = sorted(dir(engine))
    print(f"Total members on engine: {len(names)}")
    print("Members (one-per-line):")
    for n in names:
        print(" ", n)
    print()

    # classify members: builtin vs python callables vs attributes
    python_callables = []
    builtin_callables = []
    attrs = []
    for name in names:
        try:
            member = getattr(engine, name)
        except Exception as e:
            attrs.append((name, f"<unreadable: {e}>"))
            continue
        if inspect.isbuiltin(member) or inspect.ismethoddescriptor(member):
            builtin_callables.append(name)
        elif callable(member):
            python_callables.append(name)
        else:
            attrs.append((name, repr(member)))

    print("Builtin / C-extension callables (subset):")
    for n in builtin_callables[:200]:
        print("  ", n)
    print()
    print("Python-callable members (subset):")
    for n in python_callables[:200]:
        print("  ", n)
    print()
    print("Non-callable attributes (subset):")
    for n, val in attrs[:200]:
        print("  ", n, "=", short(val))
    print()

    # Detailed description for callables: signature + short doc
    print("Detailed callable member info (best-effort):")
    for name in sorted(python_callables + builtin_callables):
        try:
            obj = getattr(engine, name)
        except Exception as e:
            print(f"  {name}: <unreadable: {e}>")
            continue
        sig = safe_sig(obj)
        doc = short(getattr(obj, "__doc__", None))
        kind = "builtin" if (inspect.isbuiltin(obj) or inspect.ismethoddescriptor(obj)) else "python"
        if sig:
            print(f"  {name}{sig}  [{kind}]  # {doc}")
        else:
            print(f"  {name}()  [{kind}]  # {doc}")

    print()
    print("Safe example calls (only if methods exist):")
    # get_binding_name(0)
    try:
        if "get_binding_name" in names:
            try:
                print("  get_binding_name(0) ->", engine.get_binding_name(0))
            except Exception as e:
                print("  get_binding_name(0) raised:", e)
    except Exception:
        pass

    # get_binding_index('image_input')
    try:
        if "get_binding_index" in names:
            sample_name = "image_input"
            try:
                idx = engine.get_binding_index(sample_name)
                print(f"  get_binding_index('{sample_name}') ->", idx)
            except Exception as e:
                print(f"  get_binding_index('{sample_name}') raised:", e)
    except Exception:
        pass

    # binding_is_input(0)
    try:
        if "binding_is_input" in names:
            try:
                print("  binding_is_input(0) ->", engine.binding_is_input(0))
            except Exception as e:
                print("  binding_is_input(0) raised:", e)
    except Exception:
        pass

    # get_binding_dtype(0) and nptype conversion
    try:
        if "get_binding_dtype" in names:
            try:
                dt = engine.get_binding_dtype(0)
                print("  get_binding_dtype(0) ->", dt, "nptype->", trt.nptype(dt))
            except Exception as e:
                print("  get_binding_dtype(0) raised:", e)
    except Exception:
        pass

    # context example: create and inspect a context if possible
    try:
        if "create_execution_context" in names:
            ctx = engine.create_execution_context()
            print("  created execution context:", type(ctx))
            try:
                ctx_members = sorted(dir(ctx))
                print("  context members (subset):", ctx_members[:60])
                # check for modern methods on context
                for m in ("execute_async_v3", "execute_async_v2", "get_tensor_shape", "set_tensor_address", "get_binding_shape", "set_binding_shape"):
                    print(f"    context has {m}():", hasattr(ctx, m))
            except Exception as e:
                print("  failed to inspect context:", e)
    except Exception:
        pass

    print("\nDone.")

if __name__ == "__main__":
    main()