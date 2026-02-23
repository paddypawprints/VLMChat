"""Example: Integrated metrics and memory monitoring.

Shows how memory tracking integrates with the metrics collector.
"""

import time
import numpy as np
from camera_framework import (
    Context,
    BaseTask,
    Runner,
    Buffer,
    drop_oldest_policy,
    Collector,
    MemoryInstrument,
    AvgDurationInstrument,
    CounterInstrument,
)


class CameraTask(BaseTask):
    """Camera source with integrated metrics."""
    
    def __init__(self, collector: Collector):
        super().__init__(name="camera", interval=0.033)  # ~30fps
        self.collector = collector
        self.frame_count = 0
    
    def process(self) -> None:
        with self.collector.duration_timer("camera.capture_ms"):
            # Simulate camera capture
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Track memory
            obj_id = id(frame)
            self.collector.record("memory.track", frame.nbytes, attributes={
                "type": "numpy.ndarray",
                "obj_id": str(obj_id),
                "shape": str(frame.shape)
            })
            # Get memory instrument and track object
            mem_inst = self.collector.get_instrument("memory.images")
            if mem_inst:
                mem_inst.track_object(frame, obj_id)
            
            # Create context
            ctx = Context()
            ctx.append("frame", frame)
            
            # Track context
            ctx_id = id(ctx)
            self.collector.record("memory.track", 1024, attributes={  # Estimate size
                "type": "Context",
                "obj_id": str(ctx_id)
            })
            if mem_inst:
                mem_inst.track_object(ctx, ctx_id)
            
            # Record frame counter
            self.collector.record("camera.frames", 1)
            self.frame_count += 1
        
        # Write to output
        for buf in self.outputs:
            buf.put(ctx)


class ProcessorTask(BaseTask):
    """Processing with metrics."""
    
    def __init__(self, collector: Collector):
        super().__init__(name="processor")
        self.collector = collector
    
    def process(self) -> None:
        ctx = self.inputs[0].get()
        if ctx is None:
            return
        
        with self.collector.duration_timer("processor.process_ms"):
            frames = ctx.get_all("frame")
            for frame in frames:
                # Process
                processed = frame // 2
                
                # Track processed image
                obj_id = id(processed)
                self.collector.record("memory.track", processed.nbytes, attributes={
                    "type": "numpy.ndarray",
                    "obj_id": str(obj_id),
                    "stage": "processed"
                })
                mem_inst = self.collector.get_instrument("memory.images")
                if mem_inst:
                    mem_inst.track_object(processed, obj_id)
                
                ctx.append("processed", processed)
                self.collector.record("processor.frames", 1)
        
        for buf in self.outputs:
            buf.put(ctx)


class SinkTask(BaseTask):
    """Sink with metrics."""
    
    def __init__(self, collector: Collector):
        super().__init__(name="sink")
        self.collector = collector
    
    def process(self) -> None:
        ctx = self.inputs[0].get()
        if ctx is None:
            return
        
        with self.collector.duration_timer("sink.consume_ms"):
            frames = ctx.get_all("processed")
            for _ in frames:
                self.collector.record("sink.frames", 1)


def main():
    """Run pipeline with integrated metrics and memory monitoring."""
    
    # Create metrics collector
    collector = Collector("pipeline")
    
    # Add instruments
    collector.add_instrument(
        AvgDurationInstrument("camera.capture_avg"),
        "camera.capture_ms"
    )
    collector.add_instrument(
        AvgDurationInstrument("processor.process_avg"),
        "processor.process_ms"
    )
    collector.add_instrument(
        CounterInstrument("camera.frame_count"),
        "camera.frames"
    )
    collector.add_instrument(
        CounterInstrument("processor.frame_count"),
        "processor.frames"
    )
    
    # Add memory instrument (this is the key integration!)
    mem_instrument = MemoryInstrument("memory.images", leak_threshold_seconds=5.0)
    collector.add_instrument(mem_instrument, "memory.track")
    
    # Store reference so tasks can access it
    collector._mem_instrument = mem_instrument
    def get_instrument(name):
        if name == "memory.images":
            return collector._mem_instrument
        return None
    collector.get_instrument = get_instrument
    
    # Create pipeline
    camera = CameraTask(collector)
    processor = ProcessorTask(collector)
    sink = SinkTask(collector)
    
    # Wire buffers
    buf1 = Buffer(size=5, policy=drop_oldest_policy, name="camera→proc")
    buf2 = Buffer(size=5, policy=drop_oldest_policy, name="proc→sink")
    
    camera.outputs = [buf1]
    processor.inputs = [buf1]
    processor.outputs = [buf2]
    sink.inputs = [buf2]
    
    # Create runner
    runner = Runner([camera, processor, sink])
    
    print("Running pipeline with integrated metrics + memory monitoring...")
    print("="*60 + "\n")
    
    # Run for 3 seconds
    start = time.time()
    iteration = 0
    while time.time() - start < 3.0:
        runner.run_once()
        time.sleep(0.01)
        
        # Report every second
        iteration += 1
        if iteration % 100 == 0:
            print(f"\n--- Iteration {iteration} ({time.time() - start:.1f}s) ---")
            stats = collector.get_all_stats()
            
            # Performance metrics
            if "camera.capture_avg" in stats:
                s = stats["camera.capture_avg"]
                print(f"Camera capture: {s['avg']:.2f}ms avg, {s['count']} samples")
            
            if "camera.frame_count" in stats:
                print(f"Frames: {stats['camera.frame_count']['count']}")
            
            # Memory metrics
            if "memory.images" in stats:
                mem = stats["memory.images"]
                print(f"\nMemory:")
                print(f"  Alive: {mem['total_alive']} objects")
                print(f"  Tracked: {mem['total_tracked']}")
                print(f"  Cleaned: {mem['total_cleaned']}")
                print(f"  By type: {mem['by_type']}")
                
                if mem['potential_leaks']:
                    print(f"\n  ⚠️  LEAKS DETECTED:")
                    for leak in mem['potential_leaks']:
                        print(f"    {leak['type']}: {leak['count']} objects, "
                              f"{leak['total_size_bytes']/1024:.1f} KB, "
                              f"max age {leak['max_age_sec']:.1f}s")
    
    # Final report
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    
    stats = collector.get_all_stats()
    
    # Performance
    if "camera.capture_avg" in stats:
        s = stats["camera.capture_avg"]
        print(f"\nCamera Performance:")
        print(f"  Avg: {s['avg']:.2f}ms")
        print(f"  Min: {s['min']:.2f}ms")
        print(f"  Max: {s['max']:.2f}ms")
        print(f"  Samples: {s['count']}")
    
    # Throughput
    if "camera.frame_count" in stats:
        print(f"\nThroughput:")
        print(f"  Total frames: {stats['camera.frame_count']['count']}")
    
    # Memory
    if "memory.images" in stats:
        mem = stats["memory.images"]
        print(f"\nMemory:")
        print(f"  Total tracked: {mem['total_tracked']}")
        print(f"  Total cleaned: {mem['total_cleaned']}")
        print(f"  Still alive: {mem['total_alive']}")
        
        if mem['potential_leaks']:
            print(f"\n  ⚠️  LEAKS DETECTED: {len(mem['potential_leaks'])} types")
            for leak in mem['potential_leaks']:
                print(f"    - {leak['type']}: {leak['count']} objects")
                print(f"      Size: {leak['total_size_bytes']/1024:.1f} KB")
                print(f"      Max age: {leak['max_age_sec']:.1f}s")
        else:
            print("  ✅ No leaks detected!")
    
    runner.shutdown()
    print("\nDone!")


if __name__ == "__main__":
    main()
