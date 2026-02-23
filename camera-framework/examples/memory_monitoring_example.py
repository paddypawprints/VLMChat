"""Example: Using memory monitoring in production.

Shows how to integrate weakref-based memory leak detection.
"""

import time
import numpy as np
from camera_framework import (
    Context,
    BaseTask,
    Runner,
    Buffer,
    drop_oldest_policy,
    memory_monitor,
    track_context,
    track_image,
)


class CameraTask(BaseTask):
    """Camera source task with memory tracking."""
    
    def __init__(self):
        super().__init__(name="camera", interval=0.033)  # ~30fps
        self.frame_count = 0
    
    def process(self) -> None:
        # Create frame
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Track large image in memory monitor
        track_image(frame, size_bytes=frame.nbytes, metadata={
            "frame": self.frame_count,
            "source": "camera"
        })
        
        # Create context
        ctx = Context()
        ctx.append("frame", frame)
        
        # Track context lifetime
        track_context(ctx, metadata={
            "frame": self.frame_count,
            "task": self.name
        })
        
        # Write to output
        for buf in self.outputs:
            buf.put(ctx)
        
        self.frame_count += 1


class ProcessorTask(BaseTask):
    """Processing task with memory tracking."""
    
    def __init__(self):
        super().__init__(name="processor")
    
    def process(self) -> None:
        # Read from input
        ctx = self.inputs[0].get()
        if ctx is None:
            return
        
        # Process frame
        frames = ctx.get_all("frame")
        for frame in frames:
            processed = frame // 2  # Simple processing
            
            # Track processed image
            track_image(processed, size_bytes=processed.nbytes, metadata={
                "stage": "processed",
                "task": self.name
            })
            
            ctx.append("processed", processed)
        
        # Write to output
        for buf in self.outputs:
            buf.put(ctx)


class SinkTask(BaseTask):
    """Sink task that consumes data."""
    
    def __init__(self):
        super().__init__(name="sink")
        self.frame_count = 0
    
    def process(self) -> None:
        ctx = self.inputs[0].get()
        if ctx is None:
            return
        
        # Consume data (simulate display/save)
        frames = ctx.get_all("processed")
        self.frame_count += len(frames)
        
        # Context and images will be GC'd when this function returns
        # (assuming no other references exist)


def main():
    """Run pipeline with memory monitoring."""
    
    # Create pipeline
    camera = CameraTask()
    processor = ProcessorTask()
    sink = SinkTask()
    
    # Wire up buffers
    buf1 = Buffer(size=5, policy=drop_oldest_policy, name="camera→proc")
    buf2 = Buffer(size=5, policy=drop_oldest_policy, name="proc→sink")
    
    camera.outputs = [buf1]
    processor.inputs = [buf1]
    processor.outputs = [buf2]
    sink.inputs = [buf2]
    
    # Create runner
    runner = Runner([camera, processor, sink])
    
    print("Running pipeline with memory monitoring...")
    print("Watch for leak warnings in logs.\n")
    
    # Run for 10 seconds
    start = time.time()
    iteration = 0
    while time.time() - start < 10.0:
        runner.run_once()
        time.sleep(0.01)
        
        # Periodic memory report (every 100 iterations ~1 second)
        iteration += 1
        if iteration % 100 == 0:
            memory_monitor.report()
    
    # Final report
    print("\n" + "="*60)
    print("FINAL MEMORY REPORT")
    print("="*60)
    stats = memory_monitor.check()
    print(f"Total tracked: {stats['total_tracked']}")
    print(f"Total cleaned: {stats['total_cleaned']}")
    print(f"Still alive: {stats['total_alive']}")
    
    if stats['potential_leaks']:
        print(f"\n⚠️  POTENTIAL LEAKS DETECTED: {len(stats['potential_leaks'])}")
        for leak in stats['potential_leaks']:
            print(f"  - {leak['type']}: {leak['count']} objects")
            print(f"    Max age: {leak['max_age_sec']:.1f}s")
    else:
        print("\n✅ No leaks detected - all objects cleaned up properly!")
    
    print(f"\nFrames processed: {sink.frame_count}")
    
    runner.shutdown()


if __name__ == "__main__":
    main()
