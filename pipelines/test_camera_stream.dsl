@camera: camera_source(device=0, fps=5, buffer_size=100)

wait(source_name="camera") -> latest(source_name="camera") -> diagnostic(message="Frame captured") -> diagnostic(message="Processing complete")
