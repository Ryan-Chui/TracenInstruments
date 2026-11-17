"""facesim: synthetic structured-light face scans from FLAME, plus FR3DK loading.

Modules
-------
flame       FLAME head model in plain NumPy (load .pkl, sample, forward pass)
raster      pinhole camera + z-buffer rasterizer used for ray casting
rig         camera + DLP projector geometry (matches OpenCV stereo calibration)
patterns    Gray-code / phase-shift patterns and the decoder (works on real captures too)
scanner     the virtual scanner: render pattern images, decode, triangulate
scene       place a face (or a printed-photo spoof) in front of the rig
pointcloud  PLY/PCD/NPZ I/O, outlier removal, nose-tip crop, resampling
fr3dk       FR3DK dataset reader, index and summary
datasets    PyTorch Dataset wrappers (optional, needs torch)
generate    command-line dataset generator
"""

__version__ = "0.1.0"
