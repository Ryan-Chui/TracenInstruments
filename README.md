This program was updated from [this repository](https://github.com/guillaume-charron/3DScanner-GrayCode)
to work for my specific usecase. That being, on a mac machine.
Note: It requires a python version in a specific range. I used 3.10.22. 

# Structured Light 3D Scanner

<p float="left">
  <img src="data/point_clouds/Groot/720/360view.gif" width="32%" />
  <img src="data/point_clouds/Bouda/600/360view.gif" width="32%" /> 
  <img src="data/point_clouds/Elephant/360view.gif" width="32%" />
</p>

## Table of content
- [Installation](#installation)
- [Scanner module](#scanner-module)
- [Application usage](#application-usage)
- [Disclaimer](#disclaimer)
- [Citation](#citation)

This repo uses a projector and camera to scan an objet using structured light (Gray codes). The projected patterns are captured by the camera and the 3D points are triangulated using the calibration info of the stereo setup. The project was made as a final assignment for the IFT6145 class at UdeM.

## Installation

1. Make sure python is installed on your System
2. Install all dependencies using the `requirements.txt` file
3. Install scanner module by doing `pip install -e .` in the root of the repo
4. Follow the camera/projector calibration workflow in
   [calibration/README.md](calibration/README.md), then run the scan scripts
   in `src/`.

## Scanner module

The scanner module is used to open a camera with OpenCV, project, capture and decode Gray codes and triangulate the 3d points to make a point cloud.

Here's a list of the packages in this module:
- **acquisition** : Contains the Camera class to open a webcam on a separate thread
- **calibration** : Contains 2 classes
    1. **CameraCalibrator** : Used to calibrate the camera
    2. **ProjCamCalibrator** : Used to calibrate the projector and the stereo setup (projector + camera)
- **grayCode** : Functions to create the Gray code images and decode the captured frames
- **triangulation** : Triangulation class to compute the 3d points based on the camera and projector decoded points
- **utils** : Visualization functions for the decoding results and the 3d point cloud

## Application usage

The scan scripts are in `src/`: `3-capture_decode.py` captures Gray-code
patterns and `4-triangulate.py` reconstructs the point cloud. Camera and
projector calibration is documented separately in
[calibration/README.md](calibration/README.md).

The algorithm comparison harness is
[structured_light_harness.py](structured_light_harness.py). It compares
three-step phase shifting with single-shot coded-dot projection. The coded-dot
decoder uses one camera image and the known projected pattern to recover
camera/projector correspondences; it does not require a second camera.

```powershell
python .\structured_light_harness.py
```

For real 3D measurements, calibrate the camera and projector and triangulate
the returned correspondences. The synthetic coded-dot test uses a planar
homography only to evaluate matching quality.

When the projector is available, the hardware mode loads the saved
camera/projector calibration and captures one ELP frame:

```powershell
python .\structured_light_harness.py --hardware `
  --camera 1 --width 1920 --height 1080 `
  --projector-width 1920 --projector-height 1080 `
  --camera-calibration .\data\calib_results\elp_camera `
  --projector-calibration .\data\calib_results\my_proj `
  --stereo-calibration .\data\calib_results\my_stereo
```

Press `Space` in the camera window to capture. The calibration resolution
arguments must match the resolution used when the projector intrinsics were
calibrated; if the projector is currently run at another resolution, pass the
original dimensions with `--projector-calibration-width` and
`--projector-calibration-height`. The projector must be extended rather than
mirrored, and the camera/projector must not move after calibration.

The hardware run writes valid triangulated positions to
`data/point_clouds/coded_dots.ply` by default. The PLY contains geometry only,
without color. Override the destination and depth filter with `--output`,
`--min-depth`, and `--max-depth`. It can be opened in CloudCompare, MeshLab,
Blender, or Open3D. Synthetic mode evaluates decoding only; it does not
produce a meaningful 3D point cloud because it has no physical calibration.

## Disclaimer

This repository is for educational purposes only. The code and algorithm implemented are derived from the original authors' repositories and papers. All rights are reserved to the original authors of the papers and repositories this code is based on.

## Citation
This code is based on the implementation that can be found in this [repository](https://github.com/sjnarmstrong/gray-code-structured-light/tree/master?fbclid=IwAR3WoIdu-mVJEMWzUqAHYBba239Xw2OCyVLBepdD05pllM9HTA3PPwxCibw) and on this [tutorial](https://www.morethantechnical.com/blog/2017/11/17/projector-camera-calibration-the-easy-way/).
If you find this work useful, please ensure that you cite the original authors:
```
@INPROCEEDINGS{7781767,
  author={Yang, Liming and Normand, Jean-Marie and Moreau, Guillaume},
  booktitle={2016 IEEE International Symposium on Mixed and Augmented Reality (ISMAR)}, 
  title={Practical and Precise Projector-Camera Calibration}, 
  year={2016},
  volume={},
  number={},
  pages={63-70},
  doi={10.1109/ISMAR.2016.22}
}

@inproceedings{3D-Modeling,
	author = {Xu, Yi and Aliaga, Daniel G.},
	title = {Robust Pixel Classification for 3D Modeling with Structured Light},
	booktitle = {Proceedings of Graphics Interface 2007},
	series = {GI '07},
	year = {2007},
	isbn = {978-1-56881-337-0},
	location = {Montreal, Canada},
	pages = {233--240},
	numpages = {8},
	acmid = {1268556},
	publisher = {ACM},
	address = {New York, NY, USA},
	keywords = {3D reconstruction, direct and global separation, structured light},
} 

@INPROCEEDINGS{calib, 
	author={D. Moreno and G. Taubin}, 
	booktitle={2012 Second International Conference on 3D Imaging, Modeling, Processing, Visualization Transmission}, 
	title={Simple, Accurate, and Robust Projector-Camera Calibration}, 
	year={2012}, 
	volume={}, 
	number={}, 
	pages={464-471}, 
	doi={10.1109/3DIMPVT.2012.77}, 
	ISSN={1550-6185}, 
	month={Oct},}

@inproceedings{NayarSeperation,
	author = {Nayar, Shree K. and Krishnan, Gurunandan and Grossberg, Michael D. and Raskar, Ramesh},
	title = {Fast Separation of Direct and Global Components of a Scene Using High Frequency Illumination},
	booktitle = {ACM SIGGRAPH 2006 Papers},
	series = {SIGGRAPH '06},
	year = {2006},
	isbn = {1-59593-364-6},
	location = {Boston, Massachusetts},
	pages = {935--944},
	numpages = {10},
	acmid = {1141977},
	publisher = {ACM},
	address = {New York, NY, USA},
	keywords = {coded illumination, direct illumination, global illumination, image decomposition, image manipulation, interreflections, subsurface scattering, translucency, volumetric scattering},
} 
```
