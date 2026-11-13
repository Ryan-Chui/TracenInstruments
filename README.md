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
three-step phase shifting with single-shot coded-dot projection. The algorithm
implementations are separated into
[phase_shifting.py](phase_shifting.py) and
[coded_dots.py](coded_dots.py) for reuse by future camera/projector data
streams. The coded-dot decoder uses one camera image and the known projected
pattern to recover camera/projector correspondences; it does not require a
second camera.

```powershell
python .\structured_light_harness.py
```

For real 3D measurements, calibrate the camera and projector and triangulate
the returned correspondences. The synthetic coded-dot test uses a planar
homography only to evaluate matching quality.

When the projector is available, the hardware mode loads the saved
camera/projector calibration. The default `dots` algorithm captures one ELP
frame:

```powershell
python .\structured_light_harness.py --hardware `
  --camera 1 --width 1280 --height 720 `
  --projector-width 1280 --projector-height 720 `
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

The default coded-dot pattern uses 20-pixel spacing, 6-pixel radius, and 65%
cell occupancy to increase point-cloud density. The
decoder uses SIFT when available (with ORB fallback), which is more tolerant of
projector-to-camera scale and blur. These
projector dots are still separated enough for the current camera scale. If the
diagnostic output reports weak camera features, increase the radius or reduce
density, for example:

```powershell
python .\structured_light_harness.py --hardware --algorithm dots `
  --dot-spacing 24 --dot-radius 8 --dot-density 0.55 `
  --output .\data\point_clouds\dots.ply
```

Each dots capture also saves `dots_camera_capture.png` and
`dots_projector_pattern.png` beside the requested output. The console reports
projector and camera feature counts, ratio-test matches, and mutual matches;
these counts identify whether the problem is dot visibility or descriptor
matching.

To reject wall points when the ROI border contains only the background, use:

```powershell
python .\structured_light_harness.py --hardware --algorithm dots `
  --foreground-only --background-depth-tolerance 0.05 `
  --output .\data\point_clouds\dots_foreground.ply
```

The harness fits a sloped background plane from points near the ROI border and
keeps only points at least 5 cm closer to the local background depth. This
handles walls that appear at different depths on the left and right sides of
the image. Keep a visible strip of wall around the subject when selecting the
ROI.

The hardware run writes valid triangulated positions to
`data/point_clouds/coded_dots.ply` by default. The PLY contains geometry only,
without color. Override the destination and depth filter with `--output`,
`--min-depth`, and `--max-depth`. It can be opened in CloudCompare, MeshLab,
Blender, or Open3D. Synthetic mode evaluates decoding only; it does not
produce a meaningful 3D point cloud because it has no physical calibration.

To run the physical three-step phase path, capture each projected phase image
when prompted:

```powershell
python .\structured_light_harness.py --hardware --algorithm phase `
  --period 24 --output .\data\point_clouds\phase.ply
```

The phase path captures twelve real camera frames: coarse and fine sets of
three horizontal and three vertical phase shifts. The coarse full-frame phase
sets the absolute location, while the default 64-pixel fine phase improves
depth precision. Adjust the fine periods with `--phase-fine-period-x` and
`--phase-fine-period-y`.

For an experimental two-shot RGB color-stripe scan, use:

```powershell
python .\structured_light_harness.py --hardware --algorithm color_stripes `
  --output .\data\point_clouds\color_stripes.ply
```

The RGB channels carry three phase shifts in each image. The first image
varies along projector X and the second along projector Y, producing dense
two-dimensional projector coordinates with only two camera frames. This
requires the color ELP camera and is more geometrically complete than the
single-axis version, though it remains more sensitive to color-channel
calibration and motion than monochrome multi-frame phase shifting.

The color-stripe decoder applies robust per-channel normalization to reduce
camera white-balance and projector color-gain errors. It also rejects isolated
depth spikes using a local 5x5 depth-consistency check. Adjust
`--depth-outlier-tolerance` if valid sharp edges are being removed or noise
remains.

Color-stripe output also applies a robust Z-depth band around the median
foreground depth. The default band is ±20 cm; reduce it with
`--depth-band-tolerance 0.10` to suppress stronger depth smearing, or increase
it if the subject's valid depth range is being clipped.

To keep only points in front of a wall or other background surface, use the
same border-based depth filter as the other hardware modes:

```powershell
python .\structured_light_harness.py --hardware --algorithm color_stripes `
  --foreground-only --background-depth-tolerance 0.05 `
  --output .\data\point_clouds\color_stripes_foreground.ply
```

Leave a strip of background visible around the subject inside the selected
ROI so the background depth can be estimated.

By default, hardware mode first shows the live camera frame and lets you drag
a rectangle around the scan area. Press Enter or Space to accept it; press C
to cancel. No face detection is used. To skip selection and scan the complete
camera frame, use `--full-frame`.

For scripted runs, apply a camera-image rectangle `X Y WIDTH HEIGHT`:

```powershell
python .\structured_light_harness.py --hardware --algorithm phase `
  --camera-roi 250 100 700 520 `
  --output .\data\point_clouds\phase_roi.ply
```

The same option works with `--algorithm dots`. Coordinates refer to the
1280x720 camera image, with `(0, 0)` at the upper-left.

For physical coded-dot scans, ROI masking is applied before ORB feature
matching. The planar homography rejection used by the synthetic test is
disabled because a 3D face is not a single plane; mutual descriptor matching
and the ROI still reject unrelated features.

Phase decoding also reconstructs illuminated background surfaces by design.
For a mostly planar background around the ROI, remove points near its
estimated depth with:

```powershell
python .\structured_light_harness.py --hardware --algorithm phase `
  --background-depth-tolerance 0.03
```

The tolerance is in meters. This uses the ROI border as the background
estimate; keep some background visible around the object and do not use it
when the border contains the object.

For a face or other object in front of the background, discard all points on
the background side of that plane:

```powershell
python .\structured_light_harness.py --hardware --algorithm phase `
  --foreground-only --background-depth-tolerance 0.03 `
  --min-depth 0.30 --max-depth 1.20
```

Depth `z` is measured forward from the camera. Use `--min-depth` and
`--max-depth` to reject physically impossible near/far points for your setup.

Phase output keeps every valid camera pixel by default. If processing or file
size becomes excessive, use `--phase-sample-step 2` or `--phase-sample-step 4`
to decimate the point cloud.

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
