"""faceid: 3D face recognition with spoof rejection on structured-light point clouds.

Run from the generate_data folder, in this order:

    python -m faceid.cache    data/synth_v1                # 1. preprocess once
    python -m faceid.train    data/synth_v1 --out runs/edgeconv   # 2. train
    python -m faceid.evaluate runs/edgeconv                # 3. thresholds + report
    python -m faceid.gallery  runs/edgeconv                # 4. enroll / identify demo

Modules
-------
cache     preprocess every scan once into one .npz (+ the training Dataset)
model     point-cloud encoders, embedding + liveness heads, CosFace loss
train     training loop, checkpoint picked by validation-people accuracy at low FAR
evaluate  score distributions, threshold selection, open-set identification report
gallery   FaceID class for the real system: enroll people, identify a scan
"""
