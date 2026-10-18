import json
import os
import time
import cv2
import numpy as np

from scanner.acquisition import Camera, ProjectorWindow
from scanner.acquisition.projector import place_window
from scanner.grayCode.generate_codes import get_gray_codes, get_image_sequence
from scanner.grayCode.decode_codes import decode_sequence, combine_codes, codes_to_positions, remove_inconsistent
from scanner.utils import visualize

# Settings
# Camera: same as in 1-calibrate_camera.py and 2-calibrate_projcam.py
cam_src = 0
cam_width, cam_height = (1920, 1080)
cam_fps = 30

# Projector display: None = the display that isn't your Mac's built-in screen (same as part 2)
proj_screen_index = None

# Capture the whole pattern sequence this many times and merge them (fills in pixels one pass missed).
# 1 halves the scan time.
runs = 2

# Seconds to wait after showing a pattern before taking the photo. Raise it if the decoded image
# looks scrambled (the camera was still seeing the previous pattern).
wait_time = 0.3

# Minimum brightness difference (0-255) between a lit and an unlit stripe for a pixel to be trusted
min_contrast = 10

output_root = './data/recordings/'



def fresh_frame(cam, timeout=3.0):
    """A camera frame taken after this call (the pending one may have started before the pattern changed)."""
    cam.get_frame()
    t0 = time.time()
    while time.time() - t0 < timeout:
        frame = cam.get_frame()
        if frame is not None:
            return frame.copy()
        cv2.waitKey(2)
    raise RuntimeError('The camera stopped sending frames.')


def wait(seconds):
    """Keep the windows responsive while waiting; returns True if q was pressed."""
    t0 = time.time()
    while time.time() - t0 < seconds:
        if (cv2.waitKey(5) & 0xFF) == ord('q'):
            return True
    return False


if __name__ == '__main__':
    # Window covering the projector; the patterns are generated at its real resolution
    projector = ProjectorWindow('Projector', screen_index=proj_screen_index)
    proj_width, proj_height = projector.width, projector.height
    gray_codes = get_gray_codes(proj_width, proj_height)
    image_seq = get_image_sequence(gray_codes, proj_width, proj_height)
    seq_len = len(image_seq)
    print(f'Projector resolution: {proj_width}x{proj_height}, {seq_len} patterns x {runs} run(s), '
          f'about {seq_len * runs * (wait_time + 0.1):.0f} s. Keep the scene and the setup still during the scan.')

    # Camera window on your own screen
    cv2.namedWindow('Camera', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Camera', np.zeros((540, 960, 3), dtype=np.uint8))
    cv2.waitKey(50)
    place_window('Camera', projector.control_display)

    cam = Camera(cam_src, width=cam_width, height=cam_height, fps=cam_fps)

    # Aim: project white and show the camera until space is pressed
    projector.show(image_seq[1])
    print('Aim the scanner at the scene, then press space in the Camera window to scan (q = quit).')
    while True:
        frame = cam.get_frame()
        if frame is not None:
            preview = cv2.resize(frame, (960, 540))
            cv2.putText(preview, 'space = scan, q = quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Camera', preview)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            cam.stop_cam()
            raise SystemExit
        if key == ord(' '):
            break

    # Recording folder
    os.makedirs(output_root, exist_ok=True)
    k = 0
    while os.path.exists(os.path.join(output_root, f'record_{k}')):
        k += 1
    output_path = os.path.join(output_root, f'record_{k}')
    os.makedirs(output_path)

    # Capture
    captures = []
    image_id = 0
    t_start = time.time()
    for run in range(runs):
        frames = None
        for seq_id, pattern in enumerate(image_seq):
            projector.show(pattern)
            if wait(wait_time * (3 if seq_id == 0 else 1)):
                cam.stop_cam()
                raise SystemExit('Scan cancelled.')
            frame = fresh_frame(cam)
            cv2.imwrite(os.path.join(output_path, f'frame_{image_id}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_id += 1
            if frames is None:
                frames = np.empty((seq_len, frame.shape[0], frame.shape[1]), dtype=np.uint8)
            frames[seq_id] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            preview = cv2.resize(frame, (960, 540))
            cv2.putText(preview, f'run {run + 1}/{runs}  pattern {seq_id + 1}/{seq_len}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Camera', preview)
        captures.append(frames)
    print(f'Captured in {time.time() - t_start:.0f} s')

    projector.close()
    cam.stop_cam()

    # Decode
    print('Decoding...')
    col_codes, row_codes = [], []
    for frames in captures:
        c, r = decode_sequence(frames, min_contrast=min_contrast)
        col_codes.append(c)
        row_codes.append(r)
    h_pixels = codes_to_positions(combine_codes(col_codes), proj_width)
    v_pixels = codes_to_positions(combine_codes(row_codes), proj_height)
    # Drop pixels that disagree with their neighbors (object outlines, shadow edges, noise)
    h_pixels = remove_inconsistent(h_pixels)
    v_pixels = remove_inconsistent(v_pixels)

    np.save(os.path.join(output_path, 'h_pixels.npy'), h_pixels)
    np.save(os.path.join(output_path, 'v_pixels.npy'), v_pixels)
    valid = (h_pixels >= 0) & (v_pixels >= 0)
    with open(os.path.join(output_path, 'scan.json'), 'w') as fp:
        json.dump(dict(proj_width=proj_width, proj_height=proj_height, cam_width=int(h_pixels.shape[1]),
                       cam_height=int(h_pixels.shape[0]), runs=runs, wait_time=wait_time, min_contrast=min_contrast,
                       decoded_fraction=float(valid.mean())), fp)
    print(f'Decoded {valid.mean() * 100:.0f}% of camera pixels. Saved to {output_path}')
    print('Next: run 4-triangulate.py (it uses the newest recording).')

    # Display decoded gray codes (smooth colors = good, black = not decoded, speckle = noisy)
    color_map_img = visualize.create_color_map(proj_width, proj_height)
    visualize.plot_decoded_graycodes(h_pixels, v_pixels, color_map_img, save_to=os.path.join(output_path, 'decoded.png'))
