import glob
import json
import os
import cv2
import numpy as np


def infer_projector_size(image_folder):
    """
    Work out the projector resolution of a calibration session from where the dot grid was placed:
    the grid visits a 3x3 set of positions spread from one margin of the projector image to the other.
    Returns (width, height, circle_r), or None if the right column or bottom row of positions was never photographed.
    """
    tops, radii = set(), []
    for path in glob.glob(os.path.join(image_folder, '*_proj.npy')):
        pts = np.load(path).reshape(-1, 2)
        tops.add((int(round(pts[:, 0].min())), int(round(pts[:, 1].min()))))
        radii.append((pts[:, 1].max() - pts[:, 1].min()) / 30.0)   # 11 rows, 3 radii apart
    if not radii:
        return None
    r = int(round(float(np.median(radii))))
    margin = 3 * r + 20
    xs = sorted({x for x, _ in tops})
    ys = sorted({y for _, y in tops})
    if len(xs) < 3 or len(ys) < 3 or xs[0] != margin or ys[0] != margin:
        return None
    width, height = xs[-1] + 21 * r + margin, ys[-1] + 30 * r + margin
    if max(6, int(round(15 * height / 1080))) != r:
        return None
    return width, height, r


def calibration_projector_size(image_folder):
    """
    Projector resolution used for a calibration session: from its projector.json, else worked out from
    the saved dot positions. Returns (width, height, circle_r, source) or None.
    """
    session_file = os.path.join(image_folder, 'projector.json')
    if os.path.exists(session_file):
        with open(session_file) as fp:
            s = json.load(fp)
        return s['width'], s['height'], s.get('circle_r', 15), 'the capture session'
    guess = infer_projector_size(image_folder)
    if guess is not None:
        return guess + ('the saved dot positions',)
    return None


class ProjCamCalibrator(object):
    def __init__(self,
                 cam_width=1920,
                 cam_height=1080,
                 cam_mtx=None,
                 cam_dist=None,
                 proj_width=1920,
                 proj_height=1080,
                 proj_mtx=None,
                 proj_dist=None,
                 circle_grid_size = (4, 11),
                 circle_r=15,
                 board_offset_m=0.0):

        # Define the charuco board parameters
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        # calib.io board: 7 columns x 5 rows, 25 mm squares, 18 mm markers, DICT_6X6
        self.charuco_board = cv2.aruco.CharucoBoard((7, 5), 0.025, 0.018, self.dictionary)
        self.params = cv2.aruco.DetectorParameters()
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

        # Marker id -> its 4 corners on the board plane (x, y), used for the image -> board homography
        board_ids = np.array(self.charuco_board.getIds()).flatten()
        board_obj = np.array(self.charuco_board.getObjPoints(), dtype=np.float32)
        self.marker_obj_xy = {int(i): board_obj[k, :, :2] for k, i in enumerate(board_ids)}

        # Camera parameters (image size and intrinsic parameters)
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.cam_mtx = cam_mtx
        self.cam_dist = cam_dist

        # Projector parameters (image size and intrinsic parameters)
        self.proj_width = proj_width
        self.proj_height = proj_height
        self.proj_mtx = proj_mtx
        self.proj_dist = proj_dist

        # Generate circle grid
        self.circle_grid_size = circle_grid_size
        self.circle_r = circle_r
        self.circle_2d_pts = self.build_circle_grid_pts(circle_grid_size, circle_r)
        self.circle_2d = None

        # How far the printed ChArUco sits above the white surface the dots land on (meters).
        # Positive = ChArUco raised toward the camera, e.g. printed on a separate backing laid on the board.
        self.board_offset_m = float(board_offset_m)

    def _undistort(self, pts):
        """Remove camera lens distortion from pixel points, returning ideal pixel coordinates."""
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        if self.cam_mtx is None or self.cam_dist is None:
            return pts
        return cv2.undistortPoints(pts, self.cam_mtx, self.cam_dist, P=self.cam_mtx)

    def _offset_plane_homography(self, H):
        """
        Image -> board homography for the white surface, which is parallel to the ChArUco
        but board_offset_m farther from the camera. H is the image -> ChArUco-plane homography.
        """
        K = self.cam_mtx
        M = np.linalg.inv(K) @ np.linalg.inv(H)          # proportional to [r1 r2 t]
        scale = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
        r1, r2, t = M[:, 0] * scale, M[:, 1] * scale, M[:, 2] * scale
        if t[2] < 0:                                     # the board is in front of the camera
            r1, r2, t = -r1, -r2, -t
        r3 = np.cross(r1, r2)
        r3 /= np.linalg.norm(r3)
        # Shift the plane along its normal, to the side facing away from the camera
        z = np.sign(r3 @ t) * self.board_offset_m
        H_off = K @ np.column_stack([r1, r2, t + z * r3])  # white-surface (x, y) -> image
        return np.linalg.inv(H_off)

    def detect_markers(self, frame, gray,draw=True):
        """
        Detect markers and corners of the charuco board in the image

        Parameters:
        ---------
        frame : Mat
            image to draw detected markers
        gray : Mat
            gray version of the image
        draw : bool, optional (default = True)
            draw detected markers or not

        Returns:
        -------
        ret : bool
            True if at least one marker and the board were detected
        frame : Mat
            Image with detected markers
        H : Mat
            Homography matrix of the camera to the charuco board
        """
        ret = False
        H = None
        # SUB PIXEL CORNER DETECTION CRITERION
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        # Detect markers and corners
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)

        # Keep only markers that belong to this board (a misread marker id would otherwise crash the lookup below)
        if ids is not None:
            keep = [k for k, i in enumerate(ids.flatten()) if int(i) in self.marker_obj_xy]
            corners = tuple(corners[k] for k in keep)
            ids = ids[keep] if keep else None

        # Need several markers for a reliable homography onto the board plane
        if ids is not None and len(ids) >= 4:
            if draw:
                frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for corner in corners:
                cv2.cornerSubPix(gray, corner, (3,3), (-1,-1), criteria)

            n, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.charuco_board)
            if charuco_ids is not None and len(charuco_ids) >= 6:
                if draw:
                    frame = cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

                # Board pose from the ChArUco corners and marker corners, using the camera calibration.
                # (A free homography fitted to the markers alone is unstable when extended onto the dot area.)
                pose = self._board_pose(charuco_corners, charuco_ids, corners, ids)
                if pose is not None:
                    rvec, tvec = pose
                    if draw:
                        frame = cv2.drawFrameAxes(frame, self.cam_mtx, self.cam_dist, rvec, tvec, 0.1)
                    # Homography from (undistorted) camera pixels to board-plane coordinates in meters
                    R, _ = cv2.Rodrigues(rvec)
                    H = np.linalg.inv(self.cam_mtx @ np.column_stack([R[:, 0], R[:, 1], tvec.ravel()]))
                    ret = True

        return ret, frame, H

    def _board_pose(self, charuco_corners, charuco_ids, marker_corners, marker_ids):
        """Board pose (rvec, tvec) from ChArUco and marker corners; drops corners that disagree and re-solves once."""
        board_corners = np.array(self.charuco_board.getChessboardCorners(), dtype=np.float32)
        obj = [board_corners[charuco_ids.flatten()]]
        img = [charuco_corners.reshape(-1, 2)]
        for c, i in zip(marker_corners, marker_ids.flatten()):
            obj.append(np.c_[self.marker_obj_xy[int(i)], np.zeros(4)])
            img.append(c.reshape(4, 2))
        obj = np.concatenate(obj).astype(np.float32)
        img = np.concatenate(img).astype(np.float32)

        rvec = tvec = None
        for _ in range(2):
            if len(obj) < 6:
                return None
            ok, rvec, tvec = cv2.solvePnP(obj, img, self.cam_mtx, self.cam_dist, flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                return None
            rvec, tvec = cv2.solvePnPRefineLM(obj, img, self.cam_mtx, self.cam_dist, rvec, tvec)
            proj, _ = cv2.projectPoints(obj, rvec, tvec, self.cam_mtx, self.cam_dist)
            err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
            good = err < max(3.0 * float(np.median(err)), 1.5)
            if good.all():
                break
            obj, img = obj[good], img[good]
        if tvec[2, 0] <= 0:
            return None
        return rvec, tvec

    def detect_circle_grid(self, frame, gray, H, draw=True):
        """
        Detect the circle grid in the image

        Parameters:
        ---------
        frame : Mat
            image to draw detected circles grid
        gray : Mat
            gray version of the image
        H : Mat
            Homography matrix of the camera to the charuco board
        draw : bool, optional (default = True)
            draw detected circles or not

        Returns:
        -------
        ret : bool
            True if the circle grid was detected
        frame : Mat
            Image with detected circles
        circles : Mat
            2D coordinates of the detected circles in the camera
        circles3D : Mat
            3D coordinates of the detected circles in the charuco board reference frame (z=0)
        """
        ret, circles = cv2.findCirclesGrid(gray, self.circle_grid_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)

        circles3D = None
        if ret:
            if draw:
                frame = cv2.drawChessboardCorners(frame, self.circle_grid_size, circles, ret)

            # Apply homography to get the 3D coordinates of the circles in the charuco board reference frame
            # (when the ChArUco is raised, map onto the white surface's own plane instead)
            H_plane = self._offset_plane_homography(H) if self.board_offset_m else H
            circles3D = cv2.perspectiveTransform(self._undistort(circles), H_plane)
            # Add z=0 to the 3D coordinates
            circles3D = np.pad(circles3D, ((0,0), (0,0), (0,1)), 'constant', constant_values=0)
            circles3D = circles3D.astype(np.float32)

        return ret, frame, circles, circles3D

    def build_circle_grid_pts(self, grid_size, circle_r):
        """
        Build the 2D coordinates of the circle grid

        Parameters:
        ---------
        grid_size : tuple
            Number of circles in the grid (nb_col, nb_row)
        circle_r : int
            Radius of the circles

        Returns:
        -------
        circle_2d_pts : Mat
            2D coordinates of the circles in the grid
        """
        nb_col, nb_row = grid_size
        circle_2d_pts = np.zeros((nb_col*nb_row, 2), dtype=np.int32)
        count = 0
        for i in range(nb_row-1, -1, -1):
            for j in range(nb_col-1, -1, -1):
                if i % 2 == 0:
                    pos_x = j * 6 * circle_r + (3 * circle_r)
                else:
                    pos_x = j * 6 * circle_r
                pos_y = i * 3 * circle_r
                circle_2d_pts[count] = [pos_x, pos_y]
                count += 1
        return circle_2d_pts

    def grid_positions(self, cols=3, rows=3, margin=None):
        """
        Top-left positions that spread the circle grid across the whole projector image,
        so the projector calibration sees every region of the projector (not one small patch).
        """
        pad = 3 * self.circle_r
        margin = pad + 20 if margin is None else margin
        grid_w = int(self.circle_2d_pts[:, 0].max())
        grid_h = int(self.circle_2d_pts[:, 1].max())
        xs = np.linspace(margin, self.proj_width - grid_w - margin, cols).astype(int)
        ys = np.linspace(margin, self.proj_height - grid_h - margin, rows).astype(int)
        return [(int(x), int(y)) for y in ys for x in xs]

    def get_circle_grid_image(self, x, y):
        """
        Return an image of the circle grid

        Parameters:
        ---------
        x : int
            x position of the grid
        y : int
            y position of the grid

        Returns:
        -------
        img_circle : Mat
            Image of the circle grid (3-channel BGR, black background)
        """
        img_circle = np.zeros((self.proj_height, self.proj_width, 3), dtype=np.uint8)
        self.circle_2d = self.circle_2d_pts + [x, y]
        white_padding = 3*self.circle_r
        min_x = max(int(np.min(self.circle_2d[:,0])) - white_padding, 0)
        max_x = min(int(np.max(self.circle_2d[:,0])) + white_padding, self.proj_width)
        min_y = max(int(np.min(self.circle_2d[:,1])) - white_padding, 0)
        max_y = min(int(np.max(self.circle_2d[:,1])) + white_padding, self.proj_height)
        img_circle[min_y:max_y, min_x:max_x] = 255
        for c in self.circle_2d:
            img_circle = cv2.circle(img_circle, tuple(c.astype(np.int32)), self.circle_r, (0,0,0), cv2.FILLED, cv2.LINE_AA)
        return img_circle

    @staticmethod
    def proj_points_path(image_path):
        """Side file that stores where the circle grid was in the projector image for one photo."""
        return os.path.splitext(image_path)[0] + '_proj.npy'

    def calibrate(self, image_folder, calibrate_proj=True):
        """
        Calibrate the camera and the projector using the charuco board and the circle grid

        Parameters:
        ----------
        image_folder : str
            path to the folder containing the calibration images
        calibrate_proj : bool, optional (default = True)
            calibrate the projector or not

        Returns:
        -------
        proj_mtx : Mat
            projector intrinsic parameters matrix
        proj_dist : Mat
            projector distortion coefficients
        proj_R : Mat
            rotation matrix between the camera and the projector
        proj_T : Mat
            translation vector between the camera and the projector
        R1 : Mat
            rectification transform (rotation matrix) for the camera
        R2 : Mat
            rectification transform (rotation matrix) for the projector
        P1 : Mat
            projection matrix in the new (rectified) coordinate systems for the camera
        P2 : Mat
            projection matrix in the new (rectified) coordinate systems for the projector
        Q : Mat
            disparity-to-depth mapping matrix
        """
        if self.cam_mtx is None or self.cam_dist is None:
            raise ValueError('Camera intrinsics are missing. Run 1-calibrate_camera.py first.')

        names = []
        proj_obj_pts = []
        proj_circle_pts = []
        cam_circle_pts = []

        # Calibrate projector-camera stereo setup
        images = sorted(glob.glob(os.path.join(image_folder,'*.jpg')))
        for fname in images:
            # Where the circle grid was in the projector image when this photo was taken
            side = self.proj_points_path(fname)
            if os.path.exists(side):
                proj_pts = np.load(side).astype(np.float32).reshape(-1, 1, 2)
            elif self.circle_2d is not None:
                proj_pts = np.expand_dims(self.circle_2d.astype(np.float32), axis=1)
            else:
                print('Skipping (no projector position saved):', fname)
                continue

            img = cv2.imread(fname)
            # Convert to gray
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect markers and corners
            ret, img, H = self.detect_markers(img, gray, draw=False)

            if ret:
                # Find projected circles
                ret, img, circles_2d_cam, circles_3d = self.detect_circle_grid(img, gray, H, draw=False)
            if ret:
                names.append(os.path.basename(fname))
                proj_obj_pts.append(circles_3d)
                proj_circle_pts.append(proj_pts)
                cam_circle_pts.append(circles_2d_cam)
            else:
                print('Bad image:', fname)

        print(f'\nUsing {len(proj_obj_pts)} of {len(images)} photos')
        if len(proj_obj_pts) < 5:
            raise ValueError('Need at least 5 good photos (aim for 15-25) before calibrating.')

        if not calibrate_proj and (self.proj_mtx is None or self.proj_dist is None):
            raise ValueError('Projector intrinsics are missing. Set calibrate_proj = True.')

        def calibrate_projector(keep):
            # Start from a generic guess, not a previous projector's result.
            # Radial distortion only: tangential terms trade off against the principal point and make it unreliable.
            proj_mtx_init = np.array([[ 2000.,    0., self.proj_width/2.],
                                      [    0., 2000., self.proj_height/2.],
                                      [    0.,    0.,           1.]])
            return cv2.calibrateCamera([proj_obj_pts[i] for i in keep], [proj_circle_pts[i] for i in keep],
                                       (self.proj_width, self.proj_height), proj_mtx_init, None,
                                       flags=cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST)[:3]

        def calibrate_stereo(keep, proj_mtx, proj_dist):
            out = cv2.stereoCalibrateExtended([proj_obj_pts[i] for i in keep], [cam_circle_pts[i] for i in keep],
                                              [proj_circle_pts[i] for i in keep], self.cam_mtx, self.cam_dist,
                                              proj_mtx, proj_dist, (self.cam_width, self.cam_height), None, None,
                                              flags=cv2.CALIB_FIX_INTRINSIC)
            error, R, T, per_view = out[0], out[5], out[6], np.asarray(out[11]).reshape(-1, 2)
            return error, R, T, per_view

        # First pass with every photo, to find photos that disagree with the rest
        keep = list(range(len(names)))
        proj_mtx, proj_dist = self.proj_mtx, self.proj_dist
        if calibrate_proj:
            _, proj_mtx, proj_dist = calibrate_projector(keep)
        error, proj_R, proj_T, per_view = calibrate_stereo(keep, proj_mtx, proj_dist)
        combined = np.sqrt((per_view ** 2).mean(axis=1))
        limit = max(2.0 * float(np.median(combined)), 1.0)
        worst = list(np.argsort(-combined))
        drop = [i for i in worst if combined[i] > limit][:max(1, len(keep) // 4)]
        if len(keep) - len(drop) < 5:
            drop = []

        print('\nPer-photo error in pixels (camera side / projector side), worst first:')
        for i in worst[:8]:
            flag = '   <- dropped' if i in drop else ''
            print(f'  {names[i]:22s} {per_view[i, 0]:5.2f} / {per_view[i, 1]:5.2f}{flag}')
        if drop:
            print(f'Dropping {len(drop)} photo(s) over {limit:.2f} px and recalibrating with {len(keep) - len(drop)}.')
            keep = [i for i in keep if i not in drop]
        else:
            print(f'No photo is far worse than the others (limit {limit:.2f} px).')

        # Final calibration
        if calibrate_proj:
            proj_err, proj_mtx, proj_dist = calibrate_projector(keep)
            print('\nProjector calibration')
            print(proj_mtx)
            print(proj_dist)
            print('Error', proj_err)
            self.proj_mtx = proj_mtx
            self.proj_dist = proj_dist
        error, proj_R, proj_T, per_view = calibrate_stereo(keep, self.proj_mtx, self.proj_dist)
        cam_mtx, cam_dist, proj_mtx, proj_dist = self.cam_mtx, self.cam_dist, self.proj_mtx, self.proj_dist

        print('\nStereo calibration')
        print('Camera parameters')
        print(cam_mtx, cam_dist)
        print('Projector parameters')
        print(proj_mtx, proj_dist)
        print('Rotation and translation')
        print(proj_R)
        print(proj_T)
        side_cam = float(np.sqrt((per_view[:, 0] ** 2).mean()))
        side_proj = float(np.sqrt((per_view[:, 1] ** 2).mean()))
        print('Error', error, f'  (camera side {side_cam:.2f} px, projector side {side_proj:.2f} px)')
        print(f'Camera-to-projector distance: {np.linalg.norm(proj_T) * 100:.1f} cm (compare with your tape measure)')

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cam_mtx, cam_dist, proj_mtx, proj_dist, (self.cam_width,self.cam_height), proj_R, proj_T)

        return proj_mtx, proj_dist, proj_R, proj_T, R1, R2, P1, P2, Q
