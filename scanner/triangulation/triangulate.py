import numpy as np
import cv2


class Triangulate(object):
    """
    3D points from decoded Gray codes.

    Each camera pixel sees one projector column. That column, together with the projector's lens center,
    forms a (locally) flat sheet of light; the 3D point is where the camera pixel's ray crosses that sheet.
    Points are returned in the camera's coordinate frame, in meters: x right, y down, z = distance in front.
    """
    def __init__(self,
                 h_pixels=None,
                 v_pixels=None,
                 cam_size=(1920, 1080),
                 cam_mtx=None,
                 cam_dist=None,
                 proj_size=(1920, 1080),
                 proj_calib_size=(1920, 1080),
                 proj_mtx=None,
                 proj_dist=None,
                 proj_R=None,
                 proj_T=None,
                 image_folder=None):
        self.h_pixels = h_pixels
        self.v_pixels = v_pixels
        self.cam_w, self.cam_h = cam_size
        self.cam_mtx = None if cam_mtx is None else np.array(cam_mtx, dtype=np.float64)
        self.cam_dist = None if cam_dist is None else np.array(cam_dist, dtype=np.float64)
        self.proj_w, self.proj_h = proj_size
        self.proj_dist = None if proj_dist is None else np.array(proj_dist, dtype=np.float64)

        # If the projector ran at a different resolution than during calibration, rescale its intrinsics
        self.proj_mtx = None
        if proj_mtx is not None:
            K = np.array(proj_mtx, dtype=np.float64)        # copy: never change the caller's array
            calib_w, calib_h = proj_calib_size
            sx, sy = self.proj_w / calib_w, self.proj_h / calib_h
            K[0, 0] *= sx
            K[1, 1] *= sy
            K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
            K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
            self.proj_mtx = K

        self.proj_R = None if proj_R is None else np.array(proj_R, dtype=np.float64)
        self.proj_T = None if proj_T is None else np.array(proj_T, dtype=np.float64).reshape(3, 1)
        self.image_folder = image_folder

    def get_cam_proj_pts(self, img_white=None):
        """
        Camera pixels that were decoded, the projector position each one saw, and its color.

        Returns:
            cam_pts: (N, 2) camera pixels (x, y)
            proj_pts: (N, 2) projector positions (column, row)
            colors: (N, 3) colors in 0..1 from img_white (RGB), or None
        """
        valid = (self.h_pixels >= 0) & (self.v_pixels >= 0)
        ys, xs = np.nonzero(valid)
        cam_pts = np.stack([xs, ys], axis=1).astype(np.float32)
        proj_pts = np.stack([self.h_pixels[ys, xs], self.v_pixels[ys, xs]], axis=1).astype(np.float32)
        colors = None
        if img_white is not None:
            colors = img_white[ys, xs, :3].astype(np.float64) / 255.0
        return cam_pts, proj_pts, colors

    def triangulate(self, cam_pts, proj_pts):
        """
        Triangulate 3D points.

        Parameters:
            cam_pts: (N, 2) camera pixels
            proj_pts: (N, 2) projector (column, row) for each camera pixel

        Returns:
            points: (3, N) points in the camera frame, meters; NaN where the rays don't meet in front
        """
        R, T = self.proj_R, self.proj_T.ravel()

        # Camera rays (camera frame), as directions with z = 1
        rc = cv2.undistortPoints(cam_pts.reshape(-1, 1, 2), self.cam_mtx, self.cam_dist).reshape(-1, 2)
        d_cam = np.column_stack([rc, np.ones(len(rc))])
        d_proj = d_cam @ R.T                    # same rays, expressed in the projector frame
        origin = T                              # camera center in the projector frame

        # Sheet of light of each projector column, tangent at the decoded row (handles lens distortion)
        u, v = proj_pts[:, 0].astype(np.float64), proj_pts[:, 1].astype(np.float64)
        step = 4.0
        a = cv2.undistortPoints(np.stack([u, v - step], 1).reshape(-1, 1, 2), self.proj_mtx, self.proj_dist).reshape(-1, 2)
        b = cv2.undistortPoints(np.stack([u, v + step], 1).reshape(-1, 1, 2), self.proj_mtx, self.proj_dist).reshape(-1, 2)
        normals = np.cross(np.column_stack([a, np.ones(len(a))]), np.column_stack([b, np.ones(len(b))]))

        # Ray-sheet intersection: normal . (origin + s * d) = 0
        denom = np.einsum('ij,ij->i', normals, d_proj)
        with np.errstate(divide='ignore', invalid='ignore'):
            s = -(normals @ origin) / denom
        s[~np.isfinite(s) | (s <= 0) | (np.abs(denom) < 1e-12)] = np.nan

        # Point in the camera frame: the camera ray scaled by s (its z is the depth)
        return (d_cam * s[:, None]).T

    def filter_3d_pts(self, Pts, colors, z_range=(0.2, 3.0)):
        """
        Keep points with a valid depth inside z_range (meters).

        Returns
        ---------
        Pts_filtered: (3, M)
        colors_filtered : (M, 3) or None
        """
        z = Pts[2]
        keep = np.isfinite(z) & (z > z_range[0]) & (z < z_range[1])
        return Pts[:, keep], (None if colors is None else colors[keep])
