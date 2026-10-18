import os
import numpy as np


def arr_creat(upperleft, upperright, lowerleft, lowerright, arrwidth, arrheight):
    """Creates an array of size (arrheight, arrwidth, 1) interpolating between the four corner values."""
    arr = np.linspace(np.linspace(lowerleft, lowerright, arrwidth),
                      np.linspace(upperleft, upperright, arrwidth), arrheight, dtype=int)
    return arr[:, :, None]


def create_color_map(width, height):
    """
    Creates a color map for the decoded gray codes: each projector pixel gets its own color.
    """
    r = arr_creat(0,   255, 0,   255, width, height)
    g = arr_creat(0,   0,   255, 0, width, height)
    b = arr_creat(255, 255, 0,   0, width, height)
    return np.concatenate([r, g, b], axis=2)


def decoded_image(h_pixels, v_pixels, color_map_img):
    """Camera image colored by the projector pixel each camera pixel decoded to (black = not decoded)."""
    height, width = color_map_img.shape[:2]
    valid = (h_pixels >= 0) & (v_pixels >= 0)
    cols = np.clip(np.round(h_pixels), 0, width - 1).astype(np.int64)
    rows = np.clip(np.round(v_pixels), 0, height - 1).astype(np.int64)
    result = color_map_img[rows, cols].astype(np.uint8)
    result[~valid] = 0
    return result


def plot_decoded_graycodes(h_pixels, v_pixels, color_map_img, save_to=None):
    """
    Shows (and optionally saves) the decoded Gray codes: smooth color gradients mean good decoding,
    black means the pixel couldn't be decoded, speckle means noisy decoding.
    """
    import matplotlib.pyplot as plt
    result_img = decoded_image(h_pixels, v_pixels, color_map_img)
    if save_to is not None:
        plt.imsave(save_to, result_img)
    plt.figure('Decoded projector positions')
    plt.imshow(result_img)
    plt.axis('off')
    plt.show()


def write_ply(path, Pts, colors=None):
    """Write a point cloud as a binary PLY file (opens in MeshLab, CloudCompare, Blender, Open3D)."""
    pts = np.asarray(Pts, dtype=np.float32)
    if pts.shape[0] == 3 and pts.shape[1] != 3:
        pts = pts.T
    n = len(pts)
    if colors is None:
        dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
        data = np.empty(n, dtype=dtype)
    else:
        dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        data = np.empty(n, dtype=dtype)
        rgb = np.clip(np.asarray(colors) * 255.0, 0, 255).astype(np.uint8)
        data['red'], data['green'], data['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    data['x'], data['y'], data['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    header = ['ply', 'format binary_little_endian 1.0', f'element vertex {n}',
              'property float x', 'property float y', 'property float z']
    if colors is not None:
        header += ['property uchar red', 'property uchar green', 'property uchar blue']
    header += ['end_header']
    with open(path, 'wb') as f:
        f.write(('\n'.join(header) + '\n').encode('ascii'))
        f.write(data.tobytes())


def plot_point_cloud(Pts, colors, save_to, remove_outliers=True):
    """
    Saves the point cloud to save_to/cloud.ply and shows it.

    Uses Open3D for viewing when it's installed (K = black background, . = screenshot);
    otherwise falls back to a matplotlib 3D plot of a subsample.
    """
    pts = np.asarray(Pts).T if np.asarray(Pts).shape[0] == 3 else np.asarray(Pts)
    try:
        import open3d as o3d
    except ImportError:
        o3d = None

    if o3d is None:
        path = os.path.join(save_to, 'cloud.ply')
        write_ply(path, pts, colors)
        print(f'Saved {path} ({len(pts)} points). Open3D is not installed, showing a matplotlib preview instead.')
        import matplotlib.pyplot as plt
        idx = np.random.default_rng(0).choice(len(pts), size=min(len(pts), 40000), replace=False)
        fig = plt.figure('Point cloud')
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pts[idx, 0], pts[idx, 2], -pts[idx, 1], s=0.5, c=None if colors is None else colors[idx])
        ax.set_xlabel('x (m)'); ax.set_ylabel('depth (m)'); ax.set_zlabel('up (m)')
        ax.set_box_aspect(np.ptp(np.c_[pts[idx, 0], pts[idx, 2], -pts[idx, 1]], axis=0))
        plt.show()
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))

    if remove_outliers and len(pts) > 100:
        print('Removing stray points')
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    path = os.path.join(save_to, 'cloud.ply')
    o3d.io.write_point_cloud(path, pcd)
    print(f'Saved {path} ({len(pcd.points)} points)')

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def capture_image(vis):
        import matplotlib.pyplot as plt
        image = vis.capture_screen_float_buffer()
        k = 0
        while os.path.exists(os.path.join(save_to, f'cloud_{k}.png')):
            k += 1
        plt.imsave(os.path.join(save_to, f'cloud_{k}.png'), np.asarray(image))
        return False

    key_to_callback = {ord("K"): change_background_to_black, ord("."): capture_image}
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)
