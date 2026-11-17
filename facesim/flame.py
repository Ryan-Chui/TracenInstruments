"""
FLAME head model in plain NumPy.

Loads the FLAME model file, as downloaded from https://flame.is.tue.mpg.de
and evaluates the forward pass:

    vertices = FLAME(shape, expression, pose)

No PyTorch or chumpy is needed. Older FLAME pickles that contain chumpy objects
are handled with a stub, so you don't have to install chumpy.

FLAME conventions (checked against the FLAME template mesh):
    units are meters, +y is up, the face looks toward +z.
    joints: 0 global/root, 1 neck, 2 jaw, 3 left eye, 4 right eye.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

# Loading

_ALLOWED_MODULE_ROOTS = {"numpy", "scipy", "collections", "copyreg", "_codecs"}
_ALLOWED_BUILTINS = {"object", "list", "dict", "tuple", "set", "frozenset", "int",
                     "float", "complex", "str", "bytes", "bytearray", "bool", "slice"}


class _ChumpyStub:
    """
    Stand-in for chumpy objects so old FLAME pickles load without chumpy.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state

    def to_numpy(self) -> np.ndarray:
        d = self.__dict__
        if isinstance(d.get("x"), np.ndarray):
            return d["x"]
        arrays = [v for v in d.values() if isinstance(v, np.ndarray)]
        if arrays:
            return max(arrays, key=lambda a: a.size)
        raise ValueError("Could not extract an array from a chumpy object in the FLAME file.")


class _SafeUnpickler(pickle.Unpickler):
    """
    Only lets numpy/scipy objects through (pickles can run code otherwise).
    """

    def find_class(self, module, name):
        root = module.split(".")[0]
        if root == "chumpy":
            return _ChumpyStub
        if root == "builtins":
            if name in _ALLOWED_BUILTINS:
                return super().find_class(module, name)
        elif root in _ALLOWED_MODULE_ROOTS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Refusing to load {module}.{name} from model file")


def _to_array(x) -> np.ndarray:
    if isinstance(x, _ChumpyStub):
        return np.asarray(x.to_numpy())
    if hasattr(x, "toarray"):  # scipy.sparse
        return x.toarray()
    return np.asarray(x)


def load_model_dict(path: str | Path) -> dict:
    """
    Read a FLAME model file (.pkl, or a pickled dict saved as .npy/.npz).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FLAME model not found: {path}\n"
            "Register at https://flame.is.tue.mpg.de, open Downloads and get "
            "'FLAME 2023 Open' (flame2023_Open.pkl).")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True, encoding="latin1").item()
    with open(path, "rb") as f:
        return _SafeUnpickler(f, encoding="latin1").load()


def load_flame_masks(path: str | Path) -> dict[str, np.ndarray]:
    """
    Optional FLAME_masks.pkl (region name -> vertex indices), from the FLAME downloads page.
    """
    with open(path, "rb") as f:
        d = _SafeUnpickler(f, encoding="latin1").load()
    return {k: np.asarray(v, dtype=np.int64) for k, v in d.items()}


# Math helpers

def rodrigues(r: np.ndarray) -> np.ndarray:
    """
    Axis-angle vectors (J,3) -> rotation matrices (J,3,3).
    """
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))
    theta = np.linalg.norm(r, axis=1)
    k = r / np.maximum(theta, 1e-12)[:, None]
    K = np.zeros((len(r), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s = np.sin(theta)[:, None, None]
    c = np.cos(theta)[:, None, None]
    return np.eye(3)[None] + s * K + (1.0 - c) * (K @ K)


def euler_to_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Head rotation in FLAME axes (radians): yaw about +y, pitch about +x, roll about +z.
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # area-weighted
    vn = np.zeros_like(verts)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


# The model

class FlameModel:
    """
    FLAME forward pass in NumPy.

    Parameters are in FLAME's own units: shape and expression coefficients are
    roughly standard-normal (about -3..3), poses are axis-angle radians.
    """

    def __init__(self, path: str | Path, masks_path: str | Path | None = None):
        d = load_model_dict(path)
        self.path = str(path)
        self.v_template = _to_array(d["v_template"]).astype(np.float64)
        self.faces = _to_array(d["f"]).astype(np.int64)
        V = len(self.v_template)

        sd = _to_array(d["shapedirs"]).astype(np.float64)
        if sd.ndim == 2:
            sd = sd.reshape(V, 3, -1)
        n_basis = sd.shape[2]
        self.n_expr = 100 if n_basis > 100 else 0
        self.shapedirs = sd[:, :, : n_basis - self.n_expr]
        self.exprdirs = sd[:, :, n_basis - self.n_expr:]
        self.n_shape = self.shapedirs.shape[2]

        pd = _to_array(d["posedirs"]).astype(np.float64)
        if pd.ndim == 2:  # (V*3, P) or (P, V*3)
            pd = pd.reshape(V, 3, -1) if pd.shape[0] == V * 3 else pd.T.reshape(V, 3, -1)
        self.posedirs = pd

        self.J_regressor = _to_array(d["J_regressor"]).astype(np.float64)
        parents = _to_array(d["kintree_table"])[0].astype(np.int64)
        parents[0] = -1
        self.parents = parents
        self.weights = _to_array(d["weights"]).astype(np.float64)
        self.n_joints = self.J_regressor.shape[0]
        self.reoriented = self._canonicalize_axes()

        # Regions used for albedo/labels.
        if self.n_joints >= 5:
            self.eye_mask = self.weights[:, 3:5].sum(1) > 0.5
        else:
            self.eye_mask = np.zeros(V, bool)
        self.masks = load_flame_masks(masks_path) if masks_path else {}
        if "left_eyeball" in self.masks and "right_eyeball" in self.masks:
            self.eye_mask = np.zeros(V, bool)
            self.eye_mask[self.masks["left_eyeball"]] = True
            self.eye_mask[self.masks["right_eyeball"]] = True

        z = np.where(self.eye_mask, -np.inf, self.v_template[:, 2])
        self.nose_tip_index = int(np.argmax(z))
        eyes = self.v_template[self.eye_mask] if self.eye_mask.any() else self.v_template
        self.eye_center = eyes.mean(0)

    def head_axes(self, verts: np.ndarray | None = None):
        """
        Measure the head's (side, up, forward) directions from the mesh itself.

        side    = the line through the two eye joints (direction not used)
        up      = from the neck-base joint toward the eyes
        forward = from the middle of the head toward the eyes; the eyes sit well
                  in front of the skull's center, so this is where the face looks
        Doesn't depend on which eye joint is 'left' in the file.
        """
        v = self.v_template if verts is None else verts
        J = self.J_regressor @ v
        eyes = (J[3] + J[4]) / 2
        side = J[3] - J[4]
        side /= np.linalg.norm(side)
        up = eyes - J[0]
        forward = eyes - v.mean(0)
        return side, up / np.linalg.norm(up), forward / np.linalg.norm(forward)

    def _canonicalize_axes(self) -> bool:
        """
        Make sure the model is in FLAME's usual frame (y up, face looking
        toward +z). If the file uses other axes, rotate it by whole multiples of
        90 degrees so the face doesn't get scanned side-on or from behind.
        Returns True if the model was rotated.
        """
        if self.n_joints < 5:
            return False
        side, up, forward = self.head_axes()
        k_side = int(np.argmax(np.abs(side)))
        scores = np.abs(up).copy()
        scores[k_side] = -1
        k_up = int(np.argmax(scores))
        k_fwd = 3 - k_side - k_up
        u = np.eye(3)[k_up] * np.sign(up[k_up])
        f = np.eye(3)[k_fwd] * np.sign(forward[k_fwd])
        Q = np.stack([np.cross(u, f), u, f])     # rows: old-frame axes that become x, y, z
        if np.allclose(Q, np.eye(3)):
            return False
        # rotate the rest shape and every displacement basis into the new frame
        self.v_template = self.v_template @ Q.T
        self.shapedirs = np.einsum("ij,vjk->vik", Q, self.shapedirs)
        self.exprdirs = np.einsum("ij,vjk->vik", Q, self.exprdirs)
        # pose correctives are indexed by flattened (R_joint - I) blocks; those
        # blocks transform as Q F Q^T, so remap the input side too
        P = self.posedirs
        nb = P.shape[2] // 9
        if nb * 9 == P.shape[2] and nb > 0:
            P = P.reshape(P.shape[0], 3, nb, 3, 3)
            P = np.einsum("ai,vijcd,ec,fd->vajef", Q, P, Q, Q)
            self.posedirs = P.reshape(P.shape[0], 3, nb * 9)
        else:
            self.posedirs = np.einsum("ij,vjk->vik", Q, self.posedirs)
        print(f"note: {Path(self.path).name} uses different axes than expected; rotated it so the "
              f"face looks toward +z (axis map rows: {Q.astype(int).tolist()})")
        return True

    def forward(self, shape=None, expression=None, jaw_pose=None, neck_pose=None,
                global_pose=None, eye_pose=None) -> np.ndarray:
        """
        Return vertices (V,3) in FLAME coordinates (meters).

        shape: up to n_shape coefficients (shorter arrays are zero-padded)
        expression: up to 100 coefficients
        jaw_pose, neck_pose, global_pose: axis-angle (3,), radians
        eye_pose: (6,) = left eye (3) + right eye (3)
        """
        betas_s = _pad(shape, self.n_shape)
        betas_e = _pad(expression, self.n_expr)
        v = self.v_template + self.shapedirs @ betas_s + (self.exprdirs @ betas_e if self.n_expr else 0)

        J = self.J_regressor @ v
        full_pose = np.zeros((self.n_joints, 3))
        if global_pose is not None:
            full_pose[0] = global_pose
        if self.n_joints > 1 and neck_pose is not None:
            full_pose[1] = neck_pose
        if self.n_joints > 2 and jaw_pose is not None:
            full_pose[2] = jaw_pose
        if self.n_joints > 4 and eye_pose is not None:
            full_pose[3:5] = np.asarray(eye_pose, dtype=np.float64).reshape(2, 3)

        R = rodrigues(full_pose)
        pose_feature = (R[1:] - np.eye(3)).reshape(-1)
        n_pf = min(len(pose_feature), self.posedirs.shape[2])
        v = v + self.posedirs[:, :, :n_pf] @ pose_feature[:n_pf]

        G = np.zeros((self.n_joints, 4, 4))
        for i in range(self.n_joints):
            T = np.eye(4)
            T[:3, :3] = R[i]
            p = self.parents[i]
            T[:3, 3] = J[i] - (J[p] if p >= 0 else 0.0)
            G[i] = T if p < 0 else G[p] @ T
        A = G.copy()
        A[:, :3, 3] -= np.einsum("jab,jb->ja", G[:, :3, :3], J)
        Tv = np.einsum("vj,jab->vab", self.weights, A)
        return np.einsum("vab,vb->va", Tv[:, :3, :3], v) + Tv[:, :3, 3]

    def region_labels(self, verts: np.ndarray, hairline_above_eyes: float | None = 0.075) -> np.ndarray:
        """
        Per-vertex label: 1 skin, 2 eye, 3 hair.

        Uses FLAME_masks.pkl ('scalp') when it was given; otherwise a simple
        hairline heuristic in FLAME coordinates. Pass None for a bald head.
        """
        lab = np.ones(len(verts), np.uint8)
        if hairline_above_eyes is not None:
            if "scalp" in self.masks:
                lab[self.masks["scalp"]] = 3
            else:
                eye = verts[self.eye_mask].mean(0) if self.eye_mask.any() else self.eye_center
                rel = verts - eye
                hair = (rel[:, 1] > hairline_above_eyes) | ((rel[:, 2] < -0.075) & (rel[:, 1] > -0.01))
                lab[hair] = 3
        lab[self.eye_mask] = 2
        return lab


def _pad(x, n):
    out = np.zeros(n)
    if x is not None and n:
        x = np.asarray(x, dtype=np.float64).ravel()[:n]
        out[: len(x)] = x
    return out
