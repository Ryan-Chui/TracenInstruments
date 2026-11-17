"""
Face point cloud in, embedding and liveness answers out

    points (B, 1024, 3);            nose at the origin, units of 0.1 m
        |
        |  encoder: EdgeConv
        V
    face summary (B, 1024)
        |----------------------------------|
        V                                  V
    embedding                          liveness (real vs photo)
    (B, 128), length 1                 (B,) one number: > 0 real face, < 0 printed photo

Why an embedding:
    A classifier can only answer "which of the people I was trained on is this?"
    and it always names somebody, even for a stranger. The embedding head instead
    places every scan on a 128-D sphere so that scans of the same person land close
    together. To recognize someone you compare their embedding with the stored ones
    (cosine similarity = dot product, since all have length 1) and only accept a
    match above a threshold. New people are enrolled without retraining, and
    strangers fall below the threshold instead of being forced into a class.

Why EdgeConv (instead of PointNet):
    PointNet runs the same MLP on every point by itself, then takes the max.
    It never sees how a point sits relative to its neighbors, and that is where
    the shape of a nose bridge or brow ridge lives. EdgeConv (from DGCNN, Wang et
    al. 2019) gives each point its k nearest neighbors and learns from the offsets
    (neighbor - point), i.e. the local surface shape. It is PointNet plus one idea.
    Here the neighbors are found once in 3D (the paper refinds them in feature
    space at every layer), which for a face, that is simpler and nearly as good.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# EdgeConv encoder

@torch.no_grad()
def knn(points: torch.Tensor, k: int) -> torch.Tensor:
    """(B, N, 3) -> (B, N, k): indices of each point's k nearest neighbors (not itself)."""
    dist = torch.cdist(points, points) # (B, N, N) all pairwise distances
    dist.diagonal(dim1=1, dim2=2).fill_(float("inf")) # a point is not its own neighbor
    return dist.topk(k, dim=-1, largest=False).indices


def gather_neighbors(feat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """feat (B, C, N), idx (B, N, k) -> (B, C, N, k): the features of each point's neighbors."""
    B, C, N = feat.shape
    k = idx.shape[-1]
    flat = idx.reshape(B, 1, N * k).expand(-1, C, -1) # same neighbor list for every channel
    return feat.gather(2, flat).view(B, C, N, k)


class EdgeConv(nn.Module):
    """
    For every point i: out_i = max over neighbors j of  A*(f_j - f_i)  +  B*f_i
                       then BatchNorm + LeakyReLU.

    A*(f_j - f_i) looks at where each neighbor sits relative to the point (local
    shape); B*f_i keeps what the point itself is. On the first layer f is (x, y, z),
    so for each of the 64 learned directions in A the max says how far the surface
    around the point reaches in that direction: slope and curvature.

    It is computed the cheap way. Since A*(f_j - f_i) = A*f_j - A*f_i, we apply A
    once per point, gather the results per neighbor, take the max, then subtract.
    That is k times less multiplication than running a layer on every
    (point, neighbor) pair, and ~4x faster to train. (The DGCNN paper applies
    BatchNorm + activation before the max; after it is the same up to the sign
    of BatchNorm's scale, and much cheaper.)
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.A = nn.Conv1d(c_in, c_out, 1, bias=False) # a 1x1 Conv1d = the same Linear layer on every point
        self.B = nn.Conv1d(c_in, c_out, 1, bias=False)
        self.post = nn.Sequential(nn.BatchNorm1d(c_out), nn.LeakyReLU(0.2))

    def forward(self, feat, idx): # feat (B, C, N), idx (B, N, k)
        a = self.A(feat) # (B, c_out, N); A*f for every point
        a_max = gather_neighbors(a, idx).max(dim=-1).values # (B, c_out, N) max of A*f_j over the neighbors
        return self.post(a_max - a + self.B(feat)) # max_j A*(f_j - f_i) + B*f_i


class EdgeConvEncoder(nn.Module):
    out_dim = 1024

    def __init__(self, k: int = 16, widths=(64, 64, 128, 256)):
        super().__init__()
        self.k = k
        layers, c = [], 3
        for w in widths:
            layers.append(EdgeConv(c, w))
            c = w
        self.layers = nn.ModuleList(layers)
        # mix the features of all layers (fine detail from early ones, larger shape from later ones)
        self.mix = nn.Sequential(nn.Conv1d(sum(widths), 512, 1, bias=False),
                                 nn.BatchNorm1d(512), nn.LeakyReLU(0.2))

    def forward(self, points): # (B, N, 3)
        idx = knn(points, self.k) # neighbors found once, in 3D
        f = points.transpose(1, 2) # (B, 3, N): Conv layers want channels first
        outs = []
        for layer in self.layers:
            f = layer(f, idx)
            outs.append(f)
        f = self.mix(torch.cat(outs, dim=1)) # (B, 512, N)
        # summarize the face: max (strongest feature anywhere) and mean (overall shape)
        return torch.cat([f.max(dim=2).values, f.mean(dim=2)], dim=1)   # (B, 1024)


# PointNet encoder (baseline): the per-point layers of tinypointnet.py

class PointNetEncoder(nn.Module):
    out_dim = 1024

    def __init__(self, **_):
        super().__init__()
        self.per_point = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )

    def forward(self, points):
        return self.per_point(points.transpose(1, 2)).max(dim=2).values # (B, 1024)


# The full network

class FaceNet(nn.Module):
    def __init__(self, encoder: str = "edgeconv", emb_dim: int = 128, k: int = 16):
        super().__init__()
        self.config = {"encoder": encoder, "emb_dim": emb_dim, "k": k}
        self.encoder = EdgeConvEncoder(k=k) if encoder == "edgeconv" else PointNetEncoder()
        d = self.encoder.out_dim
        self.embed_head = nn.Sequential(nn.Linear(d, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(512, emb_dim))
        self.live_head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, points):
        g = self.encoder(points)
        emb = F.normalize(self.embed_head(g), dim=1) # length 1, so dot product = cosine
        live_logit = self.live_head(g).squeeze(1)
        return emb, live_logit


class CosFaceLoss(nn.Module):
    """
    Softmax over the training people, changed in two ways (CosFace, Wang et al. 2018):
      1. the score for person c is the cosine between the embedding and a learned
         direction ("center") for c, so only the direction of the embedding matters
      2. the correct person's cosine has a margin m subtracted before the softmax,
         so it must beat every other person by at least m to get a low loss
    The margin is what pushes different people apart on the sphere, which is what
    lets a single threshold separate "same person" from "different person" later,
    including for people the model has never seen. The centers are thrown away
    after training; only the embedding network is used.
    """

    def __init__(self, emb_dim: int, n_people: int, s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.s, self.m = s, m
        self.centers = nn.Parameter(torch.randn(n_people, emb_dim) * 0.01)

    def forward(self, emb, labels):
        cos = emb @ F.normalize(self.centers, dim=1).T # (B, n_people)
        margin = self.m * F.one_hot(labels, cos.shape[1]) # m only at the right person
        loss = F.cross_entropy(self.s * (cos - margin), labels) # s spreads the scores out for softmax
        return loss, cos.detach()


# Helpers shared by train / evaluate / gallery

def pick_device(name: str | None = None) -> torch.device:
    """'cuda' on an NVIDIA GPU, 'mps' on an Apple-silicon Mac, else 'cpu'."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def embed_clouds(model: FaceNet, clouds: np.ndarray, n_points: int = 1024, views: int = 4,
                 batch_size: int = 64, device=None):
    """
    Embed preprocessed scans for matching.

    clouds: (S, P, 3) preprocessed clouds (cache entries), P >= n_points * views.
    The cached points are in random order, so points[:, :1024], points[:, 1024:2048], ...
    are independent random samples of the same face. Each is embedded and the results
    averaged ("views"), which steadies the embedding at almost no extra cost.

    Returns (embeddings (S, D) float32 with length 1, live_prob (S,) float32).
    """
    device = device or next(model.parameters()).device
    model.eval()
    S = len(clouds)
    views = max(1, min(views, clouds.shape[1] // n_points))
    emb_sum = np.zeros((S, model.config["emb_dim"]), np.float32)
    live_sum = np.zeros(S, np.float32)
    for v in range(views):
        for a in range(0, S, batch_size):
            pts = torch.from_numpy(np.asarray(clouds[a:a + batch_size, v * n_points:(v + 1) * n_points],
                                              np.float32)).to(device)
            emb, live_logit = model(pts)
            emb_sum[a:a + batch_size] += emb.cpu().numpy()
            live_sum[a:a + batch_size] += torch.sigmoid(live_logit).cpu().numpy()
    emb = emb_sum / np.linalg.norm(emb_sum, axis=1, keepdims=True)
    return emb, live_sum / views


def save_checkpoint(path, model: FaceNet, n_points: int, extra: dict | None = None):
    torch.save({"state_dict": model.state_dict(), "config": model.config,
                "n_points": n_points, **(extra or {})}, path)


def load_checkpoint(path, device=None):
    """Returns (model in eval mode on `device`, the checkpoint dict)."""
    device = pick_device(device) if not isinstance(device, torch.device) else device
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = FaceNet(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
