import os
import cv2
import numpy as np


def read_images(folder):
    """
    Reads all images from a folder, sorted by name length then name (frame_2 before frame_10).

    Returns:
    -------
    images : np.array
        The images.
    file_names : np.array
        The names of the images.
    """
    files = sorted(os.listdir(folder), key=lambda f: (len(f), f))
    images = None
    file_names = []
    for i, file_name in enumerate(files):
        im = cv2.imread(os.path.join(folder, file_name))
        if images is None:
            images = np.empty((len(files), im.shape[0], im.shape[1], im.shape[2]), dtype=np.uint8)
        file_names.append(file_name)
        images[i] = im
    return images, np.array(file_names)


def to_gray(images):
    """Converts a set of BGR images to grayscale (uint8)."""
    return np.stack([cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY) for image in images])


def split_sequence(images):
    """
    Split a captured sequence (see generate_codes.get_image_sequence for the order) into its parts.

    Returns black, white, column patterns, column inverses, row patterns, row inverses;
    pattern stacks are ordered most significant bit first.
    """
    n = (len(images) - 2) // 4
    if len(images) != 4 * n + 2:
        raise ValueError(f'A Gray code sequence has 4n+2 images, got {len(images)}')
    patterns = images[2:]
    normal, inverse = patterns[:2 * n], patterns[2 * n:]
    col_norm, col_inv = normal[0::2], inverse[0::2]                  # column bits, most significant first
    row_norm, row_inv = normal[1::2][::-1], inverse[1::2][::-1]      # row bits were least significant first
    return images[0], images[1], col_norm, col_inv, row_norm, row_inv


# Robust pixel classification from Xu & Aliaga, "Robust Pixel Classification for 3D Modeling with
# Structured Light" (GI 2007), using the direct/global separation of Nayar et al. (SIGGRAPH 2006).
def get_direct_global(high_freq):
    """
    Direct and global light per pixel from high-frequency patterns and their inverses (room light removed).

    Parameters:
    ----------
    high_freq : np.array
        Stack of high-frequency patterns and their inverses, with the black image already subtracted.
    """
    L_max = high_freq.max(axis=0)
    L_min = high_freq.min(axis=0)
    L_d = L_max - L_min
    L_g = 2.0 * L_min
    return L_d, L_g


def classify_bits(norm, inv, L_d, L_g, m=10.0, eps=1.0):
    """
    Decide for each bit and pixel: 1 = lit, 0 = unlit, -1 = can't tell.
    The rules are tried in order and the first one that applies wins.
    """
    L_d = L_d[None]
    L_g = L_g[None]
    conditions = [
        np.broadcast_to(L_d < m, norm.shape),           # too little projector light: can't tell
        (L_d > L_g + eps) & (norm > inv + eps),          # direct light dominates
        (L_d > L_g + eps) & (norm + eps < inv),
        (norm + eps < L_d) & (inv > L_g + eps),          # global light dominates
        (norm > L_g + eps) & (inv + eps < L_d),
    ]
    return np.select(conditions, [-1, 1, 0, 0, 1], default=-1).astype(np.int8)


def decode_sequence(images, min_contrast=10.0, eps=1.0, n_high_freq=3, band=120):
    """
    Turn one captured Gray code sequence into bits.

    Parameters:
    ----------
    images : np.array
        Grayscale captures (4n+2, H, W) in projection order.
    min_contrast : float
        Minimum direct light (0-255 scale) for a pixel to be trusted.
    n_high_freq : int
        Number of finest column and row patterns used to separate direct and global light.
    band : int
        Rows processed at a time (keeps memory low).

    Returns:
    -------
    col_codes, row_codes : np.array
        int8 (n, H, W), most significant bit first; 1 = lit, 0 = unlit, -1 = can't tell.
    """
    black, white, col_norm, col_inv, row_norm, row_inv = split_sequence(images)
    n, H, W = col_norm.shape
    col_codes = np.empty((n, H, W), dtype=np.int8)
    row_codes = np.empty((n, H, W), dtype=np.int8)
    for y0 in range(0, H, band):
        sl = slice(y0, min(H, y0 + band))
        b = black[sl].astype(np.float32)
        cn = np.clip(col_norm[:, sl].astype(np.float32) - b, 0, None)
        ci = np.clip(col_inv[:, sl].astype(np.float32) - b, 0, None)
        rn = np.clip(row_norm[:, sl].astype(np.float32) - b, 0, None)
        ri = np.clip(row_inv[:, sl].astype(np.float32) - b, 0, None)
        k = n_high_freq
        L_d, L_g = get_direct_global(np.concatenate([cn[-k:], ci[-k:], rn[-k:], ri[-k:]]))
        col_codes[:, sl] = classify_bits(cn, ci, L_d, L_g, min_contrast, eps)
        row_codes[:, sl] = classify_bits(rn, ri, L_d, L_g, min_contrast, eps)
    return col_codes, row_codes


def combine_codes(code_list):
    """
    Merge the bits of several captures of the same scene: keep agreeing bits, fill bits one capture
    couldn't read from another, and mark bits the captures disagree on as unreadable.
    """
    out = code_list[0].copy()
    for codes in code_list[1:]:
        out = np.where(out == codes, out,
              np.where(out < 0, codes,
              np.where(codes < 0, out, -1))).astype(np.int8)
    return out


def codes_to_positions(codes, limit, band=120):
    """
    Projector column (or row) seen by each camera pixel.

    A pixel with every bit readable gets that projector pixel. A pixel with exactly one unreadable bit
    usually sits on the edge between two neighboring projector pixels; it gets the half-way position.

    Parameters:
    ----------
    codes : np.array
        int8 (n, H, W), most significant bit first, -1 = unreadable.
    limit : int
        Projector width (for columns) or height (for rows); positions beyond it are invalid.

    Returns:
    -------
    positions : np.array
        float32 (H, W), -1 where the pixel couldn't be decoded.
    """
    n, H, W = codes.shape
    weights = (1 << np.arange(n - 1, -1, -1)).astype(np.int32)[:, None, None]
    positions = np.full((H, W), -1.0, dtype=np.float32)

    def to_value(bits):
        binary = np.bitwise_xor.accumulate(bits.astype(np.int32), axis=0)
        return (binary * weights).sum(axis=0)

    for y0 in range(0, H, band):
        sl = slice(y0, min(H, y0 + band))
        c = codes[:, sl]
        unknown = c < 0
        n_unknown = unknown.sum(axis=0)
        v0 = to_value(np.where(unknown, 0, c))
        v1 = to_value(np.where(unknown, 1, c))
        pos = np.full(v0.shape, -1.0, dtype=np.float32)
        exact = n_unknown == 0
        pos[exact] = v0[exact]
        edge = (n_unknown == 1) & (np.abs(v0 - v1) == 1)
        pos[edge] = (v0[edge] + v1[edge]) / 2.0
        pos[pos > limit - 1] = -1.0
        positions[sl] = pos
    return positions


def get_codes(images):
    """
    Decode the Gray codes from images (kept for compatibility).

    Returns:
    -------
    h_codes : np.array
        Column bits, most significant first.
    v_codes : np.array
        Row bits, least significant first (as in the original code).
    """
    col_codes, row_codes = decode_sequence(images)
    return col_codes, row_codes[::-1]


# https://stackoverflow.com/a/72028021
def gray_decode(n):
    """Gray code to integer."""
    m = n >> 1
    while m:
        n ^= m
        m >>= 1
    return n


def gray_to_decimal(gray_code_list):
    """Gray code given as a list of bits (most significant first) to decimal; -1 if any bit is unknown."""
    if any(int(b) < 0 for b in gray_code_list):
        return -1
    return gray_decode(int(''.join(str(int(b)) for b in gray_code_list), 2))


def remove_inconsistent(positions, max_diff=2.0, min_neighbors=5):
    """
    Drop decoded pixels that disagree with their neighbors. Real surfaces change projector position smoothly
    from one camera pixel to the next; isolated jumps come from pixels that see two surfaces at once
    (object outlines, shadow edges) or from noise.

    Parameters:
    ----------
    positions : np.array
        float32 (H, W) projector columns or rows, -1 = not decoded.
    max_diff : float
        Allowed difference (projector pixels) from the median of the 8 neighbors.
    min_neighbors : int
        Minimum number of decoded neighbors.
    """
    H, W = positions.shape
    p = np.where(positions >= 0, positions, np.nan).astype(np.float32)
    padded = np.pad(p, 1, constant_values=np.nan)
    out = positions.copy()
    for y0 in range(0, H, 256):
        y1 = min(H, y0 + 256)
        neigh = np.stack([padded[y0 + 1 + dy:y1 + 1 + dy, 1 + dx:W + 1 + dx]
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
        count = np.isfinite(neigh).sum(axis=0)
        with np.errstate(all='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                med = np.nanmedian(neigh, axis=0)
        center = p[y0:y1]
        bad = np.isfinite(center) & ((count < min_neighbors) | ~(np.abs(center - med) <= max_diff))
        out[y0:y1][bad] = -1.0
    return out
