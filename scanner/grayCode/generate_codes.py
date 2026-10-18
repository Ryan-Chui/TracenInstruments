import cv2
import numpy as np

# Inspired from https://github.com/sjnarmstrong/gray-code-structured-light/blob/master/CaptureImages/GrayImages.py
def get_n_bits(width, height):
    """Number of Gray code bits needed to give every projector column and row its own code."""
    return int(np.ceil(np.log2(max(width, height))))


def get_gray_codes(width, height):
    """
    Generates Gray codes for a given resolution.

    Parameters:
    ----------
    width : int
        The width of the image.
    height : int
        The height of the image.

    Returns:
    -------
    codes : np.array
        The Gray codes, one row per value 0..max(width, height)-1, most significant bit first.
    """
    n_bits = get_n_bits(width, height)
    values = np.arange(max(width, height), dtype=np.int64)
    gray = values ^ (values >> 1)
    shifts = np.arange(n_bits - 1, -1, -1)
    return ((gray[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


def get_image_sequence(gray_codes, width, height):
    """
    Generates the sequence of Gray code images projected during a scan.

    Order (n = number of bits):
        0: all black, 1: all white,
        2 .. 2n+1: column bits (most significant first) interleaved with row bits (least significant first),
        2n+2 .. 4n+1: the inverse of each of those.
    Column bits use vertical stripes, row bits use horizontal stripes.

    Parameters:
    ----------
    gray_codes : np.array
        The Gray codes from get_gray_codes.
    width : int
        The width of the image.
    height : int
        The height of the image.

    Returns:
    -------
    images : np.array
        The Gray code images, uint8, shape (4n+2, height, width).
    """
    n = gray_codes.shape[1]
    images = np.zeros((4 * n + 2, height, width), dtype=np.uint8)
    images[1] = 255

    col_bits = gray_codes[:width]      # (width, n)
    row_bits = gray_codes[:height]     # (height, n)
    for j in range(n):                 # j = 0 is the most significant bit
        id_v = 2 * j + 2
        id_h = 2 * (n - (j + 1)) + 3
        images[id_v] = (col_bits[:, j] * 255)[None, :]
        images[id_h] = (row_bits[:, j] * 255)[:, None]
        images[id_v + 2 * n] = 255 - images[id_v]
        images[id_h + 2 * n] = 255 - images[id_h]
    return images


def display_gray_code(gray_codes, width, height, repeat_n, fps, write_video_seq=False):
    """
    Displays the Gray code pattern.

    Parameters:
    ----------
    gray_codes : np.array
        The Gray codes.
    width : int
        The width of the image.
    height : int
        The height of the image.
    repeat_n : int
        The number of times to repeat the sequence.
    fps : int
        The number of frames per second.
    write_video_seq : bool
        Whether to write the sequence to a video file.
    """
    images = get_image_sequence(gray_codes, width, height)

    if write_video_seq:
        out = cv2.VideoWriter('./data/gray_sequence.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), isColor=False)
    for n in range(repeat_n):
        for i in range(len(images)):
            # Display the image
            cv2.namedWindow('Gray Code Pattern', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Gray Code Pattern', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Gray Code Pattern", images[i])
            cv2.waitKey(int(1000/fps))

            if write_video_seq and n == 0:
                out.write(images[i])

    cv2.destroyAllWindows()

    if write_video_seq:
        out.release()
