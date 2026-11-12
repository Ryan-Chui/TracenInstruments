import sys
import time
import cv2
import numpy as np


def pump(seconds):
    """Let OpenCV's window system process events for a while (key presses are discarded)."""
    t0 = time.time()
    while time.time() - t0 < seconds:
        cv2.waitKey(20)


def list_displays():
    """
    Return the connected displays as dicts (macOS only, needs pyobjc-framework-Cocoa).

    x, y, w, h are in macOS points, in global screen coordinates (origin at the bottom-left of the main display);
    px_w, px_h are the real pixel size the pattern images must have.
    """
    try:
        from AppKit import NSScreen
    except ImportError:
        raise RuntimeError('Finding the projector display needs pyobjc. Install it with:\n'
                           '    python -m pip install pyobjc-framework-Cocoa')
    displays = []
    for k, screen in enumerate(NSScreen.screens()):
        f = screen.frame()
        scale = float(screen.backingScaleFactor())
        try:
            name = str(screen.localizedName())
        except Exception:
            name = f'Display {k}'
        displays.append(dict(index=k, name=name,
                             x=float(f.origin.x), y=float(f.origin.y), w=float(f.size.width), h=float(f.size.height),
                             scale=scale,
                             px_w=int(round(f.size.width * scale)), px_h=int(round(f.size.height * scale)),
                             builtin=('built-in' in name.lower() or 'built in' in name.lower())))
    return displays


def pick_displays(displays, screen_index=None):
    """
    Return (projector, control): the display to project on, and the display for your own windows.
    By default the projector is the display that isn't the Mac's built-in screen.
    """
    if len(displays) < 2:
        raise RuntimeError('Only one display found. Connect the projector and set it to extend your desktop '
                           '(System Settings > Displays: "Extended display", not "Mirror").')
    if screen_index is None:
        builtin = [d for d in displays if d['builtin']]
        control = builtin[0] if builtin else displays[0]
        projector = next(d for d in displays if d['index'] != control['index'])
    else:
        if not 0 <= screen_index < len(displays):
            raise ValueError(f'proj_screen_index must be between 0 and {len(displays) - 1}')
        projector = displays[screen_index]
        others = [d for d in displays if d['index'] != projector['index']]
        builtin = [d for d in others if d['builtin']]
        control = builtin[0] if builtin else others[0]
    return projector, control


def _ns_window(name):
    """The macOS window behind an OpenCV window, found by its title."""
    from AppKit import NSApplication
    for window in NSApplication.sharedApplication().windows():
        if str(window.title()) == name:
            return window
    return None


def place_window(name, display, left=40, top=60):
    """Move an OpenCV window onto a display, bring it to the front and give it keyboard focus."""
    if sys.platform != 'darwin' or display is None:
        cv2.moveWindow(name, left, top)
        return
    window = _ns_window(name)
    if window is None:
        print(f"Could not find the '{name}' window to move it.")
        return
    window.setFrameTopLeftPoint_((display['x'] + left, display['y'] + display['h'] - top))
    from AppKit import NSApplication
    NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
    window.makeKeyAndOrderFront_(None)


class ProjectorWindow(object):
    """
    A borderless window covering the whole projector display.

    The projector must be connected as a separate display (extended desktop, not mirrored).
    On macOS this does not use OpenCV's fullscreen mode: that switches to a macOS full-screen Space,
    which can leave every display black. Instead the window loses its title bar and is placed exactly
    over the projector display, above its menu bar.
    """
    def __init__(self, name='Projector', screen_index=None, size=None, fallback_x=1920):
        self.name = name
        self.display = None
        self.control_display = None

        if sys.platform == 'darwin':
            displays = list_displays()
            self.display, self.control_display = pick_displays(displays, screen_index)
            self.width, self.height = self.display['px_w'], self.display['px_h']
            for d in displays:
                if d['index'] == self.display['index']:
                    tag = '  <- projector'
                elif d['index'] == self.control_display['index']:
                    tag = '  <- your windows'
                else:
                    tag = ''
                print(f"Display {d['index']}: {d['name']}  {d['px_w']}x{d['px_h']} px{tag}")
            self._open_mac()
        else:
            if size is None:
                raise ValueError('Pass size=(width, height) of the projector on this OS.')
            self.width, self.height = size
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
            cv2.imshow(self.name, np.zeros((self.height, self.width, 3), dtype=np.uint8))
            cv2.moveWindow(self.name, fallback_x, 0)
            cv2.waitKey(50)
            cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            pump(0.5)

    def _open_mac(self):
        from AppKit import NSScreen
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.name, np.zeros((self.height, self.width, 3), dtype=np.uint8))
        pump(0.2)

        window = _ns_window(self.name)
        if window is None:
            raise RuntimeError('Could not find the projector window to move it onto the projector.')

        window.setStyleMask_(0)                                   # borderless: no title bar
        window.setHasShadow_(False)
        window.setLevel_(1000)                                    # screen-saver level: above the menu bar and Dock
        window.setCollectionBehavior_((1 << 0) | (1 << 4) | (1 << 6))  # on all Spaces, stationary, skipped by Cmd-`
        window.setFrame_display_(NSScreen.screens()[self.display['index']].frame(), True)
        window.orderFrontRegardless()
        pump(0.3)

        f, d = window.frame(), self.display
        ok = (abs(f.origin.x - d['x']) < 1 and abs(f.origin.y - d['y']) < 1 and
              abs(f.size.width - d['w']) < 1 and abs(f.size.height - d['h']) < 1)
        print(f"Projector window: {f.size.width:.0f}x{f.size.height:.0f} at ({f.origin.x:.0f}, {f.origin.y:.0f}), "
              f"projector display: {d['w']:.0f}x{d['h']:.0f} at ({d['x']:.0f}, {d['y']:.0f})  "
              f"{'OK' if ok else 'MISMATCH - the window did not land on the projector'}")

    def show(self, img):
        if img.shape[1] != self.width or img.shape[0] != self.height:
            raise ValueError(f'Pattern is {img.shape[1]}x{img.shape[0]} but the projector is {self.width}x{self.height}')
        cv2.imshow(self.name, img)

    def close(self):
        try:
            cv2.destroyWindow(self.name)
        except cv2.error:
            pass
