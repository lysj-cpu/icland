# @title imports & utils
import base64
import io
import subprocess
import time

import jax
import jax.numpy as jnp
import matplotlib.pylab as pl
import numpy as np
import PIL
from IPython.display import HTML, Image, display
from jax._src import pjit
from sdfs import box_sdf, ramp_sdf


def np2pil(a):
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1) * 255)
    return PIL.Image.fromarray(a)


def imwrite(f, a, fmt=None):
    a = np.asarray(a)
    if isinstance(f, str):
        fmt = f.rsplit(".", 1)[-1].lower()
        if fmt == "jpg":
            fmt = "jpeg"
        f = open(f, "wb")
    np2pil(a).save(f, fmt, quality=95)


def imencode(a, fmt="jpeg"):
    a = np.asarray(a)
    if len(a.shape) == 3 and a.shape[-1] == 4:
        fmt = "png"
    f = io.BytesIO()
    imwrite(f, a, fmt)
    return f.getvalue()


def imshow(a, fmt="jpeg", display=display):
    return display(Image(data=imencode(a, fmt)))


def norm(v, axis=-1, keepdims=False, eps=0.0):
    return jnp.sqrt((v * v).sum(axis, keepdims=keepdims).clip(eps))


def normalize(v, axis=-1, eps=1e-20):
    return v / norm(v, axis, keepdims=True, eps=eps)


class VideoWriter:
    def __init__(self, filename="_autoplay.mp4", fps=30.0):
        self.ffmpeg = None
        self.filename = filename
        self.fps = fps
        self.view = display(display_id=True)
        self.last_preview_time = 0.0

    def add(self, img):
        img = np.asarray(img)
        h, w = img.shape[:2]
        if self.ffmpeg is None:
            self.ffmpeg = self._open(w, h)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.ffmpeg.stdin.write(img.tobytes())
        t = time.time()
        if self.view and t - self.last_preview_time > 1:
            self.last_preview_time = t
            imshow(img, display=self.view.update)

    def __call__(self, img):
        return self.add(img)

    def _open(self, w, h):
        cmd = f"""ffmpeg -y -f rawvideo -vcodec rawvideo -s {w}x{h}
      -pix_fmt rgb24 -r {self.fps} -i - -pix_fmt yuv420p
      -c:v libx264 -crf 20 {self.filename}""".split()
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def close(self):
        if self.ffmpeg:
            self.ffmpeg.stdin.close()
            self.ffmpeg.wait()
            self.ffmpeg = None

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()
        if self.filename == "_autoplay.mp4":
            self.show()

    def show(self):
        self.close()
        if not self.view:
            return
        b64 = base64.b64encode(open(self.filename, "rb").read()).decode("utf8")
        s = f"""<video controls loop>
 <source src="data:video/mp4;base64,{b64}" type="video/mp4">
 Your browser does not support the video tag.</video>"""
        self.view.update(HTML(s))


def animate(f, duration_sec, fps=60):
    with VideoWriter(fps=fps) as vid:
        for t in jnp.linspace(0, 1, int(duration_sec * fps)):
            vid(f(t))


# --- Start here ---
# @title (show_slice)
def show_slice(sdf, z=0.0, w=50, r=3.5):
    y, x = jnp.mgrid[-r : r : w * 1j, -r : r : w * 1j].reshape(2, -1)
    p = jnp.c_[x, y, x * 0.0 + z]
    d = jax.vmap(sdf)(p).reshape(w, w)
    pl.figure(figsize=(5, 5))
    kw = dict(extent=(-r, r, -r, r), vmin=-r, vmax=r)
    pl.contourf(d, 16, cmap="bwr", **kw)
    pl.contour(d, levels=[0.0], colors="black", **kw)
    pl.axis("equal")
    pl.xlabel("x")
    pl.ylabel("y")


def _translate(
    p: jax.Array,
    x: jnp.float32,
    y: jnp.float32,
    rot: jnp.int32,
    w: jnp.float32,
    h: jnp.float32,
    fun: pjit.JitWrapped,
) -> pjit.JitWrapped:
    angle = -jnp.pi * rot / 2
    cos_t = jnp.cos(angle)
    sin_t = jnp.sin(angle)
    transformed = jnp.matmul(
        jnp.linalg.inv(
            jnp.array(
                [
                    [cos_t, -sin_t, x],
                    [sin_t, cos_t, y],
                    [0, 0, 1],
                ]
            )
        ),
        p,
    )
    return fun(transformed, w, h)


def _scene_sdf_from_tilemap(tilemap: jax.Array, p: jax.Array, floor_level=-3.0):
    w, h = tilemap.shape[0], tilemap.shape[1]
    dists = jnp.arange(w * h, dtype=jnp.int32)
    tile_width = 1
    process_tile = lambda p, x, y, tile: jax.lax.switch(
        tile[0],
        [
            lambda p, x, y, rot, w, h: _translate(p, x, y, rot, w, h, box_sdf),
            lambda p, x, y, rot, w, h: _translate(p, x, y, rot, w, h, ramp_sdf),
            lambda p, x, y, rot, w, h: _translate(p, x, y, rot, w, h, box_sdf),
        ],
        p,
        x,
        y,
        tile[1],
        tile_width,
        tile[3],
    )
    tile_dists = jax.vmap(
        lambda i: process_tile(p, i // w, i % w, tilemap[i // w, i % w])
    )(dists)

    floor = p[1] - floor_level

    return jnp.minimum(floor, tile_dists.min())


def raycast(sdf, p0, dir, step_n=50):
    def f(_, p):
        return p + sdf(p) * dir

    return jax.lax.fori_loop(0, step_n, f, p0)


def camera_rays(forward, view_size, fx=0.6):
    world_up = jnp.array([0.0, 1.0, 0.0])
    right = jnp.cross(forward, world_up)
    down = jnp.cross(right, forward)
    R = normalize(jnp.vstack([right, down, forward]))
    w, h = view_size
    fy = fx / w * h
    y, x = jnp.mgrid[fy : -fy : h * 1j, -fx : fx : w * 1j].reshape(2, -1)
    return normalize(jnp.c_[x, y, jnp.ones_like(x)]) @ R
