import jax
import jax.numpy as jnp
import struct
import numpy as np
import sys


def sdf_ratio(xy):
    s = sdf_fun(xy)
    dx, dy = sdf_grad(xy)
    return s / jnp.hypot(dx, dy)


def sdf_fun(xy):
    x, y = xy
    r0 = jnp.hypot(x - x0, y - y0)
    r1 = jnp.hypot(x - x1, y - y1)
    r2 = jnp.hypot(x - x2, y - y2)
    r3 = jnp.hypot(x - x3, y - y3)
    return jnp.exp(-r0 / s0) + jnp.exp(-r1 / s1) + jnp.exp(-r2 / s2) + jnp.exp(
        -r3 / s2) - 1.0


x0, y0, s0 = 0.0, 0.0, 1.0
x1, y1, s1 = 2.0, 1.0, 2.0
x2, y2, s2 = -1.0, -1.0, 1.0
x3, y3, s3 = -1.0, 2.0, 1.0
sdf_grad = jax.jacrev(sdf_fun)
xh = yh = 10
xl = yl = -10
nr, np0 = 800, 800
n = 256
length = 2
rmax = 4 * length
x = jnp.linspace(-rmax, rmax, n)
y = jnp.linspace(-rmax, rmax, n)
XY = jnp.meshgrid(x, y, indexing='ij')
dx = x[1] - x[0]
dy = y[1] - y[0]
phi = sdf_fun(XY)
inside = phi > 0
area = jnp.sum(inside) * dx * dy
X, Y = jnp.meshgrid(x, y, indexing='ij')
XY = jnp.stack([X, Y], axis=0)
rc = jnp.sum(jnp.where(inside, XY, 0.0), axis=(1, 2)) * dx * dy / area
scale = 1 / jnp.sqrt(area)

sdf_ratio_batch = jax.vmap(sdf_ratio)
dr = rmax / nr
dp = 2 * jnp.pi / (np0 - 1)
r = jnp.arange(1, nr + 1) * dr
p = jnp.arange(np0) * dp
R, P = jnp.meshgrid(r, p, indexing="ij")
x = R * jnp.cos(P)
y = R * jnp.sin(P)
xy = jnp.stack([x.ravel(), y.ravel()], axis=-1)
sdf = sdf_ratio_batch(xy / scale + rc).reshape(x.shape) * scale

sys.stderr.write("%g\n" % (length * scale))
with open("blob.raw", "wb") as f:
    f.write(b"SDF")
    f.write(struct.pack("ffii", length * scale, rmax, nr, np0))
    f.write(np.asarray(sdf, dtype=np.float32).tobytes())
'''
for xi, yi, si in zip(x.ravel(), y.ravel(), sdf.ravel()):
    if si > 0:
        print(xi, yi, si)
'''
