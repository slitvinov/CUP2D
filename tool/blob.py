import jax
import jax.numpy as jnp
import struct
import numpy as np

x0, y0, s0 = 0.0, 0.0, 1.0
x1, y1, s1 = 2.0, 1.0, 2.0
x2, y2, s2 = -1.0, -1.0, 1.0
x3, y3, s3 = -1.0, 2.0, 1.0


def sdf_fun(xy):
    x, y = xy
    r0 = jnp.hypot(x - x0, y - y0)
    r1 = jnp.hypot(x - x1, y - y1)
    r2 = jnp.hypot(x - x2, y - y2)
    r3 = jnp.hypot(x - x3, y - y3)
    return jnp.exp(-r0 / s0) + jnp.exp(-r1 / s1) + jnp.exp(-r2 / s2) + jnp.exp(
        -r3 / s2) - 1.0


sdf_grad = jax.jacrev(sdf_fun)
def sdf_ratio(xy):
    s = sdf_fun(xy)
    dx, dy = sdf_grad(xy)
    return s / jnp.hypot(dx, dy)


sdf_ratio_batch = jax.vmap(sdf_ratio)
nr, np0 = 800, 800
length = 2 * max(s0, s1)
rmax = 4 * length
dr = rmax / nr
dp = 2 * jnp.pi / (np0 - 1)
r = jnp.arange(1, nr + 1) * dr
p = jnp.arange(np0) * dp
R, P = jnp.meshgrid(r, p, indexing="ij")
x = R * jnp.cos(P)
y = R * jnp.sin(P)
xy = jnp.stack([x.ravel(), y.ravel()], axis=-1)
sdf = sdf_ratio_batch(xy).reshape(x.shape)
mask = sdf > 0
weights = (R**2) * mask
area = jnp.sum(weights)
xc = jnp.sum(weights * x) / area
yc = jnp.sum(weights * y) / area
area *=  dr * dp
scale = 1 / jnp.sqrt(area)
x_shift = (x + xc) / scale
y_shift = (y + yc) / scale
xy_shift = jnp.stack([x_shift.ravel(), y_shift.ravel()], axis=-1)
sdf_shifted = scale * sdf_ratio_batch(xy_shift).reshape(x.shape)
with open("blob.raw", "wb") as f:
    f.write(b"SDF")
    f.write(struct.pack("ffii", float(length * scale), float(rmax * scale), nr, np0))
    f.write(np.asarray(sdf_shifted, dtype=np.float32).tobytes())
for xi, yi, si in zip(x.ravel(), y.ravel(), sdf_shifted.ravel()):
    if si > 0:
        print(xi, yi, si)
