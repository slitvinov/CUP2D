import math
import struct
import sys
import os


def sdf2_segment(x, y, ax, ay, bx, by):
    x -= ax
    y -= ay
    bx -= ax
    by -= ay
    bb = by * by + bx * bx
    h = (by * y + bx * x) / bb
    h = 1.0 if h > 1.0 else 0.0 if h < 0.0 else h
    cx = x - bx * h
    cy = y - by * h
    return cx * cx + cy * cy


length = 0.5
ax = -length / 2
bx = length / 2
ext = length / 10
nr = 400
np = 400
rmax = 4 * length
ay = 0
by = 0
Sdf = []
for i in range(nr):
    for j in range(np):
        r = (i + 1) * rmax / nr
        p = j * (2 * math.pi) / (np - 2)
        x = r * math.cos(p)
        y = r * math.sin(p)
        sdf2 = sdf2_segment(x, y, ax, ay, bx, by)
        sdf = -math.sqrt(sdf2) + length / 10
        Sdf.append(sdf)

mass = 10
J = 8.80277e-06
with open("sdf.raw", "wb") as f:
    f.write(b"SDF")
    f.write(struct.pack("fffii", mass, J, rmax, nr, np))
    assert len(Sdf) == nr * np
    for sdf in Sdf:
        f.write(struct.pack("f", sdf))
