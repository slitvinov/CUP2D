import math
import struct
import sys
import os


def sdf_box(x, y, bx, by):
    dx = abs(x) - bx
    dy = abs(y) - by
    mx = max(dx, 0.0)
    my = max(dy, 0.0)
    outside_dist = math.hypot(mx, my)
    inside_dist = min(max(dx, dy), 0.0)
    return -outside_dist - inside_dist


bx = by = 0.5
length = 2 * max(bx, by)
rmax = 4 * length
nr = 400
np = 400
area = 0
J = 0
Sdf = []
for i in range(nr):
    for j in range(np):
        r = (i + 1) * rmax / nr
        p = j * (2 * math.pi) / (np - 2)
        x = r * math.cos(p)
        y = r * math.sin(p)
        sdf = sdf_box(x, y, bx, by)
        Sdf.append(sdf)
        if sdf > 0:
            area += r
            J += r**2
dr = rmax / nr
dp = 2 * math.pi / (np - 1)
area *= dr * dp
J *= dr * dp

print(length, area, J)
with open("box.raw", "wb") as f:
    f.write(b"SDF")
    f.write(struct.pack("ffffii", length, area, J, rmax, nr, np))
    assert len(Sdf) == nr * np
    for sdf in Sdf:
        f.write(struct.pack("f", sdf))
