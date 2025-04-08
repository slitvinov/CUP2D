import math
import struct
import sys
import os


def sdf_box(x, y):
    r0 = math.hypot(x - x0, y - y0)
    r1 = math.hypot(x - x1, y - y1)
    r2 = math.hypot(x - x2, y - y2)
    return math.exp(-r0 / s0) + math.exp(-r1 / s1) + math.exp(-r2 / s2) - 1


x0 = 0
y0 = 0
s0 = 1

x1 = 1
y1 = 1
s1 = 2

x2 = 1
y2 = -1
s2 = 1
length = 2 * max(s0, s1)
rmax = 3 * length
nr = 1600
np = 1600
dr = rmax / nr
dp = 2 * math.pi / (np - 1)
area = J = xc = yc = 0
for i in range(nr):
    for j in range(np):
        r = (i + 1) * rmax / nr
        p = j * (2 * math.pi) / (np - 2)
        x = r * math.cos(p)
        y = r * math.sin(p)
        sdf = sdf_box(x, y)
        if sdf > 0:
            area += r**2
            xc += x * r
            yc += y * r

Sdf = []
xc = 0
yc = 0
for i in range(nr):
    for j in range(np):
        r = (i + 1) * rmax / nr
        p = j * (2 * math.pi) / (np - 2)
        x = r * math.cos(p)
        y = r * math.sin(p)
        sdf = sdf_box(x - xc, y - yc)
        Sdf.append(sdf)
        if sdf > 0:
            area += r**2
            xc += x * r
            yc += y * r

xc /= area
yc /= area
area *= dr * dp
J *= dr * dp

print(length, area, J, xc, yc)
with open("blob.raw", "wb") as f:
    f.write(b"SDF")
    f.write(struct.pack("ffii", length, rmax, nr, np))
    assert len(Sdf) == nr * np
    for sdf in Sdf:
        f.write(struct.pack("f", sdf))
