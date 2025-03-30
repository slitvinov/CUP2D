#include <math.h>
#include <stdio.h>

typedef float Real;
static Real sdf2_segment(Real x, Real y, Real ax, Real ay, Real bx, Real by) {
  Real h, bb, cx, cy;
  x -= ax;
  y -= ay;
  bx -= ax;
  by -= ay;
  bb = by * by + bx * bx;
  h = (by * y + bx * x) / bb;
  h = h > 1 ? 1 : h < 0 ? 0 : h;
  cx = x - bx * h;
  cy = y - by * h;
  return cx * cx + cy * cy;
}

int main() {
  int i, j, n;
  FILE *f;
  Real sdf, ax = 2, ay = 3, bx = 5, by = 6;
  Real lo, hi, x, y, h;
  f = fopen("segment.raw", "w");
  n = 1000;
  lo = -10;
  hi = 10;
  h = (hi - lo) / (n - 1);
  for (j = 0; j < n; j++)
    for (i = 0; i < n; i++) {
      x = lo + i * h;
      y = lo + j * h;
      sdf = sdf2_segment(x, y, ax, ay, bx, by);
      sdf = sqrt(sdf);
      fwrite(&sdf, sizeof sdf, 1, f);
    }
  fclose(f);
  fprintf(stdout,
          "<Xdmf\n"
          "    Version=\"2\">\n"
          "  <Domain>\n"
          "    <Grid>\n"
          "      <Topology\n"
          "	  TopologyType=\"3DRectMesh\"\n"
          "	  Dimensions=\"1 %d %d\"/>\n"
          "    <Geometry\n"
          "        GeometryType=\"ORIGIN_DXDY\">\n"
          "      <DataItem\n"
          "         Dimensions=\"3\">\n"
          "	  %.16e\n"
          "	  %.16e\n"
          "	  0.0\n"	  
          "	 </DataItem>\n"
          "	 <DataItem\n"
          "	    Dimensions=\"3\">\n"
          "	  %.16e\n"
          "	  %.16e\n"
          "	  1.0\n"	  
          "	 </DataItem>\n"
          "    </Geometry>\n"
          "    <Attribute\n"
          "        Name=\"u\">\n"
          "      <DataItem\n"
          "          Format=\"Binary\"\n"
          "          Dimensions=\"1 %d %d\">\n"
          "         segment.raw\n"
          "        </DataItem>\n"
          "      </Attribute>\n"
          "    </Grid>\n"
          "  </Domain>\n"
          "</Xdmf>\n",
          n, n, lo, lo, h, h, n, n);
}
