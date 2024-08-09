#include <algorithm>
#include <iterator>
#include <mpi.h>
#include "AdaptTheMesh.h"
#include "advDiff.h"
#include "ComputeForces.h"
#include "HDF5Dumper.h"
#include "Helpers.h"
#include "Operator.h"
#include "PressureSingle.h"
#include "PutObjectsOnGrid.h"
#include "Shape.h"
#include "SimulationData.h"
#include "ObstacleBlock.h"
#include "Operator.h"
#include "FishUtilities.h"
#define profile( func ) do { } while (0)
using namespace cubism;

struct FishSkin
{
  const size_t Npoints;
  Real * const xSurf;
  Real * const ySurf;
  Real * const normXSurf;
  Real * const normYSurf;
  Real * const midX;
  Real * const midY;
  FishSkin(const FishSkin& c) : Npoints(c.Npoints),
    xSurf(new Real[Npoints]), ySurf(new Real[Npoints])
    , normXSurf(new Real[Npoints-1]), normYSurf(new Real[Npoints-1])
    , midX(new Real[Npoints-1]), midY(new Real[Npoints-1])
    { }

  FishSkin(const size_t N): Npoints(N),
    xSurf(new Real[Npoints]), ySurf(new Real[Npoints])
    , normXSurf(new Real[Npoints-1]), normYSurf(new Real[Npoints-1])
    , midX(new Real[Npoints-1]), midY(new Real[Npoints-1])
    { }

  ~FishSkin() { delete [] xSurf; delete [] ySurf;
      delete [] normXSurf; delete [] normYSurf; delete [] midX; delete [] midY;
  }
};

struct FishData
{
 public:
  // Length and minimal gridspacing
  const Real length, h;

  // Midline is discretized by more points in first fraction and last fraction:
  const Real fracRefined = 0.1, fracMid = 1 - 2*fracRefined;
  const Real dSmid_tgt = h / std::sqrt(2);
  const Real dSrefine_tgt = 0.125 * h;

  //// Nm should be divisible by 8, see Fish.cpp - 3)
  // thus Nmid enforced to be divisible by 8
  const int Nmid = (int)std::ceil(length * fracMid / dSmid_tgt / 8) * 8;
  const Real dSmid = length * fracMid / Nmid;

  // thus Nend enforced to be divisible by 4
  const int Nend = (int)std::ceil(fracRefined * length * 2 / (dSmid + dSrefine_tgt) / 4) * 4;
  const Real dSref = fracRefined * length * 2 / Nend - dSmid;

  const int Nm = Nmid + 2 * Nend + 1; // plus 1 because we contain 0 and L

  Real * const rS; // arclength discretization points
  Real * const rX; // coordinates of midline discretization points
  Real * const rY;
  Real * const vX; // midline discretization velocities
  Real * const vY;
  Real * const norX; // normal vector to the midline discretization points
  Real * const norY;
  Real * const vNorX;
  Real * const vNorY;
  Real * const width;

  Real linMom[2], area, J, angMom; // for diagnostics
  // start and end indices in the arrays where the fish starts and ends (to ignore the extensions when interpolating the shapes)
  FishSkin upperSkin = FishSkin(Nm);
  FishSkin lowerSkin = FishSkin(Nm);
  virtual void resetAll();

 protected:

  template<typename T>
  inline void _rotate2D(const Real Rmatrix2D[2][2], T &x, T &y) const {
    const T p[2] = {x, y};
    x = Rmatrix2D[0][0]*p[0] + Rmatrix2D[0][1]*p[1];
    y = Rmatrix2D[1][0]*p[0] + Rmatrix2D[1][1]*p[1];
  }
  template<typename T>
  inline void _translateAndRotate2D(const T pos[2], const Real Rmatrix2D[2][2], Real&x, Real&y) const {
    const Real p[2] = { x-pos[0], y-pos[1] };
    x = Rmatrix2D[0][0]*p[0] + Rmatrix2D[0][1]*p[1];
    y = Rmatrix2D[1][0]*p[0] + Rmatrix2D[1][1]*p[1];
  }

  static Real* _alloc(const int N) {
    return new Real[N];
  }
  template<typename T>
  static void _dealloc(T * ptr) {
    if(ptr not_eq nullptr) { delete [] ptr; ptr=nullptr; }
  }

  inline Real _d_ds(const int idx, const Real*const vals, const int maxidx) const {
    if(idx==0) return (vals[idx+1]-vals[idx])/(rS[idx+1]-rS[idx]);
    else if(idx==maxidx-1) return (vals[idx]-vals[idx-1])/(rS[idx]-rS[idx-1]);
    else return ( (vals[idx+1]-vals[idx])/(rS[idx+1]-rS[idx]) +
                  (vals[idx]-vals[idx-1])/(rS[idx]-rS[idx-1]) )/2;
  }
  inline Real _integrationFac1(const int idx) const {
    return 2*width[idx];
  }
  inline Real _integrationFac2(const int idx) const {
    const Real dnorXi = _d_ds(idx, norX, Nm);
    const Real dnorYi = _d_ds(idx, norY, Nm);
    return 2*std::pow(width[idx],3)*(dnorXi*norY[idx] - dnorYi*norX[idx])/3;
  }
  inline Real _integrationFac3(const int idx) const {
    return 2*std::pow(width[idx],3)/3;
  }

  virtual void _computeMidlineNormals() const;

  virtual Real _width(const Real s, const Real L) = 0;

  void _computeWidth() {
    for(int i=0; i<Nm; ++i) width[i] = _width(rS[i], length);
  }

 public:
  FishData(Real L, Real _h);
  virtual ~FishData();

  Real integrateLinearMomentum(Real CoM[2], Real vCoM[2]);
  Real integrateAngularMomentum(Real & angVel);

  void changeToCoMFrameLinear(const Real CoM_internal[2], const Real vCoM_internal[2]) const;
  void changeToCoMFrameAngular(const Real theta_internal, const Real angvel_internal) const;

  void computeSurface() const;
  void surfaceToCOMFrame(const Real theta_internal,
                         const Real CoM_internal[2]) const;
  void surfaceToComputationalFrame(const Real theta_comp,
                                   const Real CoM_interpolated[2]) const;
  void computeSkinNormals(const Real theta_comp, const Real CoM_comp[3]) const;
  void writeMidline2File(const int step_id, std::string filename);

  virtual void computeMidline(const Real time, const Real dt) = 0;
};

struct AreaSegment
{
  const Real safe_distance;
  const std::pair<int, int> s_range;
  Real w[2], c[2];
  // should be normalized and >=0:
  Real normalI[2] = {(Real)1 , (Real)0};
  Real normalJ[2] = {(Real)0 , (Real)1};
  Real objBoxLabFr[2][2] = {{0,0}, {0,0}};
  Real objBoxObjFr[2][2] = {{0,0}, {0,0}};

  AreaSegment(std::pair<int,int> sr,const Real bb[2][2],const Real safe):
  safe_distance(safe), s_range(sr),
  w{ (bb[0][1]-bb[0][0])/2 + safe, (bb[1][1]-bb[1][0])/2 + safe },
  c{ (bb[0][1]+bb[0][0])/2,        (bb[1][1]+bb[1][0])/2 }
  { assert(w[0]>0); assert(w[1]>0); }

  void changeToComputationalFrame(const Real position[2], const Real angle);
  bool isIntersectingWithAABB(const Real start[2],const Real end[2]) const;
};

struct PutFishOnBlocks
{
  const FishData & cfish;
  const Real position[2];
  const Real angle;
  const Real Rmatrix2D[2][2] = {
      {std::cos(angle), -std::sin(angle)},
      {std::sin(angle),  std::cos(angle)}
  };
  static inline Real eulerDistSq2D(const Real a[2], const Real b[2]) {
    return std::pow(a[0]-b[0],2) +std::pow(a[1]-b[1],2);
  }
  void changeVelocityToComputationalFrame(Real x[2]) const {
    const Real p[2] = {x[0], x[1]};
    x[0]=Rmatrix2D[0][0]*p[0] + Rmatrix2D[0][1]*p[1]; // rotate (around CoM)
    x[1]=Rmatrix2D[1][0]*p[0] + Rmatrix2D[1][1]*p[1];
  }
  template<typename T>
  void changeToComputationalFrame(T x[2]) const {
    const T p[2] = {x[0], x[1]};
    x[0] = Rmatrix2D[0][0]*p[0] + Rmatrix2D[0][1]*p[1];
    x[1] = Rmatrix2D[1][0]*p[0] + Rmatrix2D[1][1]*p[1];
    x[0]+= position[0]; // translate
    x[1]+= position[1];
  }
  template<typename T>
  void changeFromComputationalFrame(T x[2]) const {
    const T p[2] = { x[0]-(T)position[0], x[1]-(T)position[1] };
    // rotate back around CoM
    x[0]=Rmatrix2D[0][0]*p[0] + Rmatrix2D[1][0]*p[1];
    x[1]=Rmatrix2D[0][1]*p[0] + Rmatrix2D[1][1]*p[1];
  }

  PutFishOnBlocks(const FishData& cf, const Real pos[2],
    const Real ang): cfish(cf), position{(Real)pos[0],(Real)pos[1]}, angle(ang) { }
  virtual ~PutFishOnBlocks() {}

  void operator()(const cubism::BlockInfo& i, ScalarBlock& b,
    ObstacleBlock* const o, const std::vector<AreaSegment*>& v) const;
  virtual void constructSurface(  const cubism::BlockInfo& i, ScalarBlock& b,
    ObstacleBlock* const o, const std::vector<AreaSegment*>& v) const;
  virtual void constructInternl(  const cubism::BlockInfo& i, ScalarBlock& b,
    ObstacleBlock* const o, const std::vector<AreaSegment*>& v) const;
  virtual void signedDistanceSqrt(const cubism::BlockInfo& i, ScalarBlock& b,
    ObstacleBlock* const o, const std::vector<AreaSegment*>& v) const;
};


FishData::FishData(Real L, Real _h):
 length(L), h(_h), rS(_alloc(Nm)),rX(_alloc(Nm)),rY(_alloc(Nm)),vX(_alloc(Nm)),vY(_alloc(Nm)), norX(_alloc(Nm)), norY(_alloc(Nm)), vNorX(_alloc(Nm)), vNorY(_alloc(Nm)), width(_alloc(Nm))
{
  if( dSref <= 0 ){
    std::cout << "[CUP2D] dSref <= 0. Aborting..." << std::endl;
    fflush(0);
    abort();
  }

  rS[0] = 0;
  int k = 0;
  // extension head
  for(int i=0; i<Nend; ++i, k++)
    rS[k+1] = rS[k] + dSref +(dSmid-dSref) *         i /((Real)Nend-1.);
  // interior points
  for(int i=0; i<Nmid; ++i, k++) rS[k+1] = rS[k] + dSmid;
  // extension tail
  for(int i=0; i<Nend; ++i, k++)
    rS[k+1] = rS[k] + dSref +(dSmid-dSref) * (Nend-i-1)/((Real)Nend-1.);
  assert(k+1==Nm);
  // cout << "Discrepancy of midline length: " << std::fabs(rS[k]-L) << endl;
  rS[k] = std::min(rS[k], (Real)L);
  std::fill(rX, rX+Nm, 0);
  std::fill(rY, rY+Nm, 0);
  std::fill(vX, vX+Nm, 0);
  std::fill(vY, vY+Nm, 0);
}

FishData::~FishData()
{
  _dealloc(rS); _dealloc(rX); _dealloc(rY); _dealloc(vX); _dealloc(vY);
  _dealloc(norX); _dealloc(norY); _dealloc(vNorX); _dealloc(vNorY);
  _dealloc(width);
  // if(upperSkin not_eq nullptr) { delete upperSkin; upperSkin=nullptr; }
  // if(lowerSkin not_eq nullptr) { delete lowerSkin; lowerSkin=nullptr; }
}

void FishData::resetAll()
{
}

void FishData::writeMidline2File(const int step_id, std::string filename)
{
  char buf[500];
  sprintf(buf, "%s_midline_%07d.txt", filename.c_str(), step_id);
  FILE * f = fopen(buf, "a");
  fprintf(f, "s x y vX vY\n");
  for (int i=0; i<Nm; i++) {
    //dummy.changeToComputationalFrame(temp);
    //dummy.changeVelocityToComputationalFrame(udef);
    fprintf(f, "%g %g %g %g %g %g\n", (double)rS[i],(double)rX[i],(double)rY[i],(double)vX[i],(double)vY[i],(double)width[i]);
  }
  fflush(0);
}

void FishData::_computeMidlineNormals() const
{
  #pragma omp parallel for schedule(static)
  for(int i=0; i<Nm-1; i++) {
    const auto ds = rS[i+1]-rS[i];
    const auto tX = rX[i+1]-rX[i];
    const auto tY = rY[i+1]-rY[i];
    const auto tVX = vX[i+1]-vX[i];
    const auto tVY = vY[i+1]-vY[i];
    norX[i] = -tY/ds;
    norY[i] =  tX/ds;
    vNorX[i] = -tVY/ds;
    vNorY[i] =  tVX/ds;
  }
  norX[Nm-1] = norX[Nm-2];
  norY[Nm-1] = norY[Nm-2];
  vNorX[Nm-1] = vNorX[Nm-2];
  vNorY[Nm-1] = vNorY[Nm-2];
}

Real FishData::integrateLinearMomentum(Real CoM[2], Real vCoM[2])
{
  // already worked out the integrals for r, theta on paper
  // remaining integral done with composite trapezoidal rule
  // minimize rhs evaluations --> do first and last point separately
  Real _area=0, _cmx=0, _cmy=0, _lmx=0, _lmy=0;
  #pragma omp parallel for schedule(static) reduction(+:_area,_cmx,_cmy,_lmx,_lmy)
  for(int i=0; i<Nm; ++i)
  {
    const Real ds = (i==0) ? rS[1]-rS[0] :
        ((i==Nm-1) ? rS[Nm-1]-rS[Nm-2] :rS[i+1]-rS[i-1]);
    const Real fac1 = _integrationFac1(i);
    const Real fac2 = _integrationFac2(i);
    _area +=                        fac1 *ds/2;
    _cmx  += (rX[i]*fac1 +  norX[i]*fac2)*ds/2;
    _cmy  += (rY[i]*fac1 +  norY[i]*fac2)*ds/2;
    _lmx  += (vX[i]*fac1 + vNorX[i]*fac2)*ds/2;
    _lmy  += (vY[i]*fac1 + vNorY[i]*fac2)*ds/2;
  }
  area      = _area;
  CoM[0]    = _cmx;
  CoM[1]    = _cmy;
  linMom[0] = _lmx;
  linMom[1] = _lmy;
  assert(area> std::numeric_limits<Real>::epsilon());
  CoM[0] /= area;
  CoM[1] /= area;
  vCoM[0] = linMom[0]/area;
  vCoM[1] = linMom[1]/area;
  //printf("%f %f %f %f %f\n",CoM[0],CoM[1],vCoM[0],vCoM[1], vol);
  return area;
}

Real FishData::integrateAngularMomentum(Real& angVel)
{
  // assume we have already translated CoM and vCoM to nullify linear momentum
  // already worked out the integrals for r, theta on paper
  // remaining integral done with composite trapezoidal rule
  // minimize rhs evaluations --> do first and last point separately
  Real _J = 0, _am = 0;
  #pragma omp parallel for reduction(+:_J,_am) schedule(static)
  for(int i=0; i<Nm; ++i)
  {
    const Real ds =   (i==   0) ? rS[1]   -rS[0] :
                    ( (i==Nm-1) ? rS[Nm-1]-rS[Nm-2]
                                : rS[i+1] -rS[i-1] );
    const Real fac1 = _integrationFac1(i);
    const Real fac2 = _integrationFac2(i);
    const Real fac3 = _integrationFac3(i);
    const Real tmp_M = (rX[i]*vY[i] - rY[i]*vX[i])*fac1
      + (rX[i]*vNorY[i] -rY[i]*vNorX[i] +vY[i]*norX[i] -vX[i]*norY[i])*fac2
      + (norX[i]*vNorY[i] - norY[i]*vNorX[i])*fac3;

    const Real tmp_J = (rX[i]*rX[i]   + rY[i]*rY[i]  )*fac1
                   + 2*(rX[i]*norX[i] + rY[i]*norY[i])*fac2 + fac3;

    _am += tmp_M*ds/2;
    _J  += tmp_J*ds/2;
  }
  J      = _J;
  angMom = _am;
  assert(J>std::numeric_limits<Real>::epsilon());
  angVel = angMom/J;
  return J;
}

void FishData::changeToCoMFrameLinear(const Real Cin[2],
                                      const Real vCin[2]) const
{
  #pragma omp parallel for schedule(static)
  for(int i=0;i<Nm;++i) { // subtract internal CM and vCM
   rX[i] -= Cin[0]; rY[i] -= Cin[1]; vX[i] -= vCin[0]; vY[i] -= vCin[1];
  }
}

void FishData::changeToCoMFrameAngular(const Real Ain, const Real vAin) const
{
  const Real Rmatrix2D[2][2] = {
    { std::cos(Ain),-std::sin(Ain)},
    { std::sin(Ain), std::cos(Ain)}
  };

  #pragma omp parallel for schedule(static)
  for(int i=0;i<Nm;++i) { // subtract internal angvel and rotate position by ang
    vX[i] += vAin*rY[i];
    vY[i] -= vAin*rX[i];
    _rotate2D(Rmatrix2D, rX[i], rY[i]);
    _rotate2D(Rmatrix2D, vX[i], vY[i]);
  }
  _computeMidlineNormals();
}

void FishData::computeSurface() const
{
  // Compute surface points by adding width to the midline points
  #pragma omp parallel for schedule(static)
  for(size_t i=0; i<lowerSkin.Npoints; ++i)
  {
    Real norm[2] = {norX[i], norY[i]};
    Real const norm_mod1 = std::sqrt(norm[0]*norm[0] + norm[1]*norm[1]);
    norm[0] /= norm_mod1;
    norm[1] /= norm_mod1;
    assert(width[i] >= 0);
    lowerSkin.xSurf[i] = rX[i] - width[i] * norm[0];
    lowerSkin.ySurf[i] = rY[i] - width[i] * norm[1];
    upperSkin.xSurf[i] = rX[i] + width[i] * norm[0];
    upperSkin.ySurf[i] = rY[i] + width[i] * norm[1];
  }
}

void FishData::computeSkinNormals(const Real theta_comp,
                                  const Real CoM_comp[3]) const
{
  const Real Rmatrix2D[2][2]={ { std::cos(theta_comp),-std::sin(theta_comp)},
                               { std::sin(theta_comp), std::cos(theta_comp)} };

  for(int i=0; i<Nm; ++i) {
    _rotate2D(Rmatrix2D, rX[i], rY[i]);
    _rotate2D(Rmatrix2D, norX[i], norY[i]);
    rX[i] += CoM_comp[0];
    rY[i] += CoM_comp[1];
  }

  // Compute midpoints as they will be pressure targets
  #pragma omp parallel for
  for(size_t i=0; i<lowerSkin.Npoints-1; ++i)
  {
    lowerSkin.midX[i] = (lowerSkin.xSurf[i] + lowerSkin.xSurf[i+1])/2;
    upperSkin.midX[i] = (upperSkin.xSurf[i] + upperSkin.xSurf[i+1])/2;
    lowerSkin.midY[i] = (lowerSkin.ySurf[i] + lowerSkin.ySurf[i+1])/2;
    upperSkin.midY[i] = (upperSkin.ySurf[i] + upperSkin.ySurf[i+1])/2;

    lowerSkin.normXSurf[i]=  (lowerSkin.ySurf[i+1]-lowerSkin.ySurf[i]);
    upperSkin.normXSurf[i]=  (upperSkin.ySurf[i+1]-upperSkin.ySurf[i]);
    lowerSkin.normYSurf[i]= -(lowerSkin.xSurf[i+1]-lowerSkin.xSurf[i]);
    upperSkin.normYSurf[i]= -(upperSkin.xSurf[i+1]-upperSkin.xSurf[i]);

    const Real normL = std::sqrt( std::pow(lowerSkin.normXSurf[i],2) +
                                  std::pow(lowerSkin.normYSurf[i],2) );
    const Real normU = std::sqrt( std::pow(upperSkin.normXSurf[i],2) +
                                  std::pow(upperSkin.normYSurf[i],2) );

    lowerSkin.normXSurf[i] /= normL;
    upperSkin.normXSurf[i] /= normU;
    lowerSkin.normYSurf[i] /= normL;
    upperSkin.normYSurf[i] /= normU;

    //if too close to the head or tail, consider a point further in, so that we are pointing out for sure
    const int ii = (i<8) ? 8 : ((i > lowerSkin.Npoints-9) ? lowerSkin.Npoints-9 : i);

    const Real dirL =
      lowerSkin.normXSurf[i] * (lowerSkin.midX[i]-rX[ii]) +
      lowerSkin.normYSurf[i] * (lowerSkin.midY[i]-rY[ii]);
    const Real dirU =
      upperSkin.normXSurf[i] * (upperSkin.midX[i]-rX[ii]) +
      upperSkin.normYSurf[i] * (upperSkin.midY[i]-rY[ii]);

    if(dirL < 0) {
        lowerSkin.normXSurf[i] *= -1.0;
        lowerSkin.normYSurf[i] *= -1.0;
    }
    if(dirU < 0) {
        upperSkin.normXSurf[i] *= -1.0;
        upperSkin.normYSurf[i] *= -1.0;
    }
  }
}

void FishData::surfaceToCOMFrame(const Real theta_internal,
                                 const Real CoM_internal[2]) const
{
  const Real Rmatrix2D[2][2] = {
    { std::cos(theta_internal),-std::sin(theta_internal)},
    { std::sin(theta_internal), std::cos(theta_internal)}
  };
  // Surface points rotation and translation

  #pragma omp parallel for schedule(static)
  for(size_t i=0; i<upperSkin.Npoints; ++i)
  //for(int i=0; i<upperSkin.Npoints-1; ++i)
  {
    upperSkin.xSurf[i] -= CoM_internal[0];
    upperSkin.ySurf[i] -= CoM_internal[1];
    _rotate2D(Rmatrix2D, upperSkin.xSurf[i], upperSkin.ySurf[i]);
    lowerSkin.xSurf[i] -= CoM_internal[0];
    lowerSkin.ySurf[i] -= CoM_internal[1];
    _rotate2D(Rmatrix2D, lowerSkin.xSurf[i], lowerSkin.ySurf[i]);
  }
}

void FishData::surfaceToComputationalFrame(const Real theta_comp,
                                           const Real CoM_interpolated[2]) const
{
  const Real Rmatrix2D[2][2] = {
    { std::cos(theta_comp),-std::sin(theta_comp)},
    { std::sin(theta_comp), std::cos(theta_comp)}
  };

  #pragma omp parallel for schedule(static)
  for(size_t i=0; i<upperSkin.Npoints; ++i)
  {
    _rotate2D(Rmatrix2D, upperSkin.xSurf[i], upperSkin.ySurf[i]);
    upperSkin.xSurf[i] += CoM_interpolated[0];
    upperSkin.ySurf[i] += CoM_interpolated[1];
    _rotate2D(Rmatrix2D, lowerSkin.xSurf[i], lowerSkin.ySurf[i]);
    lowerSkin.xSurf[i] += CoM_interpolated[0];
    lowerSkin.ySurf[i] += CoM_interpolated[1];
  }
}

void AreaSegment::changeToComputationalFrame(const Real pos[2],const Real angle)
{
  // we are in CoM frame and change to comp frame --> first rotate around CoM (which is at (0,0) in CoM frame), then update center
  const Real Rmatrix2D[2][2] = {
      {std::cos(angle), -std::sin(angle)},
      {std::sin(angle),  std::cos(angle)}
  };
  const Real p[2] = {c[0],c[1]};

  const Real nx[2] = {normalI[0],normalI[1]};
  const Real ny[2] = {normalJ[0],normalJ[1]};

  for(int i=0;i<2;++i) {
      c[i] = Rmatrix2D[i][0]*p[0] + Rmatrix2D[i][1]*p[1];

      normalI[i] = Rmatrix2D[i][0]*nx[0] + Rmatrix2D[i][1]*nx[1];
      normalJ[i] = Rmatrix2D[i][0]*ny[0] + Rmatrix2D[i][1]*ny[1];
  }

  c[0] += pos[0];
  c[1] += pos[1];

  const Real magI = std::sqrt(normalI[0]*normalI[0]+normalI[1]*normalI[1]);
  const Real magJ = std::sqrt(normalJ[0]*normalJ[0]+normalJ[1]*normalJ[1]);
  assert(magI > std::numeric_limits<Real>::epsilon());
  assert(magJ > std::numeric_limits<Real>::epsilon());
  const Real invMagI = 1/magI, invMagJ = 1/magJ;

  for(int i=0;i<2;++i) {
    // also take absolute value since thats what we need when doing intersection checks later
    normalI[i]=std::fabs(normalI[i])*invMagI;
    normalJ[i]=std::fabs(normalJ[i])*invMagJ;
  }

  assert(normalI[0]>=0 && normalI[1]>=0);
  assert(normalJ[0]>=0 && normalJ[1]>=0);

  // Find the x,y,z max extents in lab frame ( exploit normal(I,J,K)[:] >=0 )
  const Real widthXvec[] = {w[0]*normalI[0], w[0]*normalI[1]};
  const Real widthYvec[] = {w[1]*normalJ[0], w[1]*normalJ[1]};

  for(int i=0; i<2; ++i) {
    objBoxLabFr[i][0] = c[i] -widthXvec[i] -widthYvec[i];
    objBoxLabFr[i][1] = c[i] +widthXvec[i] +widthYvec[i];
    objBoxObjFr[i][0] = c[i] -w[i];
    objBoxObjFr[i][1] = c[i] +w[i];
  }
}

bool AreaSegment::isIntersectingWithAABB(const Real start[2],const Real end[2]) const
{
  // Remember: Incoming coordinates are cell centers, not cell faces
  //start and end are two diagonally opposed corners of grid block
  // GN halved the safety here but added it back to w[] in prepare
  const Real AABB_w[2] = { //half block width + safe distance
      (end[0] - start[0])/2 + safe_distance,
      (end[1] - start[1])/2 + safe_distance
  };

  const Real AABB_c[2] = { //block center
    (end[0] + start[0])/2, (end[1] + start[1])/2
  };

  const Real AABB_box[2][2] = {
    {AABB_c[0] - AABB_w[0],  AABB_c[0] + AABB_w[0]},
    {AABB_c[1] - AABB_w[1],  AABB_c[1] + AABB_w[1]}
  };

  assert(AABB_w[0]>0 && AABB_w[1]>0);

  // Now Identify the ones that do not intersect
  Real intersectionLabFrame[2][2] = {
  { std::max(objBoxLabFr[0][0],AABB_box[0][0]),
    std::min(objBoxLabFr[0][1],AABB_box[0][1]) },
  { std::max(objBoxLabFr[1][0],AABB_box[1][0]),
    std::min(objBoxLabFr[1][1],AABB_box[1][1]) }
  };

  if ( intersectionLabFrame[0][1] - intersectionLabFrame[0][0] < 0
    || intersectionLabFrame[1][1] - intersectionLabFrame[1][0] < 0 )
    return false;

  // This is x-width of box, expressed in fish frame
  const Real widthXbox[2] = {AABB_w[0]*normalI[0], AABB_w[0]*normalJ[0]};
  // This is y-width of box, expressed in fish frame
  const Real widthYbox[2] = {AABB_w[1]*normalI[1], AABB_w[1]*normalJ[1]};

  const Real boxBox[2][2] = {
    { AABB_c[0] -widthXbox[0] -widthYbox[0],
      AABB_c[0] +widthXbox[0] +widthYbox[0]},
    { AABB_c[1] -widthXbox[1] -widthYbox[1],
      AABB_c[1] +widthXbox[1] +widthYbox[1]}
  };

  Real intersectionFishFrame[2][2] = {
   { std::max(boxBox[0][0],objBoxObjFr[0][0]),
     std::min(boxBox[0][1],objBoxObjFr[0][1])},
   { std::max(boxBox[1][0],objBoxObjFr[1][0]),
     std::min(boxBox[1][1],objBoxObjFr[1][1])}
  };

  if ( intersectionFishFrame[0][1] - intersectionFishFrame[0][0] < 0
    || intersectionFishFrame[1][1] - intersectionFishFrame[1][0] < 0)
    return false;

  return true;
}

void PutFishOnBlocks::operator()(const BlockInfo& i, ScalarBlock& b,
  ObstacleBlock* const o, const std::vector<AreaSegment*>& v) const
{
  //std::chrono::time_point<std::chrono::high_resolution_clock> t0, t1, t2, t3;
  //t0 = std::chrono::high_resolution_clock::now();
  constructSurface(i, b, o, v);
  //t1 = std::chrono::high_resolution_clock::now();
  constructInternl(i, b, o, v);
  //t2 = std::chrono::high_resolution_clock::now();
  signedDistanceSqrt(i, b, o, v);
  //t3 = std::chrono::high_resolution_clock::now();
  //printf("%g %g %g\n",std::chrono::duration<Real>(t1-t0).count(),
  //                    std::chrono::duration<Real>(t2-t1).count(),
  //                    std::chrono::duration<Real>(t3-t2).count());
}

void PutFishOnBlocks::signedDistanceSqrt(const BlockInfo& info, ScalarBlock& b,
  ObstacleBlock* const o, const std::vector<AreaSegment*>& vSegments) const
{
  // finalize signed distance function in tmpU
  static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
  for(int iy=0; iy<ScalarBlock::sizeY; iy++)
  for(int ix=0; ix<ScalarBlock::sizeX; ix++) {
    const Real normfac = o->chi[iy][ix] > EPS ? o->chi[iy][ix] : 1;
    o->udef[iy][ix][0] /= normfac; o->udef[iy][ix][1] /= normfac;
    // change from signed squared distance function to normal sdf
    o->dist[iy][ix] = o->dist[iy][ix]>=0 ?  std::sqrt( o->dist[iy][ix])
                                         : -std::sqrt(-o->dist[iy][ix]);
    b(ix,iy).s = std::max(b(ix,iy).s, o->dist[iy][ix]);;
  }
  static constexpr int BS[2] = {ScalarBlock::sizeX, ScalarBlock::sizeY};
  std::fill(o->chi [0], o->chi [0] + BS[1]*BS[0],  0);
}

void PutFishOnBlocks::constructSurface(const BlockInfo& info, ScalarBlock& b,
  ObstacleBlock* const o, const std::vector<AreaSegment*>& vSegments) const
{
  Real org[2];
  info.pos(org, 0, 0);
  #ifndef NDEBUG
    static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
  #endif
  const Real h = info.h, invh = 1.0/info.h;
  const Real * const rX = cfish.rX, * const norX = cfish.norX;
  const Real * const rY = cfish.rY, * const norY = cfish.norY;
  const Real * const vX = cfish.vX, * const vNorX = cfish.vNorX;
  const Real * const vY = cfish.vY, * const vNorY = cfish.vNorY;
  const Real * const width = cfish.width;
  static constexpr int BS[2] = {ScalarBlock::sizeX, ScalarBlock::sizeY};
  std::fill(o->dist[0], o->dist[0] + BS[1]*BS[0], -1);
  std::fill(o->chi [0], o->chi [0] + BS[1]*BS[0],  0);

  // construct the shape (P2M with min(distance) as kernel) onto ObstacleBlock
  for(int i=0; i<(int)vSegments.size(); ++i)
  {
    //iterate over segments contained in the vSegm intersecting this block:
    const int firstSegm = std::max(vSegments[i]->s_range.first,           1);
    const int lastSegm =  std::min(vSegments[i]->s_range.second, cfish.Nm-2);
    for(int ss = firstSegm; ss <= lastSegm; ++ss)
    {
      assert(width[ss]>0);
      //for each segment, we have one point to left and right of midl
      for(int signp = -1; signp <= 1; signp+=2)
      {
        // create a surface point
        // special treatment of tail (width = 0 --> no ellipse, just line)
        Real myP[2]= { rX[ss+0] +width[ss+0]*signp*norX[ss+0],
                       rY[ss+0] +width[ss+0]*signp*norY[ss+0]  };
        changeToComputationalFrame(myP);
        const int iap[2] = {  (int)std::floor((myP[0]-org[0])*invh),
                              (int)std::floor((myP[1]-org[1])*invh) };
        if(iap[0]+3 <= 0 || iap[0]-1 >= BS[0]) continue; // NearNeigh loop
        if(iap[1]+3 <= 0 || iap[1]-1 >= BS[1]) continue; // does not intersect

        Real pP[2] = { rX[ss+1] +width[ss+1]*signp*norX[ss+1],
                       rY[ss+1] +width[ss+1]*signp*norY[ss+1]  };
        changeToComputationalFrame(pP);
        Real pM[2] = { rX[ss-1] +width[ss-1]*signp*norX[ss-1],
                       rY[ss-1] +width[ss-1]*signp*norY[ss-1]  };
        changeToComputationalFrame(pM);
        Real udef[2] = { vX[ss+0] +width[ss+0]*signp*vNorX[ss+0],
                         vY[ss+0] +width[ss+0]*signp*vNorY[ss+0]    };
        changeVelocityToComputationalFrame(udef);
        // support is two points left, two points right --> Towers Chi will
        // be one point left, one point right, but needs SDF wider
        for(int sy =std::max(0, iap[1]-2); sy <std::min(iap[1]+4, BS[1]); ++sy)
        for(int sx =std::max(0, iap[0]-2); sx <std::min(iap[0]+4, BS[0]); ++sx)
        {
          Real p[2];
          info.pos(p, sx, sy);
          const Real dist0 = eulerDistSq2D(p, myP);
          const Real distP = eulerDistSq2D(p,  pP);
          const Real distM = eulerDistSq2D(p,  pM);

          if(std::fabs(o->dist[sy][sx])<std::min({dist0,distP,distM}))
            continue;

          changeFromComputationalFrame(p);
          #ifndef NDEBUG // check that change of ref frame does not affect dist
            const Real p0[2] = {rX[ss] +width[ss]*signp*norX[ss],
                                rY[ss] +width[ss]*signp*norY[ss] };
            const Real distC = eulerDistSq2D(p, p0);
            assert(std::fabs(distC-dist0)<EPS);
          #endif

          int close_s = ss, secnd_s = ss + (distP<distM? 1 : -1);
          Real dist1 = dist0, dist2 = distP<distM? distP : distM;
          if(distP < dist0 || distM < dist0) { // switch nearest surf point
            dist1 = dist2; dist2 = dist0;
            close_s = secnd_s; secnd_s = ss;
          }

          const Real dSsq = std::pow(rX[close_s]-rX[secnd_s], 2)
                           +std::pow(rY[close_s]-rY[secnd_s], 2);
          assert(dSsq > 2.2e-16);
          const Real cnt2ML = std::pow( width[close_s],2);
          const Real nxt2ML = std::pow( width[secnd_s],2);
          const Real safeW = std::max( width[close_s], width[secnd_s] ) + 2*h;
          const Real xMidl[2] = {rX[close_s], rY[close_s]};
          const Real grd2ML = eulerDistSq2D(p, xMidl);
          const Real diffH = std::fabs( width[close_s] - width[secnd_s] );
          Real sign2d = 0;
          //If width changes slowly or if point is very far away, this is safer:
          if( dSsq > diffH*diffH || grd2ML > safeW*safeW )
          { // if no abrupt changes in width we use nearest neighbour
            sign2d = grd2ML > cnt2ML ? -1 : 1;
          }
          else
          {
            // else we model the span between ellipses as a spherical segment
            // http://mathworld.wolfram.com/SphericalSegment.html
            const Real corr = 2*std::sqrt(cnt2ML*nxt2ML);
            const Real Rsq = (cnt2ML +nxt2ML -corr +dSsq) //radius of the sphere
                            *(cnt2ML +nxt2ML +corr +dSsq)/4/dSsq;
            const Real maxAx = std::max(cnt2ML, nxt2ML);
            const int idAx1 = cnt2ML> nxt2ML? close_s : secnd_s;
            const int idAx2 = idAx1==close_s? secnd_s : close_s;
            // 'submerged' fraction of radius:
            const Real d = std::sqrt((Rsq - maxAx)/dSsq); // (divided by ds)
            // position of the centre of the sphere:
            const Real xCentr[2] = {rX[idAx1] +(rX[idAx1]-rX[idAx2])*d,
                                    rY[idAx1] +(rY[idAx1]-rY[idAx2])*d};
            const Real grd2Core = eulerDistSq2D(p, xCentr);
            sign2d = grd2Core > Rsq ? -1 : 1; // as always, neg outside
          }

          if(std::fabs(o->dist[sy][sx]) > dist1) {
            const Real W = 1 - std::min((Real)1, std::sqrt(dist1) * (invh / 3));
            // W behaves like hat interpolation kernel that is used for internal
            // fish points. Introducing W (used to be W=1) smoothens transition
            // from surface to internal points. In fact, later we plus equal
            // udef*hat of internal points. If hat>0, point should behave like
            // internal point, meaning that fish-section udef rotation should
            // multiply distance from midline instead of entire half-width.
            // Remember that uder will become udef / chi, so W simplifies out.
            assert(W >= 0);
            o->udef[sy][sx][0] = W * udef[0];
            o->udef[sy][sx][1] = W * udef[1];
            o->dist[sy][sx] = sign2d * dist1;
            o->chi [sy][sx] = W;
          }
          // Not chi yet, I stored squared distance from analytical boundary
          // distSq is updated only if curr value is smaller than the old one
        }
      }
    }
  }
}

void PutFishOnBlocks::constructInternl(const BlockInfo& info, ScalarBlock& b,
  ObstacleBlock* const o, const std::vector<AreaSegment*>& vSegments) const
{
  Real org[2];
  info.pos(org, 0, 0);
  const Real h = info.h, invh = 1.0/info.h;
  static constexpr int BS[2] = {ScalarBlock::sizeX, ScalarBlock::sizeY};
  // construct the deformation velocities (P2M with hat function as kernel)
  for(int i=0; i<(int)vSegments.size(); ++i)
  {
    const int firstSegm = std::max(vSegments[i]->s_range.first,           1);
    const int lastSegm =  std::min(vSegments[i]->s_range.second, cfish.Nm-2);
    for(int ss=firstSegm; ss<=lastSegm; ++ss)
    {
      // P2M udef of a slice at this s
      const Real myWidth = cfish.width[ss];
      assert(myWidth > 0);
      //here we process also all inner points. Nw to the left and right of midl
      // add xtension here to make sure we have it in each direction:
      const int Nw = std::floor(myWidth/h); //floor bcz we already did interior
      for(int iw = -Nw+1; iw < Nw; ++iw)
      {
        const Real offsetW = iw * h;
        Real xp[2] = { cfish.rX[ss] + offsetW*cfish.norX[ss],
                       cfish.rY[ss] + offsetW*cfish.norY[ss] };
        changeToComputationalFrame(xp);
        xp[0] = (xp[0]-org[0])*invh; // how many grid points from this block
        xp[1] = (xp[1]-org[1])*invh; // origin is this fishpoint located at?
        const Real ap[2] = { std::floor(xp[0]), std::floor(xp[1]) };
        const int iap[2] = { (int)ap[0], (int)ap[1] };
        if(iap[0]+2 <= 0 || iap[0] >= BS[0]) continue; // hatP2M loop
        if(iap[1]+2 <= 0 || iap[1] >= BS[1]) continue; // does not intersect

        Real udef[2] = { cfish.vX[ss] + offsetW*cfish.vNorX[ss],
                         cfish.vY[ss] + offsetW*cfish.vNorY[ss] };
        changeVelocityToComputationalFrame(udef);
        Real wghts[2][2]; // P2M weights
        for(int c=0; c<2; ++c) {
          const Real t[2] = { // we floored, hat between xp and grid point +-1
              std::fabs(xp[c] -ap[c]), std::fabs(xp[c] -(ap[c] +1))
          };
          wghts[c][0] = 1 - t[0];
          wghts[c][1] = 1 - t[1];
        }

        for(int idy =std::max(0, iap[1]); idy <std::min(iap[1]+2, BS[1]); ++idy)
        for(int idx =std::max(0, iap[0]); idx <std::min(iap[0]+2, BS[0]); ++idx)
        {
          const int sx = idx - iap[0], sy = idy - iap[1];
          const Real wxwy = wghts[1][sy] * wghts[0][sx];
          assert(idx>=0 && idx<ScalarBlock::sizeX && wxwy>=0);
          assert(idy>=0 && idy<ScalarBlock::sizeY && wxwy<=1);
          o->udef[idy][idx][0] += wxwy*udef[0];
          o->udef[idy][idx][1] += wxwy*udef[1];
          o->chi [idy][idx] += wxwy;
          // set sign for all interior points
          static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
          if( std::fabs(o->dist[idy][idx]+1) < EPS ) o->dist[idy][idx] = 1;
        }
      }
    }
  }
}

class FactoryFileLineParser: public cubism::ArgumentParser
{
protected:
    // from stackoverflow

    // trim from start
    inline std::string &ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
    }

    // trim from end
    inline std::string &rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
    }

    // trim from both ends
    inline std::string &trim(std::string &s) {
        return ltrim(rtrim(s));
    }

public:

    FactoryFileLineParser(std::istringstream & is_line)
    : cubism::ArgumentParser(0, NULL, '#') // last char is comment leader
    {
        std::string key,value;
        while( std::getline(is_line, key, '=') )
        {
            if( std::getline(is_line, value, ' ') )
            {
                // add "-" because then we can use the same code for parsing factory as command lines
                //mapArguments["-"+trim(key)] = Value(trim(value));
                mapArguments[trim(key)] = cubism::Value(trim(value));
            }
        }

        mute();
    }
};


struct FishData;
class Fish: public Shape
{
 public:
  const Real length, Tperiod, phaseShift;
  FishData * myFish = nullptr;
 protected:
  Real area_internal = 0, J_internal = 0;
  Real CoM_internal[2] = {0, 0}, vCoM_internal[2] = {0, 0};
  Real theta_internal = 0, angvel_internal = 0, angvel_internal_prev = 0;

  Fish(SimulationData&s, cubism::ArgumentParser&p, Real C[2]) : Shape(s,p,C),
  length(p("-L").asDouble(0.1)), Tperiod(p("-T").asDouble(1)),
  phaseShift(p("-phi").asDouble(0))  {}
  virtual ~Fish() override;

 public:
  Real getCharLength() const override {
    return length;
  }
  void removeMoments(const std::vector<cubism::BlockInfo>& vInfo) override;
  virtual void resetAll() override;
  virtual void updatePosition(Real dt) override;
  virtual void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  virtual void saveRestart( FILE * f ) override;
  virtual void loadRestart( FILE * f ) override;
};

void Fish::create(const std::vector<BlockInfo>& vInfo)
{
  //// 0) clear obstacle blocks
  for(auto & entry : obstacleBlocks) 
    delete entry;
  obstacleBlocks.clear();

  //// 1) Update Midline and compute surface
  assert(myFish!=nullptr);
  profile(push_start("midline"));
  myFish->computeMidline(sim.time, sim.dt);
  myFish->computeSurface();
  profile(pop_stop());

  if( sim.rank == 0 && sim.bDump() )
    myFish->writeMidline2File(0, "appending");

  //// 2) Integrate Linear and Angular Momentum and shift Fish accordingly
  profile(push_start("2dmoments"));
  // returns area, CoM_internal, vCoM_internal:
  area_internal = myFish->integrateLinearMomentum(CoM_internal, vCoM_internal);
  // takes CoM_internal, vCoM_internal, puts CoM in and nullifies  lin mom:
  myFish->changeToCoMFrameLinear(CoM_internal, vCoM_internal);
  angvel_internal_prev = angvel_internal;
  // returns mom of intertia and angvel:
  J_internal = myFish->integrateAngularMomentum(angvel_internal);
  // rotates fish midline to current angle and removes angular moment:
  myFish->changeToCoMFrameAngular(theta_internal, angvel_internal);
  #if 0 //ndef NDEBUG
  {
    Real dummy_CoM_internal[2], dummy_vCoM_internal[2], dummy_angvel_internal;
    // check that things are zero
    const Real area_internal_check =
    myFish->integrateLinearMomentum(dummy_CoM_internal, dummy_vCoM_internal);
    myFish->integrateAngularMomentum(dummy_angvel_internal);
    const Real EPS = 10*std::numeric_limits<Real>::epsilon();
    assert(std::fabs(dummy_CoM_internal[0])<EPS);
    assert(std::fabs(dummy_CoM_internal[1])<EPS);
    assert(std::fabs(myFish->linMom[0])<EPS);
    assert(std::fabs(myFish->linMom[1])<EPS);
    assert(std::fabs(myFish->angMom)<EPS);
    assert(std::fabs(area_internal - area_internal_check) < EPS);
  }
  #endif
  profile(pop_stop());
  myFish->surfaceToCOMFrame(theta_internal, CoM_internal);

  //// 3) Create Bounding Boxes around Fish
  //- performance of create seems to decrease if VolumeSegment_OBB are bigger
  //- this code groups segments together and finds a bounding box (maximal
  //  x and y coords) to then be able to check intersection with cartesian grid
  const int Nsegments = (myFish->Nm-1)/8, Nm = myFish->Nm;
  assert((Nm-1)%Nsegments==0);
  profile(push_start("boxes"));

  std::vector<AreaSegment*> vSegments(Nsegments, nullptr);
  const Real h = sim.getH();
  #pragma omp parallel for schedule(static)
  for(int i=0; i<Nsegments; ++i)
  {
    const int next_idx = (i+1)*(Nm-1)/Nsegments, idx = i * (Nm-1)/Nsegments;
    // find bounding box based on this
    Real bbox[2][2] = {{1e9, -1e9}, {1e9, -1e9}};
    for(int ss=idx; ss<=next_idx; ++ss)
    {
      const Real xBnd[2]={myFish->rX[ss] -myFish->norX[ss]*myFish->width[ss],
                          myFish->rX[ss] +myFish->norX[ss]*myFish->width[ss]};
      const Real yBnd[2]={myFish->rY[ss] -myFish->norY[ss]*myFish->width[ss],
                          myFish->rY[ss] +myFish->norY[ss]*myFish->width[ss]};
      const Real maxX=std::max(xBnd[0],xBnd[1]), minX=std::min(xBnd[0],xBnd[1]);
      const Real maxY=std::max(yBnd[0],yBnd[1]), minY=std::min(yBnd[0],yBnd[1]);
      bbox[0][0] = std::min(bbox[0][0], minX);
      bbox[0][1] = std::max(bbox[0][1], maxX);
      bbox[1][0] = std::min(bbox[1][0], minY);
      bbox[1][1] = std::max(bbox[1][1], maxY);
    }
    const Real DD = 4*h; //two points on each side
    //const Real safe_distance = info.h; // one point on each side
    AreaSegment*const tAS=new AreaSegment(std::make_pair(idx,next_idx),bbox,DD);
    tAS->changeToComputationalFrame(center, orientation);
    vSegments[i] = tAS;
  }
  profile(pop_stop());

  //// 4) Interpolate shape with computational grid
  profile(push_start("intersect"));
  const auto N = vInfo.size();
  std::vector<std::vector<AreaSegment*>*> segmentsPerBlock (N, nullptr);
  obstacleBlocks = std::vector<ObstacleBlock*> (N, nullptr);

  #pragma omp parallel for schedule(static)
  for(size_t i=0; i<vInfo.size(); ++i)
  {
    const BlockInfo & info = vInfo[i];
    Real pStart[2], pEnd[2];
    info.pos(pStart, 0, 0);
    info.pos(pEnd, ScalarBlock::sizeX-1, ScalarBlock::sizeY-1);

    for(size_t s=0; s<vSegments.size(); ++s)
      if(vSegments[s]->isIntersectingWithAABB(pStart,pEnd))
      {
        if(segmentsPerBlock[info.blockID] == nullptr)
          segmentsPerBlock[info.blockID] = new std::vector<AreaSegment*>(0);
        segmentsPerBlock[info.blockID]->push_back(vSegments[s]);
      }

    // allocate new blocks if necessary
    if(segmentsPerBlock[info.blockID] not_eq nullptr)
    {
      assert(obstacleBlocks[info.blockID] == nullptr);
      ObstacleBlock * const block = new ObstacleBlock();
      assert(block not_eq nullptr);
      obstacleBlocks[info.blockID] = block;
      block->clear();
    }
  }
  assert(not segmentsPerBlock.empty());
  assert(segmentsPerBlock.size() == obstacleBlocks.size());
  profile(pop_stop());

  #pragma omp parallel
  {
    const PutFishOnBlocks putfish(*myFish, center, orientation);

    #pragma omp for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++)
    {
      const auto pos = segmentsPerBlock[vInfo[i].blockID];
      if(pos not_eq nullptr)
      {
        ObstacleBlock*const block = obstacleBlocks[vInfo[i].blockID];
        assert(block not_eq nullptr);
        putfish(vInfo[i], *(ScalarBlock*)vInfo[i].ptrBlock, block, *pos);
      }
    }
  }

  // clear vSegments
  for(auto & E : vSegments) { if(E not_eq nullptr) delete E; }
  for(auto & E : segmentsPerBlock)  { if(E not_eq nullptr) delete E; }

  profile(pop_stop());
  if (sim.step % 100 == 0 && sim.verbose)
  {
    profile(printSummary());
    profile(reset());
  }
}

void Fish::updatePosition(Real dt)
{
  // update position and angles
  Shape::updatePosition(dt);
  theta_internal -= dt*angvel_internal; // negative: we subtracted this angvel
}

void Fish::resetAll()
{
  CoM_internal[0] = 0; CoM_internal[1] = 0;
  vCoM_internal[0] = 0; vCoM_internal[1] = 0;
  theta_internal = 0; angvel_internal = 0; angvel_internal_prev = 0;
  Shape::resetAll();
  myFish->resetAll();
}

Fish::~Fish()
{
  if(myFish not_eq nullptr) {
    delete myFish;
    myFish = nullptr;
  }
}

void Fish::removeMoments(const std::vector<cubism::BlockInfo>& vInfo)
{
  Shape::removeMoments(vInfo);
  myFish->surfaceToComputationalFrame(orientation, centerOfMass);
  myFish->computeSkinNormals(orientation, centerOfMass);
  #if 0
  {
    std::stringstream ssF;
    ssF<<"skinPoints"<<std::setfill('0')<<std::setw(9)<<sim.step<<".dat";
    std::ofstream ofs (ssF.str().c_str(), std::ofstream::out);
    for(size_t i=0; i<myFish->upperSkin.Npoints; ++i)
      ofs<<myFish->upperSkin.xSurf[i]  <<" "<<myFish->upperSkin.ySurf[i]<<" " <<myFish->upperSkin.normXSurf[i]  <<" "<<myFish->upperSkin.normYSurf[i]  <<"\n";
    for(size_t i=myFish->lowerSkin.Npoints; i>0; --i)
      ofs<<myFish->lowerSkin.xSurf[i-1]<<" "<<myFish->lowerSkin.ySurf[i-1]<<" "<<myFish->lowerSkin.normXSurf[i-1]<<" "<<myFish->lowerSkin.normYSurf[i-1]<<"\n";
    ofs.flush();
    ofs.close();
  }
  #endif
}

void Fish::saveRestart( FILE * f ) {
  assert(f != NULL);
  Shape::saveRestart(f);
  fprintf(f, "theta_internal: %20.20e\n",  (double)theta_internal );
  fprintf(f, "angvel_internal: %20.20e\n", (double)angvel_internal );
}

void Fish::loadRestart( FILE * f ) {
  assert(f != NULL);
  Shape::loadRestart(f);
  bool ret = true;
  double in_theta_internal, in_angvel_internal;
  ret = ret && 1==fscanf(f, "theta_internal: %le\n",  &in_theta_internal );
  ret = ret && 1==fscanf(f, "angvel_internal: %le\n", &in_angvel_internal );
  theta_internal  = in_theta_internal;
  angvel_internal = in_angvel_internal;
  if( (not ret) ) {
    printf("Error reading restart file. Aborting...\n");
    fflush(0); abort();
  }
}

using namespace cubism;
class StefanFish: public Fish
{
  const bool bCorrectTrajectory;
  const bool bCorrectPosition;
 public:
  void act(const Real lTact, const std::vector<Real>& a) const;
  Real getLearnTPeriod() const;
  Real getPhase(const Real t) const;

  void resetAll() override;
  StefanFish(SimulationData&s, cubism::ArgumentParser&p, Real C[2]);
  void create(const std::vector<cubism::BlockInfo>& vInfo) override;

  // member functions for state in RL
  std::vector<Real> state( const std::vector<double>& origin ) const;
  std::vector<Real> state3D( ) const;

  // Helpers for state function
  ssize_t holdingBlockID(const std::array<Real,2> pos) const;
  std::array<Real, 2> getShear(const std::array<Real,2> pSurf) const;

  // Old Helpers (here for backward compatibility)
  ssize_t holdingBlockID(const std::array<Real,2> pos, const std::vector<cubism::BlockInfo>& velInfo) const;
  std::array<int, 2> safeIdInBlock(const std::array<Real,2> pos, const std::array<Real,2> org, const Real invh ) const;
  std::array<Real, 2> getShear(const std::array<Real,2> pSurf, const std::array<Real,2> normSurf, const std::vector<cubism::BlockInfo>& velInfo) const;

  // Helpers to restart simulation
  virtual void saveRestart( FILE * f ) override;
  virtual void loadRestart( FILE * f ) override;
};

class CurvatureFish : public FishData
{
  const Real amplitudeFactor, phaseShift, Tperiod;
 public:
  // PID controller of body curvature:
  Real curv_PID_fac = 0;
  Real curv_PID_dif = 0;
  // exponential averages:
  Real avgDeltaY = 0;
  Real avgDangle = 0;
  Real avgAngVel = 0;
  // stored past action for RL state:
  Real lastTact = 0;
  Real lastCurv = 0;
  Real oldrCurv = 0;
  // quantities needed to correctly control the speed of the midline maneuvers:
  Real periodPIDval = Tperiod;
  Real periodPIDdif = 0;
  bool TperiodPID = false;
  // quantities needed for rl:
  Real time0 = 0;
  Real timeshift = 0;
  // aux quantities for PID controllers:
  Real lastTime = 0;
  Real lastAvel = 0;

  // next scheduler is used to ramp-up the curvature from 0 during first period:
  Schedulers::ParameterSchedulerVector<6> curvatureScheduler;
  // next scheduler is used for midline-bending control points for RL:
  Schedulers::ParameterSchedulerLearnWave<7> rlBendingScheduler;

  // next scheduler is used to ramp-up the period
  Schedulers::ParameterSchedulerScalar periodScheduler;
  Real current_period    = Tperiod;
  Real next_period       = Tperiod;
  Real transition_start  = 0.0;
  Real transition_duration = 0.1*Tperiod;

 protected:
  Real * const rK;
  Real * const vK;
  Real * const rC;
  Real * const vC;
  Real * const rB;
  Real * const vB;

 public:

  CurvatureFish(Real L, Real T, Real phi, Real _h, Real _A)
  : FishData(L, _h), amplitudeFactor(_A),  phaseShift(phi),  Tperiod(T), rK(_alloc(Nm)), vK(_alloc(Nm)),
    rC(_alloc(Nm)), vC(_alloc(Nm)), rB(_alloc(Nm)), vB(_alloc(Nm)) 
    {
      _computeWidth();
      writeMidline2File(0, "initialCheck");
    }

  void resetAll() override {
    curv_PID_fac = 0;
    curv_PID_dif = 0;
    avgDeltaY = 0;
    avgDangle = 0;
    avgAngVel = 0;
    lastTact = 0;
    lastCurv = 0;
    oldrCurv = 0;
    periodPIDval = Tperiod;
    periodPIDdif = 0;
    TperiodPID = false;
    time0 = 0;
    timeshift = 0;
    lastTime = 0;
    lastAvel = 0;
    curvatureScheduler.resetAll();
    periodScheduler.resetAll();
    rlBendingScheduler.resetAll();

    FishData::resetAll();
  }

  void correctTrajectory(const Real dtheta, const Real vtheta,
                                         const Real t, const Real dt)
  {
    curv_PID_fac = dtheta;
    curv_PID_dif = vtheta;
  }

  void correctTailPeriod(const Real periodFac,const Real periodVel,
                                        const Real t, const Real dt)
  {
    assert(periodFac>0 && periodFac<2); // would be crazy

    const Real lastArg = (lastTime-time0)/periodPIDval + timeshift;
    time0 = lastTime;
    timeshift = lastArg;
    // so that new arg is only constant (prev arg) + dt / periodPIDval
    // with the new l_Tp:
    periodPIDval = Tperiod * periodFac;
    periodPIDdif = Tperiod * periodVel;
    lastTime = t;
    TperiodPID = true;
  }

  // Execute takes as arguments the current simulation time and the time
  // the RL action should have actually started. This is important for the midline
  // bending because it relies on matching the splines with the half period of
  // the sinusoidal describing the swimming motion (in order to exactly amplify
  // or dampen the undulation). Therefore, for Tp=1, t_rlAction might be K * 0.5
  // while t_current would be K * 0.5 plus a fraction of the timestep. This
  // because the new RL discrete step is detected as soon as t_current>=t_rlAction
  void execute(const Real t_current, const Real t_rlAction,
                              const std::vector<Real>&a)
  {
    assert(t_current >= t_rlAction);
    oldrCurv = lastCurv; // store action
    lastCurv = a[0]; // store action

    rlBendingScheduler.Turn(a[0], t_rlAction);

    if (a.size()>1) // also modify the swimming period
    {
      if (TperiodPID) std::cout << "Warning: PID controller should not be used with RL." << std::endl;
      lastTact = a[1]; // store action
      current_period = periodPIDval;
      next_period = Tperiod * (1 + a[1]);
      transition_start = t_rlAction;
    }
  }

  ~CurvatureFish() override {
    _dealloc(rK); _dealloc(vK); _dealloc(rC); _dealloc(vC);
    _dealloc(rB); _dealloc(vB);
  }

  void computeMidline(const Real time, const Real dt) override;
  Real _width(const Real s, const Real L) override
  {
    const Real sb=.04*length, st=.95*length, wt=.01*length, wh=.04*length;
    if(s<0 or s>L) return 0;
    const Real w = (s<sb ? std::sqrt(2*wh*s -s*s) :
           (s<st ? wh-(wh-wt)*std::pow((s-sb)/(st-sb),1) : // pow(.,2) is 3D
           (wt * (L-s)/(L-st))));
    // std::cout << "s=" << s << ", w=" << w << std::endl;
    assert( w >= 0 );
    return w;
  }
};

void StefanFish::resetAll() {
  CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  if(cFish == nullptr) { printf("Someone touched my fish\n"); abort(); }
  cFish->resetAll();
  Fish::resetAll();
}

void StefanFish::saveRestart( FILE * f ) {
  assert(f != NULL);
  Fish::saveRestart(f);
  CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  std::stringstream ss;
  ss<<std::setfill('0')<<std::setw(7)<<"_"<<obstacleID<<"_";
  std::string filename = "Schedulers"+ ss.str() + ".restart";
  {
     std::ofstream savestream;
     savestream.setf(std::ios::scientific);
     savestream.precision(std::numeric_limits<Real>::digits10 + 1);
     savestream.open(filename);
     {
       const auto & c = cFish->curvatureScheduler;
       savestream << c.t0 << "\t" << c.t1 << std::endl;
       for(int i=0;i<c.npoints;++i)
         savestream << c.parameters_t0[i]  << "\t"
                    << c.parameters_t1[i]  << "\t"
                    << c.dparameters_t0[i] << std::endl;
     }
     {
       const auto & c = cFish->periodScheduler;
       savestream << c.t0 << "\t" << c.t1 << std::endl;
       for(int i=0;i<c.npoints;++i)
         savestream << c.parameters_t0[i]  << "\t"
                    << c.parameters_t1[i]  << "\t"
                    << c.dparameters_t0[i] << std::endl;
     }
     {
      const auto & c = cFish->rlBendingScheduler;
       savestream << c.t0 << "\t" << c.t1 << std::endl;
       for(int i=0;i<c.npoints;++i)
         savestream << c.parameters_t0[i]  << "\t"
                    << c.parameters_t1[i]  << "\t"
                    << c.dparameters_t0[i] << std::endl;
     }
     savestream.close();
  }

  //Save these numbers for PID controller and other stuff. Maybe not all of them are needed
  //but we don't care, it's only a few numbers.
  fprintf(f, "curv_PID_fac: %20.20e\n", (double)cFish->curv_PID_fac);
  fprintf(f, "curv_PID_dif: %20.20e\n", (double)cFish->curv_PID_dif);
  fprintf(f, "avgDeltaY   : %20.20e\n", (double)cFish->avgDeltaY   );
  fprintf(f, "avgDangle   : %20.20e\n", (double)cFish->avgDangle   );
  fprintf(f, "avgAngVel   : %20.20e\n", (double)cFish->avgAngVel   );
  fprintf(f, "lastTact    : %20.20e\n", (double)cFish->lastTact    );
  fprintf(f, "lastCurv    : %20.20e\n", (double)cFish->lastCurv    );
  fprintf(f, "oldrCurv    : %20.20e\n", (double)cFish->oldrCurv    );
  fprintf(f, "periodPIDval: %20.20e\n", (double)cFish->periodPIDval);
  fprintf(f, "periodPIDdif: %20.20e\n", (double)cFish->periodPIDdif);
  fprintf(f, "time0       : %20.20e\n", (double)cFish->time0       );
  fprintf(f, "timeshift   : %20.20e\n", (double)cFish->timeshift   );
  fprintf(f, "lastTime    : %20.20e\n", (double)cFish->lastTime    );
  fprintf(f, "lastAvel    : %20.20e\n", (double)cFish->lastAvel    );
}

void StefanFish::loadRestart( FILE * f ) {
  assert(f != NULL);
  Fish::loadRestart(f);
  CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  std::stringstream ss;
  ss<<std::setfill('0')<<std::setw(7)<<"_"<<obstacleID<<"_";
  std::ifstream restartstream;
  std::string filename = "Schedulers"+ ss.str() + ".restart";
  restartstream.open(filename);
  {
     auto & c = cFish->curvatureScheduler;
     restartstream >> c.t0 >> c.t1;
     for(int i=0;i<c.npoints;++i)
       restartstream >> c.parameters_t0[i] >> c.parameters_t1[i] >> c.dparameters_t0[i];
  }
  {
     auto & c = cFish->periodScheduler;
     restartstream >> c.t0 >> c.t1;
     for(int i=0;i<c.npoints;++i)
       restartstream >> c.parameters_t0[i] >> c.parameters_t1[i] >> c.dparameters_t0[i];
  }
  {
     auto & c = cFish->rlBendingScheduler;
     restartstream >> c.t0 >> c.t1;
     for(int i=0;i<c.npoints;++i)
       restartstream >> c.parameters_t0[i] >> c.parameters_t1[i] >> c.dparameters_t0[i];
  }
  restartstream.close();

  bool ret = true;
  double in_curv_PID_fac, in_curv_PID_dif, in_avgDeltaY, in_avgDangle, in_avgAngVel, in_lastTact, in_lastCurv, in_oldrCurv, in_periodPIDval, in_periodPIDdif, in_time0, in_timeshift, in_lastTime, in_lastAvel; 
  ret = ret && 1==fscanf(f, "curv_PID_fac: %le\n", &in_curv_PID_fac);
  ret = ret && 1==fscanf(f, "curv_PID_dif: %le\n", &in_curv_PID_dif);
  ret = ret && 1==fscanf(f, "avgDeltaY   : %le\n", &in_avgDeltaY   );
  ret = ret && 1==fscanf(f, "avgDangle   : %le\n", &in_avgDangle   );
  ret = ret && 1==fscanf(f, "avgAngVel   : %le\n", &in_avgAngVel   );
  ret = ret && 1==fscanf(f, "lastTact    : %le\n", &in_lastTact    );
  ret = ret && 1==fscanf(f, "lastCurv    : %le\n", &in_lastCurv    );
  ret = ret && 1==fscanf(f, "oldrCurv    : %le\n", &in_oldrCurv    );
  ret = ret && 1==fscanf(f, "periodPIDval: %le\n", &in_periodPIDval);
  ret = ret && 1==fscanf(f, "periodPIDdif: %le\n", &in_periodPIDdif);
  ret = ret && 1==fscanf(f, "time0       : %le\n", &in_time0       );
  ret = ret && 1==fscanf(f, "timeshift   : %le\n", &in_timeshift   );
  ret = ret && 1==fscanf(f, "lastTime    : %le\n", &in_lastTime    );
  ret = ret && 1==fscanf(f, "lastAvel    : %le\n", &in_lastAvel    );
  cFish->curv_PID_fac = (Real) in_curv_PID_fac;
  cFish->curv_PID_dif = (Real) in_curv_PID_dif;
  cFish->avgDeltaY    = (Real) in_avgDeltaY   ;
  cFish->avgDangle    = (Real) in_avgDangle   ;
  cFish->avgAngVel    = (Real) in_avgAngVel   ;
  cFish->lastTact     = (Real) in_lastTact    ;
  cFish->lastCurv     = (Real) in_lastCurv    ;
  cFish->oldrCurv     = (Real) in_oldrCurv    ;
  cFish->periodPIDval = (Real) in_periodPIDval;
  cFish->periodPIDdif = (Real) in_periodPIDdif;
  cFish->time0        = (Real) in_time0       ;
  cFish->timeshift    = (Real) in_timeshift   ;
  cFish->lastTime     = (Real) in_lastTime    ;
  cFish->lastAvel     = (Real) in_lastAvel    ;
  if( (not ret) ) {
    printf("Error reading restart file. Aborting...\n");
    fflush(0); abort();
  }
}


StefanFish::StefanFish(SimulationData&s, ArgumentParser&p, Real C[2]):
 Fish(s,p,C), bCorrectTrajectory(p("-pid").asInt(0)),
 bCorrectPosition(p("-pidpos").asInt(0))
{
 #if 0
  // parse tau
  tau = parser("-tau").asDouble(1.0);
  // parse curvature controlpoint values
  curvature_values[0] = parser("-k1").asDouble(0.82014);
  curvature_values[1] = parser("-k2").asDouble(1.46515);
  curvature_values[2] = parser("-k3").asDouble(2.57136);
  curvature_values[3] = parser("-k4").asDouble(3.75425);
  curvature_values[4] = parser("-k5").asDouble(5.09147);
  curvature_values[5] = parser("-k6").asDouble(5.70449);
  // if nonzero && Learnfreq<0 your fish is gonna keep turning
  baseline_values[0] = parser("-b1").asDouble(0.0);
  baseline_values[1] = parser("-b2").asDouble(0.0);
  baseline_values[2] = parser("-b3").asDouble(0.0);
  baseline_values[3] = parser("-b4").asDouble(0.0);
  baseline_values[4] = parser("-b5").asDouble(0.0);
  baseline_values[5] = parser("-b6").asDouble(0.0);
  // curvature points are distributed by default but can be overridden
  curvature_points[0] = parser("-pk1").asDouble(0.00)*length;
  curvature_points[1] = parser("-pk2").asDouble(0.15)*length;
  curvature_points[2] = parser("-pk3").asDouble(0.40)*length;
  curvature_points[3] = parser("-pk4").asDouble(0.65)*length;
  curvature_points[4] = parser("-pk5").asDouble(0.90)*length;
  curvature_points[5] = parser("-pk6").asDouble(1.00)*length;
  baseline_points[0] = parser("-pb1").asDouble(curvature_points[0]/length)*length;
  baseline_points[1] = parser("-pb2").asDouble(curvature_points[1]/length)*length;
  baseline_points[2] = parser("-pb3").asDouble(curvature_points[2]/length)*length;
  baseline_points[3] = parser("-pb4").asDouble(curvature_points[3]/length)*length;
  baseline_points[4] = parser("-pb5").asDouble(curvature_points[4]/length)*length;
  baseline_points[5] = parser("-pb6").asDouble(curvature_points[5]/length)*length;
  printf("created IF2D_StefanFish: xpos=%3.3f ypos=%3.3f angle=%3.3f L=%3.3f Tp=%3.3f tau=%3.3f phi=%3.3f\n",position[0],position[1],angle,length,Tperiod,tau,phaseShift);
  printf("curvature points: pk1=%3.3f pk2=%3.3f pk3=%3.3f pk4=%3.3f pk5=%3.3f pk6=%3.3f\n",curvature_points[0],curvature_points[1],curvature_points[2],curvature_points[3],curvature_points[4],curvature_points[5]);
  printf("curvature values (normalized to L=1): k1=%3.3f k2=%3.3f k3=%3.3f k4=%3.3f k5=%3.3f k6=%3.3f\n",curvature_values[0],curvature_values[1],curvature_values[2],curvature_values[3],curvature_values[4],curvature_values[5]);
  printf("baseline points: pb1=%3.3f pb2=%3.3f pb3=%3.3f pb4=%3.3f pb5=%3.3f pb6=%3.3f\n",baseline_points[0],baseline_points[1],baseline_points[2],baseline_points[3],baseline_points[4],baseline_points[5]);
  printf("baseline values (normalized to L=1): b1=%3.3f b2=%3.3f b3=%3.3f b4=%3.3f b5=%3.3f b6=%3.3f\n",baseline_values[0],baseline_values[1],baseline_values[2],baseline_values[3],baseline_values[4],baseline_values[5]);
  // make curvature dimensional for this length
  for(int i=0; i<6; ++i) curvature_values[i]/=length;
 #endif

  const Real ampFac = p("-amplitudeFactor").asDouble(1.0);
  myFish = new CurvatureFish(length, Tperiod, phaseShift, sim.minH, ampFac);
  if( sim.rank == 0 && s.verbose ) printf("[CUP2D] - CurvatureFish %d %f %f %f %f %f %f\n",myFish->Nm, (double)length,(double)myFish->dSref,(double)myFish->dSmid,(double)sim.minH, (double)Tperiod, (double)phaseShift);
}

//static inline Real sgn(const Real val) { return (0 < val) - (val < 0); }
void StefanFish::create(const std::vector<BlockInfo>& vInfo)
{
  // If PID controller to keep position or swim straight enabled
  if (bCorrectPosition || bCorrectTrajectory)
  {
    CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
    if(cFish == nullptr) { printf("Someone touched my fish\n"); abort(); }
    const Real DT = sim.dt/Tperiod;//, time = sim.time;
    // Control pos diffs
    const Real   xDiff = (centerOfMass[0] - origC[0])/length;
    const Real   yDiff = (centerOfMass[1] - origC[1])/length;
    const Real angDiff =  orientation     - origAng;
    const Real relU = (u + sim.uinfx) / length;
    const Real relV = (v + sim.uinfy) / length;
    const Real angVel = omega, lastAngVel = cFish->lastAvel;
    // compute ang vel at t - 1/2 dt such that we have a better derivative:
    const Real aVelMidP = (angVel + lastAngVel)*Tperiod/2;
    const Real aVelDiff = (angVel - lastAngVel)*Tperiod/sim.dt;
    cFish->lastAvel = angVel; // store for next time

    // derivatives of following 2 exponential averages:
    const Real velDAavg = (angDiff-cFish->avgDangle)/Tperiod + DT * angVel;
    const Real velDYavg = (  yDiff-cFish->avgDeltaY)/Tperiod + DT * relV;
    const Real velAVavg = 10*((aVelMidP-cFish->avgAngVel)/Tperiod +DT*aVelDiff);
    // exponential averages
    cFish->avgDangle = (1.0 -DT) * cFish->avgDangle +    DT * angDiff;
    cFish->avgDeltaY = (1.0 -DT) * cFish->avgDeltaY +    DT *   yDiff;
    // faster average:
    cFish->avgAngVel = (1-10*DT) * cFish->avgAngVel + 10*DT *aVelMidP;
    const Real avgDangle = cFish->avgDangle, avgDeltaY = cFish->avgDeltaY;

    // integral (averaged) and proportional absolute DY and their derivative
    const Real absPy = std::fabs(yDiff), absIy = std::fabs(avgDeltaY);
    const Real velAbsPy =     yDiff>0 ? relV     : -relV;
    const Real velAbsIy = avgDeltaY>0 ? velDYavg : -velDYavg;
    assert(origAng<2e-16 && "TODO: rotate pos and vel to fish POV to enable \
                             PID to work even for non-zero angles");

    if (bCorrectPosition && sim.dt>0)
    {
      //If angle is positive: positive curvature only if Dy<0 (must go up)
      //If angle is negative: negative curvature only if Dy>0 (must go down)
      const Real IangPdy = (avgDangle *     yDiff < 0)? avgDangle * absPy : 0;
      const Real PangIdy = (angDiff   * avgDeltaY < 0)? angDiff   * absIy : 0;
      const Real IangIdy = (avgDangle * avgDeltaY < 0)? avgDangle * absIy : 0;

      // derivatives multiplied by 0 when term is inactive later:
      const Real velIangPdy = velAbsPy * avgDangle + absPy * velDAavg;
      const Real velPangIdy = velAbsIy * angDiff   + absIy * angVel;
      const Real velIangIdy = velAbsIy * avgDangle + absIy * velDAavg;

      //zero also the derivatives when appropriate
      const Real coefIangPdy = avgDangle *     yDiff < 0 ? 1 : 0;
      const Real coefPangIdy = angDiff   * avgDeltaY < 0 ? 1 : 0;
      const Real coefIangIdy = avgDangle * avgDeltaY < 0 ? 1 : 0;

      const Real valIangPdy = coefIangPdy *    IangPdy;
      const Real difIangPdy = coefIangPdy * velIangPdy;
      const Real valPangIdy = coefPangIdy *    PangIdy;
      const Real difPangIdy = coefPangIdy * velPangIdy;
      const Real valIangIdy = coefIangIdy *    IangIdy;
      const Real difIangIdy = coefIangIdy * velIangIdy;
      const Real periodFac = 1.0 - xDiff;
      const Real periodVel =     - relU;
#if 0
      if(not sim.muteAll) {
        std::ofstream filePID;
        std::stringstream ssF;
        ssF<<sim.path2file<<"/PID_"<<obstacleID<<".dat";
        filePID.open(ssF.str().c_str(), std::ios::app);
        filePID<<time<<" "<<valIangPdy<<" "<<difIangPdy
                     <<" "<<valPangIdy<<" "<<difPangIdy
                     <<" "<<valIangIdy<<" "<<difIangIdy
                     <<" "<<periodFac <<" "<<periodVel <<"\n";
      }
#endif
      const Real totalTerm = valIangPdy + valPangIdy + valIangIdy;
      const Real totalDiff = difIangPdy + difPangIdy + difIangIdy;
      cFish->correctTrajectory(totalTerm, totalDiff, sim.time, sim.dt);
      cFish->correctTailPeriod(periodFac, periodVel, sim.time, sim.dt);
    }
    // if absIy<EPS then we have just one fish that the simulation box follows
    // therefore we control the average angle but not the Y disp (which is 0)
    else if (bCorrectTrajectory && sim.dt>0)
    {
      const Real avgAngVel = cFish->avgAngVel, absAngVel = std::fabs(avgAngVel);
      const Real absAvelDiff = avgAngVel>0? velAVavg : -velAVavg;
      const Real coefInst = angDiff*avgAngVel>0 ? 0.01 : 1, coefAvg = 0.1;
      const Real termInst = angDiff*absAngVel;
      const Real diffInst = angDiff*absAvelDiff + angVel*absAngVel;
      const Real totalTerm = coefInst*termInst + coefAvg*avgDangle;
      const Real totalDiff = coefInst*diffInst + coefAvg*velDAavg;

#if 0
      if(not sim.muteAll) {
        std::ofstream filePID;
        std::stringstream ssF;
        ssF<<sim.path2file<<"/PID_"<<obstacleID<<".dat";
        filePID.open(ssF.str().c_str(), std::ios::app);
        filePID<<time<<" "<<coefInst*termInst<<" "<<coefInst*diffInst
                     <<" "<<coefAvg*avgDangle<<" "<<coefAvg*velDAavg<<"\n";
      }
#endif
      cFish->correctTrajectory(totalTerm, totalDiff, sim.time, sim.dt);
    }
  }
  Fish::create(vInfo);
}

void StefanFish::act(const Real t_rlAction, const std::vector<Real>& a) const
{
  CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  cFish->execute(sim.time, t_rlAction, a);
}

Real StefanFish::getLearnTPeriod() const
{
  const CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  //return cFish->periodPIDval;
  return cFish->next_period;
}

Real StefanFish::getPhase(const Real t) const
{
  const CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  const Real T0 = cFish->time0;
  const Real Ts = cFish->timeshift;
  const Real Tp = cFish->periodPIDval;
  const Real arg  = 2*M_PI*((t-T0)/Tp +Ts) + M_PI*phaseShift;
  const Real phase = std::fmod(arg, 2*M_PI);
  return (phase<0) ? 2*M_PI + phase : phase;
}

std::vector<Real> StefanFish::state( const std::vector<double>& origin ) const
{
  const CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  std::vector<Real> S(16,0);
  S[0] = ( center[0] - origin[0] )/ length;
  S[1] = ( center[1] - origin[1] )/ length;
  S[2] = getOrientation();
  S[3] = getPhase( sim.time );
  S[4] = getU() * Tperiod / length;
  S[5] = getV() * Tperiod / length;
  S[6] = getW() * Tperiod;
  S[7] = cFish->lastTact;
  S[8] = cFish->lastCurv;
  S[9] = cFish->oldrCurv;

  //Shear stress computation at three sensors
  //******************************************
  // Get fish skin
  const auto &DU = myFish->upperSkin;
  const auto &DL = myFish->lowerSkin;

  // index for sensors on the side of head
  int iHeadSide = 0;
  for(int i=0; i<myFish->Nm-1; ++i)
    if( myFish->rS[i] <= 0.04*length && myFish->rS[i+1] > 0.04*length )
      iHeadSide = i;
  assert(iHeadSide>0);

  //sensor locations
  const std::array<Real,2> locFront = {DU.xSurf[0]       , DU.ySurf[0]       };
  const std::array<Real,2> locUpper = {DU.midX[iHeadSide], DU.midY[iHeadSide]};
  const std::array<Real,2> locLower = {DL.midX[iHeadSide], DL.midY[iHeadSide]};

  //compute shear stress force (x,y) components
  std::array<Real,2> shearFront = getShear( locFront );
  std::array<Real,2> shearUpper = getShear( locLower );
  std::array<Real,2> shearLower = getShear( locUpper );

  //normal vectors at sensor locations (these vectors already have unit length)
  // first point of the two skins is the same normal should be almost the same: take the mean
  const std::array<Real,2> norFront = {0.5*(DU.normXSurf[0] + DL.normXSurf[0]), 0.5*(DU.normYSurf[0] + DL.normYSurf[0]) };
  const std::array<Real,2> norUpper = { DU.normXSurf[iHeadSide], DU.normYSurf[iHeadSide]};
  const std::array<Real,2> norLower = { DL.normXSurf[iHeadSide], DL.normYSurf[iHeadSide]};

  //tangent vectors at sensor locations (these vectors already have unit length)
  //signs alternate so that both upper and lower tangent vectors point towards fish tail
  const std::array<Real,2> tanFront = { norFront[1],-norFront[0]};
  const std::array<Real,2> tanUpper = {-norUpper[1], norUpper[0]};
  const std::array<Real,2> tanLower = { norLower[1],-norLower[0]};

  // project three stresses to normal and tangent directions
  const double shearFront_n = shearFront[0]*norFront[0]+shearFront[1]*norFront[1];
  const double shearUpper_n = shearUpper[0]*norUpper[0]+shearUpper[1]*norUpper[1];
  const double shearLower_n = shearLower[0]*norLower[0]+shearLower[1]*norLower[1];
  const double shearFront_t = shearFront[0]*tanFront[0]+shearFront[1]*tanFront[1];
  const double shearUpper_t = shearUpper[0]*tanUpper[0]+shearUpper[1]*tanUpper[1];
  const double shearLower_t = shearLower[0]*tanLower[0]+shearLower[1]*tanLower[1];

  // put non-dimensional results into state into state
  S[10] = shearFront_n * Tperiod / length;
  S[11] = shearFront_t * Tperiod / length;
  S[12] = shearLower_n * Tperiod / length;
  S[13] = shearLower_t * Tperiod / length;
  S[14] = shearUpper_n * Tperiod / length;
  S[15] = shearUpper_t * Tperiod / length;

  return S;
}


std::vector<Real> StefanFish::state3D() const
{
  const CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  std::vector<Real> S(25);
  S[0 ] = center[0];
  S[1 ] = center[1];
  S[2 ] = 1.0;
  
  //convert angle to quaternion
  S[3 ] = cos(0.5*getOrientation());
  S[4 ] = 0.0;
  S[5 ] = 0.0;
  S[6 ] = sin(0.5*getOrientation());

  S[7 ] = getPhase( sim.time );

  S[8 ] = getU() * Tperiod / length;
  S[9 ] = getV() * Tperiod / length;
  S[10] = 0.0;

  S[11] = 0.0;
  S[12] = 0.0;
  S[13] = getW() * Tperiod;

  S[14] = cFish->lastCurv;
  S[15] = cFish->oldrCurv;

  //Shear stress computation at three sensors
  //******************************************
  // Get fish skin
  const auto &DU = myFish->upperSkin;
  const auto &DL = myFish->lowerSkin;

  // index for sensors on the side of head
  int iHeadSide = 0;
  for(int i=0; i<myFish->Nm-1; ++i)
    if( myFish->rS[i] <= 0.04*length && myFish->rS[i+1] > 0.04*length )
      iHeadSide = i;
  assert(iHeadSide>0);

  //sensor locations
  const std::array<Real,2> locFront = {DU.xSurf[0]       , DU.ySurf[0]       };
  const std::array<Real,2> locUpper = {DU.midX[iHeadSide], DU.midY[iHeadSide]};
  const std::array<Real,2> locLower = {DL.midX[iHeadSide], DL.midY[iHeadSide]};

  //compute shear stress force (x,y) components
  std::array<Real,2> shearFront = getShear( locFront );
  std::array<Real,2> shearUpper = getShear( locLower );
  std::array<Real,2> shearLower = getShear( locUpper );
  S[16] = shearFront[0]* Tperiod / length;
  S[17] = shearFront[1]* Tperiod / length;
  S[18] = 0.0;
  S[19] = shearLower[0]* Tperiod / length;
  S[20] = shearLower[1]* Tperiod / length;
  S[21] = 0.0;
  S[22] = shearUpper[0]* Tperiod / length;
  S[23] = shearUpper[1]* Tperiod / length;
  S[24] = 0.0;
  #if 0
  //normal vectors at sensor locations (these vectors already have unit length)
  // first point of the two skins is the same normal should be almost the same: take the mean
  const std::array<Real,2> norFront = {0.5*(DU.normXSurf[0] + DL.normXSurf[0]), 0.5*(DU.normYSurf[0] + DL.normYSurf[0]) };
  const std::array<Real,2> norUpper = { DU.normXSurf[iHeadSide], DU.normYSurf[iHeadSide]};
  const std::array<Real,2> norLower = { DL.normXSurf[iHeadSide], DL.normYSurf[iHeadSide]};

  //tangent vectors at sensor locations (these vectors already have unit length)
  //signs alternate so that both upper and lower tangent vectors point towards fish tail
  const std::array<Real,2> tanFront = { norFront[1],-norFront[0]};
  const std::array<Real,2> tanUpper = {-norUpper[1], norUpper[0]};
  const std::array<Real,2> tanLower = { norLower[1],-norLower[0]};

  // project three stresses to normal and tangent directions
  const double shearFront_n = shearFront[0]*norFront[0]+shearFront[1]*norFront[1];
  const double shearUpper_n = shearUpper[0]*norUpper[0]+shearUpper[1]*norUpper[1];
  const double shearLower_n = shearLower[0]*norLower[0]+shearLower[1]*norLower[1];
  const double shearFront_t = shearFront[0]*tanFront[0]+shearFront[1]*tanFront[1];
  const double shearUpper_t = shearUpper[0]*tanUpper[0]+shearUpper[1]*tanUpper[1];
  const double shearLower_t = shearLower[0]*tanLower[0]+shearLower[1]*tanLower[1];

  // put non-dimensional results into state into state
  S[10] = shearFront_n * Tperiod / length;
  S[11] = shearFront_t * Tperiod / length;
  S[12] = shearLower_n * Tperiod / length;
  S[13] = shearLower_t * Tperiod / length;
  S[14] = shearUpper_n * Tperiod / length;
  S[15] = shearUpper_t * Tperiod / length;
  #endif

  return S;
}

/* helpers to compute sensor information */

// function that finds block id of block containing pos (x,y)
ssize_t StefanFish::holdingBlockID(const std::array<Real,2> pos) const
{
  const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();
  for(size_t i=0; i<velInfo.size(); ++i)
  {
    // compute lower left and top right corners of block (+- 0.5 h because pos returns cell centers)
    std::array<Real,2> MIN = velInfo[i].pos<Real>(0                   , 0                   );
    std::array<Real,2> MAX = velInfo[i].pos<Real>(VectorBlock::sizeX-1, VectorBlock::sizeY-1);
    MIN[0] -= 0.5 * velInfo[i].h;
    MIN[1] -= 0.5 * velInfo[i].h;
    MAX[0] += 0.5 * velInfo[i].h;
    MAX[1] += 0.5 * velInfo[i].h;

    // check whether point is inside block
    if( pos[0] >= MIN[0] && pos[1] >= MIN[1] && pos[0] <= MAX[0] && pos[1] <= MAX[1] )
    {
      return i;
    }
  }
  return -1; // rank does not contain point
};

// returns shear at given surface location
std::array<Real, 2> StefanFish::getShear(const std::array<Real,2> pSurf) const
{
  const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo(); 

  Real myF[2] = {0,0};
  
  // Get blockId of block that contains point pSurf.
  ssize_t blockIdSurf = holdingBlockID(pSurf);
  char error = false;
  if( blockIdSurf >= 0 )
  {
    const auto & skinBinfo = velInfo[blockIdSurf];

    // check whether obstacle block exists
    if(obstacleBlocks[blockIdSurf] == nullptr )
    {
      printf("[CUP2D, rank %u] velInfo[%lu] contains point (%f,%f), but obstacleBlocks[%lu] is a nullptr! obstacleBlocks.size()=%lu\n", sim.rank, blockIdSurf, (double)pSurf[0], (double)pSurf[1], blockIdSurf, obstacleBlocks.size());
      const std::vector<cubism::BlockInfo>& chiInfo = sim.chi->getBlocksInfo();
      const auto& chiBlock = chiInfo[blockIdSurf];
      ScalarBlock & __restrict__ CHI = *(ScalarBlock*) chiBlock.ptrBlock;
      for( size_t i = 0; i<ScalarBlock::sizeX; i++) 
      for( size_t j = 0; j<ScalarBlock::sizeY; j++)
      {
        const auto pos = chiBlock.pos<Real>(i, j);
        printf("i,j=%ld,%ld: pos=(%f,%f) with chi=%f\n", i, j, (double)pos[0], (double)pos[1], (double)CHI(i,j).s);
      }
      fflush(0);
      error = true;
    }
    else
    {
      Real dmin = 1e10;
      ObstacleBlock * const O = obstacleBlocks[blockIdSurf];
      for(size_t k = 0; k < O->n_surfPoints; ++k)
      {
        const int ix = O->surface[k]->ix;
        const int iy = O->surface[k]->iy;
        const std::array<Real,2> p = skinBinfo.pos<Real>(ix, iy);
        const Real d = (p[0]-pSurf[0])*(p[0]-pSurf[0])+(p[1]-pSurf[1])*(p[1]-pSurf[1]);
        if (d < dmin)
        {
          dmin = d;
          myF[0] = O->fXv_s[k];
          myF[1] = O->fYv_s[k];
        }
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, myF, 2, MPI_Real, MPI_SUM, sim.chi->getWorldComm());

  // DEBUG purposes
  #if 1
    MPI_Allreduce(MPI_IN_PLACE, &blockIdSurf, 1, MPI_INT64_T, MPI_MAX, sim.chi->getWorldComm());
    if( sim.rank == 0 && blockIdSurf == -1 )
    {
      printf("ABORT: coordinate (%g,%g) could not be associated to ANY obstacle block\n", (double)pSurf[0], (double)pSurf[1]);
      fflush(0);
      abort();
    }
    MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_CHAR, MPI_LOR, sim.chi->getWorldComm());
    if( error )
    {
      sim.dumpAll("failed");
      abort();
    }
  #endif

  // return shear
  return std::array<Real, 2>{{myF[0],myF[1]}};
};

void CurvatureFish::computeMidline(const Real t, const Real dt)
{
  periodScheduler.transition(t,transition_start,transition_start+transition_duration,current_period,next_period);
  periodScheduler.gimmeValues(t,periodPIDval,periodPIDdif);
  if (transition_start < t && t < transition_start+transition_duration)//timeshift also rampedup
  {
	  timeshift = (t - time0)/periodPIDval + timeshift;
	  time0 = t;
  }

  // define interpolation points on midline
  const std::array<Real ,6> curvaturePoints = { (Real)0, (Real).15*length,
    (Real).4*length, (Real).65*length, (Real).9*length, length
  };
  // define values of curvature at interpolation points
  const std::array<Real ,6> curvatureValues = {
      (Real)0.82014/length, (Real)1.46515/length, (Real)2.57136/length,
      (Real)3.75425/length, (Real)5.09147/length, (Real)5.70449/length
  };
  // define interpolation points for RL action
  const std::array<Real,7> bendPoints = {(Real)-.5, (Real)-.25, (Real)0,
    (Real).25, (Real).5, (Real).75, (Real)1};

  // transition curvature from 0 to target values
  #if 1 // ramp-up over Tperiod
  //Set 0.01*curvatureValues as initial values (not zeros).
  //This prevents the Poisson solver from exploding in some cases, when starting from zero residuals.
  const std::array<Real,6> curvatureZeros = {
	  0.01*curvatureValues[0],
	  0.01*curvatureValues[1],
	  0.01*curvatureValues[2],
	  0.01*curvatureValues[3],
	  0.01*curvatureValues[4],
	  0.01*curvatureValues[5],
  };
  curvatureScheduler.transition(0,0,Tperiod,curvatureZeros ,curvatureValues);
  #else // no rampup for debug
  curvatureScheduler.transition(t,0,Tperiod,curvatureValues,curvatureValues);
  #endif

  // write curvature values
  curvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rC, vC);
  rlBendingScheduler.gimmeValues(t, periodPIDval, length, bendPoints, Nm, rS, rB, vB);

  // next term takes into account the derivative of periodPIDval in darg:
  const Real diffT = 1 - (t-time0)*periodPIDdif/periodPIDval;
  // time derivative of arg:
  const Real darg = 2*M_PI/periodPIDval * diffT;
  const Real arg0 = 2*M_PI*((t-time0)/periodPIDval +timeshift) +M_PI*phaseShift;

  #pragma omp parallel for schedule(static)
  for(int i=0; i<Nm; ++i) {
    const Real arg = arg0 - 2*M_PI*rS[i]/length;
    rK[i] = amplitudeFactor* rC[i]*(std::sin(arg)     + rB[i] +curv_PID_fac);
    vK[i] = amplitudeFactor*(vC[i]*(std::sin(arg)     + rB[i] +curv_PID_fac)
                            +rC[i]*(std::cos(arg)*darg+ vB[i] +curv_PID_dif));
    assert(not std::isnan(rK[i]));
    assert(not std::isinf(rK[i]));
    assert(not std::isnan(vK[i]));
    assert(not std::isinf(vK[i]));
  }

  // solve frenet to compute midline parameters
  IF2D_Frenet2D::solve(Nm, rS, rK,vK, rX,rY, vX,vY, norX,norY, vNorX,vNorY);
  #if 0
   {
    FILE * f = fopen("stefan_profile","w");
    for(int i=0;i<Nm;++i)
      fprintf(f,"%d %g %g %g %g %g %g %g %g %g\n",
        i,rS[i],rX[i],rY[i],vX[i],vY[i],
        vNorX[i],vNorY[i],width[i],height[i]);
    fclose(f);
   }
  #endif
}

/***** Old Helpers (here for backward compatibility) ******/

// function that finds block id of block containing pos (x,y)
ssize_t StefanFish::holdingBlockID(const std::array<Real,2> pos, const std::vector<cubism::BlockInfo>& velInfo) const
{
  for(size_t i=0; i<velInfo.size(); ++i)
  {
    // get gridspacing in block
    const Real h = velInfo[i].h;

    // compute lower left corner of block
    std::array<Real,2> MIN = velInfo[i].pos<Real>(0, 0);
    for(int j=0; j<2; ++j)
      MIN[j] -= 0.5 * h; // pos returns cell centers

    // compute top right corner of block
    std::array<Real,2> MAX = velInfo[i].pos<Real>(VectorBlock::sizeX-1, VectorBlock::sizeY-1);
    for(int j=0; j<2; ++j)
      MAX[j] += 0.5 * h; // pos returns cell centers

    // check whether point is inside block
    if( pos[0] >= MIN[0] && pos[1] >= MIN[1] && pos[0] <= MAX[0] && pos[1] <= MAX[1] )
    {
      // point lies inside this block
      return i;
    }
  }
  // rank does not contain point
  return -1;
};

// function that gives indice of point in block
std::array<int, 2> StefanFish::safeIdInBlock(const std::array<Real,2> pos, const std::array<Real,2> org, const Real invh ) const
{
  const int indx = (int) std::round((pos[0] - org[0])*invh);
  const int indy = (int) std::round((pos[1] - org[1])*invh);
  const int ix = std::min( std::max(0, indx), VectorBlock::sizeX-1);
  const int iy = std::min( std::max(0, indy), VectorBlock::sizeY-1);
  return std::array<int, 2>{{ix, iy}};
};

// returns shear at given surface location
std::array<Real, 2> StefanFish::getShear(const std::array<Real,2> pSurf, const std::array<Real,2> normSurf, const std::vector<cubism::BlockInfo>& velInfo) const
{
  // Buffer to broadcast velcities and gridspacing
  Real velocityH[3] = {0.0, 0.0, 0.0};

  // 1. Compute surface velocity on surface
  // get blockId of surface
  ssize_t blockIdSurf = holdingBlockID(pSurf, velInfo);

  // get surface velocity if block containing point found
  char error = false;
  if( blockIdSurf >= 0 ) {
    // get block
    const auto& skinBinfo = velInfo[blockIdSurf];

    // check whether obstacle block exists
    if(obstacleBlocks[blockIdSurf] == nullptr ){
      printf("[CUP2D, rank %u] velInfo[%lu] contains point (%f,%f), but obstacleBlocks[%lu] is a nullptr! obstacleBlocks.size()=%lu\n", sim.rank, blockIdSurf, pSurf[0], pSurf[1], blockIdSurf, obstacleBlocks.size());
      const std::vector<cubism::BlockInfo>& chiInfo = sim.chi->getBlocksInfo();
      const auto& chiBlock = chiInfo[blockIdSurf];
      ScalarBlock & __restrict__ CHI = *(ScalarBlock*) chiBlock.ptrBlock;
      for( size_t i = 0; i<ScalarBlock::sizeX; i++) 
      for( size_t j = 0; j<ScalarBlock::sizeY; j++)
      {
        const auto pos = chiBlock.pos<Real>(i, j);
        printf("i,j=%ld,%ld: pos=(%f,%f) with chi=%f\n", i, j, pos[0], pos[1], CHI(i,j).s);
      }
      fflush(0);
      error = true;
      // abort();
    }
    else{
      // get origin of block
      const std::array<Real,2> oBlockSkin = skinBinfo.pos<Real>(0, 0);

      // get gridspacing on this block
      velocityH[2] = velInfo[blockIdSurf].h;

      // get index of point in block
      const std::array<int,2> iSkin = safeIdInBlock(pSurf, oBlockSkin, 1/velocityH[2]);

      // get deformation velocity
      const Real udefX = obstacleBlocks[blockIdSurf]->udef[iSkin[1]][iSkin[0]][0];
      const Real udefY = obstacleBlocks[blockIdSurf]->udef[iSkin[1]][iSkin[0]][1];

      // compute velocity of skin point
      velocityH[0] = u - omega * (pSurf[1]-centerOfMass[1]) + udefX;
      velocityH[1] = v + omega * (pSurf[0]-centerOfMass[0]) + udefY;
    }
  }

  // DEBUG purposes
  #if 1
  MPI_Allreduce(MPI_IN_PLACE, &blockIdSurf, 1, MPI_INT64_T, MPI_MAX, sim.chi->getWorldComm());
  if( sim.rank == 0 && blockIdSurf == -1 )
  {
    printf("ABORT: coordinate (%g,%g) could not be associated to ANY obstacle block\n", (double)pSurf[0], (double)pSurf[1]);
    fflush(0);
    abort();
  }

  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_CHAR, MPI_LOR, sim.chi->getWorldComm());
  if( error )
  {
    sim.dumpAll("failed");
    abort();
  }
  #endif

  // Allreduce to Bcast surface velocity
  MPI_Allreduce(MPI_IN_PLACE, velocityH, 3, MPI_Real, MPI_SUM, sim.chi->getWorldComm());

  // Assign skin velocities and grid-spacing
  const Real uSkin = velocityH[0];
  const Real vSkin = velocityH[1];
  const Real h     = velocityH[2];
  const Real invh = 1/h;

  // Reset buffer to 0
  velocityH[0] = 0.0; velocityH[1] = 0.0; velocityH[2] = 0.0;

  // 2. Compute flow velocity away from surface
  // compute point on lifted surface
  const std::array<Real,2> pLiftedSurf = { pSurf[0] + h * normSurf[0],
                                           pSurf[1] + h * normSurf[1] };

  // get blockId of lifted surface
  const ssize_t blockIdLifted = holdingBlockID(pLiftedSurf, velInfo);

  // get surface velocity if block containing point found
  if( blockIdLifted >= 0 ) {
    // get block
    const auto& liftedBinfo = velInfo[blockIdLifted];

    // get origin of block
    const std::array<Real,2> oBlockLifted = liftedBinfo.pos<Real>(0, 0);

    // get inverse gridspacing in block
    const Real invhLifted = 1/velInfo[blockIdLifted].h;

    // get index for sensor
    const std::array<int,2> iSens = safeIdInBlock(pLiftedSurf, oBlockLifted, invhLifted);

    // get velocity field at point
    const VectorBlock& b = * (const VectorBlock*) liftedBinfo.ptrBlock;
    velocityH[0] = b(iSens[0], iSens[1]).u[0];
    velocityH[1] = b(iSens[0], iSens[1]).u[1];
  }

  // Allreduce to Bcast flow velocity
  MPI_Allreduce(MPI_IN_PLACE, velocityH, 3, MPI_Real, MPI_SUM, sim.chi->getWorldComm());

  // Assign lifted skin velocities
  const Real uLifted = velocityH[0];
  const Real vLifted = velocityH[1];

  // return shear
  return std::array<Real, 2>{{(uLifted - uSkin) * invh,
                              (vLifted - vSkin) * invh }};

};

class Simulation
{
 public:
  SimulationData sim;
  std::vector<std::shared_ptr<Operator>> pipeline;
 protected:
  cubism::ArgumentParser parser;

  void createShapes();
  void parseRuntime();

public:
  Simulation(int argc, char ** argv, MPI_Comm comm);
  ~Simulation();

  /// Find the first operator in the pipeline that matches the given type.
  /// Returns `nullptr` if nothing was found.
  template <typename Op>
  Op *findOperator() const
  {
    for (const auto &ptr : pipeline) {
      Op *out = dynamic_cast<Op *>(ptr.get());
      if (out != nullptr)
	return out;
    }
    return nullptr;
  }

  /// Insert the operator at the end of the pipeline.
  void insertOperator(std::shared_ptr<Operator> op);

  /// Insert an operator after the operator of the given name.
  /// Throws an exception if the name is not found.
  void insertOperatorAfter(std::shared_ptr<Operator> op, const std::string &name);

  void reset();
  void resetRL();
  void init();
  void startObstacles();
  void simulate();
  Real calcMaxTimestep();
  void advance(const Real dt);

  const std::vector<std::shared_ptr<Shape>>& getShapes() { return sim.shapes; }
};

BCflag cubismBCX;
BCflag cubismBCY;

static const char kHorLine[] =
    "=======================================================================\n";

static inline std::vector<std::string> split(const std::string&s,const char dlm)
{
  std::stringstream ss(s); std::string item; std::vector<std::string> tokens;
  while (std::getline(ss, item, dlm)) tokens.push_back(item);
  return tokens;
}

Simulation::Simulation(int argc, char ** argv, MPI_Comm comm) : parser(argc,argv)
{
  sim.comm = comm;
  int size;
  MPI_Comm_size(sim.comm,&size);
  MPI_Comm_rank(sim.comm,&sim.rank);
  if (sim.rank == 0)
  {
    std::cout <<"=======================================================================\n";
    std::cout <<"    CubismUP 2D (velocity-pressure 2D incompressible Navier-Stokes)    \n";
    std::cout <<"=======================================================================\n";
    parser.print_args();
    #pragma omp parallel
    {
      int numThreads = omp_get_num_threads();
      #pragma omp master
      printf("[CUP2D] Running with %d rank(s) and %d thread(s).\n", size, numThreads);
    }
  }
}

Simulation::~Simulation() = default;

void Simulation::insertOperator(std::shared_ptr<Operator> op)
{
  pipeline.push_back(std::move(op));
}
void Simulation::insertOperatorAfter(
    std::shared_ptr<Operator> op, const std::string &name)
{
  for (size_t i = 0; i < pipeline.size(); ++i) {
    if (pipeline[i]->getName() == name) {
      pipeline.insert(pipeline.begin() + i + 1, std::move(op));
      return;
    }
  }
  std::string msg;
  msg.reserve(300);
  msg += "operator '";
  msg += name;
  msg += "' not found, available: ";
  for (size_t i = 0; i < pipeline.size(); ++i) {
    if (i > 0)
      msg += ", ";
    msg += pipeline[i]->getName();
  }
  msg += " (ensure that init() is called before inserting custom operators)";
  throw std::runtime_error(std::move(msg));
}

void Simulation::init()
{
  // parse field variables
  if ( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Parsing Simulation Configuration..." << std::endl;
  parseRuntime();
  // allocate the grid
  if( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Allocating Grid..." << std::endl;
  sim.allocateGrid();
  // create shapes
  if( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Creating Shapes..." << std::endl;
  createShapes();
  // impose field initial condition
  if( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Imposing Initial Conditions..." << std::endl;
  if( sim.ic == "random" )
  {
    randomIC ic(sim);
    ic(0);
  }
  else
  {
    IC ic(sim);
    ic(0);
  }
  // create compute pipeline
  if( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Creating Computational Pipeline..." << std::endl;

  pipeline.push_back(std::make_shared<AdaptTheMesh>(sim));
  pipeline.push_back(std::make_shared<PutObjectsOnGrid>(sim));
  pipeline.push_back(std::make_shared<advDiff>(sim));
  pipeline.push_back(std::make_shared<PressureSingle>(sim));
  pipeline.push_back(std::make_shared<ComputeForces>(sim));

  if( sim.rank == 0 && sim.verbose )
  {
    std::cout << "[CUP2D] Operator ordering:\n";
    for (size_t c=0; c<pipeline.size(); c++)
      std::cout << "[CUP2D] - " << pipeline[c]->getName() << "\n";
  }

  // Put Object on Intially defined Mesh and impose obstacle velocities
  startObstacles();
}

void Simulation::parseRuntime()
{
  // restart the simulation?
  sim.bRestart = parser("-restart").asBool(false);

  /* parameters that have to be given */
  /************************************/
  parser.set_strict_mode();

  // set initial number of blocks
  sim.bpdx = parser("-bpdx").asInt();
  sim.bpdy = parser("-bpdy").asInt();

  // maximal number of refinement levels
  sim.levelMax = parser("-levelMax").asInt();

  // refinement/compression tolerance for vorticity magnitude
  sim.Rtol = parser("-Rtol").asDouble();
  sim.Ctol = parser("-Ctol").asDouble();

  parser.unset_strict_mode();
  /************************************/
  /************************************/

  // refiment according to Qcriterion instead of |omega|
  sim.Qcriterion = parser("-Qcriterion").asBool(false);

  // check for refinement every this many timesteps
  sim.AdaptSteps = parser("-AdaptSteps").asInt(20);

  // boolean to switch between refinement according to chi or grad(chi)
  sim.bAdaptChiGradient = parser("-bAdaptChiGradient").asInt(1);

  // initial level of refinement
  sim.levelStart = parser("-levelStart").asInt(-1);
  if (sim.levelStart == -1) sim.levelStart = sim.levelMax - 1;

  // simulation extent
  sim.extent = parser("-extent").asDouble(1);

  // timestep / CFL number
  sim.dt = parser("-dt").asDouble(0);
  sim.CFL = parser("-CFL").asDouble(0.2);
  sim.rampup = parser("-rampup").asInt(0);

  // simulation ending parameters
  sim.nsteps = parser("-nsteps").asInt(0);
  sim.endTime = parser("-tend").asDouble(0);

  // penalisation coefficient
  sim.lambda = parser("-lambda").asDouble(1e7);

  // constant for explicit penalisation lambda=dlm/dt
  sim.dlm = parser("-dlm").asDouble(0);

  // kinematic viscocity
  sim.nu = parser("-nu").asDouble(1e-2);

  // forcing
  sim.bForcing = parser("-bForcing").asInt(0);
  sim.forcingWavenumber = parser("-forcingWavenumber").asDouble(4);
  sim.forcingCoefficient = parser("-forcingCoefficient").asDouble(4);

  // Smagorinsky Model
  sim.smagorinskyCoeff = parser("-smagorinskyCoeff").asDouble(0);
  sim.bDumpCs = parser("-dumpCs").asInt(0);

  // Flag for initial condition
  sim.ic = parser("-ic").asString("");

  // Boundary conditions (freespace or periodic)
  std::string BC_x = parser("-BC_x").asString("freespace");
  std::string BC_y = parser("-BC_y").asString("freespace");
  cubismBCX = string2BCflag(BC_x);
  cubismBCY = string2BCflag(BC_y);

  // poisson solver parameters
  sim.poissonSolver = parser("-poissonSolver").asString("iterative");
  sim.PoissonTol = parser("-poissonTol").asDouble(1e-6);
  sim.PoissonTolRel = parser("-poissonTolRel").asDouble(0);
  sim.maxPoissonRestarts = parser("-maxPoissonRestarts").asInt(30);
  sim.maxPoissonIterations = parser("-maxPoissonIterations").asInt(1000);
  sim.bMeanConstraint = parser("-bMeanConstraint").asInt(0);

  // output parameters
  sim.profilerFreq = parser("-profilerFreq").asInt(0);
  sim.dumpFreq = parser("-fdump").asInt(0);
  sim.dumpTime = parser("-tdump").asDouble(0);
  sim.path2file = parser("-file").asString("./");
  sim.path4serialization = parser("-serialization").asString(sim.path2file);
  sim.verbose = parser("-verbose").asInt(1);
  sim.muteAll = parser("-muteAll").asInt(0);
  sim.DumpUniform = parser("-DumpUniform").asBool(false);
  if(sim.muteAll) sim.verbose = 0;
}

void Simulation::createShapes()
{
  const std::string shapeArg = parser("-shapes").asString("");
  std::stringstream descriptors( shapeArg );
  std::string lines;

  while (std::getline(descriptors, lines))
  {
    std::replace(lines.begin(), lines.end(), '_', ' ');
    const std::vector<std::string> vlines = split(lines, ',');

    for (const auto& line: vlines)
    {
      std::istringstream line_stream(line);
      std::string objectName;
      if( sim.rank == 0 && sim.verbose )
	std::cout << "[CUP2D] " << line << std::endl;
      line_stream >> objectName;
      // Comments and empty lines ignored:
      if(objectName.empty() or objectName[0]=='#') continue;
      FactoryFileLineParser ffparser(line_stream);
      Real center[2] = {
	ffparser("-xpos").asDouble(.5*sim.extents[0]),
	ffparser("-ypos").asDouble(.5*sim.extents[1])
      };
      //ffparser.print_args();
      Shape* shape = nullptr;
      if (objectName=="stefanfish")
	shape = new StefanFish(       sim, ffparser, center);
      else
	throw std::invalid_argument("unrecognized shape: " + objectName);
      sim.addShape(std::shared_ptr<Shape>{shape});
    }
  }

  if( sim.shapes.size() ==  0 && sim.rank == 0)
    std::cout << "Did not create any obstacles." << std::endl;
}

void Simulation::reset()
{
  // reset field variables and shapes
  if( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Resetting Simulation..." << std::endl;
  sim.resetAll();
  // impose field initial condition
  if( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Imposing Initial Conditions..." << std::endl;
  IC ic(sim);
  ic(0);
  // Put Object on Intially defined Mesh and impose obstacle velocities
  startObstacles();
}

void Simulation::resetRL()
{
  // reset simulation (not shape)
  if( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Resetting Simulation..." << std::endl;
  sim.resetAll();
  // impose field initial condition
  if( sim.rank == 0 && sim.verbose )
    std::cout << "[CUP2D] Imposing Initial Conditions..." << std::endl;
  IC ic(sim);
  ic(0);
}

void Simulation::startObstacles()
{
  Checker check (sim);

  // put obstacles to grid and compress
  if( sim.rank == 0 && sim.verbose && !sim.bRestart)
    std::cout << "[CUP2D] Initial PutObjectsOnGrid and Compression of Grid\n";
  PutObjectsOnGrid * const putObjectsOnGrid = findOperator<PutObjectsOnGrid>();
  AdaptTheMesh * const adaptTheMesh = findOperator<AdaptTheMesh>();
  assert(putObjectsOnGrid != nullptr && adaptTheMesh != nullptr);
  if( not sim.bRestart )
  for( int i = 0; i<sim.levelMax; i++ )
  {
    (*putObjectsOnGrid)(0.0);
    (*adaptTheMesh)(0.0);
  }
  (*putObjectsOnGrid)(0.0);

  // impose velocity of obstacles
  if( not sim.bRestart )
  {
    if( sim.rank == 0 && sim.verbose )
      std::cout << "[CUP2D] Imposing Initial Velocity of Objects on field\n";
    ApplyObjVel initVel(sim);
    initVel(0);
  }
}

void Simulation::simulate() {
  if (sim.rank == 0 && !sim.muteAll)
    std::cout << kHorLine << "[CUP2D] Starting Simulation...\n" << std::flush;

  while (1)
	{
    Real dt = calcMaxTimestep();

    bool done = false;

    // Ignore the final time step if `dt` is way too small.
    if (!done || dt > 2e-16)
      advance(dt);

    if (!done)
      done = sim.bOver();

    if (sim.rank == 0 && sim.profilerFreq > 0 && sim.step % sim.profilerFreq == 0)
      sim.printResetProfiler();

    if (done)
    {
      const bool bDump = sim.bDump();
      if( bDump ) {
	if( sim.rank == 0 && sim.verbose )
	  std::cout << "[CUP2D] dumping field...\n";
	sim.registerDump();
	sim.dumpAll("_");
      }
      if (sim.rank == 0 && !sim.muteAll)
      {
	std::cout << kHorLine << "[CUP2D] Simulation Over... Profiling information:\n";
	sim.printResetProfiler();
	std::cout << kHorLine;
      }
      break;
    }
  }
}

Real Simulation::calcMaxTimestep()
{
  sim.dt_old2 = sim.dt_old;
  sim.dt_old = sim.dt;
  Real CFL = sim.CFL;
  const Real h = sim.getH();
  const auto findMaxU_op = findMaxU(sim);
  sim.uMax_measured = findMaxU_op.run();

  if( CFL > 0 )
  {
    const Real dtDiffusion = 0.25*h*h/(sim.nu+0.25*h*sim.uMax_measured);
    const Real dtAdvection = h / ( sim.uMax_measured + 1e-8 );

    //non-constant timestep introduces a source term = (1-dt_new/dt_old) \nabla^2 P_{old}
    //in the Poisson equation. Thus, we try to modify the timestep less often
    if (sim.step < sim.rampup)
    {
      const Real x = (sim.step + 1.0)/sim.rampup;
      const Real rampupFactor = std::exp(std::log(1e-3)*(1-x));
      sim.dt = rampupFactor*std::min({ dtDiffusion, CFL * dtAdvection});
    }
    else
    {
      sim.dt = std::min({ dtDiffusion, CFL * dtAdvection});
    }
  }

  if( sim.dt <= 0 ){
    std::cout << "[CUP2D] dt <= 0. Aborting..." << std::endl;
    fflush(0);
    abort();
  }

  if(sim.dlm > 0) sim.lambda = sim.dlm / sim.dt;
  return sim.dt;
}

void Simulation::advance(const Real dt)
{

  const Real CFL = ( sim.uMax_measured + 1e-8 ) * sim.dt / sim.getH();
  if (sim.rank == 0 && !sim.muteAll)
  {
    std::cout << kHorLine;
    printf("[CUP2D] step:%d, blocks:%zu, time:%f, dt=%f, uinf:[%f %f], maxU:%f, CFL:%f\n",
	   sim.step, sim.chi->getBlocksInfo().size(),
	   (double)sim.time, (double)dt,
	   (double)sim.uinfx, (double)sim.uinfy, (double)sim.uMax_measured, (double)CFL);
  }

  // dump field
  const bool bDump = sim.bDump();
  if( bDump ) {
    if( sim.rank == 0 && sim.verbose )
      std::cout << "[CUP2D] dumping field...\n";
    sim.registerDump();
    sim.dumpAll("_");
  }

  for (size_t c=0; c<pipeline.size(); c++) {
    if( sim.rank == 0 && sim.verbose )
      std::cout << "[CUP2D] running " << pipeline[c]->getName() << "...\n";
    (*pipeline[c])(dt);
  }
  sim.time += dt;
  sim.step++;
}

int main(int argc, char **argv)
{
  int threadSafety;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadSafety);

  double time = -MPI_Wtime();

  Simulation* sim = new Simulation(argc, argv, MPI_COMM_WORLD);
  sim->init();
  sim->simulate();
  time += MPI_Wtime();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0)
    std::cout << "Runtime = " << time << std::endl;
  delete sim;
  MPI_Finalize();
  return 0;
}
