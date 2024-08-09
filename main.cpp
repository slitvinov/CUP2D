#include <algorithm>
#include <fstream>
#include <iterator>
#include <memory>
#include <mpi.h>
#include <sstream>
#include <string>
#include <math.h>

#include "BufferedLogger.h"
#include "DefinitionsCup.h"
#include "FluxCorrection.h"
#include "HDF5Dumper.h"
#include "Helpers.h"
#include "LocalSpMatDnVec.h"
#include "ObstacleBlock.h"
#include "Operator.h"
#include "Shape.h"
#include "SimulationData.h"
#include "DefinitionsCup.h"

struct IF2D_Frenet2D
{
  static void solve( const unsigned Nm, const Real*const rS,
    const Real*const curv, const Real*const curv_dt,
    Real*const rX, Real*const rY, Real*const vX, Real*const vY,
    Real*const norX, Real*const norY, Real*const vNorX, Real*const vNorY )
  {
    // initial conditions
    rX[0] = 0.0;
    rY[0] = 0.0;
    norX[0] = 0.0;
    norY[0] = 1.0;
    Real ksiX = 1.0;
    Real ksiY = 0.0;
    // velocity variables
    vX[0] = 0.0;
    vY[0] = 0.0;
    vNorX[0] = 0.0;
    vNorY[0] = 0.0;
    Real vKsiX = 0.0;
    Real vKsiY = 0.0;

    for(unsigned i=1; i<Nm; i++) {
      // compute derivatives positions
      const Real dksiX = curv[i-1]*norX[i-1];
      const Real dksiY = curv[i-1]*norY[i-1];
      const Real dnuX = -curv[i-1]*ksiX;
      const Real dnuY = -curv[i-1]*ksiY;
      // compute derivatives velocity
      const Real dvKsiX = curv_dt[i-1]*norX[i-1] + curv[i-1]*vNorX[i-1];
      const Real dvKsiY = curv_dt[i-1]*norY[i-1] + curv[i-1]*vNorY[i-1];
      const Real dvNuX = -curv_dt[i-1]*ksiX - curv[i-1]*vKsiX;
      const Real dvNuY = -curv_dt[i-1]*ksiY - curv[i-1]*vKsiY;
      // compute current ds
      const Real ds = rS[i] - rS[i-1];
      // update
      rX[i] = rX[i-1] + ds*ksiX;
      rY[i] = rY[i-1] + ds*ksiY;
      norX[i] = norX[i-1] + ds*dnuX;
      norY[i] = norY[i-1] + ds*dnuY;
      ksiX += ds * dksiX;
      ksiY += ds * dksiY;
      // update velocities
      vX[i] = vX[i-1] + ds*vKsiX;
      vY[i] = vY[i-1] + ds*vKsiY;
      vNorX[i] = vNorX[i-1] + ds*dvNuX;
      vNorY[i] = vNorY[i-1] + ds*dvNuY;
      vKsiX += ds * dvKsiX;
      vKsiY += ds * dvKsiY;
      // normalize unit vectors
      const Real d1 = ksiX*ksiX + ksiY*ksiY;
      const Real d2 = norX[i]*norX[i] + norY[i]*norY[i];
      if(d1>std::numeric_limits<Real>::epsilon()) {
        const Real normfac = 1/std::sqrt(d1);
        ksiX*=normfac;
        ksiY*=normfac;
      }
      if(d2>std::numeric_limits<Real>::epsilon()) {
        const Real normfac = 1/std::sqrt(d2);
        norX[i]*=normfac;
        norY[i]*=normfac;
      }
    }
  }
};

class IF2D_Interpolation1D
{
 public:

  static void naturalCubicSpline(const Real*x, const Real*y,
    const unsigned n, const Real*xx, Real*yy, const unsigned nn) {
      return naturalCubicSpline(x,y,n,xx,yy,nn,0);
  }

  static void naturalCubicSpline(const Real*x, const Real*y, const unsigned n,
    const Real*xx, Real*yy, const unsigned nn, const Real offset)
  {
    std::vector<Real> y2(n), u(n-1);

    y2[0] = 0;
    u[0] = 0;
    for(unsigned i=1; i<n-1; i++) {
      const Real sig = (x[i]-x[i-1])/(x[i+1]-x[i-1]);
      const Real p = sig*y2[i-1] +2;
      y2[i] = (sig-1)/p;
      u[i] = (y[i+1]-y[i])/(x[i+1]-x[i])-(y[i]-y[i-1])/(x[i]-x[i-1]);
      u[i] = (6*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p;
    }

    const Real qn = 0;
    const Real un = 0;
    y2[n-1] = (un-qn*u[n-2])/(qn*y2[n-2] +1);

    for(unsigned k=n-2; k>0; k--) y2[k] = y2[k]*y2[k+1] +u[k];

    //#pragma omp parallel for schedule(static)
    for(unsigned j=0; j<nn; j++) {
      unsigned int klo = 0;
      unsigned int khi = n-1;
      unsigned int k = 0;
      while(khi-klo>1) {
        k=(khi+klo)>>1;
        if( x[k]>(xx[j]+offset)) khi=k;
        else                     klo=k;
      }

      const Real h = x[khi] - x[klo];
      if(h<=0.0) {
        std::cout<<"Interpolation points must be distinct!"<<std::endl; abort();
      }
      const Real a = (x[khi]-(xx[j]+offset))/h;
      const Real b = ((xx[j]+offset)-x[klo])/h;
      yy[j] = a*y[klo]+b*y[khi]+((a*a*a-a)*y2[klo]+(b*b*b-b)*y2[khi])*(h*h)/6;
    }
  }

  static void cubicInterpolation(const Real x0, const Real x1, const Real x,
    const Real y0,const Real y1,const Real dy0,const Real dy1, Real&y, Real&dy)
  {
    const Real xrel = (x-x0);
    const Real deltax = (x1-x0);

    const Real a = (dy0+dy1)/(deltax*deltax) - 2*(y1-y0)/(deltax*deltax*deltax);
    const Real b = (-2*dy0-dy1)/deltax + 3*(y1-y0)/(deltax*deltax);
    const Real c = dy0;
    const Real d = y0;

    y = a*xrel*xrel*xrel + b*xrel*xrel + c*xrel + d;
    dy = 3*a*xrel*xrel + 2*b*xrel + c;
  }

  static void cubicInterpolation(const Real x0, const Real x1, const Real x,
    const Real y0, const Real y1, Real & y, Real & dy) {
    return cubicInterpolation(x0,x1,x,y0,y1,0,0,y,dy); // 0 slope at end points
  }

  static void linearInterpolation(const Real x0, const Real x1, const Real x,
    const Real y0,const Real y1, Real&y, Real&dy)
  {
    y = (y1 - y0) / (x1 - x0) * (x - x0) + y0;
    dy = (y1 - y0) / (x1 - x0);
  }
};

namespace Schedulers
{
template<int Npoints>
struct ParameterScheduler
{
  static constexpr int npoints = Npoints;
  std::array<Real, Npoints>  parameters_t0; // parameters at t0
  std::array<Real, Npoints>  parameters_t1; // parameters at t1
  std::array<Real, Npoints> dparameters_t0; // derivative at t0
  Real t0, t1; // t0 and t1

  void save(std::string filename) {
    std::ofstream savestream;
    savestream.setf(std::ios::scientific);
    savestream.precision(std::numeric_limits<Real>::digits10 + 1);
    savestream.open(filename);

    savestream << t0 << "\t" << t1 << std::endl;
    for(int i=0;i<Npoints;++i)
      savestream << parameters_t0[i]  << "\t"
                 << parameters_t1[i]  << "\t"
                 << dparameters_t0[i] << std::endl;
    savestream.close();
  }

  void restart(std::string filename)
  {
    std::ifstream restartstream;
    restartstream.open(filename);
    restartstream >> t0 >> t1;
    for(int i=0;i<Npoints;++i)
    restartstream >> parameters_t0[i] >> parameters_t1[i] >> dparameters_t0[i];
    restartstream.close();
  }
  virtual void resetAll()
  {
    parameters_t0 = std::array<Real, Npoints>();
    parameters_t1 = std::array<Real, Npoints>();
    dparameters_t0 = std::array<Real, Npoints>();
    t0 = -1;
    t1 =  0;
  }

  ParameterScheduler()
  {
    t0=-1; t1=0;
    parameters_t0 = std::array<Real, Npoints>();
    parameters_t1 = std::array<Real, Npoints>();
    dparameters_t0 = std::array<Real, Npoints>();
  }
  virtual ~ParameterScheduler() {}

  void transition(const Real t, const Real tstart, const Real tend,
      const std::array<Real, Npoints> parameters_tend,
      const bool UseCurrentDerivative = false)
  {
    if(t<tstart or t>tend) return; // this transition is out of scope
    //if(tstart<t0) return; // this transition is not relevant: we are doing a next one already

    // we transition from whatever state we are in to a new state
    // the start point is where we are now: lets find out
    std::array<Real, Npoints> parameters;
    std::array<Real, Npoints> dparameters;
    gimmeValues(tstart,parameters,dparameters);

    // fill my members
    t0 = tstart;
    t1 = tend;
    parameters_t0 = parameters;
    parameters_t1 = parameters_tend;
    dparameters_t0 = UseCurrentDerivative ? dparameters : std::array<Real, Npoints>();
  }

  void transition(const Real t, const Real tstart, const Real tend,
      const std::array<Real, Npoints> parameters_tstart,
      const std::array<Real, Npoints> parameters_tend)
  {
    if(t<tstart or t>tend) return; // this transition is out of scope
    if(tstart<t0) return; // this transition is not relevant: we are doing a next one already

    // fill my members
    t0 = tstart;
    t1 = tend;
    parameters_t0 = parameters_tstart;
    parameters_t1 = parameters_tend;
  }

  void gimmeValues(const Real t, std::array<Real, Npoints>& parameters, std::array<Real, Npoints>& dparameters)
  {
    // look at the different cases
    if(t<t0 or t0<0) { // no transition, we are in state 0
      parameters = parameters_t0;
      dparameters = std::array<Real, Npoints>();
    } else if(t>t1) { // no transition, we are in state 1
      parameters = parameters_t1;
      dparameters = std::array<Real, Npoints>();
    } else { // we are within transition: interpolate
      for(int i=0;i<Npoints;++i)
        IF2D_Interpolation1D::cubicInterpolation(t0,t1,t,parameters_t0[i],parameters_t1[i],dparameters_t0[i],0.0,parameters[i],dparameters[i]);
    }
  }

  void gimmeValuesLinear(const Real t, std::array<Real, Npoints>& parameters, std::array<Real, Npoints>& dparameters)
  {
    // look at the different cases
    if(t<t0 or t0<0) { // no transition, we are in state 0
      parameters = parameters_t0;
      dparameters = std::array<Real, Npoints>();
    } else if(t>t1) { // no transition, we are in state 1
      parameters = parameters_t1;
      dparameters = std::array<Real, Npoints>();
    } else { // we are within transition: interpolate
      for(int i=0;i<Npoints;++i)
        IF2D_Interpolation1D::linearInterpolation(t0,t1,t,parameters_t0[i],parameters_t1[i],parameters[i],dparameters[i]);
    }
  }

  void gimmeValues(const Real t, std::array<Real, Npoints>& parameters)
  {
    std::array<Real, Npoints> dparameters_whocares; // no derivative info
    return gimmeValues(t,parameters,dparameters_whocares);
  }
};

struct ParameterSchedulerScalar : ParameterScheduler<1>
{
  void transition(const Real t, const Real tstart, const Real tend,
    const Real parameter_tend, const bool keepSlope = false) {
    const std::array<Real, 1> myParameter = {parameter_tend};
    return
      ParameterScheduler<1>::transition(t,tstart,tend,myParameter,keepSlope);
  }

  void transition(const Real t, const Real tstart, const Real tend,
                  const Real parameter_tstart, const Real parameter_tend)
  {
    const std::array<Real, 1> myParameterStart = {parameter_tstart};
    const std::array<Real, 1> myParameterEnd = {parameter_tend};
    return ParameterScheduler<1>::transition(t,tstart,tend,myParameterStart,myParameterEnd);
  }

  void gimmeValues(const Real t, Real & parameter, Real & dparameter)
  {
    std::array<Real, 1> myParameter, mydParameter;
    ParameterScheduler<1>::gimmeValues(t, myParameter, mydParameter);
    parameter = myParameter[0];
    dparameter = mydParameter[0];
  }

  void gimmeValues(const Real t, Real & parameter)
  {
    std::array<Real, 1> myParameter;
    ParameterScheduler<1>::gimmeValues(t, myParameter);
    parameter = myParameter[0];
  }
};

template<int Npoints>
struct ParameterSchedulerVector : ParameterScheduler<Npoints>
{
  void gimmeValues(const Real t, const std::array<Real, Npoints>& positions,
    const int Nfine, const Real*const positions_fine,
    Real*const parameters_fine, Real * const dparameters_fine) {
    // we interpolate in space the start and end point
    Real* parameters_t0_fine  = new Real[Nfine];
    Real* parameters_t1_fine  = new Real[Nfine];
    Real* dparameters_t0_fine = new Real[Nfine];

    IF2D_Interpolation1D::naturalCubicSpline(positions.data(),
      this->parameters_t0.data(), Npoints, positions_fine, parameters_t0_fine,
      Nfine);
    IF2D_Interpolation1D::naturalCubicSpline(positions.data(),
      this->parameters_t1.data(), Npoints, positions_fine, parameters_t1_fine,
      Nfine);
    IF2D_Interpolation1D::naturalCubicSpline(positions.data(),
      this->dparameters_t0.data(),Npoints, positions_fine, dparameters_t0_fine,
      Nfine);

    // look at the different cases
    if(t<this->t0 or this->t0<0) { // no transition, we are in state 0
      memcpy (parameters_fine, parameters_t0_fine, Nfine*sizeof(Real) );
      memset (dparameters_fine, 0, Nfine*sizeof(Real) );
    } else if(t>this->t1) { // no transition, we are in state 1
      memcpy (parameters_fine, parameters_t1_fine, Nfine*sizeof(Real) );
      memset (dparameters_fine, 0, Nfine*sizeof(Real) );
    } else {
      // we are within transition: interpolate in time for each point of the fine discretization
      //#pragma omp parallel for schedule(static)
      for(int i=0;i<Nfine;++i)
        IF2D_Interpolation1D::cubicInterpolation(this->t0, this->t1, t,
          parameters_t0_fine[i], parameters_t1_fine[i], dparameters_t0_fine[i],
          0, parameters_fine[i], dparameters_fine[i]);
    }
    delete [] parameters_t0_fine;
    delete [] parameters_t1_fine;
    delete [] dparameters_t0_fine;
  }

  void gimmeValues(const Real t, std::array<Real, Npoints>& parameters) {
    ParameterScheduler<Npoints>::gimmeValues(t, parameters);
  }

  void gimmeValues(const Real t, std::array<Real, Npoints> & parameters, std::array<Real, Npoints> & dparameters) {
    ParameterScheduler<Npoints>::gimmeValues(t, parameters, dparameters);
  }
};

template<int Npoints>
struct ParameterSchedulerLearnWave : ParameterScheduler<Npoints>
{
  template<typename T>
  void gimmeValues(const Real t, const Real Twave, const Real Length,
    const std::array<Real, Npoints> & positions, const int Nfine,
    const T*const positions_fine, T*const parameters_fine, Real*const dparameters_fine)
  {
    const Real _1oL = 1./Length;
    const Real _1oT = 1./Twave;
    // the fish goes through (as function of t and s) a wave function that describes the curvature
    //#pragma omp parallel for schedule(static)
    for(int i=0;i<Nfine;++i) {
      const Real c = positions_fine[i]*_1oL - (t - this->t0)*_1oT; //traveling wave coord
      bool bCheck = true;

      if (c < positions[0]) { // Are you before latest wave node?
        IF2D_Interpolation1D::cubicInterpolation(
          c, positions[0], c,
          this->parameters_t0[0], this->parameters_t0[0],
          parameters_fine[i], dparameters_fine[i]);
        bCheck = false;
      }
      else if (c > positions[Npoints-1]) {// Are you after oldest wave node?
          IF2D_Interpolation1D::cubicInterpolation(
          positions[Npoints-1], c, c,
          this->parameters_t0[Npoints-1], this->parameters_t0[Npoints-1],
          parameters_fine[i], dparameters_fine[i]);
        bCheck = false;
      } else {
        for (int j=1; j<Npoints; ++j) { // Check at which point of the travelling wave we are
          if (( c >= positions[j-1] ) && ( c <= positions[j] )) {
            IF2D_Interpolation1D::cubicInterpolation(
              positions[j-1], positions[j], c,
              this->parameters_t0[j-1], this->parameters_t0[j],
              parameters_fine[i], dparameters_fine[i]);
            dparameters_fine[i] = -dparameters_fine[i]*_1oT; // df/dc * dc/dt
            bCheck = false;
          }
        }
      }
      if (bCheck) { std::cout << "Ciaone2!" << std::endl; abort(); }
    }
  }

  void Turn(const Real b, const Real t_turn) // each decision adds a node at the beginning of the wave (left, right, straight) and pops last node
  {
    this->t0 = t_turn;

    for(int i=Npoints-1; i>1; --i)
        this->parameters_t0[i] = this->parameters_t0[i-2];
    this->parameters_t0[1] = b;
    this->parameters_t0[0] = 0;
  }
};

/*********************** NEURO-KINEMATIC FISH *******************************/

class Synapse
{
public:
    Real g = 0;
    Real dg = 0;
    const Real tau1 = 0.006 / 0.044;
    const Real tau2 = 0.008 / 0.044;
    Real prevTime = 0.0;
    std::vector<Real> activationTimes;
    std::vector<Real> activationAmplitudes;
public:
    void reset() {
        g = 0.0;
        dg = 0.0;
        prevTime = 0.0;
        activationTimes.clear();
        activationAmplitudes.clear();
    }
    void advance(const Real t) {
//        printf("[Synapse][advance]\n");
        dg = 0;
        Real dt = t - prevTime;
//        printf("[Synapse][advance] activationTimes.size() %ld\n", activationTimes.size());
        for (size_t i=0;i<activationTimes.size();i++) {
            const Real deltaT = t - activationTimes.at(i);
//            printf("[Synapse][advance] deltaT %f\n", deltaT);
            const Real dBiExp = -1 / tau2 * std::exp(-deltaT / tau2) + 1 / tau1 * std::exp(-deltaT / tau1);
//            printf("[Synapse][advance] dBiExp %f\n", dBiExp);
            dg += activationAmplitudes.at(i) * dBiExp;
//            printf("[Synapse][advance] dg %f\n", dg);
        }
        g += dg * dt;
        prevTime = t;
        forget(t);
//        printf("[Synapse][advance][end]\n");
    }
    void excite(const Real t, const Real amp) {
//        printf("[Synapse][excite]\n");
        activationTimes.push_back(t);
        activationAmplitudes.push_back(amp);
//        printf("[Synapse][excite][end]\n");
    }
    void forget(const Real t)
    {
//        printf("[Synapse][forget]\n");
        if (activationTimes.size() != 0) {
//            printf("[Synapse][forget] Number of activated synapses %ld\n", activationTimes.size());
//            printf("[Synapse][forget] t: %f, activationTime0: %f\n", t, activationTimes.at(0));
//            printf("[Synapse][forget] tau1tau2: %f\n", tau1+tau2);
            if (t - activationTimes.at(0) > tau1 + tau2) {
//                printf("Forgetting an activation. Current activation size is %ld\n", activationTimes.size());
                activationTimes.erase(activationTimes.begin());
                activationAmplitudes.erase(activationAmplitudes.begin());
            }
        }
//        printf("[Synapse][forget][end]\n");
    }
    Real value()
    {
        return g;
    }
    Real speed()
    {
        return dg;
    }
};

template<int Npoints>
class Oscillation
{
public:
    Real d = 0.0;
    Real t0 = 0.0;
    Real prev_fmod = 0.0;
    std::vector<Real> signal = std::vector<Real>(Npoints, 0.0);
    std::vector<Real> signal_out = std::vector<Real>(Npoints, 0.0);
public:
    void reset()
    {
        d = 0.0;
        t0 = 0.0;
        prev_fmod = 0.0;
        signal.clear();
        signal_out.clear();
    }
    void modify(const Real t0_in, const Real f_in, const Real d_in) {
//        printf("[Oscillation][modify]\n");
        d = d_in;
        t0 = t0_in;
        prev_fmod = 0;

        signal = std::vector<Real>(Npoints, 0.0);
        signal.at(0) = f_in;
        signal.at(static_cast<int>(std::ceil(static_cast<float>(Npoints + 1)/2.0) - 1.0)) = -f_in;
        signal_out = signal;
//        printf("[Oscillation][modify][end]\n");
    }
    void advance(const Real t)
    {
//        printf("[Oscillation][advance]\n");
        if (fmod(t - t0, d) < prev_fmod && t>t0) {
            signal.insert(signal.begin(), signal.back());
            signal.pop_back();
            signal_out = signal;
        } else if (t == t0) {
            signal_out = signal;
        } else {
            signal_out = std::vector<Real>(Npoints, 0.0);
        }
        prev_fmod = fmod(t - t0, d);
//        printf("[Oscillation][advance][end]\n");
    }
};

template<int Npoints>
struct ParameterSchedulerNeuroKinematic : ParameterScheduler<Npoints>
{
    Real prevTime = 0.0;
    int numActiveSpikes = 0;
    const Real tau1 = 0.006 / 0.044; //1 ms
    const Real tau2 = 0.008 / 0.044; //6 ms (AMPA)

    std::array<Real, Npoints> neuroSignal_t_coarse = std::array<Real, Npoints>();
    std::array<Real, Npoints> timeActivated_coarse = std::array<Real, Npoints>(); // array of time each synapse has been activated for
    std::array<Real, Npoints> muscSignal_t_coarse = std::array<Real, Npoints>();
    std::array<Real, Npoints> dMuscSignal_t_coarse = std::array<Real, Npoints>();
    std::vector<std::array<Real, Npoints>> neuroSignalVec_coarse;
    std::vector<std::array<Real, Npoints>> timeActivatedVec_coarse;
    std::vector<std::array<Real, Npoints>> muscSignalVec_coarse;
    std::vector<std::array<Real, Npoints>> dMuscSignalVec_coarse;
    std::vector<Real> amplitudeVec;


    virtual void resetAll()
    {
        prevTime = 0.0;
        numActiveSpikes = 0;
        neuroSignal_t_coarse = std::array<Real, Npoints>();
        timeActivated_coarse = std::array<Real, Npoints>();
        muscSignal_t_coarse = std::array<Real, Npoints>();
        dMuscSignal_t_coarse = std::array<Real, Npoints>();
        neuroSignalVec_coarse.clear();
        timeActivatedVec_coarse.clear();
        muscSignalVec_coarse.clear();
        dMuscSignalVec_coarse.clear();
        amplitudeVec.clear();
    }

    template<typename T>
    void gimmeValues(const Real t, const Real Length,
                     const std::array<Real, Npoints> & positions, const int Nfine,
                     const T*const positions_fine, T*const muscSignal_t_fine, Real*const dMuscSignal_t_fine,
                     Real*const spatialDerivativeMuscSignal, Real*const spatialDerivativeDMuscSignal)
    {
        // Advance arrays
        if (numActiveSpikes > 0) {
            this->dMuscSignal_t_coarse = std::array<Real, Npoints>();

            // Delete spikes that are no longer relevant
            for (int i=0;i<numActiveSpikes;i++) {
                const Real relaxationTime = (Npoints + 1) * tau1 + tau2;
                const Real activeSpikeTime = t-timeActivatedVec_coarse.at(i).at(0);
                if (activeSpikeTime >= relaxationTime) {
                    numActiveSpikes -= 1;
                    this->neuroSignalVec_coarse.erase(neuroSignalVec_coarse.begin() + i);
                    this->timeActivatedVec_coarse.erase(timeActivatedVec_coarse.begin() + i);
                    this->muscSignalVec_coarse.erase(muscSignalVec_coarse.begin() + i);
                    this->dMuscSignalVec_coarse.erase(dMuscSignalVec_coarse.begin() + i);
                }
            }

            advanceCoarseArrays(t);

            // Set previous time for next gimmeValues call
            this->prevTime = t;

            // Construct spine with cubic spline
            IF2D_Interpolation1D::naturalCubicSpline(positions.data(),
                                                     this->muscSignal_t_coarse.data(), Npoints, positions_fine,
                                                     muscSignal_t_fine, Nfine);
            IF2D_Interpolation1D::naturalCubicSpline(positions.data(),
                                                     this->dMuscSignal_t_coarse.data(), Npoints, positions_fine,
                                                     dMuscSignal_t_fine, Nfine);
        }
    }


    void advanceCoarseArrays(const Real time_current) {
//        printf("[numActiveSpikes][%d]\n", numActiveSpikes);
        const Real delta_t = time_current - this->prevTime;
        for (int i = 0; i < numActiveSpikes; i++) {
            for (int j = 0; j < Npoints; j++) {
                const Real deltaT = time_current - this->timeActivatedVec_coarse.at(i).at(j);
                if (deltaT >= 0) {
//                    printf("[i=%d][j=%d]\n", i, j);
                    // Activate current node but don't switch off previous one.
                    if (j > 0) {
                        this->neuroSignalVec_coarse.at(i).at(j) = this->neuroSignalVec_coarse.at(i)[j - 1];
                    }
                    // Begin the muscle response at the new node.
                    const Real dBiExp = -1 / this->tau2 * std::exp(-deltaT / this->tau2) +
                                          1 / this->tau1 * std::exp(-deltaT / this->tau1);

                    this->dMuscSignalVec_coarse.at(i).at(j) = this->neuroSignalVec_coarse.at(i).at(j) * dBiExp;

                    // Increment the overall muscle signal and write the overall derivative
                    this->dMuscSignal_t_coarse.at(j) += this->dMuscSignalVec_coarse.at(i).at(j);
                    this->muscSignal_t_coarse.at(j) += delta_t * this->dMuscSignalVec_coarse.at(i).at(j);
                }
            }
        }
    }

    // Deal with residual signal from previous firing time action (you can increment the signal with itself)
    void Spike(const Real t_spike, const Real aCmd, const Real dCmd, const Real deltaTFireCmd)
    {
        this->t0 = t_spike;
        this->prevTime = t_spike;
        this->numActiveSpikes += 1;
        this->neuroSignalVec_coarse.push_back(std::array<Real, Npoints>());
        this->timeActivatedVec_coarse.push_back(std::array<Real, Npoints>());
        this->muscSignalVec_coarse.push_back(std::array<Real, Npoints>());
        this->dMuscSignalVec_coarse.push_back(std::array<Real, Npoints>());

        for(int j=0; j < Npoints; j++){
            this->timeActivatedVec_coarse.at(numActiveSpikes-1).at(j) = this->t0 + j*dCmd;
        }
        // Activate the 0th node
        this->neuroSignalVec_coarse.at(numActiveSpikes-1).at(0) = aCmd;
    }
};

template<int Npoints>
struct ParameterSchedulerNeuroKinematicObject : ParameterScheduler<Npoints>
{
    std::array<Synapse, Npoints> synapses;
    Oscillation<Npoints> oscillation;

    std::array<Real, Npoints> muscle_value = std::array<Real, Npoints>();
    std::array<Real, Npoints> muscle_speed = std::array<Real, Npoints>();

    virtual void resetAll()
    {
        for (int i=0;i<Npoints;i++){
            synapses.at(i).reset();
        }
        oscillation.reset();
    }

    template<typename T>
    void gimmeValues(const Real t, const Real Length,
                     const std::array<Real, Npoints> & positions, const int Nfine,
                     const T*const positions_fine, T*const muscle_value_fine, Real*const muscle_speed_fine)
    {
        advance(t);

        // Construct spine with cubic spline
        IF2D_Interpolation1D::naturalCubicSpline(positions.data(),
                                                 this->muscle_value.data(), Npoints, positions_fine,
                                                 muscle_value_fine, Nfine);
        IF2D_Interpolation1D::naturalCubicSpline(positions.data(),
                                                 this->muscle_speed.data(), Npoints, positions_fine,
                                                 muscle_speed_fine, Nfine);
    }


    void advance(const Real t)
    {
        oscillation.advance(t);
        for (int i=0; i<Npoints; i++) {
//            printf("[Scheduler][advance]\n");
            const Real oscAmp = oscillation.signal_out.at(i);
            printf("[Scheduler][advance] signal_i %f\n", (double)oscillation.signal.at(i));
//            printf("[Scheduler][advance] oscAmp_i %f\n", oscAmp);
            if (oscAmp != 0) {synapses.at(i).excite(t, oscAmp);}
            synapses.at(i).advance(t);
            muscle_value.at(i) = synapses.at(i).value();
            muscle_speed.at(i) = synapses.at(i).speed();

            if (i==0) {printf("[Scheduler][advance] muscle_value_0 %f\n", (double)muscle_value.at(0));}
//            if (i==0) {printf("[Scheduler][advance] synapse_0 amplitude %f\n", synapses.at(0).activationAmplitudes.at(0));}
            if (i==0) {printf("[Scheduler][advance] synapse_0 numActivations %ld\n", synapses.at(0).activationAmplitudes.size());}
            if (i==10) {printf("[Scheduler][advance] muscle_value_10 %f\n", (double)muscle_value.at(10));}
//            if (i==9) {printf("[Scheduler][advance] synapse_9 amplitude %f\n", synapses.at(9).activationAmplitudes.at(0));}
            if (i==10) {printf("[Scheduler][advance] synapse_10 numActivations %ld\n", synapses.at(10).activationAmplitudes.size());}

//            printf("[Scheduler][advance] muscle_value_i %f\n", muscle_value.at(i));
//            printf("[Scheduler][advance] muscle_speed_i %f\n", muscle_speed.at(i));
//            printf("[Scheduler][advance][end]\n");
        }
    }

    void Spike(const Real t_spike, const Real aCmd, const Real dCmd, const Real deltaTFireCmd)
    {
        oscillation.modify(t_spike, aCmd, dCmd);
//        synapses.at(0).excite(t_spike, aCmd);
//        synapses.at(static_cast<int>(std::ceil(static_cast<float>(Npoints + 1)/2.0) - 1.0)).excite(t_spike, -aCmd);
//        printf("Activated synapse 0 and synapse %d", static_cast<int>(std::ceil(static_cast<float>(Npoints + 1)/2.0) - 1.0));
    }
};
}

class Shape;

class PutObjectsOnGrid : public Operator
{
protected:
  const std::vector<cubism::BlockInfo>& velInfo   = sim.vel->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& tmpInfo   = sim.tmp->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& chiInfo   = sim.chi->getBlocksInfo();

  void putChiOnGrid(Shape * const shape) const;

 public:
  using Operator::Operator;

  void operator()(Real dt) override;
  void advanceShapes(Real dt);
  void putObjectsOnGrid();

  std::string getName() override
  {
    return "PutObjectsOnGrid";
  }
};
static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
struct ComputeSurfaceNormals
{
  ComputeSurfaceNormals(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  cubism::StencilInfo stencil {-1, -1, 0, 2, 2, 1, false, {0}};
  cubism::StencilInfo stencil2{-1, -1, 0, 2, 2, 1, false, {0}};
  void operator()(ScalarLab & labChi, ScalarLab & labSDF, const cubism::BlockInfo& infoChi, const cubism::BlockInfo& infoSDF) const
  {
    for(const auto& shape : sim.shapes)
    {
      const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
      if(OBLOCK[infoChi.blockID] == nullptr) continue; //obst not in block
      const Real h = infoChi.h;
      ObstacleBlock& o = * OBLOCK[infoChi.blockID];
      const Real i2h = 0.5/h;
      const Real fac = 0.5*h;
      for(int iy=0; iy<ScalarBlock::sizeY; iy++)
      for(int ix=0; ix<ScalarBlock::sizeX; ix++)
      {
          const Real gradHX = labChi(ix+1,iy).s-labChi(ix-1,iy).s;
          const Real gradHY = labChi(ix,iy+1).s-labChi(ix,iy-1).s;
          if (gradHX*gradHX + gradHY*gradHY < 1e-12) continue;
          const Real gradUX = i2h*(labSDF(ix+1,iy).s-labSDF(ix-1,iy).s);
          const Real gradUY = i2h*(labSDF(ix,iy+1).s-labSDF(ix,iy-1).s);
          const Real gradUSq = (gradUX * gradUX + gradUY * gradUY) + EPS;
          const Real D = fac*(gradHX*gradUX + gradHY*gradUY)/gradUSq;
          if (std::fabs(D) > EPS) o.write(ix, iy, D, gradUX, gradUY);
      }
      o.allocate_surface();
    }
  }
};

struct PutChiOnGrid
{
  PutChiOnGrid(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  const cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0}};
  const std::vector<cubism::BlockInfo>& chiInfo = sim.chi->getBlocksInfo();
  void operator()(ScalarLab & lab, const cubism::BlockInfo& info) const
  {
    for(const auto& shape : sim.shapes)
    {
      const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
      if(OBLOCK[info.blockID] == nullptr) continue; //obst not in block
      const Real h = info.h;
      const Real h2 = h*h;
      ObstacleBlock& o = * OBLOCK[info.blockID];
      CHI_MAT & __restrict__ X = o.chi;
      const CHI_MAT & __restrict__ sdf = o.dist;
      o.COM_x = 0;
      o.COM_y = 0;
      o.Mass  = 0;
      auto & __restrict__ CHI  = *(ScalarBlock*) chiInfo[info.blockID].ptrBlock;
      for(int iy=0; iy<ScalarBlock::sizeY; iy++)
      for(int ix=0; ix<ScalarBlock::sizeX; ix++)
      {
        #if 0
        X[iy][ix] = sdf[iy][ix] > 0 ? 1 : 0;
        #else //Towers mollified Heaviside
        if (sdf[iy][ix] > +h || sdf[iy][ix] < -h)
        {
          X[iy][ix] = sdf[iy][ix] > 0 ? 1 : 0;
        }
        else
        {
          const Real distPx = lab(ix+1,iy).s;
          const Real distMx = lab(ix-1,iy).s;
          const Real distPy = lab(ix,iy+1).s;
          const Real distMy = lab(ix,iy-1).s;
          const Real IplusX = std::max((Real)0.0,distPx);
          const Real IminuX = std::max((Real)0.0,distMx);
          const Real IplusY = std::max((Real)0.0,distPy);
          const Real IminuY = std::max((Real)0.0,distMy);
          const Real gradIX = IplusX-IminuX;
          const Real gradIY = IplusY-IminuY;
          const Real gradUX = distPx-distMx;
          const Real gradUY = distPy-distMy;
          const Real gradUSq = (gradUX * gradUX + gradUY * gradUY) + EPS;
          X[iy][ix] = (gradIX*gradUX + gradIY*gradUY)/ gradUSq;
        }
        #endif
        CHI(ix,iy).s = std::max(CHI(ix,iy).s,X[iy][ix]);
        if(X[iy][ix] > 0)
        {
          Real p[2];
          info.pos(p, ix, iy);
          o.COM_x += X[iy][ix] * h2 * (p[0] - shape->centerOfMass[0]);
          o.COM_y += X[iy][ix] * h2 * (p[1] - shape->centerOfMass[1]);
          o.Mass  += X[iy][ix] * h2;
        }
      }
    }
  }
};

void PutObjectsOnGrid::operator()(const Real dt)
{
  sim.startProfiler("PutObjectsGrid");

  advanceShapes(dt);
  putObjectsOnGrid();

  sim.stopProfiler();
}

void PutObjectsOnGrid::advanceShapes(const Real dt)
{
  // Update laboratory frame of reference
  int nSum[2] = {0, 0}; Real uSum[2] = {0, 0};
  for (const auto& shape : sim.shapes)
    shape->updateLabVelocity(nSum, uSum);
  if(nSum[0]>0) {sim.uinfx_old = sim.uinfx; sim.uinfx = uSum[0]/nSum[0];}
  if(nSum[1]>0) {sim.uinfy_old = sim.uinfy; sim.uinfy = uSum[1]/nSum[1];}
  // Update position of object r^{t+1}=r^t+dt*v, \theta^{t+1}=\theta^t+dt*\omega
  for (const auto& shape : sim.shapes)
  {
    shape->updatePosition(dt);

    // .. and check if shape is outside the simulation domain
    Real p[2] = {0,0};
    shape->getCentroid(p);
    const auto& extent = sim.extents;
    if (p[0]<0 || p[0]>extent[0] || p[1]<0 || p[1]>extent[1]) {
      printf("[CUP2D] ABORT: Body out of domain [0,%f]x[0,%f] CM:[%e,%e]\n",
        (double)extent[0], (double)extent[1], (double)p[0], (double)p[1]);
      fflush(0);
      abort();
    }
  }
}

void PutObjectsOnGrid::putObjectsOnGrid()
{
  const size_t Nblocks = velInfo.size();

  // 1) Clear fields related to obstacle
  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  {
    ( (ScalarBlock*)  chiInfo[i].ptrBlock )->clear();
    ( (ScalarBlock*)  tmpInfo[i].ptrBlock )->set(-1);
  }

  // 2) Compute signed dist function and udef
  for(const auto& shape : sim.shapes)
    shape->create(tmpInfo);

  // 3) Compute chi and shape center of mass
  const PutChiOnGrid K(sim);
  cubism::compute<ScalarLab>(K,sim.tmp);
  const ComputeSurfaceNormals K1(sim);
  cubism::compute<ComputeSurfaceNormals,ScalarGrid,ScalarLab,ScalarGrid,ScalarLab>(K1,*sim.chi,*sim.tmp);
  for(const auto& shape : sim.shapes)
  {
    Real com[3] = {0.0, 0.0, 0.0};
    const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
    #pragma omp parallel for reduction(+ : com[:3])
    for (size_t i=0; i<OBLOCK.size(); i++)
    {
      if(OBLOCK[i] == nullptr) continue;
      com[0] += OBLOCK[i]->Mass;
      com[1] += OBLOCK[i]->COM_x;
      com[2] += OBLOCK[i]->COM_y;
    }
    MPI_Allreduce(MPI_IN_PLACE, com, 3, MPI_Real, MPI_SUM, sim.chi->getWorldComm());
    shape->M = com[0];
    shape->centerOfMass[0] += com[1]/com[0];
    shape->centerOfMass[1] += com[2]/com[0];
  }

  // 4) remove moments from characteristic function and put on grid U_s
  for(const auto& shape : sim.shapes)
  {
    shape->removeMoments(chiInfo);
  }

  // 5) do anything else needed by some shapes
  for(const auto& shape : sim.shapes)
  {
    shape->finalize();
  }

}

class AdaptTheMesh : public Operator
{
 public:
  ScalarAMR * tmp_amr  = nullptr;
  ScalarAMR * chi_amr  = nullptr;
  ScalarAMR * pres_amr = nullptr;
  ScalarAMR * pold_amr = nullptr;
  VectorAMR * vel_amr  = nullptr;
  VectorAMR * vOld_amr = nullptr;
  VectorAMR * tmpV_amr = nullptr;
  ScalarAMR * Cs_amr   = nullptr;

  AdaptTheMesh(SimulationData& s) : Operator(s)
  {
    tmp_amr  = new ScalarAMR(*sim.tmp ,sim.Rtol,sim.Ctol);
    chi_amr  = new ScalarAMR(*sim.chi ,sim.Rtol,sim.Ctol);
    pres_amr = new ScalarAMR(*sim.pres,sim.Rtol,sim.Ctol);
    pold_amr = new ScalarAMR(*sim.pold,sim.Rtol,sim.Ctol);
    vel_amr  = new VectorAMR(*sim.vel ,sim.Rtol,sim.Ctol);
    vOld_amr = new VectorAMR(*sim.vOld,sim.Rtol,sim.Ctol);
    tmpV_amr = new VectorAMR(*sim.tmpV,sim.Rtol,sim.Ctol);
    if( sim.smagorinskyCoeff != 0 )
      Cs_amr = new ScalarAMR(*sim.Cs,sim.Rtol,sim.Ctol);
  }

  ~AdaptTheMesh()
  {
    if( tmp_amr  not_eq nullptr ) delete tmp_amr ;
    if( chi_amr  not_eq nullptr ) delete chi_amr ;
    if( pres_amr not_eq nullptr ) delete pres_amr;
    if( pold_amr not_eq nullptr ) delete pold_amr;
    if( vel_amr  not_eq nullptr ) delete vel_amr ;
    if( vOld_amr not_eq nullptr ) delete vOld_amr;
    if( tmpV_amr not_eq nullptr ) delete tmpV_amr;
    if( Cs_amr   not_eq nullptr ) delete Cs_amr  ;
  }

  void operator() (const Real dt) override;
  void adapt();

  std::string getName() override
  {
    return "AdaptTheMesh";
  }
};

struct GradChiOnTmp
{
  GradChiOnTmp(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  //const StencilInfo stencil{-2, -2, 0, 3, 3, 1, true, {0}};
  const cubism::StencilInfo stencil{-4, -4, 0, 5, 5, 1, true, {0}};
  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  void operator()(ScalarLab & lab, const cubism::BlockInfo& info) const
  {
    auto& __restrict__ TMP = *(ScalarBlock*) tmpInfo[info.blockID].ptrBlock;
    if (sim.Qcriterion)
      for(int y=0; y<VectorBlock::sizeY; ++y)
      for(int x=0; x<VectorBlock::sizeX; ++x)
        TMP(x,y).s = std::max(TMP(x,y).s,(Real)0.0);//compress if Q<0

    //Loop over block and halo cells and set TMP(0,0) to a value which will cause mesh refinement
    //if any of the cells have:
    // 1. chi > 0 (if bAdaptChiGradient=false)
    // 2. chi > 0 and chi < 0.9 (if bAdaptChiGradient=true)
    // Option 2 is equivalent to grad(chi) != 0
    //const int offset = (info.level == sim.tmp->getlevelMax()-1) ? 2 : 1;
    const int offset = (info.level == sim.tmp->getlevelMax()-1) ? 4 : 2;
    const Real threshold = sim.bAdaptChiGradient ? 0.9 : 1e4;
    for(int y=-offset; y<VectorBlock::sizeY+offset; ++y)
    for(int x=-offset; x<VectorBlock::sizeX+offset; ++x)
    {
      lab(x,y).s = std::min(lab(x,y).s,(Real)1.0);
      lab(x,y).s = std::max(lab(x,y).s,(Real)0.0);
      if (lab(x,y).s > 0.0 && lab(x,y).s < threshold)
      {
        TMP(VectorBlock::sizeX/2-1,VectorBlock::sizeY/2  ).s = 2*sim.Rtol;
        TMP(VectorBlock::sizeX/2-1,VectorBlock::sizeY/2-1).s = 2*sim.Rtol;
        TMP(VectorBlock::sizeX/2  ,VectorBlock::sizeY/2  ).s = 2*sim.Rtol;
        TMP(VectorBlock::sizeX/2  ,VectorBlock::sizeY/2-1).s = 2*sim.Rtol;
        break;
      }
    }

    #ifdef CUP2D_CYLINDER_REF
    //Hardcoded refinement close the wall, for the high Re cylinder cases.
    //Cylinder center is supposed to be at (1.0,1.0) and its radius is 0.1
    for(int y=0; y<VectorBlock::sizeY; ++y)
    for(int x=0; x<VectorBlock::sizeX; ++x)
    {
      double p[2];
      info.pos(p,x,y);
      p[0] -= 1.0;
      p[1] -= 1.0;
      const double r = p[0]*p[0]+p[1]*p[1];
      if (r>0.1*0.1 && r < 0.11*0.11)
      {
        TMP(VectorBlock::sizeX/2-1,VectorBlock::sizeY/2  ).s = 2*sim.Rtol;
        TMP(VectorBlock::sizeX/2-1,VectorBlock::sizeY/2-1).s = 2*sim.Rtol;
        TMP(VectorBlock::sizeX/2  ,VectorBlock::sizeY/2  ).s = 2*sim.Rtol;
        TMP(VectorBlock::sizeX/2  ,VectorBlock::sizeY/2-1).s = 2*sim.Rtol;
        break;
      }
    }
    #endif
  }
};


void AdaptTheMesh::operator()(const Real dt)
{  
  if (sim.step > 10 && sim.step % sim.AdaptSteps != 0) return;
  adapt();
}

void AdaptTheMesh::adapt()
{
  sim.startProfiler("AdaptTheMesh");

  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();

  // compute vorticity (and use it as refinement criterion) and store it to tmp. 
  if (sim.Qcriterion)
  {
     auto K1 = computeQ(sim);
     K1(0);
  }
  else{
     auto K1 = computeVorticity(sim);
     K1(0);
  }

  // compute grad(chi) and if it's >0 set tmp = infinity
  GradChiOnTmp K2(sim);
  cubism::compute<ScalarLab>(K2,sim.chi);

  tmp_amr ->Tag();
  chi_amr ->TagLike(tmpInfo);
  pres_amr->TagLike(tmpInfo);
  pold_amr->TagLike(tmpInfo);
  vel_amr ->TagLike(tmpInfo);
  vOld_amr->TagLike(tmpInfo);
  tmpV_amr->TagLike(tmpInfo);
  if( sim.smagorinskyCoeff != 0 )
    Cs_amr->TagLike(tmpInfo);

  tmp_amr ->Adapt(sim.time, sim.rank == 0 && !sim.muteAll, false);
  chi_amr ->Adapt(sim.time, false, false);
  vel_amr ->Adapt(sim.time, false, false);
  vOld_amr->Adapt(sim.time, false, false);
  pres_amr->Adapt(sim.time, false, false);
  pold_amr->Adapt(sim.time, false, false);
  tmpV_amr->Adapt(sim.time, false, true);
  if( sim.smagorinskyCoeff != 0 )
    Cs_amr->Adapt(sim.time, false, true);

  sim.stopProfiler();
}

class advDiff : public Operator
{
protected:
  const std::vector<cubism::BlockInfo>& velInfo   = sim.vel->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& tmpVInfo  = sim.tmpV->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& vOldInfo  = sim.vOld->getBlocksInfo();

 public:
  advDiff(SimulationData& s) : Operator(s) { }

  void operator() (const Real dt) override;

  std::string getName() override
  {
    return "advDiff";
  }
};


#ifdef CUP2D_PRESERVE_SYMMETRY
#define CUP2D_DISABLE_OPTIMIZATIONS __attribute__((optimize("-O1")))
#else
#define CUP2D_DISABLE_OPTIMIZATIONS
#endif

CUP2D_DISABLE_OPTIMIZATIONS
static inline Real weno5_plus(const Real um2, const Real um1, const Real u, const Real up1, const Real up2)
{
  const Real exponent = 2;
  const Real e = 1e-6;
  const Real b1 = 13.0/12.0*pow((um2+u)-2*um1,2)+0.25*pow((um2+3*u)-4*um1,2);
  const Real b2 = 13.0/12.0*pow((um1+up1)-2*u,2)+0.25*pow(um1-up1,2);
  const Real b3 = 13.0/12.0*pow((u+up2)-2*up1,2)+0.25*pow((3*u+up2)-4*up1,2);
  const Real g1 = 0.1;
  const Real g2 = 0.6;
  const Real g3 = 0.3;
  const Real what1 = g1/pow(b1+e,exponent);
  const Real what2 = g2/pow(b2+e,exponent);
  const Real what3 = g3/pow(b3+e,exponent);
  const Real aux = 1.0/((what1+what3)+what2);
  const Real w1 = what1*aux;
  const Real w2 = what2*aux;
  const Real w3 = what3*aux;
  const Real f1 = (11.0/6.0)*u + ( ( 1.0/3.0)*um2- (7.0/6.0)*um1);
  const Real f2 = (5.0 /6.0)*u + ( (-1.0/6.0)*um1+ (1.0/3.0)*up1);
  const Real f3 = (1.0 /3.0)*u + ( (+5.0/6.0)*up1- (1.0/6.0)*up2);
  return (w1*f1+w3*f3)+w2*f2;
}

CUP2D_DISABLE_OPTIMIZATIONS
static inline Real weno5_minus(const Real um2, const Real um1, const Real u, const Real up1, const Real up2)
{
  const Real exponent = 2;
  const Real e = 1e-6;
  const Real b1 = 13.0/12.0*pow((um2+u)-2*um1,2)+0.25*pow((um2+3*u)-4*um1,2);
  const Real b2 = 13.0/12.0*pow((um1+up1)-2*u,2)+0.25*pow(um1-up1,2);
  const Real b3 = 13.0/12.0*pow((u+up2)-2*up1,2)+0.25*pow((3*u+up2)-4*up1,2);
  const Real g1 = 0.3;
  const Real g2 = 0.6;
  const Real g3 = 0.1;
  const Real what1 = g1/pow(b1+e,exponent);
  const Real what2 = g2/pow(b2+e,exponent);
  const Real what3 = g3/pow(b3+e,exponent);
  const Real aux = 1.0/((what1+what3)+what2);
  const Real w1 = what1*aux;
  const Real w2 = what2*aux;
  const Real w3 = what3*aux;
  const Real f1 = ( 1.0/3.0)*u + ( (-1.0/6.0)*um2+ (5.0/6.0)*um1);
  const Real f2 = ( 5.0/6.0)*u + ( ( 1.0/3.0)*um1- (1.0/6.0)*up1);
  const Real f3 = (11.0/6.0)*u + ( (-7.0/6.0)*up1+ (1.0/3.0)*up2);
  return (w1*f1+w3*f3)+w2*f2;
}

static inline Real derivative(const Real U, const Real um3, const Real um2, const Real um1,
                                            const Real u  ,
                                            const Real up1, const Real up2, const Real up3)
{
  Real fp = 0.0;
  Real fm = 0.0;
  if (U > 0)
  {
    fp = weno5_plus (um2,um1,u,up1,up2);
    fm = weno5_plus (um3,um2,um1,u,up1);
  }
  else
  {
    fp = weno5_minus(um1,u,up1,up2,up3);
    fm = weno5_minus(um2,um1,u,up1,up2);
  }
  return (fp-fm);
}

static inline Real dU_adv_dif(const VectorLab&V, const Real uinf[2], const Real advF, const Real difF, const int ix, const int iy)
{
  const Real u    = V(ix,iy).u[0];
  const Real v    = V(ix,iy).u[1];
  const Real UU   = u + uinf[0];
  const Real VV   = v + uinf[1];

  const Real up1x = V(ix+1,iy).u[0];
  const Real up2x = V(ix+2,iy).u[0];
  const Real up3x = V(ix+3,iy).u[0];
  const Real um1x = V(ix-1,iy).u[0];
  const Real um2x = V(ix-2,iy).u[0];
  const Real um3x = V(ix-3,iy).u[0];

  const Real up1y = V(ix,iy+1).u[0];
  const Real up2y = V(ix,iy+2).u[0];
  const Real up3y = V(ix,iy+3).u[0];
  const Real um1y = V(ix,iy-1).u[0];
  const Real um2y = V(ix,iy-2).u[0];
  const Real um3y = V(ix,iy-3).u[0];
  
  const Real dudx = derivative(UU,um3x,um2x,um1x,u,up1x,up2x,up3x);
  const Real dudy = derivative(VV,um3y,um2y,um1y,u,up1y,up2y,up3y);

  return advF*(UU*dudx+VV*dudy) + difF*( ((up1x + um1x) + (up1y  + um1y)) - 4*u);
}
  
static inline Real dV_adv_dif(const VectorLab&V, const Real uinf[2], const Real advF, const Real difF, const int ix, const int iy)
{
  const Real u    = V(ix,iy).u[0];
  const Real v    = V(ix,iy).u[1];
  const Real UU   = u + uinf[0];
  const Real VV   = v + uinf[1];

  const Real vp1x = V(ix+1,iy).u[1];
  const Real vp2x = V(ix+2,iy).u[1];
  const Real vp3x = V(ix+3,iy).u[1];
  const Real vm1x = V(ix-1,iy).u[1];
  const Real vm2x = V(ix-2,iy).u[1];
  const Real vm3x = V(ix-3,iy).u[1];

  const Real vp1y = V(ix,iy+1).u[1];
  const Real vp2y = V(ix,iy+2).u[1];
  const Real vp3y = V(ix,iy+3).u[1];
  const Real vm1y = V(ix,iy-1).u[1];
  const Real vm2y = V(ix,iy-2).u[1];
  const Real vm3y = V(ix,iy-3).u[1];

  const Real dvdx = derivative(UU,vm3x,vm2x,vm1x,v,vp1x,vp2x,vp3x);
  const Real dvdy = derivative(VV,vm3y,vm2y,vm1y,v,vp1y,vp2y,vp3y);

  return advF*(UU*dvdx+VV*dvdy) + difF*( ((vp1x+ vm1x) + (vp1y+ vm1y)) - 4*v);
}


struct KernelAdvectDiffuse
{
  KernelAdvectDiffuse(const SimulationData & s) : sim(s)
  {
    uinf[0] = sim.uinfx;
    uinf[1] = sim.uinfy;
  }
  const SimulationData & sim;
  Real uinf [2];
  const cubism::StencilInfo stencil{-3, -3, 0, 4, 4, 1, true, {0,1}};
  const std::vector<cubism::BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();

  void operator()(VectorLab& lab, const cubism::BlockInfo& info) const
  {
    const Real h = info.h;
    const Real dfac = sim.nu*sim.dt;
    const Real afac = -sim.dt*h;
    VectorBlock & __restrict__ TMP = *(VectorBlock*) tmpVInfo[info.blockID].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      TMP(ix,iy).u[0] = dU_adv_dif(lab,uinf,afac,dfac,ix,iy);
      TMP(ix,iy).u[1] = dV_adv_dif(lab,uinf,afac,dfac,ix,iy);
    }
    cubism::BlockCase<VectorBlock> * tempCase = (cubism::BlockCase<VectorBlock> *)(tmpVInfo[info.blockID].auxiliary);
    VectorBlock::ElementType * faceXm = nullptr;
    VectorBlock::ElementType * faceXp = nullptr;
    VectorBlock::ElementType * faceYm = nullptr;
    VectorBlock::ElementType * faceYp = nullptr;

    const Real aux_coef = dfac;

    if (tempCase != nullptr)
    {
      faceXm = tempCase -> storedFace[0] ?  & tempCase -> m_pData[0][0] : nullptr;
      faceXp = tempCase -> storedFace[1] ?  & tempCase -> m_pData[1][0] : nullptr;
      faceYm = tempCase -> storedFace[2] ?  & tempCase -> m_pData[2][0] : nullptr;
      faceYp = tempCase -> storedFace[3] ?  & tempCase -> m_pData[3][0] : nullptr;
    }
    if (faceXm != nullptr)
    {
      int ix = 0;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      {
        faceXm[iy].u[0] = aux_coef*(lab(ix,iy).u[0] - lab(ix-1,iy).u[0]);
        faceXm[iy].u[1] = aux_coef*(lab(ix,iy).u[1] - lab(ix-1,iy).u[1]);
      }
    }
    if (faceXp != nullptr)
    {
      int ix = VectorBlock::sizeX-1;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      {
        faceXp[iy].u[0] = aux_coef*(lab(ix,iy).u[0] - lab(ix+1,iy).u[0]);
        faceXp[iy].u[1] = aux_coef*(lab(ix,iy).u[1] - lab(ix+1,iy).u[1]);
      }
    }
    if (faceYm != nullptr)
    {
      int iy = 0;
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        faceYm[ix].u[0] = aux_coef*(lab(ix,iy).u[0] - lab(ix,iy-1).u[0]);
        faceYm[ix].u[1] = aux_coef*(lab(ix,iy).u[1] - lab(ix,iy-1).u[1]);
      }
    }
    if (faceYp != nullptr)
    {
      int iy = VectorBlock::sizeY-1;
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        faceYp[ix].u[0] = aux_coef*(lab(ix,iy).u[0] - lab(ix,iy+1).u[0]);
        faceYp[ix].u[1] = aux_coef*(lab(ix,iy).u[1] - lab(ix,iy+1).u[1]);
      }
    }
  }
};


void advDiff::operator()(const Real dt)
{
  sim.startProfiler("advDiff");
  const size_t Nblocks = velInfo.size();
  KernelAdvectDiffuse Step1(sim) ;

  //1.Save u^{n} to dataOld
  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock & __restrict__ Vold  = *(VectorBlock*) vOldInfo[i].ptrBlock;
    const VectorBlock & __restrict__ V  = *(VectorBlock*)  velInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      Vold(ix,iy).u[0] = V(ix,iy).u[0];
      Vold(ix,iy).u[1] = V(ix,iy).u[1];
    }
  }

  /********************************************************************/
  // 2. Set u^{n+1/2} = u^{n} + 0.5*dt*RHS(u^{n})
  //   2a) Compute 0.5*dt*RHS(u^{n}) and store it to tmpU,tmpV,tmpW
  cubism::compute<VectorLab>(Step1,sim.vel,sim.tmpV);

  //   2b) Set u^{n+1/2} = u^{n} + 0.5*dt*RHS(u^{n})
  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock & __restrict__ V  = *(VectorBlock*)  velInfo[i].ptrBlock;
    const VectorBlock & __restrict__ Vold = *(VectorBlock*) vOldInfo[i].ptrBlock;
    const VectorBlock & __restrict__ tmpV = *(VectorBlock*) tmpVInfo[i].ptrBlock;
    const Real ih2 = 1.0/(velInfo[i].h*velInfo[i].h);
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      V(ix,iy).u[0] = Vold(ix,iy).u[0] + (0.5*tmpV(ix,iy).u[0])*ih2;
      V(ix,iy).u[1] = Vold(ix,iy).u[1] + (0.5*tmpV(ix,iy).u[1])*ih2;
    }
  }
  /********************************************************************/

  /********************************************************************/
  // 3. Set u^{n+1} = u^{n} + dt*RHS(u^{n+1/2})
  //   3a) Compute dt*RHS(u^{n+1/2}) and store it to tmpU,tmpV,tmpW
  cubism::compute<VectorLab>(Step1,sim.vel,sim.tmpV);
  //   3b) Set u^{n+1} = u^{n} + dt*RHS(u^{n+1/2})
  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  {
    VectorBlock & __restrict__ V  = *(VectorBlock*)  velInfo[i].ptrBlock;
    const VectorBlock & __restrict__ Vold = *(VectorBlock*) vOldInfo[i].ptrBlock;
    const VectorBlock & __restrict__ tmpV = *(VectorBlock*) tmpVInfo[i].ptrBlock;
    const Real ih2 = 1.0/(velInfo[i].h*velInfo[i].h);
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      V(ix,iy).u[0] = Vold(ix,iy).u[0] + tmpV(ix,iy).u[0]*ih2;
      V(ix,iy).u[1] = Vold(ix,iy).u[1] + tmpV(ix,iy).u[1]*ih2;
    }
  }
  /********************************************************************/

  sim.stopProfiler();
}

class Shape;

class ComputeForces : public Operator
{
  const std::vector<cubism::BlockInfo>& presInfo = sim.pres->getBlocksInfo();

public:
  void operator() (const Real dt) override;

  ComputeForces(SimulationData& s);
  ~ComputeForces() {}

  std::string getName() override
  {
    return "ComputeForces";
  }
};


using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];

struct KernelComputeForces
{
  const int big   = 5;
  const int small = -4;
  KernelComputeForces(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  cubism::StencilInfo stencil {small, small, 0, big, big, 1, true, {0,1}};
  cubism::StencilInfo stencil2{small, small, 0, big, big, 1, true, {0}};

  const int bigg = ScalarBlock::sizeX + big-1;
  const int stencil_start[3] = {small,small,small}, stencil_end[3] = {big,big,big};
  const Real c0 = -137./60.;
  const Real c1 =    5.    ;
  const Real c2 = -  5.    ;
  const Real c3 =   10./ 3.;
  const Real c4 = -  5./ 4.;
  const Real c5 =    1./ 5.;

  inline bool inrange(const int i) const
  {
    return (i >= small && i < bigg);
  }


  const std::vector<cubism::BlockInfo>& presInfo = sim.pres->getBlocksInfo();

  void operator()(VectorLab & lab, ScalarLab & chi, const cubism::BlockInfo& info, const cubism::BlockInfo& info2) const
  {
    VectorLab & V = lab;
    ScalarBlock & __restrict__ P = *(ScalarBlock*) presInfo[info.blockID].ptrBlock;

    //const int big   = ScalarBlock::sizeX + 4;
    //const int small = -4;
    for(const auto& _shape : sim.shapes)
    {
      const Shape * const shape = _shape.get();
      const std::vector<ObstacleBlock*> & OBLOCK = shape->obstacleBlocks;
      const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];
      const Real vel_norm = std::sqrt(shape->u*shape->u + shape->v*shape->v);
      const Real vel_unit[2] = {
        vel_norm>0? (Real) shape->u / vel_norm : (Real)0,
        vel_norm>0? (Real) shape->v / vel_norm : (Real)0
      };
   
      const Real NUoH = sim.nu / info.h; // 2 nu / 2 h
      ObstacleBlock * const O = OBLOCK[info.blockID];
      if (O == nullptr) continue;
      assert(O->filled);
      for(size_t k = 0; k < O->n_surfPoints; ++k)
      {
        const int ix = O->surface[k]->ix, iy = O->surface[k]->iy;
        const std::array<Real,2> p = info.pos<Real>(ix, iy);

        const Real normX = O->surface[k]->dchidx; //*h^3 (multiplied in dchidx)
        const Real normY = O->surface[k]->dchidy; //*h^3 (multiplied in dchidy)
        const Real norm = 1.0/std::sqrt(normX*normX+normY*normY);
        const Real dx = normX*norm;
        const Real dy = normY*norm;
        //shear stresses
        //"lifted" surface: derivatives make no sense when the values used are in the object, 
        // so we take one-sided stencils with values outside of the object
        //Real D11 = 0.0;
        //Real D22 = 0.0;
        //Real D12 = 0.0;
        Real DuDx;
        Real DuDy;
        Real DvDx;
        Real DvDy;
        {
          //The integers x and y will be the coordinates of the point on the lifted surface.
          //To find them, we move along the normal vector to the surface, until we find a point
          //outside of the object (where chi = 0).
          int x = ix;
          int y = iy;
          for (int kk = 0 ; kk < 5 ; kk++) //5 is arbitrary
          {
            const int dxi = round(kk*dx);
            const int dyi = round(kk*dy);
            if (ix + dxi + 1 >= ScalarBlock::sizeX + big-1 || ix + dxi -1 < small) continue;
            if (iy + dyi + 1 >= ScalarBlock::sizeY + big-1 || iy + dyi -1 < small) continue;
            x  = ix + dxi; 
            y  = iy + dyi;
            if (chi(x,y).s < 0.01 ) break;
          }


          //Now that we found the (x,y) of the point, we compute grad(u) there.
          //grad(u) is computed with biased stencils. If available, larger stencils are used.
          //Then, we compute higher order derivatives that are used to form a Taylor expansion
          //around (x,y). Finally, this expansion is used to extrapolate grad(u) to (ix,iy) of 
          //the actual solid surface. 

          const auto & l = lab;
          const int sx = normX > 0 ? +1:-1;
          const int sy = normY > 0 ? +1:-1;

          VectorElement dveldx;
          if      (inrange(x+5*sx)) dveldx = sx*(  c0*l(x,y)+ c1*l(x+sx,y)+ c2*l(x+2*sx,y)+c3*l(x+3*sx,y)+c4*l(x+4*sx,y)+c5*l(x+5*sx,y));
          else if (inrange(x+2*sx)) dveldx = sx*(-1.5*l(x,y)+2.0*l(x+sx,y)-0.5*l(x+2*sx,y));
          else                      dveldx = sx*(l(x+sx,y)-l(x,y));
          VectorElement dveldy;
          if      (inrange(y+5*sy)) dveldy = sy*(  c0*l(x,y)+ c1*l(x,y+sy)+ c2*l(x,y+2*sy)+c3*l(x,y+3*sy)+c4*l(x,y+4*sy)+c5*l(x,y+5*sy));
          else if (inrange(y+2*sy)) dveldy = sy*(-1.5*l(x,y)+2.0*l(x,y+sy)-0.5*l(x,y+2*sy));
          else                      dveldy = sx*(l(x,y+sy)-l(x,y));

          const VectorElement dveldx2 = l(x-1,y)-2.0*l(x,y)+ l(x+1,y);
          const VectorElement dveldy2 = l(x,y-1)-2.0*l(x,y)+ l(x,y+1);

          VectorElement dveldxdy;
          if (inrange(x+2*sx) && inrange(y+2*sy)) dveldxdy = sx*sy*(-0.5*( -1.5*l(x+2*sx,y     )+2*l(x+2*sx,y+  sy)  -0.5*l(x+2*sx,y+2*sy)       ) + 2*(-1.5*l(x+sx,y)+2*l(x+sx,y+sy)-0.5*l(x+sx,y+2*sy)) -1.5*(-1.5*l(x,y)+2*l(x,y+sy)-0.5*l(x,y+2*sy)));
          else                                    dveldxdy = sx*sy*(            l(x+  sx,y+  sy)-  l(x+  sx,y     )) -   (l(x     ,y  +sy)-l(x,y));

          DuDx = dveldx.u[0] + dveldx2.u[0]*(ix-x) + dveldxdy.u[0]*(iy-y);
          DvDx = dveldx.u[1] + dveldx2.u[1]*(ix-x) + dveldxdy.u[1]*(iy-y);
          DuDy = dveldy.u[0] + dveldy2.u[0]*(iy-y) + dveldxdy.u[0]*(ix-x);
          DvDy = dveldy.u[1] + dveldy2.u[1]*(iy-y) + dveldxdy.u[1]*(ix-x);
        }//shear stress computation ends here

        //normals computed with Towers 2009
        // Actually using the volume integral, since (/iint -P /hat{n} dS) =
        // (/iiint - /nabla P dV). Also, P*/nabla /Chi = /nabla P
        // penalty-accel and surf-force match up if resolution is high enough
        //const Real fXV = D11*normX + D12*normY, fXP = - P(ix,iy).s * normX;
        //const Real fYV = D12*normX + D22*normY, fYP = - P(ix,iy).s * normY;
        const Real fXV = NUoH*DuDx*normX + NUoH*DuDy*normY, fXP = - P(ix,iy).s * normX;
        const Real fYV = NUoH*DvDx*normX + NUoH*DvDy*normY, fYP = - P(ix,iy).s * normY;

        const Real fXT = fXV + fXP, fYT = fYV + fYP;

        //store:
        O->x_s    [k] = p[0];
        O->y_s    [k] = p[1];
        O->p_s    [k] = P(ix,iy).s;
        O->u_s    [k] = V(ix,iy).u[0];
        O->v_s    [k] = V(ix,iy).u[1];
        O->nx_s   [k] = dx;
        O->ny_s   [k] = dy;
        O->omega_s[k] = (DvDx - DuDy)/info.h;
        O->uDef_s [k] = O->udef[iy][ix][0];
        O->vDef_s [k] = O->udef[iy][ix][1];
        O->fX_s   [k] = -P(ix,iy).s * dx + NUoH*DuDx*dx + NUoH*DuDy*dy;//scale by 1/h
        O->fY_s   [k] = -P(ix,iy).s * dy + NUoH*DvDx*dx + NUoH*DvDy*dy;//scale by 1/h
        O->fXv_s  [k] = NUoH*DuDx*dx + NUoH*DuDy*dy;//scale by 1/h
        O->fYv_s  [k] = NUoH*DvDx*dx + NUoH*DvDy*dy;//scale by 1/h

        //perimeter:
        O->perimeter += std::sqrt(normX*normX + normY*normY);
        O->circulation += normX * O->v_s[k] - normY * O->u_s[k];
        //forces (total, visc, pressure):
        O->forcex += fXT;
        O->forcey += fYT;
        O->forcex_V += fXV;
        O->forcey_V += fYV;
        O->forcex_P += fXP;
        O->forcey_P += fYP;
        //torque:
        O->torque   += (p[0] - Cx) * fYT - (p[1] - Cy) * fXT;
        O->torque_P += (p[0] - Cx) * fYP - (p[1] - Cy) * fXP;
        O->torque_V += (p[0] - Cx) * fYV - (p[1] - Cy) * fXV;
        //thrust, drag:
        const Real forcePar = fXT * vel_unit[0] + fYT * vel_unit[1];
        O->thrust += .5*(forcePar + std::fabs(forcePar));
        O->drag   -= .5*(forcePar - std::fabs(forcePar));
        const Real forcePerp = fXT * vel_unit[1] - fYT * vel_unit[0];
        O->lift   += forcePerp;
        //power output (and negative definite variant which ensures no elastic energy absorption)
        // This is total power, for overcoming not only deformation, but also the oncoming velocity. Work done by fluid, not by the object (for that, just take -ve)
        const Real powOut = fXT * O->u_s[k]    + fYT * O->v_s[k];
        //deformation power output (and negative definite variant which ensures no elastic energy absorption)
        const Real powDef = fXT * O->uDef_s[k] + fYT * O->vDef_s[k];
        O->Pout        += powOut;
        O->defPower    += powDef;
        O->PoutBnd     += std::min((Real)0, powOut);
        O->defPowerBnd += std::min((Real)0, powDef);
      }
      O->PoutNew = O->forcex*shape->u +  O->forcey*shape->v;
    } 
  }
};

void ComputeForces::operator()(const Real dt)
{
  sim.startProfiler("ComputeForces");
  KernelComputeForces K(sim);
  cubism::compute<KernelComputeForces,VectorGrid,VectorLab,ScalarGrid,ScalarLab>(K,*sim.vel,*sim.chi);

  // finalize partial sums
  for (const auto& shape : sim.shapes)
    shape->computeForces();
  sim.stopProfiler();
}

ComputeForces::ComputeForces(SimulationData& s) : Operator(s) { }

#define profile( func ) do { } while (0)
using namespace cubism;
class PoissonSolver
{
public:
  virtual ~PoissonSolver() = default;
  virtual void solve(const ScalarGrid *input, ScalarGrid *output) = 0;
};

class ExpAMRSolver : public PoissonSolver
{
  /*
  Method used to solve Poisson's equation: https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
  */
public:
  std::string getName() {
    // ExpAMRSolver == AMRSolver for explicit linear system
    return "ExpAMRSolver";
  }
  // Constructor and destructor
  ExpAMRSolver(SimulationData& s);
  ~ExpAMRSolver() = default;

  //main function used to solve Poisson's equation
  void solve(
      const ScalarGrid *input, 
      ScalarGrid * const output);

protected:
  //this struct contains information such as the currect timestep size, fluid properties and many others
  SimulationData& sim; 

  int rank_;
  MPI_Comm m_comm_;
  int comm_size_;

  static constexpr int BSX_ = VectorBlock::sizeX;
  static constexpr int BSY_ = VectorBlock::sizeY;
  static constexpr int BLEN_ = BSX_ * BSY_;

  //This returns element K_{I1,I2}. It is used when we invert K
  double getA_local(int I1,int I2);

  // Method to add off-diagonal matrix element associated to cell in 'rhsNei' block
  class EdgeCellIndexer; // forward declaration
  void makeFlux(
      const cubism::BlockInfo &rhs_info,
      const int ix,
      const int iy,
      const cubism::BlockInfo &rhsNei,
      const EdgeCellIndexer &indexer,
      SpRowInfo &row) const;

  // Method to compute A and b for the current mesh
  void getMat(); // update LHS and RHS after refinement
  void getVec(); // update initial guess and RHS vecs only

  // Distributed linear system which uses local indexing
  std::unique_ptr<LocalSpMatDnVec> LocalLS_;

  std::vector<long long> Nblocks_xcumsum_;
  std::vector<long long> Nrows_xcumsum_;

  // Edge descriptors to allow algorithmic access to cell indices regardless of edge type
  class CellIndexer{
    public:
      CellIndexer(const ExpAMRSolver& pSolver) : ps(pSolver) {}
      ~CellIndexer() = default;

      long long This(const cubism::BlockInfo &info, const int ix, const int iy) const
      { return blockOffset(info) + (long long)(iy*BSX_ + ix); }

      static bool validXm(const int ix, const int iy)
      { return ix > 0; }
      static bool validXp(const int ix, const int iy)
      { return ix < BSX_ - 1; }
      static bool validYm(const int ix, const int iy)
      { return iy > 0; }
      static bool validYp(const int ix, const int iy)
      { return iy < BSY_ - 1; }

      long long Xmin(const cubism::BlockInfo &info, const int ix, const int iy, const int offset = 0) const
      { return blockOffset(info) + (long long)(iy*BSX_ + offset); }
      long long Xmax(const cubism::BlockInfo &info, const int ix, const int iy, const int offset = 0) const
      { return blockOffset(info) + (long long)(iy*BSX_ + (BSX_-1-offset)); }
      long long Ymin(const cubism::BlockInfo &info, const int ix, const int iy, const int offset = 0) const
      { return blockOffset(info) + (long long)(offset*BSX_ + ix); }
      long long Ymax(const cubism::BlockInfo &info, const int ix, const int iy, const int offset = 0) const
      { return blockOffset(info) + (long long)((BSY_-1-offset)*BSX_ + ix); }

    protected:
      long long blockOffset(const cubism::BlockInfo &info) const
      { return (info.blockID + ps.Nblocks_xcumsum_[ps.sim.tmp->Tree(info).rank()])*BLEN_; }
      static int ix_f(const int ix) { return (ix % (BSX_/2)) * 2; }
      static int iy_f(const int iy) { return (iy % (BSY_/2)) * 2; }

      const ExpAMRSolver &ps; // poisson solver
  };

  class EdgeCellIndexer : public CellIndexer
  {
    public:
      EdgeCellIndexer(const ExpAMRSolver& pSolver) : CellIndexer(pSolver) {}

      // When I am uniform with the neighbouring block
      virtual long long neiUnif(const cubism::BlockInfo &nei_info, const int ix, const int iy) const = 0;

      // When I am finer than neighbouring block
      virtual long long neiInward(const cubism::BlockInfo &info, const int ix, const int iy) const = 0;
      virtual double taylorSign(const int ix, const int iy) const = 0;

      // Indices of coarses cells in neighbouring blocks, to be overridden where appropriate
      virtual int ix_c(const cubism::BlockInfo &info, const int ix) const
      { return info.index[0] % 2 == 0 ? ix/2 : ix/2 + BSX_/2; }
      virtual int iy_c(const cubism::BlockInfo &info, const int iy) const
      { return info.index[1] % 2 == 0 ? iy/2 : iy/2 + BSY_/2; }

      // When I am coarser than neighbouring block
      // neiFine1 must correspond to cells where taylorSign == -1., neiFine2 must correspond to taylorSign == 1.
      virtual long long neiFine1(const cubism::BlockInfo &nei_info, const int ix, const int iy, const int offset = 0) const = 0;
      virtual long long neiFine2(const cubism::BlockInfo &nei_info, const int ix, const int iy, const int offset = 0) const = 0;

      // Indexing aids for derivatives in Taylor approximation in coarse cell
      virtual bool isBD(const int ix, const int iy) const = 0;
      virtual bool isFD(const int ix, const int iy) const = 0;
      virtual long long Nei(const cubism::BlockInfo &info, const int ix, const int iy, const int dist) const = 0; 

      // When I am coarser and need to determine which Zchild I'm next to
      virtual long long Zchild(const cubism::BlockInfo &nei_info, const int ix, const int iy) const = 0;
  };

  // ----------------------------------------------------- Edges perpendicular to x-axis -----------------------------------
  class XbaseIndexer : public EdgeCellIndexer
  {
    public:
      XbaseIndexer(const ExpAMRSolver& pSolver) : EdgeCellIndexer(pSolver) {}

      double taylorSign(const int ix, const int iy) const override
      { return iy % 2 == 0 ? -1.: 1.; }
      bool isBD(const int ix, const int iy) const override 
      { return iy == BSY_ -1 || iy == BSY_/2 - 1; }
      bool isFD(const int ix, const int iy) const override 
      { return iy == 0 || iy == BSY_/2; }
      long long Nei(const cubism::BlockInfo &info, const int ix, const int iy, const int dist) const override
      { return This(info, ix, iy+dist); }
  };

  class XminIndexer : public XbaseIndexer
  {
    public:
      XminIndexer(const ExpAMRSolver& pSolver) : XbaseIndexer(pSolver) {}

      long long neiUnif(const cubism::BlockInfo &nei_info, const int ix, const int iy) const override
      { return Xmax(nei_info, ix, iy); }

      long long neiInward(const cubism::BlockInfo &info, const int ix, const int iy) const override
      { return This(info, ix+1, iy); }

      int ix_c(const cubism::BlockInfo &info, const int ix) const override
      { return BSX_ - 1; }

      long long neiFine1(const cubism::BlockInfo &nei_info, const int ix, const int iy, const int offset = 0) const override
      { return Xmax(nei_info, ix_f(ix), iy_f(iy), offset); }
      long long neiFine2(const cubism::BlockInfo &nei_info, const int ix, const int iy, const int offset = 0) const override
      { return Xmax(nei_info, ix_f(ix), iy_f(iy)+1, offset); }

      long long Zchild(const cubism::BlockInfo &nei_info, const int ix, const int iy) const override
      { return nei_info.Zchild[1][int(iy >= BSY_/2)][0]; }
  };

  class XmaxIndexer : public XbaseIndexer
  {
    public:
      XmaxIndexer(const ExpAMRSolver& pSolver) : XbaseIndexer(pSolver) {}

      long long neiUnif(const cubism::BlockInfo &nei_info, const int ix, const int iy) const override
      { return Xmin(nei_info, ix, iy); }

      long long neiInward(const cubism::BlockInfo &info, const int ix, const int iy) const override
      { return This(info, ix-1, iy); }

      int ix_c(const cubism::BlockInfo &info, const int ix) const override
      { return 0; }

      long long neiFine1(const cubism::BlockInfo &nei_info, const int ix, const int iy, const int offset = 0) const override
      { return Xmin(nei_info, ix_f(ix), iy_f(iy), offset); }
      long long neiFine2(const cubism::BlockInfo &nei_info, const int ix, const int iy, const int offset = 0) const override
      { return Xmin(nei_info, ix_f(ix), iy_f(iy)+1, offset); }

      long long Zchild(const cubism::BlockInfo &nei_info, const int ix, const int iy) const override
      { return nei_info.Zchild[0][int(iy >= BSY_/2)][0]; }
  };

  // ----------------------------------------------------- Edges perpendicular to y-axis -----------------------------------
  class YbaseIndexer : public EdgeCellIndexer
  {
    public:
      YbaseIndexer(const ExpAMRSolver& pSolver) : EdgeCellIndexer(pSolver) {}

      double taylorSign(const int ix, const int iy) const override
      { return ix % 2 == 0 ? -1.: 1.; }
      bool isBD(const int ix, const int iy) const override 
      { return ix == BSX_ -1 || ix == BSX_/2 - 1; }
      bool isFD(const int ix, const int iy) const override 
      { return ix == 0 || ix == BSX_/2; }
      long long Nei(const cubism::BlockInfo &info, const int ix, const int iy, const int dist) const override
      { return This(info, ix+dist, iy); }
  };

  class YminIndexer : public YbaseIndexer
  {
    public:
      YminIndexer(const ExpAMRSolver& pSolver) : YbaseIndexer(pSolver) {}

      long long neiUnif(const cubism::BlockInfo &nei_info, const int ix, const int iy) const override
      { return Ymax(nei_info, ix, iy); }

      long long neiInward(const cubism::BlockInfo &info, const int ix, const int iy) const override
      { return This(info, ix, iy+1); }

      int iy_c(const cubism::BlockInfo &info, const int iy) const override
      { return BSY_ - 1; }

      long long neiFine1(const cubism::BlockInfo &nei_info, const int ix, const int iy, const int offset = 0) const override
      { return Ymax(nei_info, ix_f(ix), iy_f(iy), offset); }
      long long neiFine2(const cubism::BlockInfo &nei_info, const int ix, const int iy, const int offset = 0) const override
      { return Ymax(nei_info, ix_f(ix)+1, iy_f(iy), offset); }

      long long Zchild(const cubism::BlockInfo &nei_info, const int ix, const int iy) const override
      { return nei_info.Zchild[int(ix >= BSX_/2)][1][0]; }
  };

  class YmaxIndexer : public YbaseIndexer
  {
    public:
      YmaxIndexer(const ExpAMRSolver& pSolver) : YbaseIndexer(pSolver) {}

      long long neiUnif(const cubism::BlockInfo &nei_info, const int ix, const int iy) const override
      { return Ymin(nei_info, ix, iy); }

      long long neiInward(const cubism::BlockInfo &info, const int ix, const int iy) const override
      { return This(info, ix, iy-1); }

      int iy_c(const cubism::BlockInfo &info, const int iy) const override
      { return 0; }

      long long neiFine1(const cubism::BlockInfo &nei_info, const int ix, const int iy, const int offset = 0) const override
      { return Ymin(nei_info, ix_f(ix), iy_f(iy), offset); }
      long long neiFine2(const cubism::BlockInfo &nei_info, const int ix, const int iy, const int offset = 0) const override
      { return Ymin(nei_info, ix_f(ix)+1, iy_f(iy), offset); }

      long long Zchild(const cubism::BlockInfo &nei_info, const int ix, const int iy) const override
      { return nei_info.Zchild[int(ix >= BSX_/2)][0][0]; }
  };

  CellIndexer GenericCell;
  XminIndexer XminCell;
  XmaxIndexer XmaxCell;
  YminIndexer YminCell;
  YmaxIndexer YmaxCell;
  // Array of pointers for the indexers above for polymorphism in makeFlux
  std::array<const EdgeCellIndexer*, 4> edgeIndexers;

  std::array<std::pair<long long, double>, 3> D1(const cubism::BlockInfo &info, const EdgeCellIndexer &indexer, const int ix, const int iy) const
  {
    // Scale D1 by h^l/4
    if (indexer.isBD(ix, iy)) 
      return {{ {indexer.Nei(info, ix, iy, -2),  1./8.}, 
                {indexer.Nei(info, ix, iy, -1), -1./2.}, 
                {indexer.This(info, ix, iy),     3./8.} }};
    else if (indexer.isFD(ix, iy)) 
      return {{ {indexer.Nei(info, ix, iy, 2), -1./8.}, 
                {indexer.Nei(info, ix, iy, 1),  1./2.}, 
                {indexer.This(info, ix, iy),   -3./8.} }};

    return {{ {indexer.Nei(info, ix, iy, -1), -1./8.}, 
              {indexer.Nei(info, ix, iy,  1),  1./8.}, 
              {indexer.This(info, ix, iy),     0.} }};
  }

  std::array<std::pair<long long, double>, 3> D2(const cubism::BlockInfo &info, const EdgeCellIndexer &indexer, const int ix, const int iy) const
  {
    // Scale D2 by 0.5*(h^l/4)^2
    if (indexer.isBD(ix, iy)) 
      return {{ {indexer.Nei(info, ix, iy, -2),  1./32.}, 
                {indexer.Nei(info, ix, iy, -1), -1./16.}, 
                {indexer.This(info, ix, iy),     1./32.} }};
    else if (indexer.isFD(ix, iy)) 
      return {{ {indexer.Nei(info, ix, iy, 2),  1./32.}, 
                {indexer.Nei(info, ix, iy, 1), -1./16.}, 
                {indexer.This(info, ix, iy),    1./32.} }};

    return {{ {indexer.Nei(info, ix, iy, -1),  1./32.}, 
              {indexer.Nei(info, ix, iy,  1),  1./32.}, 
              {indexer.This(info, ix, iy),    -1./16.} }};
  }

  void interpolate(
      const cubism::BlockInfo &info_c, const int ix_c, const int iy_c,
      const cubism::BlockInfo &info_f, const long long fine_close_idx, const long long fine_far_idx,
      const double signI, const double signT,
      const EdgeCellIndexer &indexer, SpRowInfo& row) const;
};

double ExpAMRSolver::getA_local(int I1,int I2) //matrix for Poisson's equation on a uniform grid
{
   int j1 = I1 / BSX_;
   int i1 = I1 % BSX_;
   int j2 = I2 / BSX_;
   int i2 = I2 % BSX_;
   if (i1==i2 && j1==j2)
     return 4.0;
   else if (abs(i1-i2) + abs(j1-j2) == 1)
     return -1.0;
   else
     return 0.0;
}

ExpAMRSolver::ExpAMRSolver(SimulationData& s)
  : sim(s), m_comm_(sim.comm), GenericCell(*this),
    XminCell(*this), XmaxCell(*this), YminCell(*this), YmaxCell(*this),
    edgeIndexers{&XminCell, &XmaxCell, &YminCell, &YmaxCell}
{
  // MPI
  MPI_Comm_rank(m_comm_, &rank_);
  MPI_Comm_size(m_comm_, &comm_size_);

  Nblocks_xcumsum_.resize(comm_size_ + 1);
  Nrows_xcumsum_.resize(comm_size_ + 1);

  std::vector<std::vector<double>> L; // lower triangular matrix of Cholesky decomposition
  std::vector<std::vector<double>> L_inv; // inverse of L

  L.resize(BLEN_);
  L_inv.resize(BLEN_);
  for (int i(0); i<BLEN_ ; i++)
  {
    L[i].resize(i+1);
    L_inv[i].resize(i+1);
    // L_inv will act as right block in GJ algorithm, init as identity
    for (int j(0); j<=i; j++){
      L_inv[i][j] = (i == j) ? 1. : 0.;
    }
  }

  // compute the Cholesky decomposition of the preconditioner with Cholesky-Crout
  for (int i(0); i<BLEN_ ; i++)
  {
    double s1 = 0;
    for (int k(0); k<=i-1; k++)
      s1 += L[i][k]*L[i][k];
    L[i][i] = sqrt(getA_local(i,i) - s1);
    for (int j(i+1); j<BLEN_; j++)
    {
      double s2 = 0;
      for (int k(0); k<=i-1; k++)
        s2 += L[i][k]*L[j][k];
      L[j][i] = (getA_local(j,i)-s2) / L[i][i];
    }
  }

  /* Compute the inverse of the Cholesky decomposition L using Gauss-Jordan elimination.
     L will act as the left block (it does not need to be modified in the process), 
     L_inv will act as the right block and at the end of the algo will contain the inverse */
  for (int br(0); br<BLEN_; br++)
  { // 'br' - base row in which all columns up to L_lb[br][br] are already zero
    const double bsf = 1. / L[br][br];
    for (int c(0); c<=br; c++)
      L_inv[br][c] *= bsf;

    for (int wr(br+1); wr<BLEN_; wr++)
    { // 'wr' - working row where elements below L_lb[br][br] will be set to zero
      const double wsf = L[wr][br];
      for (int c(0); c<=br; c++)
        L_inv[wr][c] -= (wsf * L_inv[br][c]);
    }
  }

  // P_inv_ holds inverse preconditionner in row major order!
  std::vector<double> P_inv(BLEN_ * BLEN_);
  for (int i(0); i<BLEN_; i++)
  for (int j(0); j<BLEN_; j++)
  {
    double aux = 0.;
    for (int k(0); k<BLEN_; k++) // P_inv_ = (L^T)^{-1} L^{-1}
      aux += (i <= k && j <=k) ? L_inv[k][i] * L_inv[k][j] : 0.;

    P_inv[i*BLEN_+j] = -aux; // Up to now Cholesky of negative P to avoid complex numbers
  }

  // Create Linear system and backend solver objects
  LocalLS_ = std::make_unique<LocalSpMatDnVec>(m_comm_, BSX_*BSY_, sim.bMeanConstraint, P_inv);
}
void ExpAMRSolver::interpolate(
    const BlockInfo &info_c, const int ix_c, const int iy_c,
    const BlockInfo &info_f, const long long fine_close_idx, const long long fine_far_idx,
    const double signInt, const double signTaylor, // sign of interpolation and sign of taylor
    const EdgeCellIndexer &indexer, SpRowInfo& row) const
{
  const int rank_c = sim.tmp->Tree(info_c).rank();
  const int rank_f = sim.tmp->Tree(info_f).rank();

  // 2./3.*p_fine_close_idx - 1./5.*p_fine_far_idx
  row.mapColVal(rank_f, fine_close_idx, signInt * 2./3.);
  row.mapColVal(rank_f, fine_far_idx,  -signInt * 1./5.);

  // 8./15 * p_T, constant term
  const double tf = signInt * 8./15.; // common factor for all terms of Taylor expansion
  row.mapColVal(rank_c, indexer.This(info_c, ix_c, iy_c), tf);

  std::array<std::pair<long long, double>, 3> D;

  // first derivative
  D = D1(info_c, indexer, ix_c, iy_c);
  for (int i(0); i < 3; i++)
    row.mapColVal(rank_c, D[i].first, signTaylor * tf * D[i].second);

  // second derivative
  D = D2(info_c, indexer, ix_c, iy_c);
  for (int i(0); i < 3; i++)
    row.mapColVal(rank_c, D[i].first, tf * D[i].second);
}

// Methods for cell centric construction of discrete Laplace operator
void ExpAMRSolver::makeFlux(
  const BlockInfo &rhs_info,
  const int ix,
  const int iy,
  const BlockInfo &rhsNei,
  const EdgeCellIndexer &indexer,
  SpRowInfo &row) const
{
  const long long sfc_idx = indexer.This(rhs_info, ix, iy);

  if (this->sim.tmp->Tree(rhsNei).Exists())
  { 
    const int nei_rank = sim.tmp->Tree(rhsNei).rank();
    const long long nei_idx = indexer.neiUnif(rhsNei, ix, iy);

    // Map flux associated to out-of-block edges at the same level of refinement
    row.mapColVal(nei_rank, nei_idx, 1.);
    row.mapColVal(sfc_idx, -1.);
  }
  else if (this->sim.tmp->Tree(rhsNei).CheckCoarser())
  {
    const BlockInfo &rhsNei_c = this->sim.tmp->getBlockInfoAll(rhs_info.level - 1 , rhsNei.Zparent);
    const int ix_c = indexer.ix_c(rhs_info, ix);
    const int iy_c = indexer.iy_c(rhs_info, iy);
    const long long inward_idx = indexer.neiInward(rhs_info, ix, iy);
    const double signTaylor = indexer.taylorSign(ix, iy);

    interpolate(rhsNei_c, ix_c, iy_c, rhs_info, sfc_idx, inward_idx, 1., signTaylor, indexer, row);
    row.mapColVal(sfc_idx, -1.);
  }
  else if (this->sim.tmp->Tree(rhsNei).CheckFiner())
  {
    const BlockInfo &rhsNei_f = this->sim.tmp->getBlockInfoAll(rhs_info.level + 1, indexer.Zchild(rhsNei, ix, iy));
    const int nei_rank = this->sim.tmp->Tree(rhsNei_f).rank();

    // F1
    long long fine_close_idx = indexer.neiFine1(rhsNei_f, ix, iy, 0);
    long long fine_far_idx   = indexer.neiFine1(rhsNei_f, ix, iy, 1);
    row.mapColVal(nei_rank, fine_close_idx, 1.);
    interpolate(rhs_info, ix, iy, rhsNei_f, fine_close_idx, fine_far_idx, -1., -1., indexer, row);
    // F2
    fine_close_idx = indexer.neiFine2(rhsNei_f, ix, iy, 0);
    fine_far_idx   = indexer.neiFine2(rhsNei_f, ix, iy, 1);
    row.mapColVal(nei_rank, fine_close_idx, 1.);
    interpolate(rhs_info, ix, iy, rhsNei_f, fine_close_idx, fine_far_idx, -1.,  1., indexer, row);
  }
  else { throw std::runtime_error("Neighbour doesn't exist, isn't coarser, nor finer..."); }
}

void ExpAMRSolver::getMat()
{
  sim.startProfiler("Poisson solver: LS");

  //This returns an array with the blocks that the coarsest possible 
  //mesh would have (i.e. all blocks are at level 0)
  std::array<int, 3> blocksPerDim = sim.pres->getMaxBlocks();

  //Get a vector of all BlockInfos of the grid we're interested in
  sim.tmp->UpdateBlockInfoAll_States(true); // update blockID's for blocks from other ranks
  std::vector<cubism::BlockInfo>&  RhsInfo = sim.tmp->getBlocksInfo();
  const int Nblocks = RhsInfo.size();
  const int N = BSX_*BSY_*Nblocks;

  // Reserve sufficient memory for LS proper to the rank
  LocalLS_->reserve(N);

  // Calculate cumulative sums for blocks and rows for correct global indexing
  const long long Nblocks_long = Nblocks;
  MPI_Allgather(&Nblocks_long, 1, MPI_LONG_LONG, Nblocks_xcumsum_.data(), 1, MPI_LONG_LONG, m_comm_);
  for (int i(Nblocks_xcumsum_.size()-1); i > 0; i--)
  {
    Nblocks_xcumsum_[i] = Nblocks_xcumsum_[i-1]; // shift to right for rank 'i+1' to have cumsum of rank 'i'
  }
  
  // Set cumsum for rank 0 to zero
  Nblocks_xcumsum_[0] = 0;
  Nrows_xcumsum_[0] = 0;

  // Perform cumulative sum
  for (size_t i(1); i < Nblocks_xcumsum_.size(); i++)
  {
    Nblocks_xcumsum_[i] += Nblocks_xcumsum_[i-1];
    Nrows_xcumsum_[i] = BLEN_*Nblocks_xcumsum_[i];
  }

  // No parallel for to ensure COO are ordered at construction
  for(int i=0; i<Nblocks; i++)
  {    
    const BlockInfo &rhs_info = RhsInfo[i];

    //1.Check if this is a boundary block
    const int aux = 1 << rhs_info.level; // = 2^level
    const int MAX_X_BLOCKS = blocksPerDim[0]*aux - 1; //this means that if level 0 has blocksPerDim[0] blocks in the x-direction, level rhs.level will have this many blocks
    const int MAX_Y_BLOCKS = blocksPerDim[1]*aux - 1; //this means that if level 0 has blocksPerDim[1] blocks in the y-direction, level rhs.level will have this many blocks

    //index is the (i,j) coordinates of a block at the current level 
    std::array<bool, 4> isBoundary;
    isBoundary[0] = (rhs_info.index[0] == 0           ); // Xm, same order as faceIndexers made in constructor!
    isBoundary[1] = (rhs_info.index[0] == MAX_X_BLOCKS); // Xp
    isBoundary[2] = (rhs_info.index[1] == 0           ); // Ym
    isBoundary[3] = (rhs_info.index[1] == MAX_Y_BLOCKS); // Yp

    std::array<bool, 2> isPeriodic; // same dimension ordering as isBoundary
    isPeriodic[0] = (cubismBCX == periodic);
    isPeriodic[1] = (cubismBCY == periodic);

    //2.Access the block's neighbors (for the Poisson solve in two dimensions we care about four neighbors in total)
    std::array<long long, 4> Z;
    Z[0] = rhs_info.Znei[1-1][1][1]; // Xm
    Z[1] = rhs_info.Znei[1+1][1][1]; // Xp
    Z[2] = rhs_info.Znei[1][1-1][1]; // Ym
    Z[3] = rhs_info.Znei[1][1+1][1]; // Yp
    //rhs.Z == rhs.Znei[1][1][1] is true always

    std::array<const BlockInfo*, 4> rhsNei;
    rhsNei[0] = &(this->sim.tmp->getBlockInfoAll(rhs_info.level, Z[0]));
    rhsNei[1] = &(this->sim.tmp->getBlockInfoAll(rhs_info.level, Z[1]));
    rhsNei[2] = &(this->sim.tmp->getBlockInfoAll(rhs_info.level, Z[2]));
    rhsNei[3] = &(this->sim.tmp->getBlockInfoAll(rhs_info.level, Z[3]));

    // Record local index of row which is to be modified with bMeanConstraint reduction result
    if (sim.bMeanConstraint &&
        rhs_info.index[0] == 0 &&
        rhs_info.index[1] == 0 &&
        rhs_info.index[2] == 0)
      LocalLS_->set_bMeanRow(GenericCell.This(rhs_info, 0, 0) - Nrows_xcumsum_[rank_]);

    //For later: there's a total of three boolean variables:
    // I.   grid->Tree(rhsNei_west).Exists()
    // II.  grid->Tree(rhsNei_west).CheckCoarser()
    // III. grid->Tree(rhsNei_west).CheckFiner()
    // And only one of them is true

    // Add matrix elements associated to interior cells of a block
    for(int iy=0; iy<BSY_; iy++)
    for(int ix=0; ix<BSX_; ix++)
    { // Following logic needs to be in for loop to assure cooRows are ordered
      const long long sfc_idx = GenericCell.This(rhs_info, ix, iy);

      if ((ix > 0 && ix < BSX_-1) && (iy > 0 && iy < BSY_-1))
      { // Inner cells, push back in ascending order for column index
        LocalLS_->cooPushBackVal(1, sfc_idx, GenericCell.This(rhs_info, ix, iy-1));
        LocalLS_->cooPushBackVal(1, sfc_idx, GenericCell.This(rhs_info, ix-1, iy));
        LocalLS_->cooPushBackVal(-4, sfc_idx, sfc_idx);
        LocalLS_->cooPushBackVal(1, sfc_idx, GenericCell.This(rhs_info, ix+1, iy));
        LocalLS_->cooPushBackVal(1, sfc_idx, GenericCell.This(rhs_info, ix, iy+1));
      }
      else
      { // See which edge is shared with a cell from different block
        std::array<bool, 4> validNei;
        validNei[0] = GenericCell.validXm(ix, iy); 
        validNei[1] = GenericCell.validXp(ix, iy); 
        validNei[2] = GenericCell.validYm(ix, iy); 
        validNei[3] = GenericCell.validYp(ix, iy);  

        // Get index of cell accross the edge (correct only for cells in this block)
        std::array<long long, 4> idxNei;
        idxNei[0] = GenericCell.This(rhs_info, ix-1, iy);
        idxNei[1] = GenericCell.This(rhs_info, ix+1, iy);
        idxNei[2] = GenericCell.This(rhs_info, ix, iy-1);
        idxNei[3] = GenericCell.This(rhs_info, ix, iy+1);

				SpRowInfo row(sim.tmp->Tree(rhs_info).rank(), sfc_idx, 8);
        for (int j(0); j < 4; j++)
        { // Iterate over each edge of cell
          if (validNei[j])
          { // This edge is 'inner' wrt to the block
            row.mapColVal(idxNei[j], 1);
            row.mapColVal(sfc_idx, -1);
          }
          else if (!isBoundary[j] || (isBoundary[j] && isPeriodic[j/2]))
            this->makeFlux(rhs_info, ix, iy, *rhsNei[j], *edgeIndexers[j], row);
        }

        LocalLS_->cooPushBackRow(row);
      }
    } // for(int iy=0; iy<BSY_; iy++) for(int ix=0; ix<BSX_; ix++)
  } // for(int i=0; i< Nblocks; i++)

  LocalLS_->make(Nrows_xcumsum_);

  sim.stopProfiler();
}

void ExpAMRSolver::getVec()
{
  //Get a vector of all BlockInfos of the grid we're interested in
  std::vector<cubism::BlockInfo>&  RhsInfo = sim.tmp->getBlocksInfo();
  std::vector<cubism::BlockInfo>&  zInfo = sim.pres->getBlocksInfo();
  const int Nblocks = RhsInfo.size();
  std::vector<double>& x  = LocalLS_->get_x();
  std::vector<double>& b  = LocalLS_->get_b();
  std::vector<double>& h2 = LocalLS_->get_h2();
  const long long shift = -Nrows_xcumsum_[rank_];

  // Copy RHS LHS vec initial guess, if LS was updated, updateMat reallocates sufficient memory
  #pragma omp parallel for
  for(int i=0; i< Nblocks; i++)
  {    
    const BlockInfo &rhs_info = RhsInfo[i];
    const ScalarBlock & __restrict__ rhs  = *(ScalarBlock*) RhsInfo[i].ptrBlock;
    const ScalarBlock & __restrict__ p  = *(ScalarBlock*) zInfo[i].ptrBlock;

    h2[i] = RhsInfo[i].h * RhsInfo[i].h;
    // Construct RHS and x_0 vectors for linear system
    for(int iy=0; iy<BSY_; iy++)
    for(int ix=0; ix<BSX_; ix++)
    {
      const long long sfc_loc = GenericCell.This(rhs_info, ix, iy) + shift;
      if (sim.bMeanConstraint &&
          rhs_info.index[0] == 0 &&
          rhs_info.index[1] == 0 &&
          rhs_info.index[2] == 0 &&
          ix == 0 && iy == 0)
        b[sfc_loc] = 0.;
      else
        b[sfc_loc] = rhs(ix,iy).s;

      x[sfc_loc] = p(ix,iy).s;
    }
  }
}

void ExpAMRSolver::solve(
    const ScalarGrid *input, 
    ScalarGrid * const output)
{

  if (rank_ == 0) {
    if (sim.verbose)
      std::cout << "--------------------- Calling on ExpAMRSolver.solve() ------------------------\n";
    else
      std::cout << '\n';
  }

  const double max_error = this->sim.step < 10 ? 0.0 : sim.PoissonTol;
  const double max_rel_error = this->sim.step < 10 ? 0.0 : sim.PoissonTolRel;
  const int max_restarts = this->sim.step < 10 ? 100 : sim.maxPoissonRestarts;

  if (sim.pres->UpdateFluxCorrection)
  {
    sim.pres->UpdateFluxCorrection = false;
    this->getMat();
    this->getVec();
    LocalLS_->solveWithUpdate(max_error, max_rel_error, max_restarts);
  }
  else
  {
    this->getVec();
    LocalLS_->solveNoUpdate(max_error, max_rel_error, max_restarts);
  }

  //Now that we found the solution, we just substract the mean to get a zero-mean solution. 
  //This can be done because the solver only cares about grad(P) = grad(P-mean(P))
  std::vector<cubism::BlockInfo>&  zInfo = sim.pres->getBlocksInfo();
  const int Nblocks = zInfo.size();
  const std::vector<double>& x = LocalLS_->get_x();

  double avg = 0;
  double avg1 = 0;
  #pragma omp parallel for reduction (+:avg,avg1)
  for(int i=0; i< Nblocks; i++)
  {
     ScalarBlock& P  = *(ScalarBlock*) zInfo[i].ptrBlock;
     const double vv = zInfo[i].h*zInfo[i].h;
     for(int iy=0; iy<BSY_; iy++)
     for(int ix=0; ix<BSX_; ix++)
     {
         P(ix,iy).s = x[i*BSX_*BSY_ + iy*BSX_ + ix];
         avg += P(ix,iy).s * vv;
         avg1 += vv;
     }
  }
  double quantities[2] = {avg,avg1};
  MPI_Allreduce(MPI_IN_PLACE, &quantities, 2, MPI_DOUBLE, MPI_SUM, m_comm_);
  avg = quantities[0]; avg1 = quantities[1] ;
  avg = avg/avg1;
  #pragma omp parallel for 
  for(int i=0; i< Nblocks; i++)
  {
     ScalarBlock& P  = *(ScalarBlock*) zInfo[i].ptrBlock;
     for(int iy=0; iy<BSY_; iy++)
     for(int ix=0; ix<BSX_; ix++)
        P(ix,iy).s += -avg;
  }
}

std::shared_ptr<PoissonSolver> makePoissonSolver(SimulationData& s);
std::shared_ptr<PoissonSolver> makePoissonSolver(SimulationData& s)
{
  if (s.poissonSolver == "cuda_iterative") 
  {
#ifdef GPU_POISSON
    if (! _DOUBLE_PRECISION_ )
      throw std::runtime_error( 
          "Poisson solver: \"" + s.poissonSolver + "\" must be compiled with in double precision mode!" );
    return std::make_shared<ExpAMRSolver>(s);
#else
    throw std::runtime_error(
        "Poisson solver: \"" + s.poissonSolver + "\" must be compiled with the -DGPU_POISSON flag!"); 
#endif
  } 
  else {
    throw std::invalid_argument(
        "Poisson solver: \"" + s.poissonSolver + "\" unrecognized!");
  }
}

class Shape;

class PressureSingle : public Operator
{
protected:
  const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();

  std::shared_ptr<PoissonSolver> pressureSolver;

  void preventCollidingObstacles() const;
  void pressureCorrection(const Real dt);
  void integrateMomenta(Shape * const shape) const;
  void penalize(const Real dt) const;

 public:
  void operator() (const Real dt) override;

  PressureSingle(SimulationData& s);
  ~PressureSingle();

  std::string getName() override
  {
    return "PressureSingle";
  }
};

using namespace cubism;

using CHI_MAT = Real[VectorBlock::sizeY][VectorBlock::sizeX];
using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];

//#define EXPL_INTEGRATE_MOM

namespace {

void ComputeJ(const Real * Rc, const Real * R, const Real * N, const Real * I, Real *J)
{
    //Invert I
    const Real m00 = 1.0; //I[0]; //set to these values for 2D!
    const Real m01 = 0.0; //I[3]; //set to these values for 2D!
    const Real m02 = 0.0; //I[4]; //set to these values for 2D!
    const Real m11 = 1.0; //I[1]; //set to these values for 2D!
    const Real m12 = 0.0; //I[5]; //set to these values for 2D!
    const Real m22 = I[5];//I[2]; //set to these values for 2D!
    Real a00 = m22*m11 - m12*m12;
    Real a01 = m02*m12 - m22*m01;
    Real a02 = m01*m12 - m02*m11;
    Real a11 = m22*m00 - m02*m02;
    Real a12 = m01*m02 - m00*m12;
    Real a22 = m00*m11 - m01*m01;
    const Real determinant =  1.0/((m00 * a00) + (m01 * a01) + (m02 * a02));
    a00 *= determinant;
    a01 *= determinant;
    a02 *= determinant;
    a11 *= determinant;
    a12 *= determinant;
    a22 *= determinant;

    const Real aux_0 = ( Rc[1] - R[1] )*N[2] - ( Rc[2] - R[2] )*N[1];
    const Real aux_1 = ( Rc[2] - R[2] )*N[0] - ( Rc[0] - R[0] )*N[2];
    const Real aux_2 = ( Rc[0] - R[0] )*N[1] - ( Rc[1] - R[1] )*N[0];
    J[0] = a00*aux_0 + a01*aux_1 + a02*aux_2;
    J[1] = a01*aux_0 + a11*aux_1 + a12*aux_2;
    J[2] = a02*aux_0 + a12*aux_1 + a22*aux_2;
}


void ElasticCollision (const Real  m1,const Real  m2,
                       const Real *I1,const Real *I2,
                       const Real *v1,const Real *v2,
                       const Real *o1,const Real *o2,
                       Real *hv1,Real *hv2,
                       Real *ho1,Real *ho2,
                       const Real *C1,const Real *C2,
                       const Real  NX,const Real  NY,const Real NZ,
                       const Real  CX,const Real  CY,const Real CZ,
                       Real *vc1,Real *vc2)
{
    const Real e = 1.0; // coefficient of restitution
    const Real N[3] ={NX,NY,NZ};
    const Real C[3] ={CX,CY,CZ};

    const Real k1[3] = { N[0]/m1, N[1]/m1, N[2]/m1};
    const Real k2[3] = {-N[0]/m2,-N[1]/m2,-N[2]/m2};
    Real J1[3];
    Real J2[3]; 
    ComputeJ(C,C1,N,I1,J1);
    ComputeJ(C,C2,N,I2,J2);
    J2[0] = -J2[0];
    J2[1] = -J2[1];
    J2[2] = -J2[2];

    Real u1DEF[3];
    u1DEF[0] = vc1[0] - v1[0] - ( o1[1]*(C[2]-C1[2]) - o1[2]*(C[1]-C1[1]) );
    u1DEF[1] = vc1[1] - v1[1] - ( o1[2]*(C[0]-C1[0]) - o1[0]*(C[2]-C1[2]) );
    u1DEF[2] = vc1[2] - v1[2] - ( o1[0]*(C[1]-C1[1]) - o1[1]*(C[0]-C1[0]) );
    Real u2DEF[3];
    u2DEF[0] = vc2[0] - v2[0] - ( o2[1]*(C[2]-C2[2]) - o2[2]*(C[1]-C2[1]) );
    u2DEF[1] = vc2[1] - v2[1] - ( o2[2]*(C[0]-C2[0]) - o2[0]*(C[2]-C2[2]) );
    u2DEF[2] = vc2[2] - v2[2] - ( o2[0]*(C[1]-C2[1]) - o2[1]*(C[0]-C2[0]) );

    const Real nom = e*( (vc1[0]-vc2[0])*N[0] + 
                           (vc1[1]-vc2[1])*N[1] + 
                           (vc1[2]-vc2[2])*N[2] )
                       + ( (v1[0]-v2[0] + u1DEF[0] - u2DEF[0] )*N[0] + 
                           (v1[1]-v2[1] + u1DEF[1] - u2DEF[1] )*N[1] + 
                           (v1[2]-v2[2] + u1DEF[2] - u2DEF[2] )*N[2] )
                  +( (o1[1]*(C[2]-C1[2]) - o1[2]*(C[1]-C1[1]) )* N[0]+
                     (o1[2]*(C[0]-C1[0]) - o1[0]*(C[2]-C1[2]) )* N[1]+
                     (o1[0]*(C[1]-C1[1]) - o1[1]*(C[0]-C1[0]) )* N[2])
                  -( (o2[1]*(C[2]-C2[2]) - o2[2]*(C[1]-C2[1]) )* N[0]+
                     (o2[2]*(C[0]-C2[0]) - o2[0]*(C[2]-C2[2]) )* N[1]+
                     (o2[0]*(C[1]-C2[1]) - o2[1]*(C[0]-C2[0]) )* N[2]);

    const Real denom = -(1.0/m1+1.0/m2) + 
               +( ( J1[1]*(C[2]-C1[2]) - J1[2]*(C[1]-C1[1]) ) *(-N[0])+
                  ( J1[2]*(C[0]-C1[0]) - J1[0]*(C[2]-C1[2]) ) *(-N[1])+
                  ( J1[0]*(C[1]-C1[1]) - J1[1]*(C[0]-C1[0]) ) *(-N[2]))
               -( ( J2[1]*(C[2]-C2[2]) - J2[2]*(C[1]-C2[1]) ) *(-N[0])+
                  ( J2[2]*(C[0]-C2[0]) - J2[0]*(C[2]-C2[2]) ) *(-N[1])+
                  ( J2[0]*(C[1]-C2[1]) - J2[1]*(C[0]-C2[0]) ) *(-N[2]));
    const Real impulse = nom/(denom+1e-21);
    hv1[0] = v1[0] + k1[0]*impulse;
    hv1[1] = v1[1] + k1[1]*impulse;
    hv1[2] = v1[2] + k1[2]*impulse;
    hv2[0] = v2[0] + k2[0]*impulse;
    hv2[1] = v2[1] + k2[1]*impulse;
    hv2[2] = v2[2] + k2[2]*impulse;
    ho1[0] = o1[0] + J1[0]*impulse;
    ho1[1] = o1[1] + J1[1]*impulse;
    ho1[2] = o1[2] + J1[2]*impulse;
    ho2[0] = o2[0] + J2[0]*impulse;
    ho2[1] = o2[1] + J2[1]*impulse;
    ho2[2] = o2[2] + J2[2]*impulse;
}

}//namespace

struct pressureCorrectionKernel
{
  pressureCorrectionKernel(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  const cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0}};
  const std::vector<cubism::BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();

  void operator()(ScalarLab & P, const cubism::BlockInfo& info) const
  {
    const Real h = info.h, pFac = -0.5*sim.dt*h;
    VectorBlock&__restrict__ tmpV = *(VectorBlock*)  tmpVInfo[info.blockID].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      tmpV(ix,iy).u[0] = pFac *(P(ix+1,iy).s-P(ix-1,iy).s);
      tmpV(ix,iy).u[1] = pFac *(P(ix,iy+1).s-P(ix,iy-1).s);
    }
    BlockCase<VectorBlock> * tempCase = (BlockCase<VectorBlock> *)(tmpVInfo[info.blockID].auxiliary);
    VectorBlock::ElementType * faceXm = nullptr;
    VectorBlock::ElementType * faceXp = nullptr;
    VectorBlock::ElementType * faceYm = nullptr;
    VectorBlock::ElementType * faceYp = nullptr;
    if (tempCase != nullptr)
    {
      faceXm = tempCase -> storedFace[0] ?  & tempCase -> m_pData[0][0] : nullptr;
      faceXp = tempCase -> storedFace[1] ?  & tempCase -> m_pData[1][0] : nullptr;
      faceYm = tempCase -> storedFace[2] ?  & tempCase -> m_pData[2][0] : nullptr;
      faceYp = tempCase -> storedFace[3] ?  & tempCase -> m_pData[3][0] : nullptr;
    }
    if (faceXm != nullptr)
    {
      int ix = 0;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      {
        faceXm[iy].clear();
        faceXm[iy].u[0] = pFac*(P(ix-1,iy).s+P(ix,iy).s);
      }
    }
    if (faceXp != nullptr)
    {
      int ix = VectorBlock::sizeX-1;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      {
        faceXp[iy].clear();
        faceXp[iy].u[0] = -pFac*(P(ix+1,iy).s+P(ix,iy).s);
      }
    }
    if (faceYm != nullptr)
    {
      int iy = 0;
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        faceYm[ix].clear();
        faceYm[ix].u[1] = pFac*(P(ix,iy-1).s+P(ix,iy).s);
      }
    }
    if (faceYp != nullptr)
    {
      int iy = VectorBlock::sizeY-1;
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        faceYp[ix].clear();
        faceYp[ix].u[1] = -pFac*(P(ix,iy+1).s+P(ix,iy).s);
      }
    }
  }
};

void PressureSingle::pressureCorrection(const Real dt)
{
  const pressureCorrectionKernel K(sim);
  cubism::compute<ScalarLab>(K,sim.pres,sim.tmpV);

  std::vector<cubism::BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
  #pragma omp parallel for
  for (size_t i=0; i < velInfo.size(); i++)
  {
      const Real ih2 = 1.0/velInfo[i].h/velInfo[i].h;
      VectorBlock&__restrict__   V = *(VectorBlock*)  velInfo[i].ptrBlock;
      VectorBlock&__restrict__   tmpV = *(VectorBlock*) tmpVInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        V(ix,iy).u[0] += tmpV(ix,iy).u[0]*ih2;
        V(ix,iy).u[1] += tmpV(ix,iy).u[1]*ih2;
      }
  }
}

void PressureSingle::integrateMomenta(Shape * const shape) const
{
  const size_t Nblocks = velInfo.size();

  const std::vector<ObstacleBlock*> & OBLOCK = shape->obstacleBlocks;
  const Real Cx = shape->centerOfMass[0];
  const Real Cy = shape->centerOfMass[1];
  Real PM=0, PJ=0, PX=0, PY=0, UM=0, VM=0, AM=0; //linear momenta

  #pragma omp parallel for reduction(+:PM,PJ,PX,PY,UM,VM,AM)
  for(size_t i=0; i<Nblocks; i++)
  {
    const VectorBlock& __restrict__ VEL = *(VectorBlock*)velInfo[i].ptrBlock;
    const Real hsq = velInfo[i].h*velInfo[i].h;

    if(OBLOCK[velInfo[i].blockID] == nullptr) continue;
    const CHI_MAT & __restrict__ chi = OBLOCK[velInfo[i].blockID]->chi;
    const UDEFMAT & __restrict__ udef = OBLOCK[velInfo[i].blockID]->udef;
    #ifndef EXPL_INTEGRATE_MOM
      const Real lambdt = sim.lambda * sim.dt;
    #endif

    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      if (chi[iy][ix] <= 0) continue;
      const Real udiff[2] = {
        VEL(ix,iy).u[0] - udef[iy][ix][0], VEL(ix,iy).u[1] - udef[iy][ix][1]
      };
      #ifdef EXPL_INTEGRATE_MOM
        const Real F = hsq * chi[iy][ix];
      #else
        //const Real Xlamdt = chi[iy][ix] * lambdt;
        //need to use unmollified version when H(x) appears in fractions
        const Real Xlamdt = chi[iy][ix] >= 0.5 ? lambdt:0.0;
        const Real F = hsq * Xlamdt / (1 + Xlamdt);
      #endif
      Real p[2]; velInfo[i].pos(p, ix, iy); p[0] -= Cx; p[1] -= Cy;
      PM += F;
      PJ += F * (p[0]*p[0] + p[1]*p[1]);
      PX += F * p[0];  PY += F * p[1];
      UM += F * udiff[0]; VM += F * udiff[1];
      AM += F * (p[0]*udiff[1] - p[1]*udiff[0]);
    }
  }
  Real quantities[7] = {PM,PJ,PX,PY,UM,VM,AM};
  MPI_Allreduce(MPI_IN_PLACE, quantities, 7, MPI_Real, MPI_SUM, sim.chi->getWorldComm());
  PM = quantities[0]; 
  PJ = quantities[1]; 
  PX = quantities[2]; 
  PY = quantities[3]; 
  UM = quantities[4]; 
  VM = quantities[5]; 
  AM = quantities[6];

  shape->fluidAngMom = AM; shape->fluidMomX = UM; shape->fluidMomY = VM;
  shape->penalDX=PX; shape->penalDY=PY; shape->penalM=PM; shape->penalJ=PJ;
}

void PressureSingle::penalize(const Real dt) const
{
  std::vector<cubism::BlockInfo>& chiInfo   = sim.chi->getBlocksInfo();

  const size_t Nblocks = velInfo.size();

  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  for (const auto& shape : sim.shapes)
  {
    const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
    const ObstacleBlock*const o = OBLOCK[velInfo[i].blockID];
    if (o == nullptr) continue;

    const Real u_s = shape->u;
    const Real v_s = shape->v;
    const Real omega_s = shape->omega;
    const Real Cx = shape->centerOfMass[0];
    const Real Cy = shape->centerOfMass[1];

    const CHI_MAT & __restrict__ X = o->chi;
    const UDEFMAT & __restrict__ UDEF = o->udef;
    const ScalarBlock& __restrict__ CHI = *(ScalarBlock*)chiInfo[i].ptrBlock;
          VectorBlock& __restrict__   V = *(VectorBlock*)velInfo[i].ptrBlock;

    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      // What if multiple obstacles share a block? Do not write udef onto
      // grid if CHI stored on the grid is greater than obst's CHI.
      if(CHI(ix,iy).s > X[iy][ix]) continue;
      if(X[iy][ix] <= 0) continue; // no need to do anything

      Real p[2];
      velInfo[i].pos(p, ix, iy);
      p[0] -= Cx;
      p[1] -= Cy;
      #ifndef EXPL_INTEGRATE_MOM
        //const Real alpha = 1/(1 + sim.lambda * dt * X[iy][ix]);
        //need to use unmollified version when H(x) appears in fractions
        const Real alpha = X[iy][ix] > 0.5 ? 1/(1 + sim.lambda * dt) : 1;
      #else
        const Real alpha = 1 - X[iy][ix];
      #endif

      const Real US = u_s - omega_s * p[1] + UDEF[iy][ix][0];
      const Real VS = v_s + omega_s * p[0] + UDEF[iy][ix][1];
      V(ix,iy).u[0] = alpha*V(ix,iy).u[0] + (1-alpha)*US;
      V(ix,iy).u[1] = alpha*V(ix,iy).u[1] + (1-alpha)*VS;
    }
  }
}

struct updatePressureRHS
{
  // RHS of Poisson equation is div(u) - chi * div(u_def)
  // It is computed here and stored in TMP

  updatePressureRHS(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0,1}};
  cubism::StencilInfo stencil2{-1, -1, 0, 2, 2, 1, false, {0,1}};
  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& chiInfo = sim.chi->getBlocksInfo();

  void operator()(VectorLab & velLab, VectorLab & uDefLab, const cubism::BlockInfo& info, const cubism::BlockInfo& info2) const
  {
    const Real h = info.h;
    const Real facDiv = 0.5*h/sim.dt;
    ScalarBlock& __restrict__ TMP = *(ScalarBlock*) tmpInfo[info.blockID].ptrBlock;
    ScalarBlock& __restrict__ CHI = *(ScalarBlock*) chiInfo[info.blockID].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      TMP(ix, iy).s  =   facDiv                *( (velLab(ix+1,iy).u[0] -  velLab(ix-1,iy).u[0])
                                               +  (velLab(ix,iy+1).u[1] -  velLab(ix,iy-1).u[1]));
      TMP(ix, iy).s += - facDiv * CHI(ix,iy).s *((uDefLab(ix+1,iy).u[0] - uDefLab(ix-1,iy).u[0])
                                               + (uDefLab(ix,iy+1).u[1] - uDefLab(ix,iy-1).u[1]));
    }
    BlockCase<ScalarBlock> * tempCase = (BlockCase<ScalarBlock> *)(tmpInfo[info.blockID].auxiliary);
    ScalarBlock::ElementType * faceXm = nullptr;
    ScalarBlock::ElementType * faceXp = nullptr;
    ScalarBlock::ElementType * faceYm = nullptr;
    ScalarBlock::ElementType * faceYp = nullptr;
    if (tempCase != nullptr)
    {
      faceXm = tempCase -> storedFace[0] ?  & tempCase -> m_pData[0][0] : nullptr;
      faceXp = tempCase -> storedFace[1] ?  & tempCase -> m_pData[1][0] : nullptr;
      faceYm = tempCase -> storedFace[2] ?  & tempCase -> m_pData[2][0] : nullptr;
      faceYp = tempCase -> storedFace[3] ?  & tempCase -> m_pData[3][0] : nullptr;
    }
    if (faceXm != nullptr)
    {
      int ix = 0;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      {
        faceXm[iy].s  =  facDiv                *( velLab(ix-1,iy).u[0] +  velLab(ix,iy).u[0]) ;
        faceXm[iy].s += -(facDiv * CHI(ix,iy).s)*(uDefLab(ix-1,iy).u[0] + uDefLab(ix,iy).u[0]) ;
      }
    }
    if (faceXp != nullptr)
    {
      int ix = VectorBlock::sizeX-1;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      {
        faceXp[iy].s  = -facDiv               *( velLab(ix+1,iy).u[0] +  velLab(ix,iy).u[0]);
        faceXp[iy].s -= -(facDiv *CHI(ix,iy).s)*(uDefLab(ix+1,iy).u[0] + uDefLab(ix,iy).u[0]);
      }
    }
    if (faceYm != nullptr)
    {
      int iy = 0;
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        faceYm[ix].s  =  facDiv               *( velLab(ix,iy-1).u[1] +  velLab(ix,iy).u[1]);
        faceYm[ix].s += -(facDiv *CHI(ix,iy).s)*(uDefLab(ix,iy-1).u[1] + uDefLab(ix,iy).u[1]);
      }
    }
    if (faceYp != nullptr)
    {
      int iy = VectorBlock::sizeY-1;
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        faceYp[ix].s  = -facDiv               *( velLab(ix,iy+1).u[1] +  velLab(ix,iy).u[1]);
        faceYp[ix].s -= -(facDiv *CHI(ix,iy).s)*(uDefLab(ix,iy+1).u[1] + uDefLab(ix,iy).u[1]);
      }
    }
  }
};

struct updatePressureRHS1
{
  // RHS of Poisson equation is div(u) - chi * div(u_def)
  // It is computed here and stored in TMP

  updatePressureRHS1(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0}};
  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& poldInfo = sim.pold->getBlocksInfo();

  void operator()(ScalarLab & lab, const cubism::BlockInfo& info) const
  {
    ScalarBlock& __restrict__ TMP = *(ScalarBlock*) tmpInfo[info.blockID].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      TMP(ix, iy).s  -=  ( ((lab(ix-1,iy).s + lab(ix+1,iy).s) + (lab(ix,iy-1).s + lab(ix,iy+1).s)) - 4.0*lab(ix,iy).s);

    BlockCase<ScalarBlock> * tempCase = (BlockCase<ScalarBlock> *)(tmpInfo[info.blockID].auxiliary);
    ScalarBlock::ElementType * faceXm = nullptr;
    ScalarBlock::ElementType * faceXp = nullptr;
    ScalarBlock::ElementType * faceYm = nullptr;
    ScalarBlock::ElementType * faceYp = nullptr;
    if (tempCase != nullptr)
    {
      faceXm = tempCase -> storedFace[0] ?  & tempCase -> m_pData[0][0] : nullptr;
      faceXp = tempCase -> storedFace[1] ?  & tempCase -> m_pData[1][0] : nullptr;
      faceYm = tempCase -> storedFace[2] ?  & tempCase -> m_pData[2][0] : nullptr;
      faceYp = tempCase -> storedFace[3] ?  & tempCase -> m_pData[3][0] : nullptr;
    }
    if (faceXm != nullptr)
    {
      int ix = 0;
      for(int iy=0; iy<ScalarBlock::sizeY; ++iy)
        faceXm[iy] = lab(ix-1,iy) - lab(ix,iy);
    }
    if (faceXp != nullptr)
    {
      int ix = ScalarBlock::sizeX-1;
      for(int iy=0; iy<ScalarBlock::sizeY; ++iy)
        faceXp[iy] = lab(ix+1,iy) - lab(ix,iy);
    }
    if (faceYm != nullptr)
    {
      int iy = 0;
      for(int ix=0; ix<ScalarBlock::sizeX; ++ix)
        faceYm[ix] = lab(ix,iy-1) - lab(ix,iy);
    }
    if (faceYp != nullptr)
    {
      int iy = ScalarBlock::sizeY-1;
      for(int ix=0; ix<ScalarBlock::sizeX; ++ix)
        faceYp[ix] = lab(ix,iy+1) - lab(ix,iy);
    }
  }
};

void PressureSingle::preventCollidingObstacles() const
{
    const auto& shapes = sim.shapes;
    const auto & infos  = sim.chi->getBlocksInfo();
    const size_t N = shapes.size();
    sim.bCollisionID.clear();

    struct CollisionInfo // hitter and hittee, symmetry but we do things twice
    {
        Real iM = 0;
        Real iPosX = 0;
        Real iPosY = 0;
        Real iPosZ = 0;
        Real iMomX = 0;
        Real iMomY = 0;
        Real iMomZ = 0;
        Real ivecX = 0;
        Real ivecY = 0;
        Real ivecZ = 0;
        Real jM = 0;
        Real jPosX = 0;
        Real jPosY = 0;
        Real jPosZ = 0;
        Real jMomX = 0;
        Real jMomY = 0;
        Real jMomZ = 0;
        Real jvecX = 0;
        Real jvecY = 0;
        Real jvecZ = 0;
    };
    std::vector<CollisionInfo> collisions(N);

    std::vector <Real> n_vec(3*N,0.0);

    #pragma omp parallel for schedule(static)
    for (size_t i=0; i<N; ++i)
    for (size_t j=0; j<N; ++j)
    {
        if(i==j) continue;
        auto & coll = collisions[i];

        const auto& iBlocks = shapes[i]->obstacleBlocks;
        const Real iU0      = shapes[i]->u;
        const Real iU1      = shapes[i]->v;
        //const Real iU2      = 0; //set to 0 for 2D
        //const Real iomega0  = 0; //set to 0 for 2D
        //const Real iomega1  = 0; //set to 0 for 2D
        const Real iomega2  = shapes[i]->omega;
        const Real iCx      = shapes[i]->centerOfMass[0];
        const Real iCy      = shapes[i]->centerOfMass[1];
        //const Real iCz      = 0; //set to 0 for 2D

        const auto& jBlocks = shapes[j]->obstacleBlocks;
        const Real jU0      = shapes[j]->u;
        const Real jU1      = shapes[j]->v;
        //const Real jU2      = 0; //set to 0 for 2D
        //const Real jomega0  = 0; //set to 0 for 2D
        //const Real jomega1  = 0; //set to 0 for 2D
        const Real jomega2  = shapes[j]->omega;
        const Real jCx      = shapes[j]->centerOfMass[0];
        const Real jCy      = shapes[j]->centerOfMass[1];
        //const Real jCz      = 0; //set to 0 for 2D

        assert(iBlocks.size() == jBlocks.size());

        const size_t nBlocks = iBlocks.size();
        for (size_t k=0; k<nBlocks; ++k)
        {
            if ( iBlocks[k] == nullptr || jBlocks[k] == nullptr ) continue;

            const auto & iSDF  = iBlocks[k]->dist;
            const auto & jSDF  = jBlocks[k]->dist;

            const CHI_MAT & iChi  = iBlocks[k]->chi;
            const CHI_MAT & jChi  = jBlocks[k]->chi;

            const UDEFMAT & iUDEF = iBlocks[k]->udef;
            const UDEFMAT & jUDEF = jBlocks[k]->udef;

            for(int iy=0; iy<VectorBlock::sizeY; ++iy)
            for(int ix=0; ix<VectorBlock::sizeX; ++ix)
            {
                if(iChi[iy][ix] <= 0.0 || jChi[iy][ix] <= 0.0 ) continue;

                const auto pos = infos[k].pos<Real>(ix, iy);

                const Real iUr0 = - iomega2*(pos[1]-iCy);
                const Real iUr1 =   iomega2*(pos[0]-iCx);
                coll.iM    += iChi[iy][ix];
                coll.iPosX += iChi[iy][ix] * pos[0];
                coll.iPosY += iChi[iy][ix] * pos[1];
                coll.iMomX += iChi[iy][ix] * (iU0 + iUr0 + iUDEF[iy][ix][0]);
                coll.iMomY += iChi[iy][ix] * (iU1 + iUr1 + iUDEF[iy][ix][1]);

                const Real jUr0 = - jomega2*(pos[1]-jCy);
                const Real jUr1 =   jomega2*(pos[0]-jCx);
                coll.jM    += jChi[iy][ix];
                coll.jPosX += jChi[iy][ix] * pos[0];
                coll.jPosY += jChi[iy][ix] * pos[1];
                coll.jMomX += jChi[iy][ix] * (jU0 + jUr0 + jUDEF[iy][ix][0]);
                coll.jMomY += jChi[iy][ix] * (jU1 + jUr1 + jUDEF[iy][ix][1]);
              
                Real dSDFdx_i;
                Real dSDFdx_j;
                if (ix == 0)
                {
                  dSDFdx_i = iSDF[iy][ix+1] - iSDF[iy][ix];
                  dSDFdx_j = jSDF[iy][ix+1] - jSDF[iy][ix];
                }
                else if (ix == VectorBlock::sizeX - 1)
                {
                  dSDFdx_i = iSDF[iy][ix] - iSDF[iy][ix-1];
                  dSDFdx_j = jSDF[iy][ix] - jSDF[iy][ix-1];
                }
                else
                {
                  dSDFdx_i = 0.5*(iSDF[iy][ix+1] - iSDF[iy][ix-1]);
                  dSDFdx_j = 0.5*(jSDF[iy][ix+1] - jSDF[iy][ix-1]);
                }

                Real dSDFdy_i;
                Real dSDFdy_j;
                if (iy == 0)
                {
                  dSDFdy_i = iSDF[iy+1][ix] - iSDF[iy][ix];
                  dSDFdy_j = jSDF[iy+1][ix] - jSDF[iy][ix];
                }
                else if (iy == VectorBlock::sizeY - 1)
                {
                  dSDFdy_i = iSDF[iy][ix] - iSDF[iy-1][ix];
                  dSDFdy_j = jSDF[iy][ix] - jSDF[iy-1][ix];
                }
                else
                {
                  dSDFdy_i = 0.5*(iSDF[iy+1][ix] - iSDF[iy-1][ix]);
                  dSDFdy_j = 0.5*(jSDF[iy+1][ix] - jSDF[iy-1][ix]);
                }

                coll.ivecX += iChi[iy][ix] * dSDFdx_i;
                coll.ivecY += iChi[iy][ix] * dSDFdy_i;

                coll.jvecX += jChi[iy][ix] * dSDFdx_j;
                coll.jvecY += jChi[iy][ix] * dSDFdy_j;
            }
        }
    }

    std::vector<Real> buffer(20*N); //CollisionInfo holds 20 Reals
    for (size_t i = 0 ; i < N ; i++)
    {
        auto & coll = collisions[i];
        buffer[20*i     ] = coll.iM   ;
        buffer[20*i + 1 ] = coll.iPosX;
        buffer[20*i + 2 ] = coll.iPosY;
        buffer[20*i + 3 ] = coll.iPosZ;
        buffer[20*i + 4 ] = coll.iMomX;
        buffer[20*i + 5 ] = coll.iMomY;
        buffer[20*i + 6 ] = coll.iMomZ;
        buffer[20*i + 7 ] = coll.ivecX;
        buffer[20*i + 8 ] = coll.ivecY;
        buffer[20*i + 9 ] = coll.ivecZ;
        buffer[20*i + 10] = coll.jM   ;
        buffer[20*i + 11] = coll.jPosX;
        buffer[20*i + 12] = coll.jPosY;
        buffer[20*i + 13] = coll.jPosZ;
        buffer[20*i + 14] = coll.jMomX;
        buffer[20*i + 15] = coll.jMomY;
        buffer[20*i + 16] = coll.jMomZ;
        buffer[20*i + 17] = coll.jvecX;
        buffer[20*i + 18] = coll.jvecY;
        buffer[20*i + 19] = coll.jvecZ;

    }
    MPI_Allreduce(MPI_IN_PLACE, buffer.data(), buffer.size(), MPI_Real, MPI_SUM, sim.chi->getWorldComm());
    for (size_t i = 0 ; i < N ; i++)
    {
        auto & coll = collisions[i];
        coll.iM    = buffer[20*i     ];
        coll.iPosX = buffer[20*i + 1 ];
        coll.iPosY = buffer[20*i + 2 ];
        coll.iPosZ = buffer[20*i + 3 ];
        coll.iMomX = buffer[20*i + 4 ];
        coll.iMomY = buffer[20*i + 5 ];
        coll.iMomZ = buffer[20*i + 6 ];
        coll.ivecX = buffer[20*i + 7 ];
        coll.ivecY = buffer[20*i + 8 ];
        coll.ivecZ = buffer[20*i + 9 ];
        coll.jM    = buffer[20*i + 10];
        coll.jPosX = buffer[20*i + 11];
        coll.jPosY = buffer[20*i + 12];
        coll.jPosZ = buffer[20*i + 13];
        coll.jMomX = buffer[20*i + 14];
        coll.jMomY = buffer[20*i + 15];
        coll.jMomZ = buffer[20*i + 16];
        coll.jvecX = buffer[20*i + 17];
        coll.jvecY = buffer[20*i + 18];
        coll.jvecZ = buffer[20*i + 19];
    }

    #pragma omp parallel for schedule(static)
    for (size_t i=0; i<N; ++i)
    for (size_t j=i+1; j<N; ++j)
    {
        if (i==j) continue;
        const Real m1 = shapes[i]->M;
        const Real m2 = shapes[j]->M;
        const Real v1[3]={shapes[i]->u,shapes[i]->v,0.0};
        const Real v2[3]={shapes[j]->u,shapes[j]->v,0.0};
        const Real o1[3]={0,0,shapes[i]->omega};
        const Real o2[3]={0,0,shapes[j]->omega};
        const Real C1[3]={shapes[i]->centerOfMass[0],shapes[i]->centerOfMass[1],0};
        const Real C2[3]={shapes[j]->centerOfMass[0],shapes[j]->centerOfMass[1],0};
        const Real I1[6]={1.0,0,0,0,0,shapes[i]->J};
        const Real I2[6]={1.0,0,0,0,0,shapes[j]->J};

        auto & coll       = collisions[i];
        auto & coll_other = collisions[j];
        // less than one fluid element of overlap: wait to get closer. no hit
        if(coll.iM       < 2.0 || coll.jM       < 2.0) continue; //object i did not collide
        if(coll_other.iM < 2.0 || coll_other.jM < 2.0) continue; //object j did not collide

        if (std::fabs(coll.iPosX/coll.iM  - coll_other.iPosX/coll_other.iM ) > shapes[i]->getCharLength() ||
            std::fabs(coll.iPosY/coll.iM  - coll_other.iPosY/coll_other.iM ) > shapes[i]->getCharLength() )
        {
            continue; // then both objects i and j collided, but not with each other!
        }

        // A collision happened!
        sim.bCollision = true;
        #pragma omp critical
        {
            sim.bCollisionID.push_back(i);
            sim.bCollisionID.push_back(j);
        }

        const bool iForced = shapes[i]->bForced;
        const bool jForced = shapes[j]->bForced;
        if (iForced || jForced)
        {
            std::cout << "[CUP2D] WARNING: Forced objects not supported for collision." << std::endl;
            // MPI_Abort(sim.chi->getWorldComm(),1);
        }

        Real ho1[3];
        Real ho2[3];
        Real hv1[3];
        Real hv2[3];

        //1. Compute collision normal vector (NX,NY,NZ)
        const Real norm_i = std::sqrt(coll.ivecX*coll.ivecX + coll.ivecY*coll.ivecY + coll.ivecZ*coll.ivecZ);
        const Real norm_j = std::sqrt(coll.jvecX*coll.jvecX + coll.jvecY*coll.jvecY + coll.jvecZ*coll.jvecZ);
        const Real mX = coll.ivecX/norm_i - coll.jvecX/norm_j;
        const Real mY = coll.ivecY/norm_i - coll.jvecY/norm_j;
        const Real mZ = coll.ivecZ/norm_i - coll.jvecZ/norm_j;
        const Real inorm = 1.0/std::sqrt(mX*mX + mY*mY + mZ*mZ);
        const Real NX = mX * inorm;
        const Real NY = mY * inorm;
        const Real NZ = mZ * inorm;

        //If objects are already moving away from each other, don't do anything
        //if( (v2[0]-v1[0])*NX + (v2[1]-v1[1])*NY + (v2[2]-v1[2])*NZ <= 0 ) continue;
        const Real hitVelX = coll.jMomX / coll.jM - coll.iMomX / coll.iM;
        const Real hitVelY = coll.jMomY / coll.jM - coll.iMomY / coll.iM;
        const Real hitVelZ = coll.jMomZ / coll.jM - coll.iMomZ / coll.iM;
        const Real projVel = hitVelX * NX + hitVelY * NY + hitVelZ * NZ;

        /*const*/ Real vc1[3] = {coll.iMomX/coll.iM, coll.iMomY/coll.iM, coll.iMomZ/coll.iM};
        /*const*/ Real vc2[3] = {coll.jMomX/coll.jM, coll.jMomY/coll.jM, coll.jMomZ/coll.jM};


        if(projVel<=0) continue; // vel goes away from collision: no need to bounce

        //2. Compute collision location
        const Real inv_iM = 1.0/coll.iM;
        const Real inv_jM = 1.0/coll.jM;
        const Real iPX = coll.iPosX * inv_iM; // object i collision location
        const Real iPY = coll.iPosY * inv_iM;
        const Real iPZ = coll.iPosZ * inv_iM;
        const Real jPX = coll.jPosX * inv_jM; // object j collision location
        const Real jPY = coll.jPosY * inv_jM;
        const Real jPZ = coll.jPosZ * inv_jM;
        const Real CX = 0.5*(iPX+jPX);
        const Real CY = 0.5*(iPY+jPY);
        const Real CZ = 0.5*(iPZ+jPZ);

        //3. Take care of the collision. Assume elastic collision (kinetic energy is conserved)
        ElasticCollision(m1,m2,I1,I2,v1,v2,o1,o2,hv1,hv2,ho1,ho2,C1,C2,NX,NY,NZ,CX,CY,CZ,vc1,vc2);
        shapes[i]->u = hv1[0];
        shapes[i]->v = hv1[1];
        //shapes[i]->transVel[2] = hv1[2];
        shapes[j]->u = hv2[0];
        shapes[j]->v = hv2[1];
        //shapes[j]->transVel[2] = hv2[2];
        //shapes[i]->angVel[0] = ho1[0];
        //shapes[i]->angVel[1] = ho1[1];
        shapes[i]->omega = ho1[2];
        //shapes[j]->angVel[0] = ho2[0];
        //shapes[j]->angVel[1] = ho2[1];
        shapes[j]->omega = ho2[2];

        if ( sim.rank == 0)
        {
            #pragma omp critical
            {
                std::cout << "Collision between objects " << i << " and " << j << std::endl;
                std::cout << " iM   (0) = " << collisions[i].iM    << " jM   (1) = " << collisions[j].jM << std::endl;
                std::cout << " jM   (0) = " << collisions[i].jM    << " jM   (1) = " << collisions[j].iM << std::endl;
                std::cout << " Normal vector = (" << NX << "," << NY << "," << NZ << std::endl;
                std::cout << " Location      = (" << CX << "," << CY << "," << CZ << std::endl;
                std::cout << " Shape " << i << " before collision u    =(" <<  v1[0] << "," <<  v1[1] << "," <<  v1[2] << ")" << std::endl;
                std::cout << " Shape " << i << " after  collision u    =(" << hv1[0] << "," << hv1[1] << "," << hv1[2] << ")" << std::endl;
                std::cout << " Shape " << j << " before collision u    =(" <<  v2[0] << "," <<  v2[1] << "," <<  v2[2] << ")" << std::endl;
                std::cout << " Shape " << j << " after  collision u    =(" << hv2[0] << "," << hv2[1] << "," << hv2[2] << ")" << std::endl;
                std::cout << " Shape " << i << " before collision omega=(" <<  o1[0] << "," <<  o1[1] << "," <<  o1[2] << ")" << std::endl;
                std::cout << " Shape " << i << " after  collision omega=(" << ho1[0] << "," << ho1[1] << "," << ho1[2] << ")" << std::endl;
                std::cout << " Shape " << j << " before collision omega=(" <<  o2[0] << "," <<  o2[1] << "," <<  o2[2] << ")" << std::endl;
                std::cout << " Shape " << j << " after  collision omega=(" << ho2[0] << "," << ho2[1] << "," << ho2[2] << ")" << std::endl;
            }
        }
    }
}


void PressureSingle::operator()(const Real dt)
{
  sim.startProfiler("Pressure");
  const size_t Nblocks = velInfo.size();

  // update velocity of obstacle
  for(const auto& shape : sim.shapes) {
    integrateMomenta(shape.get());
    shape->updateVelocity(dt);
  }
  // take care if two obstacles collide
  preventCollidingObstacles();

  // apply penalization force
  penalize(dt);

  // compute pressure RHS
  // first we put uDef to tmpV so that we can create a VectorLab to compute div(uDef)
  const std::vector<cubism::BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& chiInfo = sim.chi->getBlocksInfo();
  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  {
    ( (VectorBlock*) tmpVInfo[i].ptrBlock )->clear();
  }
  for(const auto& shape : sim.shapes)
  {
    const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
    #pragma omp parallel for
    for (size_t i=0; i < Nblocks; i++)
    {
      if(OBLOCK[tmpVInfo[i].blockID] == nullptr) continue; //obst not in block
      const UDEFMAT & __restrict__ udef = OBLOCK[tmpVInfo[i].blockID]->udef;
      const CHI_MAT & __restrict__ chi  = OBLOCK[tmpVInfo[i].blockID]->chi;
      auto & __restrict__ UDEF = *(VectorBlock*)tmpVInfo[i].ptrBlock; // dest
      const ScalarBlock&__restrict__ CHI  = *(ScalarBlock*) chiInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
      {
         if( chi[iy][ix] < CHI(ix,iy).s) continue;
         Real p[2]; tmpVInfo[i].pos(p, ix, iy);
         UDEF(ix, iy).u[0] += udef[iy][ix][0];
         UDEF(ix, iy).u[1] += udef[iy][ix][1];
      }
    }
  }
  updatePressureRHS K(sim);
  compute<updatePressureRHS,VectorGrid,VectorLab,VectorGrid,VectorLab,ScalarGrid>(K,*sim.vel,*sim.tmpV,true,sim.tmp);


  //Add p_old (+dp/dt) to RHS
  const std::vector<cubism::BlockInfo>& presInfo = sim.pres->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& poldInfo = sim.pold->getBlocksInfo();
  
  //initial guess etc.
  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  {
    ScalarBlock & __restrict__   PRES = *(ScalarBlock*)  presInfo[i].ptrBlock;
    ScalarBlock & __restrict__   POLD = *(ScalarBlock*)  poldInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      POLD  (ix,iy).s = PRES (ix,iy).s;
      PRES  (ix,iy).s = 0;
    }
  }
  updatePressureRHS1 K1(sim);
  cubism::compute<ScalarLab>(K1,sim.pold,sim.tmp);

  pressureSolver->solve(sim.tmp, sim.pres);

  Real avg = 0;
  Real avg1 = 0;
  #pragma omp parallel for reduction (+:avg,avg1)
  for(size_t i=0; i< Nblocks; i++)
  {
    ScalarBlock& P  = *(ScalarBlock*) presInfo[i].ptrBlock;
    const Real vv = presInfo[i].h*presInfo[i].h;
    for(int iy=0; iy<VectorBlock::sizeY; iy++)
    for(int ix=0; ix<VectorBlock::sizeX; ix++)
    {
      avg += P(ix,iy).s * vv;
      avg1 += vv;
    }
  }
  Real quantities[2] = {avg,avg1};
  MPI_Allreduce(MPI_IN_PLACE,&quantities,2,MPI_Real,MPI_SUM,sim.comm);
  avg = quantities[0]; avg1 = quantities[1] ;
  avg = avg/avg1;
  #pragma omp parallel for
  for(size_t i=0; i< Nblocks; i++)
  {
    ScalarBlock& P  = *(ScalarBlock*) presInfo[i].ptrBlock;
    const ScalarBlock & __restrict__   POLD = *(ScalarBlock*)  poldInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; iy++)
    for(int ix=0; ix<VectorBlock::sizeX; ix++)
      P(ix,iy).s += POLD(ix,iy).s - avg;
  }

  // apply pressure correction
  pressureCorrection(dt);

  sim.stopProfiler();
}

PressureSingle::PressureSingle(SimulationData& s) :
  Operator{s},
  pressureSolver{makePoissonSolver(s)}
{ }

PressureSingle::~PressureSingle() = default;
struct SimulationData;
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
