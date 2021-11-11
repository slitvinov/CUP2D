//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once
#include "Fish.h"
#include "FishUtilities.h"

class ExperimentFish: public Fish
{
  const Real timeStart, dtDataset;

 public:
  ExperimentFish(SimulationData&s, cubism::ArgumentParser&p, double C[2]);

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  void updatePosition(double dt) override;
  void updateVelocity(double dt) override;
};


class ExperimentDataFish : public FishData
{
  // Conainers for experimentally measured midlines and center of mass
  std::vector<std::vector<Real>> midlineData;
  std::vector<std::vector<Real>> centerOfMassData;

  // Counter for time and index of current frames from experimental dataset
  Real tLast = 0.0, tNext = 0.0;
  size_t idxLast = 0, idxNext = 0;

  // Current velocities
  Real u = 0.0, v = 0.0, omega = 0.0;

  // Scheduler to interpolate midline between frames
  Schedulers::ParameterSchedulerVector<6> midlineScheduler;

public:
  
  ExperimentDataFish(Real L, std::string path, Real _h)
  : FishData(L, _h) { 
    loadCenterOfMass( path );
    loadCenterlines( path );
    _computeWidth(); 
  }

  void loadCenterOfMass( const std::string path );
  void loadCenterlines( const std::string path );
  void computeMidline(const Real time, const Real dt) override;

  Real _width(const Real s, const Real L) override
  {
    const Real sb=.04*length, st=.95*length, wt=.01*length, wh=.04*length;
    if(s<0 or s>L) return 0;
    return (s<sb ? std::sqrt(2*wh*s -s*s) :
           (s<st ? wh-(wh-wt)*std::pow((s-sb)/(st-sb),1) : // pow(.,2) is 3D
           (wt * (L-s)/(L-st))));
  }
};
