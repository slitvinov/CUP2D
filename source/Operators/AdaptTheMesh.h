#pragma once

#include "../Operator.h"
#include "Cubism/AMR_MeshAdaptation.h"
#include "Helpers.h"

class AdaptTheMesh : public Operator
{
 public:
  int count;
  computeVorticity findOmega;

  AdaptTheMesh(SimulationData& s) : Operator(s), findOmega(s) {count=0;}

  void operator()(const double dt);

  std::string getName()
  {
    return "AdaptTheMesh";
  }
};