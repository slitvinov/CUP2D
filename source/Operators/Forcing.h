//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "../Operator.h"

class Forcing : public Operator
{
 public:
  Forcing(SimulationData& s) : Operator(s) { }

  void operator()(const Real dt);

  std::string getName()
  {
    return "Forcing";
  }
};