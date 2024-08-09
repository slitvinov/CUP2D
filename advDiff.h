#pragma once

#include "Operator.h"
#include "FluxCorrection.h"

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
