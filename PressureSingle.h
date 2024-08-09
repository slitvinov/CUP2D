#pragma once

#include "Operator.h"

class Shape;

#include "Base.h"

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
