#pragma once

#include <memory>
#include "DefinitionsCup.h"

struct SimulationData;

class PoissonSolver
{
public:
  virtual ~PoissonSolver() = default;
  virtual void solve(const ScalarGrid *input, ScalarGrid *output) = 0;
};

std::shared_ptr<PoissonSolver> makePoissonSolver(SimulationData& s);
