//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "advDiffGrav.h"

using namespace cubism;

void advDiffGrav::operator()(const double dt)
{
  sim.startProfiler("advDiffGrav");
  static constexpr int BSX = VectorBlock::sizeX, BSY = VectorBlock::sizeY;
  static constexpr int BX=0, EX=BSX-1, BY=0, EY=BSY-1;
  const auto isW = [&](const BlockInfo& info) {
    return info.index[0] == 0;
  };
  const auto isE = [&](const BlockInfo& info) {
    return info.index[0] == sim.bpdx-1;
  };
  const auto isS = [&](const BlockInfo& info) {
    return info.index[1] == 0;
  };
  const auto isN = [&](const BlockInfo& info) {
    return info.index[1] == sim.bpdy-1;
  };

  const Real UINF[2]= {sim.uinfx, sim.uinfy}, h = sim.getH();
  const std::array<Real,2>& G = sim.gravity;
  //const Real G[]= {sim.gravity[0],sim.gravity[1]};
  const Real dfac = (sim.nu/h)*(dt/h), afac = -0.5*dt/h;

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    VectorLab vellab; vellab.prepare(*(sim.vel), stenBeg, stenEnd, 1);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      vellab.load(velInfo[i], 0); VectorLab & __restrict__ V = vellab;
      VectorBlock & __restrict__ TMP = *(VectorBlock*) tmpVInfo[i].ptrBlock;
      const ScalarBlock& __restrict__ invRho=*(ScalarBlock*)rhoInfo[i].ptrBlock;

      if ( isW(velInfo[i]) ) // west
        for(int iy=0; iy<=VectorBlock::sizeY; ++iy) {
          const Real uAdvV = std::max(V(BX,iy).u[0], V(BX,iy-1).u[0]);
          V(BX-1, iy).u[0] = V(BX,iy).u[0]+UINF[0]>0? 0 : V(BX-1, iy).u[0];
          V(BX-1, iy).u[1] = uAdvV+UINF[0]>0? 0 : V(BX-1, iy).u[1];
        }
      if( isS(velInfo[i]) ) // south
        for(int ix=0; ix<=VectorBlock::sizeX; ++ix) {
          const Real vAdvU = std::max(V(ix,BY).u[1], V(ix-1,BY).u[1]);
          V(ix, BY-1).u[1] = V(ix,BY).u[1]+UINF[1]>0? 0 : V(ix, BY-1).u[1];
          V(ix, BY-1).u[0] = vAdvU+UINF[1]>0? 0 : V(ix, BY-1).u[0];
        }

      if ( isE(velInfo[i]) ) // east
        for(int iy=-1; iy<VectorBlock::sizeY; ++iy)
          V(EX+1, iy).u[0] = V(EX,iy).u[0]+UINF[0]<0? 0 : V(EX+1, iy).u[0];
      if ( isN(velInfo[i]) ) // north
        for(int ix=-1; ix<VectorBlock::sizeX; ++ix)
          V(ix, EY+1).u[1] = V(ix,EY).u[1]+UINF[1]<0? 0 : V(ix, EY+1).u[1];

      if( isE(velInfo[i]) ) // east
      for(int iy=0; iy<VectorBlock::sizeY; ++iy) {
        const Real uAdvV = std::min(V(EX+1, iy).u[0], V(EX+1, iy-1).u[0]);
        V(EX+1, iy).u[1] = uAdvV+UINF[0]<0? 0 : V(EX+1, iy).u[1];
      }

      if( isN(velInfo[i]) ) // north
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
        const Real vAdvU = std::min(V(ix, EY+1).u[1], V(ix-1, EY+1).u[1]);
        V(ix, EY+1).u[0] = vAdvU+UINF[1]<0? 0 : V(ix, EY+1).u[0];
      }

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        const Real gravFac = dt * (1 - invRho(ix,iy).s);
        const Real upx = V(ix+1, iy).u[0], upy = V(ix, iy+1).u[0];
        const Real ulx = V(ix-1, iy).u[0], uly = V(ix, iy-1).u[0];
        const Real vpx = V(ix+1, iy).u[1], vpy = V(ix, iy+1).u[1];
        const Real vlx = V(ix-1, iy).u[1], vly = V(ix, iy-1).u[1];
        const Real ucc = V(ix  , iy).u[0], vcc = V(ix, iy  ).u[1];
        const Real dudx = upx-ulx, dudy = upy-uly;
        const Real dvdx = vpx-vlx, dvdy = vpy-vly;
        const Real dUdif = upx + upy + ulx + uly - 4 * ucc;
        const Real dVdif = vpx + vpy + vlx + vly - 4 * vcc;
        {
          const Real u =  ucc + UINF[0];
          const Real v = (vcc + vlx + vpy + V(ix-1,iy+1).u[1])/4 + UINF[1];
          const Real dUAdvDiff = afac*(u*dudx + v*dudy) + dfac*dUdif;
          TMP(ix,iy).u[0] = V(ix,iy).u[0] + dUAdvDiff + G[0]*gravFac;
        }
        {
          const Real u = (ucc + uly + upx + V(ix+1,iy-1).u[0])/4 + UINF[0];
          const Real v =  vcc + UINF[1];
          const Real dVAdvDiff = afac*(u*dvdx + v*dvdy) + dfac*dVdif;
          TMP(ix,iy).u[1] = V(ix,iy).u[1] + dVAdvDiff + G[1]*gravFac;
        }
      }
    }
  }

  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++)
  {
          VectorBlock & __restrict__ V  = *(VectorBlock*)  velInfo[i].ptrBlock;
    const VectorBlock & __restrict__ T  = *(VectorBlock*) tmpVInfo[i].ptrBlock;
    V.copy(T);
  }
  sim.stopProfiler();
}
