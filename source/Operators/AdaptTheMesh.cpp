#include "AdaptTheMesh.h"

using namespace cubism;

void AdaptTheMesh::operator()(const double dt)
{
  count ++;

  if ((count-1) % 10 != 0 && count > 10) return;

  sim.startProfiler("AdaptTheMesh");

  // Write total divergence and number of blocks to file
  auto K = computeDivergence(sim); 
  K.run();

  findOmega.run();//store vorticity in tmp
  {
    //Refine according to chi and omega. Set omega=inf wherever chi > 0.
    const std::vector<BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
    const std::vector<BlockInfo>& chiInfo = sim.chi->getBlocksInfo();
    const size_t Nblocks = tmpInfo.size();

    #pragma omp parallel
    {
      static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
      ScalarLab chilab;
      chilab.prepare(*(sim.chi), stenBeg, stenEnd, 1);
      #pragma omp parallel for
      for (size_t i=0; i < Nblocks; i++)
      {
        chilab.load(chiInfo[i], 0);
        auto& __restrict__ TMP = *(ScalarBlock*)  tmpInfo[i].ptrBlock;
        const double i2h = 0.5/chiInfo[i].h;
        for(int y=0; y<VectorBlock::sizeY; ++y)
        for(int x=0; x<VectorBlock::sizeX; ++x)
        {
          const double dcdx = i2h*(chilab(x+1,y).s-chilab(x-1,y).s);
          const double dcdy = i2h*(chilab(x,y+1).s-chilab(x,y-1).s);
          const double norm = dcdx*dcdx+dcdy*dcdy;
          if (norm > 0.1) TMP(x,y).s = 1e10;
        }
      }
    }
  }

  double Rtol = sim.Rtol;
  double Ctol = sim.Ctol;

  bool verbose = sim.verbose;
  ScalarAMR tmp_amr ( *sim.tmp ,Rtol,Ctol,verbose);//refine according to tmp (vorticity magnitude)
  verbose = false;
  ScalarAMR chi_amr ( *sim.chi ,0.05,0.01,verbose);
  VectorAMR vel_amr ( *sim.vel ,Rtol,Ctol,verbose);
  ScalarAMR pres_amr( *sim.pres,Rtol,Ctol,verbose);
  //VectorAMR uDef_amr( *sim.uDef,Rtol,Ctol,verbose);
  //VectorAMR tmpV_amr( *sim.tmpV,Rtol,Ctol,verbose);

  //MeshAdaptation_basic<ScalarGrid,ScalarGrid> chi_amr(*sim.chi );
  MeshAdaptation_basic<VectorGrid,ScalarGrid>tmpV_amr(*sim.tmpV);
  MeshAdaptation_basic<VectorGrid,ScalarGrid>uDef_amr(*sim.uDef);
  MeshAdaptation_basic<  DumpGrid,ScalarGrid>dump_amr(*sim.dump);
  MeshAdaptation_basic<ScalarGrid,ScalarGrid>pOld_amr(*sim.pOld);

  tmp_amr .AdaptTheMesh();
  chi_amr .AdaptLikeOther(*sim.tmp);
  vel_amr .AdaptLikeOther(*sim.tmp);
  pres_amr.AdaptLikeOther(*sim.tmp);
  uDef_amr.AdaptLikeOther(*sim.tmp);
  dump_amr.AdaptLikeOther(*sim.tmp);
  tmpV_amr.AdaptLikeOther(*sim.tmp);
  pOld_amr.AdaptLikeOther(*sim.tmp);

  sim.chi ->SortBlocks();
  sim.vel ->SortBlocks();
  sim.pres->SortBlocks();
  sim.pOld->SortBlocks();
  sim.tmpV->SortBlocks();
  sim.tmp ->SortBlocks();
  sim.uDef->SortBlocks();
  sim.dump->SortBlocks();
  sim.stopProfiler();
}
