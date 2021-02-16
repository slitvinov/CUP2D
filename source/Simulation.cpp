//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "Simulation.h"

#include <Cubism/HDF5Dumper.h>
//#include <ZBinDumper.h>

#include "Operators/Helpers.h"
#include "Operators/PressureSingle.h"
#include "Operators/PressureVarRho_proper.h"
#include "Operators/PressureIterator_unif.h"
#include "Operators/PressureIterator_approx.h"
#include "Operators/PutObjectsOnGrid.h"
#include "Operators/PutObjectsOnGridStaggered.h"
#include "Operators/UpdateObjects.h"
#include "Operators/ComputeForces.h"
#include "Operators/UpdateObjectsStaggered.h"
#include "Operators/advDiff.h"
#include "Operators/AdaptTheMesh.h"

#include "Utils/FactoryFileLineParser.h"

#include "Obstacles/ShapesSimple.h"
#include "Obstacles/CarlingFish.h"
#include "Obstacles/StefanFish.h"
#include "Obstacles/CStartFish.h"
#include "Obstacles/ZebraFish.h"
#include "Obstacles/NeuroKinematicFish.h"
#include "Obstacles/BlowFish.h"
#include "Obstacles/SmartCylinder.h"
#include "Obstacles/Glider.h"
#include "Obstacles/Naca.h"

//#include <regex>
#include <algorithm>
#include <iterator>
using namespace cubism;

static inline std::vector<std::string> split(const std::string&s,const char dlm)
{
  std::stringstream ss(s); std::string item; std::vector<std::string> tokens;
  while (std::getline(ss, item, dlm)) tokens.push_back(item);
  return tokens;
}

Simulation::Simulation(int argc, char ** argv) : parser(argc,argv)
{
  std::cout
  <<"=======================================================================\n";
  std::cout
  <<"    CubismUP 2D (velocity-pressure 2D incompressible Navier-Stokes)    \n";
  std::cout
  <<"=======================================================================\n";
 parser.print_args();
}

Simulation::~Simulation()
{
  while( not pipeline.empty() ) {
    Operator * g = pipeline.back();
    pipeline.pop_back();
    if(g not_eq nullptr) delete g;
  }
}

void Simulation::init()
{
  // parse field variables
  std::cout << "[CUP2D] Parsing Simulation Configuration..." << std::endl;
  parseRuntime();
  // allocate the grid
  if(sim.verbose)
    std::cout << "[CUP2D] Allocating Grid..." << std::endl;
  sim.allocateGrid();
  // create shapes
  if(sim.verbose)
    std::cout << "[CUP2D] Creating Shapes..." << std::endl;
  createShapes();
  // impose field initial condition
  if(sim.verbose)
    std::cout << "[CUP2D] Imposing Initial Conditions..." << std::endl;
  IC ic(sim);
  ic(0);
  // create compute pipeline
  if(sim.verbose)
    std::cout << "[CUP2D] Creating Computational Pipeline..." << std::endl;
  pipeline.push_back( new AdaptTheMesh(sim) );

  if(sim.bVariableDensity)
  {
    std::cout << "Variable density not implemented for AMR. " << std::endl;
    abort();
  }
  else
  {
    sim.bStaggeredGrid = false;
    pipeline.push_back( new PutObjectsOnGrid(sim) );
    pipeline.push_back( new advDiff(sim) );
    //pipeline.push_back( new FadeOut(sim) );
    //pipeline.push_back( new PressureVarRho(sim) );
    //pipeline.push_back( new PressureVarRho_proper(sim) );

    if(sim.iterativePenalization)
    {
      std::cout << "Iterative Penalization with AMR not ready." << std::endl;
      abort();
      pipeline.push_back( new PressureIterator_unif(sim) );
    }
    else {
      pipeline.push_back( new PressureSingle(sim) );
      //pipeline.push_back( new UpdateObjects(sim) );
    }
    //pipeline.push_back( new FadeOut(sim) );
  }
  pipeline.push_back( new ComputeForces(sim) );

  std::cout << "[CUP2D] Operator ordering:\n";
  for (size_t c=0; c<pipeline.size(); c++)
    std::cout << "[CUP2D] - " << pipeline[c]->getName() << "\n";

  // Initial PutObjectToGrid
  std::cout << "[CUP2D] Initial PutObjectsOnGrid\n";
  (*pipeline[1])(0);
  // initial compression of the grid
  std::cout << "[CUP2D] Initial Compression of Grid\n";
  for( int i = 0; i<sim.levelMax; i++ )
    (*pipeline[0])(0);
  // PutObjectToGrid for compressed grid
  std::cout << "[CUP2D] Compressed PutObjectsOnGrid\n";
  (*pipeline[1])(0);
  // impose velocity of obstacles
  std::cout << "[CUP2D] Imposing Initial Velocity of Objects on field\n";
  ApplyObjVel initVel(sim);
  initVel(0);
}

void Simulation::parseRuntime()
{
  // restart the simulation?
  sim.bRestart = parser("-restart").asBool(false);

  parser.set_strict_mode();

  // set initial number of blocks
  sim.bpdx = parser("-bpdx").asInt();
  sim.bpdy = parser("-bpdy").asInt();

  // set number of refinement levels
  sim.levelMax = parser("-levelMax").asInt();

  // set tolerance for refinement/compression according to vorticity magnitude
  sim.Rtol = parser("-Rtol").asDouble();
  sim.Ctol = parser("-Ctol").asDouble();

  parser.unset_strict_mode();

  // set simulation extent
  sim.extent = parser("-extent").asDouble(1);

  // simulation ending parameters
  sim.nsteps = parser("-nsteps").asInt(0);
  sim.endTime = parser("-tend").asDouble(0);

  // simulation settings
  sim.CFL = parser("-CFL").asDouble(0.1);
  sim.lambda = parser("-lambda").asDouble(1e3 / sim.CFL);
  sim.dlm = parser("-dlm").asDouble(0);
  sim.nu = parser("-nu").asDouble(1e-2);
  sim.fadeLenX = parser("-fadeLen").asDouble(0.01) * sim.extent;
  sim.fadeLenY = parser("-fadeLen").asDouble(0.01) * sim.extent;

  // output parameters
  sim.dumpFreq = parser("-fdump").asInt(0);
  sim.dumpTime = parser("-tdump").asDouble(0);
  sim.path2file = parser("-file").asString("./");
  sim.path4serialization = parser("-serialization").asString(sim.path2file);

  // select Poisson solver
  sim.poissonType = parser("-poissonType").asString("");

  // boolean to enable iterative penalisation
  sim.iterativePenalization = parser("-iterativePenalization").asInt(0);

  // set output vebosity
  sim.verbose = parser("-verbose").asInt(1);
  sim.muteAll = parser("-muteAll").asInt(0);
  if(sim.muteAll) sim.verbose = 0;
}

void Simulation::createShapes()
{
  const std::string shapeArg = parser("-shapes").asString("");
  std::stringstream descriptors( shapeArg );
  std::string lines;
  unsigned k = 0;

  while (std::getline(descriptors, lines))
  {
    std::replace(lines.begin(), lines.end(), '_', ' ');
    const std::vector<std::string> vlines = split(lines, ',');

    for (const auto& line: vlines)
    {
      std::istringstream line_stream(line);
      std::string objectName;
      if( sim.verbose )
        std::cout << "[CUP2D] " << line << std::endl;
      line_stream >> objectName;
      // Comments and empty lines ignored:
      if(objectName.empty() or objectName[0]=='#') continue;
      FactoryFileLineParser ffparser(line_stream);
      double center[2] = {
        ffparser("-xpos").asDouble(.5*sim.extents[0]),
        ffparser("-ypos").asDouble(.5*sim.extents[1])
      };
      //ffparser.print_args();
      Shape* shape = nullptr;
      if (objectName=="disk")
        shape = new Disk(             sim, ffparser, center);
      else if (objectName=="smartDisk")
        shape = new SmartCylinder(    sim, ffparser, center);
      else if (objectName=="halfDisk")
        shape = new HalfDisk(         sim, ffparser, center);
      else if (objectName=="ellipse")
        shape = new Ellipse(          sim, ffparser, center);
      else if (objectName=="diskVarDensity")
        shape = new DiskVarDensity(   sim, ffparser, center);
      else if (objectName=="ellipseVarDensity")
        shape = new EllipseVarDensity(sim, ffparser, center);
      else if (objectName=="blowfish")
        shape = new BlowFish(         sim, ffparser, center);
      else if (objectName=="glider")
        shape = new Glider(           sim, ffparser, center);
      else if (objectName=="stefanfish")
        shape = new StefanFish(       sim, ffparser, center);
      else if (objectName=="cstartfish")
        shape = new CStartFish(       sim, ffparser, center);
      else if (objectName=="zebrafish")
          shape = new ZebraFish(      sim, ffparser, center);
      else if (objectName=="neurokinematicfish")
          shape = new NeuroKinematicFish(      sim, ffparser, center);
      else if (objectName=="carlingfish")
        shape = new CarlingFish(      sim, ffparser, center);
      else if ( objectName=="NACA" )
        shape = new Naca(             sim, ffparser, center);
      else {
        std::cout << "FATAL - shape is not recognized!" << std::endl; 
        fflush(0);
        abort();
      }
      assert(shape not_eq nullptr);
      shape->obstacleID = k++;
      sim.shapes.push_back(shape);
    }
  }

  if( sim.shapes.size() ==  0) {
    std::cout << "FATAL - Did not create any obstacles." << std::endl;
    fflush(0);
    abort();
  }

  // check if we have variable rho object:
  sim.checkVariableDensity();
}

void Simulation::reset()
{
  // reset field variables and shapes
  if(sim.verbose)
    std::cout << "[CUP2D] Resetting Field Variables and Shapes..." << std::endl;
  sim.resetAll();
  // impose field initial condition
  if(sim.verbose)
    std::cout << "[CUP2D] Imposing Initial Conditions..." << std::endl;
  IC ic(sim);
  ic(0);
  // Initial PutObjectToGrid
  std::cout << "[CUP2D] Initial PutObjectsOnGrid\n";
  (*pipeline[1])(0);
  // initial compression of the grid
  std::cout << "[CUP2D] Initial Compression of Grid\n";
  for( int i = 0; i<sim.levelMax; i++ )
    (*pipeline[0])(0);
  // PutObjectToGrid for compressed grid
  std::cout << "[CUP2D] Compressed PutObjectsOnGrid\n";
  (*pipeline[1])(0);
  // impose velocity of obstacles
  std::cout << "[CUP2D] Imposing Initial Velocity of Objects on field\n";
  ApplyObjVel initVel(sim);
  initVel(0);
}

void Simulation::simulate()
{
  if(sim.verbose){
    std::cout
  <<"=======================================================================\n";
    std::cout << "[CUP2D] Starting Simulation..." << std::endl;
  }
  while (1)
  {
    // sim.startProfiler("DT");
    const double dt = calcMaxTimestep();
    // sim.stopProfiler();
    if (advance(dt)) break;
  }
}

double Simulation::calcMaxTimestep()
{
  const auto findMaxU_op = findMaxU(sim);
  sim.uMax_measured = findMaxU_op.run();
  assert(sim.uMax_measured>=0);

  const double h = sim.getH();
  #if 0 // CFL condition for centered scheme
  const double dtFourier = h*h/sim.nu;
  const double dtCFL = sim.uMax_measured<2.2e-16? 1 : h/sim.uMax_measured;
  const double maxUb = sim.maxRelSpeed(), dtBody = maxUb<2.2e-16? 1 : h/maxUb;
  #else // CFL for QUICK scheme
  // stability condition sigma^2<=2d
  const double dtBalance = sim.uMax_measured < 2.2e-16 ? 1 : 2*sim.nu / (sim.uMax_measured*sim.uMax_measured);

  // stability condition sigma+4d<=2
  const double coeffAdvection = sim.uMax_measured / h;
  const double coeffDiffusion = 4*sim.nu / (h*h);
  const double dtAbs     = 2*sim.CFL/(coeffAdvection+coeffDiffusion);
  #endif

  // ramp up CFL
  if (sim.step < 100)
  {
    const double x = (sim.step+1.0)/100;
    const double rampCFL = std::exp(std::log(1e-3)*(1-x) + std::log(sim.CFL)*x);
    #if 0
    sim.dt = rampCFL * std::min({dtCFL, dtFourier, dtBody});
    #else
    sim.dt = rampCFL * std::min({ dtBalance, dtAbs });
    #endif
  }
  else
  {
    #if 0
    sim.dt = sim.CFL * std::min({dtCFL, dtFourier, dtBody});
    #else
    sim.dt = sim.CFL*std::min({ dtBalance, dtAbs });
    #endif
  }

  std::cout
  <<"=======================================================================\n";
    printf("[CUP2D] step:%d, time:%f, dt=%f, uinf:[%f %f], maxU:%f\n",
      sim.step, sim.time, sim.dt, sim.uinfx, sim.uinfy, sim.uMax_measured); 

  if(sim.dlm > 0) sim.lambda = sim.dlm / sim.dt;
  return sim.dt;
}

bool Simulation::advance(const double dt)
{
  assert(dt>2.2e-16);
  if( sim.step == 0 ){
    if(sim.verbose)
      std::cout << "[CUP2D] dumping IC...\n";
    sim.dumpAll("IC");
  }
  const bool bDump = sim.bDump();

  for (size_t c=0; c<pipeline.size(); c++) {
    if(sim.verbose)
      std::cout << "[CUP2D] running " << pipeline[c]->getName() << "...\n";
    (*pipeline[c])(sim.dt);
    //sim.dumpAll( pipeline[c]->getName() );
  }

  // For debuging state function
  // Shape *object = getShapes()[0];
  // StefanFish *agent = dynamic_cast<StefanFish *>(getShapes()[1]);
  // auto state = agent->state(object);
  // std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
  // std::cout << "[CUP2D] Computed state:" << std::endl;
  // for( auto s : state )
  //   std::cout << s << std::endl;
  // std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;


  sim.time += sim.dt;
  sim.step++;

  // dump field
  if( bDump ) {
    if(sim.verbose)
      std::cout << "[CUP2D] dumping field...\n";
    sim.registerDump();
    sim.dumpAll("avemaria_"); 
  }

  const bool bOver = sim.bOver();

  if (bOver){
    std::cout
  <<"=======================================================================\n";
    std::cout << "[CUP2D] Simulation Over... Profiling information:\n";
    sim.printResetProfiler();
    std::cout
  <<"=======================================================================\n";
  }

  return bOver;
}
