//
//  common.h
//  CubismUP_2D
//
//  Created by Christian Conti on 1/8/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_common_h
#define CubismUP_2D_common_h

#include <cassert>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace std;
#define FREESPACE
#ifndef _FLOAT_PRECISION_
typedef double Real;
#else // _FLOAT_PRECISION_
typedef float Real;
#endif // _FLOAT_PRECISION_

//this is all cubism file we need
#include <ArgumentParser.h>
#include <Grid.h>
#include <BlockInfo.h>
#include <SerializerIO_ImageVTK.h>
//#include <ZBinDumper.h>
#include <BlockLab.h>
#include <Profiler.h>
#include "StencilInfo.h"

#endif
