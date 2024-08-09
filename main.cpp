#include "cuda.h"
#include <algorithm>
#include <array>
#include <assert.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gsl/gsl_linalg.h>
#include <hdf5.h>
#include <iomanip>
#include <ios>
#include <iosfwd>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <math.h>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <omp.h>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

#define OMPI_SKIP_MPICXX 1
#ifdef _FLOAT_PRECISION_
using Real = float;
#define MPI_Real MPI_FLOAT
#endif
#ifdef _DOUBLE_PRECISION_
using Real = double;
#define MPI_Real MPI_DOUBLE
#endif
#ifdef _LONG_DOUBLE_PRECISION_
using Real = long double;
#define MPI_Real MPI_LONG_DOUBLE
#endif

namespace cubism {

class Value {
private:
  std::string content;

public:
  Value() = default;
  Value(const std::string &content_) : content(content_) {}
  Value(const Value &c) = default;

  Value &operator=(const Value &rhs) {
    if (this != &rhs)
      content = rhs.content;
    return *this;
  }
  Value &operator+=(const Value &rhs) {
    content += " " + rhs.content;
    return *this;
  }
  Value operator+(const Value &rhs) {
    return Value(content + " " + rhs.content);
  }

  double asDouble(double def = 0);
  int asInt(int def = 0);
  bool asBool(bool def = false);
  std::string asString(const std::string &def = std::string());
  friend std::ostream &operator<<(std::ostream &lhs, const Value &rhs);
};

class CommandlineParser {
private:
  const int iArgC;
  char **vArgV;
  bool bStrictMode, bVerbose;

  bool _isnumber(const std::string &s) const;

protected:
  std::map<std::string, Value> mapArguments;

public:
  CommandlineParser(int argc, char **argv);

  Value &operator()(std::string key);
  bool check(std::string key) const;

  int getargc() const { return iArgC; }
  char **getargv() const { return vArgV; }

  void set_strict_mode() { bStrictMode = true; }

  void unset_strict_mode() { bStrictMode = false; }

  void mute() { bVerbose = false; }

  void loud() { bVerbose = true; }

  void save_options(const std::string &path = ".");
  void print_args();
};

class ArgumentParser : public CommandlineParser {
  typedef std::map<std::string, Value> ArgMap;
  typedef std::map<std::string, Value *> pArgMap;
  typedef std::map<std::string, ArgMap *> FileMap;

  const char commentStart;

  // keep a reference from option origin
  ArgMap from_commandline;
  FileMap from_files;
  pArgMap from_code;

  // for runtime interaction (we keep the original map)
  ArgMap mapRuntime;

  // helper
  void _ignoreComments(std::istream &stream, char commentChar);
  void _parseFile(std::ifstream &stream, ArgMap &container);

public:
  ArgumentParser(const int _argc, char **_argv, const char cstart = '#')
      : CommandlineParser(_argc, _argv), commentStart(cstart) {
    from_commandline = mapArguments;
  }

  virtual ~ArgumentParser() {
    for (FileMap::iterator it = from_files.begin(); it != from_files.end();
         it++)
      delete it->second;
  }

  void readFile(const std::string &filepath);
  Value &operator()(std::string key);

  inline bool exist(const std::string &key) const { return check(key); }

  void write_runtime_environment() const;
  void read_runtime_environment();

  Value &parseRuntime(std::string key);
  void print_args(void);
};

} // namespace cubism

namespace cubism {

///////////////////////////////////////////////////////////
// Value
///////////////////////////////////////////////////////////
double Value::asDouble(double def) {
  if (content == "") {
    std::ostringstream sbuf;
    sbuf << def;
    content = sbuf.str();
  }
  return (double)atof(content.c_str());
}

int Value::asInt(int def) {
  if (content == "") {
    std::ostringstream sbuf;
    sbuf << def;
    content = sbuf.str();
  }
  return atoi(content.c_str());
}

bool Value::asBool(bool def) {
  if (content == "") {
    if (def)
      content = "true";
    else
      content = "false";
  }
  if (content == "0")
    return false;
  if (content == "false")
    return false;

  return true;
}

std::string Value::asString(const std::string &def) {
  if (content == "")
    content = def;

  return content;
}

std::ostream &operator<<(std::ostream &lhs, const Value &rhs) {
  lhs << rhs.content;
  return lhs;
}

///////////////////////////////////////////////////////////
// CommandlineParser
///////////////////////////////////////////////////////////
static inline void _normalizeKey(std::string &key) {
  if (key[0] == '-')
    key.erase(0, 1);
  if (key[0] == '+')
    key.erase(0, 1);
}

static inline bool _existKey(const std::string &key,
                             const std::map<std::string, Value> &container) {
  return container.find(key) != container.end();
}

Value &CommandlineParser::operator()(std::string key) {
  _normalizeKey(key);
  if (bStrictMode) {
    if (!_existKey(key, mapArguments)) {
      printf("Runtime option NOT SPECIFIED! ABORTING! name: %s\n", key.data());
      abort();
    }
  }

  if (bVerbose)
    printf("%s is %s\n", key.data(), mapArguments[key].asString().data());
  return mapArguments[key];
}

bool CommandlineParser::check(std::string key) const {
  _normalizeKey(key);
  return _existKey(key, mapArguments);
}

bool CommandlineParser::_isnumber(const std::string &s) const {
  char *end = NULL;
  strtod(s.c_str(), &end);
  return end != s.c_str(); // only care if the number is numeric or not.  This
                           // includes nan and inf
}

CommandlineParser::CommandlineParser(const int argc, char **argv)
    : iArgC(argc), vArgV(argv), bStrictMode(false), bVerbose(true) {
  // parse commandline <key> <value> pairs.  Key passed on the command
  // line must start with a leading dash (-). For example:
  // -mykey myvalue0 [myvalue1 ...]
  for (int i = 1; i < argc; i++)
    if (argv[i][0] == '-') {
      std::string values = "";
      int itemCount = 0;

      // check if the current key i is a list of values. If yes,
      // concatenate them into a string
      for (int j = i + 1; j < argc; j++) {
        // if the current value is numeric and (possibly) negative,
        // do not interpret it as a key.
        // XXX: [fabianw@mavt.ethz.ch; 2019-03-28] WARNING:
        // This will treat -nan as a NUMBER and not as a KEY
        std::string sval(argv[j]);
        const bool leadingDash = (sval[0] == '-');
        const bool isNumeric = _isnumber(sval);
        if (leadingDash && !isNumeric)
          break;
        else {
          if (std::strcmp(values.c_str(), ""))
            values += ' ';

          values += argv[j];
          itemCount++;
        }
      }

      if (itemCount == 0)
        values = "true";

      std::string key(argv[i]);
      key.erase(0, 1);   // remove leading '-'
      if (key[0] == '+') // for key concatenation
      {
        key.erase(0, 1);
        if (!_existKey(key, mapArguments))
          mapArguments[key] = Value(values); // skip leading white space
        else
          mapArguments[key] += Value(values);
      } else // regular key
      {
        if (!_existKey(key, mapArguments))
          mapArguments[key] = Value(values);
      }

      i += itemCount;
    }

  mute();
  // printf("found %ld arguments of %d\n",mapArguments.size(),argc);
}

void CommandlineParser::save_options(const std::string &path) {
  std::string options;
  for (std::map<std::string, Value>::iterator it = mapArguments.begin();
       it != mapArguments.end(); it++) {
    options += it->first + " " + it->second.asString() + " ";
  }
  std::string filepath = path + "/argumentparser.log";
  FILE *f = fopen(filepath.data(), "a");
  if (f == NULL) {
    fprintf(stderr, "impossible to write %s.\n", filepath.data());
    return;
  }
  fprintf(f, "%s\n", options.data());
  fclose(f);
}

void CommandlineParser::print_args() {
  for (std::map<std::string, Value>::iterator it = mapArguments.begin();
       it != mapArguments.end(); it++) {
    std::cout.width(50);
    std::cout.fill('.');
    std::cout << std::left << it->first;
    std::cout << ": " << it->second.asString() << std::endl;
  }
}

///////////////////////////////////////////////////////////
// ArgumentParser
///////////////////////////////////////////////////////////
void ArgumentParser::_ignoreComments(std::istream &stream,
                                     const char commentChar) {
  stream >> std::ws;
  int nextchar = stream.peek();
  while (nextchar == commentChar) {
    stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    stream >> std::ws;
    nextchar = stream.peek();
  }
}

void ArgumentParser::_parseFile(std::ifstream &stream, ArgMap &container) {
  // read (key value) pairs from input file, ignore comments
  // beginning with commentStart
  _ignoreComments(stream, commentStart);
  while (!stream.eof()) {
    std::string line, key, val;
    std::getline(stream, line);
    std::istringstream lineStream(line);
    lineStream >> key;
    lineStream >> val;
    _ignoreComments(lineStream, commentStart);
    while (!lineStream.eof()) {
      std::string multiVal;
      lineStream >> multiVal;
      val += (" " + multiVal);
      _ignoreComments(lineStream, commentStart);
    }

    const Value V(val);
    if (key[0] == '-')
      key.erase(0, 1);

    if (key[0] == '+') {
      key.erase(0, 1);
      if (!_existKey(key, container)) // skip leading white space
        container[key] = V;
      else
        container[key] += V;
    } else if (!_existKey(key, container))
      container[key] = V;
    _ignoreComments(stream, commentStart);
  }
}

void ArgumentParser::readFile(const std::string &filepath) {
  from_files[filepath] = new ArgMap;
  ArgMap &myFMap = *(from_files[filepath]);

  std::ifstream confFile(filepath.c_str());
  if (confFile.good()) {
    _parseFile(confFile, mapArguments);
    confFile.clear();
    confFile.seekg(0, std::ios::beg);
    _parseFile(confFile,
               myFMap); // we keep a reference for each separate file read
  }
  confFile.close();
}

Value &ArgumentParser::operator()(std::string key) {
  _normalizeKey(key);
  const bool bDefaultInCode = !_existKey(key, mapArguments);
  Value &retval = CommandlineParser::operator()(key);
  if (bDefaultInCode)
    from_code[key] = &retval;
  return retval;
}

void ArgumentParser::write_runtime_environment() const {
  time_t rawtime;
  std::time(&rawtime);
  struct tm *timeinfo = std::localtime(&rawtime);
  char buf[256];
  std::strftime(buf, 256, "%A, %h %d %Y, %r", timeinfo);

  std::ofstream runtime("runtime_environment.conf");
  runtime << commentStart << " RUNTIME ENVIRONMENT SETTINGS" << std::endl;
  runtime << commentStart << " ============================" << std::endl;
  runtime << commentStart << " " << buf << std::endl;
  runtime << commentStart
          << " Use this file to set runtime parameter interactively."
          << std::endl;
  runtime << commentStart
          << " The parameter are read every \"refreshperiod\" steps."
          << std::endl;
  runtime << commentStart
          << " When editing this file, you may use comments and string "
             "concatenation."
          << std::endl;
  runtime << commentStart
          << " The simulation can be terminated without killing it by setting "
             "\"exit\" to true."
          << std::endl;
  runtime << commentStart
          << " (This will write a serialized restart state. Set \"exitsave\" "
             "to false if not desired.)"
          << std::endl;
  runtime << commentStart << std::endl;
  runtime << commentStart
          << " !!! WARNING !!! EDITING THIS FILE CAN POTENTIALLY CRASH YOUR "
             "SIMULATION !!! WARNING !!!"
          << std::endl;
  for (typename std::map<std::string, Value>::const_iterator it =
           mapArguments.begin();
       it != mapArguments.end(); ++it)
    runtime << it->first << '\t' << it->second << std::endl;
}

void ArgumentParser::read_runtime_environment() {
  mapRuntime.clear();
  std::ifstream runtime("runtime_environment.conf");
  if (runtime.good())
    _parseFile(runtime, mapRuntime);
  runtime.close();
}

Value &ArgumentParser::parseRuntime(std::string key) {
  _normalizeKey(key);
  if (!_existKey(key, mapRuntime)) {
    printf("ERROR: Runtime parsing for key %s NOT FOUND!! Check your "
           "runtime_environment.conf file\n",
           key.data());
    abort();
  }
  return mapRuntime[key];
}

void ArgumentParser::print_args() {
  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
               "~~~~~~~"
            << std::endl;
  std::cout << "* Summary:" << std::endl;
  std::cout << "*    Parameter read from command line:                "
            << from_commandline.size() << std::endl;
  size_t nFiles = 0;
  size_t nFileParameter = 0;
  for (FileMap::const_iterator it = from_files.begin(); it != from_files.end();
       ++it) {
    if (it->second->size() > 0) {
      ++nFiles;
      nFileParameter += it->second->size();
    }
  }
  std::cout << "*    Parameter read from " << std::setw(3) << std::right
            << nFiles << " file(s):                 " << nFileParameter
            << std::endl;
  std::cout << "*    Parameter read from defaults in code:            "
            << from_code.size() << std::endl;
  std::cout << "*    Total number of parameter read from all sources: "
            << mapArguments.size() << std::endl;
  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
               "~~~~~~~"
            << std::endl;

  // command line given arguments
  if (!from_commandline.empty()) {
    std::cout << "* Command Line:" << std::endl;
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                 "~~~~~~~~~"
              << std::endl;
    for (ArgMap::iterator it = from_commandline.begin();
         it != from_commandline.end(); it++) {
      std::cout.width(50);
      std::cout.fill('.');
      std::cout << std::left << it->first;
      std::cout << ": " << it->second.asString() << std::endl;
    }
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                 "~~~~~~~~~"
              << std::endl;
  }

  // options read from input files
  if (!from_files.empty()) {
    for (FileMap::iterator itFile = from_files.begin();
         itFile != from_files.end(); itFile++) {
      if (!itFile->second->empty()) {
        std::cout << "* File: " << itFile->first << std::endl;
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                     "~~~~~~~~~~~~~"
                  << std::endl;
        ArgMap &fileArgs = *(itFile->second);
        for (ArgMap::iterator it = fileArgs.begin(); it != fileArgs.end();
             it++) {
          std::cout.width(50);
          std::cout.fill('.');
          std::cout << std::left << it->first;
          std::cout << ": " << it->second.asString() << std::endl;
        }
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                     "~~~~~~~~~~~~~"
                  << std::endl;
      }
    }
  }

  // defaults defined in code
  if (!from_code.empty()) {
    std::cout << "* Defaults in Code:" << std::endl;
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                 "~~~~~~~~~"
              << std::endl;
    for (pArgMap::iterator it = from_code.begin(); it != from_code.end();
         it++) {
      std::cout.width(50);
      std::cout.fill('.');
      std::cout << std::left << it->first;
      std::cout << ": " << it->second->asString() << std::endl;
    }
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                 "~~~~~~~~~"
              << std::endl;
  }
}

} // namespace cubism

class BufferedLogger {
  struct Stream {
    std::stringstream stream;
    int requests_since_last_flush = 0;
    // GN: otherwise icpc complains
    Stream(const Stream &c) {}
    Stream() {}
  };
  typedef std::unordered_map<std::string, Stream> container_type;
  container_type files;

  /*
   * Flush a single stream and reset the counter.
   */
  void flush(container_type::iterator it);

public:
  ~BufferedLogger() { flush(); }

  /*
   * Get or create a string for a given file name.
   *
   * The stream is automatically flushed if accessed
   * many times since last flush.
   */
  std::stringstream &get_stream(const std::string &filename);

  /*
   * Flush all streams.
   */
  inline void flush(void) {
    for (auto it = files.begin(); it != files.end(); ++it)
      flush(it);
  }
};

extern BufferedLogger logger; // Declared in BufferedLogger.cpp.

BufferedLogger logger;

static constexpr int AUTO_FLUSH_COUNT = 100;

void BufferedLogger::flush(BufferedLogger::container_type::iterator it) {
  std::ofstream savestream;
  savestream.open(it->first, std::ios::app | std::ios::out);
  savestream << it->second.stream.rdbuf();
  savestream.close();
  it->second.requests_since_last_flush = 0;
}

std::stringstream &BufferedLogger::get_stream(const std::string &filename) {
  auto it = files.find(filename);
  if (it != files.end()) {
    if (++it->second.requests_since_last_flush == AUTO_FLUSH_COUNT)
      flush(it);
    return it->second.stream;
  } else {
    // With request_since_last_flush == 0,
    // the first flush will have AUTO_FLUSH_COUNT frames.
    auto new_it = files.emplace(filename, Stream()).first;
    return new_it->second.stream;
  }
}
//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

namespace cubism // AMR_CUBISM
{

/**
 * @brief Hilbert Space-Filling Curve(SFC) in 2D.
 *
 * The Quadtree of GridBlocks of a simulation is traversed by an SFC.
 * Each node of the Quadtree (aka each GridBlock) is associated with
 * (i) a refinement level
 * (ii) indices (i,j) that indicate its coordinates in a uniform grid of the
 * same refinement level (iii) a Z-order index which is a unique integer along
 * an SFC that would traverse a uniform grid of the same refinement level (iv) a
 * unique integer (blockID_2). This class provides trasformations from each of
 * these attributes to the others.
 */
class SpaceFillingCurve2D {
protected:
  int BX;         ///< number of blocks in the x-direction at the coarsest level
  int BY;         ///< number of blocks in the y-direction at the coarsest level
  int levelMax;   ///< maximum level allowed
  bool isRegular; ///< true if BX,BY,BZ are powers of 2
  int base_level; ///< minimum (starting) level (determined from BX,BY,BZ)
  std::vector<std::vector<long long>>
      Zsave; ///< option to save block indices instead of computing them every
             ///< time
  std::vector<std::vector<int>>
      i_inverse; ///< option to save blocks i index instead of computing it
                 ///< every time
  std::vector<std::vector<int>>
      j_inverse; ///< option to save blocks j index instead of computing it
                 ///< every time

  /// convert (x,y) to index
  long long AxestoTranspose(const int *X_in, int b) const {
    int x = X_in[0];
    int y = X_in[1];
    int n = 1 << b;
    int rx, ry, s, d = 0;
    for (s = n / 2; s > 0; s /= 2) {
      rx = (x & s) > 0;
      ry = (y & s) > 0;
      d += s * s * ((3 * rx) ^ ry);
      rot(n, &x, &y, rx, ry);
    }
    return d;
  }

  /// convert index to (x,y)
  void TransposetoAxes(long long index, int *X, int b) const {
    // position, #bits, dimension
    int n = 1 << b;
    long long rx, ry, s, t = index;
    X[0] = 0;
    X[1] = 0;
    for (s = 1; s < n; s *= 2) {
      rx = 1 & (t / 2);
      ry = 1 & (t ^ rx);
      rot(s, &X[0], &X[1], rx, ry);
      X[0] += s * rx;
      X[1] += s * ry;
      t /= 4;
    }
  }

  /// rotate/flip a quadrant appropriately
  void rot(long long n, int *x, int *y, long long rx, long long ry) const {
    if (ry == 0) {
      if (rx == 1) {
        *x = n - 1 - *x;
        *y = n - 1 - *y;
      }

      // Swap x and y
      int t = *x;
      *x = *y;
      *y = t;
    }
  }

public:
  SpaceFillingCurve2D() {};

  SpaceFillingCurve2D(int a_BX, int a_BY, int lmax)
      : BX(a_BX), BY(a_BY), levelMax(lmax) {
    const int n_max = std::max(BX, BY);
    base_level = (log(n_max) / log(2));
    if (base_level < (double)(log(n_max) / log(2)))
      base_level++;

    i_inverse.resize(lmax);
    j_inverse.resize(lmax);
    Zsave.resize(lmax);
    {
      const int l = 0;
      const int aux = pow(pow(2, l), 2);
      i_inverse[l].resize(BX * BY * aux, -1);
      j_inverse[l].resize(BX * BY * aux, -1);
      Zsave[l].resize(BX * BY * aux, -1);
    }

    isRegular = true;
#pragma omp parallel for collapse(2)
    for (int j = 0; j < BY; j++)
      for (int i = 0; i < BX; i++) {
        const int c[2] = {i, j};
        long long index = AxestoTranspose(c, base_level);
        long long substract = 0;
        for (long long h = 0; h < index; h++) {
          int X[2] = {0, 0};
          TransposetoAxes(h, X, base_level);
          if (X[0] >= BX || X[1] >= BY)
            substract++;
        }
        index -= substract;
        if (substract > 0)
          isRegular = false;
        i_inverse[0][index] = i;
        j_inverse[0][index] = j;
        Zsave[0][j * BX + i] = index;
      }
  }

  /// space-filling curve (i,j) --> 1D index (given level l)
  long long forward(const int l, const int i, const int j) // const
  {
    const int aux = 1 << l;

    if (l >= levelMax)
      return 0;
    long long retval;
    if (!isRegular) {
      const int I = i / aux;
      const int J = j / aux;
      const int c2_a[2] = {i - I * aux, j - J * aux};
      retval = AxestoTranspose(c2_a, l);
      retval += IJ_to_index(I, J) * aux * aux;
    } else {
      const int c2_a[2] = {i, j};
      retval = AxestoTranspose(c2_a, l + base_level);
    }
    return retval;
  }

  /// space-filling curve Z-index --> (i,j) (given level l)
  void inverse(long long Z, int l, int &i, int &j) {
    if (isRegular) {
      int X[2] = {0, 0};
      TransposetoAxes(Z, X, l + base_level);
      i = X[0];
      j = X[1];
    } else {
      int aux = 1 << l;
      long long Zloc = Z % (aux * aux);
      int X[2] = {0, 0};
      TransposetoAxes(Zloc, X, l);
      long long index = Z / (aux * aux);
      int I, J;
      index_to_IJ(index, I, J);
      i = X[0] + I * aux;
      j = X[1] + J * aux;
    }
    return;
  }

  /// space-filling curve (i,j) --> 1D index (at level 0)
  long long IJ_to_index(int I, int J) {
    // int index = (J + K * BY) * BX + I;
    long long index = Zsave[0][J * BX + I];
    return index;
  }

  /// space-filling curve Z-index --> (i,j) (at level 0)
  void index_to_IJ(long long index, int &I, int &J) {
    I = i_inverse[0][index];
    J = j_inverse[0][index];
    return;
  }

  /// convert Z-index, level and ij index to single unique number
  long long Encode(int level, long long Z, int index[2]) {
    int lmax = levelMax;
    long long retval = 0;

    int ix = index[0];
    int iy = index[1];
    for (int l = level; l >= 0; l--) {
      long long Zp = forward(l, ix, iy);
      retval += Zp;
      ix /= 2;
      iy /= 2;
    }

    ix = 2 * index[0];
    iy = 2 * index[1];
    for (int l = level + 1; l < lmax; l++) {
      long long Zc = forward(l, ix, iy);

      Zc -= Zc % 4;
      retval += Zc;

      int ix1, iy1;
      inverse(Zc, l, ix1, iy1);
      ix = 2 * ix1;
      iy = 2 * iy1;
    }

    retval += level;

    return retval;
  }
};

} // namespace cubism

namespace cubism {

enum State : signed char { Leave = 0, Refine = 1, Compress = -1 };

/// Single integer used to recognize if a Block exists in the Grid and by which
/// MPI rank it is owned.
struct TreePosition {
  int position{-3};
  bool CheckCoarser() const { return position == -2; }
  bool CheckFiner() const { return position == -1; }
  bool Exists() const { return position >= 0; }
  int rank() const { return position; }
  void setrank(const int r) { position = r; }
  void setCheckCoarser() { position = -2; }
  void setCheckFiner() { position = -1; }
};

/** @brief Meta-data for each GridBlock.
 *
 * This struct holds information such as the grid spacing and the level of
 * refinement of each GridBlock. It is also used to access the data of the
 * GridBlock through a relevant pointer. Importantly, all blocks are organized
 * in a single Octree/Quadtree, regardless of the number of fields/variables
 * used in the simulation (and the number of Grids). For this reason, only one
 * instance of SpaceFillingCurve is needed, which is owned as a static member of
 * the BlockInfo struct. The functions of the SpaceFillingCurve are also
 * accessed through static functions of BlockInfo.
 */
struct BlockInfo {
  long long
      blockID; ///< all n BlockInfos owned by one rank have blockID=0,1,...,n-1
  long long blockID_2;     ///< unique index of each BlockInfo, based on its
                           ///< refinement level and Z-order curve index
  long long Z;             ///< Z-order curve index of this block
  long long Znei[3][3][3]; ///< Z-order curve index of 26 neighboring boxes
                           ///< (Znei[1][1][1] = Z)
  long long halo_block_id; ///< all m blocks at the boundary of a rank are
                           ///< numbered by halo_block_id=0,1,...,m-1
  long long Zparent; ///< Z-order curve index of parent block (after comression)
  long long Zchild[2][2][2]; ///< Z-order curve index of blocks that replace
                             ///< this one during refinement
  double h;                  ///< grid spacing
  double origin[3];          ///<(x,y,z) of block's origin
  int index[3]; ///<(i,j,k) coordinates of block at given refinement level
  int level;    ///< refinement level
  void *ptrBlock{nullptr}; ///< Pointer to data stored in user-defined Block
  void *auxiliary;         ///< Pointer to blockcase
  bool changed2; ///< =true if block will be refined/compressed; used to update
                 ///< State of neighbouring blocks among ranks
  State state;   ///< Refine/Compress/Leave this block

  /// Static function used to initialize static SFC
  static int levelMax(int l = 0) {
    static int lmax = l;
    return lmax;
  }

#if DIMENSION == 3

  /// Static function used to initialize static SFC
  static int blocks_per_dim(int i, int nx = 0, int ny = 0, int nz = 0) {
    static int a[3] = {nx, ny, nz};
    return a[i];
  }

  /// Pointer to single instance of SFC used
  static SpaceFillingCurve *SFC() {
    static SpaceFillingCurve Zcurve(blocks_per_dim(0), blocks_per_dim(1),
                                    blocks_per_dim(2), levelMax());
    return &Zcurve;
  }

  /// get Z-order index for coordinates (ix,iy,iz) and refinement level
  static long long forward(int level, int ix, int iy, int iz) {
    return (*SFC()).forward(level, ix, iy, iz);
  }

  /// get unique blockID_2 index from refinement level, Z-order index and
  /// coordinates
  static long long Encode(int level, long long Z, int index[3]) {
    return (*SFC()).Encode(level, Z, index);
  }

  /// get coordinates from refinement level and Z-order index
  static void inverse(long long Z, int l, int &i, int &j, int &k) {
    (*SFC()).inverse(Z, l, i, j, k);
  }

#else

  /// Static function used to initialize static SFC (same as above but in 2D)
  static int blocks_per_dim(int i, int nx = 0, int ny = 0) {
    static int a[2] = {nx, ny};
    return a[i];
  }

  /// Pointer to single instance of SFC used (same as above but in 2D)
  static SpaceFillingCurve2D *SFC() {
    static SpaceFillingCurve2D Zcurve(blocks_per_dim(0), blocks_per_dim(1),
                                      levelMax());
    return &Zcurve;
  }

  /// get Z-order index for coordinates (ix,iy,iz) and refinement level
  static long long forward(int level, int ix, int iy) {
    return (*SFC()).forward(level, ix, iy);
  }

  /// get unique blockID_2 index from refinement level, Z-order index and
  /// coordinates
  static long long Encode(int level, long long Z, int index[2]) {
    return (*SFC()).Encode(level, Z, index);
  }

  /// get coordinates from refinement level and Z-order index
  static void inverse(long long Z, int l, int &i, int &j) {
    (*SFC()).inverse(Z, l, i, j);
  }

#endif

#if DIMENSION == 3
  /// return position (x,y,z) in 3D, given indices of grid point
  template <typename T> inline void pos(T p[3], int ix, int iy, int iz) const {
    p[0] = origin[0] + h * (ix + 0.5);
    p[1] = origin[1] + h * (iy + 0.5);
    p[2] = origin[2] + h * (iz + 0.5);
  }

  /// return position (x,y,z) in 3D, given indices of grid point
  template <typename T>
  inline std::array<T, 3> pos(int ix, int iy, int iz) const {
    std::array<T, 3> result;
    pos(result.data(), ix, iy, iz);
    return result;
  }
#else
  /// return position (x,y) in 2D, given indices of grid point
  template <typename T> inline void pos(T p[2], int ix, int iy) const {
    p[0] = origin[0] + h * (ix + 0.5);
    p[1] = origin[1] + h * (iy + 0.5);
  }

  /// return position (x,y) in 2D, given indices of grid point
  template <typename T> inline std::array<T, 2> pos(int ix, int iy) const {
    std::array<T, 2> result;
    pos(result.data(), ix, iy);
    return result;
  }
#endif

  /// used to order/sort blocks based on blockID_2, which is only a function of
  /// Z and level
  bool operator<(const BlockInfo &other) const {
    return (blockID_2 < other.blockID_2);
  }

  /// constructor will do nothing, 'setup' needs to be called instead
  BlockInfo() {};

  /// Provide level, grid spacing, (x,y,z) origin and Z-index to
  /// setup/initialize a blockinfo
  void setup(const int a_level, const double a_h, const double a_origin[3],
             const long long a_Z) {
    level = a_level;
    Z = a_Z;
    state = Leave;
    level = a_level;
    h = a_h;
    origin[0] = a_origin[0];
    origin[1] = a_origin[1];
    origin[2] = a_origin[2];
    changed2 = true;
    auxiliary = nullptr;

    const int TwoPower = 1 << level;

// Now we also set the indices of the neighbouring blocks, parent block and
// child blocks.
#if DIMENSION == 3
    inverse(Z, level, index[0], index[1], index[2]);

    const int Bmax[3] = {blocks_per_dim(0) * TwoPower,
                         blocks_per_dim(1) * TwoPower,
                         blocks_per_dim(2) * TwoPower};
    for (int i = -1; i < 2; i++)
      for (int j = -1; j < 2; j++)
        for (int k = -1; k < 2; k++)
          Znei[i + 1][j + 1][k + 1] =
              forward(level, (index[0] + i + Bmax[0]) % Bmax[0],
                      (index[1] + j + Bmax[1]) % Bmax[1],
                      (index[2] + k + Bmax[2]) % Bmax[2]);

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
          Zchild[i][j][k] = forward(level + 1, 2 * index[0] + i,
                                    2 * index[1] + j, 2 * index[2] + k);

    Zparent = (level == 0)
                  ? 0
                  : forward(level - 1, (index[0] / 2 + Bmax[0]) % Bmax[0],
                            (index[1] / 2 + Bmax[1]) % Bmax[1],
                            (index[2] / 2 + Bmax[2]) % Bmax[2]);
#else
    inverse(Z, level, index[0], index[1]);
    index[2] = 0;

    const int Bmax[3] = {blocks_per_dim(0) * TwoPower,
                         blocks_per_dim(1) * TwoPower, 1};
    for (int i = -1; i < 2; i++)
      for (int j = -1; j < 2; j++)
        for (int k = -1; k < 2; k++)
          Znei[i + 1][j + 1][k + 1] =
              forward(level, (index[0] + i + Bmax[0]) % Bmax[0],
                      (index[1] + j + Bmax[1]) % Bmax[1]);

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
          Zchild[i][j][k] =
              forward(level + 1, 2 * index[0] + i, 2 * index[1] + j);

    Zparent = (level == 0)
                  ? 0
                  : forward(level - 1, (index[0] / 2 + Bmax[0]) % Bmax[0],
                            (index[1] / 2 + Bmax[1]) % Bmax[1]);
#endif
    blockID_2 = Encode(level, Z, index);
    blockID = blockID_2;
  }

  /// used for easier access of Znei[][][]
  long long Znei_(const int i, const int j, const int k) const {
    assert(abs(i) <= 1);
    assert(abs(j) <= 1);
    assert(abs(k) <= 1);
    return Znei[1 + i][1 + j][1 + k];
  }
};
} // namespace cubism

namespace cubism {

/**
 * @brief Auxiliary struct used to perform flux corrections at coarse-fine block
 * interfaces.
 *
 * This struct can save the fluxes passing though the six faces of one
 * GridBlock. Each BlockInfo owns a pointer to its own BlockCase. The pointer is
 * not a nullptr only if any of the six faces of the block have a neighboring
 * block at a different refinement level. When a stencil computation is
 * performed, each block can fill its own BlockCase with the fluxes passing
 * through its faces. Then, the FluxCorrection class will replace the coarse
 * fluxes with the sum of the fine fluxes, which ensures conservation of the
 * quantity whose flux we compute.
 *
 * @tparam BlockType The user-defined GridBlock
 * @tparam ElementType The type of elements stored by the user-defined GridBlock
 */
template <typename BlockType,
          typename ElementType = typename BlockType::ElementType>
struct BlockCase {
  std::vector<std::vector<ElementType>>
      m_pData;             ///< six vectors, one for each face
  unsigned int m_vSize[3]; ///< sizes of the faces (in x,y and z)
  bool storedFace[6]; ///< boolean variables, one for each face (=true if this
                      ///< face needs flux corrections because it is at a
                      ///< coarse-fine interface)
  int level;          ///< refinement level of the associated block
  long long Z;        ///< Z-order index of the associated block

  /// Constructor.
  BlockCase(bool _storedFace[6], unsigned int nX, unsigned int nY,
            unsigned int nZ, int _level, long long _Z) {
    m_vSize[0] = nX;
    m_vSize[1] = nY;
    m_vSize[2] = nZ;

    storedFace[0] = _storedFace[0];
    storedFace[1] = _storedFace[1];
    storedFace[2] = _storedFace[2];
    storedFace[3] = _storedFace[3];
    storedFace[4] = _storedFace[4];
    storedFace[5] = _storedFace[5];

    m_pData.resize(6);

    for (int d = 0; d < 3; d++) {
      int d1 = (d + 1) % 3;
      int d2 = (d + 2) % 3;

      // assume everything is initialized to 0!!!!!!!!
      if (storedFace[2 * d])
        m_pData[2 * d].resize(m_vSize[d1] * m_vSize[d2]);
      if (storedFace[2 * d + 1])
        m_pData[2 * d + 1].resize(m_vSize[d1] * m_vSize[d2]);
    }
    level = _level;
    Z = _Z;
  }

  ~BlockCase() {}
};

/**
 * @brief Performs flux corrections at coarse-fine block interfaces.
 *
 * This class can replace the coarse fluxes stored at BlockCases with the sum of
 * the fine fluxes (also stored at BlockCases). This ensures conservation of the
 * quantity whose flux we compute.
 *
 * @tparam TGrid The user-defined Grid/GridMPI
 * @tparam BlockType The user-defined GridBlock used by TGrid
 */
template <typename TGrid> class FluxCorrection {
public:
  using GridType = TGrid; ///< should be a 'Grid', 'GridMPI' or derived class
  typedef typename GridType::BlockType BlockType;
  typedef typename BlockType::ElementType ElementType;
  typedef typename ElementType::RealType Real;
  typedef BlockCase<BlockType> Case;
  int rank{
      0}; ///< MPI process ID (set to zero here, for a serial implementation)

protected:
  std::map<std::array<long long, 2>, Case *>
      MapOfCases; ///< map between BlockCases and BlockInfos (two integers:
                  ///< refinement level and Z-order index)
  TGrid *grid;    ///< grid for which we perform the flux corrections
  std::vector<Case> Cases; ///< BlockCases owned by FluxCorrection; BlockInfos
                           ///< have pointers to these (in needed)

  /// Perform flux correction for BlockInfo 'info' in the direction/face
  /// specified by 'code'
  void FillCase(BlockInfo &info, const int *const code) {
    const int myFace = abs(code[0]) * std::max(0, code[0]) +
                       abs(code[1]) * (std::max(0, code[1]) + 2) +
                       abs(code[2]) * (std::max(0, code[2]) + 4);
    const int otherFace = abs(-code[0]) * std::max(0, -code[0]) +
                          abs(-code[1]) * (std::max(0, -code[1]) + 2) +
                          abs(-code[2]) * (std::max(0, -code[2]) + 4);

    std::array<long long, 2> temp = {(long long)info.level, info.Z};
    auto search = MapOfCases.find(temp);

    Case &CoarseCase = (*search->second);
    std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];

    assert(myFace / 2 == otherFace / 2);
    assert(search != MapOfCases.end());
    assert(CoarseCase.Z == info.Z);
    assert(CoarseCase.level == info.level);

#if DIMENSION == 3
    for (int B = 0; B <= 3;
         B++) // loop over fine blocks that make up coarse face
#else
    for (int B = 0; B <= 1;
         B++) // loop over fine blocks that make up coarse face
#endif
    {
      const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
#if DIMENSION == 3
      const long long Z = (*grid).getZforward(
          info.level + 1,
          2 * info.index[0] + std::max(code[0], 0) + code[0] +
              (B % 2) * std::max(0, 1 - abs(code[0])),
          2 * info.index[1] + std::max(code[1], 0) + code[1] +
              aux * std::max(0, 1 - abs(code[1])),
          2 * info.index[2] + std::max(code[2], 0) + code[2] +
              (B / 2) * std::max(0, 1 - abs(code[2])));
#else
      const long long Z = (*grid).getZforward(
          info.level + 1,
          2 * info.index[0] + std::max(code[0], 0) + code[0] +
              (B % 2) * std::max(0, 1 - abs(code[0])),
          2 * info.index[1] + std::max(code[1], 0) + code[1] +
              aux * std::max(0, 1 - abs(code[1])));
#endif

      const int other_rank = grid->Tree(info.level + 1, Z).rank();
      if (other_rank != rank)
        continue;
      auto search1 = MapOfCases.find({info.level + 1, Z});

      Case &FineCase = (*search1->second);
      std::vector<ElementType> &FineFace = FineCase.m_pData[otherFace];
      const int d = myFace / 2;
      const int d1 = std::max((d + 1) % 3, (d + 2) % 3);
      const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
      const int N1F = FineCase.m_vSize[d1];
      const int N2F = FineCase.m_vSize[d2];
      const int N1 = N1F;
      const int N2 = N2F;
      int base = 0;
      if (B == 1)
        base = (N2 / 2) + (0) * N2;
      else if (B == 2)
        base = (0) + (N1 / 2) * N2;
      else if (B == 3)
        base = (N2 / 2) + (N1 / 2) * N2;
      assert(search1 != MapOfCases.end());
      assert(N1F == (int)CoarseCase.m_vSize[d1]);
      assert(N2F == (int)CoarseCase.m_vSize[d2]);
      assert(FineFace.size() == CoarseFace.size());
#if DIMENSION == 3
      for (int i1 = 0; i1 < N1; i1 += 2)
        for (int i2 = 0; i2 < N2; i2 += 2) {
          CoarseFace[base + (i2 / 2) + (i1 / 2) * N2] +=
              FineFace[i2 + i1 * N2] + FineFace[i2 + 1 + i1 * N2] +
              FineFace[i2 + (i1 + 1) * N2] + FineFace[i2 + 1 + (i1 + 1) * N2];
          FineFace[i2 + i1 * N2].clear();
          FineFace[i2 + 1 + i1 * N2].clear();
          FineFace[i2 + (i1 + 1) * N2].clear();
          FineFace[i2 + 1 + (i1 + 1) * N2].clear();
        }
#else
      for (int i2 = 0; i2 < N2; i2 += 2) {
        CoarseFace[base + i2 / 2] += FineFace[i2] + FineFace[i2 + 1];
        FineFace[i2].clear();
        FineFace[i2 + 1].clear();
      }
#endif
    }
  }

public:
  /// Prepare the FluxCorrection class for a given 'grid' by allocating
  /// BlockCases at each coarse-fine interface
  virtual void prepare(TGrid &_grid) {
    if (_grid.UpdateFluxCorrection == false)
      return;
    _grid.UpdateFluxCorrection = false;

    Cases.clear();
    MapOfCases.clear();
    grid = &_grid;
    std::vector<BlockInfo> &B = (*grid).getBlocksInfo();

    std::array<int, 3> blocksPerDim = (*grid).getMaxBlocks();
    std::array<int, 6> icode = {1 * 2 + 3 * 1 + 9 * 1, 1 * 0 + 3 * 1 + 9 * 1,
                                1 * 1 + 3 * 2 + 9 * 1, 1 * 1 + 3 * 0 + 9 * 1,
                                1 * 1 + 3 * 1 + 9 * 2, 1 * 1 + 3 * 1 + 9 * 0};

    for (auto &info : B) {
      grid->getBlockInfoAll(info.level, info.Z).auxiliary = nullptr;
      const int aux = 1 << info.level;

      const bool xskin =
          info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
      const bool yskin =
          info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
      const bool zskin =
          info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;

      bool storeFace[6] = {false, false, false, false, false, false};
      bool stored = false;

      for (int f = 0; f < 6; f++) {
        const int code[3] = {icode[f] % 3 - 1, (icode[f] / 3) % 3 - 1,
                             (icode[f] / 9) % 3 - 1};
        if (!_grid.xperiodic && code[0] == xskip && xskin)
          continue;
        if (!_grid.yperiodic && code[1] == yskip && yskin)
          continue;
        if (!_grid.zperiodic && code[2] == zskip && zskin)
          continue;
#if DIMENSION == 2
        if (code[2] != 0)
          continue;
#endif

        if (!grid->Tree(info.level, info.Znei_(code[0], code[1], code[2]))
                 .Exists()) {
          storeFace[abs(code[0]) * std::max(0, code[0]) +
                    abs(code[1]) * (std::max(0, code[1]) + 2) +
                    abs(code[2]) * (std::max(0, code[2]) + 4)] = true;
          stored = true;
        }
      }
      if (stored) {
        Cases.push_back(Case(storeFace, BlockType::sizeX, BlockType::sizeY,
                             BlockType::sizeZ, info.level, info.Z));
      }
    }
    size_t Cases_index = 0;
    if (Cases.size() > 0)
      for (auto &info : B) {
        if (Cases_index == Cases.size())
          break;
        if (Cases[Cases_index].level == info.level &&
            Cases[Cases_index].Z == info.Z) {
          MapOfCases.insert(std::pair<std::array<long long, 2>, Case *>(
              {Cases[Cases_index].level, Cases[Cases_index].Z},
              &Cases[Cases_index]));
          grid->getBlockInfoAll(Cases[Cases_index].level, Cases[Cases_index].Z)
              .auxiliary = &Cases[Cases_index];
          info.auxiliary = &Cases[Cases_index];
          Cases_index++;
        }
      }
  }

  /// Go over each coarse-fine interface and perform the flux corrections,
  /// assuming the associated BlockCases have been filled with the fluxes by the
  /// user
  virtual void FillBlockCases() {
    // This assumes that the BlockCases have been filled by the user somehow...
    std::vector<BlockInfo> &B = (*grid).getBlocksInfo();

    std::array<int, 3> blocksPerDim = (*grid).getMaxBlocks();

    std::array<int, 6> icode = {1 * 2 + 3 * 1 + 9 * 1, 1 * 0 + 3 * 1 + 9 * 1,
                                1 * 1 + 3 * 2 + 9 * 1, 1 * 1 + 3 * 0 + 9 * 1,
                                1 * 1 + 3 * 1 + 9 * 2, 1 * 1 + 3 * 1 + 9 * 0};

#pragma omp parallel for
    for (size_t i = 0; i < B.size(); i++) {
      BlockInfo &info = B[i];
      const int aux = 1 << info.level;
      const bool xskin =
          info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
      const bool yskin =
          info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
      const bool zskin =
          info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;

      for (int f = 0; f < 6; f++) {
        const int code[3] = {icode[f] % 3 - 1, (icode[f] / 3) % 3 - 1,
                             (icode[f] / 9) % 3 - 1};

        if (!grid->xperiodic && code[0] == xskip && xskin)
          continue;
        if (!grid->yperiodic && code[1] == yskip && yskin)
          continue;
        if (!grid->zperiodic && code[2] == zskip && zskin)
          continue;
#if DIMENSION == 2
        if (code[2] != 0)
          continue;
#endif

        bool checkFiner =
            grid->Tree(info.level, info.Znei_(code[0], code[1], code[2]))
                .CheckFiner();

        if (checkFiner) {
          FillCase(info, code);

          const int myFace = abs(code[0]) * std::max(0, code[0]) +
                             abs(code[1]) * (std::max(0, code[1]) + 2) +
                             abs(code[2]) * (std::max(0, code[2]) + 4);
          std::array<long long, 2> temp = {(long long)info.level, info.Z};
          auto search = MapOfCases.find(temp);
          assert(search != MapOfCases.end());
          Case &CoarseCase = (*search->second);
          std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];
          const int d = myFace / 2;
          const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
          const int N2 = CoarseCase.m_vSize[d2];
          BlockType &block = *(BlockType *)info.ptrBlock;

#if DIMENSION == 3
          // WARNING: tmp indices are tmp[z][y][x][Flow Quantity]!
          const int d1 = std::max((d + 1) % 3, (d + 2) % 3);
          const int N1 = CoarseCase.m_vSize[d1];
          if (d == 0) {
            const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeX - 1;
            for (int i1 = 0; i1 < N1; i1++)
              for (int i2 = 0; i2 < N2; i2++) {
                block(j, i2, i1) += CoarseFace[i2 + i1 * N2];
                CoarseFace[i2 + i1 * N2].clear();
              }
          } else if (d == 1) {
            const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeY - 1;
            for (int i1 = 0; i1 < N1; i1++)
              for (int i2 = 0; i2 < N2; i2++) {
                block(i2, j, i1) += CoarseFace[i2 + i1 * N2];
                CoarseFace[i2 + i1 * N2].clear();
              }
          } else {
            const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeZ - 1;
            for (int i1 = 0; i1 < N1; i1++)
              for (int i2 = 0; i2 < N2; i2++) {
                block(i2, i1, j) += CoarseFace[i2 + i1 * N2];
                CoarseFace[i2 + i1 * N2].clear();
              }
          }
#else
          assert(d != 2);
          if (d == 0) {
            const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeX - 1;
            for (int i2 = 0; i2 < N2; i2++) {
              block(j, i2) += CoarseFace[i2];
              CoarseFace[i2].clear();
            }
          } else // if (d == 1)
          {
            const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeY - 1;
            for (int i2 = 0; i2 < N2; i2++) {
              block(i2, j) += CoarseFace[i2];
              CoarseFace[i2].clear();
            }
          }
#endif
        }
      }
    }
  }
};

} // namespace cubism

namespace cubism {

/** When dumping a Grid, blocks are grouped into larger rectangular regions
 *  of uniform resolution. These regions (BlockGroups) have blocks with the
 *  same level and with various Space-filling-curve coordinates Z.
 *  They have NXX x NYY x NZZ grid points, grid spacing h, an origin and a
 *  minimum and maximum index (indices of bottom left and top right blocks).
 */
struct BlockGroup {
  int i_min[3];             ///< min (i,j,k) index of a block of this group
  int i_max[3];             ///< max (i,j,k) index of a block of this group
  int level;                ///< refinement level
  std::vector<long long> Z; ///< Z-order indices of blocks of this group
  size_t ID;                ///< unique group number
  double origin[3];         ///< Coordinates (x,y,z) of origin
  double h;                 ///< Grid spacing of the group
  int NXX;                  ///< Grid points of the group in the x-direction
  int NYY;                  ///< Grid points of the group in the y-direction
  int NZZ;                  ///< Grid points of the group in the z-direction
};

/** Holds the GridBlocks and their meta-data (BlockInfos).
 * This class provides information about the current state of the Octree of
 * blocks in the simulation. The user can request if a particular block is
 * present in the Octree or if its parent/ children block(s) are present
 * instead. This class also provides access to the raw data from the simulation.
 */
template <typename Block,
          template <typename X> class allocator = std::allocator>
class Grid {
public:
  typedef Block BlockType;
  using ElementType = typename Block::ElementType; ///< Blocks hold ElementTypes
  typedef typename Block::RealType Real; ///< Blocks must provide `RealType`.

#ifdef CUBISM_USE_ONETBB
  tbb::concurrent_unordered_map<long long, BlockInfo *> BlockInfoAll;
  tbb::concurrent_unordered_map<long long, TreePosition> Octree;
#else

  /** A map from unique BlockInfo IDs to pointers to BlockInfos.
   *  Should be accessed through function 'getBlockInfoAll'. If a Block does not
   * belong to this rank and it is not adjacent to it, this map should not
   * return something meaningful.
   */
  std::unordered_map<long long, BlockInfo *> BlockInfoAll;

  /** A map from unique BlockInfo IDs to pointers to integers (TreePositions)
   * that encode whether a BlockInfo is present in the Octree (and to which rank
   * it belongs to) or not. This is a seperate object from BlockInfoAll because
   * all the information we need for some blocks is merely whether they exist or
   * not (i.e. we don't need their grid spacing or other meta-data) held by
   * BlockInfos.
   */
  std::unordered_map<long long, TreePosition> Octree;
#endif

  /** Meta-data for blocks that belong to this rank.
   *  This vector holds all the BlockInfos for blocks that belong to this rank.
   * When the mesh changes, the contents of this vector become outdated and need
   * to be updated. This is done through the FillPos() function. This array
   * should be used when iterating over the blocks owned by a Grid.
   */
  std::vector<BlockInfo> m_vInfo;

  const int NX;           ///< Total # of blocks for level 0 in X-direction
  const int NY;           ///< Total # of blocks for level 0 in Y-direction
  const int NZ;           ///< Total # of blocks for level 0 in Z-direction
  const double maxextent; ///< Maximum domain extent
  const int levelMax;     ///< Maximum refinement level allowed
  const int levelStart;   ///< Initial refinement level
  const bool xperiodic;   ///< grid periodicity in x-direction
  const bool yperiodic;   ///< grid periodicity in y-direction
  const bool zperiodic;   ///< grid periodicity in z-direction
  std::vector<BlockGroup> MyGroups; ///< used for dumping data
  std::vector<long long>
      level_base; ///< auxiliary array used when searching is std::unordered_map
  bool UpdateFluxCorrection{
      true}; ///< FluxCorrection updates only when grid is refined/compressed
  bool UpdateGroups{
      true}; ///< (inactive) BlockGroups updated only when this is true
  bool FiniteDifferences{
      true}; ///< used by BlockLab, to determine what kind of coarse-fine
             ///< interface interpolation to make.true means that biased
             ///< stencils will be used to get an O(h^3) approximation
  FluxCorrection<Grid> CorrectorGrid; ///< used for AMR flux-corrections at
                                      ///< coarse-fine interfaces

  /// Get the TreePosition of block with Z-order index 'm', at refinement level
  /// 'n'.
  TreePosition &Tree(const int m, const long long n) {
    /*
     * Return the position in the Octree of a Block at level m and SFC
     * coordinate n.
     */
    const long long aux = level_base[m] + n;
    const auto retval = Octree.find(aux);
    if (retval == Octree.end()) {
#ifndef CUBISM_USE_ONETBB
#pragma omp critical
#endif
      {
        const auto retval1 = Octree.find(aux);
        if (retval1 == Octree.end()) {
          TreePosition dum;
          Octree[aux] = dum;
        }
      }
      return Tree(m, n);
    } else {
      return retval->second;
    }
  }
  /// Get the TreePosition of block with BlockInfo 'info'.
  TreePosition &Tree(BlockInfo &info) { return Tree(info.level, info.Z); }
  /// Get the TreePosition of block with BlockInfo 'info'.
  TreePosition &Tree(const BlockInfo &info) { return Tree(info.level, info.Z); }

  /// Called in constructor to allocate all blocks at level=levelStart.
  void _alloc() {
    const int m = levelStart;
    const int TwoPower = 1 << m;
    for (long long n = 0; n < NX * NY * NZ * pow(TwoPower, DIMENSION); n++) {
      Tree(m, n).setrank(0);
      _alloc(m, n);
    }
    if (m - 1 >= 0) {
      for (long long n = 0; n < NX * NY * NZ * pow((1 << (m - 1)), DIMENSION);
           n++)
        Tree(m - 1, n).setCheckFiner();
    }
    if (m + 1 < levelMax) {
      for (long long n = 0; n < NX * NY * NZ * pow((1 << (m + 1)), DIMENSION);
           n++)
        Tree(m + 1, n).setCheckCoarser();
    }
    FillPos();
  }

  /// Called to allocate a block with Z-order index 'm' at refinement level 'n',
  /// when the grid is refined.
  void _alloc(const int m, const long long n) {
    allocator<Block> alloc;

    BlockInfo &new_info = getBlockInfoAll(m, n);
    new_info.ptrBlock = alloc.allocate(1);
#pragma omp critical
    { m_vInfo.push_back(new_info); }
    Tree(m, n).setrank(rank());
  }

  /// Called in destructor to deallocate all blocks.
  void _deallocAll() {
    allocator<Block> alloc;
    for (size_t i = 0; i < m_vInfo.size(); i++) {
      const int m = m_vInfo[i].level;
      const long long n = m_vInfo[i].Z;
      alloc.deallocate((Block *)getBlockInfoAll(m, n).ptrBlock, 1);
    }
    std::vector<long long> aux;
    for (auto &m : BlockInfoAll)
      aux.push_back(m.first);
    for (size_t i = 0; i < aux.size(); i++) {
      const auto retval = BlockInfoAll.find(aux[i]);
      if (retval != BlockInfoAll.end()) {
        delete retval->second;
      }
    }
    m_vInfo.clear();
    BlockInfoAll.clear();
    Octree.clear();
  }

  /// Called to deallocate a block with Z-order index 'm' at refinement level
  /// 'n', when the grid is compressed.
  void _dealloc(const int m, const long long n) {
    allocator<Block> alloc;
    alloc.deallocate((Block *)getBlockInfoAll(m, n).ptrBlock, 1);
    for (size_t j = 0; j < m_vInfo.size(); j++) {
      if (m_vInfo[j].level == m && m_vInfo[j].Z == n) {
        m_vInfo.erase(m_vInfo.begin() + j);
        return;
      }
    }
  }

  /// Called to deallocate many blocks with blockIDs in the vector 'dealloc_IDs'
  void dealloc_many(const std::vector<long long> &dealloc_IDs) {
    for (size_t j = 0; j < m_vInfo.size(); j++)
      m_vInfo[j].changed2 = false;

    allocator<Block> alloc;

    for (size_t i = 0; i < dealloc_IDs.size(); i++)
      for (size_t j = 0; j < m_vInfo.size(); j++) {
        if (m_vInfo[j].blockID_2 == dealloc_IDs[i]) {
          const int m = m_vInfo[j].level;
          const long long n = m_vInfo[j].Z;
          m_vInfo[j].changed2 = true;
          alloc.deallocate((Block *)getBlockInfoAll(m, n).ptrBlock, 1);
          break;
        }
      }
    // for c++20
    // std::erase_if(m_vInfo, [](BlockInfo & x) { return x.changed2; });
    // for c++17
    m_vInfo.erase(std::remove_if(m_vInfo.begin(), m_vInfo.end(),
                                 [](const BlockInfo &x) { return x.changed2; }),
                  m_vInfo.end());
  }

  /** Used when Block at level m_new with SFC coordinate n_new is added to the
   * Grid as a result of compression of Block (m,n). Sets the state of the newly
   * added Block. It also replaces BlockInfo(m,n) and Block(m,n) with
   *  BlockInfo(m_new,n_new) and Block(m_new,n_new).
   * @param m: Refinement level of the GridBlock that is compressed.
   * @param n: Z-order index of the GridBlock that is compressed.
   * @param m_new: Refinement level of the GridBlock that will replace the
   * compressed GridBlock.
   * @param n_new: Z-order index of the GridBlock that will replace the
   * compressed GridBlock.
   */
  void FindBlockInfo(const int m, const long long n, const int m_new,
                     const long long n_new) {
    for (size_t j = 0; j < m_vInfo.size(); j++)
      if (m == m_vInfo[j].level && n == m_vInfo[j].Z) {
        BlockInfo &correct_info = getBlockInfoAll(m_new, n_new);
        correct_info.state = Leave;
        m_vInfo[j] = correct_info;
        return;
      }
  }

  /** The data in BlockInfoAll is always correct (states, blockIDs etc.), but
   * this is not the case for m_vInfo, whose content might be outdated after
   * grid refinement/compression or exchange of blocks between different ranks.
   * This function updates their content.
   * @param CopyInfos: set to true if the correct BlockInfos from BlockInfoAll
   * should be copied to m_vInfo. Otherwise only selected variables are copied.
   */
  virtual void FillPos(bool CopyInfos = true) {
    std::sort(m_vInfo.begin(), m_vInfo.end()); // sort according to blockID_2

#ifndef CUBISM_USE_ONETBB
    // The following will reserve memory for the unordered map.
    // This will result in a thread-safe Tree(m,n) function
    // as Octree will not change size when it is accessed by
    // multiple threads. The number m_vInfo.size()/8 is arbitrary.
    Octree.reserve(Octree.size() + m_vInfo.size() / 8);
#endif

    if (CopyInfos)
      for (size_t j = 0; j < m_vInfo.size(); j++) {
        const int m = m_vInfo[j].level;
        const long long n = m_vInfo[j].Z;
        BlockInfo &correct_info = getBlockInfoAll(m, n);
        correct_info.blockID = j;
        m_vInfo[j] = correct_info;
        assert(Tree(m, n).Exists());
      }
    else
      for (size_t j = 0; j < m_vInfo.size(); j++) {
        const int m = m_vInfo[j].level;
        const long long n = m_vInfo[j].Z;
        BlockInfo &correct_info = getBlockInfoAll(m, n);
        correct_info.blockID = j;
        m_vInfo[j].blockID = j;
        m_vInfo[j].state = correct_info.state;
        assert(Tree(m, n).Exists());
      }
  }

  /** Constructor.
   * @param _NX: total number of blocks in the x-direction, at the coarsest
   * refinement level.
   * @param _NY: total number of blocks in the y-direction, at the coarsest
   * refinement level.
   * @param _NZ: total number of blocks in the z-direction, at the coarsest
   * refinement level.
   * @param _maxextent: maximum extent of the simulation (largest side of the
   * rectangular domain).
   * @param _levelStart: refinement level where all allocated GridBlocks will be
   * @param _levelMax: maximum refinement level allowed
   * @param AllocateBlocks: true if GridBlocks should be allocated (false if
   * they are allocated by a derived class)
   * @param a_xperiodic: true if the domain is periodic in the x-direction
   * @param a_yperiodic: true if the domain is periodic in the y-direction
   * @param a_zperiodic: true if the domain is periodic in the z-direction
   */
  Grid(const unsigned int _NX, const unsigned int _NY = 1,
       const unsigned int _NZ = 1, const double _maxextent = 1,
       const unsigned int _levelStart = 0, const unsigned int _levelMax = 1,
       const bool AllocateBlocks = true, const bool a_xperiodic = true,
       const bool a_yperiodic = true, const bool a_zperiodic = true)
      : NX(_NX), NY(_NY), NZ(_NZ), maxextent(_maxextent), levelMax(_levelMax),
        levelStart(_levelStart), xperiodic(a_xperiodic), yperiodic(a_yperiodic),
        zperiodic(a_zperiodic) {
    BlockInfo dummy;
#if DIMENSION == 3
    const int nx = dummy.blocks_per_dim(0, NX, NY, NZ);
    const int ny = dummy.blocks_per_dim(1, NX, NY, NZ);
    const int nz = dummy.blocks_per_dim(2, NX, NY, NZ);
#else
    const int nx = dummy.blocks_per_dim(0, NX, NY);
    const int ny = dummy.blocks_per_dim(1, NX, NY);
    const int nz = 1;
#endif
    const int lvlMax = dummy.levelMax(levelMax);

    for (int m = 0; m < lvlMax; m++) {
      const int TwoPower = 1 << m;
      const long long Ntot = nx * ny * nz * pow(TwoPower, DIMENSION);
      if (m == 0)
        level_base.push_back(Ntot);
      if (m > 0)
        level_base.push_back(level_base[m - 1] + Ntot);
    }
    if (AllocateBlocks)
      _alloc();
  }

  /// Destructor
  virtual ~Grid() { _deallocAll(); }

  /// Returns GridBlock at level 'm' with Z-index 'n'
  virtual Block *avail(const int m, const long long n) {
    return (Block *)getBlockInfoAll(m, n).ptrBlock;
  }

  /// Returns MPI ranks of this Grid
  virtual int rank() const { return 0; }

  /**Given two vectors with the SFC coordinate (Z) and the level of each block,
   * this function will erase the current structure of the grid and create a new
   * one, with the given blocks. This is used when reading data from file
   * (possibly to restart) or when initializing the simulation.*/
  virtual void initialize_blocks(const std::vector<long long> &blocksZ,
                                 const std::vector<short int> &blockslevel) {
    _deallocAll();

    for (size_t i = 0; i < blocksZ.size(); i++) {
      const int level = blockslevel[i];
      const long long Z = blocksZ[i];

      _alloc(level, Z);
      Tree(level, Z).setrank(rank());

#if DIMENSION == 3
      int p[3];
      BlockInfo::inverse(Z, level, p[0], p[1], p[2]);
      if (level < levelMax - 1)
        for (int k1 = 0; k1 < 2; k1++)
          for (int j1 = 0; j1 < 2; j1++)
            for (int i1 = 0; i1 < 2; i1++) {
              const long long nc = getZforward(level + 1, 2 * p[0] + i1,
                                               2 * p[1] + j1, 2 * p[2] + k1);
              Tree(level + 1, nc).setCheckCoarser();
            }
      if (level > 0) {
        const long long nf =
            getZforward(level - 1, p[0] / 2, p[1] / 2, p[2] / 2);
        Tree(level - 1, nf).setCheckFiner();
      }
#else
      int p[2];
      BlockInfo::inverse(Z, level, p[0], p[1]);
      if (level < levelMax - 1)
        for (int j1 = 0; j1 < 2; j1++)
          for (int i1 = 0; i1 < 2; i1++) {
            const long long nc =
                getZforward(level + 1, 2 * p[0] + i1, 2 * p[1] + j1);
            Tree(level + 1, nc).setCheckCoarser();
          }
      if (level > 0) {
        const long long nf = getZforward(level - 1, p[0] / 2, p[1] / 2);
        Tree(level - 1, nf).setCheckFiner();
      }
#endif
    }
    FillPos();
    UpdateFluxCorrection = true;
    UpdateGroups = true;
  }

#if DIMENSION == 3
  /// Returns Z-index of GridBlock with indices ijk (ix,iy,iz) at level 'level'
  long long getZforward(const int level, const int i, const int j,
                        const int k) const {
    const int TwoPower = 1 << level;
    const int ix = (i + TwoPower * NX) % (NX * TwoPower);
    const int iy = (j + TwoPower * NY) % (NY * TwoPower);
    const int iz = (k + TwoPower * NZ) % (NZ * TwoPower);
    return BlockInfo::forward(level, ix, iy, iz);
  }

  /// Returns GridBlock with indices ijk (ix,iy,iz) at level 'm'
  Block *avail1(const int ix, const int iy, const int iz, const int m) {
    const long long n = getZforward(m, ix, iy, iz);
    return avail(m, n);
  }

#else // DIMENSION = 2

  /// Returns Z-index of GridBlock with indices ij (ix,iy) at level 'level'
  long long getZforward(const int level, const int i, const int j) const {
    const int TwoPower = 1 << level;
    const int ix = (i + TwoPower * NX) % (NX * TwoPower);
    const int iy = (j + TwoPower * NY) % (NY * TwoPower);
    return BlockInfo::forward(level, ix, iy);
  }

  /// Returns GridBlock with indices ij (ix,iy) at level 'm'
  Block *avail1(const int ix, const int iy, const int m) {
    const long long n = getZforward(m, ix, iy);
    return avail(m, n);
  }

#endif

  /// Used to iterate though all blocks (ID=0,...,m_vInfo.size()-1)
  Block &operator()(const long long ID) {
    return *(Block *)m_vInfo[ID].ptrBlock;
  }

  /// Returns the number of blocks at refinement level 0
  std::array<int, 3> getMaxBlocks() const { return {NX, NY, NZ}; }

  /// Returns the number of blocks at refinement level 'levelMax-1'
  std::array<int, 3> getMaxMostRefinedBlocks() const {
    return {
        NX << (levelMax - 1),
        NY << (levelMax - 1),
        DIMENSION == 3 ? (NZ << (levelMax - 1)) : 1,
    };
  }

  /// Returns the number of grid points at refinement level 'levelMax-1'
  std::array<int, 3> getMaxMostRefinedCells() const {
    const auto b = getMaxMostRefinedBlocks();
    return {b[0] * Block::sizeX, b[1] * Block::sizeY, b[2] * Block::sizeZ};
  }

  /// Returns the maximum refinement level allowed
  inline int getlevelMax() const { return levelMax; }

  /**
   * Access BlockInfo at level m with Space-Filling-Curve coordinate n.
   * If the BlockInfo has not been allocated (not found in the
   * std::unordered_map), allocate it as well.
   */
  BlockInfo &getBlockInfoAll(const int m, const long long n) {
    const long long aux = level_base[m] + n;
    const auto retval = BlockInfoAll.find(aux);
    if (retval != BlockInfoAll.end()) {
      return *retval->second;
    } else {
#ifndef CUBISM_USE_ONETBB
#pragma omp critical
#endif
      {
        const auto retval1 = BlockInfoAll.find(aux);
        if (retval1 == BlockInfoAll.end()) {
          BlockInfo *dumm = new BlockInfo();
          const int TwoPower = 1 << m;
          const double h0 = (maxextent / std::max(NX * Block::sizeX,
                                                  std::max(NY * Block::sizeY,
                                                           NZ * Block::sizeZ)));
          const double h = h0 / TwoPower;
          double origin[3];
          int i, j, k;
#if DIMENSION == 3
          BlockInfo::inverse(n, m, i, j, k);
#else
          BlockInfo::inverse(n, m, i, j);
          k = 0;
#endif
          origin[0] = i * Block::sizeX * h;
          origin[1] = j * Block::sizeY * h;
          origin[2] = k * Block::sizeZ * h;
          dumm->setup(m, h, origin, n);
          BlockInfoAll[aux] = dumm;
        }
      }
      return getBlockInfoAll(m, n);
    }
  }

  /// Returns the vector of BlockInfos of this Grid
  std::vector<BlockInfo> &getBlocksInfo() { return m_vInfo; }

  /// Returns the vector of BlockInfos of this Grid
  const std::vector<BlockInfo> &getBlocksInfo() const { return m_vInfo; }

  /// Returns the total number of MPI processes
  virtual int get_world_size() const { return 1; }

  /// Does nothing for a single rank (no MPI)
  virtual void UpdateBoundary(bool clean = false) {}

  /// Used to create BlockGroups, when the Grid is to be dumped to a file
  void UpdateMyGroups() {
    /*
     * This function is used before dumping the Grid. It groups adjacent blocks
     * of the same resolution (and owned by the same MPI rank) to BlockGroups
     * that will be dumped as a collection of rectangular uniform grids.
     */

    // if (!UpdateGroups) return; //TODO : does not work for CUP2D
    if (rank() == 0)
      std::cout << "Updating groups..." << std::endl;

    const unsigned int nX = BlockType::sizeX;
    const unsigned int nY = BlockType::sizeY;
    const size_t Ngrids = getBlocksInfo().size();
    const auto &MyInfos = getBlocksInfo();
    UpdateGroups = false;
    MyGroups.clear();
    std::vector<bool> added(MyInfos.size(), false);

#if DIMENSION == 3
    const unsigned int nZ = BlockType::sizeZ;
    for (unsigned int m = 0; m < Ngrids; m++) {
      const BlockInfo &I = MyInfos[m];

      if (added[I.blockID])
        continue;
      added[I.blockID] = true;
      BlockGroup newGroup;

      newGroup.level = I.level;
      newGroup.h = I.h;
      newGroup.Z.push_back(I.Z);

      const int base[3] = {I.index[0], I.index[1], I.index[2]};
      int i_off[6] = {};
      bool ready_[6] = {};

      int d = 0;
      auto blk = getMaxBlocks();
      do {
        if (ready_[d] == false) {
          bool valid = true;
          i_off[d]++;
          const int i0 =
              (d < 3) ? (base[d] - i_off[d]) : (base[d - 3] + i_off[d]);
          const int d0 = (d < 3) ? (d) % 3 : (d - 3) % 3;
          const int d1 = (d < 3) ? (d + 1) % 3 : (d - 3 + 1) % 3;
          const int d2 = (d < 3) ? (d + 2) % 3 : (d - 3 + 2) % 3;

          for (int i2 = base[d2] - i_off[d2]; i2 <= base[d2] + i_off[d2 + 3];
               i2++)
            for (int i1 = base[d1] - i_off[d1]; i1 <= base[d1] + i_off[d1 + 3];
                 i1++) {
              if (valid == false)
                break;

              if (i0 < 0 || i1 < 0 || i2 < 0 ||
                  i0 >= blk[d0] * (1 << I.level) ||
                  i1 >= blk[d1] * (1 << I.level) ||
                  i2 >= blk[d2] * (1 << I.level)) {
                valid = false;
                break;
              }
              long long n;
              if (d == 0 || d == 3)
                n = getZforward(I.level, i0, i1, i2);
              else if (d == 1 || d == 4)
                n = getZforward(I.level, i2, i0, i1);
              else /*if (d==2||d==5)*/
                n = getZforward(I.level, i1, i2, i0);

              if (Tree(I.level, n).rank() != rank()) {
                valid = false;
                break;
              }
              if (added[getBlockInfoAll(I.level, n).blockID] == true) {
                valid = false;
              }
            }

          if (valid == false) {
            i_off[d]--;
            ready_[d] = true;
          } else {
            for (int i2 = base[d2] - i_off[d2]; i2 <= base[d2] + i_off[d2 + 3];
                 i2++)
              for (int i1 = base[d1] - i_off[d1];
                   i1 <= base[d1] + i_off[d1 + 3]; i1++) {
                long long n;
                if (d == 0 || d == 3)
                  n = getZforward(I.level, i0, i1, i2);
                else if (d == 1 || d == 4)
                  n = getZforward(I.level, i2, i0, i1);
                else /*if (d==2||d==5)*/
                  n = getZforward(I.level, i1, i2, i0);
                newGroup.Z.push_back(n);
                added[getBlockInfoAll(I.level, n).blockID] = true;
              }
          }
        }
        d = (d + 1) % 6;
      } while (ready_[0] == false || ready_[1] == false || ready_[2] == false ||
               ready_[3] == false || ready_[4] == false || ready_[5] == false);

      const int ix_min = base[0] - i_off[0];
      const int iy_min = base[1] - i_off[1];
      const int iz_min = base[2] - i_off[2];
      const int ix_max = base[0] + i_off[3];
      const int iy_max = base[1] + i_off[4];
      const int iz_max = base[2] + i_off[5];

      long long n_base = getZforward(I.level, ix_min, iy_min, iz_min);

      newGroup.i_min[0] = ix_min;
      newGroup.i_min[1] = iy_min;
      newGroup.i_min[2] = iz_min;

      newGroup.i_max[0] = ix_max;
      newGroup.i_max[1] = iy_max;
      newGroup.i_max[2] = iz_max;

      const BlockInfo &info = getBlockInfoAll(I.level, n_base);
      newGroup.origin[0] = info.origin[0];
      newGroup.origin[1] = info.origin[1];
      newGroup.origin[2] = info.origin[2];

      newGroup.NXX = (newGroup.i_max[0] - newGroup.i_min[0] + 1) * nX + 1;
      newGroup.NYY = (newGroup.i_max[1] - newGroup.i_min[1] + 1) * nY + 1;
      newGroup.NZZ = (newGroup.i_max[2] - newGroup.i_min[2] + 1) * nZ + 1;

      MyGroups.push_back(newGroup);
    }
#else
    for (unsigned int m = 0; m < Ngrids; m++) {
      const BlockInfo &I = MyInfos[m];

      if (added[I.blockID])
        continue;
      added[I.blockID] = true;
      BlockGroup newGroup;

      newGroup.level = I.level;
      newGroup.h = I.h;
      newGroup.Z.push_back(I.Z);

      const int base[3] = {I.index[0], I.index[1], 0}; // I.index[2]
      int i_off[4] = {};
      bool ready_[4] = {};

      int d = 0;
      auto blk = getMaxBlocks();
      do {
        if (ready_[d] == false) {
          bool valid = true;
          i_off[d]++;
          const int i0 =
              (d < 2) ? (base[d] - i_off[d]) : (base[d - 2] + i_off[d]);
          const int d0 = (d < 2) ? (d) % 2 : (d - 2) % 2;
          const int d1 = (d < 2) ? (d + 1) % 2 : (d - 2 + 1) % 2;

          for (int i1 = base[d1] - i_off[d1]; i1 <= base[d1] + i_off[d1 + 2];
               i1++) {
            if (valid == false)
              break;

            if (i0 < 0 || i1 < 0 || i0 >= blk[d0] * (1 << I.level) ||
                i1 >= blk[d1] * (1 << I.level)) {
              valid = false;
              break;
            }
            long long n = (d == 0 || d == 2) ? getZforward(I.level, i0, i1)
                                             : getZforward(I.level, i1, i0);

            if (Tree(I.level, n).rank() != rank()) {
              valid = false;
              break;
            }
            if (added[getBlockInfoAll(I.level, n).blockID] == true) {
              valid = false;
            }
          }

          if (valid == false) {
            i_off[d]--;
            ready_[d] = true;
          } else {
            for (int i1 = base[d1] - i_off[d1]; i1 <= base[d1] + i_off[d1 + 2];
                 i1++) {
              long long n = (d == 0 || d == 2) ? getZforward(I.level, i0, i1)
                                               : getZforward(I.level, i1, i0);
              newGroup.Z.push_back(n);
              added[getBlockInfoAll(I.level, n).blockID] = true;
            }
          }
        }
        d = (d + 1) % 4;
      } while (ready_[0] == false || ready_[1] == false || ready_[2] == false ||
               ready_[3] == false);

      const int ix_min = base[0] - i_off[0];
      const int iy_min = base[1] - i_off[1];
      const int iz_min = 0; // base[2] - i_off[2];
      const int ix_max = base[0] + i_off[2];
      const int iy_max = base[1] + i_off[3];
      const int iz_max = 0; // base[2] + i_off[5];

      long long n_base = getZforward(I.level, ix_min, iy_min);

      newGroup.i_min[0] = ix_min;
      newGroup.i_min[1] = iy_min;
      newGroup.i_min[2] = iz_min;

      newGroup.i_max[0] = ix_max;
      newGroup.i_max[1] = iy_max;
      newGroup.i_max[2] = iz_max;

      const BlockInfo &info = getBlockInfoAll(I.level, n_base);
      newGroup.origin[0] = info.origin[0];
      newGroup.origin[1] = info.origin[1];
      newGroup.origin[2] = info.origin[2];

      newGroup.NXX = (newGroup.i_max[0] - newGroup.i_min[0] + 1) * nX + 1;
      newGroup.NYY = (newGroup.i_max[1] - newGroup.i_min[1] + 1) * nY + 1;
      newGroup.NZZ = 2; //(newGroup.i_max[2] - newGroup.i_min[2] + 1)*nZ + 1;

      MyGroups.push_back(newGroup);
    }
#endif
  }
};

} // namespace cubism
/*
 *  GridMPI.h
 *
 *  Created by Michalis Chatzimanolakis
 *  Copyright 2020 ETH Zurich. All rights reserved.
 *
 */

namespace cubism {

/**
 * @brief Describes a stencil of points.
 *
 * This struct is used by BlockLab to determine what halo cells need to be
 * communicated between different GridBlocks. For a gridpoint (i,j,k) the
 * gridpoints included in the stencil are points (i+ix,j+iy,k+iz), where
 * ix,iy,iz are determined by the stencil starts and ends.
 */
struct StencilInfo {
  int sx; ///< start of stencil in the x-direction (sx <= ix)
  int sy; ///< start of stencil in the y-direction (sy <= iy)
  int sz; ///< start of stencil in the z-direction (sz <= iz)
  int ex; ///< end of stencil (+1) in the x-direction (ix < ex)
  int ey; ///< end of stencil (+1) in the y-direction (iy < ey)
  int ez; ///< end of stencil (+1) in the z-direction (iz < ez)
  std::vector<int>
      selcomponents; ///< Components ('members') of Element that will be used
  bool tensorial;    ///< if false, stencil only includes points with
                     ///< |ix|+|iy|+|iz| <= 1

  /// Empty constructor.
  StencilInfo() {}

  /// Constructor
  StencilInfo(int _sx, int _sy, int _sz, int _ex, int _ey, int _ez,
              bool _tensorial, const std::vector<int> &components)
      : sx(_sx), sy(_sy), sz(_sz), ex(_ex), ey(_ey), ez(_ez),
        selcomponents(components), tensorial(_tensorial) {
    assert(selcomponents.size() > 0);

    if (!isvalid()) {
      std::cout << "Stencilinfo instance not valid. Aborting\n";
      abort();
    }
  }

  /// Copy constructor.
  StencilInfo(const StencilInfo &c)
      : sx(c.sx), sy(c.sy), sz(c.sz), ex(c.ex), ey(c.ey), ez(c.ez),
        selcomponents(c.selcomponents), tensorial(c.tensorial) {}

  /// Return a vector with all integers that make up this StencilInfo.
  std::vector<int> _all() const {
    int extra[] = {sx, sy, sz, ex, ey, ez, (int)tensorial};
    std::vector<int> all(selcomponents);
    all.insert(all.end(), extra, extra + sizeof(extra) / sizeof(int));

    return all;
  }

  /// Check if one stencil is contained in another.
  bool operator<(StencilInfo s) const {
    std::vector<int> me = _all(), you = s._all();

    const int N = std::min(me.size(), you.size());

    for (int i = 0; i < N; ++i)
      if (me[i] < you[i])
        return true;
      else if (me[i] > you[i])
        return false;

    return me.size() < you.size();
  }

  /// Check if the ends are smaller than the starts of this stencil.
  bool isvalid() const {
    const bool not0 = selcomponents.size() == 0;
    const bool not1 = sx > 0 || ex <= 0 || sx > ex;
    const bool not2 = sy > 0 || ey <= 0 || sy > ey;
    const bool not3 = sz > 0 || ez <= 0 || sz > ez;

    return !(not0 || not1 || not2 || not3);
  }
};

} // namespace cubism

namespace cubism {

template <typename Real>
inline void pack(const Real *const srcbase, Real *const dst,
                 const unsigned int gptfloats, int *selected_components,
                 const int ncomponents, const int xstart, const int ystart,
                 const int zstart, const int xend, const int yend,
                 const int zend, const int BSX, const int BSY) {
  if (gptfloats == 1) {
    const int mod = (xend - xstart) % 4;
    for (int idst = 0, iz = zstart; iz < zend; ++iz)
      for (int iy = ystart; iy < yend; ++iy) {
        for (int ix = xstart; ix < xend - mod; ix += 4, idst += 4) {
          dst[idst + 0] = srcbase[ix + 0 + BSX * (iy + BSY * iz)];
          dst[idst + 1] = srcbase[ix + 1 + BSX * (iy + BSY * iz)];
          dst[idst + 2] = srcbase[ix + 2 + BSX * (iy + BSY * iz)];
          dst[idst + 3] = srcbase[ix + 3 + BSX * (iy + BSY * iz)];
        }
        for (int ix = xend - mod; ix < xend; ix++, idst++) {
          dst[idst] = srcbase[ix + BSX * (iy + BSY * iz)];
        }
      }
  } else {
    for (int idst = 0, iz = zstart; iz < zend; ++iz)
      for (int iy = ystart; iy < yend; ++iy)
        for (int ix = xstart; ix < xend; ++ix) {
          const Real *src = srcbase + gptfloats * (ix + BSX * (iy + BSY * iz));
          for (int ic = 0; ic < ncomponents; ic++, idst++)
            dst[idst] = src[selected_components[ic]];
        }
  }
}

template <typename Real>
inline void unpack_subregion(
    const Real *const pack, Real *const dstbase, const unsigned int gptfloats,
    const int *const selected_components, const int ncomponents,
    const int srcxstart, const int srcystart, const int srczstart, const int LX,
    const int LY, const int dstxstart, const int dstystart, const int dstzstart,
    const int dstxend, const int dstyend, const int dstzend, const int xsize,
    const int ysize, const int zsize) {
  if (gptfloats == 1) {
    const int mod = (dstxend - dstxstart) % 4;
    for (int zd = dstzstart; zd < dstzend; ++zd)
      for (int yd = dstystart; yd < dstyend; ++yd) {
        const int offset = -dstxstart + srcxstart +
                           LX * (yd - dstystart + srcystart +
                                 LY * (zd - dstzstart + srczstart));
        const int offset_dst = xsize * (yd + ysize * zd);
        for (int xd = dstxstart; xd < dstxend - mod; xd += 4) {
          dstbase[xd + 0 + offset_dst] = pack[xd + 0 + offset];
          dstbase[xd + 1 + offset_dst] = pack[xd + 1 + offset];
          dstbase[xd + 2 + offset_dst] = pack[xd + 2 + offset];
          dstbase[xd + 3 + offset_dst] = pack[xd + 3 + offset];
        }
        for (int xd = dstxend - mod; xd < dstxend; ++xd) {
          dstbase[xd + offset_dst] = pack[xd + offset];
        }
      }
  } else {
    for (int zd = dstzstart; zd < dstzend; ++zd)
      for (int yd = dstystart; yd < dstyend; ++yd)
        for (int xd = dstxstart; xd < dstxend; ++xd) {
          Real *const dst =
              dstbase + gptfloats * (xd + xsize * (yd + ysize * zd));
          const Real *src =
              pack + ncomponents * (xd - dstxstart + srcxstart +
                                    LX * (yd - dstystart + srcystart +
                                          LY * (zd - dstzstart + srczstart)));
          for (int c = 0; c < ncomponents; ++c)
            dst[selected_components[c]] = src[c];
        }
  }
}

} // namespace cubism

namespace cubism {

/** \brief Auxiliary class for SynchronizerMPI_AMR; similar to std::vector
 * however, the stored data does not decrease in size, it can only increase (the
 * use of this class instead of std::vector in AMR_Synchronizer resulted in
 * faster performance). */
template <typename T> class GrowingVector {
  size_t pos;
  size_t s;

public:
  std::vector<T> v;
  GrowingVector() {
    pos = 0;
    s = 0;
  }
  GrowingVector(size_t size) { resize(size); }
  GrowingVector(size_t size, T value) { resize(size, value); }

  void resize(size_t new_size, T value) {
    v.resize(new_size, value);
    s = new_size;
  }
  void resize(size_t new_size) {
    v.resize(new_size);
    s = new_size;
  }

  size_t size() { return s; }

  void clear() {
    pos = 0;
    s = 0;
  }

  void push_back(T value) {
    if (pos < v.size())
      v[pos] = value;
    else
      v.push_back(value);
    pos++;
    s++;
  }

  T *data() { return v.data(); }

  T &operator[](size_t i) { return v[i]; }

  T &back() { return v[pos - 1]; }
  typename std::vector<T>::iterator begin() { return v.begin(); }
  typename std::vector<T>::iterator end() { return v.begin() + pos; }

  void EraseAll() {
    v.clear();
    pos = 0;
    s = 0;
  }

  ~GrowingVector() { v.clear(); }
};

/** \brief Auxiliary struct for SynchronizerMPI_AMR; describes how two adjacent
 * blocks touch.*/
struct Interface {
  BlockInfo *infos[2]; ///< the two blocks of the interface
  int icode[2]; ///< Two integers from 0 to 26. Each integer can be decoded to a
                ///< 3-digit number ABC. icode[0] = 1-10 (A=1,B=-1,C=0) means
                ///< Block 1 is at the +x,-y side of Block 0.
  bool CoarseStencil; ///< =true if the blocks need to exchange cells of their
                      ///< parent blocks
  bool ToBeKept; ///< false if this inteface is a subset of another inteface
                 ///< that will be sent anyway
  int dis;       ///< auxiliary variable

  /// Class constructor
  Interface(BlockInfo &i0, BlockInfo &i1, const int a_icode0,
            const int a_icode1) {
    infos[0] = &i0;
    infos[1] = &i1;
    icode[0] = a_icode0;
    icode[1] = a_icode1;
    CoarseStencil = false;
    ToBeKept = true;
    dis = 0;
  }

  bool operator<(const Interface &other) const {
    if (infos[0]->blockID_2 == other.infos[0]->blockID_2) {
      if (icode[0] == other.icode[0]) {
        if (infos[1]->blockID_2 == other.infos[1]->blockID_2) {
          return (icode[1] < other.icode[1]);
        }
        return (infos[1]->blockID_2 < other.infos[1]->blockID_2);
      }
      return (icode[0] < other.icode[0]);
    }
    return (infos[0]->blockID_2 < other.infos[0]->blockID_2);
  }
};

/** Auxiliary struct for SynchronizerMPI_AMR; similar to StencilInfo.
 * It is possible that the halo cells needed by two or more blocks overlap. To
 * avoid sending the same data twice, this struct has the option to keep track
 * of other MyRanges that are contained in it and do not need to be
 * sent/received.
 */
struct MyRange {
  std::vector<int>
      removedIndices;  ///< keep track of all 'index' from other MyRange
                       ///< instances that are contained in this one
  int index;           ///< index of this instance of MyRange
  int sx;              ///< stencil start in x-direction
  int sy;              ///< stencil start in y-direction
  int sz;              ///< stencil start in z-direction
  int ex;              ///< stencil end in x-direction
  int ey;              ///< stencil end in y-direction
  int ez;              ///< stencil end in z-direction
  bool needed{true};   ///< set to false if this MyRange is contained in another
  bool avg_down{true}; ///< set to true if gridpoints of this MyRange will be
                       ///< averaged down for coarse stencil interpolation

  /// check if another MyRange is contained here
  bool contains(MyRange &r) const {
    if (avg_down != r.avg_down)
      return false;
    int V = (ez - sz) * (ey - sy) * (ex - sx);
    int Vr = (r.ez - r.sz) * (r.ey - r.sy) * (r.ex - r.sx);
    return (sx <= r.sx && r.ex <= ex) && (sy <= r.sy && r.ey <= ey) &&
           (sz <= r.sz && r.ez <= ez) && (Vr < V);
  }

  /// keep track of indices of other MyRanges that are contained here
  void Remove(const MyRange &other) {
    size_t s = removedIndices.size();
    removedIndices.resize(s + other.removedIndices.size());
    for (size_t i = 0; i < other.removedIndices.size(); i++)
      removedIndices[s + i] = other.removedIndices[i];
  }
};

/** Auxiliary struct for SynchronizerMPI_AMR; Meta-data of buffers sent among
 * processes. Data is received in one contiguous buffer. This struct helps
 * unpack the buffer and put data in the correct locations.
 */
struct UnPackInfo {
  int offset;    ///< Offset in the buffer where the data related to this
                 ///< UnPackInfo starts.
  int lx;        ///< Total size of data in x-direction
  int ly;        ///< Total size of data in y-direction
  int lz;        ///< Total size of data in z-direction
  int srcxstart; ///< Where in x-direction to start receiving data
  int srcystart; ///< Where in y-direction to start receiving data
  int srczstart; ///< Where in z-direction to start receiving data
  int LX;
  int LY;
  int CoarseVersionOffset; ///< Offset in the buffer where the coarsened data
                           ///< related to this UnPackInfo starts.
  int CoarseVersionLX;
  int CoarseVersionLY;
  int CoarseVersionsrcxstart; ///< Where in x-direction to start receiving
                              ///< coarsened data
  int CoarseVersionsrcystart; ///< Where in y-direction to start receiving
                              ///< coarsened data
  int CoarseVersionsrczstart; ///< Where in z-direction to start receiving
                              ///< coarsened data
  int level;                  ///< refinement level of data
  int icode; ///< Integer from 0 to 26, can be decoded to a 3-digit number ABC.
             ///< icode = 1-10 (A=1,B=-1,C=0) means Block 1 is at the +x,-y side
             ///< of Block 0.
  int rank;  ///< rank from which this data is received
  int index_0;          ///< index of Block in x-direction that sent this data
  int index_1;          ///< index of Block in y-direction that sent this data
  int index_2;          ///< index of Block in z-direction that sent this data
  long long IDreceiver; ///< unique blockID2 of receiver
};

/** Auxiliary struct for SynchronizerMPI_AMR; keeps track of stencil and range
 * sizes that need to be sent/received. For a block in 3D, there are a total of
 * 26 possible directions that might require halo cells. There are also four
 * types of halo cells to exchange, based on the refinement level of the two
 *  neighboring blocks: 1)same level 2)coarse-fine 3)fine-coarse 4)same level
 * that also need to exchange averaged down data, in order to perform
 * coarse-fine interpolation for other blocks. This class creates 4 x 26
 * (actually 4 x 27) MyRange instances, based on a given StencilInfo.
 */
struct StencilManager {
  const StencilInfo stencil; ///< stencil to send/receive
  const StencilInfo
      Cstencil; ///< stencil used by BlockLab for coarse-fine interpolation
  int nX;       ///< Block size if x-direction
  int nY;       ///< Block size if y-direction
  int nZ;       ///< Block size if z-direction
  int sLength[3 * 27 * 3]; ///< Length of all possible stencils to send/receive
  std::array<MyRange, 3 * 27>
      AllStencils;      ///< All possible stencils to send/receive
  MyRange Coarse_Range; ///< range for Cstencil

  /// Class constructor
  StencilManager(StencilInfo a_stencil, StencilInfo a_Cstencil, int a_nX,
                 int a_nY, int a_nZ)
      : stencil(a_stencil), Cstencil(a_Cstencil), nX(a_nX), nY(a_nY), nZ(a_nZ) {
    const int sC[3] = {(stencil.sx - 1) / 2 + Cstencil.sx,
                       (stencil.sy - 1) / 2 + Cstencil.sy,
                       (stencil.sz - 1) / 2 + Cstencil.sz};
    const int eC[3] = {stencil.ex / 2 + Cstencil.ex,
                       stencil.ey / 2 + Cstencil.ey,
                       stencil.ez / 2 + Cstencil.ez};

    for (int icode = 0; icode < 27; icode++) {
      const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                           (icode / 9) % 3 - 1};

      // This also works for DIMENSION=2 and code[2]=0
      // Same level sender and receiver
      MyRange &range0 = AllStencils[icode];
      range0.sx = code[0] < 1 ? (code[0] < 0 ? nX + stencil.sx : 0) : 0;
      range0.sy = code[1] < 1 ? (code[1] < 0 ? nY + stencil.sy : 0) : 0;
      range0.sz = code[2] < 1 ? (code[2] < 0 ? nZ + stencil.sz : 0) : 0;
      range0.ex = code[0] < 1 ? nX : stencil.ex - 1;
      range0.ey = code[1] < 1 ? nY : stencil.ey - 1;
      range0.ez = code[2] < 1 ? nZ : stencil.ez - 1;
      sLength[3 * icode + 0] = range0.ex - range0.sx;
      sLength[3 * icode + 1] = range0.ey - range0.sy;
      sLength[3 * icode + 2] = range0.ez - range0.sz;

      // Fine sender, coarse receiver
      // Fine sender just needs to send "double" the stencil, so that what it
      // sends gets averaged down
      MyRange &range1 = AllStencils[icode + 27];
      range1.sx = code[0] < 1 ? (code[0] < 0 ? nX + 2 * stencil.sx : 0) : 0;
      range1.sy = code[1] < 1 ? (code[1] < 0 ? nY + 2 * stencil.sy : 0) : 0;
      range1.sz = code[2] < 1 ? (code[2] < 0 ? nZ + 2 * stencil.sz : 0) : 0;
      range1.ex = code[0] < 1 ? nX : 2 * (stencil.ex - 1);
      range1.ey = code[1] < 1 ? nY : 2 * (stencil.ey - 1);
      range1.ez = code[2] < 1 ? nZ : 2 * (stencil.ez - 1);
      sLength[3 * (icode + 27) + 0] = (range1.ex - range1.sx) / 2;
      sLength[3 * (icode + 27) + 1] = (range1.ey - range1.sy) / 2;
#if DIMENSION == 3
      sLength[3 * (icode + 27) + 2] = (range1.ez - range1.sz) / 2;
#else
      sLength[3 * (icode + 27) + 2] = 1;
#endif

      // Coarse sender, fine receiver
      // Coarse sender just needs to send "half" the stencil plus extra cells
      // for coarse-fine interpolation
      MyRange &range2 = AllStencils[icode + 2 * 27];
      range2.sx = code[0] < 1 ? (code[0] < 0 ? nX / 2 + sC[0] : 0) : 0;
      range2.sy = code[1] < 1 ? (code[1] < 0 ? nY / 2 + sC[1] : 0) : 0;
      range2.ex = code[0] < 1 ? nX / 2 : eC[0] - 1;
      range2.ey = code[1] < 1 ? nY / 2 : eC[1] - 1;
#if DIMENSION == 3
      range2.sz = code[2] < 1 ? (code[2] < 0 ? nZ / 2 + sC[2] : 0) : 0;
      range2.ez = code[2] < 1 ? nZ / 2 : eC[2] - 1;
#else
      range2.sz = 0;
      range2.ez = 1;
#endif
      sLength[3 * (icode + 2 * 27) + 0] = range2.ex - range2.sx;
      sLength[3 * (icode + 2 * 27) + 1] = range2.ey - range2.sy;
      sLength[3 * (icode + 2 * 27) + 2] = range2.ez - range2.sz;
    }
  }

  /// Return stencil XxYxZ dimensions for Cstencil, based on integer icode
  void CoarseStencilLength(const int icode, int *L) const {
    L[0] = sLength[3 * (icode + 2 * 27) + 0];
    L[1] = sLength[3 * (icode + 2 * 27) + 1];
    L[2] = sLength[3 * (icode + 2 * 27) + 2];
  }

  /// Return stencil XxYxZ dimensions for Cstencil, based on integer icode and
  /// refinement level of sender/receiver
  void DetermineStencilLength(const int level_sender, const int level_receiver,
                              const int icode, int *L) {
    if (level_sender == level_receiver) {
      L[0] = sLength[3 * icode + 0];
      L[1] = sLength[3 * icode + 1];
      L[2] = sLength[3 * icode + 2];
    } else if (level_sender > level_receiver) {
      L[0] = sLength[3 * (icode + 27) + 0];
      L[1] = sLength[3 * (icode + 27) + 1];
      L[2] = sLength[3 * (icode + 27) + 2];
    } else {
      L[0] = sLength[3 * (icode + 2 * 27) + 0];
      L[1] = sLength[3 * (icode + 2 * 27) + 1];
      L[2] = sLength[3 * (icode + 2 * 27) + 2];
    }
  }

  /// Determine which stencil to send, based on interface type of two blocks
  MyRange &DetermineStencil(const Interface &f, bool CoarseVersion = false) {
    if (CoarseVersion) {
      AllStencils[f.icode[1] + 2 * 27].needed = true;
      return AllStencils[f.icode[1] + 2 * 27];
    } else {
      if (f.infos[0]->level == f.infos[1]->level) {
        AllStencils[f.icode[1]].needed = true;
        return AllStencils[f.icode[1]];
      } else if (f.infos[0]->level > f.infos[1]->level) {
        AllStencils[f.icode[1] + 27].needed = true;
        return AllStencils[f.icode[1] + 27];
      } else {
        Coarse_Range.needed = true;
        const int code[3] = {f.icode[1] % 3 - 1, (f.icode[1] / 3) % 3 - 1,
                             (f.icode[1] / 9) % 3 - 1};

        const int s[3] = {
            code[0] < 1
                ? (code[0] < 0 ? ((stencil.sx - 1) / 2 + Cstencil.sx) : 0)
                : nX / 2,
            code[1] < 1
                ? (code[1] < 0 ? ((stencil.sy - 1) / 2 + Cstencil.sy) : 0)
                : nY / 2,
            code[2] < 1
                ? (code[2] < 0 ? ((stencil.sz - 1) / 2 + Cstencil.sz) : 0)
                : nZ / 2};

        const int e[3] = {
            code[0] < 1 ? (code[0] < 0 ? 0 : nX / 2)
                        : nX / 2 + stencil.ex / 2 + Cstencil.ex - 1,
            code[1] < 1 ? (code[1] < 0 ? 0 : nY / 2)
                        : nY / 2 + stencil.ey / 2 + Cstencil.ey - 1,
            code[2] < 1 ? (code[2] < 0 ? 0 : nZ / 2)
                        : nZ / 2 + stencil.ez / 2 + Cstencil.ez - 1};

        const int base[3] = {(f.infos[1]->index[0] + code[0]) % 2,
                             (f.infos[1]->index[1] + code[1]) % 2,
                             (f.infos[1]->index[2] + code[2]) % 2};

        int Cindex_true[3];
        for (int d = 0; d < 3; d++)
          Cindex_true[d] = f.infos[1]->index[d] + code[d];

        int CoarseEdge[3];

        CoarseEdge[0] = (code[0] == 0) ? 0
                        : (((f.infos[1]->index[0] % 2 == 0) &&
                            (Cindex_true[0] > f.infos[1]->index[0])) ||
                           ((f.infos[1]->index[0] % 2 == 1) &&
                            (Cindex_true[0] < f.infos[1]->index[0])))
                            ? 1
                            : 0;
        CoarseEdge[1] = (code[1] == 0) ? 0
                        : (((f.infos[1]->index[1] % 2 == 0) &&
                            (Cindex_true[1] > f.infos[1]->index[1])) ||
                           ((f.infos[1]->index[1] % 2 == 1) &&
                            (Cindex_true[1] < f.infos[1]->index[1])))
                            ? 1
                            : 0;
        CoarseEdge[2] = (code[2] == 0) ? 0
                        : (((f.infos[1]->index[2] % 2 == 0) &&
                            (Cindex_true[2] > f.infos[1]->index[2])) ||
                           ((f.infos[1]->index[2] % 2 == 1) &&
                            (Cindex_true[2] < f.infos[1]->index[2])))
                            ? 1
                            : 0;

        Coarse_Range.sx = s[0] + std::max(code[0], 0) * nX / 2 +
                          (1 - abs(code[0])) * base[0] * nX / 2 - code[0] * nX +
                          CoarseEdge[0] * code[0] * nX / 2;
        Coarse_Range.sy = s[1] + std::max(code[1], 0) * nY / 2 +
                          (1 - abs(code[1])) * base[1] * nY / 2 - code[1] * nY +
                          CoarseEdge[1] * code[1] * nY / 2;
#if DIMENSION == 3
        Coarse_Range.sz = s[2] + std::max(code[2], 0) * nZ / 2 +
                          (1 - abs(code[2])) * base[2] * nZ / 2 - code[2] * nZ +
                          CoarseEdge[2] * code[2] * nZ / 2;
#else
        Coarse_Range.sz = 0;
#endif

        Coarse_Range.ex = e[0] + std::max(code[0], 0) * nX / 2 +
                          (1 - abs(code[0])) * base[0] * nX / 2 - code[0] * nX +
                          CoarseEdge[0] * code[0] * nX / 2;
        Coarse_Range.ey = e[1] + std::max(code[1], 0) * nY / 2 +
                          (1 - abs(code[1])) * base[1] * nY / 2 - code[1] * nY +
                          CoarseEdge[1] * code[1] * nY / 2;
#if DIMENSION == 3
        Coarse_Range.ez = e[2] + std::max(code[2], 0) * nZ / 2 +
                          (1 - abs(code[2])) * base[2] * nZ / 2 - code[2] * nZ +
                          CoarseEdge[2] * code[2] * nZ / 2;
#else
        Coarse_Range.ez = 1;
#endif

        return Coarse_Range;
      }
    }
  }

  /// Fix MyRange classes that contain other MyRange classes, in order to avoid
  /// sending the same data twice
  void __FixDuplicates(const Interface &f, const Interface &f_dup, int lx,
                       int ly, int lz, int lx_dup, int ly_dup, int lz_dup,
                       int &sx, int &sy, int &sz) {
    const BlockInfo &receiver = *f.infos[1];
    const BlockInfo &receiver_dup = *f_dup.infos[1];
    if (receiver.level >= receiver_dup.level) {
      int icode_dup = f_dup.icode[1];
      const int code_dup[3] = {icode_dup % 3 - 1, (icode_dup / 3) % 3 - 1,
                               (icode_dup / 9) % 3 - 1};
      sx = (lx == lx_dup || code_dup[0] != -1) ? 0 : lx - lx_dup;
      sy = (ly == ly_dup || code_dup[1] != -1) ? 0 : ly - ly_dup;
      sz = (lz == lz_dup || code_dup[2] != -1) ? 0 : lz - lz_dup;
    } else {
      MyRange &range = DetermineStencil(f);
      MyRange &range_dup = DetermineStencil(f_dup);
      sx = range_dup.sx - range.sx;
      sy = range_dup.sy - range.sy;
      sz = range_dup.sz - range.sz;
    }
  }

  /// Fix MyRange classes that contain other MyRange classes, in order to avoid
  /// sending the same data twice
  void __FixDuplicates2(const Interface &f, const Interface &f_dup, int &sx,
                        int &sy, int &sz) {
    if (f.infos[0]->level != f.infos[1]->level ||
        f_dup.infos[0]->level != f_dup.infos[1]->level)
      return;
    MyRange &range = DetermineStencil(f, true);
    MyRange &range_dup = DetermineStencil(f_dup, true);
    sx = range_dup.sx - range.sx;
    sy = range_dup.sy - range.sy;
    sz = range_dup.sz - range.sz;
  }
};

/** Auxiliary struct for SynchronizerMPI_AMR; stored a number of halo blocks
 * that received their halo cells from a particular set of ranks.
 */
struct HaloBlockGroup {
  std::vector<BlockInfo *> myblocks; ///< Halo blocks for this group.
  std::set<int> myranks;             ///< MPI ranks for this group.
  bool ready =
      false; ///< Check whether communication for this group has completed.
};

/**
 *  @brief Class responsible for halo cell exchange between different MPI
 * processes. This class works together with BlockLabMPI to fill the halo cells
 * needed for each GridBlock. To overlap communication and computation, it
 * distinguishes between 'inner' blocks and 'halo' blocks. Inner blocks do not
 * need halo cells from other MPI processes, so they can be immediately filled;
 * halo blocks are at the boundary of a rank and require cells owned by other
 * ranks. This class will initiate communication for halo blocks and will
 * provide an array with pointers to inner blocks. While waiting for
 * communication to complete, the user can operate on inner blocks, which allows
 * for communication-computation overlap.
 *
 *  An instance of this class is constructed by providing the StencilInfo (aka
 * the stencil) for a particular computation, in the class constructor. Then,
 * for a fixed mesh configuration, one call to '_Setup()' is required. This
 * identifies the boundaries of each rank, its neighbors and the types of
 * interfaces (faces/edges/corners) shared by two blocks that belong to two
 * different ranks. '_Setup()' will then have to be called again only when the
 * mesh changes (this call is done by the MeshAdaptation class).
 *
 *  To use this class and send/receive halo cells, the 'sync()' function needs
 * to be called. This initiates communication and MPI 'sends' and 'receives'.
 * Once called, the inner and halo blocks (with their halo cells) can be
 * accessed through 'avail_inner' and 'avail_halo'. Note that calling
 * 'avail_halo' will result in waiting time, for the communication of halo cells
 * to complete. Therefore, 'avail_inner' should be called first, while
 * communication is performed in the background. Once the inner blocks are
 * processed, 'avail_halo' should be used to process the outer/halo blocks.
 *
 *  @tparam Real: type of data to be sent/received (double/float etc.)
 *  @tparam TGrid: type of grid to operate on (should be GridMPI)
 */
template <typename Real, typename TGrid> class SynchronizerMPI_AMR {
  MPI_Comm comm; ///< MPI communicator, same as the communicator from 'grid'
  int rank;      ///< MPI process ID, same as the ID from 'grid'
  int size;      ///< total number of processes, same as number from 'grid'
  StencilInfo
      stencil; ///< stencil associated with kernel (advection,diffusion etc.)
  StencilInfo Cstencil; ///< stencil required to do coarse-fine interpolation
  TGrid *grid;          ///< grid which owns blocks that need ghost cells
  int nX;               ///< size of each block in x-direction
  int nY;               ///< size of each block in y-direction
  int nZ;               ///< size of each block in z-direction
  MPI_Datatype MPIREAL; ///< MPI datatype matching template parameter 'Real'

  std::vector<BlockInfo *>
      inner_blocks; ///< will contain inner blocks with loaded ghost cells
  std::vector<BlockInfo *>
      halo_blocks; ///< will contain outer blocks with loaded ghost cells

  std::vector<GrowingVector<Real>>
      send_buffer; ///< send_buffer[i] contains data to send to rank i
  std::vector<GrowingVector<Real>>
      recv_buffer; ///< recv_buffer[i] will receive data from rank i

  std::vector<MPI_Request>
      requests; ///< requests for non-blocking sends/receives

  std::vector<int> send_buffer_size; ///< sizes of send_buffer (communicated
                                     ///< before actual data)
  std::vector<int> recv_buffer_size; ///< sizes of recv_buffer (communicated
                                     ///< before actual data)

  std::set<int> Neighbors; ///< IDs of neighboring MPI processes

  GrowingVector<GrowingVector<UnPackInfo>>
      myunpacks; ///< vector of vectors of UnPackInfos; unpacks[i] contains all
                 ///< UnPackInfos needed for a block with halo_blockID=i

  StencilManager SM;

  const unsigned int
      gptfloats; ///< number of Reals (doubles/float) each Element from Grid has
  const int NC;  ///< number of components from each Element to send/receive

  /// meta-data for the parts of a particular block that will be sent to another
  /// rank
  struct PackInfo {
    Real *block; ///< Pointer to the first element of the block whose data will
                 ///< be sent
    Real *pack;  ///< Pointer to the buffer where the block's elements will be
                 ///< copied
    int sx; ///< Start of the block's subset that will be sent (in x-direction)
    int sy; ///< Start of the block's subset that will be sent (in y-direction)
    int sz; ///< Start of the block's subset that will be sent (in z-direction)
    int ex; ///< End of the block's subset that will be sent (in x-direction)
    int ey; ///< End of the block's subset that will be sent (in y-direction)
    int ez; ///< End of the block's subset that will be sent (in z-direction)
  };
  std::vector<GrowingVector<PackInfo>>
      send_packinfos; ///< vector of vectors of PackInfos; send_packinfos[i]
                      ///< contains all the PackInfos to send to rank i

  std::vector<GrowingVector<Interface>>
      send_interfaces; ///< vector of vectors of Interfaces; send_interfaces[i]
                       ///< contains all the Interfaces this rank will send to
                       ///< rank i
  std::vector<GrowingVector<Interface>>
      recv_interfaces; ///< vector of vectors of Interfaces; recv_interfaces[i]
                       ///< contains all the Interfaces this rank will receive
                       ///< from rank i

  std::vector<std::vector<int>>
      ToBeAveragedDown; ///< vector of vectors of Interfaces that need to be
                        ///< averaged down when sent

  bool use_averages; ///< if true, fine blocks average down their cells to
                     ///< provide halo cells for coarse blocks (2nd order
                     ///< accurate). If false, they perform a 3rd-order accurate
                     ///< interpolation instead (which is the accuracy needed to
                     ///< compute 2nd derivatives).

  std::unordered_map<std::string, HaloBlockGroup>
      mapofHaloBlockGroups; ///< Maps groups of ranks (encoded to strings) to
                            ///< groups of halo blocks for communication.

  std::unordered_map<int, MPI_Request *>
      mapofrequests; ///< Maps each request for communication to an integer

  /// Auxiliary struct used to avoid sending the same data twice
  struct DuplicatesManager {
    /// Auxiliary struct to detect and remove duplicate Interfaces
    struct cube // could be more efficient
    {
      GrowingVector<MyRange>
          compass[27]; ///< All possible MyRange stencil that will be exchanged

      void clear() {
        for (int i = 0; i < 27; i++)
          compass[i].clear();
      }

      cube() {}

      /// Returns the MyRange objects that will be kept
      std::vector<MyRange *> keepEl() {
        std::vector<MyRange *> retval;
        for (int i = 0; i < 27; i++)
          for (size_t j = 0; j < compass[i].size(); j++)
            if (compass[i][j].needed)
              retval.push_back(&compass[i][j]);

        return retval;
      }

      /// Will return the indices of the removed MyRange objects (in v)
      void __needed(std::vector<int> &v) {
        static constexpr std::array<int, 3> faces_and_edges[18] = {
            {0, 1, 1}, {2, 1, 1}, {1, 0, 1}, {1, 2, 1}, {1, 1, 0}, {1, 1, 2},

            {0, 0, 1}, {0, 2, 1}, {2, 0, 1}, {2, 2, 1}, {1, 0, 0}, {1, 0, 2},
            {1, 2, 0}, {1, 2, 2}, {0, 1, 0}, {0, 1, 2}, {2, 1, 0}, {2, 1, 2}};

        for (auto &f : faces_and_edges)
          if (compass[f[0] + f[1] * 3 + f[2] * 9].size() != 0) {
            bool needme = false;
            auto &me = compass[f[0] + f[1] * 3 + f[2] * 9];

            for (size_t j1 = 0; j1 < me.size(); j1++)
              if (me[j1].needed) {
                needme = true;
                for (size_t j2 = 0; j2 < me.size(); j2++)
                  if (me[j2].needed && me[j2].contains(me[j1])) {
                    me[j1].needed = false;
                    me[j2].removedIndices.push_back(me[j1].index);
                    me[j2].Remove(me[j1]);
                    v.push_back(me[j1].index);
                    break;
                  }
              }

            if (!needme)
              continue;

            const int imax = (f[0] == 1) ? 2 : f[0];
            const int imin = (f[0] == 1) ? 0 : f[0];
            const int jmax = (f[1] == 1) ? 2 : f[1];
            const int jmin = (f[1] == 1) ? 0 : f[1];
            const int kmax = (f[2] == 1) ? 2 : f[2];
            const int kmin = (f[2] == 1) ? 0 : f[2];

            for (int k = kmin; k <= kmax; k++)
              for (int j = jmin; j <= jmax; j++)
                for (int i = imin; i <= imax; i++) {
                  if (i == f[0] && j == f[1] && k == f[2])
                    continue;
                  auto &other = compass[i + j * 3 + k * 9];

                  for (size_t j1 = 0; j1 < other.size(); j1++) {
                    auto &o = other[j1];
                    if (o.needed)
                      for (size_t k1 = 0; k1 < me.size(); k1++) {
                        auto &m = me[k1];
                        if (m.needed && m.contains(o)) {
                          o.needed = false;
                          m.removedIndices.push_back(o.index);
                          m.Remove(o);
                          v.push_back(o.index);
                          break;
                        }
                      }
                  }
                }
          }
      }
    };
    cube C;

    std::vector<int> offsets; ///< As the send buffer for each rank is being
                              ///< filled, offset[i] is the current offset where
                              ///< sent data is located in the send buffer.
    std::vector<int>
        offsets_recv; ///< As the send buffer for each rank is being filled,
                      ///< offset[i] is the current offset where sent data is
                      ///< located in the send buffer.
    SynchronizerMPI_AMR *Synch_ptr; ///< pointer to the SynchronizerMPI_AMR for
                                    ///< which to remove duplicate data
    std::vector<int> positions;
    std::vector<size_t> sizes;

    DuplicatesManager(SynchronizerMPI_AMR &Synch) {
      positions.resize(Synch.size);
      sizes.resize(Synch.size);
      offsets.resize(Synch.size, 0);
      offsets_recv.resize(Synch.size, 0);
      Synch_ptr = &Synch;
    }

    /// Adds an element to 'positions[r]'
    void Add(const int r, const int index) {
      if (sizes[r] == 0)
        positions[r] = index;
      sizes[r]++;
    }

    /**Remove duplicate data that will be sent to one rank.
     * @param r: the rank where the data will be sent
     * @param f: all the Interfaces between rank r and this rank
     * @param total_size: eventual size of the send buffer to rank r, after
     * duplicate Interfaces are removed.
     */
    void RemoveDuplicates(const int r, std::vector<Interface> &f,
                          int &total_size) {
      if (sizes[r] == 0)
        return;

      bool skip_needed = false;
      const int nc = Synch_ptr->getstencil().selcomponents.size();

      std::sort(f.begin() + positions[r], f.begin() + sizes[r] + positions[r]);

      C.clear();
      for (size_t i = 0; i < sizes[r]; i++) {
        C.compass[f[i + positions[r]].icode[0]].push_back(
            Synch_ptr->SM.DetermineStencil(f[i + positions[r]]));
        C.compass[f[i + positions[r]].icode[0]].back().index = i + positions[r];
        C.compass[f[i + positions[r]].icode[0]].back().avg_down =
            (f[i + positions[r]].infos[0]->level >
             f[i + positions[r]].infos[1]->level);
        if (skip_needed == false)
          skip_needed = f[i + positions[r]].CoarseStencil;
      }

      if (skip_needed == false) {
        std::vector<int> remEl;
        C.__needed(remEl);
        for (size_t k = 0; k < remEl.size(); k++)
          f[remEl[k]].ToBeKept = false;
      }

      int L[3] = {0, 0, 0};
      int Lc[3] = {0, 0, 0};
      for (auto &i : C.keepEl()) {
        const int k = i->index;
        Synch_ptr->SM.DetermineStencilLength(
            f[k].infos[0]->level, f[k].infos[1]->level, f[k].icode[1], L);
        const int V = L[0] * L[1] * L[2];
        total_size += V;
        f[k].dis = offsets[r];
        if (f[k].CoarseStencil) {
          Synch_ptr->SM.CoarseStencilLength(f[k].icode[1], Lc);
          const int Vc = Lc[0] * Lc[1] * Lc[2];
          total_size += Vc;
          offsets[r] += Vc * nc;
        }
        offsets[r] += V * nc;
        for (size_t kk = 0; kk < (*i).removedIndices.size(); kk++)
          f[i->removedIndices[kk]].dis = f[k].dis;
      }
    }

    void RemoveDuplicates_recv(std::vector<Interface> &f, int &total_size,
                               const int otherrank, const size_t start,
                               const size_t finish) {
      bool skip_needed = false;
      const int nc = Synch_ptr->getstencil().selcomponents.size();

      C.clear();
      for (size_t i = start; i < finish; i++) {
        C.compass[f[i].icode[0]].push_back(
            Synch_ptr->SM.DetermineStencil(f[i]));
        C.compass[f[i].icode[0]].back().index = i;
        C.compass[f[i].icode[0]].back().avg_down =
            (f[i].infos[0]->level > f[i].infos[1]->level);
        if (skip_needed == false)
          skip_needed = f[i].CoarseStencil;
      }

      if (skip_needed == false) {
        std::vector<int> remEl;
        C.__needed(remEl);
        for (size_t k = 0; k < remEl.size(); k++)
          f[remEl[k]].ToBeKept = false;
      }

      for (auto &i : C.keepEl()) {
        const int k = i->index;
        int L[3] = {0, 0, 0};
        int Lc[3] = {0, 0, 0};
        Synch_ptr->SM.DetermineStencilLength(
            f[k].infos[0]->level, f[k].infos[1]->level, f[k].icode[1], L);
        const int V = L[0] * L[1] * L[2];
        int Vc = 0;
        total_size += V;
        f[k].dis = offsets_recv[otherrank];
        UnPackInfo info = {f[k].dis,
                           L[0],
                           L[1],
                           L[2],
                           0,
                           0,
                           0,
                           L[0],
                           L[1],
                           -1,
                           0,
                           0,
                           0,
                           0,
                           0,
                           f[k].infos[0]->level,
                           f[k].icode[1],
                           otherrank,
                           f[k].infos[0]->index[0],
                           f[k].infos[0]->index[1],
                           f[k].infos[0]->index[2],
                           f[k].infos[1]->blockID_2};
        if (f[k].CoarseStencil) {
          Synch_ptr->SM.CoarseStencilLength(f[k].icode[1], Lc);
          Vc = Lc[0] * Lc[1] * Lc[2];
          total_size += Vc;
          offsets_recv[otherrank] += Vc * nc;
          info.CoarseVersionOffset = V * nc;
          info.CoarseVersionLX = Lc[0];
          info.CoarseVersionLY = Lc[1];
        }

        offsets_recv[otherrank] += V * nc;

        Synch_ptr->myunpacks[f[k].infos[1]->halo_block_id].push_back(info);

        for (size_t kk = 0; kk < (*i).removedIndices.size(); kk++) {
          const int remEl1 = i->removedIndices[kk];
          Synch_ptr->SM.DetermineStencilLength(f[remEl1].infos[0]->level,
                                               f[remEl1].infos[1]->level,
                                               f[remEl1].icode[1], &L[0]);
          int srcx, srcy, srcz;
          Synch_ptr->SM.__FixDuplicates(f[k], f[remEl1], info.lx, info.ly,
                                        info.lz, L[0], L[1], L[2], srcx, srcy,
                                        srcz);
          int Csrcx = 0;
          int Csrcy = 0;
          int Csrcz = 0;
          if (f[k].CoarseStencil)
            Synch_ptr->SM.__FixDuplicates2(f[k], f[remEl1], Csrcx, Csrcy,
                                           Csrcz);

          Synch_ptr->myunpacks[f[remEl1].infos[1]->halo_block_id].push_back(
              {info.offset,
               L[0],
               L[1],
               L[2],
               srcx,
               srcy,
               srcz,
               info.LX,
               info.LY,
               info.CoarseVersionOffset,
               info.CoarseVersionLX,
               info.CoarseVersionLY,
               Csrcx,
               Csrcy,
               Csrcz,
               f[remEl1].infos[0]->level,
               f[remEl1].icode[1],
               otherrank,
               f[remEl1].infos[0]->index[0],
               f[remEl1].infos[0]->index[1],
               f[remEl1].infos[0]->index[2],
               f[remEl1].infos[1]->blockID_2});

          f[remEl1].dis = info.offset;
        }
      }
    }
  };

  /// Check if blocks on the same refinement level need to exchange averaged
  /// down cells that will be used for coarse-fine interpolation.
  bool UseCoarseStencil(const Interface &f) {
    BlockInfo &a = *f.infos[0];
    BlockInfo &b = *f.infos[1];
    if (a.level == 0 || (!use_averages))
      return false;
    int imin[3];
    int imax[3];
    const int aux = 1 << a.level;
    const bool periodic[3] = {grid->xperiodic, grid->yperiodic,
                              grid->zperiodic};
    const int blocks[3] = {grid->getMaxBlocks()[0] * aux - 1,
                           grid->getMaxBlocks()[1] * aux - 1,
                           grid->getMaxBlocks()[2] * aux - 1};
    for (int d = 0; d < 3; d++) {
      imin[d] = (a.index[d] < b.index[d]) ? 0 : -1;
      imax[d] = (a.index[d] > b.index[d]) ? 0 : +1;
      if (periodic[d]) {
        if (a.index[d] == 0 && b.index[d] == blocks[d])
          imin[d] = -1;
        if (b.index[d] == 0 && a.index[d] == blocks[d])
          imax[d] = +1;
      } else {
        if (a.index[d] == 0 && b.index[d] == 0)
          imin[d] = 0;
        if (a.index[d] == blocks[d] && b.index[d] == blocks[d])
          imax[d] = 0;
      }
    }

    bool retval = false;
    for (int i2 = imin[2]; i2 <= imax[2]; i2++)
      for (int i1 = imin[1]; i1 <= imax[1]; i1++)
        for (int i0 = imin[0]; i0 <= imax[0]; i0++) {
          if ((grid->Tree(a.level, a.Znei_(i0, i1, i2))).CheckCoarser()) {
            retval = true;
            break;
          }
        }
    return retval;
  }

  /// Auxiliary function to average down data
  void AverageDownAndFill(Real *__restrict__ dst, const BlockInfo *const info,
                          const int code[3]) {
    const int s[3] = {code[0] < 1 ? (code[0] < 0 ? stencil.sx : 0) : nX,
                      code[1] < 1 ? (code[1] < 0 ? stencil.sy : 0) : nY,
                      code[2] < 1 ? (code[2] < 0 ? stencil.sz : 0) : nZ};
    const int e[3] = {
        code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + stencil.ex - 1,
        code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + stencil.ey - 1,
        code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + stencil.ez - 1};
#if DIMENSION == 3
    int pos = 0;
    const Real *src = (const Real *)(*info).ptrBlock;
    const int xStep = (code[0] == 0) ? 2 : 1;
    const int yStep = (code[1] == 0) ? 2 : 1;
    const int zStep = (code[2] == 0) ? 2 : 1;
    if (gptfloats == 1) {
      for (int iz = s[2]; iz < e[2]; iz += zStep) {
        const int ZZ = (abs(code[2]) == 1)
                           ? 2 * (iz - code[2] * nZ) + std::min(0, code[2]) * nZ
                           : iz;
        for (int iy = s[1]; iy < e[1]; iy += yStep) {
          const int YY = (abs(code[1]) == 1) ? 2 * (iy - code[1] * nY) +
                                                   std::min(0, code[1]) * nY
                                             : iy;
          for (int ix = s[0]; ix < e[0]; ix += xStep) {
            const int XX = (abs(code[0]) == 1) ? 2 * (ix - code[0] * nX) +
                                                     std::min(0, code[0]) * nX
                                               : ix;
#ifdef PRESERVE_SYMMETRY
            dst[pos] =
                ConsistentAverage(src[XX + (YY + (ZZ)*nY) * nX],
                                  src[XX + (YY + (ZZ + 1) * nY) * nX],
                                  src[XX + (YY + 1 + (ZZ)*nY) * nX],
                                  src[XX + (YY + 1 + (ZZ + 1) * nY) * nX],
                                  src[XX + 1 + (YY + (ZZ)*nY) * nX],
                                  src[XX + 1 + (YY + (ZZ + 1) * nY) * nX],
                                  src[XX + 1 + (YY + 1 + (ZZ)*nY) * nX],
                                  src[XX + 1 + (YY + 1 + (ZZ + 1) * nY) * nX]);
#else
            dst[pos] = 0.125 * (src[XX + (YY + (ZZ)*nY) * nX] +
                                src[XX + (YY + (ZZ + 1) * nY) * nX] +
                                src[XX + (YY + 1 + (ZZ)*nY) * nX] +
                                src[XX + (YY + 1 + (ZZ + 1) * nY) * nX] +
                                src[XX + 1 + (YY + (ZZ)*nY) * nX] +
                                src[XX + 1 + (YY + (ZZ + 1) * nY) * nX] +
                                src[XX + 1 + (YY + 1 + (ZZ)*nY) * nX] +
                                src[XX + 1 + (YY + 1 + (ZZ + 1) * nY) * nX]);
#endif
            pos++;
          }
        }
      }
    } else {
      for (int iz = s[2]; iz < e[2]; iz += zStep) {
        const int ZZ = (abs(code[2]) == 1)
                           ? 2 * (iz - code[2] * nZ) + std::min(0, code[2]) * nZ
                           : iz;
        for (int iy = s[1]; iy < e[1]; iy += yStep) {
          const int YY = (abs(code[1]) == 1) ? 2 * (iy - code[1] * nY) +
                                                   std::min(0, code[1]) * nY
                                             : iy;
          for (int ix = s[0]; ix < e[0]; ix += xStep) {
            const int XX = (abs(code[0]) == 1) ? 2 * (ix - code[0] * nX) +
                                                     std::min(0, code[0]) * nX
                                               : ix;
            for (int c = 0; c < NC; c++) {
              int comp = stencil.selcomponents[c];
#ifdef PRESERVE_SYMMETRY
              dst[pos] = ConsistentAverage(
                  (*(src + gptfloats * ((XX) + ((YY) + (ZZ)*nY) * nX) + comp)),
                  (*(src + gptfloats * ((XX) + ((YY) + (ZZ + 1) * nY) * nX) +
                     comp)),
                  (*(src + gptfloats * ((XX) + ((YY + 1) + (ZZ)*nY) * nX) +
                     comp)),
                  (*(src +
                     gptfloats * ((XX) + ((YY + 1) + (ZZ + 1) * nY) * nX) +
                     comp)),
                  (*(src + gptfloats * ((XX + 1) + ((YY) + (ZZ)*nY) * nX) +
                     comp)),
                  (*(src +
                     gptfloats * ((XX + 1) + ((YY) + (ZZ + 1) * nY) * nX) +
                     comp)),
                  (*(src + gptfloats * ((XX + 1) + ((YY + 1) + (ZZ)*nY) * nX) +
                     comp)),
                  (*(src +
                     gptfloats * ((XX + 1) + ((YY + 1) + (ZZ + 1) * nY) * nX) +
                     comp)));
#else
              dst[pos] =
                  0.125 *
                  ((*(src + gptfloats * ((XX) + ((YY) + (ZZ)*nY) * nX) +
                      comp)) +
                   (*(src + gptfloats * ((XX) + ((YY) + (ZZ + 1) * nY) * nX) +
                      comp)) +
                   (*(src + gptfloats * ((XX) + ((YY + 1) + (ZZ)*nY) * nX) +
                      comp)) +
                   (*(src +
                      gptfloats * ((XX) + ((YY + 1) + (ZZ + 1) * nY) * nX) +
                      comp)) +
                   (*(src + gptfloats * ((XX + 1) + ((YY) + (ZZ)*nY) * nX) +
                      comp)) +
                   (*(src +
                      gptfloats * ((XX + 1) + ((YY) + (ZZ + 1) * nY) * nX) +
                      comp)) +
                   (*(src + gptfloats * ((XX + 1) + ((YY + 1) + (ZZ)*nY) * nX) +
                      comp)) +
                   (*(src +
                      gptfloats * ((XX + 1) + ((YY + 1) + (ZZ + 1) * nY) * nX) +
                      comp)));
#endif
              pos++;
            }
          }
        }
      }
    }
#endif
#if DIMENSION == 2
    Real *src = (Real *)(*info).ptrBlock;
    const int xStep = (code[0] == 0) ? 2 : 1;
    const int yStep = (code[1] == 0) ? 2 : 1;
    int pos = 0;
    for (int iy = s[1]; iy < e[1]; iy += yStep) {
      const int YY = (abs(code[1]) == 1)
                         ? 2 * (iy - code[1] * nY) + std::min(0, code[1]) * nY
                         : iy;
      for (int ix = s[0]; ix < e[0]; ix += xStep) {
        const int XX = (abs(code[0]) == 1)
                           ? 2 * (ix - code[0] * nX) + std::min(0, code[0]) * nX
                           : ix;
        for (int c = 0; c < NC; c++) {
          int comp = stencil.selcomponents[c];
          dst[pos] =
              0.25 * (((*(src + gptfloats * (XX + (YY)*nX) + comp)) +
                       (*(src + gptfloats * (XX + 1 + (YY + 1) * nX) + comp))) +
                      ((*(src + gptfloats * (XX + (YY + 1) * nX) + comp)) +
                       (*(src + gptfloats * (XX + 1 + (YY)*nX) + comp))));
          pos++;
        }
      }
    }
#endif
  }

  /// Auxiliary function to average down data
  void AverageDownAndFill2(Real *dst, const BlockInfo *const info,
                           const int code[3]) {
    const int eC[3] = {(stencil.ex) / 2 + Cstencil.ex,
                       (stencil.ey) / 2 + Cstencil.ey,
                       (stencil.ez) / 2 + Cstencil.ez};
    const int sC[3] = {(stencil.sx - 1) / 2 + Cstencil.sx,
                       (stencil.sy - 1) / 2 + Cstencil.sy,
                       (stencil.sz - 1) / 2 + Cstencil.sz};

    const int s[3] = {code[0] < 1 ? (code[0] < 0 ? sC[0] : 0) : nX / 2,
                      code[1] < 1 ? (code[1] < 0 ? sC[1] : 0) : nY / 2,
                      code[2] < 1 ? (code[2] < 0 ? sC[2] : 0) : nZ / 2};

    const int e[3] = {
        code[0] < 1 ? (code[0] < 0 ? 0 : nX / 2) : nX / 2 + eC[0] - 1,
        code[1] < 1 ? (code[1] < 0 ? 0 : nY / 2) : nY / 2 + eC[1] - 1,
        code[2] < 1 ? (code[2] < 0 ? 0 : nZ / 2) : nZ / 2 + eC[2] - 1};

    Real *src = (Real *)(*info).ptrBlock;

    int pos = 0;

#if DIMENSION == 3
    for (int iz = s[2]; iz < e[2]; iz++) {
      const int ZZ = 2 * (iz - s[2]) + s[2] + std::max(code[2], 0) * nZ / 2 -
                     code[2] * nZ + std::min(0, code[2]) * (e[2] - s[2]);
#endif
      for (int iy = s[1]; iy < e[1]; iy++) {
        const int YY = 2 * (iy - s[1]) + s[1] + std::max(code[1], 0) * nY / 2 -
                       code[1] * nY + std::min(0, code[1]) * (e[1] - s[1]);
        for (int ix = s[0]; ix < e[0]; ix++) {
          const int XX = 2 * (ix - s[0]) + s[0] +
                         std::max(code[0], 0) * nX / 2 - code[0] * nX +
                         std::min(0, code[0]) * (e[0] - s[0]);

          for (int c = 0; c < NC; c++) {
            int comp = stencil.selcomponents[c];
#if DIMENSION == 3
#ifdef PRESERVE_SYMMETRY
            dst[pos] = ConsistentAverage(
                (*(src + gptfloats * ((XX) + ((YY) + (ZZ)*nY) * nX) + comp)),
                (*(src + gptfloats * ((XX) + ((YY) + (ZZ + 1) * nY) * nX) +
                   comp)),
                (*(src + gptfloats * ((XX) + ((YY + 1) + (ZZ)*nY) * nX) +
                   comp)),
                (*(src + gptfloats * ((XX) + ((YY + 1) + (ZZ + 1) * nY) * nX) +
                   comp)),
                (*(src + gptfloats * ((XX + 1) + ((YY) + (ZZ)*nY) * nX) +
                   comp)),
                (*(src + gptfloats * ((XX + 1) + ((YY) + (ZZ + 1) * nY) * nX) +
                   comp)),
                (*(src + gptfloats * ((XX + 1) + ((YY + 1) + (ZZ)*nY) * nX) +
                   comp)),
                (*(src +
                   gptfloats * ((XX + 1) + ((YY + 1) + (ZZ + 1) * nY) * nX) +
                   comp)));
#else
            dst[pos] =
                0.125 *
                ((*(src + gptfloats * ((XX) + ((YY) + (ZZ)*nY) * nX) + comp)) +
                 (*(src + gptfloats * ((XX) + ((YY) + (ZZ + 1) * nY) * nX) +
                    comp)) +
                 (*(src + gptfloats * ((XX) + ((YY + 1) + (ZZ)*nY) * nX) +
                    comp)) +
                 (*(src + gptfloats * ((XX) + ((YY + 1) + (ZZ + 1) * nY) * nX) +
                    comp)) +
                 (*(src + gptfloats * ((XX + 1) + ((YY) + (ZZ)*nY) * nX) +
                    comp)) +
                 (*(src + gptfloats * ((XX + 1) + ((YY) + (ZZ + 1) * nY) * nX) +
                    comp)) +
                 (*(src + gptfloats * ((XX + 1) + ((YY + 1) + (ZZ)*nY) * nX) +
                    comp)) +
                 (*(src +
                    gptfloats * ((XX + 1) + ((YY + 1) + (ZZ + 1) * nY) * nX) +
                    comp)));
#endif
#else
          dst[pos] =
              0.25 * (((*(src + gptfloats * (XX + (YY)*nX) + comp)) +
                       (*(src + gptfloats * (XX + 1 + (YY + 1) * nX) + comp))) +
                      ((*(src + gptfloats * (XX + (YY + 1) * nX) + comp)) +
                       (*(src + gptfloats * (XX + 1 + (YY)*nX) + comp))));
#endif
            pos++;
          }
        }
      }
#if DIMENSION == 3
    }
#endif
  }

#if 0
  std::string removeLeadingZeros(const std::string& input)
  {
    std::size_t firstNonZero = input.find_first_not_of('0');
    if (firstNonZero == std::string::npos)
    {
      // The input consists only of zeros
      return "0";
    }
    return input.substr(firstNonZero);
  }
  std::set<int> DecodeSet(std::string ID)
  {
    std::set<int> retval;
    for (size_t i = 0 ; i < ID.length() ; i += size)
    {
      std::string toconvert = removeLeadingZeros( ID.substr(i, size) );
      int current_rank = std::stoi ( toconvert );
      retval.insert(current_rank);
    }
    return retval;
  }
#endif

  /// Maps a set of integers to a string
  std::string EncodeSet(const std::set<int> &ranks) {
    std::string retval;
    for (auto r : ranks) {
      std::stringstream ss;
      ss << std::setw(size) << std::setfill('0') << r;
      std::string s = ss.str();
      retval += s;
    }
    return retval;
  }

public:
  /// Needs to be called whenever the grid changes because of
  /// refinement/compression
  void _Setup() {
    Neighbors.clear();
    inner_blocks.clear();
    halo_blocks.clear();
    for (int r = 0; r < size; r++) {
      send_interfaces[r].clear();
      recv_interfaces[r].clear();
      send_buffer_size[r] = 0;
    }

    for (size_t i = 0; i < myunpacks.size(); i++)
      myunpacks[i].clear();
    myunpacks.clear();

    DuplicatesManager DM(*(this));

    for (BlockInfo &info : grid->getBlocksInfo()) {
      info.halo_block_id = -1;
      const bool xskin =
          info.index[0] == 0 ||
          info.index[0] == ((grid->getMaxBlocks()[0] << info.level) - 1);
      const bool yskin =
          info.index[1] == 0 ||
          info.index[1] == ((grid->getMaxBlocks()[1] << info.level) - 1);
      const bool zskin =
          info.index[2] == 0 ||
          info.index[2] == ((grid->getMaxBlocks()[2] << info.level) - 1);
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;

      bool isInner = true;

      std::vector<int> ToBeChecked;
      bool Coarsened = false;

      for (int icode = 0; icode < 27; icode++) {
        if (icode == 1 * 1 + 3 * 1 + 9 * 1)
          continue;
        const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                             (icode / 9) % 3 - 1};

#if DIMENSION == 2
        if (code[2] != 0)
          continue;
#endif
        if (!grid->xperiodic && code[0] == xskip && xskin)
          continue;
        if (!grid->yperiodic && code[1] == yskip && yskin)
          continue;
        if (!grid->zperiodic && code[2] == zskip && zskin)
          continue;

        // if (!stencil.tensorial && !Cstencil.tensorial &&
        // abs(code[0])+abs(code[1])+abs(code[2])>1) continue; if
        // (!stencil.tensorial && use_averages == false &&
        // abs(code[0])+abs(code[1])+abs(code[2])>1) continue;

        const TreePosition &infoNeiTree =
            grid->Tree(info.level, info.Znei_(code[0], code[1], code[2]));

        if (infoNeiTree.Exists() && infoNeiTree.rank() != rank) {
          isInner = false;
          Neighbors.insert(infoNeiTree.rank());

          BlockInfo &infoNei = grid->getBlockInfoAll(
              info.level, info.Znei_(code[0], code[1], code[2]));

          const int icode2 =
              (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;

          send_interfaces[infoNeiTree.rank()].push_back(
              {info, infoNei, icode, icode2});
          recv_interfaces[infoNeiTree.rank()].push_back(
              {infoNei, info, icode2, icode});

          ToBeChecked.push_back(infoNeiTree.rank());
          ToBeChecked.push_back(
              (int)send_interfaces[infoNeiTree.rank()].size() - 1);
          ToBeChecked.push_back(
              (int)recv_interfaces[infoNeiTree.rank()].size() - 1);

          DM.Add(infoNeiTree.rank(),
                 (int)send_interfaces[infoNeiTree.rank()].size() - 1);
        } else if (infoNeiTree.CheckCoarser()) {
          Coarsened = true;
          BlockInfo &infoNei = grid->getBlockInfoAll(
              info.level, info.Znei_(code[0], code[1], code[2]));
          const int infoNeiCoarserrank =
              grid->Tree(info.level - 1, infoNei.Zparent).rank();
          if (infoNeiCoarserrank != rank) {
            isInner = false;
            Neighbors.insert(infoNeiCoarserrank);

            BlockInfo &infoNeiCoarser =
                grid->getBlockInfoAll(infoNei.level - 1, infoNei.Zparent);

            const int icode2 =
                (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;

            const int Bmax[3] = {grid->getMaxBlocks()[0] << (info.level - 1),
                                 grid->getMaxBlocks()[1] << (info.level - 1),
                                 grid->getMaxBlocks()[2] << (info.level - 1)};

            const int test_idx[3] = {
                (infoNeiCoarser.index[0] - code[0] + Bmax[0]) % Bmax[0],
                (infoNeiCoarser.index[1] - code[1] + Bmax[1]) % Bmax[1],
                (infoNeiCoarser.index[2] - code[2] + Bmax[2]) % Bmax[2]};

            if (info.index[0] / 2 == test_idx[0] &&
                info.index[1] / 2 == test_idx[1] &&
                info.index[2] / 2 == test_idx[2]) {
              send_interfaces[infoNeiCoarserrank].push_back(
                  {info, infoNeiCoarser, icode, icode2});
              recv_interfaces[infoNeiCoarserrank].push_back(
                  {infoNeiCoarser, info, icode2, icode});

              DM.Add(infoNeiCoarserrank,
                     (int)send_interfaces[infoNeiCoarserrank].size() - 1);

              if (abs(code[0]) + abs(code[1]) + abs(code[2]) ==
                  1) // if filling a face need also two edges and a corner
              {
                const int d0 = abs(
                    code[1] + 2 * code[2]); // =0 if |code[0]|=1, =1 if
                                            // |code[1]|=1, =2 if |code[2]|=1
                const int d1 = (d0 + 1) % 3;
                const int d2 = (d0 + 2) % 3;

                // corner being filled
                int code3[3];
                code3[d0] = code[d0];
                code3[d1] = -2 * (info.index[d1] % 2) + 1;
                code3[d2] = -2 * (info.index[d2] % 2) + 1;
                const int icode3 =
                    (code3[0] + 1) + (code3[1] + 1) * 3 + (code3[2] + 1) * 9;

                // edge in the d1 direction
                int code4[3];
                code4[d0] = code[d0];
                code4[d1] = code3[d1];
                code4[d2] = 0;
                const int icode4 =
                    (code4[0] + 1) + (code4[1] + 1) * 3 + (code4[2] + 1) * 9;

                // edge in the d2 direction
                int code5[3];
                code5[d0] = code[d0];
                code5[d1] = 0;
                code5[d2] = code3[d2];
                const int icode5 =
                    (code5[0] + 1) + (code5[1] + 1) * 3 + (code5[2] + 1) * 9;

#if DIMENSION == 2
                if (code3[2] == 0)
                  recv_interfaces[infoNeiCoarserrank].push_back(
                      {infoNeiCoarser, info, icode2, icode3});
                if (code4[2] == 0)
                  recv_interfaces[infoNeiCoarserrank].push_back(
                      {infoNeiCoarser, info, icode2, icode4});
                if (code5[2] == 0)
                  recv_interfaces[infoNeiCoarserrank].push_back(
                      {infoNeiCoarser, info, icode2, icode5});
#else
                recv_interfaces[infoNeiCoarserrank].push_back(
                    {infoNeiCoarser, info, icode2, icode3});
                recv_interfaces[infoNeiCoarserrank].push_back(
                    {infoNeiCoarser, info, icode2, icode4});
                recv_interfaces[infoNeiCoarserrank].push_back(
                    {infoNeiCoarser, info, icode2, icode5});
#endif
              }
#if DIMENSION == 3
              else if (abs(code[0]) + abs(code[1]) + abs(code[2]) ==
                       2) // if filling an edge need also a corner
              {
                const int d0 = (1 - abs(code[1])) + 2 * (1 - abs(code[2]));
                const int d1 = (d0 + 1) % 3;
                const int d2 = (d0 + 2) % 3;
                int code3[3];
                code3[d0] = -2 * (info.index[d0] % 2) + 1;
                code3[d1] = code[d1];
                code3[d2] = code[d2];
                const int icode3 =
                    (code3[0] + 1) + (code3[1] + 1) * 3 + (code3[2] + 1) * 9;
                recv_interfaces[infoNeiCoarserrank].push_back(
                    {infoNeiCoarser, info, icode2, icode3});
              }
#endif
            }
          }
        } else if (infoNeiTree.CheckFiner()) {
          BlockInfo &infoNei = grid->getBlockInfoAll(
              info.level, info.Znei_(code[0], code[1], code[2]));

          int Bstep = 1;
          if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))
            Bstep = 3; // edge
          else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
            Bstep = 4; // corner

          for (int B = 0; B <= 3; B += Bstep) // loop over blocks that make up
                                              // face/edge/corner (4/2/1 blocks)
          {
#if DIMENSION == 2
            if (Bstep == 1 && B >= 2)
              continue;
            if (Bstep > 1 && B >= 1)
              continue;
#endif
            const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);

            const long long nFine =
                infoNei.Zchild[std::max(-code[0], 0) +
                               (B % 2) * std::max(0, 1 - abs(code[0]))]
                              [std::max(-code[1], 0) +
                               temp * std::max(0, 1 - abs(code[1]))]
                              [std::max(-code[2], 0) +
                               (B / 2) * std::max(0, 1 - abs(code[2]))];

            const int infoNeiFinerrank =
                grid->Tree(info.level + 1, nFine).rank();

            if (infoNeiFinerrank != rank) {
              isInner = false;
              Neighbors.insert(infoNeiFinerrank);

              BlockInfo &infoNeiFiner =
                  grid->getBlockInfoAll(info.level + 1, nFine);

              const int icode2 =
                  (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;

              send_interfaces[infoNeiFinerrank].push_back(
                  {info, infoNeiFiner, icode, icode2});
              recv_interfaces[infoNeiFinerrank].push_back(
                  {infoNeiFiner, info, icode2, icode});

              DM.Add(infoNeiFinerrank,
                     (int)send_interfaces[infoNeiFinerrank].size() - 1);

              if (Bstep == 1) // if I'm filling a face then I'm also filling two
                              // edges and a corner
              {
                const int d0 = abs(
                    code[1] + 2 * code[2]); // =0 if |code[0]|=1, =1 if
                                            // |code[1]|=1, =2 if |code[2]|=1
                const int d1 = (d0 + 1) % 3;
                const int d2 = (d0 + 2) % 3;

                // corner being filled
                int code3[3];
                code3[d0] = -code[d0];
                code3[d1] = -2 * (infoNeiFiner.index[d1] % 2) + 1;
                code3[d2] = -2 * (infoNeiFiner.index[d2] % 2) + 1;
                const int icode3 =
                    (code3[0] + 1) + (code3[1] + 1) * 3 + (code3[2] + 1) * 9;

                // edge in the d1 direction
                int code4[3];
                code4[d0] = -code[d0];
                code4[d1] = code3[d1];
                code4[d2] = 0;
                const int icode4 =
                    (code4[0] + 1) + (code4[1] + 1) * 3 + (code4[2] + 1) * 9;

                // edge in the d2 direction
                int code5[3];
                code5[d0] = -code[d0];
                code5[d1] = 0;
                code5[d2] = code3[d2];
                const int icode5 =
                    (code5[0] + 1) + (code5[1] + 1) * 3 + (code5[2] + 1) * 9;

#if DIMENSION == 2
                if (code3[2] == 0) {
                  send_interfaces[infoNeiFinerrank].push_back(
                      Interface(info, infoNeiFiner, icode, icode3));
                  DM.Add(infoNeiFinerrank,
                         (int)send_interfaces[infoNeiFinerrank].size() - 1);
                }
                if (code4[2] == 0) {
                  send_interfaces[infoNeiFinerrank].push_back(
                      Interface(info, infoNeiFiner, icode, icode4));
                  DM.Add(infoNeiFinerrank,
                         (int)send_interfaces[infoNeiFinerrank].size() - 1);
                }
                if (code5[2] == 0) {
                  send_interfaces[infoNeiFinerrank].push_back(
                      Interface(info, infoNeiFiner, icode, icode5));
                  DM.Add(infoNeiFinerrank,
                         (int)send_interfaces[infoNeiFinerrank].size() - 1);
                }
#else
                send_interfaces[infoNeiFinerrank].push_back(
                    {info, infoNeiFiner, icode, icode3});
                DM.Add(infoNeiFinerrank,
                       (int)send_interfaces[infoNeiFinerrank].size() - 1);
                send_interfaces[infoNeiFinerrank].push_back(
                    {info, infoNeiFiner, icode, icode4});
                DM.Add(infoNeiFinerrank,
                       (int)send_interfaces[infoNeiFinerrank].size() - 1);
                send_interfaces[infoNeiFinerrank].push_back(
                    {info, infoNeiFiner, icode, icode5});
                DM.Add(infoNeiFinerrank,
                       (int)send_interfaces[infoNeiFinerrank].size() - 1);
#endif
              }
#if DIMENSION == 3
              else if (Bstep == 3) // if I'm filling an edge then I'm also
                                   // filling a corner
              {
                const int d0 = (1 - abs(code[1])) + 2 * (1 - abs(code[2]));
                const int d1 = (d0 + 1) % 3;
                const int d2 = (d0 + 2) % 3;
                int code3[3];
                code3[d0] = B == 0 ? 1 : -1;
                code3[d1] = -code[d1];
                code3[d2] = -code[d2];
                const int icode3 =
                    (code3[0] + 1) + (code3[1] + 1) * 3 + (code3[2] + 1) * 9;
                send_interfaces[infoNeiFinerrank].push_back(
                    {info, infoNeiFiner, icode, icode3});
                DM.Add(infoNeiFinerrank,
                       (int)send_interfaces[infoNeiFinerrank].size() - 1);
              }
#endif
            }
          }
        }
      } // icode = 0,...,26

      if (isInner) {
        info.halo_block_id = -1;
        inner_blocks.push_back(&info);
      } else {
        info.halo_block_id = halo_blocks.size();
        halo_blocks.push_back(&info);
        if (Coarsened) {
          for (size_t j = 0; j < ToBeChecked.size(); j += 3) {
            const int r = ToBeChecked[j];
            const int send = ToBeChecked[j + 1];
            const int recv = ToBeChecked[j + 2];
            const bool tmp = UseCoarseStencil(send_interfaces[r][send]);
            send_interfaces[r][send].CoarseStencil = tmp;
            recv_interfaces[r][recv].CoarseStencil = tmp;
          }
        }

        for (int r = 0; r < size; r++)
          if (DM.sizes[r] > 0) {
            DM.RemoveDuplicates(r, send_interfaces[r].v, send_buffer_size[r]);
            DM.sizes[r] = 0;
          }
      }
      grid->getBlockInfoAll(info.level, info.Z).halo_block_id =
          info.halo_block_id;
    } // i-loop

    myunpacks.resize(halo_blocks.size());

    for (int r = 0; r < size; r++) {
      recv_buffer_size[r] = 0;
      std::sort(recv_interfaces[r].begin(), recv_interfaces[r].end());

      size_t counter = 0;
      while (counter < recv_interfaces[r].size()) {
        const long long ID = recv_interfaces[r][counter].infos[0]->blockID_2;
        const size_t start = counter;
        size_t finish = start + 1;
        counter++;
        size_t j;
        for (j = counter; j < recv_interfaces[r].size(); j++) {
          if (recv_interfaces[r][j].infos[0]->blockID_2 == ID)
            finish++;
          else
            break;
        }
        counter = j;

        DM.RemoveDuplicates_recv(recv_interfaces[r].v, recv_buffer_size[r], r,
                                 start, finish);
      }

      send_buffer[r].resize(send_buffer_size[r] * NC);
      recv_buffer[r].resize(recv_buffer_size[r] * NC);
      send_packinfos[r].clear();
      ToBeAveragedDown[r].clear();
      for (int i = 0; i < (int)send_interfaces[r].size(); i++) {
        const Interface &f = send_interfaces[r][i];

        if (!f.ToBeKept)
          continue;

        if (f.infos[0]->level <= f.infos[1]->level) {
          const MyRange &range = SM.DetermineStencil(f);
          send_packinfos[r].push_back(
              {(Real *)f.infos[0]->ptrBlock, &send_buffer[r][f.dis], range.sx,
               range.sy, range.sz, range.ex, range.ey, range.ez});
          if (f.CoarseStencil) {
            const int V = (range.ex - range.sx) * (range.ey - range.sy) *
                          (range.ez - range.sz);
            ToBeAveragedDown[r].push_back(i);
            ToBeAveragedDown[r].push_back(f.dis + V * NC);
          }
        } else // receiver is coarser, so sender averages down data first
        {
          ToBeAveragedDown[r].push_back(i);
          ToBeAveragedDown[r].push_back(f.dis);
        }
      }
    }

    mapofHaloBlockGroups.clear();
    for (auto &info : halo_blocks) {
      // 1. Find ranks from which 'info' wants to receive
      const int id = info->halo_block_id;
      UnPackInfo *unpacks = myunpacks[id].data();
      std::set<int> ranks;
      for (size_t jj = 0; jj < myunpacks[id].size(); jj++) {
        const UnPackInfo &unpack = unpacks[jj];
        ranks.insert(unpack.rank);
      }
      // 2. Encode the set of ranks to one number
      auto set_ID = EncodeSet(ranks);

      // 3. Find that set and add 'info' to it. If set does not exist, create
      // it.
      const auto retval = mapofHaloBlockGroups.find(set_ID);
      if (retval == mapofHaloBlockGroups.end()) {
        HaloBlockGroup temporary;
        temporary.myranks = ranks;
        temporary.myblocks.push_back(info);
        mapofHaloBlockGroups[set_ID] = temporary;
      } else {
        (retval->second).myblocks.push_back(info);
      }
    }
  }

  // constructor
  SynchronizerMPI_AMR(StencilInfo a_stencil, StencilInfo a_Cstencil,
                      TGrid *_grid)
      : stencil(a_stencil), Cstencil(a_Cstencil),
        SM(a_stencil, a_Cstencil, TGrid::Block::sizeX, TGrid::Block::sizeY,
           TGrid::Block::sizeZ),
        gptfloats(sizeof(typename TGrid::Block::ElementType) / sizeof(Real)),
        NC(a_stencil.selcomponents.size()) {
    grid = _grid;
    use_averages = (grid->FiniteDifferences == false || stencil.tensorial ||
                    stencil.sx < -2 || stencil.sy < -2 || stencil.sz < -2 ||
                    stencil.ex > 3 || stencil.ey > 3 || stencil.ez > 3);
    comm = grid->getWorldComm();
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    nX = TGrid::Block::sizeX;
    nY = TGrid::Block::sizeY;
    nZ = TGrid::Block::sizeZ;
    send_interfaces.resize(size);
    recv_interfaces.resize(size);
    send_packinfos.resize(size);
    send_buffer_size.resize(size);
    recv_buffer_size.resize(size);
    send_buffer.resize(size);
    recv_buffer.resize(size);
    ToBeAveragedDown.resize(size);
    std::sort(stencil.selcomponents.begin(), stencil.selcomponents.end());
    if (sizeof(Real) == sizeof(double)) {
      MPIREAL = MPI_DOUBLE;
    } else if (sizeof(Real) == sizeof(long double)) {
      MPIREAL = MPI_LONG_DOUBLE;
    } else {
      MPIREAL = MPI_FLOAT;
      assert(sizeof(Real) == sizeof(float));
    }
  }

  /// Returns vector of pointers to inner blocks.
  std::vector<BlockInfo *> &avail_inner() { return inner_blocks; }

  /// Returns vector of pointers to halo blocks.
  std::vector<BlockInfo *> &avail_halo() {
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    return halo_blocks;
  }

  /// Returns vector of pointers to halo blocks without calling MPI_Wait
  std::vector<BlockInfo *> &avail_halo_nowait() { return halo_blocks; }

  /// Empty vector that avail_next() returns if no halo block groups are
  /// available
  std::vector<BlockInfo *> dummy_vector;

  /// Returns the next available (in terms of completed communication) group of
  /// halo blocks
  std::vector<BlockInfo *> &avail_next() {
    bool done = false;
    auto it = mapofHaloBlockGroups.begin();
    while (done == false) {
      done = true;
      it = mapofHaloBlockGroups.begin();
      while (it != mapofHaloBlockGroups.end()) {
        if ((it->second).ready == false) {
          std::set<int> ranks = (it->second).myranks;
          int flag = 0;
          for (auto r : ranks) {
            const auto retval = mapofrequests.find(r);
            MPI_Test(retval->second, &flag, MPI_STATUS_IGNORE);
            if (flag == false)
              break;
          }
          if (flag == 1) {
            (it->second).ready = true;
            return (it->second).myblocks;
          }
        }
        done = done && (it->second).ready;
        it++;
      }
    }
    return dummy_vector;
  }

  /// Needs to be called to initiate communication and halo cells exchange.
  void sync() {
    auto it = mapofHaloBlockGroups.begin();
    while (it != mapofHaloBlockGroups.end()) {
      (it->second).ready = false;
      it++;
    }

    const int timestamp = grid->getTimeStamp();
    mapofrequests.clear();
    requests.clear();
    requests.reserve(2 * size);

    // Post receive requests first
    for (auto r : Neighbors)
      if (recv_buffer_size[r] > 0) {
        requests.resize(requests.size() + 1);
        mapofrequests[r] = &requests.back();
        MPI_Irecv(&recv_buffer[r][0], recv_buffer_size[r] * NC, MPIREAL, r,
                  timestamp, comm, &requests.back());
      }

    // Pack data
    for (int r = 0; r < size; r++)
      if (send_buffer_size[r] != 0) {
#pragma omp parallel
        {
#pragma omp for
          for (size_t j = 0; j < ToBeAveragedDown[r].size(); j += 2) {
            const int i = ToBeAveragedDown[r][j];
            const int d = ToBeAveragedDown[r][j + 1];
            const Interface &f = send_interfaces[r][i];
            const int code[3] = {-(f.icode[0] % 3 - 1),
                                 -((f.icode[0] / 3) % 3 - 1),
                                 -((f.icode[0] / 9) % 3 - 1)};
            if (f.CoarseStencil)
              AverageDownAndFill2(send_buffer[r].data() + d, f.infos[0], code);
            else
              AverageDownAndFill(send_buffer[r].data() + d, f.infos[0], code);
          }
#pragma omp for
          for (size_t i = 0; i < send_packinfos[r].size(); i++) {
            const PackInfo &info = send_packinfos[r][i];
            pack(info.block, info.pack, gptfloats,
                 &stencil.selcomponents.front(), NC, info.sx, info.sy, info.sz,
                 info.ex, info.ey, info.ez, nX, nY);
          }
        }
      }

    // Do the sends
    for (auto r : Neighbors)
      if (send_buffer_size[r] > 0) {
        requests.resize(requests.size() + 1);
        MPI_Isend(&send_buffer[r][0], send_buffer_size[r] * NC, MPIREAL, r,
                  timestamp, comm, &requests.back());
      }
  }

  /// Get the StencilInfo of this Synchronizer
  const StencilInfo &getstencil() const { return stencil; }

  /// Check whether communication for a particular block has compelted
  bool isready(const BlockInfo &info) {
    const int id = info.halo_block_id;
    if (id < 0)
      return true;
    UnPackInfo *unpacks = myunpacks[id].data();
    for (size_t jj = 0; jj < myunpacks[id].size(); jj++) {
      const UnPackInfo &unpack = unpacks[jj];
      const int otherrank = unpack.rank;
      int flag = 0;
      const auto retval = mapofrequests.find(otherrank);
      MPI_Test(retval->second, &flag, MPI_STATUS_IGNORE);
      if (flag == 0)
        return false;
    }
    return true;
  }

  /// Used by BlockLabMPI, to get the data from the receive buffers owned by the
  /// Synchronizer and put them in its working copy of a GridBlock plus its halo
  /// cells.
  void fetch(const BlockInfo &info, const unsigned int Length[3],
             const unsigned int CLength[3], Real *cacheBlock,
             Real *coarseBlock) {
    // fetch received data for blocks that are neighbors with 'info' but are
    // owned by another rank
    const int id = info.halo_block_id;
    if (id < 0)
      return;

    // loop over all unpacks that correspond to block with this halo_block_id
    UnPackInfo *unpacks = myunpacks[id].data();
    for (size_t jj = 0; jj < myunpacks[id].size(); jj++) {
      const UnPackInfo &unpack = unpacks[jj];
      const int code[3] = {unpack.icode % 3 - 1, (unpack.icode / 3) % 3 - 1,
                           (unpack.icode / 9) % 3 - 1};
      const int otherrank = unpack.rank;

      // Based on the current unpack's icode, regions starting from 's' and
      // ending to 'e' of the current block will be filled with ghost cells.
      const int s[3] = {code[0] < 1 ? (code[0] < 0 ? stencil.sx : 0) : nX,
                        code[1] < 1 ? (code[1] < 0 ? stencil.sy : 0) : nY,
                        code[2] < 1 ? (code[2] < 0 ? stencil.sz : 0) : nZ};
      const int e[3] = {
          code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + stencil.ex - 1,
          code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + stencil.ey - 1,
          code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + stencil.ez - 1};

      if (unpack.level == info.level) // same level neighbors
      {
        Real *dst =
            cacheBlock + ((s[2] - stencil.sz) * Length[0] * Length[1] +
                          (s[1] - stencil.sy) * Length[0] + s[0] - stencil.sx) *
                             gptfloats;

        unpack_subregion(&recv_buffer[otherrank][unpack.offset], &dst[0],
                         gptfloats, &stencil.selcomponents[0],
                         stencil.selcomponents.size(), unpack.srcxstart,
                         unpack.srcystart, unpack.srczstart, unpack.LX,
                         unpack.LY, 0, 0, 0, unpack.lx, unpack.ly, unpack.lz,
                         Length[0], Length[1], Length[2]);

        if (unpack.CoarseVersionOffset >=
            0) // same level neighbors exchange averaged down ghosts
        {
          const int offset[3] = {(stencil.sx - 1) / 2 + Cstencil.sx,
                                 (stencil.sy - 1) / 2 + Cstencil.sy,
                                 (stencil.sz - 1) / 2 + Cstencil.sz};
          const int sC[3] = {
              code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : nX / 2,
              code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : nY / 2,
              code[2] < 1 ? (code[2] < 0 ? offset[2] : 0) : nZ / 2};
          Real *dst1 = coarseBlock +
                       ((sC[2] - offset[2]) * CLength[0] * CLength[1] +
                        (sC[1] - offset[1]) * CLength[0] + sC[0] - offset[0]) *
                           gptfloats;

          int L[3];
          SM.CoarseStencilLength(
              (-code[0] + 1) + 3 * (-code[1] + 1) + 9 * (-code[2] + 1), L);

          unpack_subregion(
              &recv_buffer[otherrank]
                          [unpack.offset + unpack.CoarseVersionOffset],
              &dst1[0], gptfloats, &stencil.selcomponents[0],
              stencil.selcomponents.size(), unpack.CoarseVersionsrcxstart,
              unpack.CoarseVersionsrcystart, unpack.CoarseVersionsrczstart,
              unpack.CoarseVersionLX, unpack.CoarseVersionLY, 0, 0, 0, L[0],
              L[1], L[2], CLength[0], CLength[1], CLength[2]);
        }
      } else if (unpack.level < info.level) {
        const int offset[3] = {(stencil.sx - 1) / 2 + Cstencil.sx,
                               (stencil.sy - 1) / 2 + Cstencil.sy,
                               (stencil.sz - 1) / 2 + Cstencil.sz};
        const int sC[3] = {code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : nX / 2,
                           code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : nY / 2,
                           code[2] < 1 ? (code[2] < 0 ? offset[2] : 0)
                                       : nZ / 2};
        Real *dst = coarseBlock +
                    ((sC[2] - offset[2]) * CLength[0] * CLength[1] + sC[0] -
                     offset[0] + (sC[1] - offset[1]) * CLength[0]) *
                        gptfloats;
        unpack_subregion(&recv_buffer[otherrank][unpack.offset], &dst[0],
                         gptfloats, &stencil.selcomponents[0],
                         stencil.selcomponents.size(), unpack.srcxstart,
                         unpack.srcystart, unpack.srczstart, unpack.LX,
                         unpack.LY, 0, 0, 0, unpack.lx, unpack.ly, unpack.lz,
                         CLength[0], CLength[1], CLength[2]);
      } else {
        int B;
        if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
          B = 0;                                                    // corner
        else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2)) // edge
        {
          int t;
          if (code[0] == 0)
            t = unpack.index_0 - 2 * info.index[0];
          else if (code[1] == 0)
            t = unpack.index_1 - 2 * info.index[1];
          else
            t = unpack.index_2 - 2 * info.index[2];
          assert(t == 0 || t == 1);
          B = (t == 1) ? 3 : 0;
        } else {
          int Bmod, Bdiv;
          if (abs(code[0]) == 1) {
            Bmod = unpack.index_1 - 2 * info.index[1];
            Bdiv = unpack.index_2 - 2 * info.index[2];
          } else if (abs(code[1]) == 1) {
            Bmod = unpack.index_0 - 2 * info.index[0];
            Bdiv = unpack.index_2 - 2 * info.index[2];
          } else {
            Bmod = unpack.index_0 - 2 * info.index[0];
            Bdiv = unpack.index_1 - 2 * info.index[1];
          }
          B = 2 * Bdiv + Bmod;
        }
        const int aux1 = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
        Real *dst =
            cacheBlock +
            ((abs(code[2]) * (s[2] - stencil.sz) +
              (1 - abs(code[2])) *
                  (-stencil.sz + (B / 2) * (e[2] - s[2]) / 2)) *
                 Length[0] * Length[1] +
             (abs(code[1]) * (s[1] - stencil.sy) +
              (1 - abs(code[1])) * (-stencil.sy + aux1 * (e[1] - s[1]) / 2)) *
                 Length[0] +
             abs(code[0]) * (s[0] - stencil.sx) +
             (1 - abs(code[0])) * (-stencil.sx + (B % 2) * (e[0] - s[0]) / 2)) *
                gptfloats;
        unpack_subregion(&recv_buffer[otherrank][unpack.offset], &dst[0],
                         gptfloats, &stencil.selcomponents[0],
                         stencil.selcomponents.size(), unpack.srcxstart,
                         unpack.srcystart, unpack.srczstart, unpack.LX,
                         unpack.LY, 0, 0, 0, unpack.lx, unpack.ly, unpack.lz,
                         Length[0], Length[1], Length[2]);
      }
    }
  }
};

} // namespace cubism

namespace cubism {

/**
 * @brief Performs flux corrections at coarse-fine block interfaces, with
 * multiple MPI processes.
 *
 * This class can replace the coarse fluxes stored at BlockCases with the sum of
 * the fine fluxes (also stored at BlockCases). This ensures conservation of the
 * quantity whose flux we compute.
 * @tparam TFluxCorrection The single-node version from which this class
 * inherits
 */

template <typename TFluxCorrection>
class FluxCorrectionMPI : public TFluxCorrection {
public:
  using TGrid = typename TFluxCorrection::GridType;
  typedef typename TFluxCorrection::ElementType ElementType;
  typedef typename TFluxCorrection::Real Real;
  typedef typename TFluxCorrection::BlockType BlockType;
  typedef BlockCase<BlockType> Case;
  int size;

protected:
  /// Auxiliary struct to keep track of coarse-fine interfaces between two
  /// different MPI processes
  struct face {
    BlockInfo *infos[2]; ///< the two BlockInfos of the interface
    int icode[2]; ///< encodes what face (+x,-x,+y,-y,+z,-z) is shared by the
                  ///< two Blocks
    int offset;   ///< offset in the send/recv buffers where the data for this
                  ///< face will be put
    // infos[0] : Fine block
    // infos[1] : Coarse block
    face(BlockInfo &i0, BlockInfo &i1, int a_icode0, int a_icode1) {
      infos[0] = &i0;
      infos[1] = &i1;
      icode[0] = a_icode0;
      icode[1] = a_icode1;
    }
    bool operator<(const face &other) const {
      if (infos[0]->blockID_2 == other.infos[0]->blockID_2) {
        return (icode[0] < other.icode[0]);
      } else {
        return (infos[0]->blockID_2 < other.infos[0]->blockID_2);
      }
    }
  };

  std::vector<std::vector<Real>>
      send_buffer; ///< multiple buffers to send to other ranks
  std::vector<std::vector<Real>>
      recv_buffer; ///< multiple buffers to receive from other ranks
  std::vector<std::vector<face>>
      send_faces; ///< buffers with 'faces' meta-data to send
  std::vector<std::vector<face>>
      recv_faces; ///< buffers with 'faces' meta-data to receive

  /// Perform flux correction for face 'F'
  void FillCase(face &F) {
    BlockInfo &info = *F.infos[1];
    const int icode = F.icode[1];
    const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                         (icode / 9) % 3 - 1};

    const int myFace = abs(code[0]) * std::max(0, code[0]) +
                       abs(code[1]) * (std::max(0, code[1]) + 2) +
                       abs(code[2]) * (std::max(0, code[2]) + 4);
    std::array<long long, 2> temp = {(long long)info.level, info.Z};
    auto search = TFluxCorrection::MapOfCases.find(temp);
    assert(search != TFluxCorrection::MapOfCases.end());
    Case &CoarseCase = (*search->second);
    std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];
#if DIMENSION == 3
    for (int B = 0; B <= 3;
         B++) // loop over fine blocks that make up coarse face
#else
    for (int B = 0; B <= 1;
         B++) // loop over fine blocks that make up coarse face
#endif
    {
      const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);

#if DIMENSION == 3
      const long long Z =
          (*TFluxCorrection::grid)
              .getZforward(info.level + 1,
                           2 * info.index[0] + std::max(code[0], 0) + code[0] +
                               (B % 2) * std::max(0, 1 - abs(code[0])),
                           2 * info.index[1] + std::max(code[1], 0) + code[1] +
                               aux * std::max(0, 1 - abs(code[1])),
                           2 * info.index[2] + std::max(code[2], 0) + code[2] +
                               (B / 2) * std::max(0, 1 - abs(code[2])));
#else
      const long long Z =
          (*TFluxCorrection::grid)
              .getZforward(info.level + 1,
                           2 * info.index[0] + std::max(code[0], 0) + code[0] +
                               (B % 2) * std::max(0, 1 - abs(code[0])),
                           2 * info.index[1] + std::max(code[1], 0) + code[1] +
                               aux * std::max(0, 1 - abs(code[1])));
#endif
      if (Z != F.infos[0]->Z)
        continue;

      const int d = myFace / 2;
      const int d1 = std::max((d + 1) % 3, (d + 2) % 3);
      const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
      const int N1 = CoarseCase.m_vSize[d1];
      const int N2 = CoarseCase.m_vSize[d2];

      int base = 0; //(B%2)*(N1/2)+ (B/2)*(N2/2)*N1;
      if (B == 1)
        base = (N2 / 2) + (0) * N2;
      else if (B == 2)
        base = (0) + (N1 / 2) * N2;
      else if (B == 3)
        base = (N2 / 2) + (N1 / 2) * N2;

      int r = (*TFluxCorrection::grid)
                  .Tree(F.infos[0]->level, F.infos[0]->Z)
                  .rank();
      int dis = 0;

#if DIMENSION == 3
      for (int i1 = 0; i1 < N1; i1 += 2)
        for (int i2 = 0; i2 < N2; i2 += 2) {
          for (int j = 0; j < ElementType::DIM; j++)
            CoarseFace[base + (i2 / 2) + (i1 / 2) * N2].member(j) +=
                recv_buffer[r][F.offset + dis + j];
          dis += ElementType::DIM;
        }
#else
      for (int i2 = 0; i2 < N2; i2 += 2) {
        for (int j = 0; j < ElementType::DIM; j++)
          CoarseFace[base + (i2 / 2)].member(j) +=
              recv_buffer[r][F.offset + dis + j];
        dis += ElementType::DIM;
      }
#endif
    }
  }

  /// Perform flux correction for face 'F' and direction encoded by 'code*' (for
  /// data received from other processes)
  void FillCase_2(face &F, int codex, int codey, int codez) {
    BlockInfo &info = *F.infos[1];
    const int icode = F.icode[1];
    const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                         (icode / 9) % 3 - 1};

    if (abs(code[0]) != codex)
      return;
    if (abs(code[1]) != codey)
      return;
    if (abs(code[2]) != codez)
      return;

    const int myFace = abs(code[0]) * std::max(0, code[0]) +
                       abs(code[1]) * (std::max(0, code[1]) + 2) +
                       abs(code[2]) * (std::max(0, code[2]) + 4);
    std::array<long long, 2> temp = {(long long)info.level, info.Z};
    auto search = TFluxCorrection::MapOfCases.find(temp);
    assert(search != TFluxCorrection::MapOfCases.end());
    Case &CoarseCase = (*search->second);
    std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];

    const int d = myFace / 2;
    const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
    const int N2 = CoarseCase.m_vSize[d2];
    BlockType &block = *(BlockType *)info.ptrBlock;
#if DIMENSION == 3
    const int d1 = std::max((d + 1) % 3, (d + 2) % 3);
    const int N1 = CoarseCase.m_vSize[d1];
    if (d == 0) {
      const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeX - 1;
      for (int i1 = 0; i1 < N1; i1++)
        for (int i2 = 0; i2 < N2; i2++) {
          block(j, i2, i1) += CoarseFace[i2 + i1 * N2];
          CoarseFace[i2 + i1 * N2].clear();
        }
    } else if (d == 1) {
      const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeY - 1;
      for (int i1 = 0; i1 < N1; i1++)
        for (int i2 = 0; i2 < N2; i2++) {
          block(i2, j, i1) += CoarseFace[i2 + i1 * N2];
          CoarseFace[i2 + i1 * N2].clear();
        }
    } else {
      const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeZ - 1;
      for (int i1 = 0; i1 < N1; i1++)
        for (int i2 = 0; i2 < N2; i2++) {
          block(i2, i1, j) += CoarseFace[i2 + i1 * N2];
          CoarseFace[i2 + i1 * N2].clear();
        }
    }
#else
    assert(d != 2);
    if (d == 0) {
      const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeX - 1;
      for (int i2 = 0; i2 < N2; i2++) {
        block(j, i2) += CoarseFace[i2];
        CoarseFace[i2].clear();
      }
    } else // if (d == 1)
    {
      const int j = (myFace % 2 == 0) ? 0 : BlockType::sizeY - 1;
      for (int i2 = 0; i2 < N2; i2++) {
        block(i2, j) += CoarseFace[i2];
        CoarseFace[i2].clear();
      }
    }
#endif
  }

public:
  /// Prepare the FluxCorrection class for a given 'grid' by allocating
  /// BlockCases at each coarse-fine interface
  virtual void prepare(TGrid &_grid) override {
    if (_grid.UpdateFluxCorrection == false)
      return;
    _grid.UpdateFluxCorrection = false;

    int temprank;
    MPI_Comm_size(_grid.getWorldComm(), &size);
    MPI_Comm_rank(_grid.getWorldComm(), &temprank);
    TFluxCorrection::rank = temprank;

    send_buffer.resize(size);
    recv_buffer.resize(size);
    send_faces.resize(size);
    recv_faces.resize(size);

    for (int r = 0; r < size; r++) {
      send_faces[r].clear();
      recv_faces[r].clear();
    }

    std::vector<int> send_buffer_size(size, 0);
    std::vector<int> recv_buffer_size(size, 0);

    const int NC = ElementType::DIM;

    int blocksize[3];
    blocksize[0] = BlockType::sizeX;
    blocksize[1] = BlockType::sizeY;
    blocksize[2] = BlockType::sizeZ;

    TFluxCorrection::Cases.clear();
    TFluxCorrection::MapOfCases.clear();

    TFluxCorrection::grid = &_grid;
    std::vector<BlockInfo> &BB = (*TFluxCorrection::grid).getBlocksInfo();

    std::array<int, 3> blocksPerDim = _grid.getMaxBlocks();

    std::array<int, 6> icode = {1 * 2 + 3 * 1 + 9 * 1, 1 * 0 + 3 * 1 + 9 * 1,
                                1 * 1 + 3 * 2 + 9 * 1, 1 * 1 + 3 * 0 + 9 * 1,
                                1 * 1 + 3 * 1 + 9 * 2, 1 * 1 + 3 * 1 + 9 * 0};

    for (auto &info : BB) {
      (*TFluxCorrection::grid).getBlockInfoAll(info.level, info.Z).auxiliary =
          nullptr;
      info.auxiliary = nullptr;

      const int aux = 1 << info.level;

      const bool xskin =
          info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
      const bool yskin =
          info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
      const bool zskin =
          info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;

      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;

      bool storeFace[6] = {false, false, false, false, false, false};
      bool stored = false;

      for (int f = 0; f < 6; f++) {
        const int code[3] = {icode[f] % 3 - 1, (icode[f] / 3) % 3 - 1,
                             (icode[f] / 9) % 3 - 1};

        if (!_grid.xperiodic && code[0] == xskip && xskin)
          continue;
        if (!_grid.yperiodic && code[1] == yskip && yskin)
          continue;
        if (!_grid.zperiodic && code[2] == zskip && zskin)
          continue;
#if DIMENSION == 2
        if (code[2] != 0)
          continue;
#endif

        if (!(*TFluxCorrection::grid)
                 .Tree(info.level, info.Znei_(code[0], code[1], code[2]))
                 .Exists()) {
          storeFace[abs(code[0]) * std::max(0, code[0]) +
                    abs(code[1]) * (std::max(0, code[1]) + 2) +
                    abs(code[2]) * (std::max(0, code[2]) + 4)] = true;
          stored = true;
        }

        int L[3];
        L[0] = (code[0] == 0) ? blocksize[0] / 2 : 1;
        L[1] = (code[1] == 0) ? blocksize[1] / 2 : 1;
#if DIMENSION == 3
        L[2] = (code[2] == 0) ? blocksize[2] / 2 : 1;
#else
        L[2] = 1;
#endif
        int V = L[0] * L[1] * L[2];

        if ((*TFluxCorrection::grid)
                .Tree(info.level, info.Znei_(code[0], code[1], code[2]))
                .CheckCoarser()) {
          BlockInfo &infoNei =
              (*TFluxCorrection::grid)
                  .getBlockInfoAll(info.level,
                                   info.Znei_(code[0], code[1], code[2]));
          const long long nCoarse = infoNei.Zparent;
          BlockInfo &infoNeiCoarser =
              (*TFluxCorrection::grid).getBlockInfoAll(info.level - 1, nCoarse);
          const int infoNeiCoarserrank =
              (*TFluxCorrection::grid).Tree(info.level - 1, nCoarse).rank();
          {
            int code2[3] = {-code[0], -code[1], -code[2]};
            int icode2 =
                (code2[0] + 1) + (code2[1] + 1) * 3 + (code2[2] + 1) * 9;
            send_faces[infoNeiCoarserrank].push_back(
                face(info, infoNeiCoarser, icode[f], icode2));
            send_buffer_size[infoNeiCoarserrank] += V;
          }
        } else if ((*TFluxCorrection::grid)
                       .Tree(info.level, info.Znei_(code[0], code[1], code[2]))
                       .CheckFiner()) {
          BlockInfo &infoNei =
              (*TFluxCorrection::grid)
                  .getBlockInfoAll(info.level,
                                   info.Znei_(code[0], code[1], code[2]));
          int Bstep = 1; // face
#if DIMENSION == 3
          for (int B = 0; B <= 3;
               B += Bstep) // loop over blocks that make up face
#else
          for (int B = 0; B <= 1;
               B += Bstep) // loop over blocks that make up face
#endif
          {
            const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
            const long long nFine =
                infoNei.Zchild[std::max(-code[0], 0) +
                               (B % 2) * std::max(0, 1 - abs(code[0]))]
                              [std::max(-code[1], 0) +
                               temp * std::max(0, 1 - abs(code[1]))]
                              [std::max(-code[2], 0) +
                               (B / 2) * std::max(0, 1 - abs(code[2]))];
            const int infoNeiFinerrank =
                (*TFluxCorrection::grid).Tree(infoNei.level + 1, nFine).rank();
            {
              BlockInfo &infoNeiFiner =
                  (*TFluxCorrection::grid)
                      .getBlockInfoAll(infoNei.level + 1, nFine);
              int icode2 =
                  (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
              recv_faces[infoNeiFinerrank].push_back(
                  face(infoNeiFiner, info, icode2, icode[f]));
              recv_buffer_size[infoNeiFinerrank] += V;
            }
          }
        }
      } // icode = 0,...,26

      if (stored) {
        TFluxCorrection::Cases.push_back(
            Case(storeFace, BlockType::sizeX, BlockType::sizeY,
                 BlockType::sizeZ, info.level, info.Z));
      }
    }

    size_t Cases_index = 0;
    if (TFluxCorrection::Cases.size() > 0)
      for (auto &info : BB) {
        if (Cases_index == TFluxCorrection::Cases.size())
          break;
        if (TFluxCorrection::Cases[Cases_index].level == info.level &&
            TFluxCorrection::Cases[Cases_index].Z == info.Z) {
          TFluxCorrection::MapOfCases.insert(
              std::pair<std::array<long long, 2>, Case *>(
                  {TFluxCorrection::Cases[Cases_index].level,
                   TFluxCorrection::Cases[Cases_index].Z},
                  &TFluxCorrection::Cases[Cases_index]));
          TFluxCorrection::grid
              ->getBlockInfoAll(TFluxCorrection::Cases[Cases_index].level,
                                TFluxCorrection::Cases[Cases_index].Z)
              .auxiliary = &TFluxCorrection::Cases[Cases_index];
          info.auxiliary = &TFluxCorrection::Cases[Cases_index];
          Cases_index++;
        }
      }

    // 2.Sort faces
    for (int r = 0; r < size; r++) {
      std::sort(send_faces[r].begin(), send_faces[r].end());
      std::sort(recv_faces[r].begin(), recv_faces[r].end());
    }

    // 3.Define map
    for (int r = 0; r < size; r++) {
      send_buffer[r].resize(send_buffer_size[r] * NC);
      recv_buffer[r].resize(recv_buffer_size[r] * NC);

      int offset = 0;
      for (int k = 0; k < (int)recv_faces[r].size(); k++) {
        face &f = recv_faces[r][k];

        const int code[3] = {f.icode[1] % 3 - 1, (f.icode[1] / 3) % 3 - 1,
                             (f.icode[1] / 9) % 3 - 1};

        int L[3];
        L[0] = (code[0] == 0) ? blocksize[0] / 2 : 1;
        L[1] = (code[1] == 0) ? blocksize[1] / 2 : 1;
#if DIMENSION == 3
        L[2] = (code[2] == 0) ? blocksize[2] / 2 : 1;
#else
        L[2] = 1;
#endif
        int V = L[0] * L[1] * L[2];

        f.offset = offset;

        offset += V * NC;
      }
    }
  }

  /// Go over each coarse-fine interface and perform the flux corrections,
  /// assuming the associated BlockCases have been filled with the fluxes by the
  /// user
  virtual void FillBlockCases() override {
    auto MPI_real =
        (sizeof(Real) == sizeof(float))
            ? MPI_FLOAT
            : ((sizeof(Real) == sizeof(double)) ? MPI_DOUBLE : MPI_LONG_DOUBLE);

    // This assumes that the BlockCases have been filled by the user somehow...

    // 1.Pack send data
    for (int r = 0; r < size; r++) {

      int displacement = 0;
      for (int k = 0; k < (int)send_faces[r].size(); k++) {
        face &f = send_faces[r][k];

        BlockInfo &info = *(f.infos[0]);

        auto search =
            TFluxCorrection::MapOfCases.find({(long long)info.level, info.Z});
        assert(search != TFluxCorrection::MapOfCases.end());

        Case &FineCase = (*search->second);

        int icode = f.icode[0];
        const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                             (icode / 9) % 3 - 1};
        const int myFace = abs(code[0]) * std::max(0, code[0]) +
                           abs(code[1]) * (std::max(0, code[1]) + 2) +
                           abs(code[2]) * (std::max(0, code[2]) + 4);
        std::vector<ElementType> &FineFace = FineCase.m_pData[myFace];

        const int d = myFace / 2;
        const int d2 = std::min((d + 1) % 3, (d + 2) % 3);
        const int N2 = FineCase.m_vSize[d2];
#if DIMENSION == 3
        const int d1 = std::max((d + 1) % 3, (d + 2) % 3);
        const int N1 = FineCase.m_vSize[d1];
        for (int i1 = 0; i1 < N1; i1 += 2)
          for (int i2 = 0; i2 < N2; i2 += 2) {
            ElementType avg =
                ((FineFace[i2 + i1 * N2] + FineFace[i2 + 1 + i1 * N2]) +
                 (FineFace[i2 + (i1 + 1) * N2] +
                  FineFace[i2 + 1 + (i1 + 1) * N2]));
            for (int j = 0; j < ElementType::DIM; j++)
              send_buffer[r][displacement + j] = avg.member(j);
            displacement += ElementType::DIM;
            FineFace[i2 + i1 * N2].clear();
            FineFace[i2 + 1 + i1 * N2].clear();
            FineFace[i2 + (i1 + 1) * N2].clear();
            FineFace[i2 + 1 + (i1 + 1) * N2].clear();
          }
#else
        for (int i2 = 0; i2 < N2; i2 += 2) {
          ElementType avg = FineFace[i2] + FineFace[i2 + 1];
          for (int j = 0; j < ElementType::DIM; j++)
            send_buffer[r][displacement + j] = avg.member(j);
          displacement += ElementType::DIM;
          FineFace[i2].clear();
          FineFace[i2 + 1].clear();
        }
#endif
      }
    }

    std::vector<MPI_Request> send_requests;
    std::vector<MPI_Request> recv_requests;

    const int me = TFluxCorrection::rank;
    for (int r = 0; r < size; r++)
      if (r != me) {
        if (recv_buffer[r].size() != 0) {
          MPI_Request req{};
          recv_requests.push_back(req);
          MPI_Irecv(&recv_buffer[r][0], recv_buffer[r].size(), MPI_real, r,
                    123456, (*TFluxCorrection::grid).getWorldComm(),
                    &recv_requests.back());
        }
        if (send_buffer[r].size() != 0) {
          MPI_Request req{};
          send_requests.push_back(req);
          MPI_Isend(&send_buffer[r][0], send_buffer[r].size(), MPI_real, r,
                    123456, (*TFluxCorrection::grid).getWorldComm(),
                    &send_requests.back());
        }
      }

    MPI_Request me_send_request;
    MPI_Request me_recv_request;
    if (recv_buffer[me].size() != 0) {
      MPI_Irecv(&recv_buffer[me][0], recv_buffer[me].size(), MPI_real, me,
                123456, (*TFluxCorrection::grid).getWorldComm(),
                &me_recv_request);
    }
    if (send_buffer[me].size() != 0) {
      MPI_Isend(&send_buffer[me][0], send_buffer[me].size(), MPI_real, me,
                123456, (*TFluxCorrection::grid).getWorldComm(),
                &me_send_request);
    }

    if (recv_buffer[me].size() > 0)
      MPI_Waitall(1, &me_recv_request, MPI_STATUSES_IGNORE);
    if (send_buffer[me].size() > 0)
      MPI_Waitall(1, &me_send_request, MPI_STATUSES_IGNORE);

    for (int index = 0; index < (int)recv_faces[me].size(); index++)
      FillCase(recv_faces[me][index]);

    if (recv_requests.size() > 0)
      MPI_Waitall(recv_requests.size(), &recv_requests[0], MPI_STATUSES_IGNORE);

    for (int r = 0; r < size; r++)
      if (r != me)
        for (int index = 0; index < (int)recv_faces[r].size(); index++)
          FillCase(recv_faces[r][index]);

    // first do x, then y then z. It is done like this to preserve symmetry and
    // not favor any direction
    for (int r = 0; r < size; r++) // if (r!=me)
      for (int index = 0; index < (int)recv_faces[r].size(); index++)
        FillCase_2(recv_faces[r][index], 1, 0, 0);
    for (int r = 0; r < size; r++) // if (r!=me)
      for (int index = 0; index < (int)recv_faces[r].size(); index++)
        FillCase_2(recv_faces[r][index], 0, 1, 0);
#if DIMENSION == 3
    for (int r = 0; r < size; r++) // if (r!=me)
      for (int index = 0; index < (int)recv_faces[r].size(); index++)
        FillCase_2(recv_faces[r][index], 0, 0, 1);
#endif

    if (send_requests.size() > 0)
      MPI_Waitall(send_requests.size(), &send_requests[0], MPI_STATUSES_IGNORE);
  }
};

} // namespace cubism

namespace cubism {

/** Similar to Grid, but with functionalities for multiple MPI processes.
 */
template <typename TGrid> class GridMPI : public TGrid {
public:
  typedef typename TGrid::Real Real;
  typedef typename TGrid::BlockType Block;
  typedef typename TGrid::BlockType BlockType;
  typedef SynchronizerMPI_AMR<Real, GridMPI<TGrid>> SynchronizerMPIType;

  // MPI related variables
  size_t timestamp;   ///< used as message tag during communication
  MPI_Comm worldcomm; ///< MPI communicator
  int myrank;         ///< MPI process ID
  int world_size;     ///< total number of MPI processes

  std::map<StencilInfo, SynchronizerMPIType *>
      SynchronizerMPIs; ///< map of Syncronizers need for halo cell exchange
  FluxCorrectionMPI<FluxCorrection<GridMPI<TGrid>>> Corrector;
  std::vector<BlockInfo *> boundary; ///< BlockInfos of adjacent ranks

  /// Constructor, same as the one from Grid.
  GridMPI(const int nX, const int nY = 1, const int nZ = 1,
          const double a_maxextent = 1, const int a_levelStart = 0,
          const int a_levelMax = 1, const MPI_Comm comm = MPI_COMM_WORLD,
          const bool a_xperiodic = true, const bool a_yperiodic = true,
          const bool a_zperiodic = true)
      : TGrid(nX, nY, nZ, a_maxextent, a_levelStart, a_levelMax, false,
              a_xperiodic, a_yperiodic, a_zperiodic),
        timestamp(0), worldcomm(comm) {
    MPI_Comm_size(worldcomm, &world_size);
    MPI_Comm_rank(worldcomm, &myrank);

    const long long total_blocks =
        nX * nY * nZ * pow(pow(2, a_levelStart), DIMENSION);
    long long my_blocks = total_blocks / world_size;
    if ((long long)myrank < total_blocks % world_size)
      my_blocks++;
    long long n_start = myrank * (total_blocks / world_size);
    if (total_blocks % world_size > 0) {
      if ((long long)myrank < total_blocks % world_size)
        n_start += myrank;
      else
        n_start += total_blocks % world_size;
    }

    std::vector<short int> levels(my_blocks, a_levelStart);
    std::vector<long long> Zs(my_blocks);
    for (long long n = n_start; n < n_start + my_blocks; n++)
      Zs[n - n_start] = n;
    initialize_blocks(Zs, levels);

    if (myrank == 0) {
      std::cout << "Total blocks = " << total_blocks << ", each rank gets "
                << my_blocks << std::endl;
    }
    MPI_Barrier(worldcomm);
  }

  /// Destructor.
  virtual ~GridMPI() override {
    for (auto it = SynchronizerMPIs.begin(); it != SynchronizerMPIs.end(); ++it)
      delete it->second;

    SynchronizerMPIs.clear();

    MPI_Barrier(worldcomm);
  }

  /// Return pointer to block at level 'm' with Z-index 'n' or nullptr if this
  /// block is not owned by this rank.
  virtual Block *avail(const int m, const long long n) override {
    return (TGrid::Tree(m, n).rank() == myrank)
               ? (Block *)TGrid::getBlockInfoAll(m, n).ptrBlock
               : nullptr;
  }

  /// Communicate the state (refine/compress/leave) of all blocks in the
  /// boundaries of this rank; Used when the mesh is refined, to make sure we
  /// all adjacent blocks do not differ by more than one refinement level.
  virtual void UpdateBoundary(bool clean = false) override {
    const auto blocksPerDim = TGrid::getMaxBlocks();

    int rank, size;
    MPI_Comm_rank(worldcomm, &rank);
    MPI_Comm_size(worldcomm, &size);

    std::vector<std::vector<long long>> send_buffer(size);

    std::vector<BlockInfo *> &bbb = boundary;
    std::set<int> Neighbors;
    for (size_t jjj = 0; jjj < bbb.size(); jjj++) {
      BlockInfo &info = *bbb[jjj];

      std::set<int> receivers;

      const int aux = 1 << info.level;
      const bool xskin =
          info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
      const bool yskin =
          info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
      const bool zskin =
          info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;
      for (int icode = 0; icode < 27; icode++) {
        if (icode == 1 * 1 + 3 * 1 + 9 * 1)
          continue;

        const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                             (icode / 9) % 3 - 1};

        if (!TGrid::xperiodic && code[0] == xskip && xskin)
          continue;
        if (!TGrid::yperiodic && code[1] == yskip && yskin)
          continue;
        if (!TGrid::zperiodic && code[2] == zskip && zskin)
          continue;
#if DIMENSION == 2
        if (code[2] != 0)
          continue;
#endif

        BlockInfo &infoNei = TGrid::getBlockInfoAll(
            info.level, info.Znei_(code[0], code[1], code[2]));

        const TreePosition &infoNeiTree = TGrid::Tree(infoNei.level, infoNei.Z);
        if (infoNeiTree.Exists() && infoNeiTree.rank() != rank) {
          if (infoNei.state != Refine || clean)
            infoNei.state = Leave;
          receivers.insert(infoNeiTree.rank());
          Neighbors.insert(infoNeiTree.rank());
        } else if (infoNeiTree.CheckCoarser()) {
          const long long nCoarse = infoNei.Zparent;
          BlockInfo &infoNeiCoarser =
              TGrid::getBlockInfoAll(infoNei.level - 1, nCoarse);
          const int infoNeiCoarserrank =
              TGrid::Tree(infoNei.level - 1, nCoarse).rank();
          if (infoNeiCoarserrank != rank) {
            assert(infoNeiCoarserrank >= 0);
            if (infoNeiCoarser.state != Refine || clean)
              infoNeiCoarser.state = Leave;
            receivers.insert(infoNeiCoarserrank);
            Neighbors.insert(infoNeiCoarserrank);
          }
        } else if (infoNeiTree.CheckFiner()) {
          int Bstep = 1; // face
          if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))
            Bstep = 3; // edge
          else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
            Bstep = 4; // corner

#if DIMENSION == 3
          for (int B = 0; B <= 3;
               B += Bstep) // loop over blocks that make up face/edge/corner
                           // (respectively 4,2 or 1 blocks)
#else
          for (int B = 0; B <= 1; B += Bstep)
#endif
          {
            const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
            const long long nFine =
                infoNei.Zchild[std::max(-code[0], 0) +
                               (B % 2) * std::max(0, 1 - abs(code[0]))]
                              [std::max(-code[1], 0) +
                               temp * std::max(0, 1 - abs(code[1]))]
                              [std::max(-code[2], 0) +
                               (B / 2) * std::max(0, 1 - abs(code[2]))];

            BlockInfo &infoNeiFiner =
                TGrid::getBlockInfoAll(infoNei.level + 1, nFine);
            const int infoNeiFinerrank =
                TGrid::Tree(infoNei.level + 1, nFine).rank();
            if (infoNeiFinerrank != rank) {
              if (infoNeiFiner.state != Refine || clean)
                infoNeiFiner.state = Leave;
              receivers.insert(infoNeiFinerrank);
              Neighbors.insert(infoNeiFinerrank);
            }
          }
        }
      } // icode = 0,...,26

      if (info.changed2 && info.state != Leave) {
        if (info.state == Refine)
          info.changed2 = false;

        std::set<int>::iterator it = receivers.begin();
        while (it != receivers.end()) {
          int temp = (info.state == Compress) ? 1 : 2;
          send_buffer[*it].push_back(info.level);
          send_buffer[*it].push_back(info.Z);
          send_buffer[*it].push_back(temp);
          it++;
        }
      }
    }

    std::vector<MPI_Request> requests;

    long long dummy = 0;
    for (int r : Neighbors)
      if (r != rank) {
        requests.resize(requests.size() + 1);
        if (send_buffer[r].size() != 0)
          MPI_Isend(&send_buffer[r][0], send_buffer[r].size(), MPI_LONG_LONG, r,
                    123, worldcomm, &requests[requests.size() - 1]);
        else {
          MPI_Isend(&dummy, 1, MPI_LONG_LONG, r, 123, worldcomm,
                    &requests[requests.size() - 1]);
        }
      }

    std::vector<std::vector<long long>> recv_buffer(size);
    for (int r : Neighbors)
      if (r != rank) {
        int recv_size;
        MPI_Status status;
        MPI_Probe(r, 123, worldcomm, &status);
        MPI_Get_count(&status, MPI_LONG_LONG, &recv_size);
        if (recv_size > 0) {
          recv_buffer[r].resize(recv_size);
          requests.resize(requests.size() + 1);
          MPI_Irecv(&recv_buffer[r][0], recv_buffer[r].size(), MPI_LONG_LONG, r,
                    123, worldcomm, &requests[requests.size() - 1]);
        }
      }

    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    for (int r = 0; r < size; r++)
      if (recv_buffer[r].size() > 1)
        for (int index = 0; index < (int)recv_buffer[r].size(); index += 3) {
          int level = recv_buffer[r][index];
          long long Z = recv_buffer[r][index + 1];
          TGrid::getBlockInfoAll(level, Z).state =
              (recv_buffer[r][index + 2] == 1) ? Compress : Refine;
        }
  };

  /// Called after grid refinement/compression, to update the Octree with the
  /// new block states and rank ownership.
  void UpdateBlockInfoAll_States(bool UpdateIDs = false) {
    std::vector<int> myNeighbors = FindMyNeighbors();

    const auto blocksPerDim = TGrid::getMaxBlocks();
    std::vector<long long> myData;
    for (auto &info : TGrid::m_vInfo) {
      bool myflag = false;

      const int aux = 1 << info.level;
      const bool xskin =
          info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
      const bool yskin =
          info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
      const bool zskin =
          info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;
      for (int icode = 0; icode < 27; icode++) {
        if (icode == 1 * 1 + 3 * 1 + 9 * 1)
          continue;

        const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                             (icode / 9) % 3 - 1};

        if (!TGrid::xperiodic && code[0] == xskip && xskin)
          continue;
        if (!TGrid::yperiodic && code[1] == yskip && yskin)
          continue;
        if (!TGrid::zperiodic && code[2] == zskip && zskin)
          continue;
#if DIMENSION == 2
        if (code[2] != 0)
          continue;
#endif

        BlockInfo &infoNei = TGrid::getBlockInfoAll(
            info.level, info.Znei_(code[0], code[1], code[2]));

        const TreePosition &infoNeiTree = TGrid::Tree(infoNei.level, infoNei.Z);
        if (infoNeiTree.Exists() && infoNeiTree.rank() != myrank) {
          myflag = true;
          break;
        } else if (infoNeiTree.CheckCoarser()) {
          long long nCoarse = infoNei.Zparent;
          const int infoNeiCoarserrank =
              TGrid::Tree(infoNei.level - 1, nCoarse).rank();
          if (infoNeiCoarserrank != myrank) {
            myflag = true;
            break;
          }
        } else if (infoNeiTree.CheckFiner()) {
          int Bstep = 1; // face
          if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))
            Bstep = 3; // edge
          else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
            Bstep = 4; // corner

          for (int B = 0; B <= 3;
               B += Bstep) // loop over blocks that make up face/edge/corner
                           // (respectively 4,2 or 1 blocks)
          {
            const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
            const long long nFine =
                infoNei.Zchild[std::max(-code[0], 0) +
                               (B % 2) * std::max(0, 1 - abs(code[0]))]
                              [std::max(-code[1], 0) +
                               temp * std::max(0, 1 - abs(code[1]))]
                              [std::max(-code[2], 0) +
                               (B / 2) * std::max(0, 1 - abs(code[2]))];
            const int infoNeiFinerrank =
                TGrid::Tree(infoNei.level + 1, nFine).rank();
            if (infoNeiFinerrank != myrank) {
              myflag = true;
              break;
            }
          }
        } else if (infoNeiTree.rank() < 0) {
          myflag = true;
          break;
        }
      } // icode = 0,...,26

      if (myflag) {
        myData.push_back(info.level);
        myData.push_back(info.Z);
        if (UpdateIDs)
          myData.push_back(info.blockID);
      }
    }

    std::vector<std::vector<long long>> recv_buffer(myNeighbors.size());
    std::vector<std::vector<long long>> send_buffer(myNeighbors.size());
    std::vector<int> recv_size(myNeighbors.size());

    std::vector<MPI_Request> size_requests(2 * myNeighbors.size());

    int mysize = (int)myData.size();
    int kk = 0;
    for (auto r : myNeighbors) {
      MPI_Irecv(&recv_size[kk], 1, MPI_INT, r, timestamp, worldcomm,
                &size_requests[2 * kk]);
      MPI_Isend(&mysize, 1, MPI_INT, r, timestamp, worldcomm,
                &size_requests[2 * kk + 1]);
      kk++;
    }
    kk = 0;
    for (size_t j = 0; j < myNeighbors.size(); j++) {
      send_buffer[kk].resize(myData.size());
      for (size_t i = 0; i < myData.size(); i++)
        send_buffer[kk][i] = myData[i];
      kk++;
    }

    MPI_Waitall(size_requests.size(), size_requests.data(),
                MPI_STATUSES_IGNORE);
    std::vector<MPI_Request> requests(2 * myNeighbors.size());
    kk = 0;
    for (auto r : myNeighbors) {
      recv_buffer[kk].resize(recv_size[kk]);
      MPI_Irecv(recv_buffer[kk].data(), recv_buffer[kk].size(), MPI_LONG_LONG,
                r, timestamp, worldcomm, &requests[2 * kk]);
      MPI_Isend(send_buffer[kk].data(), send_buffer[kk].size(), MPI_LONG_LONG,
                r, timestamp, worldcomm, &requests[2 * kk + 1]);
      kk++;
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    kk = -1;
    const int increment = UpdateIDs ? 3 : 2;
    for (auto r : myNeighbors) {
      kk++;
      for (size_t index__ = 0; index__ < recv_buffer[kk].size();
           index__ += increment) {
        const int level = (int)recv_buffer[kk][index__];
        const long long Z = recv_buffer[kk][index__ + 1];
        TGrid::Tree(level, Z).setrank(r);
        if (UpdateIDs)
          TGrid::getBlockInfoAll(level, Z).blockID =
              recv_buffer[kk][index__ + 2];
#if DIMENSION == 3
        int p[3];
        BlockInfo::inverse(Z, level, p[0], p[1], p[2]);

        if (level < TGrid::levelMax - 1)
          for (int k = 0; k < 2; k++)
            for (int j = 0; j < 2; j++)
              for (int i = 0; i < 2; i++) {
                const long long nc = TGrid::getZforward(
                    level + 1, 2 * p[0] + i, 2 * p[1] + j, 2 * p[2] + k);
                TGrid::Tree(level + 1, nc).setCheckCoarser();
              }
        if (level > 0) {
          const long long nf =
              TGrid::getZforward(level - 1, p[0] / 2, p[1] / 2, p[2] / 2);
          TGrid::Tree(level - 1, nf).setCheckFiner();
        }
#else
        int p[2];
        BlockInfo::inverse(Z, level, p[0], p[1]);
        if (level < TGrid::levelMax - 1)
          for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++) {
              const long long nc =
                  TGrid::getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
              TGrid::Tree(level + 1, nc).setCheckCoarser();
            }
        if (level > 0) {
          const long long nf =
              TGrid::getZforward(level - 1, p[0] / 2, p[1] / 2);
          TGrid::Tree(level - 1, nf).setCheckFiner();
        }
#endif
      }
    }
  }

  /// Returns a vector with the process IDs (ranks) or neighboring GridMPIs.
  std::vector<int> FindMyNeighbors() {
    std::vector<int> myNeighbors;
    double low[3] = {+1e20, +1e20, +1e20};
    double high[3] = {-1e20, -1e20, -1e20};
    double p_low[3];
    double p_high[3];
    for (auto &info : TGrid::m_vInfo) {
#if DIMENSION == 3
      const double h = 2 * info.h;
      info.pos(p_low, 0, 0, 0);
      info.pos(p_high, Block::sizeX - 1, Block::sizeY - 1, Block::sizeZ - 1);
      p_low[0] -= h;
      p_low[1] -= h;
      p_low[2] -= h;
      p_high[0] += h;
      p_high[1] += h;
      p_high[2] += h;
      low[0] = std::min(low[0], p_low[0]);
      low[1] = std::min(low[1], p_low[1]);
      low[2] = std::min(low[2], p_low[2]);
      high[0] = std::max(high[0], p_high[0]);
      high[1] = std::max(high[1], p_high[1]);
      high[2] = std::max(high[2], p_high[2]);
#else
      const double h = 2 * info.h;
      info.pos(p_low, 0, 0);
      info.pos(p_high, Block::sizeX - 1, Block::sizeY - 1);
      p_low[0] -= h;
      p_low[1] -= h;
      p_low[2] = 0;
      p_high[0] += h;
      p_high[1] += h;
      p_high[2] = 0;
      low[0] = std::min(low[0], p_low[0]);
      low[1] = std::min(low[1], p_low[1]);
      low[2] = 0;
      high[0] = std::max(high[0], p_high[0]);
      high[1] = std::max(high[1], p_high[1]);
      high[2] = 0;
#endif
    }
    std::vector<double> all_boxes(world_size * 6);
    double my_box[6] = {low[0], low[1], low[2], high[0], high[1], high[2]};
    MPI_Allgather(my_box, 6, MPI_DOUBLE, all_boxes.data(), 6, MPI_DOUBLE,
                  worldcomm);
    for (int i = 0; i < world_size; i++) {
      if (i == myrank)
        continue;
      if (Intersect(low, high, &all_boxes[i * 6], &all_boxes[i * 6 + 3]))
        myNeighbors.push_back(i);
    }
    return myNeighbors;
  }

  /// Check if a rectangle with bottom left point l1 and top right point h1
  /// intersects with a rectangle with bottom left point l2 and top right point
  /// h2; used when determining neighboring processes.
  bool Intersect(double *l1, double *h1, double *l2, double *h2) {
    const double h0 =
        (TGrid::maxextent / std::max(TGrid::NX * Block::sizeX,
                                     std::max(TGrid::NY * Block::sizeY,
                                              TGrid::NZ * Block::sizeZ)));
    const double extent[3] = {TGrid::NX * Block::sizeX * h0,
                              TGrid::NY * Block::sizeY * h0,
                              TGrid::NZ * Block::sizeZ * h0};

    const Real intersect[3][2] = {
        {std::max(l1[0], l2[0]), std::min(h1[0], h2[0])},
        {std::max(l1[1], l2[1]), std::min(h1[1], h2[1])},
        {std::max(l1[2], l2[2]), std::min(h1[2], h2[2])}};

    bool intersection[3];
    intersection[0] = intersect[0][1] - intersect[0][0] > 0.0;
    intersection[1] = intersect[1][1] - intersect[1][0] > 0.0;
    intersection[2] =
        DIMENSION == 3 ? (intersect[2][1] - intersect[2][0] > 0.0) : true;
    const bool isperiodic[3] = {TGrid::xperiodic, TGrid::yperiodic,
                                TGrid::zperiodic};
    for (int d = 0; d < DIMENSION; d++) {
      if (isperiodic[d]) {
        if (h2[d] > extent[d])
          intersection[d] = std::min(h1[d], h2[d] - extent[d]) -
                            std::max(l1[d], l2[d] - extent[d]);
        else if (h1[d] > extent[d])
          intersection[d] = std::min(h2[d], h1[d] - extent[d]) -
                            std::max(l2[d], l1[d] - extent[d]);
      }
      if (!intersection[d])
        return false;
    }
    return true;
  }

  /** Returns a SynchronizerMPI_AMR for a given stencil of points.
   *  Each stencil needed in the simulation has its own SynchronizerMPI_AMR. All
   * Synchronizers are owned by GridMPI in a map between the stencils and them.
   */
  SynchronizerMPIType *sync(const StencilInfo &stencil) {
    assert(stencil.isvalid());

    // Hardcoded stencil for coarse-fine interpolation: +-1 points.
    StencilInfo Cstencil(-1, -1, DIMENSION == 3 ? -1 : 0, 2, 2,
                         DIMENSION == 3 ? 2 : 1, true, stencil.selcomponents);

    SynchronizerMPIType *queryresult = nullptr;

    typename std::map<StencilInfo, SynchronizerMPIType *>::iterator
        itSynchronizerMPI = SynchronizerMPIs.find(stencil);

    if (itSynchronizerMPI == SynchronizerMPIs.end()) {
      queryresult = new SynchronizerMPIType(stencil, Cstencil, this);
      queryresult->_Setup();
      SynchronizerMPIs[stencil] = queryresult;
    } else {
      queryresult = itSynchronizerMPI->second;
    }

    queryresult->sync();

    timestamp = (timestamp + 1) % 32768;
    return queryresult;
  }

  /// same as Grid::initialize_blocks, with additional initialization for
  /// Synchronizers needed for halo cell exchange between different processes.
  virtual void
  initialize_blocks(const std::vector<long long> &blocksZ,
                    const std::vector<short int> &blockslevel) override {
    TGrid::initialize_blocks(blocksZ, blockslevel);
    UpdateBlockInfoAll_States(false);
    for (auto it = SynchronizerMPIs.begin(); it != SynchronizerMPIs.end(); ++it)
      (*it->second)._Setup();
  }

  /// Return the ID of this MPI process.
  virtual int rank() const override { return myrank; }

  /// Returns a tag value that is used when sending/receiving data with MPI.
  size_t getTimeStamp() const { return timestamp; }

  /// Return the MPI communicator of the simulation.
  MPI_Comm getWorldComm() const { return worldcomm; }

  /// Return the total number of MPI processes.
  virtual int get_world_size() const override { return world_size; }
};

} // namespace cubism

namespace cubism {

/**
 * A wrapper class for a 3D array of data.
 * @tparam DataType: the kind of data the 3D array is for
 * @tparam allocator: object responsible for allocating the data
 */
template <class DataType, template <typename T> class allocator>
class Matrix3D {
private:
  DataType *m_pData{nullptr}; ///< pointer to data
  unsigned int m_vSize[3]{
      0, 0, 0}; ///< three dimensions (X,Y,Z) (sizes) of array of data
  unsigned int m_nElements{0}; ///< total number of elements saved (XxYxZ)
  unsigned int m_nElementsPerSlice{0}; ///< shorthand for XxY

public:
  /// Deallocate existing data.
  void _Release() {
    if (m_pData != nullptr) {
      free(m_pData);
      m_pData = nullptr;
    }
  }

  /// Deallocate existing data and reallocate memory for a nSizeX x nSizeY x
  /// nSizeZ array.
  void _Setup(unsigned int nSizeX, unsigned int nSizeY, unsigned int nSizeZ) {
    _Release();

    m_vSize[0] = nSizeX;
    m_vSize[1] = nSizeY;
    m_vSize[2] = nSizeZ;

    m_nElementsPerSlice = nSizeX * nSizeY;

    m_nElements = nSizeX * nSizeY * nSizeZ;

    posix_memalign((void **)&m_pData, std::max(8, CUBISM_ALIGNMENT),
                   sizeof(DataType) * m_nElements);
    assert(m_pData != nullptr);
  }

  /// Destructor.
  ~Matrix3D() { _Release(); }

  /// Constructor, calls _Setup()
  Matrix3D(unsigned int nSizeX, unsigned int nSizeY, unsigned int nSizeZ)
      : m_pData(nullptr), m_nElements(0), m_nElementsPerSlice(0) {
    _Setup(nSizeX, nSizeY, nSizeZ);
  }

  /// Constructor, does not allocate memory.
  Matrix3D() : m_pData(nullptr), m_nElements(-1), m_nElementsPerSlice(-1) {}

  Matrix3D(const Matrix3D &m) = delete;

  /// Copy constructor.
  Matrix3D(Matrix3D &&m)
      : m_pData{m.m_pData}, m_vSize{m.m_vSize[0], m.m_vSize[1], m.m_vSize[2]},
        m_nElements{m.m_nElements}, m_nElementsPerSlice{m.m_nElementsPerSlice} {
    m.m_pData = nullptr;
  }

  /// Copy another matrix3D to this one
  inline Matrix3D &operator=(const Matrix3D &m) {
#ifndef NDEBUG
    assert(m_vSize[0] == m.m_vSize[0]);
    assert(m_vSize[1] == m.m_vSize[1]);
    assert(m_vSize[2] == m.m_vSize[2]);
#endif
    for (unsigned int i = 0; i < m_nElements; i++)
      m_pData[i] = m.m_pData[i];
    return *this;
  }

  /// Set all elements to a given element of the same datatype
  inline Matrix3D &operator=(DataType d) {
    for (unsigned int i = 0; i < m_nElements; i++)
      m_pData[i] = d;

    return *this;
  }

  /// Set all elements to a number, applicable only is data is doubles/floats
  inline Matrix3D &operator=(const double a) {
    for (unsigned int i = 0; i < m_nElements; i++)
      m_pData[i].set(a);
    return *this;
  }

  /// Access an element.
  inline DataType &Access(unsigned int ix, unsigned int iy,
                          unsigned int iz) const {
#ifndef NDEBUG
    assert(ix < m_vSize[0]);
    assert(iy < m_vSize[1]);
    assert(iz < m_vSize[2]);
#endif
    return m_pData[iz * m_nElementsPerSlice + iy * m_vSize[0] + ix];
  }

  /// Read an element withoud changing it.
  inline const DataType &Read(unsigned int ix, unsigned int iy,
                              unsigned int iz) const {
#ifndef NDEBUG
    assert(ix < m_vSize[0]);
    assert(iy < m_vSize[1]);
    assert(iz < m_vSize[2]);
#endif
    return m_pData[iz * m_nElementsPerSlice + iy * m_vSize[0] + ix];
  }

  /// Access elements of the array in sequential order, useful for pointwise
  /// operations
  inline DataType &LinAccess(unsigned int i) const {
#ifndef NDEBUG
    assert(i < m_nElements);
#endif
    return m_pData[i];
  }

  /// Get total number of elements of the array
  inline unsigned int getNumberOfElements() const { return m_nElements; }

  /// Get elements on each XY slice/plane of the array
  inline unsigned int getNumberOfElementsPerSlice() const {
    return m_nElementsPerSlice;
  }

  /// Get array of sizes for data
  inline unsigned int *getSize() const { return (unsigned int *)m_vSize; }

  /// Get array of size in the 'dim' direction
  inline unsigned int getSize(int dim) const { return m_vSize[dim]; }
};

} // namespace cubism

namespace cubism {
#define memcpy2(a, b, c) memcpy((a), (b), (c))

// default coarse-fine interpolation stencil
#if DIMENSION == 3
constexpr int default_start[3] = {-1, -1, -1};
constexpr int default_end[3] = {2, 2, 2};
#else
constexpr int default_start[3] = {-1, -1, 0};
constexpr int default_end[3] = {2, 2, 1};
#endif

/** \brief Copy of a Gridblock plus halo cells.*/
/** This class provides the user a copy of a Gridblock that is extended by a
 * layer of halo cells. To define one instance of it, the user needs to provide
 * a 'TGrid' type in the template parameters. From this, the BlockType and
 * ElementType and inferred, which are the GridBlock class and Element type
 * stored at each gridpoint of the mesh. To use a BlockLab, the user first needs
 * to call 'prepare', which will provide the BlockLab with the stencil of points
 * needed for a particular computation. To get an array of a particular
 *  GridBlock (+halo cells), the user should call 'load' and provide it with the
 * BlockInfo that is associated with the GridBlock of interest. Once this is
 * done, gridpoints in the GridBlock and halo cells can be accessed with the
 * (x,y,z) operator. For example, (-1,0,0) would access a halo cell in the -x
 * direction.
 *  @tparam TGrid: the kind of Grid/GridMPI halo cells are needed for
 *  @tparam allocator: a class responsible for allocation of memory for this
 * BlockLab
 */
template <typename TGrid,
          template <typename X> class allocator = std::allocator>
class BlockLab {
public:
  using GridType = TGrid; ///< should be a 'Grid', 'GridMPI' or derived class
  using BlockType =
      typename GridType::BlockType; ///< GridBlock type used by TGrid
  using ElementType =
      typename BlockType::ElementType; ///< Element type used by GridBlock type
  using Real = typename ElementType::RealType; ///< Number type used by Element
                                               ///< (double/float etc.)

protected:
  Matrix3D<ElementType, allocator>
      *m_cacheBlock;     ///< working array of GridBlock + halo cells.
  int m_stencilStart[3]; ///< starts of stencil for halo cells
  int m_stencilEnd[3];   ///< ends of stencil fom halo cells
  bool istensorial;      ///< whether the stencil is tensorial or not (see also
                         ///< StencilInfo struct)
  bool use_averages;     ///< if true, fine blocks average down their cells to
                         ///< provide halo cells for coarse blocks (2nd order
                     ///< accurate). If false, they perform a 3rd-order accurate
                     ///< interpolation instead (which is the accuracy needed to
                     ///< compute 2nd derivatives).
  GridType *m_refGrid; ///< Point to TGrid instance
  int NX;              ///< GridBlock size in the x-direction.
  int NY;              ///< GridBlock size in the y-direction.
  int NZ;              ///< GridBlock size in the z-direction.
  std::array<BlockType *, 27>
      myblocks; ///< Pointers to neighboring blocks of a GridBlock
  std::array<int, 27> coarsened_nei_codes; ///< If a neighbor is at a coarser
                                           ///< level, store it here
  int coarsened_nei_codes_size;            ///< Number of coarser neighbors
  int offset[3]; ///< like m_stencilStart but used when a coarse block sends
                 ///< cells to a finer block
  Matrix3D<ElementType, allocator>
      *m_CoarsenedBlock;       ///< coarsened version of given block
  int m_InterpStencilStart[3]; ///< stencil starts used for refinement (assumed
                               ///< tensorial)
  int m_InterpStencilEnd[3];   ///< stencil ends used for refinement (assumed
                               ///< tensorial)
  bool coarsened;         ///< true if block has at least one coarser neighbor
  int CoarseBlockSize[3]; ///< size of coarsened block (NX/2,NY/2,NZ/2)

  /// Coefficients used with upwind/central stencil of points with 3rd order
  /// interpolation of halo cells from fine to coarse blocks
  const double d_coef_plus[9] = {
      -0.09375, 0.4375,  0.15625,  // starting point (+2,+1,0)
      0.15625,  -0.5625, 0.90625,  // last point     (-2,-1,0)
      -0.09375, 0.4375,  0.15625}; // central point  (-1,0,+1)
  /// Coefficients used with upwind/central stencil of points with 3rd order
  /// interpolation of halo cells from fine to coarse blocks
  const double d_coef_minus[9] = {
      0.15625,  -0.5625, 0.90625,   // starting point (+2,+1,0)
      -0.09375, 0.4375,  0.15625,   // last point     (-2,-1,0)
      0.15625,  0.4375,  -0.09375}; // central point  (-1,0,+1)

public:
  /// Constructor.
  BlockLab()
      : m_cacheBlock(nullptr), m_refGrid(nullptr), m_CoarsenedBlock(nullptr) {
    m_stencilStart[0] = m_stencilStart[1] = m_stencilStart[2] = 0;
    m_stencilEnd[0] = m_stencilEnd[1] = m_stencilEnd[2] = 0;
    m_InterpStencilStart[0] = m_InterpStencilStart[1] =
        m_InterpStencilStart[2] = 0;
    m_InterpStencilEnd[0] = m_InterpStencilEnd[1] = m_InterpStencilEnd[2] = 0;

    CoarseBlockSize[0] = (int)BlockType::sizeX / 2;
    CoarseBlockSize[1] = (int)BlockType::sizeY / 2;
    CoarseBlockSize[2] = (int)BlockType::sizeZ / 2;
    if (CoarseBlockSize[0] == 0)
      CoarseBlockSize[0] = 1;
    if (CoarseBlockSize[1] == 0)
      CoarseBlockSize[1] = 1;
    if (CoarseBlockSize[2] == 0)
      CoarseBlockSize[2] = 1;
  }

  /// Return a name for this BlockLab. Useful for derived instances with custom
  /// boundary conditions.
  virtual std::string name() const { return "BlockLab"; }

  /// true if boundary conditions are periodic in x-direction
  virtual bool is_xperiodic() { return true; }

  /// true if boundary conditions are periodic in y-direction
  virtual bool is_yperiodic() { return true; }

  /// true if boundary conditions are periodic in z-direction
  virtual bool is_zperiodic() { return true; }

  /// Destructor.
  ~BlockLab() {
    _release(m_cacheBlock);
    _release(m_CoarsenedBlock);
  }

  /**
   * Get a single element from the block.
   * stencil_start and stencil_end refer to the values passed in
   * BlockLab::prepare().
   * @param ix: Index in x-direction (stencil_start[0] <= ix < BlockType::sizeX
   * + stencil_end[0] - 1).
   * @param iy: Index in y-direction (stencil_start[1] <= iy < BlockType::sizeY
   * + stencil_end[1] - 1).
   * @param iz: Index in z-direction (stencil_start[2] <= iz < BlockType::sizeZ
   * + stencil_end[2] - 1).
   */
  ElementType &operator()(int ix, int iy = 0, int iz = 0) {
    assert(ix - m_stencilStart[0] >= 0 &&
           ix - m_stencilStart[0] < (int)m_cacheBlock->getSize()[0]);
    assert(iy - m_stencilStart[1] >= 0 &&
           iy - m_stencilStart[1] < (int)m_cacheBlock->getSize()[1]);
    assert(iz - m_stencilStart[2] >= 0 &&
           iz - m_stencilStart[2] < (int)m_cacheBlock->getSize()[2]);
    return m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1],
                                iz - m_stencilStart[2]);
  }

  /// Just as BlockLab::operator() but const.
  const ElementType &operator()(int ix, int iy = 0, int iz = 0) const {
    assert(ix - m_stencilStart[0] >= 0 &&
           ix - m_stencilStart[0] < (int)m_cacheBlock->getSize()[0]);
    assert(iy - m_stencilStart[1] >= 0 &&
           iy - m_stencilStart[1] < (int)m_cacheBlock->getSize()[1]);
    assert(iz - m_stencilStart[2] >= 0 &&
           iz - m_stencilStart[2] < (int)m_cacheBlock->getSize()[2]);
    return m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1],
                                iz - m_stencilStart[2]);
  }

  /// Just as BlockLab::operator() but returning a const.
  const ElementType &read(int ix, int iy = 0, int iz = 0) const {
    assert(ix - m_stencilStart[0] >= 0 &&
           ix - m_stencilStart[0] < (int)m_cacheBlock->getSize()[0]);
    assert(iy - m_stencilStart[1] >= 0 &&
           iy - m_stencilStart[1] < (int)m_cacheBlock->getSize()[1]);
    assert(iz - m_stencilStart[2] >= 0 &&
           iz - m_stencilStart[2] < (int)m_cacheBlock->getSize()[2]);
    return m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1],
                                iz - m_stencilStart[2]);
  }

  /// Deallocate memory (used in destructor).
  void release() {
    _release(m_cacheBlock);
    _release(m_CoarsenedBlock);
  }

  /** Prepares the BlockLab for a given 'grid' and stencil of points.
   *  Allocates memory (if not already allocated) for the arrays that will hold
   * the copy of a GridBlock plus its halo cells.
   * @param grid: the Grid/GridMPI with all the GridBlocks that will need halo
   * cells
   * @param stencil: the StencilInfo for the halo cells
   * @param  Istencil_start: the starts of the stencil used for coarse-fine
   * interpolation of halo cells, set to -1 for the default interpolation.
   * @param  Istencil_end: the ends of the stencil used for coarse-fine
   * interpolation of halo cells, set to +2 for the default interpolation.
   */
  virtual void prepare(GridType &grid, const StencilInfo &stencil,
                       const int Istencil_start[3] = default_start,
                       const int Istencil_end[3] = default_end) {
    istensorial = stencil.tensorial;
    coarsened = false;
    m_stencilStart[0] = stencil.sx;
    m_stencilStart[1] = stencil.sy;
    m_stencilStart[2] = stencil.sz;
    m_stencilEnd[0] = stencil.ex;
    m_stencilEnd[1] = stencil.ey;
    m_stencilEnd[2] = stencil.ez;

    m_InterpStencilStart[0] = Istencil_start[0];
    m_InterpStencilStart[1] = Istencil_start[1];
    m_InterpStencilStart[2] = Istencil_start[2];
    m_InterpStencilEnd[0] = Istencil_end[0];
    m_InterpStencilEnd[1] = Istencil_end[1];
    m_InterpStencilEnd[2] = Istencil_end[2];

    assert(m_InterpStencilStart[0] <= m_InterpStencilEnd[0]);
    assert(m_InterpStencilStart[1] <= m_InterpStencilEnd[1]);
    assert(m_InterpStencilStart[2] <= m_InterpStencilEnd[2]);
    assert(stencil.sx <= stencil.ex);
    assert(stencil.sy <= stencil.ey);
    assert(stencil.sz <= stencil.ez);
    assert(stencil.sx >= -BlockType::sizeX);
    assert(stencil.sy >= -BlockType::sizeY);
    assert(stencil.sz >= -BlockType::sizeZ);
    assert(stencil.ex < 2 * BlockType::sizeX);
    assert(stencil.ey < 2 * BlockType::sizeY);
    assert(stencil.ez < 2 * BlockType::sizeZ);

    m_refGrid = &grid;

    if (m_cacheBlock == NULL ||
        (int)m_cacheBlock->getSize()[0] !=
            (int)BlockType::sizeX + m_stencilEnd[0] - m_stencilStart[0] - 1 ||
        (int)m_cacheBlock->getSize()[1] !=
            (int)BlockType::sizeY + m_stencilEnd[1] - m_stencilStart[1] - 1 ||
        (int)m_cacheBlock->getSize()[2] !=
            (int)BlockType::sizeZ + m_stencilEnd[2] - m_stencilStart[2] - 1) {
      if (m_cacheBlock != NULL)
        _release(m_cacheBlock);

      m_cacheBlock = allocator<Matrix3D<ElementType, allocator>>().allocate(1);

      allocator<Matrix3D<ElementType, allocator>>().construct(m_cacheBlock);

      m_cacheBlock->_Setup(
          BlockType::sizeX + m_stencilEnd[0] - m_stencilStart[0] - 1,
          BlockType::sizeY + m_stencilEnd[1] - m_stencilStart[1] - 1,
          BlockType::sizeZ + m_stencilEnd[2] - m_stencilStart[2] - 1);
    }

    offset[0] = (m_stencilStart[0] - 1) / 2 + m_InterpStencilStart[0];
    offset[1] = (m_stencilStart[1] - 1) / 2 + m_InterpStencilStart[1];
    offset[2] = (m_stencilStart[2] - 1) / 2 + m_InterpStencilStart[2];

    const int e[3] = {(m_stencilEnd[0]) / 2 + 1 + m_InterpStencilEnd[0] - 1,
                      (m_stencilEnd[1]) / 2 + 1 + m_InterpStencilEnd[1] - 1,
                      (m_stencilEnd[2]) / 2 + 1 + m_InterpStencilEnd[2] - 1};

    if (m_CoarsenedBlock == NULL ||
        (int)m_CoarsenedBlock->getSize()[0] !=
            CoarseBlockSize[0] + e[0] - offset[0] - 1 ||
        (int)m_CoarsenedBlock->getSize()[1] !=
            CoarseBlockSize[1] + e[1] - offset[1] - 1 ||
        (int)m_CoarsenedBlock->getSize()[2] !=
            CoarseBlockSize[2] + e[2] - offset[2] - 1) {
      if (m_CoarsenedBlock != NULL)
        _release(m_CoarsenedBlock);

      m_CoarsenedBlock =
          allocator<Matrix3D<ElementType, allocator>>().allocate(1);

      allocator<Matrix3D<ElementType, allocator>>().construct(m_CoarsenedBlock);

      m_CoarsenedBlock->_Setup(CoarseBlockSize[0] + e[0] - offset[0] - 1,
                               CoarseBlockSize[1] + e[1] - offset[1] - 1,
                               CoarseBlockSize[2] + e[2] - offset[2] - 1);
    }

#if DIMENSION == 3
    use_averages = (m_refGrid->FiniteDifferences == false || istensorial ||
                    m_stencilStart[0] < -2 || m_stencilStart[1] < -2 ||
                    m_stencilStart[2] < -2 || m_stencilEnd[0] > 3 ||
                    m_stencilEnd[1] > 3 || m_stencilEnd[2] > 3);
#else
    use_averages = (m_refGrid->FiniteDifferences == false || istensorial ||
                    m_stencilStart[0] < -2 || m_stencilStart[1] < -2 ||
                    m_stencilEnd[0] > 3 || m_stencilEnd[1] > 3);
#endif
  }

  /** Provide a prepared BlockLab (working copy of gridpoints+halo cells).
   *  Once called, the user can use the () operators to access the halo cells.
   * For derived instances of BlockLab, the time 't' can also be provided, in
   * order to enforce time-dependent boundary conditions.
   * @param info: the BlockInfo for the GridBlock that needs halo cells.
   * @param t: (optional) current time, for time-dependent boundary conditions
   * @param applybc: (optional, default is true) apply boundary conditions or
   * not (assume periodic if not)
   */
  virtual void load(const BlockInfo &info, const Real t = 0,
                    const bool applybc = true) {
    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    const int nZ = BlockType::sizeZ;
    const bool xperiodic = is_xperiodic();
    const bool yperiodic = is_yperiodic();
    const bool zperiodic = is_zperiodic();

    std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();

    const int aux = 1 << info.level;
    NX = blocksPerDim[0] * aux; // needed for apply_bc
    NY = blocksPerDim[1] * aux; // needed for apply_bc
    NZ = blocksPerDim[2] * aux; // needed for apply_bc

    assert(m_cacheBlock != NULL);

    // 1.load the block into the cache
    {
      BlockType &block = *(BlockType *)info.ptrBlock;
      ElementType *ptrSource = &block(0);

#if 0 // original
            for(int iz=0; iz<nZ; iz++)
            for(int iy=0; iy<nY; iy++)
            {
              ElementType * ptrDestination = &m_cacheBlock->Access(0-m_stencilStart[0], iy-m_stencilStart[1], iz-m_stencilStart[2]);
              memcpy2((char *)ptrDestination, (char *)ptrSource, sizeof(ElementType)*nX);
              ptrSource+= nX;
            }
#else
      const int nbytes = sizeof(ElementType) * nX;
      const int _iz0 = -m_stencilStart[2];
      const int _iz1 = _iz0 + nZ;
      const int _iy0 = -m_stencilStart[1];
      const int _iy1 = _iy0 + nY;
      const int m_vSize0 = m_cacheBlock->getSize(0);
      const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice();
      const int my_ix = -m_stencilStart[0];
#pragma GCC ivdep
      for (int iz = _iz0; iz < _iz1; iz++) {
        const int my_izx = iz * m_nElemsPerSlice + my_ix;
#pragma GCC ivdep
        for (int iy = _iy0; iy < _iy1; iy += 4) {
          ElementType *__restrict__ ptrDestination0 =
              &m_cacheBlock->LinAccess(my_izx + (iy)*m_vSize0);
          ElementType *__restrict__ ptrDestination1 =
              &m_cacheBlock->LinAccess(my_izx + (iy + 1) * m_vSize0);
          ElementType *__restrict__ ptrDestination2 =
              &m_cacheBlock->LinAccess(my_izx + (iy + 2) * m_vSize0);
          ElementType *__restrict__ ptrDestination3 =
              &m_cacheBlock->LinAccess(my_izx + (iy + 3) * m_vSize0);
          memcpy2(ptrDestination0, (ptrSource), nbytes);
          memcpy2(ptrDestination1, (ptrSource + nX), nbytes);
          memcpy2(ptrDestination2, (ptrSource + 2 * nX), nbytes);
          memcpy2(ptrDestination3, (ptrSource + 3 * nX), nbytes);
          ptrSource += 4 * nX;
        }
      }
#endif
    }

    // 2. put the ghosts into the cache
    {
      coarsened = false;

      const bool xskin = info.index[0] == 0 || info.index[0] == NX - 1;
      const bool yskin = info.index[1] == 0 || info.index[1] == NY - 1;
      const bool zskin = info.index[2] == 0 || info.index[2] == NZ - 1;
      const int xskip = info.index[0] == 0 ? -1 : 1;
      const int yskip = info.index[1] == 0 ? -1 : 1;
      const int zskip = info.index[2] == 0 ? -1 : 1;

      int icodes[DIMENSION == 2 ? 8 : 26]; // Could be uint8_t?
      int k = 0;
      coarsened_nei_codes_size = 0;

      for (int icode = (DIMENSION == 2 ? 9 : 0);
           icode < (DIMENSION == 2 ? 18 : 27); icode++) {
        myblocks[icode] = nullptr;
        if (icode == 1 * 1 + 3 * 1 + 9 * 1)
          continue;
        const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, icode / 9 - 1};

        if (!xperiodic && code[0] == xskip && xskin)
          continue;
        if (!yperiodic && code[1] == yskip && yskin)
          continue;
        if (!zperiodic && code[2] == zskip && zskin)
          continue;

        const auto &TreeNei =
            m_refGrid->Tree(info.level, info.Znei_(code[0], code[1], code[2]));
        if (TreeNei.Exists()) {
          icodes[k++] = icode;
        } else if (TreeNei.CheckCoarser()) {
          coarsened_nei_codes[coarsened_nei_codes_size++] = icode;
          CoarseFineExchange(info, code);
        }

        if (!istensorial && !use_averages &&
            abs(code[0]) + abs(code[1]) + abs(code[2]) > 1)
          continue;

        // s and e correspond to start and end of this lab's cells that are
        // filled by neighbors
        const int s[3] = {
            code[0] < 1 ? (code[0] < 0 ? m_stencilStart[0] : 0) : nX,
            code[1] < 1 ? (code[1] < 0 ? m_stencilStart[1] : 0) : nY,
            code[2] < 1 ? (code[2] < 0 ? m_stencilStart[2] : 0) : nZ};

        const int e[3] = {
            code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + m_stencilEnd[0] - 1,
            code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + m_stencilEnd[1] - 1,
            code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + m_stencilEnd[2] - 1};

        if (TreeNei.Exists())
          SameLevelExchange(info, code, s, e);
        else if (TreeNei.CheckFiner())
          FineToCoarseExchange(info, code, s, e);
      } // icode = 0,...,26 (3D) or 9,...,17 (2D)
      if (coarsened_nei_codes_size > 0)
        for (int i = 0; i < k; ++i) {
          const int icode = icodes[i];
          const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                               icode / 9 - 1};
          const int infoNei_index[3] = {(info.index[0] + code[0] + NX) % NX,
                                        (info.index[1] + code[1] + NY) % NY,
                                        (info.index[2] + code[2] + NZ) % NZ};
          if (UseCoarseStencil(info, infoNei_index)) {
            FillCoarseVersion(info, code);
            coarsened = true;
          }
        }

      if (m_refGrid->get_world_size() == 1) {
        post_load(info, t, applybc);
      }
    }
  }

protected:
  /** Called from 'load', to enforce boundary conditions and coarse-fine
   * interpolation. To interpolate halo cells from neighboring coarser blocks,
   * the BlockLab first fills a coarsened version of the GridBlock that requires
   * the halo cells. This coarsened version is filled with grid points from the
   * coarse neighbors and with averaged down values of this GridBlock's
   * gridpoints. Averaging down happens in this function, followed by the
   *  interpolation. Boundary conditions from derived versions of this class are
   * also enforced. Default boundary conditions are periodic.
   * @param info: the BlockInfo for the GridBlock that needs halo cells.
   * @param t: (optional) current time, for time-dependent boundary conditions
   * @param applybc: (optional, default is true) apply boundary conditions or
   * not (assume periodic if not)
   */
  void post_load(const BlockInfo &info, const Real t = 0, bool applybc = true) {
    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
#if DIMENSION == 3
    const int nZ = BlockType::sizeZ;
    if (coarsened) {
#pragma GCC ivdep
      for (int k = 0; k < nZ / 2; k++) {
#pragma GCC ivdep
        for (int j = 0; j < nY / 2; j++) {
#pragma GCC ivdep
          for (int i = 0; i < nX / 2; i++) {
            if (i > -m_InterpStencilStart[0] &&
                i < nX / 2 - m_InterpStencilEnd[0] &&
                j > -m_InterpStencilStart[1] &&
                j < nY / 2 - m_InterpStencilEnd[1] &&
                k > -m_InterpStencilStart[2] &&
                k < nZ / 2 - m_InterpStencilEnd[2])
              continue;
            const int ix = 2 * i - m_stencilStart[0];
            const int iy = 2 * j - m_stencilStart[1];
            const int iz = 2 * k - m_stencilStart[2];
            ElementType &coarseElement = m_CoarsenedBlock->Access(
                i - offset[0], j - offset[1], k - offset[2]);
            coarseElement =
                AverageDown(m_cacheBlock->Read(ix, iy, iz),
                            m_cacheBlock->Read(ix + 1, iy, iz),
                            m_cacheBlock->Read(ix, iy + 1, iz),
                            m_cacheBlock->Read(ix + 1, iy + 1, iz),
                            m_cacheBlock->Read(ix, iy, iz + 1),
                            m_cacheBlock->Read(ix + 1, iy, iz + 1),
                            m_cacheBlock->Read(ix, iy + 1, iz + 1),
                            m_cacheBlock->Read(ix + 1, iy + 1, iz + 1));
          }
        }
      }
    }
#else
    if (coarsened) {
#pragma GCC ivdep
      for (int j = 0; j < nY / 2; j++) {
#pragma GCC ivdep
        for (int i = 0; i < nX / 2; i++) {
          if (i > -m_InterpStencilStart[0] &&
              i < nX / 2 - m_InterpStencilEnd[0] &&
              j > -m_InterpStencilStart[1] &&
              j < nY / 2 - m_InterpStencilEnd[1])
            continue;
          const int ix = 2 * i - m_stencilStart[0];
          const int iy = 2 * j - m_stencilStart[1];
          ElementType &coarseElement =
              m_CoarsenedBlock->Access(i - offset[0], j - offset[1], 0);
          coarseElement = AverageDown(m_cacheBlock->Read(ix, iy, 0),
                                      m_cacheBlock->Read(ix + 1, iy, 0),
                                      m_cacheBlock->Read(ix, iy + 1, 0),
                                      m_cacheBlock->Read(ix + 1, iy + 1, 0));
        }
      }
    }
#endif
    if (applybc)
      _apply_bc(info, t, true); // apply BC to coarse block
    CoarseFineInterpolation(info);
    if (applybc)
      _apply_bc(info, t);
  }

  /** Check if blocks on the same refinement level need to exchange averaged
   * down cells. To perform coarse-fine interpolation, the BlockLab creates a
   * coarsened version of the GridBlock that needs halo cells. Filling this
   * coarsened version can require averaged down values from GridBlocks of the
   * same resolution, which would create a large enough stencil of coarse values
   * to perform the interpolation. Whether or not this is needed is determined
   *  by this function.
   * @param info: the BlockInfo for the GridBlock that needs halo cells.
   * @param b_index: the (i,j,k) index coordinates of the block that is adjacent
   * to 'info'.
   */
  bool UseCoarseStencil(const BlockInfo &a, const int *b_index) {
    if (a.level == 0 || (!use_averages))
      return false;

    std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();

    int imin[3];
    int imax[3];
    const int aux = 1 << a.level;
    const bool periodic[3] = {is_xperiodic(), is_yperiodic(), is_zperiodic()};
    const int blocks[3] = {blocksPerDim[0] * aux - 1, blocksPerDim[1] * aux - 1,
                           blocksPerDim[2] * aux - 1};
    for (int d = 0; d < 3; d++) {
      imin[d] = (a.index[d] < b_index[d]) ? 0 : -1;
      imax[d] = (a.index[d] > b_index[d]) ? 0 : +1;
      if (periodic[d]) {
        if (a.index[d] == 0 && b_index[d] == blocks[d])
          imin[d] = -1;
        if (b_index[d] == 0 && a.index[d] == blocks[d])
          imax[d] = +1;
      } else {
        if (a.index[d] == 0 && b_index[d] == 0)
          imin[d] = 0;
        if (a.index[d] == blocks[d] && b_index[d] == blocks[d])
          imax[d] = 0;
      }
    }

    for (int itest = 0; itest < coarsened_nei_codes_size; itest++)
      for (int i2 = imin[2]; i2 <= imax[2]; i2++)
        for (int i1 = imin[1]; i1 <= imax[1]; i1++)
          for (int i0 = imin[0]; i0 <= imax[0]; i0++) {
            const int icode_test = (i0 + 1) + 3 * (i1 + 1) + 9 * (i2 + 1);
            if (coarsened_nei_codes[itest] == icode_test)
              return true;
          }
    return false;
  }

  /** Exchange halo cells for blocks on the same refinement level.
   * @param info: the BlockInfo for the GridBlock that needs halo cells.
   * @param code: pointer to three integers, one for each spatial direction.
   * Possible values of each integer are -1,0,+1, based on the relative position
   * of the neighboring block and 'info'
   * @param s: the starts of the part of 'info' that will be filled
   * @param e: the ends of the part of 'info' that will be filled
   */
  void SameLevelExchange(const BlockInfo &info, const int *const code,
                         const int *const s, const int *const e) {
    const int bytes = (e[0] - s[0]) * sizeof(ElementType);
    if (!bytes)
      return;

    const int icode = (code[0] + 1) + 3 * (code[1] + 1) + 9 * (code[2] + 1);
    myblocks[icode] =
        m_refGrid->avail(info.level, info.Znei_(code[0], code[1], code[2]));
    if (myblocks[icode] == nullptr)
      return;
    const BlockType &b = *myblocks[icode];

    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    const int nZ = BlockType::sizeZ;
    const int m_vSize0 = m_cacheBlock->getSize(0);
    const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice();
    const int my_ix = s[0] - m_stencilStart[0];
    const int mod = (e[1] - s[1]) % 4;

#pragma GCC ivdep
    for (int iz = s[2]; iz < e[2]; iz++) {
      const int my_izx = (iz - m_stencilStart[2]) * m_nElemsPerSlice + my_ix;
#pragma GCC ivdep
      for (int iy = s[1]; iy < e[1] - mod; iy += 4) {
        ElementType *__restrict__ ptrDest0 = &m_cacheBlock->LinAccess(
            my_izx + (iy - m_stencilStart[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest1 = &m_cacheBlock->LinAccess(
            my_izx + (iy + 1 - m_stencilStart[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest2 = &m_cacheBlock->LinAccess(
            my_izx + (iy + 2 - m_stencilStart[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest3 = &m_cacheBlock->LinAccess(
            my_izx + (iy + 3 - m_stencilStart[1]) * m_vSize0);
        const ElementType *ptrSrc0 =
            &b(s[0] - code[0] * nX, iy - code[1] * nY, iz - code[2] * nZ);
        const ElementType *ptrSrc1 =
            &b(s[0] - code[0] * nX, iy + 1 - code[1] * nY, iz - code[2] * nZ);
        const ElementType *ptrSrc2 =
            &b(s[0] - code[0] * nX, iy + 2 - code[1] * nY, iz - code[2] * nZ);
        const ElementType *ptrSrc3 =
            &b(s[0] - code[0] * nX, iy + 3 - code[1] * nY, iz - code[2] * nZ);
        memcpy2(ptrDest0, ptrSrc0, bytes);
        memcpy2(ptrDest1, ptrSrc1, bytes);
        memcpy2(ptrDest2, ptrSrc2, bytes);
        memcpy2(ptrDest3, ptrSrc3, bytes);
      }
#pragma GCC ivdep
      for (int iy = e[1] - mod; iy < e[1]; iy++) {
        ElementType *__restrict__ ptrDest = &m_cacheBlock->LinAccess(
            my_izx + (iy - m_stencilStart[1]) * m_vSize0);
        const ElementType *ptrSrc =
            &b(s[0] - code[0] * nX, iy - code[1] * nY, iz - code[2] * nZ);
        memcpy2(ptrDest, ptrSrc, bytes);
      }
    }
  }

#if DIMENSION == 3
  /// Average down eight elements (3D)
  ElementType AverageDown(const ElementType &e0, const ElementType &e1,
                          const ElementType &e2, const ElementType &e3,
                          const ElementType &e4, const ElementType &e5,
                          const ElementType &e6, const ElementType &e7) {
#ifdef PRESERVE_SYMMETRY
    return ConsistentAverage<ElementType>(e0, e1, e2, e3, e4, e5, e6, e7);
#else
    return 0.125 * (e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7);
#endif
  }

  /** Coarse-fine interpolation function, based on interpolation stencil of +-1
   * point. This function evaluates a third-order Taylor expansion by using a
   * stencil of +-1 points around the coarse grid point that will be replaced by
   * eight finer ones. This function can be overwritten by derived versions of
   * BlockLab, to enable a custom interpolation. The +-1 points used here come
   * from the 'interpolation stencil' passed to BlockLab.
   *  @param C: pointer to the +-1 points around the coarse point (27 values in
   * total)
   *  @param R: pointer to the eight refined points around the coarse point
   *  @param x: deprecated parameter, used only in the 2D version of this
   * function
   *  @param y: deprecated parameter, used only in the 2D version of this
   * function
   *  @param z: deprecated parameter, used only in the 2D version of this
   * function
   */
  virtual void TestInterp(ElementType *C[3][3][3], ElementType *R, int x, int y,
                          int z) {
#ifdef PRESERVE_SYMMETRY
    const ElementType dudx = 0.125 * ((*C[2][1][1]) - (*C[0][1][1]));
    const ElementType dudy = 0.125 * ((*C[1][2][1]) - (*C[1][0][1]));
    const ElementType dudz = 0.125 * ((*C[1][1][2]) - (*C[1][1][0]));
    const ElementType dudxdy = 0.015625 * (((*C[0][0][1]) + (*C[2][2][1])) -
                                           ((*C[2][0][1]) + (*C[0][2][1])));
    const ElementType dudxdz = 0.015625 * (((*C[0][1][0]) + (*C[2][1][2])) -
                                           ((*C[2][1][0]) + (*C[0][1][2])));
    const ElementType dudydz = 0.015625 * (((*C[1][0][0]) + (*C[1][2][2])) -
                                           ((*C[1][2][0]) + (*C[1][0][2])));
    const ElementType lap =
        *C[1][1][1] + 0.03125 * (ConsistentSum((*C[0][1][1]) + (*C[2][1][1]),
                                               (*C[1][0][1]) + (*C[1][2][1]),
                                               (*C[1][1][0]) + (*C[1][1][2])) -
                                 6.0 * (*C[1][1][1]));
    R[0] = lap + (ConsistentSum((-1.0) * dudx, (-1.0) * dudy, (-1.0) * dudz) +
                  ConsistentSum(dudxdy, dudxdz, dudydz));
    R[1] = lap + (ConsistentSum(dudx, (-1.0) * dudy, (-1.0) * dudz) +
                  ConsistentSum((-1.0) * dudxdy, (-1.0) * dudxdz, dudydz));
    R[2] = lap + (ConsistentSum((-1.0) * dudx, dudy, (-1.0) * dudz) +
                  ConsistentSum((-1.0) * dudxdy, dudxdz, (-1.0) * dudydz));
    R[3] = lap + (ConsistentSum(dudx, dudy, (-1.0) * dudz) +
                  ConsistentSum(dudxdy, (-1.0) * dudxdz, (-1.0) * dudydz));
    R[4] = lap + (ConsistentSum((-1.0) * dudx, (-1.0) * dudy, dudz) +
                  ConsistentSum(dudxdy, (-1.0) * dudxdz, (-1.0) * dudydz));
    R[5] = lap + (ConsistentSum(dudx, (-1.0) * dudy, dudz) +
                  ConsistentSum((-1.0) * dudxdy, dudxdz, (-1.0) * dudydz));
    R[6] = lap + (ConsistentSum((-1.0) * dudx, dudy, dudz) +
                  ConsistentSum((-1.0) * dudxdy, (-1.0) * dudxdz, dudydz));
    R[7] = lap + (ConsistentSum(dudx, dudy, dudz) +
                  ConsistentSum(dudxdy, dudxdz, dudydz));
#else
    const ElementType dudx = 0.125 * ((*C[2][1][1]) - (*C[0][1][1]));
    const ElementType dudy = 0.125 * ((*C[1][2][1]) - (*C[1][0][1]));
    const ElementType dudz = 0.125 * ((*C[1][1][2]) - (*C[1][1][0]));
    const ElementType dudxdy = 0.015625 * ((*C[0][0][1]) + (*C[2][2][1]) -
                                           (*C[2][0][1]) - (*C[0][2][1]));
    const ElementType dudxdz = 0.015625 * ((*C[0][1][0]) + (*C[2][1][2]) -
                                           (*C[2][1][0]) - (*C[0][1][2]));
    const ElementType dudydz = 0.015625 * ((*C[1][0][0]) + (*C[1][2][2]) -
                                           (*C[1][2][0]) - (*C[1][0][2]));
    const ElementType lap =
        *C[1][1][1] + 0.03125 * ((*C[0][1][1]) + (*C[2][1][1]) + (*C[1][0][1]) +
                                 (*C[1][2][1]) + (*C[1][1][0]) + (*C[1][1][2]) +
                                 (-6.0) * (*C[1][1][1]));
    R[0] = lap - dudx - dudy - dudz + dudxdy + dudxdz + dudydz;
    R[1] = lap + dudx - dudy - dudz - dudxdy - dudxdz + dudydz;
    R[2] = lap - dudx + dudy - dudz - dudxdy + dudxdz - dudydz;
    R[3] = lap + dudx + dudy - dudz + dudxdy - dudxdz - dudydz;
    R[4] = lap - dudx - dudy + dudz + dudxdy - dudxdz - dudydz;
    R[5] = lap + dudx - dudy + dudz - dudxdy + dudxdz - dudydz;
    R[6] = lap - dudx + dudy + dudz - dudxdy - dudxdz + dudydz;
    R[7] = lap + dudx + dudy + dudz + dudxdy + dudxdz + dudydz;
#endif
  }
#else
  /// Average down four elements (2D)
  ElementType AverageDown(const ElementType &e0, const ElementType &e1,
                          const ElementType &e2, const ElementType &e3) {
    return 0.25 * ((e0 + e3) + (e1 + e2));
  }

  /// Auxiliary function for 3rd order coarse-fine interpolation
  void LI(ElementType &a, ElementType b, ElementType c) {
    auto kappa = ((4.0 / 15.0) * a + (6.0 / 15.0) * c) + (-10.0 / 15.0) * b;
    auto lambda = (b - c) - kappa;
    a = (4.0 * kappa + 2.0 * lambda) + c;
  }

  /// Auxiliary function for 3rd order coarse-fine interpolation
  void LE(ElementType &a, ElementType b, ElementType c) {
    auto kappa = ((4.0 / 15.0) * a + (6.0 / 15.0) * c) + (-10.0 / 15.0) * b;
    auto lambda = (b - c) - kappa;
    a = (9.0 * kappa + 3.0 * lambda) + c;
  }

  /** Coarse-fine interpolation function, based on interpolation stencil of +-1
   * point. This function evaluates a third-order Taylor expansion by using a
   * stencil of +-1 points around the coarse grid point that will be replaced by
   * eight finer ones. This function can be overwritten by derived versions of
   * BlockLab, to enable a custom interpolation. The +-1 points used here come
   * from the 'interpolation stencil' passed to BlockLab.
   *  @param C: pointer to the +-1 points around the coarse point (9 values in
   * total)
   *  @param R: pointer to the one refined points around the coarse point
   *  @param x: delta x of the point to be interpolated (+1 or -1).
   *  @param y: delta y of the point to be interpolated (+1 or -1).
   */
  virtual void TestInterp(ElementType *C[3][3], ElementType &R, int x, int y) {
    const double dx = 0.25 * (2 * x - 1);
    const double dy = 0.25 * (2 * y - 1);
    ElementType dudx = 0.5 * ((*C[2][1]) - (*C[0][1]));
    ElementType dudy = 0.5 * ((*C[1][2]) - (*C[1][0]));
    ElementType dudxdy =
        0.25 * (((*C[0][0]) + (*C[2][2])) - ((*C[2][0]) + (*C[0][2])));
    ElementType dudx2 = ((*C[0][1]) + (*C[2][1])) - 2.0 * (*C[1][1]);
    ElementType dudy2 = ((*C[1][0]) + (*C[1][2])) - 2.0 * (*C[1][1]);
    R = (*C[1][1] + (dx * dudx + dy * dudy)) +
        (((0.5 * dx * dx) * dudx2 + (0.5 * dy * dy) * dudy2) +
         (dx * dy) * dudxdy);
  }
#endif

  /** Exchange halo cells from fine to coarse blocks.
   * @param info: the BlockInfo for the GridBlock that needs halo cells.
   * @param code: pointer to three integers, one for each spatial direction.
   * Possible values of each integer are -1,0,+1, based on the relative position
   * of the neighboring block and 'info'
   * @param s: the starts of the part of 'info' that will be filled
   * @param e: the ends of the part of 'info' that will be filled
   */
  void FineToCoarseExchange(const BlockInfo &info, const int *const code,
                            const int *const s, const int *const e) {
    const int bytes = (abs(code[0]) * (e[0] - s[0]) +
                       (1 - abs(code[0])) * ((e[0] - s[0]) / 2)) *
                      sizeof(ElementType);
    if (!bytes)
      return;

    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    const int nZ = BlockType::sizeZ;
    const int m_vSize0 = m_cacheBlock->getSize(0);
    const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice();
    const int yStep = (code[1] == 0) ? 2 : 1;
    const int zStep = (code[2] == 0) ? 2 : 1;
    const int mod = ((e[1] - s[1]) / yStep) % 4;

    int Bstep = 1; // face
    if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))
      Bstep = 3; // edge
    else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
      Bstep = 4; // corner

    /*
      A corner has one finer block.
      An edge has two finer blocks, corresponding to B=0 and B=3. The block B=0
      is the one closer to the origin (0,0,0). A face has four finer blocks.
      They are numbered as follows, depending on whether the face lies on the
      xy- , yz- or xz- plane

      y                                  z                                  z
      ^                                  ^                                  ^
      |                                  |                                  |
      |                                  |                                  |
      |_________________                 |_________________ |_________________
      |        |        |                |        |        |                | |
      | |    2   |   3    |                |    2   |   3    |                |
      2   |   3    |
      |________|________|                |________|________| |________|________|
      |        |        |                |        |        |                | |
      | |    0   |    1   |                |    0   |    1   |                |
      0   |    1   |
      |________|________|------------->x |________|________|------------->x
      |________|________|------------->y

    */
    // loop over blocks that make up face/edge/corner (respectively 4,2 or 1
    // blocks)
    for (int B = 0; B <= 3; B += Bstep) {
      const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);

#if DIMENSION == 3
      BlockType *b_ptr =
          m_refGrid->avail1(2 * info.index[0] + std::max(code[0], 0) + code[0] +
                                (B % 2) * std::max(0, 1 - abs(code[0])),
                            2 * info.index[1] + std::max(code[1], 0) + code[1] +
                                aux * std::max(0, 1 - abs(code[1])),
                            2 * info.index[2] + std::max(code[2], 0) + code[2] +
                                (B / 2) * std::max(0, 1 - abs(code[2])),
                            info.level + 1);
#else
      BlockType *b_ptr =
          m_refGrid->avail1(2 * info.index[0] + std::max(code[0], 0) + code[0] +
                                (B % 2) * std::max(0, 1 - abs(code[0])),
                            2 * info.index[1] + std::max(code[1], 0) + code[1] +
                                aux * std::max(0, 1 - abs(code[1])),
                            info.level + 1);
#endif
      if (b_ptr == nullptr)
        continue;
      BlockType &b = *b_ptr;

      const int my_ix = abs(code[0]) * (s[0] - m_stencilStart[0]) +
                        (1 - abs(code[0])) * (s[0] - m_stencilStart[0] +
                                              (B % 2) * (e[0] - s[0]) / 2);
      const int XX = s[0] - code[0] * nX + std::min(0, code[0]) * (e[0] - s[0]);

#pragma GCC ivdep
      for (int iz = s[2]; iz < e[2]; iz += zStep) {
        const int ZZ = (abs(code[2]) == 1)
                           ? 2 * (iz - code[2] * nZ) + std::min(0, code[2]) * nZ
                           : iz;
        const int my_izx =
            (abs(code[2]) * (iz - m_stencilStart[2]) +
             (1 - abs(code[2])) *
                 (iz / 2 - m_stencilStart[2] + (B / 2) * (e[2] - s[2]) / 2)) *
                m_nElemsPerSlice +
            my_ix;

#pragma GCC ivdep
        for (int iy = s[1]; iy < e[1] - mod; iy += 4 * yStep) {
          ElementType *__restrict__ ptrDest0 = &m_cacheBlock->LinAccess(
              my_izx +
              (abs(code[1]) * (iy + 0 * yStep - m_stencilStart[1]) +
               (1 - abs(code[1])) * ((iy + 0 * yStep) / 2 - m_stencilStart[1] +
                                     aux * (e[1] - s[1]) / 2)) *
                  m_vSize0);
          ElementType *__restrict__ ptrDest1 = &m_cacheBlock->LinAccess(
              my_izx +
              (abs(code[1]) * (iy + 1 * yStep - m_stencilStart[1]) +
               (1 - abs(code[1])) * ((iy + 1 * yStep) / 2 - m_stencilStart[1] +
                                     aux * (e[1] - s[1]) / 2)) *
                  m_vSize0);
          ElementType *__restrict__ ptrDest2 = &m_cacheBlock->LinAccess(
              my_izx +
              (abs(code[1]) * (iy + 2 * yStep - m_stencilStart[1]) +
               (1 - abs(code[1])) * ((iy + 2 * yStep) / 2 - m_stencilStart[1] +
                                     aux * (e[1] - s[1]) / 2)) *
                  m_vSize0);
          ElementType *__restrict__ ptrDest3 = &m_cacheBlock->LinAccess(
              my_izx +
              (abs(code[1]) * (iy + 3 * yStep - m_stencilStart[1]) +
               (1 - abs(code[1])) * ((iy + 3 * yStep) / 2 - m_stencilStart[1] +
                                     aux * (e[1] - s[1]) / 2)) *
                  m_vSize0);
          const int YY0 = (abs(code[1]) == 1)
                              ? 2 * (iy + 0 * yStep - code[1] * nY) +
                                    std::min(0, code[1]) * nY
                              : iy + 0 * yStep;
          const int YY1 = (abs(code[1]) == 1)
                              ? 2 * (iy + 1 * yStep - code[1] * nY) +
                                    std::min(0, code[1]) * nY
                              : iy + 1 * yStep;
          const int YY2 = (abs(code[1]) == 1)
                              ? 2 * (iy + 2 * yStep - code[1] * nY) +
                                    std::min(0, code[1]) * nY
                              : iy + 2 * yStep;
          const int YY3 = (abs(code[1]) == 1)
                              ? 2 * (iy + 3 * yStep - code[1] * nY) +
                                    std::min(0, code[1]) * nY
                              : iy + 3 * yStep;
#if DIMENSION == 3
          const ElementType *ptrSrc_00 = &b(XX, YY0, ZZ);
          const ElementType *ptrSrc_10 = &b(XX, YY0, ZZ + 1);
          const ElementType *ptrSrc_20 = &b(XX, YY0 + 1, ZZ);
          const ElementType *ptrSrc_30 = &b(XX, YY0 + 1, ZZ + 1);
          const ElementType *ptrSrc_01 = &b(XX, YY1, ZZ);
          const ElementType *ptrSrc_11 = &b(XX, YY1, ZZ + 1);
          const ElementType *ptrSrc_21 = &b(XX, YY1 + 1, ZZ);
          const ElementType *ptrSrc_31 = &b(XX, YY1 + 1, ZZ + 1);
          const ElementType *ptrSrc_02 = &b(XX, YY2, ZZ);
          const ElementType *ptrSrc_12 = &b(XX, YY2, ZZ + 1);
          const ElementType *ptrSrc_22 = &b(XX, YY2 + 1, ZZ);
          const ElementType *ptrSrc_32 = &b(XX, YY2 + 1, ZZ + 1);
          const ElementType *ptrSrc_03 = &b(XX, YY3, ZZ);
          const ElementType *ptrSrc_13 = &b(XX, YY3, ZZ + 1);
          const ElementType *ptrSrc_23 = &b(XX, YY3 + 1, ZZ);
          const ElementType *ptrSrc_33 = &b(XX, YY3 + 1, ZZ + 1);
#pragma GCC ivdep
          for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) +
                                 (1 - abs(code[0])) * ((e[0] - s[0]) / 2));
               ee++) {
            ptrDest0[ee] = AverageDown(
                ptrSrc_00[2 * ee], ptrSrc_10[2 * ee], ptrSrc_20[2 * ee],
                ptrSrc_30[2 * ee], ptrSrc_00[2 * ee + 1], ptrSrc_10[2 * ee + 1],
                ptrSrc_20[2 * ee + 1], ptrSrc_30[2 * ee + 1]);
            ptrDest1[ee] = AverageDown(
                ptrSrc_01[2 * ee], ptrSrc_11[2 * ee], ptrSrc_21[2 * ee],
                ptrSrc_31[2 * ee], ptrSrc_01[2 * ee + 1], ptrSrc_11[2 * ee + 1],
                ptrSrc_21[2 * ee + 1], ptrSrc_31[2 * ee + 1]);
            ptrDest2[ee] = AverageDown(
                ptrSrc_02[2 * ee], ptrSrc_12[2 * ee], ptrSrc_22[2 * ee],
                ptrSrc_32[2 * ee], ptrSrc_02[2 * ee + 1], ptrSrc_12[2 * ee + 1],
                ptrSrc_22[2 * ee + 1], ptrSrc_32[2 * ee + 1]);
            ptrDest3[ee] = AverageDown(
                ptrSrc_03[2 * ee], ptrSrc_13[2 * ee], ptrSrc_23[2 * ee],
                ptrSrc_33[2 * ee], ptrSrc_03[2 * ee + 1], ptrSrc_13[2 * ee + 1],
                ptrSrc_23[2 * ee + 1], ptrSrc_33[2 * ee + 1]);
          }
#else
          const ElementType *ptrSrc_00 = &b(XX, YY0, ZZ);
          const ElementType *ptrSrc_10 = &b(XX, YY0 + 1, ZZ);
          const ElementType *ptrSrc_01 = &b(XX, YY1, ZZ);
          const ElementType *ptrSrc_11 = &b(XX, YY1 + 1, ZZ);
          const ElementType *ptrSrc_02 = &b(XX, YY2, ZZ);
          const ElementType *ptrSrc_12 = &b(XX, YY2 + 1, ZZ);
          const ElementType *ptrSrc_03 = &b(XX, YY3, ZZ);
          const ElementType *ptrSrc_13 = &b(XX, YY3 + 1, ZZ);
#pragma GCC ivdep
          for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) +
                                 (1 - abs(code[0])) * ((e[0] - s[0]) / 2));
               ee++) {
            ptrDest0[ee] = AverageDown(
                *(ptrSrc_00 + 2 * ee), *(ptrSrc_10 + 2 * ee),
                *(ptrSrc_00 + 2 * ee + 1), *(ptrSrc_10 + 2 * ee + 1));
            ptrDest1[ee] = AverageDown(
                *(ptrSrc_01 + 2 * ee), *(ptrSrc_11 + 2 * ee),
                *(ptrSrc_01 + 2 * ee + 1), *(ptrSrc_11 + 2 * ee + 1));
            ptrDest2[ee] = AverageDown(
                *(ptrSrc_02 + 2 * ee), *(ptrSrc_12 + 2 * ee),
                *(ptrSrc_02 + 2 * ee + 1), *(ptrSrc_12 + 2 * ee + 1));
            ptrDest3[ee] = AverageDown(
                *(ptrSrc_03 + 2 * ee), *(ptrSrc_13 + 2 * ee),
                *(ptrSrc_03 + 2 * ee + 1), *(ptrSrc_13 + 2 * ee + 1));
          }
#endif
        }
#pragma GCC ivdep
        for (int iy = e[1] - mod; iy < e[1]; iy += yStep) {
          ElementType *ptrDest = (ElementType *)&m_cacheBlock->LinAccess(
              my_izx + (abs(code[1]) * (iy - m_stencilStart[1]) +
                        (1 - abs(code[1])) * (iy / 2 - m_stencilStart[1] +
                                              aux * (e[1] - s[1]) / 2)) *
                           m_vSize0);
          const int YY = (abs(code[1]) == 1) ? 2 * (iy - code[1] * nY) +
                                                   std::min(0, code[1]) * nY
                                             : iy;
#if DIMENSION == 3
          const ElementType *ptrSrc_0 = &b(XX, YY, ZZ);
          const ElementType *ptrSrc_1 = &b(XX, YY, ZZ + 1);
          const ElementType *ptrSrc_2 = &b(XX, YY + 1, ZZ);
          const ElementType *ptrSrc_3 = &b(XX, YY + 1, ZZ + 1);
          const ElementType *ptrSrc_0_1 = &b(XX + 1, YY, ZZ);
          const ElementType *ptrSrc_1_1 = &b(XX + 1, YY, ZZ + 1);
          const ElementType *ptrSrc_2_1 = &b(XX + 1, YY + 1, ZZ);
          const ElementType *ptrSrc_3_1 = &b(XX + 1, YY + 1, ZZ + 1);
// average down elements of block b to send to coarser neighbor
#pragma GCC ivdep
          for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) +
                                 (1 - abs(code[0])) * ((e[0] - s[0]) / 2));
               ee++) {
            ptrDest[ee] = AverageDown(ptrSrc_0[2 * ee], ptrSrc_1[2 * ee],
                                      ptrSrc_2[2 * ee], ptrSrc_3[2 * ee],
                                      ptrSrc_0_1[2 * ee], ptrSrc_1_1[2 * ee],
                                      ptrSrc_2_1[2 * ee], ptrSrc_3_1[2 * ee]);
          }
#else
          const ElementType *ptrSrc_0 = &b(XX, YY, ZZ);
          const ElementType *ptrSrc_1 = &b(XX, YY + 1, ZZ);
// average down elements of block b to send to coarser neighbor
#pragma GCC ivdep
          for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) +
                                 (1 - abs(code[0])) * ((e[0] - s[0]) / 2));
               ee++) {
            ptrDest[ee] =
                AverageDown(*(ptrSrc_0 + 2 * ee), *(ptrSrc_1 + 2 * ee),
                            *(ptrSrc_0 + 2 * ee + 1), *(ptrSrc_1 + 2 * ee + 1));
          }
#endif
        }
      }
    } // B
  }

  /** Exchange halo cells from coarse to fine blocks.
   * @param info: the BlockInfo for the GridBlock that needs halo cells.
   * @param code: pointer to three integers, one for each spatial direction.
   * Possible values of each integer are -1,0,+1, based on the relative position
   * of the neighboring block and 'info'
   */
  void CoarseFineExchange(const BlockInfo &info, const int *const code) {
    // Coarse neighbors send their cells. Those are stored in m_CoarsenedBlock
    // and are later used in function CoarseFineInterpolation to interpolate
    // fine values.

    const int infoNei_index[3] = {(info.index[0] + code[0] + NX) % NX,
                                  (info.index[1] + code[1] + NY) % NY,
                                  (info.index[2] + code[2] + NZ) % NZ};
    const int infoNei_index_true[3] = {(info.index[0] + code[0]),
                                       (info.index[1] + code[1]),
                                       (info.index[2] + code[2])};
#if DIMENSION == 3
    BlockType *b_ptr =
        m_refGrid->avail1((infoNei_index[0]) / 2, (infoNei_index[1]) / 2,
                          (infoNei_index[2]) / 2, info.level - 1);
#else
    BlockType *b_ptr = m_refGrid->avail1(
        (infoNei_index[0]) / 2, (infoNei_index[1]) / 2, info.level - 1);
#endif

    if (b_ptr == nullptr)
      return;
    const BlockType &b = *b_ptr;

    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    const int nZ = BlockType::sizeZ;

    const int s[3] = {
        code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : CoarseBlockSize[0],
        code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : CoarseBlockSize[1],
        code[2] < 1 ? (code[2] < 0 ? offset[2] : 0) : CoarseBlockSize[2]};

    const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : CoarseBlockSize[0])
                                  : CoarseBlockSize[0] + (m_stencilEnd[0]) / 2 +
                                        m_InterpStencilEnd[0] - 1,
                      code[1] < 1 ? (code[1] < 0 ? 0 : CoarseBlockSize[1])
                                  : CoarseBlockSize[1] + (m_stencilEnd[1]) / 2 +
                                        m_InterpStencilEnd[1] - 1,
                      code[2] < 1 ? (code[2] < 0 ? 0 : CoarseBlockSize[2])
                                  : CoarseBlockSize[2] + (m_stencilEnd[2]) / 2 +
                                        m_InterpStencilEnd[2] - 1};

    const int bytes = (e[0] - s[0]) * sizeof(ElementType);
    if (!bytes)
      return;

    const int base[3] = {(info.index[0] + code[0]) % 2,
                         (info.index[1] + code[1]) % 2,
                         (info.index[2] + code[2]) % 2};

    int CoarseEdge[3];
    CoarseEdge[0] = (code[0] == 0) ? 0
                    : (((info.index[0] % 2 == 0) &&
                        (infoNei_index_true[0] > info.index[0])) ||
                       ((info.index[0] % 2 == 1) &&
                        (infoNei_index_true[0] < info.index[0])))
                        ? 1
                        : 0;
    CoarseEdge[1] = (code[1] == 0) ? 0
                    : (((info.index[1] % 2 == 0) &&
                        (infoNei_index_true[1] > info.index[1])) ||
                       ((info.index[1] % 2 == 1) &&
                        (infoNei_index_true[1] < info.index[1])))
                        ? 1
                        : 0;
    CoarseEdge[2] = (code[2] == 0) ? 0
                    : (((info.index[2] % 2 == 0) &&
                        (infoNei_index_true[2] > info.index[2])) ||
                       ((info.index[2] % 2 == 1) &&
                        (infoNei_index_true[2] < info.index[2])))
                        ? 1
                        : 0;

    const int start[3] = {
        std::max(code[0], 0) * nX / 2 + (1 - abs(code[0])) * base[0] * nX / 2 -
            code[0] * nX + CoarseEdge[0] * code[0] * nX / 2,
        std::max(code[1], 0) * nY / 2 + (1 - abs(code[1])) * base[1] * nY / 2 -
            code[1] * nY + CoarseEdge[1] * code[1] * nY / 2,
        std::max(code[2], 0) * nZ / 2 + (1 - abs(code[2])) * base[2] * nZ / 2 -
            code[2] * nZ + CoarseEdge[2] * code[2] * nZ / 2};

    const int m_vSize0 = m_CoarsenedBlock->getSize(0);
    const int m_nElemsPerSlice =
        m_CoarsenedBlock->getNumberOfElementsPerSlice();
    const int my_ix = s[0] - offset[0];
    const int mod = (e[1] - s[1]) % 4;

#pragma GCC ivdep
    for (int iz = s[2]; iz < e[2]; iz++) {
      const int my_izx = (iz - offset[2]) * m_nElemsPerSlice + my_ix;
#pragma GCC ivdep
      for (int iy = s[1]; iy < e[1] - mod; iy += 4) {
        ElementType *__restrict__ ptrDest0 = &m_CoarsenedBlock->LinAccess(
            my_izx + (iy + 0 - offset[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest1 = &m_CoarsenedBlock->LinAccess(
            my_izx + (iy + 1 - offset[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest2 = &m_CoarsenedBlock->LinAccess(
            my_izx + (iy + 2 - offset[1]) * m_vSize0);
        ElementType *__restrict__ ptrDest3 = &m_CoarsenedBlock->LinAccess(
            my_izx + (iy + 3 - offset[1]) * m_vSize0);
        const ElementType *ptrSrc0 =
            &b(s[0] + start[0], iy + 0 + start[1], iz + start[2]);
        const ElementType *ptrSrc1 =
            &b(s[0] + start[0], iy + 1 + start[1], iz + start[2]);
        const ElementType *ptrSrc2 =
            &b(s[0] + start[0], iy + 2 + start[1], iz + start[2]);
        const ElementType *ptrSrc3 =
            &b(s[0] + start[0], iy + 3 + start[1], iz + start[2]);
        memcpy2(ptrDest0, ptrSrc0, bytes);
        memcpy2(ptrDest1, ptrSrc1, bytes);
        memcpy2(ptrDest2, ptrSrc2, bytes);
        memcpy2(ptrDest3, ptrSrc3, bytes);
      }
#pragma GCC ivdep
      for (int iy = e[1] - mod; iy < e[1]; iy++) {
        ElementType *ptrDest =
            &m_CoarsenedBlock->LinAccess(my_izx + (iy - offset[1]) * m_vSize0);
        const ElementType *ptrSrc =
            &b(s[0] + start[0], iy + start[1], iz + start[2]);
        memcpy2(ptrDest, ptrSrc, bytes);
      }
    }
  }

  /** Fill coarsened version of a block, used for fine-coarse interpolation.
   * Each block will create a coarsened version of itself, with averaged down
   * values. This version is also filled with gridpoints for halo cells that are
   * received from coarser neighbors. It is then used to interpolate fine cells
   * at coarse-fine interfaces.
   * @param info: the BlockInfo for the GridBlock that needs halo cells.
   * @param code: pointer to three integers, one for each spatial direction.
   * Possible values of each integer are -1,0,+1, based on the relative position
   * of the neighboring block and 'info'
   */
  void FillCoarseVersion(const BlockInfo &info, const int *const code) {
    // If a neighboring block is on the same level it might need to average down
    // some cells and use them to fill the coarsened version of this block.
    // Those cells are needed to refine the coarsened version and obtain ghosts
    // from coarser neighbors (those cells are inside the interpolation stencil
    // for refinement).

    const int icode = (code[0] + 1) + 3 * (code[1] + 1) + 9 * (code[2] + 1);
    if (myblocks[icode] == nullptr)
      return;
    const BlockType &b = *myblocks[icode];

    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    const int nZ = BlockType::sizeZ;

    const int eC[3] = {(m_stencilEnd[0]) / 2 + m_InterpStencilEnd[0],
                       (m_stencilEnd[1]) / 2 + m_InterpStencilEnd[1],
                       (m_stencilEnd[2]) / 2 + m_InterpStencilEnd[2]};

    const int s[3] = {
        code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : CoarseBlockSize[0],
        code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : CoarseBlockSize[1],
        code[2] < 1 ? (code[2] < 0 ? offset[2] : 0) : CoarseBlockSize[2]};

    const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : CoarseBlockSize[0])
                                  : CoarseBlockSize[0] + eC[0] - 1,
                      code[1] < 1 ? (code[1] < 0 ? 0 : CoarseBlockSize[1])
                                  : CoarseBlockSize[1] + eC[1] - 1,
                      code[2] < 1 ? (code[2] < 0 ? 0 : CoarseBlockSize[2])
                                  : CoarseBlockSize[2] + eC[2] - 1};

    const int bytes = (e[0] - s[0]) * sizeof(ElementType);
    if (!bytes)
      return;

    const int start[3] = {
        s[0] + std::max(code[0], 0) * CoarseBlockSize[0] - code[0] * nX +
            std::min(0, code[0]) * (e[0] - s[0]),
        s[1] + std::max(code[1], 0) * CoarseBlockSize[1] - code[1] * nY +
            std::min(0, code[1]) * (e[1] - s[1]),
        s[2] + std::max(code[2], 0) * CoarseBlockSize[2] - code[2] * nZ +
            std::min(0, code[2]) * (e[2] - s[2])};

    const int m_vSize0 = m_CoarsenedBlock->getSize(0);
    const int m_nElemsPerSlice =
        m_CoarsenedBlock->getNumberOfElementsPerSlice();
    const int my_ix = s[0] - offset[0];
    const int XX = start[0];

#pragma GCC ivdep
    for (int iz = s[2]; iz < e[2]; iz++) {
      const int ZZ = 2 * (iz - s[2]) + start[2];
      const int my_izx = (iz - offset[2]) * m_nElemsPerSlice + my_ix;

#pragma GCC ivdep
      for (int iy = s[1]; iy < e[1]; iy++) {
        if (code[1] == 0 && code[2] == 0 && iy > -m_InterpStencilStart[1] &&
            iy < nY / 2 - m_InterpStencilEnd[1] &&
            iz > -m_InterpStencilStart[2] &&
            iz < nZ / 2 - m_InterpStencilEnd[2])
          continue;

        ElementType *__restrict__ ptrDest1 =
            &m_CoarsenedBlock->LinAccess(my_izx + (iy - offset[1]) * m_vSize0);

        const int YY = 2 * (iy - s[1]) + start[1];
#if DIMENSION == 3
        const ElementType *ptrSrc_0 = &b(XX, YY, ZZ);
        const ElementType *ptrSrc_1 = &b(XX, YY, ZZ + 1);
        const ElementType *ptrSrc_2 = &b(XX, YY + 1, ZZ);
        const ElementType *ptrSrc_3 = &b(XX, YY + 1, ZZ + 1);
// average down elements of block b to send to coarser neighbor
#pragma GCC ivdep
        for (int ee = 0; ee < e[0] - s[0]; ee++) {
          ptrDest1[ee] =
              AverageDown(*(ptrSrc_0 + 2 * ee), *(ptrSrc_1 + 2 * ee),
                          *(ptrSrc_2 + 2 * ee), *(ptrSrc_3 + 2 * ee),
                          *(ptrSrc_0 + 2 * ee + 1), *(ptrSrc_1 + 2 * ee + 1),
                          *(ptrSrc_2 + 2 * ee + 1), *(ptrSrc_3 + 2 * ee + 1));
        }
#else
        const ElementType *ptrSrc_0 = (const ElementType *)&b(XX, YY, ZZ);
        const ElementType *ptrSrc_1 = (const ElementType *)&b(XX, YY + 1, ZZ);
// average down elements of block b to send to coarser neighbor
#pragma GCC ivdep
        for (int ee = 0; ee < e[0] - s[0]; ee++) {
          ptrDest1[ee] =
              AverageDown(*(ptrSrc_0 + 2 * ee), *(ptrSrc_1 + 2 * ee),
                          *(ptrSrc_0 + 2 * ee + 1), *(ptrSrc_1 + 2 * ee + 1));
        }
#endif
      }
    }
  }

/// Perform fine-coarse interpolation, after filling coarsened version of block.
#ifdef PRESERVE_SYMMETRY
  __attribute__((optimize("-O1")))
#endif
  void
  CoarseFineInterpolation(const BlockInfo &info) {
    const int nX = BlockType::sizeX;
    const int nY = BlockType::sizeY;
    const int nZ = BlockType::sizeZ;
    const bool xperiodic = is_xperiodic();
    const bool yperiodic = is_yperiodic();
    const bool zperiodic = is_zperiodic();
    const std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();
    const int aux = 1 << info.level;
    const bool xskin =
        info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
    const bool yskin =
        info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
    const bool zskin =
        info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
    const int xskip = info.index[0] == 0 ? -1 : 1;
    const int yskip = info.index[1] == 0 ? -1 : 1;
    const int zskip = info.index[2] == 0 ? -1 : 1;

    for (int ii = 0; ii < coarsened_nei_codes_size; ++ii) {
      const int icode = coarsened_nei_codes[ii];
      if (icode == 1 * 1 + 3 * 1 + 9 * 1)
        continue;
      const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                           (icode / 9) % 3 - 1};

#if DIMENSION == 2
      if (code[2] != 0)
        continue;
#endif

      if (!xperiodic && code[0] == xskip && xskin)
        continue;
      if (!yperiodic && code[1] == yskip && yskin)
        continue;
      if (!zperiodic && code[2] == zskip && zskin)
        continue;
      if (!istensorial && !use_averages &&
          abs(code[0]) + abs(code[1]) + abs(code[2]) > 1)
        continue;

      // s and e correspond to start and end of this lab's cells that are filled
      // by neighbors
      const int s[3] = {
          code[0] < 1 ? (code[0] < 0 ? m_stencilStart[0] : 0) : nX,
          code[1] < 1 ? (code[1] < 0 ? m_stencilStart[1] : 0) : nY,
          code[2] < 1 ? (code[2] < 0 ? m_stencilStart[2] : 0) : nZ};
      const int e[3] = {
          code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + m_stencilEnd[0] - 1,
          code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + m_stencilEnd[1] - 1,
          code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + m_stencilEnd[2] - 1};

      const int sC[3] = {
          code[0] < 1 ? (code[0] < 0 ? ((m_stencilStart[0] - 1) / 2) : 0)
                      : CoarseBlockSize[0],
          code[1] < 1 ? (code[1] < 0 ? ((m_stencilStart[1] - 1) / 2) : 0)
                      : CoarseBlockSize[1],
          code[2] < 1 ? (code[2] < 0 ? ((m_stencilStart[2] - 1) / 2) : 0)
                      : CoarseBlockSize[2]};

      const int bytes = (e[0] - s[0]) * sizeof(ElementType);
      if (!bytes)
        continue;

#if DIMENSION == 3
      ElementType retval[8];
      if (use_averages)
        for (int iz = s[2]; iz < e[2]; iz += 2) {
          const int ZZ =
              (iz - s[2] - std::min(0, code[2]) * ((e[2] - s[2]) % 2)) / 2 +
              sC[2];
          const int z =
              abs(iz - s[2] - std::min(0, code[2]) * ((e[2] - s[2]) % 2)) % 2;
          const int izp = (abs(iz) % 2 == 1) ? -1 : 1;
          const int rzp = (izp == 1) ? 1 : 0;
          const int rz = (izp == 1) ? 0 : 1;

#pragma GCC ivdep
          for (int iy = s[1]; iy < e[1]; iy += 2) {
            const int YY =
                (iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) / 2 +
                sC[1];
            const int y =
                abs(iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) % 2;
            const int iyp = (abs(iy) % 2 == 1) ? -1 : 1;
            const int ryp = (iyp == 1) ? 1 : 0;
            const int ry = (iyp == 1) ? 0 : 1;

#pragma GCC ivdep
            for (int ix = s[0]; ix < e[0]; ix += 2) {
              const int XX =
                  (ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) / 2 +
                  sC[0];
              const int x =
                  abs(ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) %
                  2;
              const int ixp = (abs(ix) % 2 == 1) ? -1 : 1;
              const int rxp = (ixp == 1) ? 1 : 0;
              const int rx = (ixp == 1) ? 0 : 1;

              ElementType *Test[3][3][3];
              for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                  for (int k = 0; k < 3; k++)
                    Test[i][j][k] = &m_CoarsenedBlock->Access(
                        XX - 1 + i - offset[0], YY - 1 + j - offset[1],
                        ZZ - 1 + k - offset[2]);

              TestInterp(Test, retval, x, y, z);

              if (ix >= s[0] && ix < e[0] && iy >= s[1] && iy < e[1] &&
                  iz >= s[2] && iz < e[2])
                m_cacheBlock->Access(
                    ix - m_stencilStart[0], iy - m_stencilStart[1],
                    iz - m_stencilStart[2]) = retval[rx + 2 * ry + 4 * rz];
              if (ix + ixp >= s[0] && ix + ixp < e[0] && iy >= s[1] &&
                  iy < e[1] && iz >= s[2] && iz < e[2])
                m_cacheBlock->Access(
                    ix + ixp - m_stencilStart[0], iy - m_stencilStart[1],
                    iz - m_stencilStart[2]) = retval[rxp + 2 * ry + 4 * rz];
              if (ix >= s[0] && ix < e[0] && iy + iyp >= s[1] &&
                  iy + iyp < e[1] && iz >= s[2] && iz < e[2])
                m_cacheBlock->Access(
                    ix - m_stencilStart[0], iy + iyp - m_stencilStart[1],
                    iz - m_stencilStart[2]) = retval[rx + 2 * ryp + 4 * rz];
              if (ix + ixp >= s[0] && ix + ixp < e[0] && iy + iyp >= s[1] &&
                  iy + iyp < e[1] && iz >= s[2] && iz < e[2])
                m_cacheBlock->Access(
                    ix + ixp - m_stencilStart[0], iy + iyp - m_stencilStart[1],
                    iz - m_stencilStart[2]) = retval[rxp + 2 * ryp + 4 * rz];
              if (ix >= s[0] && ix < e[0] && iy >= s[1] && iy < e[1] &&
                  iz + izp >= s[2] && iz + izp < e[2])
                m_cacheBlock->Access(ix - m_stencilStart[0],
                                     iy - m_stencilStart[1],
                                     iz + izp - m_stencilStart[2]) =
                    retval[rx + 2 * ry + 4 * rzp];
              if (ix + ixp >= s[0] && ix + ixp < e[0] && iy >= s[1] &&
                  iy < e[1] && iz + izp >= s[2] && iz + izp < e[2])
                m_cacheBlock->Access(ix + ixp - m_stencilStart[0],
                                     iy - m_stencilStart[1],
                                     iz + izp - m_stencilStart[2]) =
                    retval[rxp + 2 * ry + 4 * rzp];
              if (ix >= s[0] && ix < e[0] && iy + iyp >= s[1] &&
                  iy + iyp < e[1] && iz + izp >= s[2] && iz + izp < e[2])
                m_cacheBlock->Access(ix - m_stencilStart[0],
                                     iy + iyp - m_stencilStart[1],
                                     iz + izp - m_stencilStart[2]) =
                    retval[rx + 2 * ryp + 4 * rzp];
              if (ix + ixp >= s[0] && ix + ixp < e[0] && iy + iyp >= s[1] &&
                  iy + iyp < e[1] && iz + izp >= s[2] && iz + izp < e[2])
                m_cacheBlock->Access(ix + ixp - m_stencilStart[0],
                                     iy + iyp - m_stencilStart[1],
                                     iz + izp - m_stencilStart[2]) =
                    retval[rxp + 2 * ryp + 4 * rzp];
            }
          }
        }
      if (m_refGrid->FiniteDifferences &&
          abs(code[0]) + abs(code[1]) + abs(code[2]) ==
              1) // Correct stencil points +-1 and +-2 at faces
      {
        const int coef_ixyz[3] = {std::min(0, code[0]) * ((e[0] - s[0]) % 2),
                                  std::min(0, code[1]) * ((e[1] - s[1]) % 2),
                                  std::min(0, code[2]) * ((e[2] - s[2]) % 2)};
        const int min_iz = std::max(s[2], -2);
        const int min_iy = std::max(s[1], -2);
        const int min_ix = std::max(s[0], -2);
        const int max_iz = std::min(e[2], nZ + 2);
        const int max_iy = std::min(e[1], nY + 2);
        const int max_ix = std::min(e[0], nX + 2);

        for (int iz = min_iz; iz < max_iz; iz++) {
          const int ZZ = (iz - s[2] - coef_ixyz[2]) / 2 + sC[2] - offset[2];
          const int z = abs(iz - s[2] - coef_ixyz[2]) % 2;
          const double dz = 0.25 * (2 * z - 1);
          const double *dz_coef = dz > 0 ? &d_coef_plus[0] : &d_coef_minus[0];
          const bool zinner = (ZZ + offset[2] != 0) &&
                              (ZZ + offset[2] != CoarseBlockSize[2] - 1);
          const bool zstart = (ZZ + offset[2] == 0);

#pragma GCC ivdep
          for (int iy = min_iy; iy < max_iy; iy++) {
            const int YY = (iy - s[1] - coef_ixyz[1]) / 2 + sC[1] - offset[1];
            const int y = abs(iy - s[1] - coef_ixyz[1]) % 2;
            const double dy = 0.25 * (2 * y - 1);
            const double *dy_coef = dy > 0 ? &d_coef_plus[0] : &d_coef_minus[0];
            const bool yinner = (YY + offset[1] != 0) &&
                                (YY + offset[1] != CoarseBlockSize[1] - 1);
            const bool ystart = (YY + offset[1] == 0);

#pragma GCC ivdep
            for (int ix = min_ix; ix < max_ix; ix++) {
              const int XX = (ix - s[0] - coef_ixyz[0]) / 2 + sC[0] - offset[0];
              const int x = abs(ix - s[0] - coef_ixyz[0]) % 2;
              const double dx = 0.25 * (2 * x - 1);
              const double *dx_coef =
                  dx > 0 ? &d_coef_plus[0] : &d_coef_minus[0];
              const bool xinner = (XX + offset[0] != 0) &&
                                  (XX + offset[0] != CoarseBlockSize[0] - 1);
              const bool xstart = (XX + offset[0] == 0);

              auto &a = m_cacheBlock->Access(ix - m_stencilStart[0],
                                             iy - m_stencilStart[1],
                                             iz - m_stencilStart[2]);
              if (code[0] != 0) // X-face
              {
                ElementType x1D, x2D, mixed;

                int YP, YM, ZP, ZM;
                double mixed_coef = 1.0;
                if (yinner) {
                  x1D =
                      (dy_coef[6] * m_CoarsenedBlock->Access(XX, YY - 1, ZZ) +
                       dy_coef[8] * m_CoarsenedBlock->Access(XX, YY + 1, ZZ)) +
                      dy_coef[7] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  YP = YY + 1;
                  YM = YY - 1;
                  mixed_coef *= 0.5;
                } else if (ystart) {
                  x1D =
                      (dy_coef[0] * m_CoarsenedBlock->Access(XX, YY + 2, ZZ) +
                       dy_coef[1] * m_CoarsenedBlock->Access(XX, YY + 1, ZZ)) +
                      dy_coef[2] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  YP = YY + 1;
                  YM = YY;
                } else {
                  x1D =
                      (dy_coef[3] * m_CoarsenedBlock->Access(XX, YY - 2, ZZ) +
                       dy_coef[4] * m_CoarsenedBlock->Access(XX, YY - 1, ZZ)) +
                      dy_coef[5] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  YP = YY;
                  YM = YY - 1;
                }
                if (zinner) {
                  x2D =
                      (dz_coef[6] * m_CoarsenedBlock->Access(XX, YY, ZZ - 1) +
                       dz_coef[8] * m_CoarsenedBlock->Access(XX, YY, ZZ + 1)) +
                      dz_coef[7] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  ZP = ZZ + 1;
                  ZM = ZZ - 1;
                  mixed_coef *= 0.5;
                } else if (zstart) {
                  x2D =
                      (dz_coef[0] * m_CoarsenedBlock->Access(XX, YY, ZZ + 2) +
                       dz_coef[1] * m_CoarsenedBlock->Access(XX, YY, ZZ + 1)) +
                      dz_coef[2] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  ZP = ZZ + 1;
                  ZM = ZZ;
                } else {
                  x2D =
                      (dz_coef[3] * m_CoarsenedBlock->Access(XX, YY, ZZ - 2) +
                       dz_coef[4] * m_CoarsenedBlock->Access(XX, YY, ZZ - 1)) +
                      dz_coef[5] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  ZP = ZZ;
                  ZM = ZZ - 1;
                }
                mixed = mixed_coef * dy * dz *
                        ((m_CoarsenedBlock->Access(XX, YM, ZM) +
                          m_CoarsenedBlock->Access(XX, YP, ZP)) -
                         (m_CoarsenedBlock->Access(XX, YP, ZM) +
                          m_CoarsenedBlock->Access(XX, YM, ZP)));
                a = (x1D + x2D) + mixed;
              } else if (code[1] != 0) // Y-face
              {
                ElementType x1D, x2D, mixed;

                int XP, XM, ZP, ZM;
                double mixed_coef = 1.0;
                if (xinner) {
                  x1D =
                      (dx_coef[6] * m_CoarsenedBlock->Access(XX - 1, YY, ZZ) +
                       dx_coef[8] * m_CoarsenedBlock->Access(XX + 1, YY, ZZ)) +
                      dx_coef[7] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  XP = XX + 1;
                  XM = XX - 1;
                  mixed_coef *= 0.5;
                } else if (xstart) {
                  x1D =
                      (dx_coef[0] * m_CoarsenedBlock->Access(XX + 2, YY, ZZ) +
                       dx_coef[1] * m_CoarsenedBlock->Access(XX + 1, YY, ZZ)) +
                      dx_coef[2] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  XP = XX + 1;
                  XM = XX;
                } else {
                  x1D =
                      (dx_coef[3] * m_CoarsenedBlock->Access(XX - 2, YY, ZZ) +
                       dx_coef[4] * m_CoarsenedBlock->Access(XX - 1, YY, ZZ)) +
                      dx_coef[5] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  XP = XX;
                  XM = XX - 1;
                }
                if (zinner) {
                  x2D =
                      (dz_coef[6] * m_CoarsenedBlock->Access(XX, YY, ZZ - 1) +
                       dz_coef[8] * m_CoarsenedBlock->Access(XX, YY, ZZ + 1)) +
                      dz_coef[7] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  ZP = ZZ + 1;
                  ZM = ZZ - 1;
                  mixed_coef *= 0.5;
                } else if (zstart) {
                  x2D =
                      (dz_coef[0] * m_CoarsenedBlock->Access(XX, YY, ZZ + 2) +
                       dz_coef[1] * m_CoarsenedBlock->Access(XX, YY, ZZ + 1)) +
                      dz_coef[2] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  ZP = ZZ + 1;
                  ZM = ZZ;
                } else {
                  x2D =
                      (dz_coef[3] * m_CoarsenedBlock->Access(XX, YY, ZZ - 2) +
                       dz_coef[4] * m_CoarsenedBlock->Access(XX, YY, ZZ - 1)) +
                      dz_coef[5] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  ZP = ZZ;
                  ZM = ZZ - 1;
                }
                mixed = mixed_coef * dx * dz *
                        ((m_CoarsenedBlock->Access(XM, YY, ZM) +
                          m_CoarsenedBlock->Access(XP, YY, ZP)) -
                         (m_CoarsenedBlock->Access(XP, YY, ZM) +
                          m_CoarsenedBlock->Access(XM, YY, ZP)));
                a = (x1D + x2D) + mixed;
              } else if (code[2] != 0) // Z-face
              {
                ElementType x1D, x2D, mixed;

                int XP, XM, YP, YM;
                double mixed_coef = 1.0;
                if (xinner) {
                  x1D =
                      (dx_coef[6] * m_CoarsenedBlock->Access(XX - 1, YY, ZZ) +
                       dx_coef[8] * m_CoarsenedBlock->Access(XX + 1, YY, ZZ)) +
                      dx_coef[7] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  XP = XX + 1;
                  XM = XX - 1;
                  mixed_coef *= 0.5;
                } else if (xstart) {
                  x1D =
                      (dx_coef[0] * m_CoarsenedBlock->Access(XX + 2, YY, ZZ) +
                       dx_coef[1] * m_CoarsenedBlock->Access(XX + 1, YY, ZZ)) +
                      dx_coef[2] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  XP = XX + 1;
                  XM = XX;
                } else {
                  x1D =
                      (dx_coef[3] * m_CoarsenedBlock->Access(XX - 2, YY, ZZ) +
                       dx_coef[4] * m_CoarsenedBlock->Access(XX - 1, YY, ZZ)) +
                      dx_coef[5] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  XP = XX;
                  XM = XX - 1;
                }
                if (yinner) {
                  x2D =
                      (dy_coef[6] * m_CoarsenedBlock->Access(XX, YY - 1, ZZ) +
                       dy_coef[8] * m_CoarsenedBlock->Access(XX, YY + 1, ZZ)) +
                      dy_coef[7] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  YP = YY + 1;
                  YM = YY - 1;
                  mixed_coef *= 0.5;
                } else if (ystart) {
                  x2D =
                      (dy_coef[0] * m_CoarsenedBlock->Access(XX, YY + 2, ZZ) +
                       dy_coef[1] * m_CoarsenedBlock->Access(XX, YY + 1, ZZ)) +
                      dy_coef[2] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  YP = YY + 1;
                  YM = YY;
                } else {
                  x2D =
                      (dy_coef[3] * m_CoarsenedBlock->Access(XX, YY - 2, ZZ) +
                       dy_coef[4] * m_CoarsenedBlock->Access(XX, YY - 1, ZZ)) +
                      dy_coef[5] * m_CoarsenedBlock->Access(XX, YY, ZZ);
                  YP = YY;
                  YM = YY - 1;
                }

                mixed = mixed_coef * dx * dy *
                        ((m_CoarsenedBlock->Access(XM, YM, ZZ) +
                          m_CoarsenedBlock->Access(XP, YP, ZZ)) -
                         (m_CoarsenedBlock->Access(XP, YM, ZZ) +
                          m_CoarsenedBlock->Access(XM, YP, ZZ)));
                a = (x1D + x2D) + mixed;
              }

              const auto &b = m_cacheBlock->Access(
                  ix - m_stencilStart[0] + (-3 * code[0] + 1) / 2 -
                      x * abs(code[0]),
                  iy - m_stencilStart[1] + (-3 * code[1] + 1) / 2 -
                      y * abs(code[1]),
                  iz - m_stencilStart[2] + (-3 * code[2] + 1) / 2 -
                      z * abs(code[2]));
              const auto &c = m_cacheBlock->Access(
                  ix - m_stencilStart[0] + (-5 * code[0] + 1) / 2 -
                      x * abs(code[0]),
                  iy - m_stencilStart[1] + (-5 * code[1] + 1) / 2 -
                      y * abs(code[1]),
                  iz - m_stencilStart[2] + (-5 * code[2] + 1) / 2 -
                      z * abs(code[2]));
              const int ccc = code[0] + code[1] + code[2];
              const int xyz =
                  abs(code[0]) * x + abs(code[1]) * y + abs(code[2]) * z;

              if (ccc == 1)
                a = (xyz == 0)
                        ? (1.0 / 15.0) * (8.0 * a + (10.0 * b - 3.0 * c))
                        : (1.0 / 15.0) * (24.0 * a + (-15.0 * b + 6 * c));
              else /*(ccc=-1)*/
                a = (xyz == 1)
                        ? (1.0 / 15.0) * (8.0 * a + (10.0 * b - 3.0 * c))
                        : (1.0 / 15.0) * (24.0 * a + (-15.0 * b + 6 * c));
            }
          }
        }
      }
#else

      if (use_averages) {
#pragma GCC ivdep
        for (int iy = s[1]; iy < e[1]; iy += 1) {
          const int YY =
              (iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) / 2 +
              sC[1];
#pragma GCC ivdep
          for (int ix = s[0]; ix < e[0]; ix += 1) {
            const int XX =
                (ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) / 2 +
                sC[0];
            ElementType *Test[3][3];
            for (int i = 0; i < 3; i++)
              for (int j = 0; j < 3; j++)
                Test[i][j] = &m_CoarsenedBlock->Access(
                    XX - 1 + i - offset[0], YY - 1 + j - offset[1], 0);
            TestInterp(
                Test,
                m_cacheBlock->Access(ix - m_stencilStart[0],
                                     iy - m_stencilStart[1], 0),
                abs(ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2,
                abs(iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) %
                    2);
          }
        }
      }
      if (m_refGrid->FiniteDifferences &&
          abs(code[0]) + abs(code[1]) ==
              1) // Correct stencil points +-1 and +-2 at faces
      {
#pragma GCC ivdep
        for (int iy = s[1]; iy < e[1]; iy += 2) {
          const int YY =
              (iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) / 2 +
              sC[1] - offset[1];
          const int y =
              abs(iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) % 2;
          const int iyp = (abs(iy) % 2 == 1) ? -1 : 1;
          const double dy = 0.25 * (2 * y - 1);

#pragma GCC ivdep
          for (int ix = s[0]; ix < e[0]; ix += 2) {
            const int XX =
                (ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) / 2 +
                sC[0] - offset[0];
            const int x =
                abs(ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2;
            const int ixp = (abs(ix) % 2 == 1) ? -1 : 1;
            const double dx = 0.25 * (2 * x - 1);
            if (ix < -2 || iy < -2 || ix > nX + 1 || iy > nY + 1)
              continue;

            if (code[0] != 0) {
              ElementType dudy, dudy2;
              if (YY + offset[1] == 0) {
                dudy = (-0.5 * m_CoarsenedBlock->Access(XX, YY + 2, 0) -
                        1.5 * m_CoarsenedBlock->Access(XX, YY, 0)) +
                       2.0 * m_CoarsenedBlock->Access(XX, YY + 1, 0);
                dudy2 = (m_CoarsenedBlock->Access(XX, YY + 2, 0) +
                         m_CoarsenedBlock->Access(XX, YY, 0)) -
                        2.0 * m_CoarsenedBlock->Access(XX, YY + 1, 0);
              } else if (YY + offset[1] == CoarseBlockSize[1] - 1) {
                dudy = (0.5 * m_CoarsenedBlock->Access(XX, YY - 2, 0) +
                        1.5 * m_CoarsenedBlock->Access(XX, YY, 0)) -
                       2.0 * m_CoarsenedBlock->Access(XX, YY - 1, 0);
                dudy2 = (m_CoarsenedBlock->Access(XX, YY - 2, 0) +
                         m_CoarsenedBlock->Access(XX, YY, 0)) -
                        2.0 * m_CoarsenedBlock->Access(XX, YY - 1, 0);
              } else {
                dudy = 0.5 * (m_CoarsenedBlock->Access(XX, YY + 1, 0) -
                              m_CoarsenedBlock->Access(XX, YY - 1, 0));
                dudy2 = (m_CoarsenedBlock->Access(XX, YY + 1, 0) +
                         m_CoarsenedBlock->Access(XX, YY - 1, 0)) -
                        2.0 * m_CoarsenedBlock->Access(XX, YY, 0);
              }
              m_cacheBlock->Access(ix - m_stencilStart[0],
                                   iy - m_stencilStart[1], 0) =
                  m_CoarsenedBlock->Access(XX, YY, 0) + dy * dudy +
                  (0.5 * dy * dy) * dudy2;
              if (iy + iyp >= s[1] && iy + iyp < e[1])
                m_cacheBlock->Access(ix - m_stencilStart[0],
                                     iy - m_stencilStart[1] + iyp, 0) =
                    m_CoarsenedBlock->Access(XX, YY, 0) - dy * dudy +
                    (0.5 * dy * dy) * dudy2;
              if (ix + ixp >= s[0] && ix + ixp < e[0])
                m_cacheBlock->Access(ix - m_stencilStart[0] + ixp,
                                     iy - m_stencilStart[1], 0) =
                    m_CoarsenedBlock->Access(XX, YY, 0) + dy * dudy +
                    (0.5 * dy * dy) * dudy2;
              if (ix + ixp >= s[0] && ix + ixp < e[0] && iy + iyp >= s[1] &&
                  iy + iyp < e[1])
                m_cacheBlock->Access(ix - m_stencilStart[0] + ixp,
                                     iy - m_stencilStart[1] + iyp, 0) =
                    m_CoarsenedBlock->Access(XX, YY, 0) - dy * dudy +
                    (0.5 * dy * dy) * dudy2;
            } else // if (code[1] != 0)
            {
              ElementType dudx, dudx2;
              if (XX + offset[0] == 0) {
                dudx = (-0.5 * m_CoarsenedBlock->Access(XX + 2, YY, 0) -
                        1.5 * m_CoarsenedBlock->Access(XX, YY, 0)) +
                       2.0 * m_CoarsenedBlock->Access(XX + 1, YY, 0);
                dudx2 = (m_CoarsenedBlock->Access(XX + 2, YY, 0) +
                         m_CoarsenedBlock->Access(XX, YY, 0)) -
                        2.0 * m_CoarsenedBlock->Access(XX + 1, YY, 0);
              } else if (XX + offset[0] == CoarseBlockSize[0] - 1) {
                dudx = (0.5 * m_CoarsenedBlock->Access(XX - 2, YY, 0) +
                        1.5 * m_CoarsenedBlock->Access(XX, YY, 0)) -
                       2.0 * m_CoarsenedBlock->Access(XX - 1, YY, 0);
                dudx2 = (m_CoarsenedBlock->Access(XX - 2, YY, 0) +
                         m_CoarsenedBlock->Access(XX, YY, 0)) -
                        2.0 * m_CoarsenedBlock->Access(XX - 1, YY, 0);
              } else {
                dudx = 0.5 * (m_CoarsenedBlock->Access(XX + 1, YY, 0) -
                              m_CoarsenedBlock->Access(XX - 1, YY, 0));
                dudx2 = (m_CoarsenedBlock->Access(XX + 1, YY, 0) +
                         m_CoarsenedBlock->Access(XX - 1, YY, 0)) -
                        2.0 * m_CoarsenedBlock->Access(XX, YY, 0);
              }
              m_cacheBlock->Access(ix - m_stencilStart[0],
                                   iy - m_stencilStart[1], 0) =
                  m_CoarsenedBlock->Access(XX, YY, 0) + dx * dudx +
                  (0.5 * dx * dx) * dudx2;
              if (iy + iyp >= s[1] && iy + iyp < e[1])
                m_cacheBlock->Access(ix - m_stencilStart[0],
                                     iy - m_stencilStart[1] + iyp, 0) =
                    m_CoarsenedBlock->Access(XX, YY, 0) + dx * dudx +
                    (0.5 * dx * dx) * dudx2;
              if (ix + ixp >= s[0] && ix + ixp < e[0])
                m_cacheBlock->Access(ix - m_stencilStart[0] + ixp,
                                     iy - m_stencilStart[1], 0) =
                    m_CoarsenedBlock->Access(XX, YY, 0) - dx * dudx +
                    (0.5 * dx * dx) * dudx2;
              if (ix + ixp >= s[0] && ix + ixp < e[0] && iy + iyp >= s[1] &&
                  iy + iyp < e[1])
                m_cacheBlock->Access(ix - m_stencilStart[0] + ixp,
                                     iy - m_stencilStart[1] + iyp, 0) =
                    m_CoarsenedBlock->Access(XX, YY, 0) - dx * dudx +
                    (0.5 * dx * dx) * dudx2;
            }
          }
        }

        for (int iy = s[1]; iy < e[1]; iy += 1) {
#pragma GCC ivdep
          for (int ix = s[0]; ix < e[0]; ix += 1) {
            if (ix < -2 || iy < -2 || ix > nX + 1 || iy > nY + 1)
              continue;
            const int x =
                abs(ix - s[0] - std::min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2;
            const int y =
                abs(iy - s[1] - std::min(0, code[1]) * ((e[1] - s[1]) % 2)) % 2;

            auto &a = m_cacheBlock->Access(ix - m_stencilStart[0],
                                           iy - m_stencilStart[1], 0);

            if (code[0] == 0 && code[1] == 1) {
              if (y == 0) // interpolation
              {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] - 1, 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] - 2, 0);
                LI(a, b, c);
              } else if (y == 1) // extrapolation
              {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] - 2, 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] - 3, 0);
                LE(a, b, c);
              }
            } else if (code[0] == 0 && code[1] == -1) {
              if (y == 1) // interpolation
              {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] + 1, 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] + 2, 0);
                LI(a, b, c);
              } else if (y == 0) // extrapolation
              {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] + 2, 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0],
                                               iy - m_stencilStart[1] + 3, 0);
                LE(a, b, c);
              }
            } else if (code[1] == 0 && code[0] == 1) {
              if (x == 0) // interpolation
              {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0] - 1,
                                               iy - m_stencilStart[1], 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0] - 2,
                                               iy - m_stencilStart[1], 0);
                LI(a, b, c);
              } else if (x == 1) // extrapolation
              {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0] - 2,
                                               iy - m_stencilStart[1], 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0] - 3,
                                               iy - m_stencilStart[1], 0);
                LE(a, b, c);
              }
            } else if (code[1] == 0 && code[0] == -1) {
              if (x == 1) // interpolation
              {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0] + 1,
                                               iy - m_stencilStart[1], 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0] + 2,
                                               iy - m_stencilStart[1], 0);
                LI(a, b, c);
              } else if (x == 0) // extrapolation
              {
                auto &b = m_cacheBlock->Access(ix - m_stencilStart[0] + 2,
                                               iy - m_stencilStart[1], 0);
                auto &c = m_cacheBlock->Access(ix - m_stencilStart[0] + 3,
                                               iy - m_stencilStart[1], 0);
                LE(a, b, c);
              }
            }
          }
        }
      }
#endif
    }
  }

  /// Enforce boundary conditions.
  virtual void _apply_bc(const BlockInfo &info, const Real t = 0,
                         bool coarse = false) {}

  /// Deallocate memory.
  template <typename T> void _release(T *&t) {
    if (t != NULL) {
      allocator<T>().destroy(t);
      allocator<T>().deallocate(t, 1);
    }
    t = NULL;
  }

private:
  BlockLab(const BlockLab &) = delete;
  BlockLab &operator=(const BlockLab &) = delete;
};

} // namespace cubism

namespace cubism {

/** \brief Similar to BlockLab, but should be used with simulations that support
 * MPI.*/
template <typename MyBlockLab> class BlockLabMPI : public MyBlockLab {
public:
  using GridType = typename MyBlockLab::GridType;
  using BlockType = typename GridType::BlockType;
  using ElementType = typename BlockType::ElementType;
  using Real = typename ElementType::RealType;

private:
  typedef SynchronizerMPI_AMR<Real, GridType> SynchronizerMPIType;
  SynchronizerMPIType *refSynchronizerMPI;

public:
  /// Same as 'prepare' from BlockLab. This will also create a
  /// SynchronizerMPI_AMR for different MPI processes.
  virtual void prepare(GridType &grid, const StencilInfo &stencil,
                       const int Istencil_start[3] = default_start,
                       const int Istencil_end[3] = default_end) override {
    auto itSynchronizerMPI = grid.SynchronizerMPIs.find(stencil);
    refSynchronizerMPI = itSynchronizerMPI->second;
    MyBlockLab::prepare(grid, stencil);
  }

  /// Same as 'load' from BlockLab. This will also fetch halo cells from
  /// different MPI processes.
  virtual void load(const BlockInfo &info, const Real t = 0,
                    const bool applybc = true) override {
    MyBlockLab::load(info, t, applybc);

    Real *dst = (Real *)&MyBlockLab ::m_cacheBlock->LinAccess(0);
    Real *dst1 = (Real *)&MyBlockLab ::m_CoarsenedBlock->LinAccess(0);

    refSynchronizerMPI->fetch(info, MyBlockLab::m_cacheBlock->getSize(),
                              MyBlockLab::m_CoarsenedBlock->getSize(), dst,
                              dst1);

    if (MyBlockLab::m_refGrid->get_world_size() > 1)
      MyBlockLab::post_load(info, t, applybc);
  }
};

} // namespace cubism

namespace cubism {

/** Takes care of load-balancing of Blocks.
 * This class will redistribute Blocks among different MPI ranks for two
 * reasons: 1) Eight (in 3D) or four (in 2D) blocks need to be compressed to
 * one, but they are owned by different ranks. PrepareCompression() will collect
 * them all to one rank, so that they can be compressed. 2) There is a load
 * imbalance after the grid is refined or compressed. If the imbalance is not
 * great (load imbalance ratio < 1.1), a 1D-diffusion based scheme is used to
 * redistribute blocks along the 1D Space-Filling-Curve. Otherwise, all blocks
 * are simply evenly redistributed among all ranks.
 * @tparam TGrid: the type of GridMPI to perform load-balancing for
 */
template <typename TGrid> class LoadBalancer {
public:
  typedef typename TGrid::Block BlockType;
  typedef typename TGrid::Block::ElementType ElementType;
  typedef typename TGrid::Block::ElementType::RealType Real;
  bool movedBlocks; ///< =true if load-balancing is performed when
                    ///< Balance_Diffusion of Balance_Global is called

protected:
  TGrid *grid; ///< grid where load balancing will be performed

  /// MPI datatype and auxiliary struct used to send/receive blocks among ranks
  MPI_Datatype MPI_BLOCK;
  struct MPI_Block {
    long long mn[2]; ///< level and Z-order index of a BlockInfo
    Real data[sizeof(BlockType) /
              sizeof(Real)]; ///< buffer array of data to send/receive

    /** Constructor; calls 'prepare'.
     * @param info: BlockInfo for block to be sent/received.
     * @param Fillptr: true if we want the data of the GridBlock to be copied to
     * this MPI_Block.
     */
    MPI_Block(const BlockInfo &info, const bool Fillptr = true) {
      prepare(info, Fillptr);
    }

    /** Prepare the MPI_Block with data from a GridBlock.
     * @param info: BlockInfo for block to be sent/received.
     * @param Fillptr: true if we want the data of the GridBlock to be copied to
     * this MPI_Block.
     */
    void prepare(const BlockInfo &info, const bool Fillptr = true) {
      mn[0] = info.level;
      mn[1] = info.Z;
      if (Fillptr) {
        Real *aux = &((BlockType *)info.ptrBlock)->data[0][0][0].member(0);
        std::memcpy(&data[0], aux, sizeof(BlockType));
      }
    }

    MPI_Block() {}
  };

  /// Allocate a block at a given level and Z-index and fill it with received
  /// data
  void AddBlock(const int level, const long long Z, Real *data) {
    // 1. Allocate the block from the grid
    grid->_alloc(level, Z);

    // 2. Fill the block with data received
    BlockInfo &info = grid->getBlockInfoAll(level, Z);
    BlockType *b1 = (BlockType *)info.ptrBlock;
    assert(b1 != NULL);
    Real *a1 = &b1->data[0][0][0].member(0);
    std::memcpy(a1, data, sizeof(BlockType));

// 3. Update status of children and parent block of newly allocated block
#if DIMENSION == 3
    int p[3];
    BlockInfo::inverse(Z, level, p[0], p[1], p[2]);
    if (level < grid->getlevelMax() - 1)
      for (int k1 = 0; k1 < 2; k1++)
        for (int j1 = 0; j1 < 2; j1++)
          for (int i1 = 0; i1 < 2; i1++) {
            const long long nc = grid->getZforward(
                level + 1, 2 * p[0] + i1, 2 * p[1] + j1, 2 * p[2] + k1);
            grid->Tree(level + 1, nc).setCheckCoarser();
          }
    if (level > 0) {
      const long long nf =
          grid->getZforward(level - 1, p[0] / 2, p[1] / 2, p[2] / 2);
      grid->Tree(level - 1, nf).setCheckFiner();
    }
#else
    int p[2];
    BlockInfo::inverse(Z, level, p[0], p[1]);
    if (level < grid->getlevelMax() - 1)
      for (int j1 = 0; j1 < 2; j1++)
        for (int i1 = 0; i1 < 2; i1++) {
          const long long nc =
              grid->getZforward(level + 1, 2 * p[0] + i1, 2 * p[1] + j1);
          grid->Tree(level + 1, nc).setCheckCoarser();
        }
    if (level > 0) {
      const long long nf = grid->getZforward(level - 1, p[0] / 2, p[1] / 2);
      grid->Tree(level - 1, nf).setCheckFiner();
    }
#endif
  }

public:
  /// Constructor
  LoadBalancer(TGrid &a_grid) {
    grid = &a_grid;
    movedBlocks = false;

    // Create MPI datatype to send/receive blocks (data) + two integers (their
    // level and Z-index)
    int array_of_blocklengths[2] = {2, sizeof(BlockType) / sizeof(Real)};
    MPI_Aint array_of_displacements[2] = {0, 2 * sizeof(long long)};
    MPI_Datatype array_of_types[2];
    array_of_types[0] = MPI_LONG_LONG;
    if (sizeof(Real) == sizeof(float))
      array_of_types[1] = MPI_FLOAT;
    else if (sizeof(Real) == sizeof(double))
      array_of_types[1] = MPI_DOUBLE;
    else if (sizeof(Real) == sizeof(long double))
      array_of_types[1] = MPI_LONG_DOUBLE;
    MPI_Type_create_struct(2, array_of_blocklengths, array_of_displacements,
                           array_of_types, &MPI_BLOCK);
    MPI_Type_commit(&MPI_BLOCK);
  }

  /// Destructor
  ~LoadBalancer() { MPI_Type_free(&MPI_BLOCK); }

  /// Compression of eight blocks requires all of them to be owned by one rank;
  /// this function collects all groups of 8 blocks to be compressed to a single
  /// rank.
  void PrepareCompression() {
    const int size = grid->get_world_size();
    const int rank = grid->rank();

    std::vector<BlockInfo> &I = grid->getBlocksInfo();
    std::vector<std::vector<MPI_Block>> send_blocks(size);
    std::vector<std::vector<MPI_Block>> recv_blocks(size);

    // Loop over blocks
    for (auto &b : I) {
#if DIMENSION == 3
      const long long nBlock =
          grid->getZforward(b.level, 2 * (b.index[0] / 2), 2 * (b.index[1] / 2),
                            2 * (b.index[2] / 2));
#else
      const long long nBlock = grid->getZforward(b.level, 2 * (b.index[0] / 2),
                                                 2 * (b.index[1] / 2));
#endif

      const BlockInfo &base = grid->getBlockInfoAll(b.level, nBlock);

      // If the 'base' block does not exist, no compression will take place.
      // Continue to next block. By now, if 'base' block is marked for
      // compression it means that the remaining 7 (3, in 2D) blocks will also
      // need compression, so we check if base.state == Compress.
      if (!grid->Tree(base).Exists() || base.state != Compress)
        continue;

      const BlockInfo &bCopy = grid->getBlockInfoAll(b.level, b.Z);
      const int baserank = grid->Tree(b.level, nBlock).rank();
      const int brank = grid->Tree(b.level, b.Z).rank();

      // if 'b' is NOT the 'base' block we send it to the rank that owns the
      // 'base' block.
      if (b.Z != nBlock) {
        if (baserank != rank && brank == rank) {
          send_blocks[baserank].push_back({bCopy});
          grid->Tree(b.level, b.Z).setrank(baserank);
        }
      }
      // if 'b' is the 'base' block we collect the remaining 7 (3, in 2D) blocks
      // that will be compressed with it.
      else {
#if DIMENSION == 3
        for (int k = 0; k < 2; k++)
#endif
          for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++) {
#if DIMENSION == 3
              const long long n = grid->getZforward(
                  b.level, b.index[0] + i, b.index[1] + j, b.index[2] + k);
#else
            const long long n =
                grid->getZforward(b.level, b.index[0] + i, b.index[1] + j);
#endif
              if (n == nBlock)
                continue;
              BlockInfo &temp = grid->getBlockInfoAll(b.level, n);
              const int temprank = grid->Tree(b.level, n).rank();
              if (temprank != rank) {
                recv_blocks[temprank].push_back({temp, false});
                grid->Tree(b.level, n).setrank(baserank);
              }
            }
      }
    }

    // 1/4 Perform the sends/receives of blocks
    std::vector<MPI_Request> requests;
    for (int r = 0; r < size; r++)
      if (r != rank) {
        if (recv_blocks[r].size() != 0) {
          MPI_Request req{};
          requests.push_back(req);
          MPI_Irecv(&recv_blocks[r][0], recv_blocks[r].size(), MPI_BLOCK, r,
                    2468, grid->getWorldComm(), &requests.back());
        }
        if (send_blocks[r].size() != 0) {
          MPI_Request req{};
          requests.push_back(req);
          MPI_Isend(&send_blocks[r][0], send_blocks[r].size(), MPI_BLOCK, r,
                    2468, grid->getWorldComm(), &requests.back());
        }
      }

    // 2/4 Do some work while sending/receiving. Here we deallocate the blocks
    // we sent.
    for (int r = 0; r < size; r++)
      for (int i = 0; i < (int)send_blocks[r].size(); i++) {
        grid->_dealloc(send_blocks[r][i].mn[0], send_blocks[r][i].mn[1]);
        grid->Tree(send_blocks[r][i].mn[0], send_blocks[r][i].mn[1])
            .setCheckCoarser();
      }

    // 3/4 Wait for communication to complete
    if (requests.size() != 0) {
      movedBlocks = true;
      MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
    }

    // 4/4 Allocate the blocks we received and copy data to them.
    for (int r = 0; r < size; r++)
      for (int i = 0; i < (int)recv_blocks[r].size(); i++) {
        const int level = (int)recv_blocks[r][i].mn[0];
        const long long Z = recv_blocks[r][i].mn[1];
        grid->_alloc(level, Z);
        BlockInfo &info = grid->getBlockInfoAll(level, Z);
        BlockType *b1 = (BlockType *)info.ptrBlock;
        assert(b1 != NULL);
        Real *a1 = &b1->data[0][0][0].member(0);
        std::memcpy(a1, recv_blocks[r][i].data, sizeof(BlockType));
      }
  }

  /// Redistributes blocks with diffusion algorithm along the 1D Space-Filling
  /// Hilbert Curve; block_distribution[i] is the number of blocks owned by rank
  /// i, for i=0,...,#of ranks -1
  void Balance_Diffusion(const bool verbose,
                         std::vector<long long> &block_distribution) {
    const int size = grid->get_world_size();
    const int rank = grid->rank();

    movedBlocks = false;
    {
      long long max_b = block_distribution[0];
      long long min_b = block_distribution[0];
      for (auto &b : block_distribution) {
        max_b = std::max(max_b, b);
        min_b = std::min(min_b, b);
      }
      const double ratio = static_cast<double>(max_b) / min_b;
      if (rank == 0 && verbose) {
        std::cout << "Load imbalance ratio = " << ratio << std::endl;
      }
      if (ratio > 1.01 || min_b == 0) {
        Balance_Global(block_distribution);
        return;
      }
    }

    const int right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    const int left = (rank == 0) ? MPI_PROC_NULL : rank - 1;

    const int my_blocks = grid->getBlocksInfo().size();

    int right_blocks, left_blocks;

    MPI_Request reqs[4];
    MPI_Irecv(&left_blocks, 1, MPI_INT, left, 123, grid->getWorldComm(),
              &reqs[0]);
    MPI_Irecv(&right_blocks, 1, MPI_INT, right, 456, grid->getWorldComm(),
              &reqs[1]);
    MPI_Isend(&my_blocks, 1, MPI_INT, left, 456, grid->getWorldComm(),
              &reqs[2]);
    MPI_Isend(&my_blocks, 1, MPI_INT, right, 123, grid->getWorldComm(),
              &reqs[3]);

    MPI_Waitall(4, &reqs[0], MPI_STATUSES_IGNORE);

    const int nu = 4;
    const int flux_left = (rank == 0) ? 0 : (my_blocks - left_blocks) / nu;
    const int flux_right =
        (rank == size - 1) ? 0 : (my_blocks - right_blocks) / nu;

    std::vector<BlockInfo> SortedInfos = grid->getBlocksInfo();
    if (flux_right != 0 || flux_left != 0)
      std::sort(SortedInfos.begin(), SortedInfos.end());

    std::vector<MPI_Block> send_left;
    std::vector<MPI_Block> recv_left;
    std::vector<MPI_Block> send_right;
    std::vector<MPI_Block> recv_right;

    std::vector<MPI_Request> request;

    if (flux_left > 0) // then I will send blocks to my left rank
    {
      send_left.resize(flux_left);
#pragma omp parallel for schedule(runtime)
      for (int i = 0; i < flux_left; i++)
        send_left[i].prepare(SortedInfos[i]);
      MPI_Request req{};
      request.push_back(req);
      MPI_Isend(&send_left[0], send_left.size(), MPI_BLOCK, left, 7890,
                grid->getWorldComm(), &request.back());
    } else if (flux_left < 0) // then I will receive blocks from my left rank
    {
      recv_left.resize(abs(flux_left));
      MPI_Request req{};
      request.push_back(req);
      MPI_Irecv(&recv_left[0], recv_left.size(), MPI_BLOCK, left, 4560,
                grid->getWorldComm(), &request.back());
    }
    if (flux_right > 0) // then I will send blocks to my right rank
    {
      send_right.resize(flux_right);
#pragma omp parallel for schedule(runtime)
      for (int i = 0; i < flux_right; i++)
        send_right[i].prepare(SortedInfos[my_blocks - i - 1]);
      MPI_Request req{};
      request.push_back(req);
      MPI_Isend(&send_right[0], send_right.size(), MPI_BLOCK, right, 4560,
                grid->getWorldComm(), &request.back());
    } else if (flux_right < 0) // then I will receive blocks from my right rank
    {
      recv_right.resize(abs(flux_right));
      MPI_Request req{};
      request.push_back(req);
      MPI_Irecv(&recv_right[0], recv_right.size(), MPI_BLOCK, right, 7890,
                grid->getWorldComm(), &request.back());
    }

    for (int i = 0; i < flux_right; i++) {
      BlockInfo &info = SortedInfos[my_blocks - i - 1];
      grid->_dealloc(info.level, info.Z);
      grid->Tree(info.level, info.Z).setrank(right);
    }

    for (int i = 0; i < flux_left; i++) {
      BlockInfo &info = SortedInfos[i];
      grid->_dealloc(info.level, info.Z);
      grid->Tree(info.level, info.Z).setrank(left);
    }

    if (request.size() != 0) {
      movedBlocks = true;
      MPI_Waitall(request.size(), &request[0], MPI_STATUSES_IGNORE);
    }
    int temp = movedBlocks ? 1 : 0;
    MPI_Request request_reduction;
    MPI_Iallreduce(MPI_IN_PLACE, &temp, 1, MPI_INT, MPI_SUM,
                   grid->getWorldComm(), &request_reduction);

    for (int i = 0; i < -flux_left; i++)
      AddBlock(recv_left[i].mn[0], recv_left[i].mn[1], recv_left[i].data);
    for (int i = 0; i < -flux_right; i++)
      AddBlock(recv_right[i].mn[0], recv_right[i].mn[1], recv_right[i].data);

    MPI_Wait(&request_reduction, MPI_STATUS_IGNORE);
    movedBlocks = (temp >= 1);
    grid->FillPos();
  }

  /// Redistributes all blocks evenly, along the 1D Space-Filling Hilbert Curve;
  /// all_b[i] is the number of blocks owned by rank i, for i=0,...,#of ranks -1
  void Balance_Global(std::vector<long long> &all_b) {
    const int size = grid->get_world_size();
    const int rank = grid->rank();

    // Redistribute all blocks evenly, along the 1D Hilbert curve.
    // all_b[i] = # of blocks currently owned by rank i.

    // sort blocks according to Z-index and level on the Hilbert curve.
    std::vector<BlockInfo> SortedInfos = grid->getBlocksInfo();
    std::sort(SortedInfos.begin(), SortedInfos.end());

    // compute the total number of blocks (total_load) and how many blocks each
    // rank should have, for a balanced load distribution
    long long total_load = 0;
    for (int r = 0; r < size; r++)
      total_load += all_b[r];
    long long my_load = total_load / size;
    if (rank < (total_load % size))
      my_load += 1;

    std::vector<long long> index_start(size);
    index_start[0] = 0;
    for (int r = 1; r < size; r++)
      index_start[r] = index_start[r - 1] + all_b[r - 1];

    long long ideal_index = (total_load / size) * rank;
    ideal_index += (rank < (total_load % size)) ? rank : (total_load % size);

    // now check the actual block distribution and mark the blocks that should
    // not be owned by a particular rank and should instead be sent to another
    // rank.
    std::vector<std::vector<MPI_Block>> send_blocks(size);
    std::vector<std::vector<MPI_Block>> recv_blocks(size);
    for (int r = 0; r < size; r++)
      if (rank != r) {
        { // check if I need to receive blocks
          const long long a1 = ideal_index;
          const long long a2 = ideal_index + my_load - 1;
          const long long b1 = index_start[r];
          const long long b2 = index_start[r] + all_b[r] - 1;
          const long long c1 = std::max(a1, b1);
          const long long c2 = std::min(a2, b2);
          if (c2 - c1 + 1 > 0)
            recv_blocks[r].resize(c2 - c1 + 1);
        }
        { // check if I need to send blocks
          long long other_ideal_index = (total_load / size) * r;
          other_ideal_index +=
              (r < (total_load % size)) ? r : (total_load % size);
          long long other_load = total_load / size;
          if (r < (total_load % size))
            other_load += 1;
          const long long a1 = other_ideal_index;
          const long long a2 = other_ideal_index + other_load - 1;
          const long long b1 = index_start[rank];
          const long long b2 = index_start[rank] + all_b[rank] - 1;
          const long long c1 = std::max(a1, b1);
          const long long c2 = std::min(a2, b2);
          if (c2 - c1 + 1 > 0)
            send_blocks[r].resize(c2 - c1 + 1);
        }
      }

    // perform the sends and receives of blocks
    int tag = 12345;
    std::vector<MPI_Request> requests;
    for (int r = 0; r < size; r++)
      if (recv_blocks[r].size() != 0) {
        MPI_Request req{};
        requests.push_back(req);
        MPI_Irecv(recv_blocks[r].data(), recv_blocks[r].size(), MPI_BLOCK, r,
                  tag, grid->getWorldComm(), &requests.back());
      }

    long long counter_S = 0;
    long long counter_E = 0;
    for (int r = 0; r < rank; r++)
      if (send_blocks[r].size() != 0) {
        for (size_t i = 0; i < send_blocks[r].size(); i++)
          send_blocks[r][i].prepare(SortedInfos[counter_S + i]);
        counter_S += send_blocks[r].size();
        MPI_Request req{};
        requests.push_back(req);
        MPI_Isend(send_blocks[r].data(), send_blocks[r].size(), MPI_BLOCK, r,
                  tag, grid->getWorldComm(), &requests.back());
      }
    for (int r = size - 1; r > rank; r--)
      if (send_blocks[r].size() != 0) {
        for (size_t i = 0; i < send_blocks[r].size(); i++)
          send_blocks[r][i].prepare(
              SortedInfos[SortedInfos.size() - 1 - (counter_E + i)]);
        counter_E += send_blocks[r].size();
        MPI_Request req{};
        requests.push_back(req);
        MPI_Isend(send_blocks[r].data(), send_blocks[r].size(), MPI_BLOCK, r,
                  tag, grid->getWorldComm(), &requests.back());
      }

    // no need to wait here, do some work first!
    // MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    // do some work while sending/receiving, by deallocating the blocks that are
    // being sent
    movedBlocks = true;
    std::vector<long long> deallocIDs;
    counter_S = 0;
    counter_E = 0;
    for (int r = 0; r < size; r++)
      if (send_blocks[r].size() != 0) {
        if (r < rank) {
          for (size_t i = 0; i < send_blocks[r].size(); i++) {
            BlockInfo &info = SortedInfos[counter_S + i];
            deallocIDs.push_back(info.blockID_2);
            grid->Tree(info.level, info.Z).setrank(r);
          }
          counter_S += send_blocks[r].size();
        } else {
          for (size_t i = 0; i < send_blocks[r].size(); i++) {
            BlockInfo &info =
                SortedInfos[SortedInfos.size() - 1 - (counter_E + i)];
            deallocIDs.push_back(info.blockID_2);
            grid->Tree(info.level, info.Z).setrank(r);
          }
          counter_E += send_blocks[r].size();
        }
      }
    grid->dealloc_many(deallocIDs);

    // wait for communication
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

// allocate received blocks
#pragma omp parallel
    {
      for (int r = 0; r < size; r++)
        if (recv_blocks[r].size() != 0) {
#pragma omp for
          for (size_t i = 0; i < recv_blocks[r].size(); i++)
            AddBlock(recv_blocks[r][i].mn[0], recv_blocks[r][i].mn[1],
                     recv_blocks[r][i].data);
        }
    }
    grid->FillPos();
  }
};

} // namespace cubism

namespace cubism {

/**
 * @brief Class responsible for mesh refinement of a GridMPI.
 *
 * This class can label each GridBlock of a GridMPI as requiring
 * refinement/compression/nothing. It can then perform interpolation of points
 * for refinement and averaging down for compression, followed by load balancing
 * of GridBlocks among different processes.
 *
 * The user should use this class through its constructor and through the
 * functions 'Tag', 'TagLike' and 'Adapt'. When constructed, the user provides
 * this class with a Grid and two numbers: the tolerance for refinement (Rtol)
 * and the tolerance for compression (Ctol). By calling 'Tag', GridBlocks with
 * gridpoints that have magnitude() > Rtol will be tagged for refinement and
 * those with magnitude() < Ctol for compression. Alternatively, 'TagLike' can
 * be used to copy the tags of blocks from another already tagged grid to this
 * grid. Once tagged, the user needs to call 'Adapt', which will adapt the mesh,
 * interpolate new points and do the load-balancing of blocks among processes.
 *
 * In order to change the default refinement criterion, a new class can inherit
 * from this class and overload the function 'TagLoadedBlock'. In order to
 * change the default refinement interpolation, a new class can inherit from
 * this class and overload the function 'RefineBlocks'.
 *
 * @tparam TLab  The BlockLab type used for halo cells exchange and boundary
 * condition enforcement, when interpolation of gridpoints happens after mesh
 * refinement.
 */
template <typename TLab> class MeshAdaptation {
protected:
  typedef typename TLab::GridType TGrid;
  typedef typename TGrid::Block BlockType;
  typedef typename TGrid::BlockType::ElementType ElementType;
  typedef typename TGrid::BlockType::ElementType::RealType Real;
  typedef SynchronizerMPI_AMR<Real, TGrid> SynchronizerMPIType;

  StencilInfo stencil;  ///< stencil of +-1 point, needed for 2nd-order
                        ///< refinement interpolation
  bool CallValidStates; ///< will be true when 'Tag' is called and some
                        ///< refinement/compression is needed
  bool boundary_needed; ///< set true to update the boundary blocks of each
                        ///< GridMPI
  LoadBalancer<TGrid> *Balancer; ///< load-balancing of blocks
  TGrid *grid;                   ///< pointer to Grid that will be adapted
  double time; ///< (optional) time of simulation, for time-dependent refinement
               ///< criteria
  bool basic_refinement; ///< set to false if no interpolation is to be
                         ///< performed after refinement
  double tolerance_for_refinement;  ///< compare 'magnitude()' of each gridpoint
                                    ///< to this number, to check if refinement
                                    ///< is needed
  double tolerance_for_compression; ///< compare 'magnitude()' of each gridpoint
                                    ///< to this number, to check if compression
                                    ///< is needed
  std::vector<long long>
      dealloc_IDs; ///< blockIDs for blocks that are deallocated because of mesh
                   ///< refinement/compression

public:
  /**
   * @brief Class constructor.
   *
   * @param g The Grid to be refined/compressed.
   * @param Rtol Tolerance for refinement.
   * @param Ctol Tolerance for compression.
   */
  MeshAdaptation(TGrid &g, double Rtol, double Ctol) {
    grid = &g;

    tolerance_for_refinement = Rtol;
    tolerance_for_compression = Ctol;

    boundary_needed = false;

    constexpr int Gx = 1;
    constexpr int Gy = 1;
    constexpr int Gz = DIMENSION == 3 ? 1 : 0;
    stencil.sx = -Gx;
    stencil.sy = -Gy;
    stencil.sz = -Gz;
    stencil.ex = Gx + 1;
    stencil.ey = Gy + 1;
    stencil.ez = Gz + 1;
    stencil.tensorial = true;
    for (int i = 0; i < ElementType::DIM; i++)
      stencil.selcomponents.push_back(i);

    Balancer = new LoadBalancer<TGrid>(*grid);
  }

  /**
   * @brief Class destructor.
   */
  virtual ~MeshAdaptation() { delete Balancer; }

  /**
   * @brief Tag each block of this grid for refinement/compression based on
   * criterion from 'TagLoadedBlock'.
   * @param t Current time of the simulation; used only for time-dependent
   * boundary conditions.
   */
  void Tag(double t = 0) {
    time = t;
    boundary_needed = true;

    SynchronizerMPI_AMR<Real, TGrid> *Synch = grid->sync(stencil);

    CallValidStates = false;
    bool Reduction = false;
    MPI_Request Reduction_req;
    int tmp;

    std::vector<BlockInfo *> &inner = Synch->avail_inner();
    TagBlocksVector(inner, Reduction, Reduction_req, tmp);

    std::vector<BlockInfo *> &halo = Synch->avail_halo();
    TagBlocksVector(halo, Reduction, Reduction_req, tmp);

    if (!Reduction) {
      tmp = CallValidStates ? 1 : 0;
      Reduction = true;
      MPI_Iallreduce(MPI_IN_PLACE, &tmp, 1, MPI_INT, MPI_SUM,
                     grid->getWorldComm(), &Reduction_req);
    }

    MPI_Wait(&Reduction_req, MPI_STATUS_IGNORE);
    CallValidStates = (tmp > 0);

    grid->boundary = halo;

    if (CallValidStates)
      ValidStates();
  }

  /**
   * @brief Refine/compress the mesh after blocks are tagged.
   * @param t Current time of the simulation; used only for time-dependent
   * boundary conditions.
   * @param verbosity Boolean variable controlling screen output.
   * @param basic Boolean variable; if set to false, no refinement interpolation
   * is performed and blocks are simply allocated (and filled with nothing)
   * after refinement.
   */
  void Adapt(double t = 0, bool verbosity = false, bool basic = false) {
    basic_refinement = basic;
    SynchronizerMPI_AMR<Real, TGrid> *Synch = nullptr;
    if (basic == false) {
      Synch = grid->sync(stencil);
      // TODO: the line below means there's no computation & communication
      // overlap here
      grid->boundary = Synch->avail_halo();
      if (boundary_needed)
        grid->UpdateBoundary();
    }

    int r = 0;
    int c = 0;

    std::vector<int> m_com;
    std::vector<int> m_ref;
    std::vector<long long> n_com;
    std::vector<long long> n_ref;

    std::vector<BlockInfo> &I = grid->getBlocksInfo();

    long long blocks_after = I.size();

    for (auto &info : I) {
      if (info.state == Refine) {
        m_ref.push_back(info.level);
        n_ref.push_back(info.Z);
        blocks_after += (1 << DIMENSION) - 1;
        r++;
      } else if (info.state == Compress && info.index[0] % 2 == 0 &&
                 info.index[1] % 2 == 0 && info.index[2] % 2 == 0) {
        m_com.push_back(info.level);
        n_com.push_back(info.Z);
        c++;
      } else if (info.state == Compress) {
        blocks_after--;
      }
    }
    MPI_Request requests[2];
    int temp[2] = {r, c};
    int result[2];
    int size;
    MPI_Comm_size(grid->getWorldComm(), &size);
    std::vector<long long> block_distribution(size);
    MPI_Iallreduce(&temp, &result, 2, MPI_INT, MPI_SUM, grid->getWorldComm(),
                   &requests[0]);
    MPI_Iallgather(&blocks_after, 1, MPI_LONG_LONG, block_distribution.data(),
                   1, MPI_LONG_LONG, grid->getWorldComm(), &requests[1]);

    dealloc_IDs.clear();

#ifdef CUBISM_USE_ONETBB
#pragma omp parallel
#endif
    {
      TLab lab;
      if (Synch != nullptr)
        lab.prepare(*grid, Synch->getstencil());
#ifdef CUBISM_USE_ONETBB
#pragma omp for
#endif
      for (size_t i = 0; i < m_ref.size(); i++) {
        refine_1(m_ref[i], n_ref[i], lab);
      }
#ifdef CUBISM_USE_ONETBB
#pragma omp for
#endif
      for (size_t i = 0; i < m_ref.size(); i++) {
        refine_2(m_ref[i], n_ref[i]);
      }
    }
    grid->dealloc_many(dealloc_IDs);

    Balancer->PrepareCompression();

    dealloc_IDs.clear();

#ifdef CUBISM_USE_ONETBB
#pragma omp parallel for
#endif
    for (size_t i = 0; i < m_com.size(); i++) {
      compress(m_com[i], n_com[i]);
    }

    grid->dealloc_many(dealloc_IDs);

    MPI_Waitall(2, requests, MPI_STATUS_IGNORE);
    if (verbosity) {
      std::cout
          << "==============================================================\n";
      std::cout << " refined:" << result[0] << "   compressed:" << result[1]
                << std::endl;
      std::cout
          << "=============================================================="
          << std::endl;
    }

    Balancer->Balance_Diffusion(verbosity, block_distribution);

    if (result[0] > 0 || result[1] > 0 || Balancer->movedBlocks) {
      grid->UpdateFluxCorrection = true;
      grid->UpdateGroups = true;

      grid->UpdateBlockInfoAll_States(false);

      auto it = grid->SynchronizerMPIs.begin();
      while (it != grid->SynchronizerMPIs.end()) {
        (*it->second)._Setup();
        it++;
      }
    }
  }

  /**
   * @brief Tag each block of this grid for refinement/compression by copying
   * the tags of the given BlockInfos.
   * @param I1 Vector of BlockInfos whose 'state' (refine/compress/leave) will
   * be copied to the BlockInfos of this grid.
   */
  void TagLike(const std::vector<BlockInfo> &I1) {
    std::vector<BlockInfo> &I2 = grid->getBlocksInfo();
    for (size_t i1 = 0; i1 < I2.size(); i1++) {
      BlockInfo &ary0 = I2[i1];
      BlockInfo &info = grid->getBlockInfoAll(ary0.level, ary0.Z);
      for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1;
           i++)
        for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1;
             j++)
#if DIMENSION == 3
          for (int k = 2 * (info.index[2] / 2);
               k <= 2 * (info.index[2] / 2) + 1; k++) {
            const long long n = grid->getZforward(info.level, i, j, k);
            BlockInfo &infoNei = grid->getBlockInfoAll(info.level, n);
            infoNei.state = Leave;
          }
#else
        {
          const long long n = grid->getZforward(info.level, i, j);
          BlockInfo &infoNei = grid->getBlockInfoAll(info.level, n);
          infoNei.state = Leave;
        }
#endif
      info.state = Leave;
      ary0.state = Leave;
    }
#pragma omp parallel for
    for (size_t i = 0; i < I1.size(); i++) {
      const BlockInfo &info1 = I1[i];
      BlockInfo &info2 = I2[i];
      BlockInfo &info3 = grid->getBlockInfoAll(info2.level, info2.Z);
      info2.state = info1.state;
      info3.state = info1.state;
      if (info2.state == Compress) {
        const int i2 = 2 * (info2.index[0] / 2);
        const int j2 = 2 * (info2.index[1] / 2);
#if DIMENSION == 3
        const int k2 = 2 * (info2.index[2] / 2);
        const long long n = grid->getZforward(info2.level, i2, j2, k2);
#else
        const long long n = grid->getZforward(info2.level, i2, j2);
#endif
        BlockInfo &infoNei = grid->getBlockInfoAll(info2.level, n);
        infoNei.state = Compress;
      }
    }
  }

protected:
  /**
   * @brief Auxiliary function to tag a vector of blocks
   * @param I Vector of BlockInfos to tag.
   * @param Reduction Boolean that will be set to true if any block is tagged;
   * setting to true will cause a call to 'ValidStates()' after tagging blocks.
   * @param Reduction_req MPI request that will be used if 'Reduction' is true
   * @param tmp Same value as 'Reduction' by this is an integer
   */
  void TagBlocksVector(std::vector<BlockInfo *> &I, bool &Reduction,
                       MPI_Request &Reduction_req, int &tmp) {
    const int levelMax = grid->getlevelMax();
#pragma omp parallel
    {
#pragma omp for schedule(dynamic, 1)
      for (size_t i = 0; i < I.size(); i++) {
        BlockInfo &info = grid->getBlockInfoAll(I[i]->level, I[i]->Z);

        I[i]->state = TagLoadedBlock(info);

        const bool maxLevel =
            (I[i]->state == Refine) && (I[i]->level == levelMax - 1);
        const bool minLevel = (I[i]->state == Compress) && (I[i]->level == 0);

        if (maxLevel || minLevel)
          I[i]->state = Leave;

        info.state = I[i]->state;
        if (info.state != Leave) {
#pragma omp critical
          {
            CallValidStates = true;
            if (!Reduction) {
              tmp = 1;
              Reduction = true;
              MPI_Iallreduce(MPI_IN_PLACE, &tmp, 1, MPI_INT, MPI_SUM,
                             grid->getWorldComm(), &Reduction_req);
            }
          }
        }
      }
    }
  }

  /**
   * @brief First step of refinement of a block.
   *
   * The new blocks are allocated and the interpolation needed for refinement is
   * perfomed. The parent block will be deallocated in step 2 (refine_2).
   *
   * @param level The refinement level of the block to be refined.
   * @param Z The Z-order index of the block to be refined.
   */
  void refine_1(const int level, const long long Z, TLab &lab) {
    BlockInfo &parent = grid->getBlockInfoAll(level, Z);
    parent.state = Leave;
    if (basic_refinement == false)
      lab.load(parent, time, true);

    const int p[3] = {parent.index[0], parent.index[1], parent.index[2]};

    assert(parent.ptrBlock != NULL);
    assert(level <= grid->getlevelMax() - 1);
#if DIMENSION == 3
    BlockType *Blocks[8];
    for (int k = 0; k < 2; k++)
      for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
          const long long nc = grid->getZforward(level + 1, 2 * p[0] + i,
                                                 2 * p[1] + j, 2 * p[2] + k);
          BlockInfo &Child = grid->getBlockInfoAll(level + 1, nc);
          Child.state = Leave;
          grid->_alloc(level + 1, nc);
          grid->Tree(level + 1, nc).setCheckCoarser();
          Blocks[k * 4 + j * 2 + i] = (BlockType *)Child.ptrBlock;
        }
#else
    BlockType *Blocks[4];
    for (int j = 0; j < 2; j++)
      for (int i = 0; i < 2; i++) {
        const long long nc =
            grid->getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
        BlockInfo &Child = grid->getBlockInfoAll(level + 1, nc);
        Child.state = Leave;
        grid->_alloc(level + 1, nc);
        grid->Tree(level + 1, nc).setCheckCoarser();
        Blocks[j * 2 + i] = (BlockType *)Child.ptrBlock;
      }
#endif
    if (basic_refinement == false)
      RefineBlocks(Blocks, lab);
  }

  /**
   * @brief Second step of refinement of a block.
   *
   * After all blocks are refined with refine_1, we can deallocate their parent
   * blocks here.
   *
   * @param level The refinement level of the block to be refined.
   * @param Z The Z-order index of the block to be refined.
   */
  void refine_2(const int level, const long long Z) {
#pragma omp critical
    { dealloc_IDs.push_back(grid->getBlockInfoAll(level, Z).blockID_2); }

    BlockInfo &parent = grid->getBlockInfoAll(level, Z);
    grid->Tree(parent).setCheckFiner();
    parent.state = Leave;

    int p[3] = {parent.index[0], parent.index[1], parent.index[2]};
#if DIMENSION == 3
    for (int k = 0; k < 2; k++)
      for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
          const long long nc = grid->getZforward(level + 1, 2 * p[0] + i,
                                                 2 * p[1] + j, 2 * p[2] + k);
          BlockInfo &Child = grid->getBlockInfoAll(level + 1, nc);
          grid->Tree(Child).setrank(grid->rank());
          if (level + 2 < grid->getlevelMax())
            for (int i0 = 0; i0 < 2; i0++)
              for (int i1 = 0; i1 < 2; i1++)
                for (int i2 = 0; i2 < 2; i2++)
                  grid->Tree(level + 2, Child.Zchild[i0][i1][i2])
                      .setCheckCoarser();
        }
#else
    for (int j = 0; j < 2; j++)
      for (int i = 0; i < 2; i++) {
        const long long nc =
            grid->getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
        BlockInfo &Child = grid->getBlockInfoAll(level + 1, nc);
        grid->Tree(Child).setrank(grid->rank());
        if (level + 2 < grid->getlevelMax())
          for (int i0 = 0; i0 < 2; i0++)
            for (int i1 = 0; i1 < 2; i1++)
              grid->Tree(level + 2, Child.Zchild[i0][i1][1]).setCheckCoarser();
      }
#endif
  }

  /**
   * @brief Compress eight blocks.
   *
   * The 'bottom left' block (i,j,k) is provided in the input, via its
   * refinement level and Z-order index. The top right block would be the block
   * (i+1,j+1,k+1).
   *
   * @param level The refinement level of the bottom left block to be refined.
   * @param Z The Z-order index of the bottom left block to be refined.
   */
  void compress(const int level, const long long Z) {
    assert(level > 0);

    BlockInfo &info = grid->getBlockInfoAll(level, Z);

    assert(info.state == Compress);

#if DIMENSION == 3
    BlockType *Blocks[8];
    for (int K = 0; K < 2; K++)
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          const int blk = K * 4 + J * 2 + I;
          const long long n = grid->getZforward(
              level, info.index[0] + I, info.index[1] + J, info.index[2] + K);
          Blocks[blk] = (BlockType *)(grid->getBlockInfoAll(level, n)).ptrBlock;
        }

    const int nx = BlockType::sizeX;
    const int ny = BlockType::sizeY;
    const int nz = BlockType::sizeZ;
    const int offsetX[2] = {0, nx / 2};
    const int offsetY[2] = {0, ny / 2};
    const int offsetZ[2] = {0, nz / 2};
    if (basic_refinement == false)
      for (int K = 0; K < 2; K++)
        for (int J = 0; J < 2; J++)
          for (int I = 0; I < 2; I++) {
            BlockType &b = *Blocks[K * 4 + J * 2 + I];
            for (int k = 0; k < nz; k += 2)
              for (int j = 0; j < ny; j += 2)
                for (int i = 0; i < nx; i += 2) {
#ifdef PRESERVE_SYMMETRY
                  const ElementType B1 = b(i, j, k) + b(i + 1, j + 1, k + 1);
                  const ElementType B2 = b(i + 1, j, k) + b(i, j + 1, k + 1);
                  const ElementType B3 = b(i, j + 1, k) + b(i + 1, j, k + 1);
                  const ElementType B4 = b(i, j, k + 1) + b(i + 1, j + 1, k);
                  (*Blocks[0])(i / 2 + offsetX[I], j / 2 + offsetY[J],
                               k / 2 + offsetZ[K]) =
                      0.125 * ConsistentSum<ElementType>(B1, B2, B3, B4);
#else
                  (*Blocks[0])(i / 2 + offsetX[I], j / 2 + offsetY[J],
                               k / 2 + offsetZ[K]) =
                      0.125 * ((b(i, j, k) + b(i + 1, j + 1, k + 1)) +
                               (b(i + 1, j, k) + b(i, j + 1, k + 1)) +
                               (b(i, j + 1, k) + b(i + 1, j, k + 1)) +
                               (b(i + 1, j + 1, k) + b(i, j, k + 1)));
#endif
                }
          }

    const long long np = grid->getZforward(
        level - 1, info.index[0] / 2, info.index[1] / 2, info.index[2] / 2);
    BlockInfo &parent = grid->getBlockInfoAll(level - 1, np);
    grid->Tree(parent.level, parent.Z).setrank(grid->rank());
    parent.ptrBlock = info.ptrBlock;
    parent.state = Leave;
    if (level - 2 >= 0)
      grid->Tree(level - 2, parent.Zparent).setCheckFiner();

    for (int K = 0; K < 2; K++)
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          const long long n = grid->getZforward(
              level, info.index[0] + I, info.index[1] + J, info.index[2] + K);
          if (I + J + K == 0) {
            grid->FindBlockInfo(level, n, level - 1, np);
          } else {
#pragma omp critical
            {
              dealloc_IDs.push_back(grid->getBlockInfoAll(level, n).blockID_2);
            }
          }
          grid->Tree(level, n).setCheckCoarser();
          grid->getBlockInfoAll(level, n).state = Leave;
        }
#endif
#if DIMENSION == 2
    BlockType *Blocks[4];
    for (int J = 0; J < 2; J++)
      for (int I = 0; I < 2; I++) {
        const int blk = J * 2 + I;
        const long long n =
            grid->getZforward(level, info.index[0] + I, info.index[1] + J);
        Blocks[blk] = (BlockType *)(grid->getBlockInfoAll(level, n)).ptrBlock;
      }

    const int nx = BlockType::sizeX;
    const int ny = BlockType::sizeY;
    const int offsetX[2] = {0, nx / 2};
    const int offsetY[2] = {0, ny / 2};
    if (basic_refinement == false)
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          BlockType &b = *Blocks[J * 2 + I];
          for (int j = 0; j < ny; j += 2)
            for (int i = 0; i < nx; i += 2) {
              ElementType average = 0.25 * ((b(i, j, 0) + b(i + 1, j + 1, 0)) +
                                            (b(i + 1, j, 0) + b(i, j + 1, 0)));
              (*Blocks[0])(i / 2 + offsetX[I], j / 2 + offsetY[J], 0) = average;
            }
        }
    const long long np =
        grid->getZforward(level - 1, info.index[0] / 2, info.index[1] / 2);
    BlockInfo &parent = grid->getBlockInfoAll(level - 1, np);
    grid->Tree(parent.level, parent.Z).setrank(grid->rank());
    parent.ptrBlock = info.ptrBlock;
    parent.state = Leave;
    if (level - 2 >= 0)
      grid->Tree(level - 2, parent.Zparent).setCheckFiner();

    for (int J = 0; J < 2; J++)
      for (int I = 0; I < 2; I++) {
        const long long n =
            grid->getZforward(level, info.index[0] + I, info.index[1] + J);
        if (I + J == 0) {
          grid->FindBlockInfo(level, n, level - 1, np);
        } else {
#pragma omp critical
          { dealloc_IDs.push_back(grid->getBlockInfoAll(level, n).blockID_2); }
        }
        grid->Tree(level, n).setCheckCoarser();
        grid->getBlockInfoAll(level, n).state = Leave;
      }
#endif
  }

  /**
   * @brief Make sure adjacent blocks of the to-be-adapted mesh do not differ by
   * more than one refinement level.
   *
   * Given a set of tagged blocks, this function will mark some additional
   * blocks for refinement, to make sure no adjacent blocks differ by more than
   * one refinement level. It will also unmark some blocks from being
   * compressed, if their adjacent blocks do not need compression and/or belong
   * to a finer refinement level.
   */
  void ValidStates() {
    const std::array<int, 3> blocksPerDim = grid->getMaxBlocks();
    const int levelMin = 0;
    const int levelMax = grid->getlevelMax();
    const bool xperiodic = grid->xperiodic;
    const bool yperiodic = grid->yperiodic;
    const bool zperiodic = grid->zperiodic;

    std::vector<BlockInfo> &I = grid->getBlocksInfo();

#pragma omp parallel for
    for (size_t j = 0; j < I.size(); j++) {
      BlockInfo &info = I[j];

      if ((info.state == Refine && info.level == levelMax - 1) ||
          (info.state == Compress && info.level == levelMin)) {
        info.state = Leave;
        (grid->getBlockInfoAll(info.level, info.Z)).state = Leave;
      }
      if (info.state != Leave) {
        info.changed2 = true;
        (grid->getBlockInfoAll(info.level, info.Z)).changed2 = info.changed2;
      }
    }

    // 1.Change states of blocks next to finer resolution blocks
    // 2.Change states of blocks next to same resolution blocks
    // 3.Compress a block only if all blocks with the same parent need
    // compression
    bool clean_boundary = true;
    for (int m = levelMax - 1; m >= levelMin; m--) {
      // 1.
      for (size_t j = 0; j < I.size(); j++) {
        BlockInfo &info = I[j];
        if (info.level == m && info.state != Refine &&
            info.level != levelMax - 1) {
          const int TwoPower = 1 << info.level;
          const bool xskin = info.index[0] == 0 ||
                             info.index[0] == blocksPerDim[0] * TwoPower - 1;
          const bool yskin = info.index[1] == 0 ||
                             info.index[1] == blocksPerDim[1] * TwoPower - 1;
          const bool zskin = info.index[2] == 0 ||
                             info.index[2] == blocksPerDim[2] * TwoPower - 1;
          const int xskip = info.index[0] == 0 ? -1 : 1;
          const int yskip = info.index[1] == 0 ? -1 : 1;
          const int zskip = info.index[2] == 0 ? -1 : 1;

          for (int icode = 0; icode < 27; icode++) {
            if (info.state == Refine)
              break;
            if (icode == 1 * 1 + 3 * 1 + 9 * 1)
              continue;
            const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                                 (icode / 9) % 3 - 1};
            if (!xperiodic && code[0] == xskip && xskin)
              continue;
            if (!yperiodic && code[1] == yskip && yskin)
              continue;
            if (!zperiodic && code[2] == zskip && zskin)
              continue;
#if DIMENSION == 2
            if (code[2] != 0)
              continue;
#endif

            if (grid->Tree(info.level, info.Znei_(code[0], code[1], code[2]))
                    .CheckFiner()) {
              if (info.state == Compress) {
                info.state = Leave;
                (grid->getBlockInfoAll(info.level, info.Z)).state = Leave;
              }
              // if (info.level == levelMax - 1) break;

              const int tmp = abs(code[0]) + abs(code[1]) + abs(code[2]);
              int Bstep = 1; // face
              if (tmp == 2)
                Bstep = 3; // edge
              else if (tmp == 3)
                Bstep = 4; // corner

// loop over blocks that make up face/edge/corner(respectively 4,2 or 1 blocks)
#if DIMENSION == 3
              for (int B = 0; B <= 3; B += Bstep)
#else
              for (int B = 0; B <= 1; B += Bstep)
#endif
              {
                const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
                const int iNei = 2 * info.index[0] + std::max(code[0], 0) +
                                 code[0] +
                                 (B % 2) * std::max(0, 1 - abs(code[0]));
                const int jNei = 2 * info.index[1] + std::max(code[1], 0) +
                                 code[1] + aux * std::max(0, 1 - abs(code[1]));
#if DIMENSION == 3
                const int kNei = 2 * info.index[2] + std::max(code[2], 0) +
                                 code[2] +
                                 (B / 2) * std::max(0, 1 - abs(code[2]));
                const long long zzz =
                    grid->getZforward(m + 1, iNei, jNei, kNei);
#else
                const long long zzz = grid->getZforward(m + 1, iNei, jNei);
#endif
                BlockInfo &FinerNei = grid->getBlockInfoAll(m + 1, zzz);
                State NeiState = FinerNei.state;
                if (NeiState == Refine) {
                  info.state = Refine;
                  (grid->getBlockInfoAll(info.level, info.Z)).state = Refine;
                  info.changed2 = true;
                  (grid->getBlockInfoAll(info.level, info.Z)).changed2 = true;
                  break;
                }
              }
            }
          }
        }
      }

      grid->UpdateBoundary(clean_boundary);
      clean_boundary = false;
      if (m == levelMin)
        break;

      // 2.
      for (size_t j = 0; j < I.size(); j++) {
        BlockInfo &info = I[j];
        if (info.level == m && info.state == Compress) {
          const int aux = 1 << info.level;
          const bool xskin =
              info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
          const bool yskin =
              info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
          const bool zskin =
              info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
          const int xskip = info.index[0] == 0 ? -1 : 1;
          const int yskip = info.index[1] == 0 ? -1 : 1;
          const int zskip = info.index[2] == 0 ? -1 : 1;

          for (int icode = 0; icode < 27; icode++) {
            if (icode == 1 * 1 + 3 * 1 + 9 * 1)
              continue;
            const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1,
                                 (icode / 9) % 3 - 1};
            if (!xperiodic && code[0] == xskip && xskin)
              continue;
            if (!yperiodic && code[1] == yskip && yskin)
              continue;
            if (!zperiodic && code[2] == zskip && zskin)
              continue;
#if DIMENSION == 2
            if (code[2] != 0)
              continue;
#endif

            BlockInfo &infoNei = grid->getBlockInfoAll(
                info.level, info.Znei_(code[0], code[1], code[2]));
            if (grid->Tree(infoNei).Exists() && infoNei.state == Refine) {
              info.state = Leave;
              (grid->getBlockInfoAll(info.level, info.Z)).state = Leave;
              break;
            }
          }
        }
      }
    } // m

    // 3.
    for (size_t jjj = 0; jjj < I.size(); jjj++) {
      BlockInfo &info = I[jjj];
      const int m = info.level;
      bool found = false;
      for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1;
           i++)
        for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1;
             j++)
          for (int k = 2 * (info.index[2] / 2);
               k <= 2 * (info.index[2] / 2) + 1; k++) {
#if DIMENSION == 3
            const long long n = grid->getZforward(m, i, j, k);
#else
            const long long n = grid->getZforward(m, i, j);
#endif
            BlockInfo &infoNei = grid->getBlockInfoAll(m, n);
            if (grid->Tree(infoNei).Exists() == false ||
                infoNei.state != Compress) {
              found = true;
              if (info.state == Compress) {
                info.state = Leave;
                (grid->getBlockInfoAll(info.level, info.Z)).state = Leave;
              }
              break;
            }
          }
      if (found)
        for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1;
             i++)
          for (int j = 2 * (info.index[1] / 2);
               j <= 2 * (info.index[1] / 2) + 1; j++)
            for (int k = 2 * (info.index[2] / 2);
                 k <= 2 * (info.index[2] / 2) + 1; k++) {
#if DIMENSION == 3
              const long long n = grid->getZforward(m, i, j, k);
#else
              const long long n = grid->getZforward(m, i, j);
#endif
              BlockInfo &infoNei = grid->getBlockInfoAll(m, n);
              if (grid->Tree(infoNei).Exists() && infoNei.state == Compress)
                infoNei.state = Leave;
            }
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // Virtual functions that can be overwritten by user
  ////////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief How cells are interpolated after refinement
   *
   * Default interpolation is a 2nd order Taylor expansion. Can be overidden by
   * a derived class, to enable a custom refinement interpolation.
   *
   * @param B Pointers to the eight new blocks to be interpolated.
   */
  virtual void RefineBlocks(BlockType *B[8], TLab &Lab) {
    const int nx = BlockType::sizeX;
    const int ny = BlockType::sizeY;

    int offsetX[2] = {0, nx / 2};
    int offsetY[2] = {0, ny / 2};

#if DIMENSION == 3
    const int nz = BlockType::sizeZ;
    int offsetZ[2] = {0, nz / 2};

    for (int K = 0; K < 2; K++)
      for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++) {
          BlockType &b = *B[K * 4 + J * 2 + I];
          b.clear();

          for (int k = 0; k < nz; k += 2)
            for (int j = 0; j < ny; j += 2)
              for (int i = 0; i < nx; i += 2) {
                const int x = i / 2 + offsetX[I];
                const int y = j / 2 + offsetY[J];
                const int z = k / 2 + offsetZ[K];
                const ElementType dudx =
                    0.5 * (Lab(x + 1, y, z) - Lab(x - 1, y, z));
                const ElementType dudy =
                    0.5 * (Lab(x, y + 1, z) - Lab(x, y - 1, z));
                const ElementType dudz =
                    0.5 * (Lab(x, y, z + 1) - Lab(x, y, z - 1));
                const ElementType dudx2 =
                    (Lab(x + 1, y, z) + Lab(x - 1, y, z)) - 2.0 * Lab(x, y, z);
                const ElementType dudy2 =
                    (Lab(x, y + 1, z) + Lab(x, y - 1, z)) - 2.0 * Lab(x, y, z);
                const ElementType dudz2 =
                    (Lab(x, y, z + 1) + Lab(x, y, z - 1)) - 2.0 * Lab(x, y, z);
                const ElementType dudxdy =
                    0.25 * ((Lab(x + 1, y + 1, z) + Lab(x - 1, y - 1, z)) -
                            (Lab(x + 1, y - 1, z) + Lab(x - 1, y + 1, z)));
                const ElementType dudxdz =
                    0.25 * ((Lab(x + 1, y, z + 1) + Lab(x - 1, y, z - 1)) -
                            (Lab(x + 1, y, z - 1) + Lab(x - 1, y, z + 1)));
                const ElementType dudydz =
                    0.25 * ((Lab(x, y + 1, z + 1) + Lab(x, y - 1, z - 1)) -
                            (Lab(x, y + 1, z - 1) + Lab(x, y - 1, z + 1)));

#ifdef PRESERVE_SYMMETRY
                const ElementType d2 =
                    0.03125 * ConsistentSum<ElementType>(dudx2, dudy2, dudz2);
                b(i, j, k) =
                    Lab(x, y, z) +
                    (0.25 * ConsistentSum<ElementType>(
                                -(1.0) * dudx, -(1.0) * dudy, -(1.0) * dudz) +
                     d2) +
                    0.0625 * ConsistentSum(dudxdy, dudxdz, dudydz);
                b(i + 1, j, k) =
                    Lab(x, y, z) +
                    (0.25 * ConsistentSum<ElementType>(dudx, -(1.0) * dudy,
                                                       -(1.0) * dudz) +
                     d2) +
                    0.0625 *
                        ConsistentSum(-(1.0) * dudxdy, -(1.0) * dudxdz, dudydz);
                b(i, j + 1, k) =
                    Lab(x, y, z) +
                    (0.25 * ConsistentSum<ElementType>(-(1.0) * dudx, dudy,
                                                       -(1.0) * dudz) +
                     d2) +
                    0.0625 *
                        ConsistentSum(-(1.0) * dudxdy, dudxdz, -(1.0) * dudydz);
                b(i + 1, j + 1, k) =
                    Lab(x, y, z) +
                    (0.25 *
                         ConsistentSum<ElementType>(dudx, dudy, -(1.0) * dudz) +
                     d2) +
                    0.0625 *
                        ConsistentSum(dudxdy, -(1.0) * dudxdz, -(1.0) * dudydz);
                b(i, j, k + 1) =
                    Lab(x, y, z) +
                    (0.25 * ConsistentSum<ElementType>(-(1.0) * dudx,
                                                       -(1.0) * dudy, dudz) +
                     d2) +
                    0.0625 *
                        ConsistentSum(dudxdy, -(1.0) * dudxdz, -(1.0) * dudydz);
                b(i + 1, j, k + 1) =
                    Lab(x, y, z) +
                    (0.25 *
                         ConsistentSum<ElementType>(dudx, -(1.0) * dudy, dudz) +
                     d2) +
                    0.0625 *
                        ConsistentSum(-(1.0) * dudxdy, dudxdz, -(1.0) * dudydz);
                b(i, j + 1, k + 1) =
                    Lab(x, y, z) +
                    (0.25 *
                         ConsistentSum<ElementType>(-(1.0) * dudx, dudy, dudz) +
                     d2) +
                    0.0625 *
                        ConsistentSum(-(1.0) * dudxdy, -(1.0) * dudxdz, dudydz);
                b(i + 1, j + 1, k + 1) =
                    Lab(x, y, z) +
                    (0.25 * ConsistentSum<ElementType>(dudx, dudy, dudz) + d2) +
                    0.0625 * ConsistentSum(dudxdy, dudxdz, dudydz);
#else
                b(i, j, k) = Lab(x, y, z) +
                             0.25 * (-(1.0) * dudx - dudy - dudz) +
                             0.03125 * (dudx2 + dudy2 + dudz2) +
                             0.0625 * (dudxdy + dudxdz + dudydz);
                b(i + 1, j, k) = Lab(x, y, z) + 0.25 * (dudx - dudy - dudz) +
                                 0.03125 * (dudx2 + dudy2 + dudz2) +
                                 0.0625 * (-(1.0) * dudxdy - dudxdz + dudydz);
                b(i, j + 1, k) = Lab(x, y, z) +
                                 0.25 * (-(1.0) * dudx + dudy - dudz) +
                                 0.03125 * (dudx2 + dudy2 + dudz2) +
                                 0.0625 * (-(1.0) * dudxdy + dudxdz - dudydz);
                b(i + 1, j + 1, k) = Lab(x, y, z) +
                                     0.25 * (dudx + dudy - dudz) +
                                     0.03125 * (dudx2 + dudy2 + dudz2) +
                                     0.0625 * (dudxdy - dudxdz - dudydz);
                b(i, j, k + 1) = Lab(x, y, z) +
                                 0.25 * (-(1.0) * dudx - dudy + dudz) +
                                 0.03125 * (dudx2 + dudy2 + dudz2) +
                                 0.0625 * (dudxdy - dudxdz - dudydz);
                b(i + 1, j, k + 1) =
                    Lab(x, y, z) + 0.25 * (dudx - dudy + dudz) +
                    0.03125 * (dudx2 + dudy2 + dudz2) +
                    0.0625 * (-(1.0) * dudxdy + dudxdz - dudydz);
                b(i, j + 1, k + 1) =
                    Lab(x, y, z) + 0.25 * (-(1.0) * dudx + dudy + dudz) +
                    0.03125 * (dudx2 + dudy2 + dudz2) +
                    0.0625 * (-(1.0) * dudxdy - dudxdz + dudydz);
                b(i + 1, j + 1, k + 1) = Lab(x, y, z) +
                                         0.25 * (dudx + dudy + dudz) +
                                         0.03125 * (dudx2 + dudy2 + dudz2) +
                                         0.0625 * (dudxdy + dudxdz + dudydz);
#endif
              }
        }
#else

    for (int J = 0; J < 2; J++)
      for (int I = 0; I < 2; I++) {
        BlockType &b = *B[J * 2 + I];
        b.clear();

        for (int j = 0; j < ny; j += 2)
          for (int i = 0; i < nx; i += 2) {
            ElementType dudx =
                0.5 * (Lab(i / 2 + offsetX[I] + 1, j / 2 + offsetY[J]) -
                       Lab(i / 2 + offsetX[I] - 1, j / 2 + offsetY[J]));
            ElementType dudy =
                0.5 * (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J] + 1) -
                       Lab(i / 2 + offsetX[I], j / 2 + offsetY[J] - 1));

            ElementType dudx2 =
                (Lab(i / 2 + offsetX[I] + 1, j / 2 + offsetY[J]) +
                 Lab(i / 2 + offsetX[I] - 1, j / 2 + offsetY[J])) -
                2.0 * Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]);
            ElementType dudy2 =
                (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J] + 1) +
                 Lab(i / 2 + offsetX[I], j / 2 + offsetY[J] - 1)) -
                2.0 * Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]);

            ElementType dudxdy =
                0.25 * ((Lab(i / 2 + offsetX[I] + 1, j / 2 + offsetY[J] + 1) +
                         Lab(i / 2 + offsetX[I] - 1, j / 2 + offsetY[J] - 1)) -
                        (Lab(i / 2 + offsetX[I] + 1, j / 2 + offsetY[J] - 1) +
                         Lab(i / 2 + offsetX[I] - 1, j / 2 + offsetY[J] + 1)));

            b(i, j, 0) =
                (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]) +
                 (-0.25 * dudx - 0.25 * dudy)) +
                ((0.03125 * dudx2 + 0.03125 * dudy2) + 0.0625 * dudxdy);
            b(i + 1, j, 0) =
                (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]) +
                 (+0.25 * dudx - 0.25 * dudy)) +
                ((0.03125 * dudx2 + 0.03125 * dudy2) - 0.0625 * dudxdy);
            b(i, j + 1, 0) =
                (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]) +
                 (-0.25 * dudx + 0.25 * dudy)) +
                ((0.03125 * dudx2 + 0.03125 * dudy2) - 0.0625 * dudxdy);
            b(i + 1, j + 1, 0) =
                (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]) +
                 (+0.25 * dudx + 0.25 * dudy)) +
                ((0.03125 * dudx2 + 0.03125 * dudy2) + 0.0625 * dudxdy);
          }
      }
#endif
  }

  /**
   * @brief Refinement criterion.
   *
   * Default refinement criterion is to compare the 'magnitude()' of each
   * gridpoint to Rtol and Ctol. Can be overidden by a derived class, to enable
   * a custom refinement criterion.
   *
   * @param info BlockInfo to be tagged.
   */
  virtual State TagLoadedBlock(BlockInfo &info) {
    const int nx = BlockType::sizeX;
    const int ny = BlockType::sizeY;
    BlockType &b = *(BlockType *)info.ptrBlock;

    double Linf = 0.0;
#if DIMENSION == 3
    const int nz = BlockType::sizeZ;
    for (int k = 0; k < nz; k++)
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++) {
          Linf = std::max(Linf, std::fabs(b(i, j, k).magnitude()));
        }
#endif
#if DIMENSION == 2
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++) {
        Linf = std::max(Linf, std::fabs(b(i, j).magnitude()));
      }
#endif

    if (Linf > tolerance_for_refinement)
      return Refine;
    else if (Linf < tolerance_for_compression)
      return Compress;

    return Leave;
  }
};

} // namespace cubism

namespace cubism {

template <typename Lab, typename Kernel, typename TGrid,
          typename TGrid_corr = TGrid>
void compute(Kernel &&kernel, TGrid *g, TGrid_corr *g_corr = nullptr) {
  // If flux corrections are needed, prepare the flux correction object
  if (g_corr != nullptr)
    g_corr->Corrector.prepare(*g_corr);

  // Start sending and receiving of data for blocks at the boundary of each rank
  cubism::SynchronizerMPI_AMR<typename TGrid::Real, TGrid> &Synch =
      *(g->sync(kernel.stencil));

  // Access the inner blocks of each rank
  std::vector<cubism::BlockInfo *> *inner = &Synch.avail_inner();

  std::vector<cubism::BlockInfo *> *halo_next;
  bool done = false;
#pragma omp parallel
  {
    Lab lab;
    lab.prepare(*g, kernel.stencil);

// First compute for inner blocks
#pragma omp for nowait
    for (const auto &I : *inner) {
      lab.load(*I, 0);
      kernel(lab, *I);
    }

    // Then compute for boundary blocks
#if 1
    while (done == false) {
#pragma omp master
      halo_next = &Synch.avail_next();
#pragma omp barrier

#pragma omp for nowait
      for (const auto &I : *halo_next) {
        lab.load(*I, 0);
        kernel(lab, *I);
      }

#pragma omp single
      {
        if (halo_next->size() == 0)
          done = true;
      }
    }
#else
    std::vector<cubism::BlockInfo> &blk = g->getBlocksInfo();
    std::vector<bool> ready(blk.size(), false);
    std::vector<cubism::BlockInfo *> &avail1 = Synch.avail_halo_nowait();
    const int Nhalo = avail1.size();
    while (done == false) {
      done = true;
      for (int i = 0; i < Nhalo; i++) {
        const cubism::BlockInfo &I = *avail1[i];
        if (ready[I.blockID] == false) {
          if (Synch.isready(I)) {
            ready[I.blockID] = true;
            lab.load(I, 0);
            kernel(lab, I);
          } else {
            done = false;
          }
        }
      }
    }
#endif
  }

  // Complete the send requests remaining
  Synch.avail_halo();

  // Carry out flux corrections
  if (g_corr != nullptr)
    g_corr->Corrector.FillBlockCases();
}

// Get two BlockLabs from two different Grids
template <typename Kernel, typename TGrid, typename LabMPI, typename TGrid2,
          typename LabMPI2, typename TGrid_corr = TGrid>
static void compute(const Kernel &kernel, TGrid &grid, TGrid2 &grid2,
                    const bool applyFluxCorrection = false,
                    TGrid_corr *corrected_grid = nullptr) {
  if (applyFluxCorrection)
    corrected_grid->Corrector.prepare(*corrected_grid);

  SynchronizerMPI_AMR<typename TGrid::Real, TGrid> &Synch =
      *grid.sync(kernel.stencil);
  Kernel kernel2 = kernel;
  kernel2.stencil.sx = kernel2.stencil2.sx;
  kernel2.stencil.sy = kernel2.stencil2.sy;
  kernel2.stencil.sz = kernel2.stencil2.sz;
  kernel2.stencil.ex = kernel2.stencil2.ex;
  kernel2.stencil.ey = kernel2.stencil2.ey;
  kernel2.stencil.ez = kernel2.stencil2.ez;
  kernel2.stencil.tensorial = kernel2.stencil2.tensorial;
  kernel2.stencil.selcomponents.clear();
  kernel2.stencil.selcomponents = kernel2.stencil2.selcomponents;

  SynchronizerMPI_AMR<typename TGrid::Real, TGrid2> &Synch2 =
      *grid2.sync(kernel2.stencil);

  const StencilInfo &stencil = Synch.getstencil();
  const StencilInfo &stencil2 = Synch2.getstencil();

  std::vector<cubism::BlockInfo> &blk = grid.getBlocksInfo();
  std::vector<bool> ready(blk.size(), false);

  std::vector<BlockInfo *> &avail0 = Synch.avail_inner();
  std::vector<BlockInfo *> &avail02 = Synch2.avail_inner();
  const int Ninner = avail0.size();
  std::vector<cubism::BlockInfo *> avail1;
  std::vector<cubism::BlockInfo *> avail12;
// bool done = false;
#pragma omp parallel
  {
    LabMPI lab;
    LabMPI2 lab2;
    lab.prepare(grid, stencil);
    lab2.prepare(grid2, stencil2);
#pragma omp for
    for (int i = 0; i < Ninner; i++) {
      const BlockInfo &I = *avail0[i];
      const BlockInfo &I2 = *avail02[i];
      lab.load(I, 0);
      lab2.load(I2, 0);
      kernel(lab, lab2, I, I2);
      ready[I.blockID] = true;
    }

#if 1
#pragma omp master
    {
      avail1 = Synch.avail_halo();
      avail12 = Synch2.avail_halo();
    }
#pragma omp barrier

    const int Nhalo = avail1.size();

#pragma omp for
    for (int i = 0; i < Nhalo; i++) {
      const cubism::BlockInfo &I = *avail1[i];
      const cubism::BlockInfo &I2 = *avail12[i];
      lab.load(I, 0);
      lab2.load(I2, 0);
      kernel(lab, lab2, I, I2);
    }
#else

#pragma omp master
    {
      avail1 = Synch.avail_halo_nowait();
      avail12 = Synch2.avail_halo_nowait();
    }
#pragma omp barrier
    const int Nhalo = avail1.size();

    while (done == false) {
#pragma omp barrier

#pragma omp single
      done = true;

#pragma omp for
      for (int i = 0; i < Nhalo; i++) {
        const cubism::BlockInfo &I = *avail1[i];
        const cubism::BlockInfo &I2 = *avail12[i];
        if (ready[I.blockID] == false) {
          bool blockready;
#pragma omp critical
          { blockready = (Synch.isready(I) && Synch.isready(I2)); }
          if (blockready) {
            ready[I.blockID] = true;
            lab.load(I, 0);
            lab2.load(I2, 0);
            kernel(lab, lab2, I, I2);
          } else {
#pragma omp atomic write
            done = false;
          }
        }
      }
    }
    avail1 = Synch.avail_halo();
    avail12 = Synch2.avail_halo();
#endif
  }

  if (applyFluxCorrection)
    corrected_grid->Corrector.FillBlockCases();
}

/// Example of a gridpoint element that is merely a scalar quantity of type
/// 'Real' (double/float).
template <typename Real = double> struct ScalarElement {
  using RealType = Real; ///< definition of 'RealType', needed by BlockLab
  Real s = 0;            ///< scalar quantity

  /// set scalar to zero
  inline void clear() { s = 0; }

  /// set scalar to a value
  inline void set(const Real v) { s = v; }

  /// copy a ScalarElement
  inline void copy(const ScalarElement &c) { s = c.s; }

  ScalarElement &operator*=(const Real a) {
    this->s *= a;
    return *this;
  }
  ScalarElement &operator+=(const ScalarElement &rhs) {
    this->s += rhs.s;
    return *this;
  }
  ScalarElement &operator-=(const ScalarElement &rhs) {
    this->s -= rhs.s;
    return *this;
  }
  ScalarElement &operator/=(const ScalarElement &rhs) {
    this->s /= rhs.s;
    return *this;
  }
  friend ScalarElement operator*(const Real a, ScalarElement el) {
    return (el *= a);
  }
  friend ScalarElement operator+(ScalarElement lhs, const ScalarElement &rhs) {
    return (lhs += rhs);
  }
  friend ScalarElement operator-(ScalarElement lhs, const ScalarElement &rhs) {
    return (lhs -= rhs);
  }
  friend ScalarElement operator/(ScalarElement lhs, const ScalarElement &rhs) {
    return (lhs /= rhs);
  }
  bool operator<(const ScalarElement &other) const { return (s < other.s); }
  bool operator>(const ScalarElement &other) const { return (s > other.s); }
  bool operator<=(const ScalarElement &other) const { return (s <= other.s); }
  bool operator>=(const ScalarElement &other) const { return (s >= other.s); }
  Real magnitude() { return s; }
  Real &member(int i) { return s; }
  static constexpr int DIM = 1;
};

/// Example of a gridpoint element that is a vector quantity of type 'Real'
/// (double/float); 'dim' are the number of dimensions of the vector
template <int dim, typename Real = double> struct VectorElement {
  using RealType = Real; ///< definition of 'RealType', needed by BlockLab
  static constexpr int DIM = dim;
  Real u[DIM]; ///< vector quantity

  VectorElement() { clear(); }

  /// set vector components to zero
  inline void clear() {
    for (int i = 0; i < DIM; ++i)
      u[i] = 0;
  }

  /// set vector components to a number
  inline void set(const Real v) {
    for (int i = 0; i < DIM; ++i)
      u[i] = v;
  }

  /// set copy one VectorElement to another
  inline void copy(const VectorElement &c) {
    for (int i = 0; i < DIM; ++i)
      u[i] = c.u[i];
  }

  VectorElement &operator=(const VectorElement &c) = default;

  VectorElement &operator*=(const Real a) {
    for (int i = 0; i < DIM; ++i)
      this->u[i] *= a;
    return *this;
  }
  VectorElement &operator+=(const VectorElement &rhs) {
    for (int i = 0; i < DIM; ++i)
      this->u[i] += rhs.u[i];
    return *this;
  }
  VectorElement &operator-=(const VectorElement &rhs) {
    for (int i = 0; i < DIM; ++i)
      this->u[i] -= rhs.u[i];
    return *this;
  }
  VectorElement &operator/=(const VectorElement &rhs) {
    for (int i = 0; i < DIM; ++i)
      this->u[i] /= rhs.u[i];
    return *this;
  }
  friend VectorElement operator*(const Real a, VectorElement el) {
    return (el *= a);
  }
  friend VectorElement operator+(VectorElement lhs, const VectorElement &rhs) {
    return (lhs += rhs);
  }
  friend VectorElement operator-(VectorElement lhs, const VectorElement &rhs) {
    return (lhs -= rhs);
  }
  friend VectorElement operator/(VectorElement lhs, const VectorElement &rhs) {
    return (lhs /= rhs);
  }
  bool operator<(const VectorElement &other) const {
    Real s1 = 0.0;
    Real s2 = 0.0;
    for (int i = 0; i < DIM; ++i) {
      s1 += u[i] * u[i];
      s2 += other.u[i] * other.u[i];
    }

    return (s1 < s2);
  }
  bool operator>(const VectorElement &other) const {
    Real s1 = 0.0;
    Real s2 = 0.0;
    for (int i = 0; i < DIM; ++i) {
      s1 += u[i] * u[i];
      s2 += other.u[i] * other.u[i];
    }

    return (s1 > s2);
  }
  bool operator<=(const VectorElement &other) const {
    Real s1 = 0.0;
    Real s2 = 0.0;
    for (int i = 0; i < DIM; ++i) {
      s1 += u[i] * u[i];
      s2 += other.u[i] * other.u[i];
    }

    return (s1 <= s2);
  }
  bool operator>=(const VectorElement &other) const {
    Real s1 = 0.0;
    Real s2 = 0.0;
    for (int i = 0; i < DIM; ++i) {
      s1 += u[i] * u[i];
      s2 += other.u[i] * other.u[i];
    }

    return (s1 >= s2);
  }
  Real magnitude() {
    Real s1 = 0.0;
    for (int i = 0; i < DIM; ++i) {
      s1 += u[i] * u[i];
    }
    return sqrt(s1);
  }
  Real &member(int i) { return u[i]; }
};

/// array of blocksize^dim gridpoints of type 'TElement'.
template <int blocksize, int dim, typename TElement> struct GridBlock {
  // these identifiers are required by cubism!
  static constexpr int BS = blocksize;
  static constexpr int sizeX = blocksize;
  static constexpr int sizeY = blocksize;
  static constexpr int sizeZ = dim > 2 ? blocksize : 1;
  static constexpr std::array<int, 3> sizeArray = {sizeX, sizeY, sizeZ};
  using ElementType = TElement;
  using RealType = typename TElement::RealType;

  ElementType data[sizeZ][sizeY][sizeX];

  /// set 'data' to zero (call 'clear()' of each TElement)
  inline void clear() {
    ElementType *const entry = &data[0][0][0];
    for (int i = 0; i < sizeX * sizeY * sizeZ; ++i)
      entry[i].clear();
  }

  /// set 'data' to a value (call 'set()' of each TElement)
  inline void set(const RealType v) {
    ElementType *const entry = &data[0][0][0];
    for (int i = 0; i < sizeX * sizeY * sizeZ; ++i)
      entry[i].set(v);
  }

  /// copy one GridBlock to another  (call 'copy()' of each TElement)
  inline void copy(const GridBlock<blocksize, dim, ElementType> &c) {
    ElementType *const entry = &data[0][0][0];
    const ElementType *const source = &c.data[0][0][0];
    for (int i = 0; i < sizeX * sizeY * sizeZ; ++i)
      entry[i].copy(source[i]);
  }

  /// Access an element of this GridBlock (const.)
  const ElementType &operator()(int ix, int iy = 0, int iz = 0) const {
    assert(ix >= 0 && iy >= 0 && iz >= 0 && ix < sizeX && iy < sizeY &&
           iz < sizeZ);
    return data[iz][iy][ix];
  }

  /// Access an element of this GridBlock
  ElementType &operator()(int ix, int iy = 0, int iz = 0) {
    assert(ix >= 0 && iy >= 0 && iz >= 0 && ix < sizeX && iy < sizeY &&
           iz < sizeZ);
    return data[iz][iy][ix];
  }
  GridBlock(const GridBlock &) = delete;
  GridBlock &operator=(const GridBlock &) = delete;
};

/** BlockLab to apply zero Neumann boundary conditions (zero normal derivative
 * to the boundary).
 * @tparam TGrid: Grid/GridMPI type to apply the boundary conditions to.
 * @tparam dim: = 2 or 3, depending on the spatial dimensions
 * @tparam allocator: allocator object, same as the one from BlockLab.
 */
template <typename TGrid, int dim,
          template <typename X> class allocator = std::allocator>
class BlockLabNeumann : public cubism::BlockLab<TGrid, allocator> {
  /*
   * Apply 2nd order Neumann boundary condition: du/dn_{i+1/2} = 0 => u_{i} =
   * u_{i+1}
   */
  static constexpr int sizeX = TGrid::BlockType::sizeX;
  static constexpr int sizeY = TGrid::BlockType::sizeY;
  static constexpr int sizeZ = TGrid::BlockType::sizeZ;
  static constexpr int DIM = dim;

protected:
  /// Apply bc on face of direction dir and side side (0 or 1):
  template <int dir, int side> void Neumann3D(const bool coarse = false) {
    int stenBeg[3];
    int stenEnd[3];
    int bsize[3];
    if (!coarse) {
      stenEnd[0] = this->m_stencilEnd[0];
      stenEnd[1] = this->m_stencilEnd[1];
      stenEnd[2] = this->m_stencilEnd[2];
      stenBeg[0] = this->m_stencilStart[0];
      stenBeg[1] = this->m_stencilStart[1];
      stenBeg[2] = this->m_stencilStart[2];
      bsize[0] = sizeX;
      bsize[1] = sizeY;
      bsize[2] = sizeZ;
    } else {
      stenEnd[0] =
          (this->m_stencilEnd[0]) / 2 + 1 + this->m_InterpStencilEnd[0] - 1;
      stenEnd[1] =
          (this->m_stencilEnd[1]) / 2 + 1 + this->m_InterpStencilEnd[1] - 1;
      stenEnd[2] =
          (this->m_stencilEnd[2]) / 2 + 1 + this->m_InterpStencilEnd[2] - 1;
      stenBeg[0] =
          (this->m_stencilStart[0] - 1) / 2 + this->m_InterpStencilStart[0];
      stenBeg[1] =
          (this->m_stencilStart[1] - 1) / 2 + this->m_InterpStencilStart[1];
      stenBeg[2] =
          (this->m_stencilStart[2] - 1) / 2 + this->m_InterpStencilStart[2];
      bsize[0] = sizeX / 2;
      bsize[1] = sizeY / 2;
      bsize[2] = sizeZ / 2;
    }

    auto *const cb = coarse ? this->m_CoarsenedBlock : this->m_cacheBlock;

    int s[3];
    int e[3];
    s[0] = dir == 0 ? (side == 0 ? stenBeg[0] : bsize[0]) : 0;
    s[1] = dir == 1 ? (side == 0 ? stenBeg[1] : bsize[1]) : 0;
    s[2] = dir == 2 ? (side == 0 ? stenBeg[2] : bsize[2]) : 0;
    e[0] = dir == 0 ? (side == 0 ? 0 : bsize[0] + stenEnd[0] - 1) : bsize[0];
    e[1] = dir == 1 ? (side == 0 ? 0 : bsize[1] + stenEnd[1] - 1) : bsize[1];
    e[2] = dir == 2 ? (side == 0 ? 0 : bsize[2] + stenEnd[2] - 1) : bsize[2];

    // Fill face
    for (int iz = s[2]; iz < e[2]; iz++)
      for (int iy = s[1]; iy < e[1]; iy++)
        for (int ix = s[0]; ix < e[0]; ix++) {
          cb->Access(ix - stenBeg[0], iy - stenBeg[1], iz - stenBeg[2]) =
              cb->Access(
                  (dir == 0 ? (side == 0 ? 0 : bsize[0] - 1) : ix) - stenBeg[0],
                  (dir == 1 ? (side == 0 ? 0 : bsize[1] - 1) : iy) - stenBeg[1],
                  (dir == 2 ? (side == 0 ? 0 : bsize[2] - 1) : iz) -
                      stenBeg[2]);
        }

    // Fill edges and corners (necessary for the coarse block)
    s[dir] = stenBeg[dir] * (1 - side) + bsize[dir] * side;
    e[dir] = (bsize[dir] - 1 + stenEnd[dir]) * side;
    const int d1 = (dir + 1) % 3;
    const int d2 = (dir + 2) % 3;
    for (int b = 0; b < 2; ++b)
      for (int a = 0; a < 2; ++a) {
        s[d1] = stenBeg[d1] + a * b * (bsize[d1] - stenBeg[d1]);
        s[d2] = stenBeg[d2] + (a - a * b) * (bsize[d2] - stenBeg[d2]);
        e[d1] = (1 - b + a * b) * (bsize[d1] - 1 + stenEnd[d1]);
        e[d2] = (a + b - a * b) * (bsize[d2] - 1 + stenEnd[d2]);
        for (int iz = s[2]; iz < e[2]; iz++)
          for (int iy = s[1]; iy < e[1]; iy++)
            for (int ix = s[0]; ix < e[0]; ix++) {
              cb->Access(ix - stenBeg[0], iy - stenBeg[1], iz - stenBeg[2]) =
                  dir == 0
                      ? cb->Access(side * (bsize[0] - 1) - stenBeg[0],
                                   iy - stenBeg[1], iz - stenBeg[2])
                      : (dir == 1
                             ? cb->Access(ix - stenBeg[0],
                                          side * (bsize[1] - 1) - stenBeg[1],
                                          iz - stenBeg[2])
                             : cb->Access(ix - stenBeg[0], iy - stenBeg[1],
                                          side * (bsize[2] - 1) - stenBeg[2]));
            }
      }
  }

  /// Apply bc on face of direction dir and side side (0 or 1):
  template <int dir, int side> void Neumann2D(const bool coarse = false) {
    int stenBeg[2];
    int stenEnd[2];
    int bsize[2];
    if (!coarse) {
      stenEnd[0] = this->m_stencilEnd[0];
      stenEnd[1] = this->m_stencilEnd[1];
      stenBeg[0] = this->m_stencilStart[0];
      stenBeg[1] = this->m_stencilStart[1];
      bsize[0] = sizeX;
      bsize[1] = sizeY;
    } else {
      stenEnd[0] =
          (this->m_stencilEnd[0]) / 2 + 1 + this->m_InterpStencilEnd[0] - 1;
      stenEnd[1] =
          (this->m_stencilEnd[1]) / 2 + 1 + this->m_InterpStencilEnd[1] - 1;
      stenBeg[0] =
          (this->m_stencilStart[0] - 1) / 2 + this->m_InterpStencilStart[0];
      stenBeg[1] =
          (this->m_stencilStart[1] - 1) / 2 + this->m_InterpStencilStart[1];
      bsize[0] = sizeX / 2;
      bsize[1] = sizeY / 2;
    }

    auto *const cb = coarse ? this->m_CoarsenedBlock : this->m_cacheBlock;

    int s[2];
    int e[2];
    s[0] = dir == 0 ? (side == 0 ? stenBeg[0] : bsize[0]) : stenBeg[0];
    s[1] = dir == 1 ? (side == 0 ? stenBeg[1] : bsize[1]) : stenBeg[1];
    e[0] = dir == 0 ? (side == 0 ? 0 : bsize[0] + stenEnd[0] - 1)
                    : bsize[0] + stenEnd[0] - 1;
    e[1] = dir == 1 ? (side == 0 ? 0 : bsize[1] + stenEnd[1] - 1)
                    : bsize[1] + stenEnd[1] - 1;

    for (int iy = s[1]; iy < e[1]; iy++)
      for (int ix = s[0]; ix < e[0]; ix++)
        cb->Access(ix - stenBeg[0], iy - stenBeg[1], 0) = cb->Access(
            (dir == 0 ? (side == 0 ? 0 : bsize[0] - 1) : ix) - stenBeg[0],
            (dir == 1 ? (side == 0 ? 0 : bsize[1] - 1) : iy) - stenBeg[1], 0);
  }

public:
  typedef typename TGrid::BlockType::ElementType ElementTypeBlock;
  typedef typename TGrid::BlockType::ElementType ElementType;
  using Real = typename ElementType::RealType; ///< Number type used by Element
                                               ///< (double/float etc.)
  /// Will return 'false' as the boundary conditions are not periodic for this
  /// BlockLab.
  virtual bool is_xperiodic() override { return false; }
  /// Will return 'false' as the boundary conditions are not periodic for this
  /// BlockLab.
  virtual bool is_yperiodic() override { return false; }
  /// Will return 'false' as the boundary conditions are not periodic for this
  /// BlockLab.
  virtual bool is_zperiodic() override { return false; }

  BlockLabNeumann() = default;
  BlockLabNeumann(const BlockLabNeumann &) = delete;
  BlockLabNeumann &operator=(const BlockLabNeumann &) = delete;

  /// Apply the boundary condition; 'coarse' is set to true if the boundary
  /// condition should be applied to the coarsened version of the BlockLab (see
  /// also BlockLab).
  void _apply_bc(const cubism::BlockInfo &info, const Real t = 0,
                 const bool coarse = false) override {
    if (DIM == 2) {
      if (info.index[0] == 0)
        this->template Neumann2D<0, 0>(coarse);
      if (info.index[0] == this->NX - 1)
        this->template Neumann2D<0, 1>(coarse);
      if (info.index[1] == 0)
        this->template Neumann2D<1, 0>(coarse);
      if (info.index[1] == this->NY - 1)
        this->template Neumann2D<1, 1>(coarse);
    } else if (DIM == 3) {
      if (info.index[0] == 0)
        this->template Neumann3D<0, 0>(coarse);
      if (info.index[0] == this->NX - 1)
        this->template Neumann3D<0, 1>(coarse);
      if (info.index[1] == 0)
        this->template Neumann3D<1, 0>(coarse);
      if (info.index[1] == this->NY - 1)
        this->template Neumann3D<1, 1>(coarse);
      if (info.index[2] == 0)
        this->template Neumann3D<2, 0>(coarse);
      if (info.index[2] == this->NZ - 1)
        this->template Neumann3D<2, 1>(coarse);
    }
  }
};

} // namespace cubism

enum BCflag { freespace, periodic, wall };
inline BCflag string2BCflag(const std::string &strFlag) {
  if (strFlag == "periodic") {
    // printf("[CUP2D] Using periodic boundary conditions\n");
    return periodic;
  } else if (strFlag == "freespace") {
    // printf("[CUP2D] Using freespace boundary conditions\n");
    return freespace;
  } else if (strFlag == "wall") {
    // printf("[CUP2D] Using freespace boundary conditions\n");
    return wall;
  } else {
    fprintf(stderr, "BC not recognized %s\n", strFlag.c_str());
    fflush(0);
    abort();
    return periodic; // dummy
  }
}
// CAREFUL THESE ARE GLOBAL VARIABLES!
extern BCflag cubismBCX;
extern BCflag cubismBCY;

template <typename TGrid,
          template <typename X> class allocator = std::allocator>
class BlockLabDirichlet : public cubism::BlockLab<TGrid, allocator> {
public:
  using ElementType = typename TGrid::BlockType::ElementType;
  static constexpr int sizeX = TGrid::BlockType::sizeX;
  static constexpr int sizeY = TGrid::BlockType::sizeY;
  static constexpr int sizeZ = TGrid::BlockType::sizeZ;

  virtual bool is_xperiodic() override { return cubismBCX == periodic; }
  virtual bool is_yperiodic() override { return cubismBCY == periodic; }
  virtual bool is_zperiodic() override { return false; }

  // Apply bc on face of direction dir and side side (0 or 1):
  template <int dir, int side>
  void applyBCface(bool wall, bool coarse = false) {

    const int A = 1 - dir;
    if (!coarse) {
      auto *const cb = this->m_cacheBlock;
      int s[3] = {0, 0, 0}, e[3] = {0, 0, 0};
      const int *const stenBeg = this->m_stencilStart;
      const int *const stenEnd = this->m_stencilEnd;
      s[0] = dir == 0 ? (side == 0 ? stenBeg[0] : sizeX) : stenBeg[0];
      s[1] = dir == 1 ? (side == 0 ? stenBeg[1] : sizeY) : stenBeg[1];
      e[0] = dir == 0 ? (side == 0 ? 0 : sizeX + stenEnd[0] - 1)
                      : sizeX + stenEnd[0] - 1;
      e[1] = dir == 1 ? (side == 0 ? 0 : sizeY + stenEnd[1] - 1)
                      : sizeY + stenEnd[1] - 1;

      if (!wall)
        for (int iy = s[1]; iy < e[1]; iy++)
          for (int ix = s[0]; ix < e[0]; ix++) {
            const int x =
                (dir == 0 ? (side == 0 ? 0 : sizeX - 1) : ix) - stenBeg[0];
            const int y =
                (dir == 1 ? (side == 0 ? 0 : sizeY - 1) : iy) - stenBeg[1];
            cb->Access(ix - stenBeg[0], iy - stenBeg[1], 0).member(1 - A) =
                (-1.0) * cb->Access(x, y, 0).member(1 - A);
            cb->Access(ix - stenBeg[0], iy - stenBeg[1], 0).member(A) =
                cb->Access(x, y, 0).member(A);
          }
      else
        for (int iy = s[1]; iy < e[1]; iy++)
          for (int ix = s[0]; ix < e[0]; ix++) {
            const int x =
                (dir == 0 ? (side == 0 ? 0 : sizeX - 1) : ix) - stenBeg[0];
            const int y =
                (dir == 1 ? (side == 0 ? 0 : sizeY - 1) : iy) - stenBeg[1];
            cb->Access(ix - stenBeg[0], iy - stenBeg[1], 0) =
                (-1.0) * cb->Access(x, y, 0);
          }
    } else {
      auto *const cb = this->m_CoarsenedBlock;

      const int eI[3] = {
          (this->m_stencilEnd[0]) / 2 + 1 + this->m_InterpStencilEnd[0] - 1,
          (this->m_stencilEnd[1]) / 2 + 1 + this->m_InterpStencilEnd[1] - 1,
          (this->m_stencilEnd[2]) / 2 + 1 + this->m_InterpStencilEnd[2] - 1};
      const int sI[3] = {
          (this->m_stencilStart[0] - 1) / 2 + this->m_InterpStencilStart[0],
          (this->m_stencilStart[1] - 1) / 2 + this->m_InterpStencilStart[1],
          (this->m_stencilStart[2] - 1) / 2 + this->m_InterpStencilStart[2]};

      const int *const stenBeg = sI;
      const int *const stenEnd = eI;

      int s[3] = {0, 0, 0}, e[3] = {0, 0, 0};

      s[0] = dir == 0 ? (side == 0 ? stenBeg[0] : sizeX / 2) : stenBeg[0];
      s[1] = dir == 1 ? (side == 0 ? stenBeg[1] : sizeY / 2) : stenBeg[1];

      e[0] = dir == 0 ? (side == 0 ? 0 : sizeX / 2 + stenEnd[0] - 1)
                      : sizeX / 2 + stenEnd[0] - 1;
      e[1] = dir == 1 ? (side == 0 ? 0 : sizeY / 2 + stenEnd[1] - 1)
                      : sizeY / 2 + stenEnd[1] - 1;

      if (!wall)
        for (int iy = s[1]; iy < e[1]; iy++)
          for (int ix = s[0]; ix < e[0]; ix++) {
            const int x =
                (dir == 0 ? (side == 0 ? 0 : sizeX / 2 - 1) : ix) - stenBeg[0];
            const int y =
                (dir == 1 ? (side == 0 ? 0 : sizeY / 2 - 1) : iy) - stenBeg[1];
            cb->Access(ix - stenBeg[0], iy - stenBeg[1], 0).member(1 - A) =
                (-1.0) * cb->Access(x, y, 0).member(1 - A);
            cb->Access(ix - stenBeg[0], iy - stenBeg[1], 0).member(A) =
                cb->Access(x, y, 0).member(A);
          }
      else
        for (int iy = s[1]; iy < e[1]; iy++)
          for (int ix = s[0]; ix < e[0]; ix++) {
            const int x =
                (dir == 0 ? (side == 0 ? 0 : sizeX / 2 - 1) : ix) - stenBeg[0];
            const int y =
                (dir == 1 ? (side == 0 ? 0 : sizeY / 2 - 1) : iy) - stenBeg[1];
            cb->Access(ix - stenBeg[0], iy - stenBeg[1], 0) =
                (-1.0) * cb->Access(x, y, 0);
          }
    }
  }

  // Called by Cubism:
  void _apply_bc(const cubism::BlockInfo &info, const Real t = 0,
                 const bool coarse = false) override {
    const BCflag BCX = cubismBCX;
    const BCflag BCY = cubismBCY;
    if (!coarse) {
      if (is_xperiodic() == false) {
        if (info.index[0] == 0)
          this->template applyBCface<0, 0>(BCX == wall);
        if (info.index[0] == this->NX - 1)
          this->template applyBCface<0, 1>(BCX == wall);
      }
      if (is_yperiodic() == false) {
        if (info.index[1] == 0)
          this->template applyBCface<1, 0>(BCY == wall);
        if (info.index[1] == this->NY - 1)
          this->template applyBCface<1, 1>(BCY == wall);
      }
    } else {
      if (is_xperiodic() == false) {
        if (info.index[0] == 0)
          this->template applyBCface<0, 0>(BCX == wall, coarse);
        if (info.index[0] == this->NX - 1)
          this->template applyBCface<0, 1>(BCX == wall, coarse);
      }
      if (is_yperiodic() == false) {
        if (info.index[1] == 0)
          this->template applyBCface<1, 0>(BCY == wall, coarse);
        if (info.index[1] == this->NY - 1)
          this->template applyBCface<1, 1>(BCY == wall, coarse);
      }
    }
  }

  BlockLabDirichlet() : cubism::BlockLab<TGrid, allocator>() {}
  BlockLabDirichlet(const BlockLabDirichlet &) = delete;
  BlockLabDirichlet &operator=(const BlockLabDirichlet &) = delete;
};

template <typename TGrid,
          template <typename X> class allocator = std::allocator>
class BlockLabNeumann : public cubism::BlockLabNeumann<TGrid, 2, allocator> {
public:
  using cubismLab = cubism::BlockLabNeumann<TGrid, 2, allocator>;
  virtual bool is_xperiodic() override { return cubismBCX == periodic; }
  virtual bool is_yperiodic() override { return cubismBCY == periodic; }
  virtual bool is_zperiodic() override { return false; }

  // Called by Cubism:
  void _apply_bc(const cubism::BlockInfo &info, const Real t = 0,
                 const bool coarse = false) override {
    if (is_xperiodic() == false) {
      if (info.index[0] == 0)
        cubismLab::template Neumann2D<0, 0>(coarse);
      if (info.index[0] == this->NX - 1)
        cubismLab::template Neumann2D<0, 1>(coarse);
    }
    if (is_yperiodic() == false) {
      if (info.index[1] == 0)
        cubismLab::template Neumann2D<1, 0>(coarse);
      if (info.index[1] == this->NY - 1)
        cubismLab::template Neumann2D<1, 1>(coarse);
    }
  }
};

using ScalarElement = cubism::ScalarElement<Real>;
using VectorElement = cubism::VectorElement<2, Real>;
using ScalarBlock = cubism::GridBlock<_BS_, 2, ScalarElement>;
using VectorBlock = cubism::GridBlock<_BS_, 2, VectorElement>;
using ScalarGrid = cubism::GridMPI<cubism::Grid<ScalarBlock, std::allocator>>;
using VectorGrid = cubism::GridMPI<cubism::Grid<VectorBlock, std::allocator>>;

using VectorLab =
    cubism::BlockLabMPI<BlockLabDirichlet<VectorGrid, std::allocator>>;
using ScalarLab =
    cubism::BlockLabMPI<BlockLabNeumann<ScalarGrid, std::allocator>>;
using ScalarAMR = cubism::MeshAdaptation<ScalarLab>;
using VectorAMR = cubism::MeshAdaptation<VectorLab>;

namespace cubism {

const bool bVerboseProfiling = false;

class ProfileAgent {
  //	typedef tbb::tick_count ClockTime;
  typedef timeval ClockTime;

  enum ProfileAgentState {
    ProfileAgentState_Created,
    ProfileAgentState_Started,
    ProfileAgentState_Stopped
  };

  ClockTime m_tStart, m_tEnd;
  ProfileAgentState m_state;
  long double m_dAccumulatedTime;
  int m_nMeasurements;
  int m_nMoney;

  static void _getTime(ClockTime &time) {
    // time = tick_count::now();
    gettimeofday(&time, NULL);
  }

  static double _getElapsedTime(const ClockTime &tS, const ClockTime &tE) {
    return (tE.tv_sec - tS.tv_sec) + 1e-6 * (tE.tv_usec - tS.tv_usec);
    // return (tE - tS).seconds();
  }

  void _reset() {
    m_tStart = ClockTime();
    m_tEnd = ClockTime();
    m_dAccumulatedTime = 0;
    m_nMeasurements = 0;
    m_nMoney = 0;
    m_state = ProfileAgentState_Created;
  }

public:
  ProfileAgent()
      : m_tStart(), m_tEnd(), m_state(ProfileAgentState_Created),
        m_dAccumulatedTime(0), m_nMeasurements(0), m_nMoney(0) {}

  void start() {
    assert(m_state == ProfileAgentState_Created ||
           m_state == ProfileAgentState_Stopped);

    if (bVerboseProfiling) {
      printf("start\n");
    }

    _getTime(m_tStart);

    m_state = ProfileAgentState_Started;
  }

  void stop(int nMoney = 0) {
    assert(m_state == ProfileAgentState_Started);

    if (bVerboseProfiling) {
      printf("stop\n");
    }

    _getTime(m_tEnd);
    m_dAccumulatedTime += _getElapsedTime(m_tStart, m_tEnd);
    m_nMeasurements++;
    m_nMoney += nMoney;
    m_state = ProfileAgentState_Stopped;
  }

  friend class Profiler;
};

struct ProfileSummaryItem {
  std::string sName;
  double dTime;
  int nMoney;
  int nSamples;
  double dAverageTime;

  ProfileSummaryItem(std::string sName_, double dTime_, int nMoney_,
                     int nSamples_)
      : sName(sName_), dTime(dTime_), nMoney(nMoney_), nSamples(nSamples_),
        dAverageTime(dTime_ / nSamples_) {}
};

class Profiler {
protected:
  std::map<std::string, ProfileAgent *> m_mapAgents;
  std::stack<std::string> m_mapStoppedAgents;

public:
  void push_start(std::string sAgentName) {
    if (m_mapStoppedAgents.size() > 0)
      getAgent(m_mapStoppedAgents.top()).stop();

    m_mapStoppedAgents.push(sAgentName);
    getAgent(sAgentName).start();
  }

  void pop_stop() {
    std::string sCurrentAgentName = m_mapStoppedAgents.top();
    getAgent(sCurrentAgentName).stop();
    m_mapStoppedAgents.pop();

    if (m_mapStoppedAgents.size() == 0)
      return;

    getAgent(m_mapStoppedAgents.top()).start();
  }

  void clear() {
    for (std::map<std::string, ProfileAgent *>::iterator it =
             m_mapAgents.begin();
         it != m_mapAgents.end(); it++) {
      delete it->second;

      it->second = NULL;
    }

    m_mapAgents.clear();
  }

  Profiler() : m_mapAgents() {}

  ~Profiler() { clear(); }

  void printSummary(FILE *outFile = NULL) const {
    std::vector<ProfileSummaryItem> v = createSummary();

    double dTotalTime = 0;
    double dTotalTime2 = 0;
    for (std::vector<ProfileSummaryItem>::const_iterator it = v.begin();
         it != v.end(); it++)
      dTotalTime += it->dTime;

    for (std::vector<ProfileSummaryItem>::const_iterator it = v.begin();
         it != v.end(); it++)
      dTotalTime2 += it->dTime - it->nSamples * 1.30e-6;

    for (std::vector<ProfileSummaryItem>::const_iterator it = v.begin();
         it != v.end(); it++) {
      const ProfileSummaryItem &item = *it;
      const double avgTime = item.dAverageTime;

      printf("[%15s]: \t%02.0f-%02.0f%%\t%03.3e (%03.3e) s\t%03.3f (%03.3f) "
             "s\t(%d samples)\n",
             item.sName.data(), 100 * item.dTime / dTotalTime,
             100 * (item.dTime - item.nSamples * 1.3e-6) / dTotalTime2, avgTime,
             avgTime - 1.30e-6, item.dTime,
             item.dTime - item.nSamples * 1.30e-6, item.nSamples);
      if (outFile)
        fprintf(outFile, "[%15s]: \t%02.2f%%\t%03.3f s\t(%d samples)\n",

                item.sName.data(), 100 * item.dTime / dTotalTime, avgTime,
                item.nSamples);
    }

    printf("[Total time]: \t%f\n", dTotalTime);
    if (outFile)
      fprintf(outFile, "[Total time]: \t%f\n", dTotalTime);
    if (outFile)
      fflush(outFile);
    if (outFile)
      fclose(outFile);
  }

  std::vector<ProfileSummaryItem>
  createSummary(bool bSkipIrrelevantEntries = true) const {
    std::vector<ProfileSummaryItem> result;
    result.reserve(m_mapAgents.size());

    for (std::map<std::string, ProfileAgent *>::const_iterator it =
             m_mapAgents.begin();
         it != m_mapAgents.end(); it++) {
      const ProfileAgent &agent = *it->second;
      if (!bSkipIrrelevantEntries || agent.m_dAccumulatedTime > 1e-3)
        result.push_back(ProfileSummaryItem(it->first, agent.m_dAccumulatedTime,
                                            agent.m_nMoney,
                                            agent.m_nMeasurements));
    }

    return result;
  }

  void reset() {
    // printf("reset\n");
    for (std::map<std::string, ProfileAgent *>::const_iterator it =
             m_mapAgents.begin();
         it != m_mapAgents.end(); it++)
      it->second->_reset();
  }

  ProfileAgent &getAgent(std::string sName) {
    if (bVerboseProfiling) {
      printf("%s ", sName.data());
    }

    std::map<std::string, ProfileAgent *>::const_iterator it =
        m_mapAgents.find(sName);

    const bool bFound = it != m_mapAgents.end();

    if (bFound)
      return *it->second;

    ProfileAgent *agent = new ProfileAgent();

    m_mapAgents[sName] = agent;

    return *agent;
  }

  friend class ProfileAgent;
};

} // namespace cubism

class Shape;

struct SimulationData {
  // MPI
  MPI_Comm comm;
  int rank;

  /* parsed parameters */
  /*********************/

  bool bRestart;

  // blocks per dimension
  int bpdx;
  int bpdy;

  // number of levels
  int levelMax;

  // initial level
  int levelStart;

  // refinement/compression tolerance for voriticy magnitude
  Real Rtol;
  Real Ctol;

  // boolean to switch between vorticity magnitude and Q-criterion refinement
  // the Q-criterion measures the difference between rotation rate and shear
  // rate Q > 0 indicates that there's a vortex Q < 0 indicates a region where
  // viscous forces are stronger if Qcriterion=true, refinement will be done
  // where Q>Rtol (Rtol>0) Generally this results in less refinement, compared
  // to refining according to vorticity magnitude. For the cases this has been
  // tested with there was no loss of accuracy, despite the fact that the mesh
  // that was refined according to Q had about 1/4 of the points the other mesh
  // had.
  bool Qcriterion{false};

  // check for mesh refinement every this many steps
  int AdaptSteps{20};

  // boolean to switch between refinement according to chi or grad(chi)
  bool bAdaptChiGradient;

  // maximal simulation extent (direction with max(bpd))
  Real extent;

  // simulation extents
  std::array<Real, 2> extents;

  // timestep / cfl condition
  Real dt;
  Real CFL;
  int rampup{0};

  // simulation ending parameters
  int nsteps;
  Real endTime;

  // penalisation coefficient
  Real lambda;

  // constant for explicit penalisation lambda=dlm/dt
  Real dlm;

  // kinematic viscosity
  Real nu;

  // forcing
  bool bForcing;
  Real forcingWavenumber;
  Real forcingCoefficient;

  // Smagorinsky Model
  Real smagorinskyCoeff;

  // Flag for initial conditions
  std::string ic;

  // poisson solver parameters
  std::string poissonSolver; // for now only "iterative"
  Real PoissonTol;           // absolute error tolerance
  Real PoissonTolRel;        // relative error tolerance
  int maxPoissonRestarts;    // maximal number of restarts of Poisson solver
  int maxPoissonIterations;  // maximal number of iterations of Poisson solver
  int bMeanConstraint;       // regularizing the poisson equation using the mean

  // output setting
  int profilerFreq = 0;
  int dumpFreq;
  Real dumpTime;
  bool verbose;
  bool muteAll;
  std::string path4serialization;
  std::string path2file;

  /*********************/

  // initialize profiler
  cubism::Profiler *profiler = new cubism::Profiler();

  // declare grids
  ScalarGrid *chi = nullptr;
  VectorGrid *vel = nullptr;
  VectorGrid *vOld = nullptr;
  ScalarGrid *pres = nullptr;
  VectorGrid *tmpV = nullptr;
  ScalarGrid *tmp = nullptr;
  ScalarGrid *pold = nullptr;
  ScalarGrid *Cs = nullptr;

  // vector containing obstacles
  std::vector<std::shared_ptr<Shape>> shapes;

  // simulation time
  Real time = 0;

  // simulation step
  int step = 0;

  // velocity of simulation frame of reference
  Real uinfx = 0;
  Real uinfy = 0;
  Real uinfx_old = 0;
  Real uinfy_old = 0;
  Real dt_old =
      1e10; // need to initialize to a big value so that restarting does not
  Real dt_old2 = 1e10; // break when these are used in PressureSingle.cpp

  // largest velocity measured
  Real uMax_measured = 0;

  // time of next dump
  Real nextDumpTime = 0;

  // bools specifying whether we dump or not
  bool _bDump = false;
  bool DumpUniform = false;
  bool bDumpCs = false;

  // bool for detecting collisions
  bool bCollision = false;
  std::vector<int> bCollisionID;

  void addShape(std::shared_ptr<Shape> shape);

  void allocateGrid();
  void resetAll();
  bool bDump();
  void registerDump();
  bool bOver() const;

  // minimal and maximal gridspacing possible
  Real minH;
  Real maxH;

  SimulationData();
  SimulationData(const SimulationData &) = delete;
  SimulationData(SimulationData &&) = delete;
  SimulationData &operator=(const SimulationData &) = delete;
  SimulationData &operator=(SimulationData &&) = delete;
  ~SimulationData();

  // minimal gridspacing present on grid
  Real getH() {
    Real minHGrid = std::numeric_limits<Real>::infinity();
    auto &infos = vel->getBlocksInfo();
    for (size_t i = 0; i < infos.size(); i++) {
      minHGrid = std::min((Real)infos[i].h, minHGrid);
    }
    MPI_Allreduce(MPI_IN_PLACE, &minHGrid, 1, MPI_Real, MPI_MIN, comm);
    return minHGrid;
  }

  void startProfiler(std::string name);
  void stopProfiler();
  void printResetProfiler();

  void writeRestartFiles();
  void readRestartFiles();

  void dumpChi(std::string name);
  void dumpPres(std::string name);
  void dumpTmp(std::string name);
  void dumpVel(std::string name);
  void dumpUdef(std::string name);
  void dumpVold(std::string name);
  void dumpPold(std::string name);
  void dumpTmpV(std::string name);
  void dumpCs(std::string name);
  void dumpAll(std::string name);
};

class Operator {
public:
  SimulationData &sim;
  Operator(SimulationData &s) : sim(s) {}
  virtual ~Operator() {}
  virtual void operator()(const Real dt) = 0;
  virtual std::string getName() = 0;
};
//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

using CHI_MAT = Real[_BS_][_BS_];
using UDEFMAT = Real[_BS_][_BS_][2];

struct surface_data {
  const int ix, iy;
  const Real dchidx, dchidy, delta;

  surface_data(const int _ix, const int _iy, const Real Xdx, const Real Xdy,
               const Real D)
      : ix(_ix), iy(_iy), dchidx(Xdx), dchidy(Xdy), delta(D) {}
};

struct ObstacleBlock {
  static const int sizeX = _BS_;
  static const int sizeY = _BS_;

  // bulk quantities:
  Real chi[sizeY][sizeX];
  Real dist[sizeY][sizeX];
  Real udef[sizeY][sizeX][2];

  // surface quantities:
  size_t n_surfPoints = 0;
  bool filled = false;
  std::vector<surface_data *> surface;

  // surface quantities of interest (only needed for post-processing
  // computations)
  Real *x_s = nullptr;     // x-coordinate
  Real *y_s = nullptr;     // y-coordinate
  Real *p_s = nullptr;     // pressure
  Real *u_s = nullptr;     // u velocity
  Real *v_s = nullptr;     // v velocity
  Real *nx_s = nullptr;    // x-component of unit normal vector
  Real *ny_s = nullptr;    // y-component of unit normal vector
  Real *omega_s = nullptr; // vorticity
  Real *uDef_s = nullptr;  // x-component of deformation velocity
  Real *vDef_s = nullptr;  // y-component of deformation velocity
  Real *fX_s = nullptr;    // x-component of total force
  Real *fY_s = nullptr;    // y-component of total force
  Real *fXv_s = nullptr;   // x-component of viscous force
  Real *fYv_s = nullptr;   // y-component of viscous force

  // additive quantities:
  Real perimeter = 0, forcex = 0, forcey = 0, forcex_P = 0, forcey_P = 0;
  Real forcex_V = 0, forcey_V = 0, torque = 0, torque_P = 0, torque_V = 0;
  Real drag = 0, thrust = 0, lift = 0, Pout = 0, PoutNew = 0, PoutBnd = 0,
       defPower = 0, defPowerBnd = 0;
  Real circulation = 0;

  // auxiliary quantities for shape center of mass
  Real COM_x = 0;
  Real COM_y = 0;
  Real Mass = 0;

  ObstacleBlock() {
    clear();
    // rough estimate of surface cutting the block diagonally
    // with 2 points needed on each side of surface
    surface.reserve(4 * _BS_);
  }
  ~ObstacleBlock() { clear_surface(); }

  void clear_surface() {
    filled = false;
    n_surfPoints = 0;
    perimeter = forcex = forcey = forcex_P = forcey_P = 0;
    forcex_V = forcey_V = torque = torque_P = torque_V = drag = thrust = lift =
        0;
    Pout = PoutBnd = defPower = defPowerBnd = circulation = 0;

    for (auto &trash : surface) {
      if (trash == nullptr)
        continue;
      delete trash;
      trash = nullptr;
    }
    surface.clear();

    if (x_s not_eq nullptr) {
      free(x_s);
      x_s = nullptr;
    }
    if (y_s not_eq nullptr) {
      free(y_s);
      y_s = nullptr;
    }
    if (p_s not_eq nullptr) {
      free(p_s);
      p_s = nullptr;
    }
    if (u_s not_eq nullptr) {
      free(u_s);
      u_s = nullptr;
    }
    if (v_s not_eq nullptr) {
      free(v_s);
      v_s = nullptr;
    }
    if (nx_s not_eq nullptr) {
      free(nx_s);
      nx_s = nullptr;
    }
    if (ny_s not_eq nullptr) {
      free(ny_s);
      ny_s = nullptr;
    }
    if (omega_s not_eq nullptr) {
      free(omega_s);
      omega_s = nullptr;
    }
    if (uDef_s not_eq nullptr) {
      free(uDef_s);
      uDef_s = nullptr;
    }
    if (vDef_s not_eq nullptr) {
      free(vDef_s);
      vDef_s = nullptr;
    }
    if (fX_s not_eq nullptr) {
      free(fX_s);
      fX_s = nullptr;
    }
    if (fY_s not_eq nullptr) {
      free(fY_s);
      fY_s = nullptr;
    }
    if (fXv_s not_eq nullptr) {
      free(fXv_s);
      fXv_s = nullptr;
    }
    if (fYv_s not_eq nullptr) {
      free(fYv_s);
      fYv_s = nullptr;
    }
  }

  void clear() {
    clear_surface();
    std::fill(dist[0], dist[0] + sizeX * sizeY, -1);
    std::fill(chi[0], chi[0] + sizeX * sizeY, 0);
    memset(udef, 0, sizeof(Real) * sizeX * sizeY * 2);
  }

  void write(const int ix, const int iy, const Real delta, const Real gradUX,
             const Real gradUY) {
    assert(!filled);

    if (delta > 0) {
      n_surfPoints++;
      // multiply by cell area h^2 and by 0.5/h due to finite diff of gradHX
      const Real dchidx = -delta * gradUX, dchidy = -delta * gradUY;
      surface.push_back(new surface_data(ix, iy, dchidx, dchidy, delta));
    }
  }

  void allocate_surface() {
    filled = true;
    assert(surface.size() == n_surfPoints);
    x_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    y_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    p_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    u_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    v_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    nx_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    ny_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    omega_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    uDef_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    vDef_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    fX_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    fY_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    fXv_s = (Real *)calloc(n_surfPoints, sizeof(Real));
    fYv_s = (Real *)calloc(n_surfPoints, sizeof(Real));
  }

  void fill_stringstream(std::stringstream &s) {
    for (size_t i = 0; i < n_surfPoints; i++)
      s << x_s[i] << ", " << y_s[i] << ", " << p_s[i] << ", " << u_s[i] << ", "
        << v_s[i] << ", " << nx_s[i] << ", " << ny_s[i] << ", " << omega_s[i]
        << ", " << uDef_s[i] << ", " << vDef_s[i] << ", " << fX_s[i] << ", "
        << fY_s[i] << ", " << fXv_s[i] << ", " << fYv_s[i] << "\n";
  }
};

class Shape {
public: // data fields
  SimulationData &sim;
  unsigned obstacleID = 0;
  std::vector<ObstacleBlock *> obstacleBlocks;
  // general quantities
  const Real origC[2], origAng;
  Real center[2]; // for single density, this corresponds to centerOfMass
  Real centerOfMass[2];
  Real d_gm[2] = {0, 0}; // distance of center of geometry to center of mass
  Real labCenterOfMass[2] = {0, 0};
  Real orientation = origAng;

  const bool bFixed;
  const bool bFixedx;
  const bool bFixedy;
  const bool bForced;
  const bool bForcedx;
  const bool bForcedy;
  const bool bBlockang;
  const Real forcedu;
  const Real forcedv;
  const Real forcedomega;
  const bool bDumpSurface;
  const Real timeForced;
  const int breakSymmetryType;
  const Real breakSymmetryStrength;
  const Real breakSymmetryTime;

  Real M = 0;
  Real J = 0;
  Real u = forcedu; // in lab frame, not sim frame
  Real v = forcedv; // in lab frame, not sim frame
  Real omega = forcedomega;
  Real fluidAngMom = 0;
  Real fluidMomX = 0;
  Real fluidMomY = 0;
  Real penalDX = 0;
  Real penalDY = 0;
  Real penalM = 0;
  Real penalJ = 0;
  Real appliedForceX = 0;
  Real appliedForceY = 0;
  Real appliedTorque = 0;

  Real perimeter = 0, forcex = 0, forcey = 0, forcex_P = 0, forcey_P = 0;
  Real forcex_V = 0, forcey_V = 0, torque = 0, torque_P = 0, torque_V = 0;
  Real drag = 0, thrust = 0, lift = 0, circulation = 0, Pout = 0, PoutNew = 0,
       PoutBnd = 0, defPower = 0;
  Real defPowerBnd = 0, Pthrust = 0, Pdrag = 0, EffPDef = 0, EffPDefBnd = 0;

  virtual void resetAll() {
    center[0] = origC[0];
    center[1] = origC[1];
    centerOfMass[0] = origC[0];
    centerOfMass[1] = origC[1];
    labCenterOfMass[0] = 0;
    labCenterOfMass[1] = 0;
    orientation = origAng;
    M = 0;
    J = 0;
    u = forcedu;
    v = forcedv;
    omega = forcedomega;
    fluidMomX = 0;
    fluidMomY = 0;
    fluidAngMom = 0;
    appliedForceX = 0;
    appliedForceY = 0;
    appliedTorque = 0;
    d_gm[0] = 0;
    d_gm[1] = 0;
    for (auto &entry : obstacleBlocks)
      delete entry;
    obstacleBlocks.clear();
  }

protected:
  /*
    inline void rotate(Real p[2]) const
    {
      const Real x = p[0], y = p[1];
      p[0] =  x*std::cos(orientation) + y*std::sin(orientation);
      p[1] = -x*std::sin(orientation) + y*std::cos(orientation);
    }
  */
public:
  Shape(SimulationData &s, cubism::ArgumentParser &p, Real C[2]);

  virtual ~Shape();

  virtual Real getCharLength() const = 0;
  virtual Real getCharSpeed() const {
    return std::sqrt(forcedu * forcedu + forcedv * forcedv);
  }
  virtual Real getCharMass() const;
  virtual Real getMaxVel() const;

  virtual void create(const std::vector<cubism::BlockInfo> &vInfo) = 0;
  virtual void finalize() {};

  virtual void updateVelocity(Real dt);
  virtual void updatePosition(Real dt);

  void setCentroid(Real C[2]) {
    this->center[0] = C[0];
    this->center[1] = C[1];
    const Real cost = std::cos(this->orientation);
    const Real sint = std::sin(this->orientation);
    this->centerOfMass[0] = C[0] - cost * this->d_gm[0] + sint * this->d_gm[1];
    this->centerOfMass[1] = C[1] - sint * this->d_gm[0] - cost * this->d_gm[1];
  }

  void setCenterOfMass(Real com[2]) {
    this->centerOfMass[0] = com[0];
    this->centerOfMass[1] = com[1];
    const Real cost = std::cos(this->orientation);
    const Real sint = std::sin(this->orientation);
    this->center[0] = com[0] + cost * this->d_gm[0] - sint * this->d_gm[1];
    this->center[1] = com[1] + sint * this->d_gm[0] + cost * this->d_gm[1];
  }

  void getCentroid(Real centroid[2]) const {
    centroid[0] = this->center[0];
    centroid[1] = this->center[1];
  }

  virtual void getCenterOfMass(Real com[2]) const {
    com[0] = this->centerOfMass[0];
    com[1] = this->centerOfMass[1];
  }

  void getLabPosition(Real com[2]) const {
    com[0] = this->labCenterOfMass[0];
    com[1] = this->labCenterOfMass[1];
  }

  Real getU() const { return u; }
  Real getV() const { return v; }
  Real getW() const { return omega; }

  Real getOrientation() const { return this->orientation; }
  void setOrientation(const Real angle) { this->orientation = angle; }

  // functions needed for restarting the simulation
  virtual void saveRestart(FILE *f);
  virtual void loadRestart(FILE *f);

  struct Integrals {
    const Real x, y, m, j, u, v, a;
    Integrals(Real _x, Real _y, Real _m, Real _j, Real _u, Real _v, Real _a)
        : x(_x), y(_y), m(_m), j(_j), u(_u), v(_v), a(_a) {}
    Integrals(const Integrals &c)
        : x(c.x), y(c.y), m(c.m), j(c.j), u(c.u), v(c.v), a(c.a) {}
  };

  Integrals integrateObstBlock(const std::vector<cubism::BlockInfo> &vInfo);

  virtual void removeMoments(const std::vector<cubism::BlockInfo> &vInfo);

  virtual void updateLabVelocity(int mSum[2], Real uSum[2]);

  void penalize();

  void diagnostics();

  virtual void computeForces();
};

class findMaxU {
  SimulationData &sim;
  const std::vector<cubism::BlockInfo> &velInfo = sim.vel->getBlocksInfo();

public:
  findMaxU(SimulationData &s) : sim(s) {}

  Real run() const;

  std::string getName() const { return "findMaxU"; }
};

class Checker {
  SimulationData &sim;
  const std::vector<cubism::BlockInfo> &velInfo = sim.vel->getBlocksInfo();

public:
  Checker(SimulationData &s) : sim(s) {}

  void run(std::string when) const;

  std::string getName() const { return "Checker"; }
};

class IC : public Operator {
protected:
  const std::vector<cubism::BlockInfo> &velInfo = sim.vel->getBlocksInfo();

public:
  IC(SimulationData &s) : Operator(s) {}

  void operator()(const Real dt);

  std::string getName() { return "IC"; }
};

class gaussianIC : public Operator {
protected:
  const std::vector<cubism::BlockInfo> &velInfo = sim.vel->getBlocksInfo();

public:
  gaussianIC(SimulationData &s) : Operator(s) {}

  void operator()(const Real dt);

  std::string getName() { return "gaussianIC"; }
};

class randomIC : public Operator {
protected:
  const std::vector<cubism::BlockInfo> &velInfo = sim.vel->getBlocksInfo();

public:
  randomIC(SimulationData &s) : Operator(s) {}

  void operator()(const Real dt);

  std::string getName() { return "randomIC"; }
};

class ApplyObjVel : public Operator {
protected:
  const std::vector<cubism::BlockInfo> &velInfo = sim.vel->getBlocksInfo();

public:
  ApplyObjVel(SimulationData &s) : Operator(s) {}

  void operator()(const Real dt);

  std::string getName() { return "ApplyObjVel"; }
};

struct KernelVorticity {
  KernelVorticity(const SimulationData &s) : sim(s) {}
  const SimulationData &sim;
  const std::vector<cubism::BlockInfo> &tmpInfo = sim.tmp->getBlocksInfo();
  const cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0, 1}};
  void operator()(VectorLab &lab, const cubism::BlockInfo &info) const {
    const Real i2h = 0.5 / info.h;
    auto &__restrict__ TMP = *(ScalarBlock *)tmpInfo[info.blockID].ptrBlock;
    for (int y = 0; y < VectorBlock::sizeY; ++y)
      for (int x = 0; x < VectorBlock::sizeX; ++x)
        TMP(x, y).s = i2h * ((lab(x, y - 1).u[0] - lab(x, y + 1).u[0]) +
                             (lab(x + 1, y).u[1] - lab(x - 1, y).u[1]));
  }
};

class computeVorticity : public Operator {
public:
  computeVorticity(SimulationData &s) : Operator(s) {}

  void operator()(const Real dt) {
    const KernelVorticity mykernel(sim);
    cubism::compute<VectorLab>(mykernel, sim.vel);

    if (!sim.muteAll)
      reportVorticity();
  }

  void reportVorticity() const {
    Real maxv = -1e10;
    Real minv = -1e10;
#pragma omp parallel for reduction(max : minv, maxv)
    for (auto &info : sim.tmp->getBlocksInfo()) {
      auto &TMP = *(ScalarBlock *)info.ptrBlock;
      for (int y = 0; y < VectorBlock::sizeY; ++y)
        for (int x = 0; x < VectorBlock::sizeX; ++x) {
          maxv = std::max(maxv, TMP(x, y).s);
          minv = std::max(minv, -TMP(x, y).s);
        }
    }
    Real buffer[2] = {maxv, minv};
    Real recvbuf[2];
    MPI_Reduce(buffer, recvbuf, 2, MPI_Real, MPI_MAX, 0,
               sim.chi->getWorldComm());
    recvbuf[1] = -recvbuf[1];
    if (sim.rank == 0)
      std::cout << " max(omega)=" << recvbuf[0] << " min(omega)=" << recvbuf[1]
                << " max(omega)+min(omega)=" << recvbuf[0] + recvbuf[1]
                << std::endl;
  }

  std::string getName() { return "computeVorticity"; }
};

struct KernelQ {
  KernelQ(const SimulationData &s) : sim(s) {}
  const SimulationData &sim;
  const std::vector<cubism::BlockInfo> &tmpInfo = sim.tmp->getBlocksInfo();
  const cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0, 1}};
  void operator()(VectorLab &lab, const cubism::BlockInfo &info) const {
    const Real i2h = 0.5 / info.h;
    auto &__restrict__ TMP = *(ScalarBlock *)tmpInfo[info.blockID].ptrBlock;
    for (int y = 0; y < VectorBlock::sizeY; ++y)
      for (int x = 0; x < VectorBlock::sizeX; ++x) {
        const Real WZ = i2h * ((lab(x, y - 1).u[0] - lab(x, y + 1).u[0]) +
                               (lab(x + 1, y).u[1] - lab(x - 1, y).u[1]));
        const Real D11 =
            i2h * (lab(x + 1, y).u[0] - lab(x - 1, y).u[0]); // shear stresses
        const Real D22 =
            i2h * (lab(x, y + 1).u[1] - lab(x, y - 1).u[1]); // shear stresses
        const Real D12 =
            i2h * ((lab(x, y + 1).u[0] - lab(x, y - 1).u[0]) +
                   (lab(x + 1, y).u[1] - lab(x - 1, y).u[1])); // shear stresses
        const Real SS = D11 * D11 + D22 * D22 + 0.5 * (D12 * D12);
        TMP(x, y).s = 0.5 * (0.5 * (WZ * WZ) - SS);
      }
  }
};

class computeQ : public Operator {
public:
  computeQ(SimulationData &s) : Operator(s) {}

  void operator()(const Real dt) {
    const KernelQ mykernel(sim);
    cubism::compute<VectorLab>(mykernel, sim.vel);
  }

  std::string getName() { return "computeQ"; }
};

struct KernelDivergence {
  KernelDivergence(const SimulationData &s) : sim(s) {}
  const SimulationData &sim;
  const std::vector<cubism::BlockInfo> &tmpInfo = sim.tmp->getBlocksInfo();
  const cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0, 1}};
  void operator()(VectorLab &lab, const cubism::BlockInfo &info) const {
    const Real h = info.h;
    const Real facDiv = 0.5 * h;
    auto &__restrict__ TMP = *(ScalarBlock *)tmpInfo[info.blockID].ptrBlock;
    for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix)
        TMP(ix, iy).s =
            facDiv * ((lab(ix + 1, iy).u[0] - lab(ix - 1, iy).u[0]) +
                      (lab(ix, iy + 1).u[1] - lab(ix, iy - 1).u[1]));
    cubism::BlockCase<ScalarBlock> *tempCase =
        (cubism::BlockCase<ScalarBlock> *)(tmpInfo[info.blockID].auxiliary);
    ScalarBlock::ElementType *faceXm = nullptr;
    ScalarBlock::ElementType *faceXp = nullptr;
    ScalarBlock::ElementType *faceYm = nullptr;
    ScalarBlock::ElementType *faceYp = nullptr;
    if (tempCase != nullptr) {
      faceXm = tempCase->storedFace[0] ? &tempCase->m_pData[0][0] : nullptr;
      faceXp = tempCase->storedFace[1] ? &tempCase->m_pData[1][0] : nullptr;
      faceYm = tempCase->storedFace[2] ? &tempCase->m_pData[2][0] : nullptr;
      faceYp = tempCase->storedFace[3] ? &tempCase->m_pData[3][0] : nullptr;
    }
    if (faceXm != nullptr) {
      int ix = 0;
      for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
        faceXm[iy].s = facDiv * (lab(ix - 1, iy).u[0] + lab(ix, iy).u[0]);
    }
    if (faceXp != nullptr) {
      int ix = VectorBlock::sizeX - 1;
      for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
        faceXp[iy].s = -facDiv * (lab(ix + 1, iy).u[0] + lab(ix, iy).u[0]);
    }
    if (faceYm != nullptr) {
      int iy = 0;
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix)
        faceYm[ix].s = facDiv * (lab(ix, iy - 1).u[1] + lab(ix, iy).u[1]);
    }
    if (faceYp != nullptr) {
      int iy = VectorBlock::sizeY - 1;
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix)
        faceYp[ix].s = -facDiv * (lab(ix, iy + 1).u[1] + lab(ix, iy).u[1]);
    }
  }
};

class computeDivergence : public Operator {
public:
  computeDivergence(SimulationData &s) : Operator(s) {}

  void operator()(const Real dt) {

    const KernelDivergence mykernel(sim);
    cubism::compute<VectorLab>(mykernel, sim.vel, sim.tmp);
#if 0
    Real total = 0.0;
    Real abs   = 0.0;
    for (auto & info: sim.tmp->getBlocksInfo())
    {
      auto & TMP = *(ScalarBlock*) info.ptrBlock;
      for(int y=0; y<VectorBlock::sizeY; ++y)
      for(int x=0; x<VectorBlock::sizeX; ++x)
      {
        abs   += std::fabs(TMP(x,y).s);
        total += TMP(x,y).s;
      }
    }
    Real sendbuf[2]={total,abs};
    Real recvbuf[2];
    MPI_Reduce(sendbuf, recvbuf, 2, MPI_Real, MPI_SUM, 0, sim.chi->getWorldComm());
    if (sim.rank == 0)
    {
      ofstream myfile;
      myfile.open ("div.txt",ios::app);
      myfile << sim.step << " " << total << " " << abs << std::endl;
      myfile.close();
    }
#endif
  }

  std::string getName() { return "computeDivergence"; }
};

namespace fs = std::filesystem;

// Function to retrieve HDF5 type (hid_t) for a given real type.
// If using custom types, the user should specialize this function.
template <typename T> hid_t get_hdf5_type();
template <> inline hid_t get_hdf5_type<long long>() { return H5T_NATIVE_LLONG; }
template <> inline hid_t get_hdf5_type<short int>() { return H5T_NATIVE_SHORT; }
template <> inline hid_t get_hdf5_type<int>() { return H5T_NATIVE_INT; }
template <> inline hid_t get_hdf5_type<float>() { return H5T_NATIVE_FLOAT; }
template <> inline hid_t get_hdf5_type<double>() { return H5T_NATIVE_DOUBLE; }
template <> inline hid_t get_hdf5_type<long double>() {
  return H5T_NATIVE_LDOUBLE;
}

namespace cubism {

/// used for dumping a ScalarElement
struct StreamerScalar {
  static constexpr int NCHANNELS = 1;
  template <typename TBlock, typename T>
  static inline void operate(TBlock &b, const int ix, const int iy,
                             const int iz, T output[NCHANNELS]) {
    output[0] = b(ix, iy, iz).s;
  }
  static std::string prefix() { return std::string(""); }
  static const char *getAttributeName() { return "Scalar"; }
};

/// used for dumping a VectorElement
struct StreamerVector {
  static constexpr int NCHANNELS = 3;
  template <typename TBlock, typename T>
  static void operate(TBlock &b, const int ix, const int iy, const int iz,
                      T output[NCHANNELS]) {
    for (int i = 0; i < TBlock::ElementType::DIM; i++)
      output[i] = b(ix, iy, iz).u[i];
  }
  static std::string prefix() { return std::string(""); }
  static const char *getAttributeName() { return "Vector"; }
};

template <typename TStreamer, typename hdf5Real, typename TGrid>
void DumpHDF5_uniform(const TGrid &grid, const typename TGrid::Real absTime,
                      const std::string &fname,
                      const std::string &dpath = ".") {
  // only for 2D!

  typedef typename TGrid::BlockType B;
  const unsigned int nX = B::sizeX;
  const unsigned int nY = B::sizeY;
  // const unsigned int nZ = B::sizeZ;

  // fname is the base filepath without file type extension
  std::ostringstream filename;
  std::ostringstream fullpath;
  filename << fname;
  fullpath << dpath << "/" << filename.str();
  std::vector<BlockInfo> MyInfos = grid.getBlocksInfo();
  const int levelMax = grid.getlevelMax();
  std::array<int, 3> bpd = grid.getMaxBlocks();
  const unsigned int unx = bpd[0] * (1 << (levelMax - 1)) * nX;
  const unsigned int uny = bpd[1] * (1 << (levelMax - 1)) * nY;
  // const int unz = bpd[2]*(1<<(levelMax-1))*nZ;
  const unsigned int NCHANNELS = TStreamer::NCHANNELS;
  double hmin = 1e10;
  for (size_t i = 0; i < MyInfos.size(); i++)
    hmin = std::min(hmin, MyInfos[i].h);
  const double h = hmin;

  // TODO: Refactor, move the interpolation logic into a separate function at
  // the level of a Grid, see copyToUniformNoInterpolation for reference.

  std::vector<float> uniform_mesh(uny * unx * NCHANNELS);
  for (size_t i = 0; i < MyInfos.size(); i++) {
    const BlockInfo &info = MyInfos[i];
    const int level = info.level;

    for (unsigned int y = 0; y < nY; y++)
      for (unsigned int x = 0; x < nX; x++) {
        B &block = *(B *)info.ptrBlock;

        float output[NCHANNELS] = {0.0};
        float dudx[NCHANNELS] = {0.0};
        float dudy[NCHANNELS] = {0.0};
        TStreamer::operate(block, x, y, 0, (float *)output);

        if (x != 0 && x != nX - 1) {
          float output_p[NCHANNELS] = {0.0};
          float output_m[NCHANNELS] = {0.0};
          TStreamer::operate(block, x + 1, y, 0, (float *)output_p);
          TStreamer::operate(block, x - 1, y, 0, (float *)output_m);
          for (unsigned int j = 0; j < NCHANNELS; ++j)
            dudx[j] = 0.5 * (output_p[j] - output_m[j]);
        } else if (x == 0) {
          float output_p[NCHANNELS] = {0.0};
          TStreamer::operate(block, x + 1, y, 0, (float *)output_p);
          for (unsigned int j = 0; j < NCHANNELS; ++j)
            dudx[j] = output_p[j] - output[j];
        } else {
          float output_m[NCHANNELS] = {0.0};
          TStreamer::operate(block, x - 1, y, 0, (float *)output_m);
          for (unsigned int j = 0; j < NCHANNELS; ++j)
            dudx[j] = output[j] - output_m[j];
        }

        if (y != 0 && y != nY - 1) {
          float output_p[NCHANNELS] = {0.0};
          float output_m[NCHANNELS] = {0.0};
          TStreamer::operate(block, x, y + 1, 0, (float *)output_p);
          TStreamer::operate(block, x, y - 1, 0, (float *)output_m);
          for (unsigned int j = 0; j < NCHANNELS; ++j)
            dudy[j] = 0.5 * (output_p[j] - output_m[j]);
        } else if (y == 0) {
          float output_p[NCHANNELS] = {0.0};
          TStreamer::operate(block, x, y + 1, 0, (float *)output_p);
          for (unsigned int j = 0; j < NCHANNELS; ++j)
            dudy[j] = output_p[j] - output[j];
        } else {
          float output_m[NCHANNELS] = {0.0};
          TStreamer::operate(block, x, y - 1, 0, (float *)output_m);
          for (unsigned int j = 0; j < NCHANNELS; ++j)
            dudy[j] = output[j] - output_m[j];
        }

        int iy_start =
            (info.index[1] * nY + y) * (1 << ((levelMax - 1) - level));
        int ix_start =
            (info.index[0] * nX + x) * (1 << ((levelMax - 1) - level));

        const int points = 1 << ((levelMax - 1) - level);
        const double dh = 1.0 / points;

        for (int iy = iy_start; iy < iy_start + (1 << ((levelMax - 1) - level));
             iy++)
          for (int ix = ix_start;
               ix < ix_start + (1 << ((levelMax - 1) - level)); ix++) {
            double cx = (ix - ix_start - points / 2 + 1 - 0.5) * dh;
            double cy = (iy - iy_start - points / 2 + 1 - 0.5) * dh;
            for (unsigned int j = 0; j < NCHANNELS; ++j)
              uniform_mesh[iy * NCHANNELS * unx + ix * NCHANNELS + j] =
                  output[j] + cx * dudx[j] + cy * dudy[j];
          }
      }
  }

  hid_t file_id, dataset_id, fspace_id, plist_id;
  H5open();
  // 1.Set up file access property list with parallel I/O access
  // 2.Create a new file collectively and release property list identifier.
  // 3.All ranks need to create datasets dset*
  hsize_t dims[4] = {1, uny, unx, NCHANNELS};

  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  file_id = H5Fcreate((fullpath.str() + "uniform.h5").c_str(), H5F_ACC_TRUNC,
                      H5P_DEFAULT, plist_id);
  fspace_id = H5Screate_simple(4, dims, NULL);
  dataset_id = H5Dcreate(file_id, "dset", H5T_NATIVE_FLOAT, fspace_id,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Pclose(plist_id);
  H5Dclose(dataset_id);
  H5Sclose(fspace_id);

  dataset_id = H5Dopen(file_id, "dset", H5P_DEFAULT);
  H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           uniform_mesh.data());
  H5Dclose(dataset_id);

  // 5.Close hdf5 file
  H5Fclose(file_id);
  H5close();

  // 6.Write grid meta-data
  {
    FILE *xmf = 0;
    xmf = fopen((fullpath.str() + "uniform.xmf").c_str(), "w");
    fprintf(xmf, "<?xml version=\"1.0\" ?>\n");
    fprintf(xmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
    fprintf(xmf, "<Xdmf Version=\"2.0\">\n");
    fprintf(xmf, " <Domain>\n");
    fprintf(xmf, "   <Grid GridType=\"Uniform\">\n");
    fprintf(xmf, "     <Time Value=\"%e\"/>\n\n", absTime);
    fprintf(xmf,
            "     <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\"%d %d "
            "%d\"/>\n\n",
            1 + 1, uny + 1, unx + 1);
    fprintf(xmf, "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
    fprintf(xmf, "       <DataItem Dimensions=\"3\" NumberType=\"Double\" "
                 "Precision=\"8\" "
                 "Format=\"XML\">\n");
    fprintf(xmf, "        %e %e %e\n", 0.0, 0.0, 0.0);
    fprintf(xmf, "       </DataItem>\n");
    fprintf(xmf, "       <DataItem Dimensions=\"3\" NumberType=\"Double\" "
                 "Precision=\"8\" "
                 "Format=\"XML\">\n");
    fprintf(xmf, "        %e %e %e\n", h, h, h);
    fprintf(xmf, "       </DataItem>\n");
    fprintf(xmf, "     </Geometry>\n\n");
    fprintf(
        xmf,
        "     <Attribute Name=\"dset\" AttributeType=\"%s\" Center=\"Cell\">\n",
        TStreamer::getAttributeName());
    fprintf(xmf,
            "       <DataItem Dimensions=\"%d %d %d %d\" NumberType=\"Float\" "
            "Precision=\"8\" Format=\"HDF\">\n",
            1, uny, unx, NCHANNELS);
    fprintf(xmf, "        %s:/dset\n", (filename.str() + "uniform.h5").c_str());
    fprintf(xmf, "       </DataItem>\n");
    fprintf(xmf, "     </Attribute>\n");
    fprintf(xmf, "   </Grid>\n");
    fprintf(xmf, " </Domain>\n");
    fprintf(xmf, "</Xdmf>\n");
    fclose(xmf);
  }
}

template <typename data_type>
void read_buffer_from_file(std::vector<data_type> &buffer, MPI_Comm &comm,
                           const std::string &name,
                           const std::string &dataset_name, const int chunk) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  hid_t file_id, dataset_id, fspace_id, fapl_id, mspace_id;

  // 1. Open file
  fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);
  file_id = H5Fopen(name.c_str(), H5F_ACC_RDONLY, fapl_id);
  H5Pclose(fapl_id);

  // 2. Dataset property list
  fapl_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(fapl_id, H5FD_MPIO_COLLECTIVE);

  // 3. Read dataset size
  dataset_id = H5Dopen2(file_id, dataset_name.c_str(), H5P_DEFAULT);
  hsize_t total = H5Dget_storage_size(dataset_id) / sizeof(data_type) / chunk;

  // 4. Determine part of the dataset to be read by this rank
  unsigned long long my_data = total / size;
  if ((hsize_t)rank < total % (hsize_t)size)
    my_data++;
  unsigned long long n_start = rank * (total / size);
  if (total % size > 0) {
    if ((hsize_t)rank < total % (hsize_t)size)
      n_start += rank;
    else
      n_start += total % size;
  }
  hsize_t offset = n_start * chunk;
  hsize_t count = my_data * chunk;
  buffer.resize(count);

  // 5. Read from file
  fspace_id = H5Dget_space(dataset_id);
  mspace_id = H5Screate_simple(1, &count, NULL);
  H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, &offset, NULL, &count, NULL);
  H5Dread(dataset_id, get_hdf5_type<data_type>(), mspace_id, fspace_id, fapl_id,
          buffer.data());

  // 6. Close stuff
  H5Pclose(fapl_id);
  H5Dclose(dataset_id);
  H5Sclose(fspace_id);
  H5Sclose(mspace_id);
  H5Fclose(file_id);
}

template <typename data_type>
void save_buffer_to_file(const std::vector<data_type> &buffer,
                         const int NCHANNELS, MPI_Comm &comm,
                         const std::string &name,
                         const std::string &dataset_name, const hid_t &file_id,
                         const hid_t &fapl_id)

{
  assert(buffer.size() % NCHANNELS == 0);
  unsigned long long MyCells = buffer.size() / NCHANNELS;
  unsigned long long TotalCells;
  MPI_Allreduce(&MyCells, &TotalCells, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                comm);

  hsize_t base_tmp[1] = {0};
  MPI_Exscan(&MyCells, &base_tmp[0], 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
  base_tmp[0] *= NCHANNELS;

  hid_t dataset_id, fspace_id, mspace_id;

  hsize_t dims[1] = {(hsize_t)TotalCells * NCHANNELS};
  fspace_id = H5Screate_simple(1, dims, NULL);
  dataset_id =
      H5Dcreate(file_id, dataset_name.c_str(), get_hdf5_type<data_type>(),
                fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t count[1] = {MyCells * NCHANNELS};
  fspace_id = H5Dget_space(dataset_id);
  mspace_id = H5Screate_simple(1, count, NULL);
  H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, base_tmp, NULL, count, NULL);
  H5Dwrite(dataset_id, get_hdf5_type<data_type>(), mspace_id, fspace_id,
           fapl_id, buffer.data());

  H5Sclose(mspace_id);
  H5Sclose(fspace_id);
  H5Dclose(dataset_id);

#if 0 // compression
    hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t cdims[1];
    cdims[0] = 8*8*8;
    if (compression==false)
    {
        const int PtsPerElement = 8;
        cdims[0] *= PtsPerElement * DIMENSION;
    }
    H5Pset_chunk(plist_id, 1, cdims);
    H5Pset_deflate(plist_id, 6);
    dataset_id = H5Dcreate(file_id, dataset_name.c_str(), get_hdf5_type<data_type>(), fspace_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    hsize_t count[1] = {MyCells*NCHANNELS};
    fspace_id = H5Dget_space(dataset_id);
    mspace_id = H5Screate_simple(1, count, NULL);
    H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, base_tmp, NULL, count, NULL);
    H5Dwrite(dataset_id, get_hdf5_type<data_type>(), mspace_id, fspace_id, fapl_id, buffer.data());
    H5Sclose(mspace_id);
    H5Sclose(fspace_id);
    H5Dclose(dataset_id);
    H5Pclose(plist_id);
#endif
}

static double latestTime{-1.0};
// The following requirements for the data TStreamer are required:
// TStreamer::NCHANNELS        : Number of data elements (1=Scalar, 3=Vector,
// 9=Tensor) TStreamer::operate          : Data access methods for read and
// write TStreamer::getAttributeName : Attribute name of the date ("Scalar",
// "Vector", "Tensor")
template <typename TStreamer, typename hdf5Real, typename TGrid>
void DumpHDF5_MPI(TGrid &grid, typename TGrid::Real absTime,
                  const std::string &fname, const std::string &dpath = ".",
                  const bool dumpGrid = true) {
  const bool SaveGrid = latestTime < absTime && dumpGrid;

  int gridCount = 0;
  for (auto &p : fs::recursive_directory_iterator(dpath)) {
    if (p.path().extension() == ".h5") {
      std::string g = p.path().stem().string();
      g.resize(4);
      if (g == "grid") {
        gridCount++;
      }
    }
  }
  if (SaveGrid == false)
    gridCount--;

  latestTime = absTime;

  typedef typename TGrid::BlockType B;
  const int nX = B::sizeX;
  const int nY = B::sizeY;
  const int nZ = B::sizeZ;
  const int NCHANNELS = TStreamer::NCHANNELS;

  MPI_Comm comm = grid.getWorldComm();
  const int rank = grid.myrank;
  std::ostringstream filename;
  std::ostringstream fullpath;
  filename << fname; // fname is the base filepath without file type extension
  fullpath << dpath << "/" << filename.str();

#if DIMENSION == 2
  const int PtsPerElement = 4;
#else
  const int PtsPerElement = 8;
#endif
  std::vector<BlockInfo> &MyInfos = grid.getBlocksInfo();
  unsigned long long MyCells = MyInfos.size() * nX * nY * nZ;
  unsigned long long TotalCells;
  MPI_Allreduce(&MyCells, &TotalCells, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                comm);

  H5open();
  hid_t file_id, fapl_id;

  // 1.Set up file access property list with parallel I/O access
  fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);

  // 2.Create a new file collectively and release property list identifier.
  file_id = H5Fcreate((fullpath.str() + ".h5").c_str(), H5F_ACC_TRUNC,
                      H5P_DEFAULT, fapl_id);
  H5Pclose(fapl_id);
  H5Fclose(file_id);

  hid_t file_id_grid, fapl_id_grid;
  std::stringstream gridFile_s;
  gridFile_s << "grid" << std::setfill('0') << std::setw(9) << gridCount
             << ".h5";
  std::string gridFile = gridFile_s.str();

  std::stringstream gridFilePath_s;
  gridFilePath_s << dpath << "/grid" << std::setfill('0') << std::setw(9)
                 << gridCount << ".h5";
  std::string gridFilePath = gridFilePath_s.str();

  if (SaveGrid) {
    // 1.Set up file access property list with parallel I/O access
    fapl_id_grid = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id_grid, comm, MPI_INFO_NULL);

    // 2.Create a new file collectively and release property list identifier.
    file_id_grid = H5Fcreate(gridFilePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                             fapl_id_grid);
    H5Pclose(fapl_id_grid);
    H5Fclose(file_id_grid);
  }

  // Write grid meta-data
  if (rank == 0 && dumpGrid) {
    std::ostringstream myfilename;
    myfilename << filename.str();
    std::stringstream s;
    s << "<?xml version=\"1.0\" ?>\n";
    s << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    s << "<Xdmf Version=\"2.0\">\n";
    s << "<Domain>\n";
    s << " <Grid Name=\"OctTree\" GridType=\"Uniform\">\n";
    s << "  <Time Value=\"" << std::scientific << absTime << "\"/>\n\n";
#if DIMENSION == 2
    s << "   <Topology NumberOfElements=\"" << TotalCells
      << "\" TopologyType=\"Quadrilateral\"/>\n";
    s << "     <Geometry GeometryType=\"XY\">\n";
#else
    s << "   <Topology NumberOfElements=\"" << TotalCells
      << "\" TopologyType=\"Hexahedron\"/>\n";
    s << "     <Geometry GeometryType=\"XYZ\">\n";
#endif
    // s << "        <DataItem ItemType=\"Uniform\"  Dimensions=\" " <<
    // TotalCells*PtsPerElement << " " << DIMENSION << "\" NumberType=\"Float\"
    // Precision=\" " << (int)sizeof(hdf5Real) << "\" Format=\"HDF\">\n";
    s << "        <DataItem ItemType=\"Uniform\"  Dimensions=\" "
      << TotalCells * PtsPerElement << " " << DIMENSION
      << "\" NumberType=\"Float\" Precision=\" " << (int)sizeof(float)
      << "\" Format=\"HDF\">\n";
    s << "            " << gridFile.c_str() << ":/"
      << "vertices"
      << "\n";
    s << "        </DataItem>\n";
    s << "     </Geometry>\n";
    s << "     <Attribute Name=\"data\" AttributeType=\""
      << TStreamer::getAttributeName() << "\" Center=\"Cell\">\n";
    s << "        <DataItem ItemType=\"Uniform\"  Dimensions=\" " << TotalCells
      << " " << NCHANNELS << "\" NumberType=\"Float\" Precision=\" "
      << (int)sizeof(hdf5Real) << "\" Format=\"HDF\">\n";
    s << "            " << (myfilename.str() + ".h5").c_str() << ":/"
      << "data"
      << "\n";
    s << "        </DataItem>\n";
    s << "     </Attribute>\n";
    s << " </Grid>\n";
    s << "</Domain>\n";
    s << "</Xdmf>\n";
    std::string st = s.str();
    FILE *xmf = 0;
    xmf = fopen((fullpath.str() + "-new.xmf").c_str(), "w");
    fprintf(xmf, "%s", st.c_str());
    fclose(xmf);
  }

  std::string name = fullpath.str() + ".h5";

  fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);
  file_id = H5Fopen(name.c_str(), H5F_ACC_RDWR, fapl_id);
  H5Pclose(fapl_id);
  fapl_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(fapl_id, H5FD_MPIO_COLLECTIVE);

  // Dump grid structure (used when restarting)
  {
    std::vector<short int> bufferlevel(MyInfos.size());
    std::vector<long long> bufferZ(MyInfos.size());
    for (size_t i = 0; i < MyInfos.size(); i++) {
      bufferlevel[i] = MyInfos[i].level;
      bufferZ[i] = MyInfos[i].Z;
    }
    save_buffer_to_file<short int>(bufferlevel, 1, comm, fullpath.str() + ".h5",
                                   "blockslevel", file_id, fapl_id);
    save_buffer_to_file<long long>(bufferZ, 1, comm, fullpath.str() + ".h5",
                                   "blocksZ", file_id, fapl_id);
  }
  // Dump vertices
  if (SaveGrid) {
    fapl_id_grid = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id_grid, comm, MPI_INFO_NULL);
    file_id_grid = H5Fopen(gridFilePath.c_str(), H5F_ACC_RDWR, fapl_id_grid);
    H5Pclose(fapl_id_grid);
    fapl_id_grid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(fapl_id_grid, H5FD_MPIO_COLLECTIVE);

    std::vector<float> buffer(MyCells * PtsPerElement * DIMENSION);
    for (size_t i = 0; i < MyInfos.size(); i++) {
      const BlockInfo &info = MyInfos[i];
      const float h2 = 0.5 * info.h;
      for (int z = 0; z < nZ; z++)
        for (int y = 0; y < nY; y++)
          for (int x = 0; x < nX; x++) {
            const int bbase = (i * nZ * nY * nX + z * nY * nX + y * nX + x) *
                              PtsPerElement * DIMENSION;
#if DIMENSION == 3
            float p[3];
            info.pos(p, x, y, z);
            //(0,0,0)
            buffer[bbase] = p[0] - h2;
            buffer[bbase + 1] = p[1] - h2;
            buffer[bbase + 2] = p[2] - h2;
            //(0,0,1)
            buffer[bbase + DIMENSION] = p[0] - h2;
            buffer[bbase + DIMENSION + 1] = p[1] - h2;
            buffer[bbase + DIMENSION + 2] = p[2] + h2;
            //(0,1,1)
            buffer[bbase + 2 * DIMENSION] = p[0] - h2;
            buffer[bbase + 2 * DIMENSION + 1] = p[1] + h2;
            buffer[bbase + 2 * DIMENSION + 2] = p[2] + h2;
            //(0,1,0)
            buffer[bbase + 3 * DIMENSION] = p[0] - h2;
            buffer[bbase + 3 * DIMENSION + 1] = p[1] + h2;
            buffer[bbase + 3 * DIMENSION + 2] = p[2] - h2;
            //(1,0,0)
            buffer[bbase + 4 * DIMENSION] = p[0] + h2;
            buffer[bbase + 4 * DIMENSION + 1] = p[1] - h2;
            buffer[bbase + 4 * DIMENSION + 2] = p[2] - h2;
            //(1,0,1)
            buffer[bbase + 5 * DIMENSION] = p[0] + h2;
            buffer[bbase + 5 * DIMENSION + 1] = p[1] - h2;
            buffer[bbase + 5 * DIMENSION + 2] = p[2] + h2;
            //(1,1,1)
            buffer[bbase + 6 * DIMENSION] = p[0] + h2;
            buffer[bbase + 6 * DIMENSION + 1] = p[1] + h2;
            buffer[bbase + 6 * DIMENSION + 2] = p[2] + h2;
            //(1,1,0)
            buffer[bbase + 7 * DIMENSION] = p[0] + h2;
            buffer[bbase + 7 * DIMENSION + 1] = p[1] + h2;
            buffer[bbase + 7 * DIMENSION + 2] = p[2] - h2;
#else
            double p[2];
            info.pos(p, x, y);
            //(0,0)
            buffer[bbase] = p[0] - h2;
            buffer[bbase + 1] = p[1] - h2;
            //(0,1)
            buffer[bbase + DIMENSION] = p[0] - h2;
            buffer[bbase + DIMENSION + 1] = p[1] + h2;
            //(1,1)
            buffer[bbase + 2 * DIMENSION] = p[0] + h2;
            buffer[bbase + 2 * DIMENSION + 1] = p[1] + h2;
            //(1,0)
            buffer[bbase + 3 * DIMENSION] = p[0] + h2;
            buffer[bbase + 3 * DIMENSION + 1] = p[1] - h2;
#endif
          }
    }
    save_buffer_to_file<float>(buffer, 1, comm, gridFilePath, "vertices",
                               file_id_grid, fapl_id_grid);

    H5Pclose(fapl_id_grid);
    H5Fclose(file_id_grid);
  }
  // Dump data
  {
    std::vector<hdf5Real> buffer(MyCells * NCHANNELS);
    for (size_t i = 0; i < MyInfos.size(); i++) {
      const BlockInfo &info = MyInfos[i];
      B &b = *(B *)info.ptrBlock;
      for (int z = 0; z < nZ; z++)
        for (int y = 0; y < nY; y++)
          for (int x = 0; x < nX; x++) {
            hdf5Real output[NCHANNELS]{0};
            TStreamer::operate(b, x, y, z, output);
            for (int nc = 0; nc < NCHANNELS; nc++) {
              buffer[(i * nZ * nY * nX + z * nY * nX + y * nX + x) * NCHANNELS +
                     nc] = output[nc];
            }
          }
    }
    save_buffer_to_file<hdf5Real>(buffer, NCHANNELS, comm,
                                  fullpath.str() + ".h5", "data", file_id,
                                  fapl_id);
  }

  H5Pclose(fapl_id);
  H5Fclose(file_id);
  H5close();
}

template <typename TStreamer, typename hdf5Real, typename TGrid>
void DumpHDF5_MPI2(TGrid &grid, typename TGrid::Real absTime,
                   const std::string &fname, const std::string &dpath = ".") {
  typedef typename TGrid::BlockType B;
  const int nX = B::sizeX;
  const int nY = B::sizeY;
  const int nZ = B::sizeZ;

  MPI_Comm comm = grid.getWorldComm();
  const int rank = grid.myrank;
  const int size = grid.world_size;
  const int NCHANNELS = TStreamer::NCHANNELS;
  std::ostringstream filename;
  std::ostringstream fullpath;
  filename << fname; // fname is the base filepath without file type extension
  fullpath << dpath << "/" << filename.str();

  if (rank == 0)
    std::filesystem::remove((fullpath.str() + ".xmf").c_str());

  std::vector<BlockGroup> &MyGroups = grid.MyGroups;
  grid.UpdateMyGroups();

#if DIMENSION == 2
  double hmin = 1e10;
  for (size_t groupID = 0; groupID < MyGroups.size(); groupID++)
    hmin = std::min(hmin, MyGroups[groupID].h);
  MPI_Allreduce(MPI_IN_PLACE, &hmin, 1, MPI_DOUBLE, MPI_MIN, comm);
#endif

  long long mycells = 0;
  for (size_t groupID = 0; groupID < MyGroups.size(); groupID++) {
    mycells += (MyGroups[groupID].NZZ - 1) * (MyGroups[groupID].NYY - 1) *
               (MyGroups[groupID].NXX - 1);
  }
  hsize_t base_tmp[1] = {0};
  MPI_Exscan(&mycells, &base_tmp[0], 1, MPI_LONG_LONG, MPI_SUM, comm);

  long long start = 0;
  // Write grid meta-data
  {
    std::ostringstream myfilename;
    myfilename << filename.str();
    std::stringstream s;
    if (rank == 0) {
      s << "<?xml version=\"1.0\" ?>\n";
      s << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
      s << "<Xdmf Version=\"2.0\">\n";
      s << "<Domain>\n";
      s << " <Grid Name=\"OctTree\" GridType=\"Collection\">\n";
      s << "  <Time Value=\"" << std::scientific << absTime << "\"/>\n\n";
    }
    for (size_t groupID = 0; groupID < MyGroups.size(); groupID++) {
      const BlockGroup &group = MyGroups[groupID];
      const int nXX = group.NXX;
      const int nYY = group.NYY;
      const int nZZ = group.NZZ;
      s << "  <Grid GridType=\"Uniform\">\n";
      s << "   <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\" " << nZZ
        << " " << nYY << " " << nXX << "\"/>\n";
      s << "   <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n";
      s << "   <DataItem Dimensions=\"3\" NumberType=\"Double\" "
           "Precision=\"8\" "
           "Format=\"XML\">\n";
      s << "    " << std::scientific << group.origin[2] << " "
        << group.origin[1] << " " << group.origin[0] << "\n";
      s << "   </DataItem>\n";
      s << "   <DataItem Dimensions=\"3\" NumberType=\"Double\" "
           "Precision=\"8\" "
           "Format=\"XML\">\n";
#if DIMENSION == 3
      s << "    " << std::scientific << group.h << " " << group.h << " "
        << group.h << "\n";
#else
      s << "    " << std::scientific << hmin << " " << group.h << " " << group.h
        << "\n";
#endif
      s << "   </DataItem>\n";
      s << "   </Geometry>\n";

      int dd = (nZZ - 1) * (nYY - 1) * (nXX - 1); //*NCHANNELS;
      s << "   <Attribute Name=\"data\" AttributeType=\""
        << "Scalar"
        << "\" Center=\"Cell\">\n";
      s << "<DataItem ItemType=\"HyperSlab\" Dimensions=\" " << 1 << " " << 1
        << " " << dd << "\" Type=\"HyperSlab\"> \n";
      s << "<DataItem Dimensions=\"3 1\" Format=\"XML\">\n";
      s << base_tmp[0] + start << "\n";
      s << 1 << "\n";
      s << dd << "\n";
      s << "</DataItem>\n";
      s << "   <DataItem ItemType=\"Uniform\"  Dimensions=\" " << dd << " "
        << "\" NumberType=\"Float\" Precision=\" " << (int)sizeof(float)
        << "\" Format=\"HDF\">\n";
      s << "    " << (myfilename.str() + ".h5").c_str() << ":/"
        << "dset"
        << "\n";
      s << "   </DataItem>\n";
      s << "   </DataItem>\n";
      s << "   </Attribute>\n";
      start += dd;
      s << "  </Grid>\n\n";
    }
    if (rank == size - 1) {
      s << " </Grid>\n";
      s << "</Domain>\n";
      s << "</Xdmf>\n";
    }
    std::string st = s.str();
    MPI_Offset offset = 0;
    MPI_Offset len = st.size() * sizeof(char);
    MPI_File xmf;
    MPI_File_open(comm, (fullpath.str() + ".xmf").c_str(),
                  MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &xmf);
    MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, comm);
    MPI_File_write_at_all(xmf, offset, st.data(), st.size(), MPI_CHAR,
                          MPI_STATUS_IGNORE);
    MPI_File_close(&xmf);
  }

  H5open();
  // Write group data to separate hdf5 file
  {
    hid_t file_id, fapl_id;
    hid_t dataset_origins, fspace_origins,
        mspace_origins; // origin[0],origin[1],origin[2],group.h : doubles
    hid_t dataset_indices, fspace_indices,
        mspace_indices; // nx,ny,nz,index[0],index[1],index[2],level : integers

    // 1.Set up file access property list with parallel I/O access
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);

    // 2.Create a new file collectively and release property list identifier.
    file_id = H5Fcreate((fullpath.str() + "-groups.h5").c_str(), H5F_ACC_TRUNC,
                        H5P_DEFAULT, fapl_id);
    H5Pclose(fapl_id);

    // 3.Create datasets
    fapl_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(fapl_id, H5FD_MPIO_COLLECTIVE);

    long long total = MyGroups.size(); // total number of groups
    MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_LONG_LONG, MPI_SUM, comm);

    hsize_t dim_origins = 4 * total;
    hsize_t dim_indices = 7 * total;
    fspace_origins = H5Screate_simple(1, &dim_origins, NULL);
    fspace_indices = H5Screate_simple(1, &dim_indices, NULL);
    dataset_origins =
        H5Dcreate(file_id, "origins", H5T_NATIVE_DOUBLE, fspace_origins,
                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dataset_indices =
        H5Dcreate(file_id, "indices", H5T_NATIVE_INT, fspace_indices,
                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<double> origins(4 * MyGroups.size());
    std::vector<int> indices(7 * MyGroups.size());
    for (size_t groupID = 0; groupID < MyGroups.size(); groupID++) {
      const BlockGroup &group = MyGroups[groupID];
      origins[4 * groupID] = group.origin[0];
      origins[4 * groupID + 1] = group.origin[1];
      origins[4 * groupID + 2] = group.origin[2];
      origins[4 * groupID + 3] = group.h;
      indices[7 * groupID] = group.NXX - 1;
      indices[7 * groupID + 1] = group.NYY - 1;
      indices[7 * groupID + 2] = group.NZZ - 1;
      indices[7 * groupID + 3] = group.i_min[0];
      indices[7 * groupID + 4] = group.i_min[1];
      indices[7 * groupID + 5] = group.i_min[2];
      indices[7 * groupID + 6] = group.level;
    }

    long long my_size = MyGroups.size(); // total number of groups
    hsize_t offset_groups = 0;
    MPI_Exscan(&my_size, &offset_groups, 1, MPI_LONG_LONG, MPI_SUM, comm);
    hsize_t offset_origins = 4 * offset_groups;
    hsize_t offset_indices = 7 * offset_groups;

    hsize_t count_origins = origins.size();
    hsize_t count_indices = indices.size();
    fspace_origins = H5Dget_space(dataset_origins);
    fspace_indices = H5Dget_space(dataset_indices);
    H5Sselect_hyperslab(fspace_origins, H5S_SELECT_SET, &offset_origins, NULL,
                        &count_origins, NULL);
    H5Sselect_hyperslab(fspace_indices, H5S_SELECT_SET, &offset_indices, NULL,
                        &count_indices, NULL);
    mspace_origins = H5Screate_simple(1, &count_origins, NULL);
    mspace_indices = H5Screate_simple(1, &count_indices, NULL);
    H5Dwrite(dataset_origins, H5T_NATIVE_DOUBLE, mspace_origins, fspace_origins,
             fapl_id, origins.data());
    H5Dwrite(dataset_indices, H5T_NATIVE_INT, mspace_indices, fspace_indices,
             fapl_id, indices.data());

    H5Sclose(mspace_origins);
    H5Sclose(mspace_indices);
    H5Sclose(fspace_origins);
    H5Sclose(fspace_indices);
    H5Dclose(dataset_origins);
    H5Dclose(dataset_indices);
    H5Pclose(fapl_id);
    H5Fclose(file_id);
  }
  // fullpath <<  std::setfill('0') << std::setw(10) << rank; //mike
  // Dump data
  hid_t file_id, dataset_id, fspace_id, fapl_id, mspace_id;
  // hid_t dataset_id_ghost, fspace_id_ghost;

  // 1.Set up file access property list with parallel I/O access
  fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);

  // 2.Create a new file collectively and release property list identifier.
  file_id = H5Fcreate((fullpath.str() + ".h5").c_str(), H5F_ACC_TRUNC,
                      H5P_DEFAULT, fapl_id);
  H5Pclose(fapl_id);

  // 3.Create dataset
  fapl_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(fapl_id, H5FD_MPIO_COLLECTIVE);
  long long total;
  MPI_Allreduce(&start, &total, 1, MPI_LONG_LONG, MPI_SUM, comm);
  // total = start;
  hsize_t dims[1] = {(hsize_t)total};
  fspace_id = H5Screate_simple(1, dims, NULL);
  dataset_id = H5Dcreate(file_id, "dset", get_hdf5_type<float>(), fspace_id,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
  // hsize_t cdims[1];
  // cdims[0] = 8*8*8;
  // H5Pset_chunk(plist_id, 1, cdims);
  // H5Pset_deflate(plist_id, 6);
  // dataset_id = H5Dcreate(file_id, "dset", get_hdf5_type<double>(), fspace_id,
  // H5P_DEFAULT, plist_id, H5P_DEFAULT);

  // 4.Dump
  long long start1 = 0;
  std::vector<float> bigArray(start);
  for (size_t groupID = 0; groupID < MyGroups.size(); groupID++) {
    const BlockGroup &group = MyGroups[groupID];
    const int nX_max = group.NXX - 1;
    const int nY_max = group.NYY - 1;
    const int nZ_max = group.NZZ - 1;
    int dd1 = nX_max * nY_max * nZ_max; // * NCHANNELS;
    std::vector<float> array_block(dd1, 0.0);
    for (int kB = group.i_min[2]; kB <= group.i_max[2]; kB++)
      for (int jB = group.i_min[1]; jB <= group.i_max[1]; jB++)
        for (int iB = group.i_min[0]; iB <= group.i_max[0]; iB++) {
#if DIMENSION == 3
          const long long Z = BlockInfo::forward(group.level, iB, jB, kB);
#else
          const long long Z = BlockInfo::forward(group.level, iB, jB);
#endif
          const cubism::BlockInfo &I = grid.getBlockInfoAll(group.level, Z);
          const auto &lab = *(B *)(I.ptrBlock);
          for (int iz = 0; iz < nZ; iz++)
            for (int iy = 0; iy < nY; iy++)
              for (int ix = 0; ix < nX; ix++) {
                float output[NCHANNELS];
                TStreamer::operate(lab, ix, iy, iz, output);
                const int iz_b = (kB - group.i_min[2]) * nZ + iz;
                const int iy_b = (jB - group.i_min[1]) * nY + iy;
                const int ix_b = (iB - group.i_min[0]) * nX + ix;
                const int base = iz_b * nX_max * nY_max + iy_b * nX_max + ix_b;
                if (NCHANNELS > 1) {
                  output[0] = output[0] * output[0] + output[1] * output[1] +
                              output[2] * output[2];
                  array_block[base] = sqrt(output[0]);
                } else {
                  array_block[base] = output[0];
                }
              }
        }
    for (int j = 0; j < dd1; j++) {
      bigArray[start1 + j] = array_block[j];
    }
    start1 += dd1;
  }
  hsize_t count[1] = {bigArray.size()};

  fspace_id = H5Dget_space(dataset_id);
  mspace_id = H5Screate_simple(1, count, NULL);

  H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, base_tmp, NULL, count, NULL);
  H5Dwrite(dataset_id, get_hdf5_type<float>(), mspace_id, fspace_id, fapl_id,
           bigArray.data());
  H5Sclose(mspace_id);
  H5Sclose(fspace_id);
  H5Dclose(dataset_id);
  H5Pclose(fapl_id);
  H5Fclose(file_id);
  // H5Pclose(plist_id);
  H5close();
}

template <typename TStreamer, typename hdf5Real, typename TGrid>
void ReadHDF5_MPI(TGrid &grid, const std::string &fname,
                  const std::string &dpath = ".") {
  typedef typename TGrid::BlockType B;
  const int nX = B::sizeX;
  const int nY = B::sizeY;
  const int nZ = B::sizeZ;
  const int NCHANNELS = TStreamer::NCHANNELS;
  const int blocksize = nX * nY * nZ * NCHANNELS;

  MPI_Comm comm = grid.getWorldComm();

  // fname is the base filepath tail without file type extension and additional
  // identifiers
  std::ostringstream filename;
  std::ostringstream fullpath;
  filename << fname;
  fullpath << dpath << "/" << filename.str();

  H5open();

  std::vector<long long> blocksZ;
  std::vector<short int> blockslevel;
  std::vector<hdf5Real> data;
  read_buffer_from_file<long long>(blocksZ, comm, fullpath.str() + ".h5",
                                   "blocksZ", 1);
  read_buffer_from_file<short int>(blockslevel, comm, fullpath.str() + ".h5",
                                   "blockslevel", 1);
  read_buffer_from_file<hdf5Real>(data, comm, fullpath.str() + ".h5", "data",
                                  blocksize);

  grid.initialize_blocks(blocksZ, blockslevel);

  std::vector<BlockInfo> &MyInfos = grid.getBlocksInfo();
  for (size_t i = 0; i < MyInfos.size(); i++) {
    const BlockInfo &info = MyInfos[i];
    B &b = *(B *)info.ptrBlock;
    for (int z = 0; z < nZ; z++)
      for (int y = 0; y < nY; y++)
        for (int x = 0; x < nX; x++)
          for (int nc = 0; nc < std::min(NCHANNELS, (int)B::ElementType::DIM);
               nc++) {
            // NCHANNELS > DIM only for 2D vectors, otherwise NCHANNELS=DIM
            b(x, y, z).member(nc) =
                data[(i * nZ * nY * nX + z * nY * nX + y * nX + x) * NCHANNELS +
                     nc];
          }
  }

  H5close();
}

} // namespace cubism

void IC::operator()(const Real dt) {
  const std::vector<cubism::BlockInfo> &chiInfo = sim.chi->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &presInfo = sim.pres->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &poldInfo = sim.pold->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &tmpVInfo = sim.tmpV->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &vOldInfo = sim.vOld->getBlocksInfo();

  if (not sim.bRestart) {
#pragma omp parallel for
    for (size_t i = 0; i < velInfo.size(); i++) {
      VectorBlock &VEL = *(VectorBlock *)velInfo[i].ptrBlock;
      VEL.clear();
      ScalarBlock &CHI = *(ScalarBlock *)chiInfo[i].ptrBlock;
      CHI.clear();
      ScalarBlock &PRES = *(ScalarBlock *)presInfo[i].ptrBlock;
      PRES.clear();
      ScalarBlock &POLD = *(ScalarBlock *)poldInfo[i].ptrBlock;
      POLD.clear();
      ScalarBlock &TMP = *(ScalarBlock *)tmpInfo[i].ptrBlock;
      TMP.clear();
      VectorBlock &TMPV = *(VectorBlock *)tmpVInfo[i].ptrBlock;
      TMPV.clear();
      VectorBlock &VOLD = *(VectorBlock *)vOldInfo[i].ptrBlock;
      VOLD.clear();
    }

    if (sim.smagorinskyCoeff != 0) {
      const std::vector<cubism::BlockInfo> &CsInfo = sim.Cs->getBlocksInfo();
#pragma omp parallel for
      for (size_t i = 0; i < CsInfo.size(); i++) {
        ScalarBlock &CS = *(ScalarBlock *)CsInfo[i].ptrBlock;
        for (int iy = 0; iy < ScalarBlock::sizeY; ++iy)
          for (int ix = 0; ix < ScalarBlock::sizeX; ++ix) {
            CS(ix, iy).s = sim.smagorinskyCoeff;
          }
      }
    }
  } else {
    // create filename from step
    sim.readRestartFiles();

    std::stringstream ss;
    ss << "_" << std::setfill('0') << std::setw(7) << sim.step;

    // The only field that is needed for restarting is velocity. Chi is derived
    // from the files we read for obstacles. Here we also read pres so that the
    // Poisson solver has the same initial guess, which in turn leads to
    // restarted simulations having the exact same result as non-restarted ones
    // (we also read pres because we need to read at least one ScalarGrid, see
    // hack below).
    cubism::ReadHDF5_MPI<cubism::StreamerVector, Real, VectorGrid>(
        *(sim.vel), "vel_" + ss.str(), sim.path4serialization);
    cubism::ReadHDF5_MPI<cubism::StreamerScalar, Real, ScalarGrid>(
        *(sim.pres), "pres_" + ss.str(), sim.path4serialization);

    // hack: need to "read" the other grids too, so that the mesh is the same
    // for every grid. So we read VectorGrids from "vel" and ScalarGrids from
    // "pres". We don't care about the grid point values (those are set to zero
    // below), we only care about the grid structure, i.e. refinement levels
    // etc.
    cubism::ReadHDF5_MPI<cubism::StreamerScalar, Real, ScalarGrid>(
        *(sim.pold), "pres_" + ss.str(), sim.path4serialization);
    cubism::ReadHDF5_MPI<cubism::StreamerScalar, Real, ScalarGrid>(
        *(sim.chi), "pres_" + ss.str(), sim.path4serialization);
    cubism::ReadHDF5_MPI<cubism::StreamerScalar, Real, ScalarGrid>(
        *(sim.tmp), "pres_" + ss.str(), sim.path4serialization);
    cubism::ReadHDF5_MPI<cubism::StreamerVector, Real, VectorGrid>(
        *(sim.tmpV), "vel_" + ss.str(), sim.path4serialization);
    cubism::ReadHDF5_MPI<cubism::StreamerVector, Real, VectorGrid>(
        *(sim.vOld), "vel_" + ss.str(), sim.path4serialization);

#pragma omp parallel for
    for (size_t i = 0; i < velInfo.size(); i++) {
      ScalarBlock &CHI = *(ScalarBlock *)chiInfo[i].ptrBlock;
      CHI.clear();
      ScalarBlock &POLD = *(ScalarBlock *)poldInfo[i].ptrBlock;
      POLD.clear();
      ScalarBlock &TMP = *(ScalarBlock *)tmpInfo[i].ptrBlock;
      TMP.clear();
      VectorBlock &TMPV = *(VectorBlock *)tmpVInfo[i].ptrBlock;
      TMPV.clear();
      VectorBlock &VOLD = *(VectorBlock *)vOldInfo[i].ptrBlock;
      VOLD.clear();
    }

    if (sim.smagorinskyCoeff != 0) {
      cubism::ReadHDF5_MPI<cubism::StreamerScalar, Real, ScalarGrid>(
          *(sim.Cs), "pres_" + ss.str(), sim.path4serialization);
      const std::vector<cubism::BlockInfo> &CsInfo = sim.Cs->getBlocksInfo();
#pragma omp parallel for
      for (size_t i = 0; i < CsInfo.size(); i++) {
        ScalarBlock &CS = *(ScalarBlock *)CsInfo[i].ptrBlock;
        for (int iy = 0; iy < ScalarBlock::sizeY; ++iy)
          for (int ix = 0; ix < ScalarBlock::sizeX; ++ix) {
            CS(ix, iy).s = sim.smagorinskyCoeff;
          }
      }
    }
  }
}

void randomIC::operator()(const Real dt) {
  const std::vector<cubism::BlockInfo> &chiInfo = sim.chi->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &presInfo = sim.pres->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &poldInfo = sim.pold->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &tmpVInfo = sim.tmpV->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &vOldInfo = sim.vOld->getBlocksInfo();

#pragma omp parallel
  {
    std::random_device seed;
    std::mt19937 gen(seed());
    std::normal_distribution<Real> dist(0.0, 0.01);

#pragma omp for
    for (size_t i = 0; i < velInfo.size(); i++) {
      VectorBlock &VEL = *(VectorBlock *)velInfo[i].ptrBlock;
      for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
        for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
          VEL(ix, iy).u[0] = 0.5 + dist(gen);
          VEL(ix, iy).u[1] = 0.5 + dist(gen);
        }

      ScalarBlock &CHI = *(ScalarBlock *)chiInfo[i].ptrBlock;
      CHI.clear();
      ScalarBlock &PRES = *(ScalarBlock *)presInfo[i].ptrBlock;
      PRES.clear();
      ScalarBlock &POLD = *(ScalarBlock *)poldInfo[i].ptrBlock;
      POLD.clear();
      ScalarBlock &TMP = *(ScalarBlock *)tmpInfo[i].ptrBlock;
      TMP.clear();
      VectorBlock &TMPV = *(VectorBlock *)tmpVInfo[i].ptrBlock;
      TMPV.clear();
      VectorBlock &VOLD = *(VectorBlock *)vOldInfo[i].ptrBlock;
      VOLD.clear();
    }
  }

  if (sim.smagorinskyCoeff != 0) {
    const std::vector<cubism::BlockInfo> &CsInfo = sim.Cs->getBlocksInfo();
#pragma omp parallel for
    for (size_t i = 0; i < CsInfo.size(); i++) {
      ScalarBlock &CS = *(ScalarBlock *)CsInfo[i].ptrBlock;
      for (int iy = 0; iy < ScalarBlock::sizeY; ++iy)
        for (int ix = 0; ix < ScalarBlock::sizeX; ++ix) {
          CS(ix, iy).s = sim.smagorinskyCoeff;
        }
    }
  }
}

Real findMaxU::run() const {
  const size_t Nblocks = velInfo.size();

  const Real UINF = sim.uinfx, VINF = sim.uinfy;
///*
#ifdef ZERO_TOTAL_MOM
  Real momX = 0, momY = 0, totM = 0;
#pragma omp parallel for schedule(static) reduction(+ : momX, momY, totM)
  for (size_t i = 0; i < Nblocks; i++) {
    const Real h = velInfo[i].h;
    const VectorBlock &VEL = *(VectorBlock *)velInfo[i].ptrBlock;
    for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        const Real facMom = h * h;
        momX += facMom * VEL(ix, iy).u[0];
        momY += facMom * VEL(ix, iy).u[1];
        totM += facMom;
      }
  }
  Real temp[3] = {momX, momY, totM};
  MPI_Allreduce(MPI_IN_PLACE, temp, 3, MPI_Real, MPI_SUM,
                sim.chi->getWorldComm());
  momX = temp[0];
  momY = temp[1];
  totM = temp[2];
  // printf("Integral of momenta X:%e Y:%e mass:%e\n", momX, momY, totM);
  const Real DU = momX / totM, DV = momY / totM;
#endif
  //*/
  Real U = 0, V = 0, u = 0, v = 0;
#pragma omp parallel for schedule(static) reduction(max : U, V, u, v)
  for (size_t i = 0; i < Nblocks; i++) {
    VectorBlock &VEL = *(VectorBlock *)velInfo[i].ptrBlock;
    for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
#ifdef ZERO_TOTAL_MOM
        VEL(ix, iy).u[0] -= DU;
        VEL(ix, iy).u[1] -= DV;
#endif
        U = std::max(U, std::fabs(VEL(ix, iy).u[0] + UINF));
        V = std::max(V, std::fabs(VEL(ix, iy).u[1] + VINF));
        u = std::max(u, std::fabs(VEL(ix, iy).u[0]));
        v = std::max(v, std::fabs(VEL(ix, iy).u[1]));
      }
  }
  Real quantities[4] = {U, V, u, v};
  MPI_Allreduce(MPI_IN_PLACE, quantities, 4, MPI_Real, MPI_MAX,
                sim.chi->getWorldComm());
  U = quantities[0];
  V = quantities[1];
  u = quantities[2];
  v = quantities[3];
  return std::max({U, V, u, v});
}

void Checker::run(std::string when) const {
  return;
  const size_t Nblocks = velInfo.size();

  const std::vector<cubism::BlockInfo> &presInfo = sim.pres->getBlocksInfo();
  bool bAbort = false;

#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    VectorBlock &VEL = *(VectorBlock *)velInfo[i].ptrBlock;
    ScalarBlock &PRES = *(ScalarBlock *)presInfo[i].ptrBlock;

    for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        if (std::isnan(VEL(ix, iy).u[0])) {
          printf("isnan( VEL(ix,iy).u[0]) %s\n", when.c_str());
          bAbort = true;
          break;
        }
        if (std::isinf(VEL(ix, iy).u[0])) {
          printf("isinf( VEL(ix,iy).u[0]) %s\n", when.c_str());
          bAbort = true;
          break;
        }
        if (std::isnan(VEL(ix, iy).u[1])) {
          printf("isnan( VEL(ix,iy).u[1]) %s\n", when.c_str());
          bAbort = true;
          break;
        }
        if (std::isinf(VEL(ix, iy).u[1])) {
          printf("isinf( VEL(ix,iy).u[1]) %s\n", when.c_str());
          bAbort = true;
          break;
        }
        if (std::isnan(PRES(ix, iy).s)) {
          printf("isnan(PRES(ix,iy).s   ) %s\n", when.c_str());
          bAbort = true;
          break;
        }
        if (std::isinf(PRES(ix, iy).s)) {
          printf("isinf(PRES(ix,iy).s   ) %s\n", when.c_str());
          bAbort = true;
          break;
        }
      }
  }

  if (bAbort) {
    std::cout << "[CUP2D] Detected NaN/INF Field Values. Dumping the field and "
                 "aborting..."
              << std::endl;
    sim.dumpAll("abort_");
    MPI_Abort(sim.comm, 1);
  }
}

void ApplyObjVel::operator()(const Real dt) {
  // We loop over each shape's obstacle blocks and copy the obstacle's
  // deformation velocity UDEF to tmpV.
  // Then, we put that velocity to the grid.

  const size_t Nblocks = velInfo.size();
  const std::vector<cubism::BlockInfo> &chiInfo = sim.chi->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &tmpVInfo = sim.tmpV->getBlocksInfo();
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    ((VectorBlock *)tmpVInfo[i].ptrBlock)->clear();
  }
  for (const auto &shape : sim.shapes) {
    const std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
#pragma omp parallel for
    for (size_t i = 0; i < Nblocks; i++) {
      if (OBLOCK[tmpVInfo[i].blockID] == nullptr)
        continue; // obst not in block
      const UDEFMAT &__restrict__ udef = OBLOCK[tmpVInfo[i].blockID]->udef;
      const CHI_MAT &__restrict__ chi = OBLOCK[tmpVInfo[i].blockID]->chi;
      auto &__restrict__ UDEF = *(VectorBlock *)tmpVInfo[i].ptrBlock; // dest
      const ScalarBlock &__restrict__ CHI = *(ScalarBlock *)chiInfo[i].ptrBlock;
      for (int iy = 0; iy < VectorBlock::sizeY; iy++)
        for (int ix = 0; ix < VectorBlock::sizeX; ix++) {
          if (chi[iy][ix] < CHI(ix, iy).s)
            continue;
          Real p[2];
          tmpVInfo[i].pos(p, ix, iy);
          UDEF(ix, iy).u[0] += udef[iy][ix][0];
          UDEF(ix, iy).u[1] += udef[iy][ix][1];
        }
    }
  }
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < Nblocks; i++) {
    VectorBlock &UF = *(VectorBlock *)velInfo[i].ptrBlock;
    VectorBlock &US = *(VectorBlock *)tmpVInfo[i].ptrBlock;
    ScalarBlock &X = *(ScalarBlock *)chiInfo[i].ptrBlock;
    for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        UF(ix, iy).u[0] =
            UF(ix, iy).u[0] * (1 - X(ix, iy).s) + US(ix, iy).u[0] * X(ix, iy).s;
        UF(ix, iy).u[1] =
            UF(ix, iy).u[1] * (1 - X(ix, iy).s) + US(ix, iy).u[1] * X(ix, iy).s;
      }
  }
}
static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
Real Shape::getCharMass() const { return 0; }
Real Shape::getMaxVel() const { return std::sqrt(u * u + v * v); }

void Shape::updateVelocity(Real dt) {
#ifdef EXPL_INTEGRATE_MOM
  if (not bForcedx || sim.time > timeForced)
    u = (fluidMomX + dt * appliedForceX) / penalM;
  if (not bForcedy || sim.time > timeForced)
    v = (fluidMomY + dt * appliedForceY) / penalM;
  if (not bBlockang || sim.time > timeForced)
    omega = (fluidAngMom + dt * appliedTorque) / penalJ;
#else
  // A and b need to be declared as double (not Real)
  double A[3][3] = {{(double)penalM, (double)0, (double)-penalDY},
                    {(double)0, (double)penalM, (double)penalDX},
                    {(double)-penalDY, (double)penalDX, (double)penalJ}};
  double b[3] = {(double)(fluidMomX + dt * appliedForceX),
                 (double)(fluidMomY + dt * appliedForceY),
                 (double)(fluidAngMom + dt * appliedTorque)};

  if (bForcedx && sim.time < timeForced) {
    A[0][1] = 0;
    A[0][2] = 0;
    b[0] = penalM * forcedu;
  }
  if (bForcedy && sim.time < timeForced) {
    A[1][0] = 0;
    A[1][2] = 0;
    b[1] = penalM * forcedv;
  }
  if (bBlockang && sim.time < timeForced) {
    A[2][0] = 0;
    A[2][1] = 0;
    b[2] = penalJ * forcedomega;
  }

  gsl_matrix_view Agsl = gsl_matrix_view_array(&A[0][0], 3, 3);
  gsl_vector_view bgsl = gsl_vector_view_array(b, 3);
  gsl_vector *xgsl = gsl_vector_alloc(3);
  int sgsl;
  gsl_permutation *permgsl = gsl_permutation_alloc(3);
  gsl_linalg_LU_decomp(&Agsl.matrix, permgsl, &sgsl);
  gsl_linalg_LU_solve(&Agsl.matrix, permgsl, &bgsl.vector, xgsl);

  if (not bForcedx || sim.time > timeForced)
    u = gsl_vector_get(xgsl, 0);
  if (not bForcedy || sim.time > timeForced)
    v = gsl_vector_get(xgsl, 1);
  if (not bBlockang || sim.time > timeForced)
    omega = gsl_vector_get(xgsl, 2);

  const double tStart = breakSymmetryTime;
  const bool shouldBreak = (sim.time > tStart && sim.time < tStart + 1.0);
  if (breakSymmetryType != 0 && shouldBreak) {
    const double strength = breakSymmetryStrength;
    const double charL = getCharLength();
    const double charV = std::abs(u);

    // Set magintude of disturbance
    if (breakSymmetryType == 1) { // add rotation
      omega = strength * charV * charL * sin(2 * M_PI * (sim.time - tStart));
    }
    if (breakSymmetryType == 2) { // add translation
      v = strength * charV * sin(2 * M_PI * (sim.time - tStart));
    }
  }

  gsl_permutation_free(permgsl);
  gsl_vector_free(xgsl);
#endif
}

void Shape::updateLabVelocity(int nSum[2], Real uSum[2]) {
  if (bFixedx) {
    (nSum[0])++;
    uSum[0] -= u;
  }
  if (bFixedy) {
    (nSum[1])++;
    uSum[1] -= v;
  }
}

void Shape::updatePosition(Real dt) {
  // Remember, uinf is -ubox, therefore we sum it to u body to get
  // velocity of shapre relative to the sim box
  centerOfMass[0] += dt * (u + sim.uinfx);
  centerOfMass[1] += dt * (v + sim.uinfy);
  labCenterOfMass[0] += dt * u;
  labCenterOfMass[1] += dt * v;

  orientation += dt * omega;
  orientation = orientation > M_PI ? orientation - 2 * M_PI : orientation;
  orientation = orientation < -M_PI ? orientation + 2 * M_PI : orientation;

  const Real cosang = std::cos(orientation), sinang = std::sin(orientation);

  center[0] = centerOfMass[0] + cosang * d_gm[0] - sinang * d_gm[1];
  center[1] = centerOfMass[1] + sinang * d_gm[0] + cosang * d_gm[1];

  const Real CX = labCenterOfMass[0], CY = labCenterOfMass[1], t = sim.time;
  const Real cx = centerOfMass[0], cy = centerOfMass[1], angle = orientation;

  // do not print/write for initial PutObjectOnGrid
  if (dt <= 0)
    return;

  if (not sim.muteAll && sim.rank == 0) {
    printf("CM:[%.02f %.02f] C:[%.02f %.02f] ang:%.02f u:%.05f v:%.05f av:%.03f"
           " M:%.02e J:%.02e\n",
           (double)cx, (double)cy, (double)center[0], (double)center[1],
           (double)angle, (double)u, (double)v, (double)omega, (double)M,
           (double)J);
    std::stringstream ssF;
    ssF << sim.path2file << "/velocity_" << obstacleID << ".dat";
    std::stringstream &fout = logger.get_stream(ssF.str());
    if (sim.step == 0)
      fout << "t dt CXsim CYsim CXlab CYlab angle u v omega M J accx accy "
              "accw\n";

    fout << t << " " << dt << " " << cx << " " << cy << " " << CX << " " << CY
         << " " << angle << " " << u << " " << v << " " << omega << " " << M
         << " " << J << " " << fluidMomX / penalM << " " << fluidMomY / penalM
         << " " << fluidAngMom / penalJ << "\n";
  }
}

Shape::Integrals
Shape::integrateObstBlock(const std::vector<cubism::BlockInfo> &vInfo) {
  Real _x = 0, _y = 0, _m = 0, _j = 0, _u = 0, _v = 0, _a = 0;
#pragma omp parallel for schedule(dynamic, 1)                                  \
    reduction(+ : _x, _y, _m, _j, _u, _v, _a)
  for (size_t i = 0; i < vInfo.size(); i++) {
    const Real hsq = std::pow(vInfo[i].h, 2);
    const auto pos = obstacleBlocks[vInfo[i].blockID];
    if (pos == nullptr)
      continue;
    const CHI_MAT &__restrict__ CHI = pos->chi;
    const UDEFMAT &__restrict__ UDEF = pos->udef;
    for (int iy = 0; iy < ObstacleBlock::sizeY; ++iy)
      for (int ix = 0; ix < ObstacleBlock::sizeX; ++ix) {
        if (CHI[iy][ix] <= 0)
          continue;
        Real p[2];
        vInfo[i].pos(p, ix, iy);
        const Real chi = CHI[iy][ix] * hsq;
        p[0] -= centerOfMass[0];
        p[1] -= centerOfMass[1];
        _x += chi * p[0];
        _y += chi * p[1];
        _m += chi;
        _j += chi * (p[0] * p[0] + p[1] * p[1]);
        _u += chi * UDEF[iy][ix][0];
        _v += chi * UDEF[iy][ix][1];
        _a += chi * (p[0] * UDEF[iy][ix][1] - p[1] * UDEF[iy][ix][0]);
      }
  }
  Real quantities[7] = {_x, _y, _m, _j, _u, _v, _a};
  MPI_Allreduce(MPI_IN_PLACE, quantities, 7, MPI_Real, MPI_SUM,
                sim.chi->getWorldComm());
  _x = quantities[0];
  _y = quantities[1];
  _m = quantities[2];
  _j = quantities[3];
  _u = quantities[4];
  _v = quantities[5];
  _a = quantities[6];
  _u /= _m;
  _v /= _m;
  _a /= _j;
  return Integrals(_x, _y, _m, _j, _u, _v, _a);
}

void Shape::removeMoments(const std::vector<cubism::BlockInfo> &vInfo) {
  Shape::Integrals I = integrateObstBlock(vInfo);
  M = I.m;
  J = I.j;

  // with current center put shape on grid, with current shape on grid we
  // updated the center of mass, now recompute the distance betweeen the two:
  const Real dCx = center[0] - centerOfMass[0];
  const Real dCy = center[1] - centerOfMass[1];
  d_gm[0] = dCx * std::cos(orientation) + dCy * std::sin(orientation);
  d_gm[1] = -dCx * std::sin(orientation) + dCy * std::cos(orientation);

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < vInfo.size(); i++) {
    const auto pos = obstacleBlocks[vInfo[i].blockID];
    if (pos == nullptr)
      continue;

    for (int iy = 0; iy < ObstacleBlock::sizeY; ++iy)
      for (int ix = 0; ix < ObstacleBlock::sizeX; ++ix) {
        Real p[2];
        vInfo[i].pos(p, ix, iy);
        p[0] -= centerOfMass[0];
        p[1] -= centerOfMass[1];
        pos->udef[iy][ix][0] -= I.u - I.a * p[1];
        pos->udef[iy][ix][1] -= I.v + I.a * p[0];
      }
  }
};

void Shape::diagnostics() {
  /*
  const std::vector<cubism::BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  const Real hsq = std::pow(vInfo[0].h, 2);
  Real _a=0, _m=0, _x=0, _y=0, _t=0;
  #pragma omp parallel for schedule(dynamic) reduction(+:_a,_m,_x,_y,_t)
  for(size_t i=0; i<vInfo.size(); i++) {
      const auto pos = obstacleBlocks[vInfo[i].blockID];
      if(pos == nullptr) continue;
      FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        if (pos->chi[iy][ix] <= 0) continue;
        const Real Xs = pos->chi[iy][ix] * hsq;
        Real p[2];
        vInfo[i].pos(p, ix, iy);
        p[0] -= centerOfMass[0];
        p[1] -= centerOfMass[1];
        const Real*const udef = pos->udef[iy][ix];
        const Real uDiff = b(ix,iy).u - (u -omega*p[1] +udef[0]);
        const Real vDiff = b(ix,iy).v - (v +omega*p[0] +udef[1]);
        _a += Xs;
        _m += Xs;
        _x += uDiff*Xs;
        _y += vDiff*Xs;
        _t += (p[0]*vDiff-p[1]*uDiff)*Xs;
      }
  }
  area_penal   = _a;
  mass_penal   = _m;
  forcex_penal = _x * sim.lambda;
  forcey_penal = _y * sim.lambda;
  torque_penal = _t * sim.lambda;
  */
}

void Shape::computeForces() {
  // additive quantities:
  perimeter = 0;
  forcex = 0;
  forcey = 0;
  forcex_P = 0;
  forcey_P = 0;
  forcex_V = 0;
  forcey_V = 0;
  torque = 0;
  torque_P = 0;
  torque_V = 0;
  drag = 0;
  thrust = 0;
  lift = 0;
  Pout = 0;
  PoutNew = 0;
  PoutBnd = 0;
  defPower = 0;
  defPowerBnd = 0;
  circulation = 0;

  for (auto &block : obstacleBlocks)
    if (block not_eq nullptr) {
      circulation += block->circulation;
      perimeter += block->perimeter;
      torque += block->torque;
      forcex += block->forcex;
      forcey += block->forcey;
      forcex_P += block->forcex_P;
      forcey_P += block->forcey_P;
      forcex_V += block->forcex_V;
      forcey_V += block->forcey_V;
      torque_P += block->torque_P;
      torque_V += block->torque_V;
      drag += block->drag;
      thrust += block->thrust;
      lift += block->lift;
      Pout += block->Pout;
      PoutNew += block->PoutNew;
      defPowerBnd += block->defPowerBnd;
      PoutBnd += block->PoutBnd;
      defPower += block->defPower;
    }
  Real quantities[19];
  quantities[0] = circulation;
  quantities[1] = perimeter;
  quantities[2] = forcex;
  quantities[3] = forcex_P;
  quantities[4] = forcex_V;
  quantities[5] = torque_P;
  quantities[6] = drag;
  quantities[7] = lift;
  quantities[8] = Pout;
  quantities[9] = PoutNew;
  quantities[10] = PoutBnd;
  quantities[11] = torque;
  quantities[12] = forcey;
  quantities[13] = forcey_P;
  quantities[14] = forcey_V;
  quantities[15] = torque_V;
  quantities[16] = thrust;
  quantities[17] = defPowerBnd;
  quantities[18] = defPower;
  MPI_Allreduce(MPI_IN_PLACE, quantities, 19, MPI_Real, MPI_SUM,
                sim.chi->getWorldComm());
  circulation = quantities[0];
  perimeter = quantities[1];
  forcex = quantities[2];
  forcex_P = quantities[3];
  forcex_V = quantities[4];
  torque_P = quantities[5];
  drag = quantities[6];
  lift = quantities[7];
  Pout = quantities[8];
  PoutNew = quantities[9];
  PoutBnd = quantities[10];
  torque = quantities[11];
  forcey = quantities[12];
  forcey_P = quantities[13];
  forcey_V = quantities[14];
  torque_V = quantities[15];
  thrust = quantities[16];
  defPowerBnd = quantities[17];
  defPower = quantities[18];

  // derived quantities:
  Pthrust = thrust * std::sqrt(u * u + v * v);
  Pdrag = drag * std::sqrt(u * u + v * v);
  const Real denUnb = Pthrust - std::min(defPower, (Real)0);
  const Real demBnd = Pthrust - defPowerBnd;
  EffPDef = Pthrust / std::max(denUnb, EPS);
  EffPDefBnd = Pthrust / std::max(demBnd, EPS);

  if (sim.dt <= 0)
    return;

  if (not sim.muteAll && sim._bDump && bDumpSurface) {
    std::stringstream s;
    if (sim.rank == 0)
      s << "x,y,p,u,v,nx,ny,omega,uDef,vDef,fX,fY,fXv,fYv\n";
    for (auto &block : obstacleBlocks)
      if (block not_eq nullptr)
        block->fill_stringstream(s);
    std::string st = s.str();
    MPI_Offset offset = 0;
    MPI_Offset len = st.size() * sizeof(char);
    MPI_File surface_file;
    std::stringstream ssF;
    ssF << sim.path2file << "/surface_" << obstacleID << "_"
        << std::setfill('0') << std::setw(7) << sim.step << ".csv";
    MPI_File_delete(ssF.str().c_str(),
                    MPI_INFO_NULL); // delete the file if it exists
    MPI_File_open(sim.chi->getWorldComm(), ssF.str().c_str(),
                  MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL,
                  &surface_file);
    MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, sim.chi->getWorldComm());
    MPI_File_write_at_all(surface_file, offset, st.data(), st.size(), MPI_CHAR,
                          MPI_STATUS_IGNORE);
    MPI_File_close(&surface_file);
  }

  int tot_blocks = 0;
  int nb = (int)sim.chi->getBlocksInfo().size();
  MPI_Reduce(&nb, &tot_blocks, 1, MPI_INT, MPI_SUM, 0, sim.chi->getWorldComm());
  if (not sim.muteAll && sim.rank == 0) {
    std::stringstream ssF, ssP;
    ssF << sim.path2file << "/forceValues_" << obstacleID << ".dat";
    ssP << sim.path2file << "/powerValues_" << obstacleID << ".dat";

    std::stringstream &fileForce = logger.get_stream(ssF.str());
    if (sim.step == 0)
      fileForce << "time Fx Fy FxPres FyPres FxVisc FyVisc tau tauPres tauVisc "
                   "drag thrust lift perimeter circulation blocks\n";

    fileForce << sim.time << " " << forcex << " " << forcey << " " << forcex_P
              << " " << forcey_P << " " << forcex_V << " " << forcey_V << " "
              << torque << " " << torque_P << " " << torque_V << " " << drag
              << " " << thrust << " " << lift << " " << perimeter << " "
              << circulation << " " << tot_blocks << "\n";

    std::stringstream &filePower = logger.get_stream(ssP.str());
    if (sim.step == 0)
      filePower << "time Pthrust Pdrag PoutBnd Pout PoutNew defPowerBnd "
                   "defPower EffPDefBnd EffPDef\n";
    filePower << sim.time << " " << Pthrust << " " << Pdrag << " " << PoutBnd
              << " " << Pout << " " << PoutNew << " " << defPowerBnd << " "
              << defPower << " " << EffPDefBnd << " " << EffPDef << "\n";
  }
}

Shape::Shape(SimulationData &s, cubism::ArgumentParser &p, Real C[2])
    : sim(s), origC{C[0], C[1]}, origAng(p("-angle").asDouble(0) * M_PI / 180),
      center{C[0], C[1]}, centerOfMass{C[0], C[1]}, orientation(origAng),
      bFixed(p("-bFixed").asBool(false)), bFixedx(p("-bFixedx").asBool(bFixed)),
      bFixedy(p("-bFixedy").asBool(bFixed)),
      bForced(p("-bForced").asBool(false)),
      bForcedx(p("-bForcedx").asBool(bForced)),
      bForcedy(p("-bForcedy").asBool(bForced)),
      bBlockang(p("-bBlockAng").asBool(bForcedx || bForcedy)),
      forcedu(-p("-xvel").asDouble(0)), forcedv(-p("-yvel").asDouble(0)),
      forcedomega(-p("-angvel").asDouble(0)),
      bDumpSurface(p("-dumpSurf").asInt(0)),
      timeForced(p("-timeForced").asDouble(std::numeric_limits<Real>::max())),
      breakSymmetryType(
          p("-breakSymmetryType").asInt(0)), // 0 is no symmetry breaking
      breakSymmetryStrength(p("-breakSymmetryStrength").asDouble(0.1)),
      breakSymmetryTime(p("-breakSymmetryTime").asDouble(1.0)) {}

Shape::~Shape() {
  for (auto &entry : obstacleBlocks)
    delete entry;
  obstacleBlocks.clear();
}

// functions needed for restarting the simulation
void Shape::saveRestart(FILE *f) {
  assert(f != NULL);
  fprintf(f, "x:     %20.20e\n", (double)centerOfMass[0]);
  fprintf(f, "y:     %20.20e\n", (double)centerOfMass[1]);
  fprintf(f, "xlab:  %20.20e\n", (double)labCenterOfMass[0]);
  fprintf(f, "ylab:  %20.20e\n", (double)labCenterOfMass[1]);
  fprintf(f, "u:     %20.20e\n", (double)u);
  fprintf(f, "v:     %20.20e\n", (double)v);
  fprintf(f, "omega: %20.20e\n", (double)omega);
  fprintf(f, "orientation: %20.20e\n", (double)orientation);
  fprintf(f, "d_gm0: %20.20e\n", (double)d_gm[0]);
  fprintf(f, "d_gm1: %20.20e\n", (double)d_gm[1]);
  fprintf(f, "center0: %20.20e\n", (double)center[0]);
  fprintf(f, "center1: %20.20e\n", (double)center[1]);
  // maybe center0,center1,d_gm0,d_gm1 are not all needed, but it's only four
  // numbers so we might as well dump them
}

void Shape::loadRestart(FILE *f) {
  assert(f != NULL);
  bool ret = true;
  double in_centerOfMass0, in_centerOfMass1, in_labCenterOfMass0,
      in_labCenterOfMass1, in_u, in_v, in_omega, in_orientation, in_d_gm0,
      in_d_gm1, in_center0, in_center1;
  ret = ret && 1 == fscanf(f, "x:     %le\n", &in_centerOfMass0);
  ret = ret && 1 == fscanf(f, "y:     %le\n", &in_centerOfMass1);
  ret = ret && 1 == fscanf(f, "xlab:  %le\n", &in_labCenterOfMass0);
  ret = ret && 1 == fscanf(f, "ylab:  %le\n", &in_labCenterOfMass1);
  ret = ret && 1 == fscanf(f, "u:     %le\n", &in_u);
  ret = ret && 1 == fscanf(f, "v:     %le\n", &in_v);
  ret = ret && 1 == fscanf(f, "omega: %le\n", &in_omega);
  ret = ret && 1 == fscanf(f, "orientation: %le\n", &in_orientation);
  ret = ret && 1 == fscanf(f, "d_gm0: %le\n", &in_d_gm0);
  ret = ret && 1 == fscanf(f, "d_gm1: %le\n", &in_d_gm1);
  ret = ret && 1 == fscanf(f, "center0: %le\n", &in_center0);
  ret = ret && 1 == fscanf(f, "center1: %le\n", &in_center1);
  if ((not ret)) {
    printf("Error reading restart file. Aborting...\n");
    fflush(0);
    abort();
  }
  centerOfMass[0] = in_centerOfMass0;
  centerOfMass[1] = in_centerOfMass1;
  labCenterOfMass[0] = in_labCenterOfMass0;
  labCenterOfMass[1] = in_labCenterOfMass1;
  u = in_u;
  v = in_v;
  omega = in_omega;
  orientation = in_orientation;
  d_gm[0] = in_d_gm0;
  d_gm[1] = in_d_gm1;
  center[0] = in_center0;
  center[1] = in_center1;
  if (sim.rank == 0)
    printf("Restarting Object.. x: %le, y: %le, xlab: %le, ylab: %le, u: %le, "
           "v: %le, omega: %le\n",
           (double)centerOfMass[0], (double)centerOfMass[1],
           (double)labCenterOfMass[0], (double)labCenterOfMass[1], (double)u,
           (double)v, (double)omega);
}

void SimulationData::addShape(std::shared_ptr<Shape> shape) {
  shape->obstacleID = (unsigned)shapes.size();
  shapes.push_back(std::move(shape));
}

void SimulationData::resetAll() {
  for (const auto &shape : shapes)
    shape->resetAll();
  time = 0;
  step = 0;
  uinfx = 0;
  uinfy = 0;
  nextDumpTime = 0;
  _bDump = false;
  bCollision = false;
}

void SimulationData::allocateGrid() {
  ScalarLab dummy;
  const bool xperiodic = dummy.is_xperiodic();
  const bool yperiodic = dummy.is_yperiodic();
  const bool zperiodic = dummy.is_zperiodic();

  chi = new ScalarGrid(bpdx, bpdy, 1, extent, levelStart, levelMax, comm,
                       xperiodic, yperiodic, zperiodic);
  vel = new VectorGrid(bpdx, bpdy, 1, extent, levelStart, levelMax, comm,
                       xperiodic, yperiodic, zperiodic);
  vOld = new VectorGrid(bpdx, bpdy, 1, extent, levelStart, levelMax, comm,
                        xperiodic, yperiodic, zperiodic);
  pres = new ScalarGrid(bpdx, bpdy, 1, extent, levelStart, levelMax, comm,
                        xperiodic, yperiodic, zperiodic);
  tmpV = new VectorGrid(bpdx, bpdy, 1, extent, levelStart, levelMax, comm,
                        xperiodic, yperiodic, zperiodic);
  tmp = new ScalarGrid(bpdx, bpdy, 1, extent, levelStart, levelMax, comm,
                       xperiodic, yperiodic, zperiodic);
  pold = new ScalarGrid(bpdx, bpdy, 1, extent, levelStart, levelMax, comm,
                        xperiodic, yperiodic, zperiodic);

  // For RL SGS learning
  if (smagorinskyCoeff != 0)
    Cs = new ScalarGrid(bpdx, bpdy, 1, extent, levelStart, levelMax, comm,
                        xperiodic, yperiodic, zperiodic);

  const std::vector<cubism::BlockInfo> &velInfo = vel->getBlocksInfo();

  if (velInfo.size() == 0) {
    std::cout << "You are using too many MPI ranks for the given initial "
                 "number of blocks.";
    std::cout << "Either increase levelStart or reduce the number of ranks."
              << std::endl;
    MPI_Abort(chi->getWorldComm(), 1);
  }
  // Compute extents, assume all blockinfos have same h at the start!!!
  int aux = pow(2, levelStart);
  extents[0] = aux * bpdx * velInfo[0].h * VectorBlock::sizeX;
  extents[1] = aux * bpdy * velInfo[0].h * VectorBlock::sizeY;
  // printf("Extents %e %e (%e)\n", extents[0], extents[1], extent);

  // compute min and max gridspacing for set AMR parameter
  int auxMax = pow(2, levelMax - 1);
  minH = extents[0] / (auxMax * bpdx * VectorBlock::sizeX);
  maxH = extents[0] / (bpdx * VectorBlock::sizeX);
}

void SimulationData::dumpChi(std::string name) {
  std::stringstream ss;
  ss << name << std::setfill('0') << std::setw(7) << step;
  cubism::DumpHDF5_MPI<cubism::StreamerScalar, Real>(
      *chi, time, "chi_" + ss.str(), path4serialization);
}
void SimulationData::dumpPres(std::string name) {
  std::stringstream ss;
  ss << name << std::setfill('0') << std::setw(7) << step;
  cubism::DumpHDF5_MPI<cubism::StreamerScalar, Real>(
      *pres, time, "pres_" + ss.str(), path4serialization);
}
void SimulationData::dumpPold(std::string name) {
  std::stringstream ss;
  ss << name << std::setfill('0') << std::setw(7) << step;
  cubism::DumpHDF5_MPI<cubism::StreamerScalar, Real>(
      *pold, time, "pold_" + ss.str(), path4serialization);
}
void SimulationData::dumpTmp(std::string name) {
  std::stringstream ss;
  ss << name << std::setfill('0') << std::setw(7) << step;
  cubism::DumpHDF5_MPI<cubism::StreamerScalar, Real>(
      *tmp, time, "tmp_" + ss.str(), path4serialization);
}
void SimulationData::dumpVel(std::string name) {
  std::stringstream ss;
  ss << name << std::setfill('0') << std::setw(7) << step;
  cubism::DumpHDF5_MPI<cubism::StreamerVector, Real>(
      *(vel), time, "vel_" + ss.str(), path4serialization);
}
void SimulationData::dumpVold(std::string name) {
  std::stringstream ss;
  ss << name << std::setfill('0') << std::setw(7) << step;
  cubism::DumpHDF5_MPI<cubism::StreamerVector, Real>(
      *(vOld), time, "vOld_" + ss.str(), path4serialization);
}
void SimulationData::dumpTmpV(std::string name) {
  std::stringstream ss;
  ss << name << std::setfill('0') << std::setw(7) << step;
  cubism::DumpHDF5_MPI<cubism::StreamerVector, Real>(
      *(tmpV), time, "tmpV_" + ss.str(), path4serialization);
}
void SimulationData::dumpCs(std::string name) {
  std::stringstream ss;
  ss << name << std::setfill('0') << std::setw(7) << step;
  cubism::DumpHDF5_MPI<cubism::StreamerScalar, Real>(
      *(Cs), time, "Cs_" + ss.str(), path4serialization);
}

void SimulationData::registerDump() { nextDumpTime += dumpTime; }

SimulationData::SimulationData() = default;

SimulationData::~SimulationData() {
  delete profiler;
  if (vel not_eq nullptr)
    delete vel;
  if (chi not_eq nullptr)
    delete chi;
  if (pres not_eq nullptr)
    delete pres;
  if (pold not_eq nullptr)
    delete pold;
  if (vOld not_eq nullptr)
    delete vOld;
  if (tmpV not_eq nullptr)
    delete tmpV;
  if (tmp not_eq nullptr)
    delete tmp;
  if (Cs not_eq nullptr)
    delete Cs;
}

bool SimulationData::bOver() const {
  const bool timeEnd = endTime > 0 && time >= endTime;
  const bool stepEnd = nsteps > 0 && step >= nsteps;
  return timeEnd || stepEnd;
}

bool SimulationData::bDump() {
  const bool timeDump = dumpTime > 0 && time >= nextDumpTime;
  const bool stepDump = dumpFreq > 0 && (step % dumpFreq) == 0;
  _bDump = stepDump || timeDump;
  return _bDump;
}

void SimulationData::startProfiler(std::string name) {
#ifndef NDEBUG
  Checker check(*this);
  check.run("before" + name);
#endif
  profiler->push_start(name);
}

void SimulationData::stopProfiler() {
  // Checker check (*this);
  // check.run("after" + profiler->currentAgentName());
  profiler->pop_stop();
}

void SimulationData::printResetProfiler() {
  profiler->printSummary();
  profiler->reset();
}

void SimulationData::dumpAll(std::string name) {
  startProfiler("Dump");

  auto K1 = computeVorticity(*this);
  K1(0);
  dumpTmp(name); // dump vorticity
  dumpChi(name);
  dumpVel(name);
  dumpPres(name);
  // dumpPold(name);
  // dumpTmpV(name);
  // dumpVold(name);
  if (bDumpCs)
    dumpCs(name);

  writeRestartFiles();

  stopProfiler();
}

void SimulationData::writeRestartFiles() {

  // write restart file for field
  if (rank == 0) {
    std::stringstream ssR;
    ssR << path4serialization + "/field.restart";
    FILE *fField = fopen(ssR.str().c_str(), "w");
    if (fField == NULL) {
      printf("Could not write %s. Aborting...\n", "field.restart");
      fflush(0);
      abort();
    }
    assert(fField != NULL);
    fprintf(fField, "time: %20.20e\n", (double)time);
    fprintf(fField, "stepid: %d\n", step);
    fprintf(fField, "uinfx: %20.20e\n", (double)uinfx);
    fprintf(fField, "uinfy: %20.20e\n", (double)uinfy);
    fprintf(fField, "dt: %20.20e\n", (double)dt);
    fclose(fField);
  }

  // write restart file for shapes
  {
    int size;
    MPI_Comm_size(comm, &size);
    const size_t tasks = shapes.size();
    size_t my_share = tasks / size;
    if (tasks % size != 0 && rank == size - 1) // last rank gets what's left
    {
      my_share += tasks % size;
    }
    const size_t my_start = rank * (tasks / size);
    const size_t my_end = my_start + my_share;

#pragma omp parallel for schedule(static, 1)
    for (size_t j = my_start; j < my_end; j++) {
      auto &shape = shapes[j];
      std::stringstream ssR;
      ssR << path4serialization + "/shape_" << shape->obstacleID << ".restart";
      FILE *fShape = fopen(ssR.str().c_str(), "w");
      if (fShape == NULL) {
        printf("Could not write %s. Aborting...\n", ssR.str().c_str());
        fflush(0);
        abort();
      }
      shape->saveRestart(fShape);
      fclose(fShape);
    }
  }
}

void SimulationData::readRestartFiles() {
  // read restart file for field
  FILE *fField = fopen("field.restart", "r");
  if (fField == NULL) {
    printf("Could not read %s. Aborting...\n", "field.restart");
    fflush(0);
    abort();
  }
  assert(fField != NULL);
  if (rank == 0 && verbose)
    printf("Reading %s...\n", "field.restart");
  bool ret = true;
  double in_time, in_uinfx, in_uinfy, in_dt;
  ret = ret && 1 == fscanf(fField, "time: %le\n", &in_time);
  ret = ret && 1 == fscanf(fField, "stepid: %d\n", &step);
  ret = ret && 1 == fscanf(fField, "uinfx: %le\n", &in_uinfx);
  ret = ret && 1 == fscanf(fField, "uinfy: %le\n", &in_uinfy);
  ret = ret && 1 == fscanf(fField, "dt: %le\n", &in_dt);
  time = (Real)in_time;
  uinfx = (Real)in_uinfx;
  uinfy = (Real)in_uinfy;
  dt = (Real)in_dt;
  fclose(fField);
  if ((not ret) || step < 0 || time < 0) {
    printf("Error reading restart file. Aborting...\n");
    fflush(0);
    abort();
  }
  if (rank == 0 && verbose)
    printf("Restarting flow.. time: %le, stepid: %d, uinfx: %le, uinfy: %le\n",
           (double)time, step, (double)uinfx, (double)uinfy);
  nextDumpTime = time + dumpTime;

  // read restart file for shapes
  for (std::shared_ptr<Shape> shape : shapes) {
    std::stringstream ssR;
    ssR << "shape_" << shape->obstacleID << ".restart";
    FILE *fShape = fopen(ssR.str().c_str(), "r");
    if (fShape == NULL) {
      printf("Could not read %s. Aborting...\n", ssR.str().c_str());
      fflush(0);
      abort();
    }
    if (rank == 0 && verbose)
      printf("Reading %s...\n", ssR.str().c_str());
    shape->loadRestart(fShape);
    fclose(fShape);
  }
}

struct IF2D_Frenet2D {
  static void solve(const unsigned Nm, const Real *const rS,
                    const Real *const curv, const Real *const curv_dt,
                    Real *const rX, Real *const rY, Real *const vX,
                    Real *const vY, Real *const norX, Real *const norY,
                    Real *const vNorX, Real *const vNorY) {
    // initial conditions
    rX[0] = 0.0;
    rY[0] = 0.0;
    norX[0] = 0.0;
    norY[0] = 1.0;
    Real ksiX = 1.0;
    Real ksiY = 0.0;
    // velocity variables
    vX[0] = 0.0;
    vY[0] = 0.0;
    vNorX[0] = 0.0;
    vNorY[0] = 0.0;
    Real vKsiX = 0.0;
    Real vKsiY = 0.0;

    for (unsigned i = 1; i < Nm; i++) {
      // compute derivatives positions
      const Real dksiX = curv[i - 1] * norX[i - 1];
      const Real dksiY = curv[i - 1] * norY[i - 1];
      const Real dnuX = -curv[i - 1] * ksiX;
      const Real dnuY = -curv[i - 1] * ksiY;
      // compute derivatives velocity
      const Real dvKsiX =
          curv_dt[i - 1] * norX[i - 1] + curv[i - 1] * vNorX[i - 1];
      const Real dvKsiY =
          curv_dt[i - 1] * norY[i - 1] + curv[i - 1] * vNorY[i - 1];
      const Real dvNuX = -curv_dt[i - 1] * ksiX - curv[i - 1] * vKsiX;
      const Real dvNuY = -curv_dt[i - 1] * ksiY - curv[i - 1] * vKsiY;
      // compute current ds
      const Real ds = rS[i] - rS[i - 1];
      // update
      rX[i] = rX[i - 1] + ds * ksiX;
      rY[i] = rY[i - 1] + ds * ksiY;
      norX[i] = norX[i - 1] + ds * dnuX;
      norY[i] = norY[i - 1] + ds * dnuY;
      ksiX += ds * dksiX;
      ksiY += ds * dksiY;
      // update velocities
      vX[i] = vX[i - 1] + ds * vKsiX;
      vY[i] = vY[i - 1] + ds * vKsiY;
      vNorX[i] = vNorX[i - 1] + ds * dvNuX;
      vNorY[i] = vNorY[i - 1] + ds * dvNuY;
      vKsiX += ds * dvKsiX;
      vKsiY += ds * dvKsiY;
      // normalize unit vectors
      const Real d1 = ksiX * ksiX + ksiY * ksiY;
      const Real d2 = norX[i] * norX[i] + norY[i] * norY[i];
      if (d1 > std::numeric_limits<Real>::epsilon()) {
        const Real normfac = 1 / std::sqrt(d1);
        ksiX *= normfac;
        ksiY *= normfac;
      }
      if (d2 > std::numeric_limits<Real>::epsilon()) {
        const Real normfac = 1 / std::sqrt(d2);
        norX[i] *= normfac;
        norY[i] *= normfac;
      }
    }
  }
};

class IF2D_Interpolation1D {
public:
  static void naturalCubicSpline(const Real *x, const Real *y, const unsigned n,
                                 const Real *xx, Real *yy, const unsigned nn) {
    return naturalCubicSpline(x, y, n, xx, yy, nn, 0);
  }

  static void naturalCubicSpline(const Real *x, const Real *y, const unsigned n,
                                 const Real *xx, Real *yy, const unsigned nn,
                                 const Real offset) {
    std::vector<Real> y2(n), u(n - 1);

    y2[0] = 0;
    u[0] = 0;
    for (unsigned i = 1; i < n - 1; i++) {
      const Real sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
      const Real p = sig * y2[i - 1] + 2;
      y2[i] = (sig - 1) / p;
      u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) -
             (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
      u[i] = (6 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
    }

    const Real qn = 0;
    const Real un = 0;
    y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1);

    for (unsigned k = n - 2; k > 0; k--)
      y2[k] = y2[k] * y2[k + 1] + u[k];

    // #pragma omp parallel for schedule(static)
    for (unsigned j = 0; j < nn; j++) {
      unsigned int klo = 0;
      unsigned int khi = n - 1;
      unsigned int k = 0;
      while (khi - klo > 1) {
        k = (khi + klo) >> 1;
        if (x[k] > (xx[j] + offset))
          khi = k;
        else
          klo = k;
      }

      const Real h = x[khi] - x[klo];
      if (h <= 0.0) {
        std::cout << "Interpolation points must be distinct!" << std::endl;
        abort();
      }
      const Real a = (x[khi] - (xx[j] + offset)) / h;
      const Real b = ((xx[j] + offset) - x[klo]) / h;
      yy[j] =
          a * y[klo] + b * y[khi] +
          ((a * a * a - a) * y2[klo] + (b * b * b - b) * y2[khi]) * (h * h) / 6;
    }
  }

  static void cubicInterpolation(const Real x0, const Real x1, const Real x,
                                 const Real y0, const Real y1, const Real dy0,
                                 const Real dy1, Real &y, Real &dy) {
    const Real xrel = (x - x0);
    const Real deltax = (x1 - x0);

    const Real a = (dy0 + dy1) / (deltax * deltax) -
                   2 * (y1 - y0) / (deltax * deltax * deltax);
    const Real b =
        (-2 * dy0 - dy1) / deltax + 3 * (y1 - y0) / (deltax * deltax);
    const Real c = dy0;
    const Real d = y0;

    y = a * xrel * xrel * xrel + b * xrel * xrel + c * xrel + d;
    dy = 3 * a * xrel * xrel + 2 * b * xrel + c;
  }

  static void cubicInterpolation(const Real x0, const Real x1, const Real x,
                                 const Real y0, const Real y1, Real &y,
                                 Real &dy) {
    return cubicInterpolation(x0, x1, x, y0, y1, 0, 0, y,
                              dy); // 0 slope at end points
  }

  static void linearInterpolation(const Real x0, const Real x1, const Real x,
                                  const Real y0, const Real y1, Real &y,
                                  Real &dy) {
    y = (y1 - y0) / (x1 - x0) * (x - x0) + y0;
    dy = (y1 - y0) / (x1 - x0);
  }
};

namespace Schedulers {
template <int Npoints> struct ParameterScheduler {
  static constexpr int npoints = Npoints;
  std::array<Real, Npoints> parameters_t0;  // parameters at t0
  std::array<Real, Npoints> parameters_t1;  // parameters at t1
  std::array<Real, Npoints> dparameters_t0; // derivative at t0
  Real t0, t1;                              // t0 and t1

  void save(std::string filename) {
    std::ofstream savestream;
    savestream.setf(std::ios::scientific);
    savestream.precision(std::numeric_limits<Real>::digits10 + 1);
    savestream.open(filename);

    savestream << t0 << "\t" << t1 << std::endl;
    for (int i = 0; i < Npoints; ++i)
      savestream << parameters_t0[i] << "\t" << parameters_t1[i] << "\t"
                 << dparameters_t0[i] << std::endl;
    savestream.close();
  }

  void restart(std::string filename) {
    std::ifstream restartstream;
    restartstream.open(filename);
    restartstream >> t0 >> t1;
    for (int i = 0; i < Npoints; ++i)
      restartstream >> parameters_t0[i] >> parameters_t1[i] >>
          dparameters_t0[i];
    restartstream.close();
  }
  virtual void resetAll() {
    parameters_t0 = std::array<Real, Npoints>();
    parameters_t1 = std::array<Real, Npoints>();
    dparameters_t0 = std::array<Real, Npoints>();
    t0 = -1;
    t1 = 0;
  }

  ParameterScheduler() {
    t0 = -1;
    t1 = 0;
    parameters_t0 = std::array<Real, Npoints>();
    parameters_t1 = std::array<Real, Npoints>();
    dparameters_t0 = std::array<Real, Npoints>();
  }
  virtual ~ParameterScheduler() {}

  void transition(const Real t, const Real tstart, const Real tend,
                  const std::array<Real, Npoints> parameters_tend,
                  const bool UseCurrentDerivative = false) {
    if (t < tstart or t > tend)
      return; // this transition is out of scope
    // if(tstart<t0) return; // this transition is not relevant: we are doing a
    // next one already

    // we transition from whatever state we are in to a new state
    // the start point is where we are now: lets find out
    std::array<Real, Npoints> parameters;
    std::array<Real, Npoints> dparameters;
    gimmeValues(tstart, parameters, dparameters);

    // fill my members
    t0 = tstart;
    t1 = tend;
    parameters_t0 = parameters;
    parameters_t1 = parameters_tend;
    dparameters_t0 =
        UseCurrentDerivative ? dparameters : std::array<Real, Npoints>();
  }

  void transition(const Real t, const Real tstart, const Real tend,
                  const std::array<Real, Npoints> parameters_tstart,
                  const std::array<Real, Npoints> parameters_tend) {
    if (t < tstart or t > tend)
      return; // this transition is out of scope
    if (tstart < t0)
      return; // this transition is not relevant: we are doing a next one
              // already

    // fill my members
    t0 = tstart;
    t1 = tend;
    parameters_t0 = parameters_tstart;
    parameters_t1 = parameters_tend;
  }

  void gimmeValues(const Real t, std::array<Real, Npoints> &parameters,
                   std::array<Real, Npoints> &dparameters) {
    // look at the different cases
    if (t < t0 or t0 < 0) { // no transition, we are in state 0
      parameters = parameters_t0;
      dparameters = std::array<Real, Npoints>();
    } else if (t > t1) { // no transition, we are in state 1
      parameters = parameters_t1;
      dparameters = std::array<Real, Npoints>();
    } else { // we are within transition: interpolate
      for (int i = 0; i < Npoints; ++i)
        IF2D_Interpolation1D::cubicInterpolation(
            t0, t1, t, parameters_t0[i], parameters_t1[i], dparameters_t0[i],
            0.0, parameters[i], dparameters[i]);
    }
  }

  void gimmeValuesLinear(const Real t, std::array<Real, Npoints> &parameters,
                         std::array<Real, Npoints> &dparameters) {
    // look at the different cases
    if (t < t0 or t0 < 0) { // no transition, we are in state 0
      parameters = parameters_t0;
      dparameters = std::array<Real, Npoints>();
    } else if (t > t1) { // no transition, we are in state 1
      parameters = parameters_t1;
      dparameters = std::array<Real, Npoints>();
    } else { // we are within transition: interpolate
      for (int i = 0; i < Npoints; ++i)
        IF2D_Interpolation1D::linearInterpolation(
            t0, t1, t, parameters_t0[i], parameters_t1[i], parameters[i],
            dparameters[i]);
    }
  }

  void gimmeValues(const Real t, std::array<Real, Npoints> &parameters) {
    std::array<Real, Npoints> dparameters_whocares; // no derivative info
    return gimmeValues(t, parameters, dparameters_whocares);
  }
};

struct ParameterSchedulerScalar : ParameterScheduler<1> {
  void transition(const Real t, const Real tstart, const Real tend,
                  const Real parameter_tend, const bool keepSlope = false) {
    const std::array<Real, 1> myParameter = {parameter_tend};
    return ParameterScheduler<1>::transition(t, tstart, tend, myParameter,
                                             keepSlope);
  }

  void transition(const Real t, const Real tstart, const Real tend,
                  const Real parameter_tstart, const Real parameter_tend) {
    const std::array<Real, 1> myParameterStart = {parameter_tstart};
    const std::array<Real, 1> myParameterEnd = {parameter_tend};
    return ParameterScheduler<1>::transition(t, tstart, tend, myParameterStart,
                                             myParameterEnd);
  }

  void gimmeValues(const Real t, Real &parameter, Real &dparameter) {
    std::array<Real, 1> myParameter, mydParameter;
    ParameterScheduler<1>::gimmeValues(t, myParameter, mydParameter);
    parameter = myParameter[0];
    dparameter = mydParameter[0];
  }

  void gimmeValues(const Real t, Real &parameter) {
    std::array<Real, 1> myParameter;
    ParameterScheduler<1>::gimmeValues(t, myParameter);
    parameter = myParameter[0];
  }
};

template <int Npoints>
struct ParameterSchedulerVector : ParameterScheduler<Npoints> {
  void gimmeValues(const Real t, const std::array<Real, Npoints> &positions,
                   const int Nfine, const Real *const positions_fine,
                   Real *const parameters_fine, Real *const dparameters_fine) {
    // we interpolate in space the start and end point
    Real *parameters_t0_fine = new Real[Nfine];
    Real *parameters_t1_fine = new Real[Nfine];
    Real *dparameters_t0_fine = new Real[Nfine];

    IF2D_Interpolation1D::naturalCubicSpline(
        positions.data(), this->parameters_t0.data(), Npoints, positions_fine,
        parameters_t0_fine, Nfine);
    IF2D_Interpolation1D::naturalCubicSpline(
        positions.data(), this->parameters_t1.data(), Npoints, positions_fine,
        parameters_t1_fine, Nfine);
    IF2D_Interpolation1D::naturalCubicSpline(
        positions.data(), this->dparameters_t0.data(), Npoints, positions_fine,
        dparameters_t0_fine, Nfine);

    // look at the different cases
    if (t < this->t0 or this->t0 < 0) { // no transition, we are in state 0
      memcpy(parameters_fine, parameters_t0_fine, Nfine * sizeof(Real));
      memset(dparameters_fine, 0, Nfine * sizeof(Real));
    } else if (t > this->t1) { // no transition, we are in state 1
      memcpy(parameters_fine, parameters_t1_fine, Nfine * sizeof(Real));
      memset(dparameters_fine, 0, Nfine * sizeof(Real));
    } else {
      // we are within transition: interpolate in time for each point of the
      // fine discretization
      // #pragma omp parallel for schedule(static)
      for (int i = 0; i < Nfine; ++i)
        IF2D_Interpolation1D::cubicInterpolation(
            this->t0, this->t1, t, parameters_t0_fine[i], parameters_t1_fine[i],
            dparameters_t0_fine[i], 0, parameters_fine[i], dparameters_fine[i]);
    }
    delete[] parameters_t0_fine;
    delete[] parameters_t1_fine;
    delete[] dparameters_t0_fine;
  }

  void gimmeValues(const Real t, std::array<Real, Npoints> &parameters) {
    ParameterScheduler<Npoints>::gimmeValues(t, parameters);
  }

  void gimmeValues(const Real t, std::array<Real, Npoints> &parameters,
                   std::array<Real, Npoints> &dparameters) {
    ParameterScheduler<Npoints>::gimmeValues(t, parameters, dparameters);
  }
};

template <int Npoints>
struct ParameterSchedulerLearnWave : ParameterScheduler<Npoints> {
  template <typename T>
  void gimmeValues(const Real t, const Real Twave, const Real Length,
                   const std::array<Real, Npoints> &positions, const int Nfine,
                   const T *const positions_fine, T *const parameters_fine,
                   Real *const dparameters_fine) {
    const Real _1oL = 1. / Length;
    const Real _1oT = 1. / Twave;
    // the fish goes through (as function of t and s) a wave function that
    // describes the curvature
    // #pragma omp parallel for schedule(static)
    for (int i = 0; i < Nfine; ++i) {
      const Real c = positions_fine[i] * _1oL -
                     (t - this->t0) * _1oT; // traveling wave coord
      bool bCheck = true;

      if (c < positions[0]) { // Are you before latest wave node?
        IF2D_Interpolation1D::cubicInterpolation(
            c, positions[0], c, this->parameters_t0[0], this->parameters_t0[0],
            parameters_fine[i], dparameters_fine[i]);
        bCheck = false;
      } else if (c >
                 positions[Npoints - 1]) { // Are you after oldest wave node?
        IF2D_Interpolation1D::cubicInterpolation(
            positions[Npoints - 1], c, c, this->parameters_t0[Npoints - 1],
            this->parameters_t0[Npoints - 1], parameters_fine[i],
            dparameters_fine[i]);
        bCheck = false;
      } else {
        for (int j = 1; j < Npoints;
             ++j) { // Check at which point of the travelling wave we are
          if ((c >= positions[j - 1]) && (c <= positions[j])) {
            IF2D_Interpolation1D::cubicInterpolation(
                positions[j - 1], positions[j], c, this->parameters_t0[j - 1],
                this->parameters_t0[j], parameters_fine[i],
                dparameters_fine[i]);
            dparameters_fine[i] = -dparameters_fine[i] * _1oT; // df/dc * dc/dt
            bCheck = false;
          }
        }
      }
      if (bCheck) {
        std::cout << "Ciaone2!" << std::endl;
        abort();
      }
    }
  }

  void
  Turn(const Real b,
       const Real t_turn) // each decision adds a node at the beginning of the
                          // wave (left, right, straight) and pops last node
  {
    this->t0 = t_turn;

    for (int i = Npoints - 1; i > 1; --i)
      this->parameters_t0[i] = this->parameters_t0[i - 2];
    this->parameters_t0[1] = b;
    this->parameters_t0[0] = 0;
  }
};

/*********************** NEURO-KINEMATIC FISH *******************************/

class Synapse {
public:
  Real g = 0;
  Real dg = 0;
  const Real tau1 = 0.006 / 0.044;
  const Real tau2 = 0.008 / 0.044;
  Real prevTime = 0.0;
  std::vector<Real> activationTimes;
  std::vector<Real> activationAmplitudes;

public:
  void reset() {
    g = 0.0;
    dg = 0.0;
    prevTime = 0.0;
    activationTimes.clear();
    activationAmplitudes.clear();
  }
  void advance(const Real t) {
    //        printf("[Synapse][advance]\n");
    dg = 0;
    Real dt = t - prevTime;
    //        printf("[Synapse][advance] activationTimes.size() %ld\n",
    //        activationTimes.size());
    for (size_t i = 0; i < activationTimes.size(); i++) {
      const Real deltaT = t - activationTimes.at(i);
      //            printf("[Synapse][advance] deltaT %f\n", deltaT);
      const Real dBiExp = -1 / tau2 * std::exp(-deltaT / tau2) +
                          1 / tau1 * std::exp(-deltaT / tau1);
      //            printf("[Synapse][advance] dBiExp %f\n", dBiExp);
      dg += activationAmplitudes.at(i) * dBiExp;
      //            printf("[Synapse][advance] dg %f\n", dg);
    }
    g += dg * dt;
    prevTime = t;
    forget(t);
    //        printf("[Synapse][advance][end]\n");
  }
  void excite(const Real t, const Real amp) {
    //        printf("[Synapse][excite]\n");
    activationTimes.push_back(t);
    activationAmplitudes.push_back(amp);
    //        printf("[Synapse][excite][end]\n");
  }
  void forget(const Real t) {
    //        printf("[Synapse][forget]\n");
    if (activationTimes.size() != 0) {
      //            printf("[Synapse][forget] Number of activated synapses
      //            %ld\n", activationTimes.size()); printf("[Synapse][forget]
      //            t: %f, activationTime0: %f\n", t, activationTimes.at(0));
      //            printf("[Synapse][forget] tau1tau2: %f\n", tau1+tau2);
      if (t - activationTimes.at(0) > tau1 + tau2) {
        //                printf("Forgetting an activation. Current activation
        //                size is %ld\n", activationTimes.size());
        activationTimes.erase(activationTimes.begin());
        activationAmplitudes.erase(activationAmplitudes.begin());
      }
    }
    //        printf("[Synapse][forget][end]\n");
  }
  Real value() { return g; }
  Real speed() { return dg; }
};

template <int Npoints> class Oscillation {
public:
  Real d = 0.0;
  Real t0 = 0.0;
  Real prev_fmod = 0.0;
  std::vector<Real> signal = std::vector<Real>(Npoints, 0.0);
  std::vector<Real> signal_out = std::vector<Real>(Npoints, 0.0);

public:
  void reset() {
    d = 0.0;
    t0 = 0.0;
    prev_fmod = 0.0;
    signal.clear();
    signal_out.clear();
  }
  void modify(const Real t0_in, const Real f_in, const Real d_in) {
    //        printf("[Oscillation][modify]\n");
    d = d_in;
    t0 = t0_in;
    prev_fmod = 0;

    signal = std::vector<Real>(Npoints, 0.0);
    signal.at(0) = f_in;
    signal.at(static_cast<int>(
        std::ceil(static_cast<float>(Npoints + 1) / 2.0) - 1.0)) = -f_in;
    signal_out = signal;
    //        printf("[Oscillation][modify][end]\n");
  }
  void advance(const Real t) {
    //        printf("[Oscillation][advance]\n");
    if (fmod(t - t0, d) < prev_fmod && t > t0) {
      signal.insert(signal.begin(), signal.back());
      signal.pop_back();
      signal_out = signal;
    } else if (t == t0) {
      signal_out = signal;
    } else {
      signal_out = std::vector<Real>(Npoints, 0.0);
    }
    prev_fmod = fmod(t - t0, d);
    //        printf("[Oscillation][advance][end]\n");
  }
};

template <int Npoints>
struct ParameterSchedulerNeuroKinematic : ParameterScheduler<Npoints> {
  Real prevTime = 0.0;
  int numActiveSpikes = 0;
  const Real tau1 = 0.006 / 0.044; // 1 ms
  const Real tau2 = 0.008 / 0.044; // 6 ms (AMPA)

  std::array<Real, Npoints> neuroSignal_t_coarse = std::array<Real, Npoints>();
  std::array<Real, Npoints> timeActivated_coarse =
      std::array<Real, Npoints>(); // array of time each synapse has been
                                   // activated for
  std::array<Real, Npoints> muscSignal_t_coarse = std::array<Real, Npoints>();
  std::array<Real, Npoints> dMuscSignal_t_coarse = std::array<Real, Npoints>();
  std::vector<std::array<Real, Npoints>> neuroSignalVec_coarse;
  std::vector<std::array<Real, Npoints>> timeActivatedVec_coarse;
  std::vector<std::array<Real, Npoints>> muscSignalVec_coarse;
  std::vector<std::array<Real, Npoints>> dMuscSignalVec_coarse;
  std::vector<Real> amplitudeVec;

  virtual void resetAll() {
    prevTime = 0.0;
    numActiveSpikes = 0;
    neuroSignal_t_coarse = std::array<Real, Npoints>();
    timeActivated_coarse = std::array<Real, Npoints>();
    muscSignal_t_coarse = std::array<Real, Npoints>();
    dMuscSignal_t_coarse = std::array<Real, Npoints>();
    neuroSignalVec_coarse.clear();
    timeActivatedVec_coarse.clear();
    muscSignalVec_coarse.clear();
    dMuscSignalVec_coarse.clear();
    amplitudeVec.clear();
  }

  template <typename T>
  void gimmeValues(const Real t, const Real Length,
                   const std::array<Real, Npoints> &positions, const int Nfine,
                   const T *const positions_fine, T *const muscSignal_t_fine,
                   Real *const dMuscSignal_t_fine,
                   Real *const spatialDerivativeMuscSignal,
                   Real *const spatialDerivativeDMuscSignal) {
    // Advance arrays
    if (numActiveSpikes > 0) {
      this->dMuscSignal_t_coarse = std::array<Real, Npoints>();

      // Delete spikes that are no longer relevant
      for (int i = 0; i < numActiveSpikes; i++) {
        const Real relaxationTime = (Npoints + 1) * tau1 + tau2;
        const Real activeSpikeTime = t - timeActivatedVec_coarse.at(i).at(0);
        if (activeSpikeTime >= relaxationTime) {
          numActiveSpikes -= 1;
          this->neuroSignalVec_coarse.erase(neuroSignalVec_coarse.begin() + i);
          this->timeActivatedVec_coarse.erase(timeActivatedVec_coarse.begin() +
                                              i);
          this->muscSignalVec_coarse.erase(muscSignalVec_coarse.begin() + i);
          this->dMuscSignalVec_coarse.erase(dMuscSignalVec_coarse.begin() + i);
        }
      }

      advanceCoarseArrays(t);

      // Set previous time for next gimmeValues call
      this->prevTime = t;

      // Construct spine with cubic spline
      IF2D_Interpolation1D::naturalCubicSpline(
          positions.data(), this->muscSignal_t_coarse.data(), Npoints,
          positions_fine, muscSignal_t_fine, Nfine);
      IF2D_Interpolation1D::naturalCubicSpline(
          positions.data(), this->dMuscSignal_t_coarse.data(), Npoints,
          positions_fine, dMuscSignal_t_fine, Nfine);
    }
  }

  void advanceCoarseArrays(const Real time_current) {
    //        printf("[numActiveSpikes][%d]\n", numActiveSpikes);
    const Real delta_t = time_current - this->prevTime;
    for (int i = 0; i < numActiveSpikes; i++) {
      for (int j = 0; j < Npoints; j++) {
        const Real deltaT =
            time_current - this->timeActivatedVec_coarse.at(i).at(j);
        if (deltaT >= 0) {
          //                    printf("[i=%d][j=%d]\n", i, j);
          // Activate current node but don't switch off previous one.
          if (j > 0) {
            this->neuroSignalVec_coarse.at(i).at(j) =
                this->neuroSignalVec_coarse.at(i)[j - 1];
          }
          // Begin the muscle response at the new node.
          const Real dBiExp = -1 / this->tau2 * std::exp(-deltaT / this->tau2) +
                              1 / this->tau1 * std::exp(-deltaT / this->tau1);

          this->dMuscSignalVec_coarse.at(i).at(j) =
              this->neuroSignalVec_coarse.at(i).at(j) * dBiExp;

          // Increment the overall muscle signal and write the overall
          // derivative
          this->dMuscSignal_t_coarse.at(j) +=
              this->dMuscSignalVec_coarse.at(i).at(j);
          this->muscSignal_t_coarse.at(j) +=
              delta_t * this->dMuscSignalVec_coarse.at(i).at(j);
        }
      }
    }
  }

  // Deal with residual signal from previous firing time action (you can
  // increment the signal with itself)
  void Spike(const Real t_spike, const Real aCmd, const Real dCmd,
             const Real deltaTFireCmd) {
    this->t0 = t_spike;
    this->prevTime = t_spike;
    this->numActiveSpikes += 1;
    this->neuroSignalVec_coarse.push_back(std::array<Real, Npoints>());
    this->timeActivatedVec_coarse.push_back(std::array<Real, Npoints>());
    this->muscSignalVec_coarse.push_back(std::array<Real, Npoints>());
    this->dMuscSignalVec_coarse.push_back(std::array<Real, Npoints>());

    for (int j = 0; j < Npoints; j++) {
      this->timeActivatedVec_coarse.at(numActiveSpikes - 1).at(j) =
          this->t0 + j * dCmd;
    }
    // Activate the 0th node
    this->neuroSignalVec_coarse.at(numActiveSpikes - 1).at(0) = aCmd;
  }
};

template <int Npoints>
struct ParameterSchedulerNeuroKinematicObject : ParameterScheduler<Npoints> {
  std::array<Synapse, Npoints> synapses;
  Oscillation<Npoints> oscillation;

  std::array<Real, Npoints> muscle_value = std::array<Real, Npoints>();
  std::array<Real, Npoints> muscle_speed = std::array<Real, Npoints>();

  virtual void resetAll() {
    for (int i = 0; i < Npoints; i++) {
      synapses.at(i).reset();
    }
    oscillation.reset();
  }

  template <typename T>
  void gimmeValues(const Real t, const Real Length,
                   const std::array<Real, Npoints> &positions, const int Nfine,
                   const T *const positions_fine, T *const muscle_value_fine,
                   Real *const muscle_speed_fine) {
    advance(t);

    // Construct spine with cubic spline
    IF2D_Interpolation1D::naturalCubicSpline(
        positions.data(), this->muscle_value.data(), Npoints, positions_fine,
        muscle_value_fine, Nfine);
    IF2D_Interpolation1D::naturalCubicSpline(
        positions.data(), this->muscle_speed.data(), Npoints, positions_fine,
        muscle_speed_fine, Nfine);
  }

  void advance(const Real t) {
    oscillation.advance(t);
    for (int i = 0; i < Npoints; i++) {
      //            printf("[Scheduler][advance]\n");
      const Real oscAmp = oscillation.signal_out.at(i);
      printf("[Scheduler][advance] signal_i %f\n",
             (double)oscillation.signal.at(i));
      //            printf("[Scheduler][advance] oscAmp_i %f\n", oscAmp);
      if (oscAmp != 0) {
        synapses.at(i).excite(t, oscAmp);
      }
      synapses.at(i).advance(t);
      muscle_value.at(i) = synapses.at(i).value();
      muscle_speed.at(i) = synapses.at(i).speed();

      if (i == 0) {
        printf("[Scheduler][advance] muscle_value_0 %f\n",
               (double)muscle_value.at(0));
      }
      //            if (i==0) {printf("[Scheduler][advance] synapse_0 amplitude
      //            %f\n", synapses.at(0).activationAmplitudes.at(0));}
      if (i == 0) {
        printf("[Scheduler][advance] synapse_0 numActivations %ld\n",
               synapses.at(0).activationAmplitudes.size());
      }
      if (i == 10) {
        printf("[Scheduler][advance] muscle_value_10 %f\n",
               (double)muscle_value.at(10));
      }
      //            if (i==9) {printf("[Scheduler][advance] synapse_9 amplitude
      //            %f\n", synapses.at(9).activationAmplitudes.at(0));}
      if (i == 10) {
        printf("[Scheduler][advance] synapse_10 numActivations %ld\n",
               synapses.at(10).activationAmplitudes.size());
      }

      //            printf("[Scheduler][advance] muscle_value_i %f\n",
      //            muscle_value.at(i)); printf("[Scheduler][advance]
      //            muscle_speed_i %f\n", muscle_speed.at(i));
      //            printf("[Scheduler][advance][end]\n");
    }
  }

  void Spike(const Real t_spike, const Real aCmd, const Real dCmd,
             const Real deltaTFireCmd) {
    oscillation.modify(t_spike, aCmd, dCmd);
    //        synapses.at(0).excite(t_spike, aCmd);
    //        synapses.at(static_cast<int>(std::ceil(static_cast<float>(Npoints
    //        + 1)/2.0) - 1.0)).excite(t_spike, -aCmd); printf("Activated
    //        synapse 0 and synapse %d",
    //        static_cast<int>(std::ceil(static_cast<float>(Npoints + 1)/2.0)
    //        - 1.0));
  }
};
} // namespace Schedulers

class Shape;

class PutObjectsOnGrid : public Operator {
protected:
  const std::vector<cubism::BlockInfo> &velInfo = sim.vel->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &chiInfo = sim.chi->getBlocksInfo();

  void putChiOnGrid(Shape *const shape) const;

public:
  using Operator::Operator;

  void operator()(Real dt) override;
  void advanceShapes(Real dt);
  void putObjectsOnGrid();

  std::string getName() override { return "PutObjectsOnGrid"; }
};
// static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
struct ComputeSurfaceNormals {
  ComputeSurfaceNormals(const SimulationData &s) : sim(s) {}
  const SimulationData &sim;
  cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0}};
  cubism::StencilInfo stencil2{-1, -1, 0, 2, 2, 1, false, {0}};
  void operator()(ScalarLab &labChi, ScalarLab &labSDF,
                  const cubism::BlockInfo &infoChi,
                  const cubism::BlockInfo &infoSDF) const {
    for (const auto &shape : sim.shapes) {
      const std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
      if (OBLOCK[infoChi.blockID] == nullptr)
        continue; // obst not in block
      const Real h = infoChi.h;
      ObstacleBlock &o = *OBLOCK[infoChi.blockID];
      const Real i2h = 0.5 / h;
      const Real fac = 0.5 * h;
      for (int iy = 0; iy < ScalarBlock::sizeY; iy++)
        for (int ix = 0; ix < ScalarBlock::sizeX; ix++) {
          const Real gradHX = labChi(ix + 1, iy).s - labChi(ix - 1, iy).s;
          const Real gradHY = labChi(ix, iy + 1).s - labChi(ix, iy - 1).s;
          if (gradHX * gradHX + gradHY * gradHY < 1e-12)
            continue;
          const Real gradUX =
              i2h * (labSDF(ix + 1, iy).s - labSDF(ix - 1, iy).s);
          const Real gradUY =
              i2h * (labSDF(ix, iy + 1).s - labSDF(ix, iy - 1).s);
          const Real gradUSq = (gradUX * gradUX + gradUY * gradUY) + EPS;
          const Real D = fac * (gradHX * gradUX + gradHY * gradUY) / gradUSq;
          if (std::fabs(D) > EPS)
            o.write(ix, iy, D, gradUX, gradUY);
        }
      o.allocate_surface();
    }
  }
};

struct PutChiOnGrid {
  PutChiOnGrid(const SimulationData &s) : sim(s) {}
  const SimulationData &sim;
  const cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0}};
  const std::vector<cubism::BlockInfo> &chiInfo = sim.chi->getBlocksInfo();
  void operator()(ScalarLab &lab, const cubism::BlockInfo &info) const {
    for (const auto &shape : sim.shapes) {
      const std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
      if (OBLOCK[info.blockID] == nullptr)
        continue; // obst not in block
      const Real h = info.h;
      const Real h2 = h * h;
      ObstacleBlock &o = *OBLOCK[info.blockID];
      CHI_MAT &__restrict__ X = o.chi;
      const CHI_MAT &__restrict__ sdf = o.dist;
      o.COM_x = 0;
      o.COM_y = 0;
      o.Mass = 0;
      auto &__restrict__ CHI = *(ScalarBlock *)chiInfo[info.blockID].ptrBlock;
      for (int iy = 0; iy < ScalarBlock::sizeY; iy++)
        for (int ix = 0; ix < ScalarBlock::sizeX; ix++) {
#if 0
        X[iy][ix] = sdf[iy][ix] > 0 ? 1 : 0;
#else // Towers mollified Heaviside
          if (sdf[iy][ix] > +h || sdf[iy][ix] < -h) {
            X[iy][ix] = sdf[iy][ix] > 0 ? 1 : 0;
          } else {
            const Real distPx = lab(ix + 1, iy).s;
            const Real distMx = lab(ix - 1, iy).s;
            const Real distPy = lab(ix, iy + 1).s;
            const Real distMy = lab(ix, iy - 1).s;
            const Real IplusX = std::max((Real)0.0, distPx);
            const Real IminuX = std::max((Real)0.0, distMx);
            const Real IplusY = std::max((Real)0.0, distPy);
            const Real IminuY = std::max((Real)0.0, distMy);
            const Real gradIX = IplusX - IminuX;
            const Real gradIY = IplusY - IminuY;
            const Real gradUX = distPx - distMx;
            const Real gradUY = distPy - distMy;
            const Real gradUSq = (gradUX * gradUX + gradUY * gradUY) + EPS;
            X[iy][ix] = (gradIX * gradUX + gradIY * gradUY) / gradUSq;
          }
#endif
          CHI(ix, iy).s = std::max(CHI(ix, iy).s, X[iy][ix]);
          if (X[iy][ix] > 0) {
            Real p[2];
            info.pos(p, ix, iy);
            o.COM_x += X[iy][ix] * h2 * (p[0] - shape->centerOfMass[0]);
            o.COM_y += X[iy][ix] * h2 * (p[1] - shape->centerOfMass[1]);
            o.Mass += X[iy][ix] * h2;
          }
        }
    }
  }
};

void PutObjectsOnGrid::operator()(const Real dt) {
  sim.startProfiler("PutObjectsGrid");

  advanceShapes(dt);
  putObjectsOnGrid();

  sim.stopProfiler();
}

void PutObjectsOnGrid::advanceShapes(const Real dt) {
  // Update laboratory frame of reference
  int nSum[2] = {0, 0};
  Real uSum[2] = {0, 0};
  for (const auto &shape : sim.shapes)
    shape->updateLabVelocity(nSum, uSum);
  if (nSum[0] > 0) {
    sim.uinfx_old = sim.uinfx;
    sim.uinfx = uSum[0] / nSum[0];
  }
  if (nSum[1] > 0) {
    sim.uinfy_old = sim.uinfy;
    sim.uinfy = uSum[1] / nSum[1];
  }
  // Update position of object r^{t+1}=r^t+dt*v, \theta^{t+1}=\theta^t+dt*\omega
  for (const auto &shape : sim.shapes) {
    shape->updatePosition(dt);

    // .. and check if shape is outside the simulation domain
    Real p[2] = {0, 0};
    shape->getCentroid(p);
    const auto &extent = sim.extents;
    if (p[0] < 0 || p[0] > extent[0] || p[1] < 0 || p[1] > extent[1]) {
      printf("[CUP2D] ABORT: Body out of domain [0,%f]x[0,%f] CM:[%e,%e]\n",
             (double)extent[0], (double)extent[1], (double)p[0], (double)p[1]);
      fflush(0);
      abort();
    }
  }
}

void PutObjectsOnGrid::putObjectsOnGrid() {
  const size_t Nblocks = velInfo.size();

// 1) Clear fields related to obstacle
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    ((ScalarBlock *)chiInfo[i].ptrBlock)->clear();
    ((ScalarBlock *)tmpInfo[i].ptrBlock)->set(-1);
  }

  // 2) Compute signed dist function and udef
  for (const auto &shape : sim.shapes)
    shape->create(tmpInfo);

  // 3) Compute chi and shape center of mass
  const PutChiOnGrid K(sim);
  cubism::compute<ScalarLab>(K, sim.tmp);
  const ComputeSurfaceNormals K1(sim);
  cubism::compute<ComputeSurfaceNormals, ScalarGrid, ScalarLab, ScalarGrid,
                  ScalarLab>(K1, *sim.chi, *sim.tmp);
  for (const auto &shape : sim.shapes) {
    Real com[3] = {0.0, 0.0, 0.0};
    const std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
#pragma omp parallel for reduction(+ : com[ : 3])
    for (size_t i = 0; i < OBLOCK.size(); i++) {
      if (OBLOCK[i] == nullptr)
        continue;
      com[0] += OBLOCK[i]->Mass;
      com[1] += OBLOCK[i]->COM_x;
      com[2] += OBLOCK[i]->COM_y;
    }
    MPI_Allreduce(MPI_IN_PLACE, com, 3, MPI_Real, MPI_SUM,
                  sim.chi->getWorldComm());
    shape->M = com[0];
    shape->centerOfMass[0] += com[1] / com[0];
    shape->centerOfMass[1] += com[2] / com[0];
  }

  // 4) remove moments from characteristic function and put on grid U_s
  for (const auto &shape : sim.shapes) {
    shape->removeMoments(chiInfo);
  }

  // 5) do anything else needed by some shapes
  for (const auto &shape : sim.shapes) {
    shape->finalize();
  }
}

class AdaptTheMesh : public Operator {
public:
  ScalarAMR *tmp_amr = nullptr;
  ScalarAMR *chi_amr = nullptr;
  ScalarAMR *pres_amr = nullptr;
  ScalarAMR *pold_amr = nullptr;
  VectorAMR *vel_amr = nullptr;
  VectorAMR *vOld_amr = nullptr;
  VectorAMR *tmpV_amr = nullptr;
  ScalarAMR *Cs_amr = nullptr;

  AdaptTheMesh(SimulationData &s) : Operator(s) {
    tmp_amr = new ScalarAMR(*sim.tmp, sim.Rtol, sim.Ctol);
    chi_amr = new ScalarAMR(*sim.chi, sim.Rtol, sim.Ctol);
    pres_amr = new ScalarAMR(*sim.pres, sim.Rtol, sim.Ctol);
    pold_amr = new ScalarAMR(*sim.pold, sim.Rtol, sim.Ctol);
    vel_amr = new VectorAMR(*sim.vel, sim.Rtol, sim.Ctol);
    vOld_amr = new VectorAMR(*sim.vOld, sim.Rtol, sim.Ctol);
    tmpV_amr = new VectorAMR(*sim.tmpV, sim.Rtol, sim.Ctol);
    if (sim.smagorinskyCoeff != 0)
      Cs_amr = new ScalarAMR(*sim.Cs, sim.Rtol, sim.Ctol);
  }

  ~AdaptTheMesh() {
    if (tmp_amr not_eq nullptr)
      delete tmp_amr;
    if (chi_amr not_eq nullptr)
      delete chi_amr;
    if (pres_amr not_eq nullptr)
      delete pres_amr;
    if (pold_amr not_eq nullptr)
      delete pold_amr;
    if (vel_amr not_eq nullptr)
      delete vel_amr;
    if (vOld_amr not_eq nullptr)
      delete vOld_amr;
    if (tmpV_amr not_eq nullptr)
      delete tmpV_amr;
    if (Cs_amr not_eq nullptr)
      delete Cs_amr;
  }

  void operator()(const Real dt) override;
  void adapt();

  std::string getName() override { return "AdaptTheMesh"; }
};

struct GradChiOnTmp {
  GradChiOnTmp(const SimulationData &s) : sim(s) {}
  const SimulationData &sim;
  // const StencilInfo stencil{-2, -2, 0, 3, 3, 1, true, {0}};
  const cubism::StencilInfo stencil{-4, -4, 0, 5, 5, 1, true, {0}};
  const std::vector<cubism::BlockInfo> &tmpInfo = sim.tmp->getBlocksInfo();
  void operator()(ScalarLab &lab, const cubism::BlockInfo &info) const {
    auto &__restrict__ TMP = *(ScalarBlock *)tmpInfo[info.blockID].ptrBlock;
    if (sim.Qcriterion)
      for (int y = 0; y < VectorBlock::sizeY; ++y)
        for (int x = 0; x < VectorBlock::sizeX; ++x)
          TMP(x, y).s = std::max(TMP(x, y).s, (Real)0.0); // compress if Q<0

    // Loop over block and halo cells and set TMP(0,0) to a value which will
    // cause mesh refinement if any of the cells have:
    //  1. chi > 0 (if bAdaptChiGradient=false)
    //  2. chi > 0 and chi < 0.9 (if bAdaptChiGradient=true)
    //  Option 2 is equivalent to grad(chi) != 0
    // const int offset = (info.level == sim.tmp->getlevelMax()-1) ? 2 : 1;
    const int offset = (info.level == sim.tmp->getlevelMax() - 1) ? 4 : 2;
    const Real threshold = sim.bAdaptChiGradient ? 0.9 : 1e4;
    for (int y = -offset; y < VectorBlock::sizeY + offset; ++y)
      for (int x = -offset; x < VectorBlock::sizeX + offset; ++x) {
        lab(x, y).s = std::min(lab(x, y).s, (Real)1.0);
        lab(x, y).s = std::max(lab(x, y).s, (Real)0.0);
        if (lab(x, y).s > 0.0 && lab(x, y).s < threshold) {
          TMP(VectorBlock::sizeX / 2 - 1, VectorBlock::sizeY / 2).s =
              2 * sim.Rtol;
          TMP(VectorBlock::sizeX / 2 - 1, VectorBlock::sizeY / 2 - 1).s =
              2 * sim.Rtol;
          TMP(VectorBlock::sizeX / 2, VectorBlock::sizeY / 2).s = 2 * sim.Rtol;
          TMP(VectorBlock::sizeX / 2, VectorBlock::sizeY / 2 - 1).s =
              2 * sim.Rtol;
          break;
        }
      }

#ifdef CUP2D_CYLINDER_REF
    // Hardcoded refinement close the wall, for the high Re cylinder cases.
    // Cylinder center is supposed to be at (1.0,1.0) and its radius is 0.1
    for (int y = 0; y < VectorBlock::sizeY; ++y)
      for (int x = 0; x < VectorBlock::sizeX; ++x) {
        double p[2];
        info.pos(p, x, y);
        p[0] -= 1.0;
        p[1] -= 1.0;
        const double r = p[0] * p[0] + p[1] * p[1];
        if (r > 0.1 * 0.1 && r < 0.11 * 0.11) {
          TMP(VectorBlock::sizeX / 2 - 1, VectorBlock::sizeY / 2).s =
              2 * sim.Rtol;
          TMP(VectorBlock::sizeX / 2 - 1, VectorBlock::sizeY / 2 - 1).s =
              2 * sim.Rtol;
          TMP(VectorBlock::sizeX / 2, VectorBlock::sizeY / 2).s = 2 * sim.Rtol;
          TMP(VectorBlock::sizeX / 2, VectorBlock::sizeY / 2 - 1).s =
              2 * sim.Rtol;
          break;
        }
      }
#endif
  }
};

void AdaptTheMesh::operator()(const Real dt) {
  if (sim.step > 10 && sim.step % sim.AdaptSteps != 0)
    return;
  adapt();
}

void AdaptTheMesh::adapt() {
  sim.startProfiler("AdaptTheMesh");

  const std::vector<cubism::BlockInfo> &tmpInfo = sim.tmp->getBlocksInfo();

  // compute vorticity (and use it as refinement criterion) and store it to tmp.
  if (sim.Qcriterion) {
    auto K1 = computeQ(sim);
    K1(0);
  } else {
    auto K1 = computeVorticity(sim);
    K1(0);
  }

  // compute grad(chi) and if it's >0 set tmp = infinity
  GradChiOnTmp K2(sim);
  cubism::compute<ScalarLab>(K2, sim.chi);

  tmp_amr->Tag();
  chi_amr->TagLike(tmpInfo);
  pres_amr->TagLike(tmpInfo);
  pold_amr->TagLike(tmpInfo);
  vel_amr->TagLike(tmpInfo);
  vOld_amr->TagLike(tmpInfo);
  tmpV_amr->TagLike(tmpInfo);
  if (sim.smagorinskyCoeff != 0)
    Cs_amr->TagLike(tmpInfo);

  tmp_amr->Adapt(sim.time, sim.rank == 0 && !sim.muteAll, false);
  chi_amr->Adapt(sim.time, false, false);
  vel_amr->Adapt(sim.time, false, false);
  vOld_amr->Adapt(sim.time, false, false);
  pres_amr->Adapt(sim.time, false, false);
  pold_amr->Adapt(sim.time, false, false);
  tmpV_amr->Adapt(sim.time, false, true);
  if (sim.smagorinskyCoeff != 0)
    Cs_amr->Adapt(sim.time, false, true);

  sim.stopProfiler();
}

class advDiff : public Operator {
protected:
  const std::vector<cubism::BlockInfo> &velInfo = sim.vel->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &tmpVInfo = sim.tmpV->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &vOldInfo = sim.vOld->getBlocksInfo();

public:
  advDiff(SimulationData &s) : Operator(s) {}

  void operator()(const Real dt) override;

  std::string getName() override { return "advDiff"; }
};

#ifdef CUP2D_PRESERVE_SYMMETRY
#define CUP2D_DISABLE_OPTIMIZATIONS __attribute__((optimize("-O1")))
#else
#define CUP2D_DISABLE_OPTIMIZATIONS
#endif

CUP2D_DISABLE_OPTIMIZATIONS
static inline Real weno5_plus(const Real um2, const Real um1, const Real u,
                              const Real up1, const Real up2) {
  const Real exponent = 2;
  const Real e = 1e-6;
  const Real b1 = 13.0 / 12.0 * pow((um2 + u) - 2 * um1, 2) +
                  0.25 * pow((um2 + 3 * u) - 4 * um1, 2);
  const Real b2 =
      13.0 / 12.0 * pow((um1 + up1) - 2 * u, 2) + 0.25 * pow(um1 - up1, 2);
  const Real b3 = 13.0 / 12.0 * pow((u + up2) - 2 * up1, 2) +
                  0.25 * pow((3 * u + up2) - 4 * up1, 2);
  const Real g1 = 0.1;
  const Real g2 = 0.6;
  const Real g3 = 0.3;
  const Real what1 = g1 / pow(b1 + e, exponent);
  const Real what2 = g2 / pow(b2 + e, exponent);
  const Real what3 = g3 / pow(b3 + e, exponent);
  const Real aux = 1.0 / ((what1 + what3) + what2);
  const Real w1 = what1 * aux;
  const Real w2 = what2 * aux;
  const Real w3 = what3 * aux;
  const Real f1 = (11.0 / 6.0) * u + ((1.0 / 3.0) * um2 - (7.0 / 6.0) * um1);
  const Real f2 = (5.0 / 6.0) * u + ((-1.0 / 6.0) * um1 + (1.0 / 3.0) * up1);
  const Real f3 = (1.0 / 3.0) * u + ((+5.0 / 6.0) * up1 - (1.0 / 6.0) * up2);
  return (w1 * f1 + w3 * f3) + w2 * f2;
}

CUP2D_DISABLE_OPTIMIZATIONS
static inline Real weno5_minus(const Real um2, const Real um1, const Real u,
                               const Real up1, const Real up2) {
  const Real exponent = 2;
  const Real e = 1e-6;
  const Real b1 = 13.0 / 12.0 * pow((um2 + u) - 2 * um1, 2) +
                  0.25 * pow((um2 + 3 * u) - 4 * um1, 2);
  const Real b2 =
      13.0 / 12.0 * pow((um1 + up1) - 2 * u, 2) + 0.25 * pow(um1 - up1, 2);
  const Real b3 = 13.0 / 12.0 * pow((u + up2) - 2 * up1, 2) +
                  0.25 * pow((3 * u + up2) - 4 * up1, 2);
  const Real g1 = 0.3;
  const Real g2 = 0.6;
  const Real g3 = 0.1;
  const Real what1 = g1 / pow(b1 + e, exponent);
  const Real what2 = g2 / pow(b2 + e, exponent);
  const Real what3 = g3 / pow(b3 + e, exponent);
  const Real aux = 1.0 / ((what1 + what3) + what2);
  const Real w1 = what1 * aux;
  const Real w2 = what2 * aux;
  const Real w3 = what3 * aux;
  const Real f1 = (1.0 / 3.0) * u + ((-1.0 / 6.0) * um2 + (5.0 / 6.0) * um1);
  const Real f2 = (5.0 / 6.0) * u + ((1.0 / 3.0) * um1 - (1.0 / 6.0) * up1);
  const Real f3 = (11.0 / 6.0) * u + ((-7.0 / 6.0) * up1 + (1.0 / 3.0) * up2);
  return (w1 * f1 + w3 * f3) + w2 * f2;
}

static inline Real derivative(const Real U, const Real um3, const Real um2,
                              const Real um1, const Real u, const Real up1,
                              const Real up2, const Real up3) {
  Real fp = 0.0;
  Real fm = 0.0;
  if (U > 0) {
    fp = weno5_plus(um2, um1, u, up1, up2);
    fm = weno5_plus(um3, um2, um1, u, up1);
  } else {
    fp = weno5_minus(um1, u, up1, up2, up3);
    fm = weno5_minus(um2, um1, u, up1, up2);
  }
  return (fp - fm);
}

static inline Real dU_adv_dif(const VectorLab &V, const Real uinf[2],
                              const Real advF, const Real difF, const int ix,
                              const int iy) {
  const Real u = V(ix, iy).u[0];
  const Real v = V(ix, iy).u[1];
  const Real UU = u + uinf[0];
  const Real VV = v + uinf[1];

  const Real up1x = V(ix + 1, iy).u[0];
  const Real up2x = V(ix + 2, iy).u[0];
  const Real up3x = V(ix + 3, iy).u[0];
  const Real um1x = V(ix - 1, iy).u[0];
  const Real um2x = V(ix - 2, iy).u[0];
  const Real um3x = V(ix - 3, iy).u[0];

  const Real up1y = V(ix, iy + 1).u[0];
  const Real up2y = V(ix, iy + 2).u[0];
  const Real up3y = V(ix, iy + 3).u[0];
  const Real um1y = V(ix, iy - 1).u[0];
  const Real um2y = V(ix, iy - 2).u[0];
  const Real um3y = V(ix, iy - 3).u[0];

  const Real dudx = derivative(UU, um3x, um2x, um1x, u, up1x, up2x, up3x);
  const Real dudy = derivative(VV, um3y, um2y, um1y, u, up1y, up2y, up3y);

  return advF * (UU * dudx + VV * dudy) +
         difF * (((up1x + um1x) + (up1y + um1y)) - 4 * u);
}

static inline Real dV_adv_dif(const VectorLab &V, const Real uinf[2],
                              const Real advF, const Real difF, const int ix,
                              const int iy) {
  const Real u = V(ix, iy).u[0];
  const Real v = V(ix, iy).u[1];
  const Real UU = u + uinf[0];
  const Real VV = v + uinf[1];

  const Real vp1x = V(ix + 1, iy).u[1];
  const Real vp2x = V(ix + 2, iy).u[1];
  const Real vp3x = V(ix + 3, iy).u[1];
  const Real vm1x = V(ix - 1, iy).u[1];
  const Real vm2x = V(ix - 2, iy).u[1];
  const Real vm3x = V(ix - 3, iy).u[1];

  const Real vp1y = V(ix, iy + 1).u[1];
  const Real vp2y = V(ix, iy + 2).u[1];
  const Real vp3y = V(ix, iy + 3).u[1];
  const Real vm1y = V(ix, iy - 1).u[1];
  const Real vm2y = V(ix, iy - 2).u[1];
  const Real vm3y = V(ix, iy - 3).u[1];

  const Real dvdx = derivative(UU, vm3x, vm2x, vm1x, v, vp1x, vp2x, vp3x);
  const Real dvdy = derivative(VV, vm3y, vm2y, vm1y, v, vp1y, vp2y, vp3y);

  return advF * (UU * dvdx + VV * dvdy) +
         difF * (((vp1x + vm1x) + (vp1y + vm1y)) - 4 * v);
}

struct KernelAdvectDiffuse {
  KernelAdvectDiffuse(const SimulationData &s) : sim(s) {
    uinf[0] = sim.uinfx;
    uinf[1] = sim.uinfy;
  }
  const SimulationData &sim;
  Real uinf[2];
  const cubism::StencilInfo stencil{-3, -3, 0, 4, 4, 1, true, {0, 1}};
  const std::vector<cubism::BlockInfo> &tmpVInfo = sim.tmpV->getBlocksInfo();

  void operator()(VectorLab &lab, const cubism::BlockInfo &info) const {
    const Real h = info.h;
    const Real dfac = sim.nu * sim.dt;
    const Real afac = -sim.dt * h;
    VectorBlock &__restrict__ TMP =
        *(VectorBlock *)tmpVInfo[info.blockID].ptrBlock;
    for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        TMP(ix, iy).u[0] = dU_adv_dif(lab, uinf, afac, dfac, ix, iy);
        TMP(ix, iy).u[1] = dV_adv_dif(lab, uinf, afac, dfac, ix, iy);
      }
    cubism::BlockCase<VectorBlock> *tempCase =
        (cubism::BlockCase<VectorBlock> *)(tmpVInfo[info.blockID].auxiliary);
    VectorBlock::ElementType *faceXm = nullptr;
    VectorBlock::ElementType *faceXp = nullptr;
    VectorBlock::ElementType *faceYm = nullptr;
    VectorBlock::ElementType *faceYp = nullptr;

    const Real aux_coef = dfac;

    if (tempCase != nullptr) {
      faceXm = tempCase->storedFace[0] ? &tempCase->m_pData[0][0] : nullptr;
      faceXp = tempCase->storedFace[1] ? &tempCase->m_pData[1][0] : nullptr;
      faceYm = tempCase->storedFace[2] ? &tempCase->m_pData[2][0] : nullptr;
      faceYp = tempCase->storedFace[3] ? &tempCase->m_pData[3][0] : nullptr;
    }
    if (faceXm != nullptr) {
      int ix = 0;
      for (int iy = 0; iy < VectorBlock::sizeY; ++iy) {
        faceXm[iy].u[0] = aux_coef * (lab(ix, iy).u[0] - lab(ix - 1, iy).u[0]);
        faceXm[iy].u[1] = aux_coef * (lab(ix, iy).u[1] - lab(ix - 1, iy).u[1]);
      }
    }
    if (faceXp != nullptr) {
      int ix = VectorBlock::sizeX - 1;
      for (int iy = 0; iy < VectorBlock::sizeY; ++iy) {
        faceXp[iy].u[0] = aux_coef * (lab(ix, iy).u[0] - lab(ix + 1, iy).u[0]);
        faceXp[iy].u[1] = aux_coef * (lab(ix, iy).u[1] - lab(ix + 1, iy).u[1]);
      }
    }
    if (faceYm != nullptr) {
      int iy = 0;
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        faceYm[ix].u[0] = aux_coef * (lab(ix, iy).u[0] - lab(ix, iy - 1).u[0]);
        faceYm[ix].u[1] = aux_coef * (lab(ix, iy).u[1] - lab(ix, iy - 1).u[1]);
      }
    }
    if (faceYp != nullptr) {
      int iy = VectorBlock::sizeY - 1;
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        faceYp[ix].u[0] = aux_coef * (lab(ix, iy).u[0] - lab(ix, iy + 1).u[0]);
        faceYp[ix].u[1] = aux_coef * (lab(ix, iy).u[1] - lab(ix, iy + 1).u[1]);
      }
    }
  }
};

void advDiff::operator()(const Real dt) {
  sim.startProfiler("advDiff");
  const size_t Nblocks = velInfo.size();
  KernelAdvectDiffuse Step1(sim);

// 1.Save u^{n} to dataOld
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    VectorBlock &__restrict__ Vold = *(VectorBlock *)vOldInfo[i].ptrBlock;
    const VectorBlock &__restrict__ V = *(VectorBlock *)velInfo[i].ptrBlock;
    for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        Vold(ix, iy).u[0] = V(ix, iy).u[0];
        Vold(ix, iy).u[1] = V(ix, iy).u[1];
      }
  }

  /********************************************************************/
  // 2. Set u^{n+1/2} = u^{n} + 0.5*dt*RHS(u^{n})
  //   2a) Compute 0.5*dt*RHS(u^{n}) and store it to tmpU,tmpV,tmpW
  cubism::compute<VectorLab>(Step1, sim.vel, sim.tmpV);

//   2b) Set u^{n+1/2} = u^{n} + 0.5*dt*RHS(u^{n})
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    VectorBlock &__restrict__ V = *(VectorBlock *)velInfo[i].ptrBlock;
    const VectorBlock &__restrict__ Vold = *(VectorBlock *)vOldInfo[i].ptrBlock;
    const VectorBlock &__restrict__ tmpV = *(VectorBlock *)tmpVInfo[i].ptrBlock;
    const Real ih2 = 1.0 / (velInfo[i].h * velInfo[i].h);
    for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        V(ix, iy).u[0] = Vold(ix, iy).u[0] + (0.5 * tmpV(ix, iy).u[0]) * ih2;
        V(ix, iy).u[1] = Vold(ix, iy).u[1] + (0.5 * tmpV(ix, iy).u[1]) * ih2;
      }
  }
  /********************************************************************/

  /********************************************************************/
  // 3. Set u^{n+1} = u^{n} + dt*RHS(u^{n+1/2})
  //   3a) Compute dt*RHS(u^{n+1/2}) and store it to tmpU,tmpV,tmpW
  cubism::compute<VectorLab>(Step1, sim.vel, sim.tmpV);
//   3b) Set u^{n+1} = u^{n} + dt*RHS(u^{n+1/2})
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    VectorBlock &__restrict__ V = *(VectorBlock *)velInfo[i].ptrBlock;
    const VectorBlock &__restrict__ Vold = *(VectorBlock *)vOldInfo[i].ptrBlock;
    const VectorBlock &__restrict__ tmpV = *(VectorBlock *)tmpVInfo[i].ptrBlock;
    const Real ih2 = 1.0 / (velInfo[i].h * velInfo[i].h);
    for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        V(ix, iy).u[0] = Vold(ix, iy).u[0] + tmpV(ix, iy).u[0] * ih2;
        V(ix, iy).u[1] = Vold(ix, iy).u[1] + tmpV(ix, iy).u[1] * ih2;
      }
  }
  /********************************************************************/

  sim.stopProfiler();
}

class Shape;

class ComputeForces : public Operator {
  const std::vector<cubism::BlockInfo> &presInfo = sim.pres->getBlocksInfo();

public:
  void operator()(const Real dt) override;

  ComputeForces(SimulationData &s);
  ~ComputeForces() {}

  std::string getName() override { return "ComputeForces"; }
};

using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];

struct KernelComputeForces {
  const int big = 5;
  const int small = -4;
  KernelComputeForces(const SimulationData &s) : sim(s) {}
  const SimulationData &sim;
  cubism::StencilInfo stencil{small, small, 0, big, big, 1, true, {0, 1}};
  cubism::StencilInfo stencil2{small, small, 0, big, big, 1, true, {0}};

  const int bigg = ScalarBlock::sizeX + big - 1;
  const int stencil_start[3] = {small, small, small},
            stencil_end[3] = {big, big, big};
  const Real c0 = -137. / 60.;
  const Real c1 = 5.;
  const Real c2 = -5.;
  const Real c3 = 10. / 3.;
  const Real c4 = -5. / 4.;
  const Real c5 = 1. / 5.;

  inline bool inrange(const int i) const { return (i >= small && i < bigg); }

  const std::vector<cubism::BlockInfo> &presInfo = sim.pres->getBlocksInfo();

  void operator()(VectorLab &lab, ScalarLab &chi, const cubism::BlockInfo &info,
                  const cubism::BlockInfo &info2) const {
    VectorLab &V = lab;
    ScalarBlock &__restrict__ P =
        *(ScalarBlock *)presInfo[info.blockID].ptrBlock;

    // const int big   = ScalarBlock::sizeX + 4;
    // const int small = -4;
    for (const auto &_shape : sim.shapes) {
      const Shape *const shape = _shape.get();
      const std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
      const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];
      const Real vel_norm =
          std::sqrt(shape->u * shape->u + shape->v * shape->v);
      const Real vel_unit[2] = {
          vel_norm > 0 ? (Real)shape->u / vel_norm : (Real)0,
          vel_norm > 0 ? (Real)shape->v / vel_norm : (Real)0};

      const Real NUoH = sim.nu / info.h; // 2 nu / 2 h
      ObstacleBlock *const O = OBLOCK[info.blockID];
      if (O == nullptr)
        continue;
      assert(O->filled);
      for (size_t k = 0; k < O->n_surfPoints; ++k) {
        const int ix = O->surface[k]->ix, iy = O->surface[k]->iy;
        const std::array<Real, 2> p = info.pos<Real>(ix, iy);

        const Real normX = O->surface[k]->dchidx; //*h^3 (multiplied in dchidx)
        const Real normY = O->surface[k]->dchidy; //*h^3 (multiplied in dchidy)
        const Real norm = 1.0 / std::sqrt(normX * normX + normY * normY);
        const Real dx = normX * norm;
        const Real dy = normY * norm;
        // shear stresses
        //"lifted" surface: derivatives make no sense when the values used are
        //in the object,
        //  so we take one-sided stencils with values outside of the object
        // Real D11 = 0.0;
        // Real D22 = 0.0;
        // Real D12 = 0.0;
        Real DuDx;
        Real DuDy;
        Real DvDx;
        Real DvDy;
        {
          // The integers x and y will be the coordinates of the point on the
          // lifted surface. To find them, we move along the normal vector to
          // the surface, until we find a point outside of the object (where chi
          // = 0).
          int x = ix;
          int y = iy;
          for (int kk = 0; kk < 5; kk++) // 5 is arbitrary
          {
            const int dxi = round(kk * dx);
            const int dyi = round(kk * dy);
            if (ix + dxi + 1 >= ScalarBlock::sizeX + big - 1 ||
                ix + dxi - 1 < small)
              continue;
            if (iy + dyi + 1 >= ScalarBlock::sizeY + big - 1 ||
                iy + dyi - 1 < small)
              continue;
            x = ix + dxi;
            y = iy + dyi;
            if (chi(x, y).s < 0.01)
              break;
          }

          // Now that we found the (x,y) of the point, we compute grad(u) there.
          // grad(u) is computed with biased stencils. If available, larger
          // stencils are used. Then, we compute higher order derivatives that
          // are used to form a Taylor expansion around (x,y). Finally, this
          // expansion is used to extrapolate grad(u) to (ix,iy) of the actual
          // solid surface.

          const auto &l = lab;
          const int sx = normX > 0 ? +1 : -1;
          const int sy = normY > 0 ? +1 : -1;

          VectorElement dveldx;
          if (inrange(x + 5 * sx))
            dveldx = sx * (c0 * l(x, y) + c1 * l(x + sx, y) +
                           c2 * l(x + 2 * sx, y) + c3 * l(x + 3 * sx, y) +
                           c4 * l(x + 4 * sx, y) + c5 * l(x + 5 * sx, y));
          else if (inrange(x + 2 * sx))
            dveldx = sx * (-1.5 * l(x, y) + 2.0 * l(x + sx, y) -
                           0.5 * l(x + 2 * sx, y));
          else
            dveldx = sx * (l(x + sx, y) - l(x, y));
          VectorElement dveldy;
          if (inrange(y + 5 * sy))
            dveldy = sy * (c0 * l(x, y) + c1 * l(x, y + sy) +
                           c2 * l(x, y + 2 * sy) + c3 * l(x, y + 3 * sy) +
                           c4 * l(x, y + 4 * sy) + c5 * l(x, y + 5 * sy));
          else if (inrange(y + 2 * sy))
            dveldy = sy * (-1.5 * l(x, y) + 2.0 * l(x, y + sy) -
                           0.5 * l(x, y + 2 * sy));
          else
            dveldy = sx * (l(x, y + sy) - l(x, y));

          const VectorElement dveldx2 =
              l(x - 1, y) - 2.0 * l(x, y) + l(x + 1, y);
          const VectorElement dveldy2 =
              l(x, y - 1) - 2.0 * l(x, y) + l(x, y + 1);

          VectorElement dveldxdy;
          if (inrange(x + 2 * sx) && inrange(y + 2 * sy))
            dveldxdy =
                sx * sy *
                (-0.5 * (-1.5 * l(x + 2 * sx, y) + 2 * l(x + 2 * sx, y + sy) -
                         0.5 * l(x + 2 * sx, y + 2 * sy)) +
                 2 * (-1.5 * l(x + sx, y) + 2 * l(x + sx, y + sy) -
                      0.5 * l(x + sx, y + 2 * sy)) -
                 1.5 * (-1.5 * l(x, y) + 2 * l(x, y + sy) -
                        0.5 * l(x, y + 2 * sy)));
          else
            dveldxdy = sx * sy * (l(x + sx, y + sy) - l(x + sx, y)) -
                       (l(x, y + sy) - l(x, y));

          DuDx =
              dveldx.u[0] + dveldx2.u[0] * (ix - x) + dveldxdy.u[0] * (iy - y);
          DvDx =
              dveldx.u[1] + dveldx2.u[1] * (ix - x) + dveldxdy.u[1] * (iy - y);
          DuDy =
              dveldy.u[0] + dveldy2.u[0] * (iy - y) + dveldxdy.u[0] * (ix - x);
          DvDy =
              dveldy.u[1] + dveldy2.u[1] * (iy - y) + dveldxdy.u[1] * (ix - x);
        } // shear stress computation ends here

        // normals computed with Towers 2009
        //  Actually using the volume integral, since (/iint -P /hat{n} dS) =
        //  (/iiint - /nabla P dV). Also, P*/nabla /Chi = /nabla P
        //  penalty-accel and surf-force match up if resolution is high enough
        // const Real fXV = D11*normX + D12*normY, fXP = - P(ix,iy).s * normX;
        // const Real fYV = D12*normX + D22*normY, fYP = - P(ix,iy).s * normY;
        const Real fXV = NUoH * DuDx * normX + NUoH * DuDy * normY,
                   fXP = -P(ix, iy).s * normX;
        const Real fYV = NUoH * DvDx * normX + NUoH * DvDy * normY,
                   fYP = -P(ix, iy).s * normY;

        const Real fXT = fXV + fXP, fYT = fYV + fYP;

        // store:
        O->x_s[k] = p[0];
        O->y_s[k] = p[1];
        O->p_s[k] = P(ix, iy).s;
        O->u_s[k] = V(ix, iy).u[0];
        O->v_s[k] = V(ix, iy).u[1];
        O->nx_s[k] = dx;
        O->ny_s[k] = dy;
        O->omega_s[k] = (DvDx - DuDy) / info.h;
        O->uDef_s[k] = O->udef[iy][ix][0];
        O->vDef_s[k] = O->udef[iy][ix][1];
        O->fX_s[k] = -P(ix, iy).s * dx + NUoH * DuDx * dx +
                     NUoH * DuDy * dy; // scale by 1/h
        O->fY_s[k] = -P(ix, iy).s * dy + NUoH * DvDx * dx +
                     NUoH * DvDy * dy;                     // scale by 1/h
        O->fXv_s[k] = NUoH * DuDx * dx + NUoH * DuDy * dy; // scale by 1/h
        O->fYv_s[k] = NUoH * DvDx * dx + NUoH * DvDy * dy; // scale by 1/h

        // perimeter:
        O->perimeter += std::sqrt(normX * normX + normY * normY);
        O->circulation += normX * O->v_s[k] - normY * O->u_s[k];
        // forces (total, visc, pressure):
        O->forcex += fXT;
        O->forcey += fYT;
        O->forcex_V += fXV;
        O->forcey_V += fYV;
        O->forcex_P += fXP;
        O->forcey_P += fYP;
        // torque:
        O->torque += (p[0] - Cx) * fYT - (p[1] - Cy) * fXT;
        O->torque_P += (p[0] - Cx) * fYP - (p[1] - Cy) * fXP;
        O->torque_V += (p[0] - Cx) * fYV - (p[1] - Cy) * fXV;
        // thrust, drag:
        const Real forcePar = fXT * vel_unit[0] + fYT * vel_unit[1];
        O->thrust += .5 * (forcePar + std::fabs(forcePar));
        O->drag -= .5 * (forcePar - std::fabs(forcePar));
        const Real forcePerp = fXT * vel_unit[1] - fYT * vel_unit[0];
        O->lift += forcePerp;
        // power output (and negative definite variant which ensures no elastic
        // energy absorption)
        //  This is total power, for overcoming not only deformation, but also
        //  the oncoming velocity. Work done by fluid, not by the object (for
        //  that, just take -ve)
        const Real powOut = fXT * O->u_s[k] + fYT * O->v_s[k];
        // deformation power output (and negative definite variant which ensures
        // no elastic energy absorption)
        const Real powDef = fXT * O->uDef_s[k] + fYT * O->vDef_s[k];
        O->Pout += powOut;
        O->defPower += powDef;
        O->PoutBnd += std::min((Real)0, powOut);
        O->defPowerBnd += std::min((Real)0, powDef);
      }
      O->PoutNew = O->forcex * shape->u + O->forcey * shape->v;
    }
  }
};

void ComputeForces::operator()(const Real dt) {
  sim.startProfiler("ComputeForces");
  KernelComputeForces K(sim);
  cubism::compute<KernelComputeForces, VectorGrid, VectorLab, ScalarGrid,
                  ScalarLab>(K, *sim.vel, *sim.chi);

  // finalize partial sums
  for (const auto &shape : sim.shapes)
    shape->computeForces();
  sim.stopProfiler();
}

ComputeForces::ComputeForces(SimulationData &s) : Operator(s) {}

#define profile(func)                                                          \
  do {                                                                         \
  } while (0)
class PoissonSolver {
public:
  virtual ~PoissonSolver() = default;
  virtual void solve(const ScalarGrid *input, ScalarGrid *output) = 0;
};

class ExpAMRSolver : public PoissonSolver {
  /*
  Method used to solve Poisson's equation:
  https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
  */
public:
  std::string getName() {
    // ExpAMRSolver == AMRSolver for explicit linear system
    return "ExpAMRSolver";
  }
  // Constructor and destructor
  ExpAMRSolver(SimulationData &s);
  ~ExpAMRSolver() = default;

  // main function used to solve Poisson's equation
  void solve(const ScalarGrid *input, ScalarGrid *const output);

protected:
  // this struct contains information such as the currect timestep size, fluid
  // properties and many others
  SimulationData &sim;

  int rank_;
  MPI_Comm m_comm_;
  int comm_size_;

  static constexpr int BSX_ = VectorBlock::sizeX;
  static constexpr int BSY_ = VectorBlock::sizeY;
  static constexpr int BLEN_ = BSX_ * BSY_;

  // This returns element K_{I1,I2}. It is used when we invert K
  double getA_local(int I1, int I2);

  // Method to add off-diagonal matrix element associated to cell in 'rhsNei'
  // block
  class EdgeCellIndexer; // forward declaration
  void makeFlux(const cubism::BlockInfo &rhs_info, const int ix, const int iy,
                const cubism::BlockInfo &rhsNei, const EdgeCellIndexer &indexer,
                SpRowInfo &row) const;

  // Method to compute A and b for the current mesh
  void getMat(); // update LHS and RHS after refinement
  void getVec(); // update initial guess and RHS vecs only

  // Distributed linear system which uses local indexing
  std::unique_ptr<LocalSpMatDnVec> LocalLS_;

  std::vector<long long> Nblocks_xcumsum_;
  std::vector<long long> Nrows_xcumsum_;

  // Edge descriptors to allow algorithmic access to cell indices regardless of
  // edge type
  class CellIndexer {
  public:
    CellIndexer(const ExpAMRSolver &pSolver) : ps(pSolver) {}
    ~CellIndexer() = default;

    long long This(const cubism::BlockInfo &info, const int ix,
                   const int iy) const {
      return blockOffset(info) + (long long)(iy * BSX_ + ix);
    }

    static bool validXm(const int ix, const int iy) { return ix > 0; }
    static bool validXp(const int ix, const int iy) { return ix < BSX_ - 1; }
    static bool validYm(const int ix, const int iy) { return iy > 0; }
    static bool validYp(const int ix, const int iy) { return iy < BSY_ - 1; }

    long long Xmin(const cubism::BlockInfo &info, const int ix, const int iy,
                   const int offset = 0) const {
      return blockOffset(info) + (long long)(iy * BSX_ + offset);
    }
    long long Xmax(const cubism::BlockInfo &info, const int ix, const int iy,
                   const int offset = 0) const {
      return blockOffset(info) + (long long)(iy * BSX_ + (BSX_ - 1 - offset));
    }
    long long Ymin(const cubism::BlockInfo &info, const int ix, const int iy,
                   const int offset = 0) const {
      return blockOffset(info) + (long long)(offset * BSX_ + ix);
    }
    long long Ymax(const cubism::BlockInfo &info, const int ix, const int iy,
                   const int offset = 0) const {
      return blockOffset(info) + (long long)((BSY_ - 1 - offset) * BSX_ + ix);
    }

  protected:
    long long blockOffset(const cubism::BlockInfo &info) const {
      return (info.blockID +
              ps.Nblocks_xcumsum_[ps.sim.tmp->Tree(info).rank()]) *
             BLEN_;
    }
    static int ix_f(const int ix) { return (ix % (BSX_ / 2)) * 2; }
    static int iy_f(const int iy) { return (iy % (BSY_ / 2)) * 2; }

    const ExpAMRSolver &ps; // poisson solver
  };

  class EdgeCellIndexer : public CellIndexer {
  public:
    EdgeCellIndexer(const ExpAMRSolver &pSolver) : CellIndexer(pSolver) {}

    // When I am uniform with the neighbouring block
    virtual long long neiUnif(const cubism::BlockInfo &nei_info, const int ix,
                              const int iy) const = 0;

    // When I am finer than neighbouring block
    virtual long long neiInward(const cubism::BlockInfo &info, const int ix,
                                const int iy) const = 0;
    virtual double taylorSign(const int ix, const int iy) const = 0;

    // Indices of coarses cells in neighbouring blocks, to be overridden where
    // appropriate
    virtual int ix_c(const cubism::BlockInfo &info, const int ix) const {
      return info.index[0] % 2 == 0 ? ix / 2 : ix / 2 + BSX_ / 2;
    }
    virtual int iy_c(const cubism::BlockInfo &info, const int iy) const {
      return info.index[1] % 2 == 0 ? iy / 2 : iy / 2 + BSY_ / 2;
    }

    // When I am coarser than neighbouring block
    // neiFine1 must correspond to cells where taylorSign == -1., neiFine2 must
    // correspond to taylorSign == 1.
    virtual long long neiFine1(const cubism::BlockInfo &nei_info, const int ix,
                               const int iy, const int offset = 0) const = 0;
    virtual long long neiFine2(const cubism::BlockInfo &nei_info, const int ix,
                               const int iy, const int offset = 0) const = 0;

    // Indexing aids for derivatives in Taylor approximation in coarse cell
    virtual bool isBD(const int ix, const int iy) const = 0;
    virtual bool isFD(const int ix, const int iy) const = 0;
    virtual long long Nei(const cubism::BlockInfo &info, const int ix,
                          const int iy, const int dist) const = 0;

    // When I am coarser and need to determine which Zchild I'm next to
    virtual long long Zchild(const cubism::BlockInfo &nei_info, const int ix,
                             const int iy) const = 0;
  };

  // ----------------------------------------------------- Edges perpendicular
  // to x-axis -----------------------------------
  class XbaseIndexer : public EdgeCellIndexer {
  public:
    XbaseIndexer(const ExpAMRSolver &pSolver) : EdgeCellIndexer(pSolver) {}

    double taylorSign(const int ix, const int iy) const override {
      return iy % 2 == 0 ? -1. : 1.;
    }
    bool isBD(const int ix, const int iy) const override {
      return iy == BSY_ - 1 || iy == BSY_ / 2 - 1;
    }
    bool isFD(const int ix, const int iy) const override {
      return iy == 0 || iy == BSY_ / 2;
    }
    long long Nei(const cubism::BlockInfo &info, const int ix, const int iy,
                  const int dist) const override {
      return This(info, ix, iy + dist);
    }
  };

  class XminIndexer : public XbaseIndexer {
  public:
    XminIndexer(const ExpAMRSolver &pSolver) : XbaseIndexer(pSolver) {}

    long long neiUnif(const cubism::BlockInfo &nei_info, const int ix,
                      const int iy) const override {
      return Xmax(nei_info, ix, iy);
    }

    long long neiInward(const cubism::BlockInfo &info, const int ix,
                        const int iy) const override {
      return This(info, ix + 1, iy);
    }

    int ix_c(const cubism::BlockInfo &info, const int ix) const override {
      return BSX_ - 1;
    }

    long long neiFine1(const cubism::BlockInfo &nei_info, const int ix,
                       const int iy, const int offset = 0) const override {
      return Xmax(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const cubism::BlockInfo &nei_info, const int ix,
                       const int iy, const int offset = 0) const override {
      return Xmax(nei_info, ix_f(ix), iy_f(iy) + 1, offset);
    }

    long long Zchild(const cubism::BlockInfo &nei_info, const int ix,
                     const int iy) const override {
      return nei_info.Zchild[1][int(iy >= BSY_ / 2)][0];
    }
  };

  class XmaxIndexer : public XbaseIndexer {
  public:
    XmaxIndexer(const ExpAMRSolver &pSolver) : XbaseIndexer(pSolver) {}

    long long neiUnif(const cubism::BlockInfo &nei_info, const int ix,
                      const int iy) const override {
      return Xmin(nei_info, ix, iy);
    }

    long long neiInward(const cubism::BlockInfo &info, const int ix,
                        const int iy) const override {
      return This(info, ix - 1, iy);
    }

    int ix_c(const cubism::BlockInfo &info, const int ix) const override {
      return 0;
    }

    long long neiFine1(const cubism::BlockInfo &nei_info, const int ix,
                       const int iy, const int offset = 0) const override {
      return Xmin(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const cubism::BlockInfo &nei_info, const int ix,
                       const int iy, const int offset = 0) const override {
      return Xmin(nei_info, ix_f(ix), iy_f(iy) + 1, offset);
    }

    long long Zchild(const cubism::BlockInfo &nei_info, const int ix,
                     const int iy) const override {
      return nei_info.Zchild[0][int(iy >= BSY_ / 2)][0];
    }
  };

  // ----------------------------------------------------- Edges perpendicular
  // to y-axis -----------------------------------
  class YbaseIndexer : public EdgeCellIndexer {
  public:
    YbaseIndexer(const ExpAMRSolver &pSolver) : EdgeCellIndexer(pSolver) {}

    double taylorSign(const int ix, const int iy) const override {
      return ix % 2 == 0 ? -1. : 1.;
    }
    bool isBD(const int ix, const int iy) const override {
      return ix == BSX_ - 1 || ix == BSX_ / 2 - 1;
    }
    bool isFD(const int ix, const int iy) const override {
      return ix == 0 || ix == BSX_ / 2;
    }
    long long Nei(const cubism::BlockInfo &info, const int ix, const int iy,
                  const int dist) const override {
      return This(info, ix + dist, iy);
    }
  };

  class YminIndexer : public YbaseIndexer {
  public:
    YminIndexer(const ExpAMRSolver &pSolver) : YbaseIndexer(pSolver) {}

    long long neiUnif(const cubism::BlockInfo &nei_info, const int ix,
                      const int iy) const override {
      return Ymax(nei_info, ix, iy);
    }

    long long neiInward(const cubism::BlockInfo &info, const int ix,
                        const int iy) const override {
      return This(info, ix, iy + 1);
    }

    int iy_c(const cubism::BlockInfo &info, const int iy) const override {
      return BSY_ - 1;
    }

    long long neiFine1(const cubism::BlockInfo &nei_info, const int ix,
                       const int iy, const int offset = 0) const override {
      return Ymax(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const cubism::BlockInfo &nei_info, const int ix,
                       const int iy, const int offset = 0) const override {
      return Ymax(nei_info, ix_f(ix) + 1, iy_f(iy), offset);
    }

    long long Zchild(const cubism::BlockInfo &nei_info, const int ix,
                     const int iy) const override {
      return nei_info.Zchild[int(ix >= BSX_ / 2)][1][0];
    }
  };

  class YmaxIndexer : public YbaseIndexer {
  public:
    YmaxIndexer(const ExpAMRSolver &pSolver) : YbaseIndexer(pSolver) {}

    long long neiUnif(const cubism::BlockInfo &nei_info, const int ix,
                      const int iy) const override {
      return Ymin(nei_info, ix, iy);
    }

    long long neiInward(const cubism::BlockInfo &info, const int ix,
                        const int iy) const override {
      return This(info, ix, iy - 1);
    }

    int iy_c(const cubism::BlockInfo &info, const int iy) const override {
      return 0;
    }

    long long neiFine1(const cubism::BlockInfo &nei_info, const int ix,
                       const int iy, const int offset = 0) const override {
      return Ymin(nei_info, ix_f(ix), iy_f(iy), offset);
    }
    long long neiFine2(const cubism::BlockInfo &nei_info, const int ix,
                       const int iy, const int offset = 0) const override {
      return Ymin(nei_info, ix_f(ix) + 1, iy_f(iy), offset);
    }

    long long Zchild(const cubism::BlockInfo &nei_info, const int ix,
                     const int iy) const override {
      return nei_info.Zchild[int(ix >= BSX_ / 2)][0][0];
    }
  };

  CellIndexer GenericCell;
  XminIndexer XminCell;
  XmaxIndexer XmaxCell;
  YminIndexer YminCell;
  YmaxIndexer YmaxCell;
  // Array of pointers for the indexers above for polymorphism in makeFlux
  std::array<const EdgeCellIndexer *, 4> edgeIndexers;

  std::array<std::pair<long long, double>, 3> D1(const cubism::BlockInfo &info,
                                                 const EdgeCellIndexer &indexer,
                                                 const int ix,
                                                 const int iy) const {
    // Scale D1 by h^l/4
    if (indexer.isBD(ix, iy))
      return {{{indexer.Nei(info, ix, iy, -2), 1. / 8.},
               {indexer.Nei(info, ix, iy, -1), -1. / 2.},
               {indexer.This(info, ix, iy), 3. / 8.}}};
    else if (indexer.isFD(ix, iy))
      return {{{indexer.Nei(info, ix, iy, 2), -1. / 8.},
               {indexer.Nei(info, ix, iy, 1), 1. / 2.},
               {indexer.This(info, ix, iy), -3. / 8.}}};

    return {{{indexer.Nei(info, ix, iy, -1), -1. / 8.},
             {indexer.Nei(info, ix, iy, 1), 1. / 8.},
             {indexer.This(info, ix, iy), 0.}}};
  }

  std::array<std::pair<long long, double>, 3> D2(const cubism::BlockInfo &info,
                                                 const EdgeCellIndexer &indexer,
                                                 const int ix,
                                                 const int iy) const {
    // Scale D2 by 0.5*(h^l/4)^2
    if (indexer.isBD(ix, iy))
      return {{{indexer.Nei(info, ix, iy, -2), 1. / 32.},
               {indexer.Nei(info, ix, iy, -1), -1. / 16.},
               {indexer.This(info, ix, iy), 1. / 32.}}};
    else if (indexer.isFD(ix, iy))
      return {{{indexer.Nei(info, ix, iy, 2), 1. / 32.},
               {indexer.Nei(info, ix, iy, 1), -1. / 16.},
               {indexer.This(info, ix, iy), 1. / 32.}}};

    return {{{indexer.Nei(info, ix, iy, -1), 1. / 32.},
             {indexer.Nei(info, ix, iy, 1), 1. / 32.},
             {indexer.This(info, ix, iy), -1. / 16.}}};
  }

  void interpolate(const cubism::BlockInfo &info_c, const int ix_c,
                   const int iy_c, const cubism::BlockInfo &info_f,
                   const long long fine_close_idx, const long long fine_far_idx,
                   const double signI, const double signT,
                   const EdgeCellIndexer &indexer, SpRowInfo &row) const;
};

double ExpAMRSolver::getA_local(
    int I1, int I2) // matrix for Poisson's equation on a uniform grid
{
  int j1 = I1 / BSX_;
  int i1 = I1 % BSX_;
  int j2 = I2 / BSX_;
  int i2 = I2 % BSX_;
  if (i1 == i2 && j1 == j2)
    return 4.0;
  else if (abs(i1 - i2) + abs(j1 - j2) == 1)
    return -1.0;
  else
    return 0.0;
}

ExpAMRSolver::ExpAMRSolver(SimulationData &s)
    : sim(s), m_comm_(sim.comm), GenericCell(*this), XminCell(*this),
      XmaxCell(*this), YminCell(*this), YmaxCell(*this),
      edgeIndexers{&XminCell, &XmaxCell, &YminCell, &YmaxCell} {
  // MPI
  MPI_Comm_rank(m_comm_, &rank_);
  MPI_Comm_size(m_comm_, &comm_size_);

  Nblocks_xcumsum_.resize(comm_size_ + 1);
  Nrows_xcumsum_.resize(comm_size_ + 1);

  std::vector<std::vector<double>>
      L; // lower triangular matrix of Cholesky decomposition
  std::vector<std::vector<double>> L_inv; // inverse of L

  L.resize(BLEN_);
  L_inv.resize(BLEN_);
  for (int i(0); i < BLEN_; i++) {
    L[i].resize(i + 1);
    L_inv[i].resize(i + 1);
    // L_inv will act as right block in GJ algorithm, init as identity
    for (int j(0); j <= i; j++) {
      L_inv[i][j] = (i == j) ? 1. : 0.;
    }
  }

  // compute the Cholesky decomposition of the preconditioner with
  // Cholesky-Crout
  for (int i(0); i < BLEN_; i++) {
    double s1 = 0;
    for (int k(0); k <= i - 1; k++)
      s1 += L[i][k] * L[i][k];
    L[i][i] = sqrt(getA_local(i, i) - s1);
    for (int j(i + 1); j < BLEN_; j++) {
      double s2 = 0;
      for (int k(0); k <= i - 1; k++)
        s2 += L[i][k] * L[j][k];
      L[j][i] = (getA_local(j, i) - s2) / L[i][i];
    }
  }

  /* Compute the inverse of the Cholesky decomposition L using Gauss-Jordan
     elimination. L will act as the left block (it does not need to be modified
     in the process), L_inv will act as the right block and at the end of the
     algo will contain the inverse */
  for (int br(0); br < BLEN_; br++) { // 'br' - base row in which all columns up
                                      // to L_lb[br][br] are already zero
    const double bsf = 1. / L[br][br];
    for (int c(0); c <= br; c++)
      L_inv[br][c] *= bsf;

    for (int wr(br + 1); wr < BLEN_;
         wr++) { // 'wr' - working row where elements below L_lb[br][br] will be
                 // set to zero
      const double wsf = L[wr][br];
      for (int c(0); c <= br; c++)
        L_inv[wr][c] -= (wsf * L_inv[br][c]);
    }
  }

  // P_inv_ holds inverse preconditionner in row major order!
  std::vector<double> P_inv(BLEN_ * BLEN_);
  for (int i(0); i < BLEN_; i++)
    for (int j(0); j < BLEN_; j++) {
      double aux = 0.;
      for (int k(0); k < BLEN_; k++) // P_inv_ = (L^T)^{-1} L^{-1}
        aux += (i <= k && j <= k) ? L_inv[k][i] * L_inv[k][j] : 0.;

      P_inv[i * BLEN_ + j] =
          -aux; // Up to now Cholesky of negative P to avoid complex numbers
    }

  // Create Linear system and backend solver objects
  LocalLS_ = std::make_unique<LocalSpMatDnVec>(m_comm_, BSX_ * BSY_,
                                               sim.bMeanConstraint, P_inv);
}
void ExpAMRSolver::interpolate(
    const cubism::BlockInfo &info_c, const int ix_c, const int iy_c,
    const cubism::BlockInfo &info_f, const long long fine_close_idx,
    const long long fine_far_idx, const double signInt,
    const double signTaylor, // sign of interpolation and sign of taylor
    const EdgeCellIndexer &indexer, SpRowInfo &row) const {
  const int rank_c = sim.tmp->Tree(info_c).rank();
  const int rank_f = sim.tmp->Tree(info_f).rank();

  // 2./3.*p_fine_close_idx - 1./5.*p_fine_far_idx
  row.mapColVal(rank_f, fine_close_idx, signInt * 2. / 3.);
  row.mapColVal(rank_f, fine_far_idx, -signInt * 1. / 5.);

  // 8./15 * p_T, constant term
  const double tf =
      signInt * 8. / 15.; // common factor for all terms of Taylor expansion
  row.mapColVal(rank_c, indexer.This(info_c, ix_c, iy_c), tf);

  std::array<std::pair<long long, double>, 3> D;

  // first derivative
  D = D1(info_c, indexer, ix_c, iy_c);
  for (int i(0); i < 3; i++)
    row.mapColVal(rank_c, D[i].first, signTaylor * tf * D[i].second);

  // second derivative
  D = D2(info_c, indexer, ix_c, iy_c);
  for (int i(0); i < 3; i++)
    row.mapColVal(rank_c, D[i].first, tf * D[i].second);
}

// Methods for cell centric construction of discrete Laplace operator
void ExpAMRSolver::makeFlux(const cubism::BlockInfo &rhs_info, const int ix,
                            const int iy, const cubism::BlockInfo &rhsNei,
                            const EdgeCellIndexer &indexer,
                            SpRowInfo &row) const {
  const long long sfc_idx = indexer.This(rhs_info, ix, iy);

  if (this->sim.tmp->Tree(rhsNei).Exists()) {
    const int nei_rank = sim.tmp->Tree(rhsNei).rank();
    const long long nei_idx = indexer.neiUnif(rhsNei, ix, iy);

    // Map flux associated to out-of-block edges at the same level of refinement
    row.mapColVal(nei_rank, nei_idx, 1.);
    row.mapColVal(sfc_idx, -1.);
  } else if (this->sim.tmp->Tree(rhsNei).CheckCoarser()) {
    const cubism::BlockInfo &rhsNei_c =
        this->sim.tmp->getBlockInfoAll(rhs_info.level - 1, rhsNei.Zparent);
    const int ix_c = indexer.ix_c(rhs_info, ix);
    const int iy_c = indexer.iy_c(rhs_info, iy);
    const long long inward_idx = indexer.neiInward(rhs_info, ix, iy);
    const double signTaylor = indexer.taylorSign(ix, iy);

    interpolate(rhsNei_c, ix_c, iy_c, rhs_info, sfc_idx, inward_idx, 1.,
                signTaylor, indexer, row);
    row.mapColVal(sfc_idx, -1.);
  } else if (this->sim.tmp->Tree(rhsNei).CheckFiner()) {
    const cubism::BlockInfo &rhsNei_f = this->sim.tmp->getBlockInfoAll(
        rhs_info.level + 1, indexer.Zchild(rhsNei, ix, iy));
    const int nei_rank = this->sim.tmp->Tree(rhsNei_f).rank();

    // F1
    long long fine_close_idx = indexer.neiFine1(rhsNei_f, ix, iy, 0);
    long long fine_far_idx = indexer.neiFine1(rhsNei_f, ix, iy, 1);
    row.mapColVal(nei_rank, fine_close_idx, 1.);
    interpolate(rhs_info, ix, iy, rhsNei_f, fine_close_idx, fine_far_idx, -1.,
                -1., indexer, row);
    // F2
    fine_close_idx = indexer.neiFine2(rhsNei_f, ix, iy, 0);
    fine_far_idx = indexer.neiFine2(rhsNei_f, ix, iy, 1);
    row.mapColVal(nei_rank, fine_close_idx, 1.);
    interpolate(rhs_info, ix, iy, rhsNei_f, fine_close_idx, fine_far_idx, -1.,
                1., indexer, row);
  } else {
    throw std::runtime_error(
        "Neighbour doesn't exist, isn't coarser, nor finer...");
  }
}

void ExpAMRSolver::getMat() {
  sim.startProfiler("Poisson solver: LS");

  // This returns an array with the blocks that the coarsest possible
  // mesh would have (i.e. all blocks are at level 0)
  std::array<int, 3> blocksPerDim = sim.pres->getMaxBlocks();

  // Get a vector of all BlockInfos of the grid we're interested in
  sim.tmp->UpdateBlockInfoAll_States(
      true); // update blockID's for blocks from other ranks
  std::vector<cubism::BlockInfo> &RhsInfo = sim.tmp->getBlocksInfo();
  const int Nblocks = RhsInfo.size();
  const int N = BSX_ * BSY_ * Nblocks;

  // Reserve sufficient memory for LS proper to the rank
  LocalLS_->reserve(N);

  // Calculate cumulative sums for blocks and rows for correct global indexing
  const long long Nblocks_long = Nblocks;
  MPI_Allgather(&Nblocks_long, 1, MPI_LONG_LONG, Nblocks_xcumsum_.data(), 1,
                MPI_LONG_LONG, m_comm_);
  for (int i(Nblocks_xcumsum_.size() - 1); i > 0; i--) {
    Nblocks_xcumsum_[i] =
        Nblocks_xcumsum_[i - 1]; // shift to right for rank 'i+1' to have cumsum
                                 // of rank 'i'
  }

  // Set cumsum for rank 0 to zero
  Nblocks_xcumsum_[0] = 0;
  Nrows_xcumsum_[0] = 0;

  // Perform cumulative sum
  for (size_t i(1); i < Nblocks_xcumsum_.size(); i++) {
    Nblocks_xcumsum_[i] += Nblocks_xcumsum_[i - 1];
    Nrows_xcumsum_[i] = BLEN_ * Nblocks_xcumsum_[i];
  }

  // No parallel for to ensure COO are ordered at construction
  for (int i = 0; i < Nblocks; i++) {
    const cubism::BlockInfo &rhs_info = RhsInfo[i];

    // 1.Check if this is a boundary block
    const int aux = 1 << rhs_info.level; // = 2^level
    const int MAX_X_BLOCKS =
        blocksPerDim[0] * aux -
        1; // this means that if level 0 has blocksPerDim[0] blocks in the
           // x-direction, level rhs.level will have this many blocks
    const int MAX_Y_BLOCKS =
        blocksPerDim[1] * aux -
        1; // this means that if level 0 has blocksPerDim[1] blocks in the
           // y-direction, level rhs.level will have this many blocks

    // index is the (i,j) coordinates of a block at the current level
    std::array<bool, 4> isBoundary;
    isBoundary[0] = (rhs_info.index[0] ==
                     0); // Xm, same order as faceIndexers made in constructor!
    isBoundary[1] = (rhs_info.index[0] == MAX_X_BLOCKS); // Xp
    isBoundary[2] = (rhs_info.index[1] == 0);            // Ym
    isBoundary[3] = (rhs_info.index[1] == MAX_Y_BLOCKS); // Yp

    std::array<bool, 2> isPeriodic; // same dimension ordering as isBoundary
    isPeriodic[0] = (cubismBCX == periodic);
    isPeriodic[1] = (cubismBCY == periodic);

    // 2.Access the block's neighbors (for the Poisson solve in two dimensions
    // we care about four neighbors in total)
    std::array<long long, 4> Z;
    Z[0] = rhs_info.Znei[1 - 1][1][1]; // Xm
    Z[1] = rhs_info.Znei[1 + 1][1][1]; // Xp
    Z[2] = rhs_info.Znei[1][1 - 1][1]; // Ym
    Z[3] = rhs_info.Znei[1][1 + 1][1]; // Yp
    // rhs.Z == rhs.Znei[1][1][1] is true always

    std::array<const cubism::BlockInfo *, 4> rhsNei;
    rhsNei[0] = &(this->sim.tmp->getBlockInfoAll(rhs_info.level, Z[0]));
    rhsNei[1] = &(this->sim.tmp->getBlockInfoAll(rhs_info.level, Z[1]));
    rhsNei[2] = &(this->sim.tmp->getBlockInfoAll(rhs_info.level, Z[2]));
    rhsNei[3] = &(this->sim.tmp->getBlockInfoAll(rhs_info.level, Z[3]));

    // Record local index of row which is to be modified with bMeanConstraint
    // reduction result
    if (sim.bMeanConstraint && rhs_info.index[0] == 0 &&
        rhs_info.index[1] == 0 && rhs_info.index[2] == 0)
      LocalLS_->set_bMeanRow(GenericCell.This(rhs_info, 0, 0) -
                             Nrows_xcumsum_[rank_]);

    // For later: there's a total of three boolean variables:
    //  I.   grid->Tree(rhsNei_west).Exists()
    //  II.  grid->Tree(rhsNei_west).CheckCoarser()
    //  III. grid->Tree(rhsNei_west).CheckFiner()
    //  And only one of them is true

    // Add matrix elements associated to interior cells of a block
    for (int iy = 0; iy < BSY_; iy++)
      for (int ix = 0; ix < BSX_; ix++) { // Following logic needs to be in for
                                          // loop to assure cooRows are ordered
        const long long sfc_idx = GenericCell.This(rhs_info, ix, iy);

        if ((ix > 0 && ix < BSX_ - 1) &&
            (iy > 0 && iy < BSY_ - 1)) { // Inner cells, push back in ascending
                                         // order for column index
          LocalLS_->cooPushBackVal(1, sfc_idx,
                                   GenericCell.This(rhs_info, ix, iy - 1));
          LocalLS_->cooPushBackVal(1, sfc_idx,
                                   GenericCell.This(rhs_info, ix - 1, iy));
          LocalLS_->cooPushBackVal(-4, sfc_idx, sfc_idx);
          LocalLS_->cooPushBackVal(1, sfc_idx,
                                   GenericCell.This(rhs_info, ix + 1, iy));
          LocalLS_->cooPushBackVal(1, sfc_idx,
                                   GenericCell.This(rhs_info, ix, iy + 1));
        } else { // See which edge is shared with a cell from different block
          std::array<bool, 4> validNei;
          validNei[0] = GenericCell.validXm(ix, iy);
          validNei[1] = GenericCell.validXp(ix, iy);
          validNei[2] = GenericCell.validYm(ix, iy);
          validNei[3] = GenericCell.validYp(ix, iy);

          // Get index of cell accross the edge (correct only for cells in this
          // block)
          std::array<long long, 4> idxNei;
          idxNei[0] = GenericCell.This(rhs_info, ix - 1, iy);
          idxNei[1] = GenericCell.This(rhs_info, ix + 1, iy);
          idxNei[2] = GenericCell.This(rhs_info, ix, iy - 1);
          idxNei[3] = GenericCell.This(rhs_info, ix, iy + 1);

          SpRowInfo row(sim.tmp->Tree(rhs_info).rank(), sfc_idx, 8);
          for (int j(0); j < 4; j++) { // Iterate over each edge of cell
            if (validNei[j]) {         // This edge is 'inner' wrt to the block
              row.mapColVal(idxNei[j], 1);
              row.mapColVal(sfc_idx, -1);
            } else if (!isBoundary[j] || (isBoundary[j] && isPeriodic[j / 2]))
              this->makeFlux(rhs_info, ix, iy, *rhsNei[j], *edgeIndexers[j],
                             row);
          }

          LocalLS_->cooPushBackRow(row);
        }
      } // for(int iy=0; iy<BSY_; iy++) for(int ix=0; ix<BSX_; ix++)
  } // for(int i=0; i< Nblocks; i++)

  LocalLS_->make(Nrows_xcumsum_);

  sim.stopProfiler();
}

void ExpAMRSolver::getVec() {
  // Get a vector of all BlockInfos of the grid we're interested in
  std::vector<cubism::BlockInfo> &RhsInfo = sim.tmp->getBlocksInfo();
  std::vector<cubism::BlockInfo> &zInfo = sim.pres->getBlocksInfo();
  const int Nblocks = RhsInfo.size();
  std::vector<double> &x = LocalLS_->get_x();
  std::vector<double> &b = LocalLS_->get_b();
  std::vector<double> &h2 = LocalLS_->get_h2();
  const long long shift = -Nrows_xcumsum_[rank_];

// Copy RHS LHS vec initial guess, if LS was updated, updateMat reallocates
// sufficient memory
#pragma omp parallel for
  for (int i = 0; i < Nblocks; i++) {
    const cubism::BlockInfo &rhs_info = RhsInfo[i];
    const ScalarBlock &__restrict__ rhs = *(ScalarBlock *)RhsInfo[i].ptrBlock;
    const ScalarBlock &__restrict__ p = *(ScalarBlock *)zInfo[i].ptrBlock;

    h2[i] = RhsInfo[i].h * RhsInfo[i].h;
    // Construct RHS and x_0 vectors for linear system
    for (int iy = 0; iy < BSY_; iy++)
      for (int ix = 0; ix < BSX_; ix++) {
        const long long sfc_loc = GenericCell.This(rhs_info, ix, iy) + shift;
        if (sim.bMeanConstraint && rhs_info.index[0] == 0 &&
            rhs_info.index[1] == 0 && rhs_info.index[2] == 0 && ix == 0 &&
            iy == 0)
          b[sfc_loc] = 0.;
        else
          b[sfc_loc] = rhs(ix, iy).s;

        x[sfc_loc] = p(ix, iy).s;
      }
  }
}

void ExpAMRSolver::solve(const ScalarGrid *input, ScalarGrid *const output) {

  if (rank_ == 0) {
    if (sim.verbose)
      std::cout << "--------------------- Calling on ExpAMRSolver.solve() "
                   "------------------------\n";
    else
      std::cout << '\n';
  }

  const double max_error = this->sim.step < 10 ? 0.0 : sim.PoissonTol;
  const double max_rel_error = this->sim.step < 10 ? 0.0 : sim.PoissonTolRel;
  const int max_restarts = this->sim.step < 10 ? 100 : sim.maxPoissonRestarts;

  if (sim.pres->UpdateFluxCorrection) {
    sim.pres->UpdateFluxCorrection = false;
    this->getMat();
    this->getVec();
    LocalLS_->solveWithUpdate(max_error, max_rel_error, max_restarts);
  } else {
    this->getVec();
    LocalLS_->solveNoUpdate(max_error, max_rel_error, max_restarts);
  }

  // Now that we found the solution, we just substract the mean to get a
  // zero-mean solution. This can be done because the solver only cares about
  // grad(P) = grad(P-mean(P))
  std::vector<cubism::BlockInfo> &zInfo = sim.pres->getBlocksInfo();
  const int Nblocks = zInfo.size();
  const std::vector<double> &x = LocalLS_->get_x();

  double avg = 0;
  double avg1 = 0;
#pragma omp parallel for reduction(+ : avg, avg1)
  for (int i = 0; i < Nblocks; i++) {
    ScalarBlock &P = *(ScalarBlock *)zInfo[i].ptrBlock;
    const double vv = zInfo[i].h * zInfo[i].h;
    for (int iy = 0; iy < BSY_; iy++)
      for (int ix = 0; ix < BSX_; ix++) {
        P(ix, iy).s = x[i * BSX_ * BSY_ + iy * BSX_ + ix];
        avg += P(ix, iy).s * vv;
        avg1 += vv;
      }
  }
  double quantities[2] = {avg, avg1};
  MPI_Allreduce(MPI_IN_PLACE, &quantities, 2, MPI_DOUBLE, MPI_SUM, m_comm_);
  avg = quantities[0];
  avg1 = quantities[1];
  avg = avg / avg1;
#pragma omp parallel for
  for (int i = 0; i < Nblocks; i++) {
    ScalarBlock &P = *(ScalarBlock *)zInfo[i].ptrBlock;
    for (int iy = 0; iy < BSY_; iy++)
      for (int ix = 0; ix < BSX_; ix++)
        P(ix, iy).s += -avg;
  }
}

std::shared_ptr<PoissonSolver> makePoissonSolver(SimulationData &s);
std::shared_ptr<PoissonSolver> makePoissonSolver(SimulationData &s) {
  if (s.poissonSolver == "cuda_iterative") {
#ifdef GPU_POISSON
    if (!_DOUBLE_PRECISION_)
      throw std::runtime_error(
          "Poisson solver: \"" + s.poissonSolver +
          "\" must be compiled with in double precision mode!");
    return std::make_shared<ExpAMRSolver>(s);
#else
    throw std::runtime_error(
        "Poisson solver: \"" + s.poissonSolver +
        "\" must be compiled with the -DGPU_POISSON flag!");
#endif
  } else {
    throw std::invalid_argument("Poisson solver: \"" + s.poissonSolver +
                                "\" unrecognized!");
  }
}

class Shape;

class PressureSingle : public Operator {
protected:
  const std::vector<cubism::BlockInfo> &velInfo = sim.vel->getBlocksInfo();

  std::shared_ptr<PoissonSolver> pressureSolver;

  void preventCollidingObstacles() const;
  void pressureCorrection(const Real dt);
  void integrateMomenta(Shape *const shape) const;
  void penalize(const Real dt) const;

public:
  void operator()(const Real dt) override;

  PressureSingle(SimulationData &s);
  ~PressureSingle();

  std::string getName() override { return "PressureSingle"; }
};

using CHI_MAT = Real[VectorBlock::sizeY][VectorBlock::sizeX];
using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];

// #define EXPL_INTEGRATE_MOM

namespace {

void ComputeJ(const Real *Rc, const Real *R, const Real *N, const Real *I,
              Real *J) {
  // Invert I
  const Real m00 = 1.0;  // I[0]; //set to these values for 2D!
  const Real m01 = 0.0;  // I[3]; //set to these values for 2D!
  const Real m02 = 0.0;  // I[4]; //set to these values for 2D!
  const Real m11 = 1.0;  // I[1]; //set to these values for 2D!
  const Real m12 = 0.0;  // I[5]; //set to these values for 2D!
  const Real m22 = I[5]; // I[2]; //set to these values for 2D!
  Real a00 = m22 * m11 - m12 * m12;
  Real a01 = m02 * m12 - m22 * m01;
  Real a02 = m01 * m12 - m02 * m11;
  Real a11 = m22 * m00 - m02 * m02;
  Real a12 = m01 * m02 - m00 * m12;
  Real a22 = m00 * m11 - m01 * m01;
  const Real determinant = 1.0 / ((m00 * a00) + (m01 * a01) + (m02 * a02));
  a00 *= determinant;
  a01 *= determinant;
  a02 *= determinant;
  a11 *= determinant;
  a12 *= determinant;
  a22 *= determinant;

  const Real aux_0 = (Rc[1] - R[1]) * N[2] - (Rc[2] - R[2]) * N[1];
  const Real aux_1 = (Rc[2] - R[2]) * N[0] - (Rc[0] - R[0]) * N[2];
  const Real aux_2 = (Rc[0] - R[0]) * N[1] - (Rc[1] - R[1]) * N[0];
  J[0] = a00 * aux_0 + a01 * aux_1 + a02 * aux_2;
  J[1] = a01 * aux_0 + a11 * aux_1 + a12 * aux_2;
  J[2] = a02 * aux_0 + a12 * aux_1 + a22 * aux_2;
}

void ElasticCollision(const Real m1, const Real m2, const Real *I1,
                      const Real *I2, const Real *v1, const Real *v2,
                      const Real *o1, const Real *o2, Real *hv1, Real *hv2,
                      Real *ho1, Real *ho2, const Real *C1, const Real *C2,
                      const Real NX, const Real NY, const Real NZ,
                      const Real CX, const Real CY, const Real CZ, Real *vc1,
                      Real *vc2) {
  const Real e = 1.0; // coefficient of restitution
  const Real N[3] = {NX, NY, NZ};
  const Real C[3] = {CX, CY, CZ};

  const Real k1[3] = {N[0] / m1, N[1] / m1, N[2] / m1};
  const Real k2[3] = {-N[0] / m2, -N[1] / m2, -N[2] / m2};
  Real J1[3];
  Real J2[3];
  ComputeJ(C, C1, N, I1, J1);
  ComputeJ(C, C2, N, I2, J2);
  J2[0] = -J2[0];
  J2[1] = -J2[1];
  J2[2] = -J2[2];

  Real u1DEF[3];
  u1DEF[0] = vc1[0] - v1[0] - (o1[1] * (C[2] - C1[2]) - o1[2] * (C[1] - C1[1]));
  u1DEF[1] = vc1[1] - v1[1] - (o1[2] * (C[0] - C1[0]) - o1[0] * (C[2] - C1[2]));
  u1DEF[2] = vc1[2] - v1[2] - (o1[0] * (C[1] - C1[1]) - o1[1] * (C[0] - C1[0]));
  Real u2DEF[3];
  u2DEF[0] = vc2[0] - v2[0] - (o2[1] * (C[2] - C2[2]) - o2[2] * (C[1] - C2[1]));
  u2DEF[1] = vc2[1] - v2[1] - (o2[2] * (C[0] - C2[0]) - o2[0] * (C[2] - C2[2]));
  u2DEF[2] = vc2[2] - v2[2] - (o2[0] * (C[1] - C2[1]) - o2[1] * (C[0] - C2[0]));

  const Real nom = e * ((vc1[0] - vc2[0]) * N[0] + (vc1[1] - vc2[1]) * N[1] +
                        (vc1[2] - vc2[2]) * N[2]) +
                   ((v1[0] - v2[0] + u1DEF[0] - u2DEF[0]) * N[0] +
                    (v1[1] - v2[1] + u1DEF[1] - u2DEF[1]) * N[1] +
                    (v1[2] - v2[2] + u1DEF[2] - u2DEF[2]) * N[2]) +
                   ((o1[1] * (C[2] - C1[2]) - o1[2] * (C[1] - C1[1])) * N[0] +
                    (o1[2] * (C[0] - C1[0]) - o1[0] * (C[2] - C1[2])) * N[1] +
                    (o1[0] * (C[1] - C1[1]) - o1[1] * (C[0] - C1[0])) * N[2]) -
                   ((o2[1] * (C[2] - C2[2]) - o2[2] * (C[1] - C2[1])) * N[0] +
                    (o2[2] * (C[0] - C2[0]) - o2[0] * (C[2] - C2[2])) * N[1] +
                    (o2[0] * (C[1] - C2[1]) - o2[1] * (C[0] - C2[0])) * N[2]);

  const Real denom =
      -(1.0 / m1 + 1.0 / m2) +
      +((J1[1] * (C[2] - C1[2]) - J1[2] * (C[1] - C1[1])) * (-N[0]) +
        (J1[2] * (C[0] - C1[0]) - J1[0] * (C[2] - C1[2])) * (-N[1]) +
        (J1[0] * (C[1] - C1[1]) - J1[1] * (C[0] - C1[0])) * (-N[2])) -
      ((J2[1] * (C[2] - C2[2]) - J2[2] * (C[1] - C2[1])) * (-N[0]) +
       (J2[2] * (C[0] - C2[0]) - J2[0] * (C[2] - C2[2])) * (-N[1]) +
       (J2[0] * (C[1] - C2[1]) - J2[1] * (C[0] - C2[0])) * (-N[2]));
  const Real impulse = nom / (denom + 1e-21);
  hv1[0] = v1[0] + k1[0] * impulse;
  hv1[1] = v1[1] + k1[1] * impulse;
  hv1[2] = v1[2] + k1[2] * impulse;
  hv2[0] = v2[0] + k2[0] * impulse;
  hv2[1] = v2[1] + k2[1] * impulse;
  hv2[2] = v2[2] + k2[2] * impulse;
  ho1[0] = o1[0] + J1[0] * impulse;
  ho1[1] = o1[1] + J1[1] * impulse;
  ho1[2] = o1[2] + J1[2] * impulse;
  ho2[0] = o2[0] + J2[0] * impulse;
  ho2[1] = o2[1] + J2[1] * impulse;
  ho2[2] = o2[2] + J2[2] * impulse;
}

} // namespace

struct pressureCorrectionKernel {
  pressureCorrectionKernel(const SimulationData &s) : sim(s) {}
  const SimulationData &sim;
  const cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0}};
  const std::vector<cubism::BlockInfo> &tmpVInfo = sim.tmpV->getBlocksInfo();

  void operator()(ScalarLab &P, const cubism::BlockInfo &info) const {
    const Real h = info.h, pFac = -0.5 * sim.dt * h;
    VectorBlock &__restrict__ tmpV =
        *(VectorBlock *)tmpVInfo[info.blockID].ptrBlock;
    for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        tmpV(ix, iy).u[0] = pFac * (P(ix + 1, iy).s - P(ix - 1, iy).s);
        tmpV(ix, iy).u[1] = pFac * (P(ix, iy + 1).s - P(ix, iy - 1).s);
      }
    cubism::BlockCase<VectorBlock> *tempCase =
        (cubism::BlockCase<VectorBlock> *)(tmpVInfo[info.blockID].auxiliary);
    VectorBlock::ElementType *faceXm = nullptr;
    VectorBlock::ElementType *faceXp = nullptr;
    VectorBlock::ElementType *faceYm = nullptr;
    VectorBlock::ElementType *faceYp = nullptr;
    if (tempCase != nullptr) {
      faceXm = tempCase->storedFace[0] ? &tempCase->m_pData[0][0] : nullptr;
      faceXp = tempCase->storedFace[1] ? &tempCase->m_pData[1][0] : nullptr;
      faceYm = tempCase->storedFace[2] ? &tempCase->m_pData[2][0] : nullptr;
      faceYp = tempCase->storedFace[3] ? &tempCase->m_pData[3][0] : nullptr;
    }
    if (faceXm != nullptr) {
      int ix = 0;
      for (int iy = 0; iy < VectorBlock::sizeY; ++iy) {
        faceXm[iy].clear();
        faceXm[iy].u[0] = pFac * (P(ix - 1, iy).s + P(ix, iy).s);
      }
    }
    if (faceXp != nullptr) {
      int ix = VectorBlock::sizeX - 1;
      for (int iy = 0; iy < VectorBlock::sizeY; ++iy) {
        faceXp[iy].clear();
        faceXp[iy].u[0] = -pFac * (P(ix + 1, iy).s + P(ix, iy).s);
      }
    }
    if (faceYm != nullptr) {
      int iy = 0;
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        faceYm[ix].clear();
        faceYm[ix].u[1] = pFac * (P(ix, iy - 1).s + P(ix, iy).s);
      }
    }
    if (faceYp != nullptr) {
      int iy = VectorBlock::sizeY - 1;
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        faceYp[ix].clear();
        faceYp[ix].u[1] = -pFac * (P(ix, iy + 1).s + P(ix, iy).s);
      }
    }
  }
};

void PressureSingle::pressureCorrection(const Real dt) {
  const pressureCorrectionKernel K(sim);
  cubism::compute<ScalarLab>(K, sim.pres, sim.tmpV);

  std::vector<cubism::BlockInfo> &tmpVInfo = sim.tmpV->getBlocksInfo();
#pragma omp parallel for
  for (size_t i = 0; i < velInfo.size(); i++) {
    const Real ih2 = 1.0 / velInfo[i].h / velInfo[i].h;
    VectorBlock &__restrict__ V = *(VectorBlock *)velInfo[i].ptrBlock;
    VectorBlock &__restrict__ tmpV = *(VectorBlock *)tmpVInfo[i].ptrBlock;
    for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        V(ix, iy).u[0] += tmpV(ix, iy).u[0] * ih2;
        V(ix, iy).u[1] += tmpV(ix, iy).u[1] * ih2;
      }
  }
}

void PressureSingle::integrateMomenta(Shape *const shape) const {
  const size_t Nblocks = velInfo.size();

  const std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
  const Real Cx = shape->centerOfMass[0];
  const Real Cy = shape->centerOfMass[1];
  Real PM = 0, PJ = 0, PX = 0, PY = 0, UM = 0, VM = 0, AM = 0; // linear momenta

#pragma omp parallel for reduction(+ : PM, PJ, PX, PY, UM, VM, AM)
  for (size_t i = 0; i < Nblocks; i++) {
    const VectorBlock &__restrict__ VEL = *(VectorBlock *)velInfo[i].ptrBlock;
    const Real hsq = velInfo[i].h * velInfo[i].h;

    if (OBLOCK[velInfo[i].blockID] == nullptr)
      continue;
    const CHI_MAT &__restrict__ chi = OBLOCK[velInfo[i].blockID]->chi;
    const UDEFMAT &__restrict__ udef = OBLOCK[velInfo[i].blockID]->udef;
#ifndef EXPL_INTEGRATE_MOM
    const Real lambdt = sim.lambda * sim.dt;
#endif

    for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        if (chi[iy][ix] <= 0)
          continue;
        const Real udiff[2] = {VEL(ix, iy).u[0] - udef[iy][ix][0],
                               VEL(ix, iy).u[1] - udef[iy][ix][1]};
#ifdef EXPL_INTEGRATE_MOM
        const Real F = hsq * chi[iy][ix];
#else
        // const Real Xlamdt = chi[iy][ix] * lambdt;
        // need to use unmollified version when H(x) appears in fractions
        const Real Xlamdt = chi[iy][ix] >= 0.5 ? lambdt : 0.0;
        const Real F = hsq * Xlamdt / (1 + Xlamdt);
#endif
        Real p[2];
        velInfo[i].pos(p, ix, iy);
        p[0] -= Cx;
        p[1] -= Cy;
        PM += F;
        PJ += F * (p[0] * p[0] + p[1] * p[1]);
        PX += F * p[0];
        PY += F * p[1];
        UM += F * udiff[0];
        VM += F * udiff[1];
        AM += F * (p[0] * udiff[1] - p[1] * udiff[0]);
      }
  }
  Real quantities[7] = {PM, PJ, PX, PY, UM, VM, AM};
  MPI_Allreduce(MPI_IN_PLACE, quantities, 7, MPI_Real, MPI_SUM,
                sim.chi->getWorldComm());
  PM = quantities[0];
  PJ = quantities[1];
  PX = quantities[2];
  PY = quantities[3];
  UM = quantities[4];
  VM = quantities[5];
  AM = quantities[6];

  shape->fluidAngMom = AM;
  shape->fluidMomX = UM;
  shape->fluidMomY = VM;
  shape->penalDX = PX;
  shape->penalDY = PY;
  shape->penalM = PM;
  shape->penalJ = PJ;
}

void PressureSingle::penalize(const Real dt) const {
  std::vector<cubism::BlockInfo> &chiInfo = sim.chi->getBlocksInfo();

  const size_t Nblocks = velInfo.size();

#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++)
    for (const auto &shape : sim.shapes) {
      const std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
      const ObstacleBlock *const o = OBLOCK[velInfo[i].blockID];
      if (o == nullptr)
        continue;

      const Real u_s = shape->u;
      const Real v_s = shape->v;
      const Real omega_s = shape->omega;
      const Real Cx = shape->centerOfMass[0];
      const Real Cy = shape->centerOfMass[1];

      const CHI_MAT &__restrict__ X = o->chi;
      const UDEFMAT &__restrict__ UDEF = o->udef;
      const ScalarBlock &__restrict__ CHI = *(ScalarBlock *)chiInfo[i].ptrBlock;
      VectorBlock &__restrict__ V = *(VectorBlock *)velInfo[i].ptrBlock;

      for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
        for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
          // What if multiple obstacles share a block? Do not write udef onto
          // grid if CHI stored on the grid is greater than obst's CHI.
          if (CHI(ix, iy).s > X[iy][ix])
            continue;
          if (X[iy][ix] <= 0)
            continue; // no need to do anything

          Real p[2];
          velInfo[i].pos(p, ix, iy);
          p[0] -= Cx;
          p[1] -= Cy;
#ifndef EXPL_INTEGRATE_MOM
          // const Real alpha = 1/(1 + sim.lambda * dt * X[iy][ix]);
          // need to use unmollified version when H(x) appears in fractions
          const Real alpha = X[iy][ix] > 0.5 ? 1 / (1 + sim.lambda * dt) : 1;
#else
          const Real alpha = 1 - X[iy][ix];
#endif

          const Real US = u_s - omega_s * p[1] + UDEF[iy][ix][0];
          const Real VS = v_s + omega_s * p[0] + UDEF[iy][ix][1];
          V(ix, iy).u[0] = alpha * V(ix, iy).u[0] + (1 - alpha) * US;
          V(ix, iy).u[1] = alpha * V(ix, iy).u[1] + (1 - alpha) * VS;
        }
    }
}

struct updatePressureRHS {
  // RHS of Poisson equation is div(u) - chi * div(u_def)
  // It is computed here and stored in TMP

  updatePressureRHS(const SimulationData &s) : sim(s) {}
  const SimulationData &sim;
  cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0, 1}};
  cubism::StencilInfo stencil2{-1, -1, 0, 2, 2, 1, false, {0, 1}};
  const std::vector<cubism::BlockInfo> &tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &chiInfo = sim.chi->getBlocksInfo();

  void operator()(VectorLab &velLab, VectorLab &uDefLab,
                  const cubism::BlockInfo &info,
                  const cubism::BlockInfo &info2) const {
    const Real h = info.h;
    const Real facDiv = 0.5 * h / sim.dt;
    ScalarBlock &__restrict__ TMP =
        *(ScalarBlock *)tmpInfo[info.blockID].ptrBlock;
    ScalarBlock &__restrict__ CHI =
        *(ScalarBlock *)chiInfo[info.blockID].ptrBlock;
    for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        TMP(ix, iy).s =
            facDiv * ((velLab(ix + 1, iy).u[0] - velLab(ix - 1, iy).u[0]) +
                      (velLab(ix, iy + 1).u[1] - velLab(ix, iy - 1).u[1]));
        TMP(ix, iy).s +=
            -facDiv * CHI(ix, iy).s *
            ((uDefLab(ix + 1, iy).u[0] - uDefLab(ix - 1, iy).u[0]) +
             (uDefLab(ix, iy + 1).u[1] - uDefLab(ix, iy - 1).u[1]));
      }
    cubism::BlockCase<ScalarBlock> *tempCase =
        (cubism::BlockCase<ScalarBlock> *)(tmpInfo[info.blockID].auxiliary);
    ScalarBlock::ElementType *faceXm = nullptr;
    ScalarBlock::ElementType *faceXp = nullptr;
    ScalarBlock::ElementType *faceYm = nullptr;
    ScalarBlock::ElementType *faceYp = nullptr;
    if (tempCase != nullptr) {
      faceXm = tempCase->storedFace[0] ? &tempCase->m_pData[0][0] : nullptr;
      faceXp = tempCase->storedFace[1] ? &tempCase->m_pData[1][0] : nullptr;
      faceYm = tempCase->storedFace[2] ? &tempCase->m_pData[2][0] : nullptr;
      faceYp = tempCase->storedFace[3] ? &tempCase->m_pData[3][0] : nullptr;
    }
    if (faceXm != nullptr) {
      int ix = 0;
      for (int iy = 0; iy < VectorBlock::sizeY; ++iy) {
        faceXm[iy].s = facDiv * (velLab(ix - 1, iy).u[0] + velLab(ix, iy).u[0]);
        faceXm[iy].s += -(facDiv * CHI(ix, iy).s) *
                        (uDefLab(ix - 1, iy).u[0] + uDefLab(ix, iy).u[0]);
      }
    }
    if (faceXp != nullptr) {
      int ix = VectorBlock::sizeX - 1;
      for (int iy = 0; iy < VectorBlock::sizeY; ++iy) {
        faceXp[iy].s =
            -facDiv * (velLab(ix + 1, iy).u[0] + velLab(ix, iy).u[0]);
        faceXp[iy].s -= -(facDiv * CHI(ix, iy).s) *
                        (uDefLab(ix + 1, iy).u[0] + uDefLab(ix, iy).u[0]);
      }
    }
    if (faceYm != nullptr) {
      int iy = 0;
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        faceYm[ix].s = facDiv * (velLab(ix, iy - 1).u[1] + velLab(ix, iy).u[1]);
        faceYm[ix].s += -(facDiv * CHI(ix, iy).s) *
                        (uDefLab(ix, iy - 1).u[1] + uDefLab(ix, iy).u[1]);
      }
    }
    if (faceYp != nullptr) {
      int iy = VectorBlock::sizeY - 1;
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        faceYp[ix].s =
            -facDiv * (velLab(ix, iy + 1).u[1] + velLab(ix, iy).u[1]);
        faceYp[ix].s -= -(facDiv * CHI(ix, iy).s) *
                        (uDefLab(ix, iy + 1).u[1] + uDefLab(ix, iy).u[1]);
      }
    }
  }
};

struct updatePressureRHS1 {
  // RHS of Poisson equation is div(u) - chi * div(u_def)
  // It is computed here and stored in TMP

  updatePressureRHS1(const SimulationData &s) : sim(s) {}
  const SimulationData &sim;
  cubism::StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0}};
  const std::vector<cubism::BlockInfo> &tmpInfo = sim.tmp->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &poldInfo = sim.pold->getBlocksInfo();

  void operator()(ScalarLab &lab, const cubism::BlockInfo &info) const {
    ScalarBlock &__restrict__ TMP =
        *(ScalarBlock *)tmpInfo[info.blockID].ptrBlock;
    for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix)
        TMP(ix, iy).s -= (((lab(ix - 1, iy).s + lab(ix + 1, iy).s) +
                           (lab(ix, iy - 1).s + lab(ix, iy + 1).s)) -
                          4.0 * lab(ix, iy).s);

    cubism::BlockCase<ScalarBlock> *tempCase =
        (cubism::BlockCase<ScalarBlock> *)(tmpInfo[info.blockID].auxiliary);
    ScalarBlock::ElementType *faceXm = nullptr;
    ScalarBlock::ElementType *faceXp = nullptr;
    ScalarBlock::ElementType *faceYm = nullptr;
    ScalarBlock::ElementType *faceYp = nullptr;
    if (tempCase != nullptr) {
      faceXm = tempCase->storedFace[0] ? &tempCase->m_pData[0][0] : nullptr;
      faceXp = tempCase->storedFace[1] ? &tempCase->m_pData[1][0] : nullptr;
      faceYm = tempCase->storedFace[2] ? &tempCase->m_pData[2][0] : nullptr;
      faceYp = tempCase->storedFace[3] ? &tempCase->m_pData[3][0] : nullptr;
    }
    if (faceXm != nullptr) {
      int ix = 0;
      for (int iy = 0; iy < ScalarBlock::sizeY; ++iy)
        faceXm[iy] = lab(ix - 1, iy) - lab(ix, iy);
    }
    if (faceXp != nullptr) {
      int ix = ScalarBlock::sizeX - 1;
      for (int iy = 0; iy < ScalarBlock::sizeY; ++iy)
        faceXp[iy] = lab(ix + 1, iy) - lab(ix, iy);
    }
    if (faceYm != nullptr) {
      int iy = 0;
      for (int ix = 0; ix < ScalarBlock::sizeX; ++ix)
        faceYm[ix] = lab(ix, iy - 1) - lab(ix, iy);
    }
    if (faceYp != nullptr) {
      int iy = ScalarBlock::sizeY - 1;
      for (int ix = 0; ix < ScalarBlock::sizeX; ++ix)
        faceYp[ix] = lab(ix, iy + 1) - lab(ix, iy);
    }
  }
};

void PressureSingle::preventCollidingObstacles() const {
  const auto &shapes = sim.shapes;
  const auto &infos = sim.chi->getBlocksInfo();
  const size_t N = shapes.size();
  sim.bCollisionID.clear();

  struct CollisionInfo // hitter and hittee, symmetry but we do things twice
  {
    Real iM = 0;
    Real iPosX = 0;
    Real iPosY = 0;
    Real iPosZ = 0;
    Real iMomX = 0;
    Real iMomY = 0;
    Real iMomZ = 0;
    Real ivecX = 0;
    Real ivecY = 0;
    Real ivecZ = 0;
    Real jM = 0;
    Real jPosX = 0;
    Real jPosY = 0;
    Real jPosZ = 0;
    Real jMomX = 0;
    Real jMomY = 0;
    Real jMomZ = 0;
    Real jvecX = 0;
    Real jvecY = 0;
    Real jvecZ = 0;
  };
  std::vector<CollisionInfo> collisions(N);

  std::vector<Real> n_vec(3 * N, 0.0);

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j) {
      if (i == j)
        continue;
      auto &coll = collisions[i];

      const auto &iBlocks = shapes[i]->obstacleBlocks;
      const Real iU0 = shapes[i]->u;
      const Real iU1 = shapes[i]->v;
      // const Real iU2      = 0; //set to 0 for 2D
      // const Real iomega0  = 0; //set to 0 for 2D
      // const Real iomega1  = 0; //set to 0 for 2D
      const Real iomega2 = shapes[i]->omega;
      const Real iCx = shapes[i]->centerOfMass[0];
      const Real iCy = shapes[i]->centerOfMass[1];
      // const Real iCz      = 0; //set to 0 for 2D

      const auto &jBlocks = shapes[j]->obstacleBlocks;
      const Real jU0 = shapes[j]->u;
      const Real jU1 = shapes[j]->v;
      // const Real jU2      = 0; //set to 0 for 2D
      // const Real jomega0  = 0; //set to 0 for 2D
      // const Real jomega1  = 0; //set to 0 for 2D
      const Real jomega2 = shapes[j]->omega;
      const Real jCx = shapes[j]->centerOfMass[0];
      const Real jCy = shapes[j]->centerOfMass[1];
      // const Real jCz      = 0; //set to 0 for 2D

      assert(iBlocks.size() == jBlocks.size());

      const size_t nBlocks = iBlocks.size();
      for (size_t k = 0; k < nBlocks; ++k) {
        if (iBlocks[k] == nullptr || jBlocks[k] == nullptr)
          continue;

        const auto &iSDF = iBlocks[k]->dist;
        const auto &jSDF = jBlocks[k]->dist;

        const CHI_MAT &iChi = iBlocks[k]->chi;
        const CHI_MAT &jChi = jBlocks[k]->chi;

        const UDEFMAT &iUDEF = iBlocks[k]->udef;
        const UDEFMAT &jUDEF = jBlocks[k]->udef;

        for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
          for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
            if (iChi[iy][ix] <= 0.0 || jChi[iy][ix] <= 0.0)
              continue;

            const auto pos = infos[k].pos<Real>(ix, iy);

            const Real iUr0 = -iomega2 * (pos[1] - iCy);
            const Real iUr1 = iomega2 * (pos[0] - iCx);
            coll.iM += iChi[iy][ix];
            coll.iPosX += iChi[iy][ix] * pos[0];
            coll.iPosY += iChi[iy][ix] * pos[1];
            coll.iMomX += iChi[iy][ix] * (iU0 + iUr0 + iUDEF[iy][ix][0]);
            coll.iMomY += iChi[iy][ix] * (iU1 + iUr1 + iUDEF[iy][ix][1]);

            const Real jUr0 = -jomega2 * (pos[1] - jCy);
            const Real jUr1 = jomega2 * (pos[0] - jCx);
            coll.jM += jChi[iy][ix];
            coll.jPosX += jChi[iy][ix] * pos[0];
            coll.jPosY += jChi[iy][ix] * pos[1];
            coll.jMomX += jChi[iy][ix] * (jU0 + jUr0 + jUDEF[iy][ix][0]);
            coll.jMomY += jChi[iy][ix] * (jU1 + jUr1 + jUDEF[iy][ix][1]);

            Real dSDFdx_i;
            Real dSDFdx_j;
            if (ix == 0) {
              dSDFdx_i = iSDF[iy][ix + 1] - iSDF[iy][ix];
              dSDFdx_j = jSDF[iy][ix + 1] - jSDF[iy][ix];
            } else if (ix == VectorBlock::sizeX - 1) {
              dSDFdx_i = iSDF[iy][ix] - iSDF[iy][ix - 1];
              dSDFdx_j = jSDF[iy][ix] - jSDF[iy][ix - 1];
            } else {
              dSDFdx_i = 0.5 * (iSDF[iy][ix + 1] - iSDF[iy][ix - 1]);
              dSDFdx_j = 0.5 * (jSDF[iy][ix + 1] - jSDF[iy][ix - 1]);
            }

            Real dSDFdy_i;
            Real dSDFdy_j;
            if (iy == 0) {
              dSDFdy_i = iSDF[iy + 1][ix] - iSDF[iy][ix];
              dSDFdy_j = jSDF[iy + 1][ix] - jSDF[iy][ix];
            } else if (iy == VectorBlock::sizeY - 1) {
              dSDFdy_i = iSDF[iy][ix] - iSDF[iy - 1][ix];
              dSDFdy_j = jSDF[iy][ix] - jSDF[iy - 1][ix];
            } else {
              dSDFdy_i = 0.5 * (iSDF[iy + 1][ix] - iSDF[iy - 1][ix]);
              dSDFdy_j = 0.5 * (jSDF[iy + 1][ix] - jSDF[iy - 1][ix]);
            }

            coll.ivecX += iChi[iy][ix] * dSDFdx_i;
            coll.ivecY += iChi[iy][ix] * dSDFdy_i;

            coll.jvecX += jChi[iy][ix] * dSDFdx_j;
            coll.jvecY += jChi[iy][ix] * dSDFdy_j;
          }
      }
    }

  std::vector<Real> buffer(20 * N); // CollisionInfo holds 20 Reals
  for (size_t i = 0; i < N; i++) {
    auto &coll = collisions[i];
    buffer[20 * i] = coll.iM;
    buffer[20 * i + 1] = coll.iPosX;
    buffer[20 * i + 2] = coll.iPosY;
    buffer[20 * i + 3] = coll.iPosZ;
    buffer[20 * i + 4] = coll.iMomX;
    buffer[20 * i + 5] = coll.iMomY;
    buffer[20 * i + 6] = coll.iMomZ;
    buffer[20 * i + 7] = coll.ivecX;
    buffer[20 * i + 8] = coll.ivecY;
    buffer[20 * i + 9] = coll.ivecZ;
    buffer[20 * i + 10] = coll.jM;
    buffer[20 * i + 11] = coll.jPosX;
    buffer[20 * i + 12] = coll.jPosY;
    buffer[20 * i + 13] = coll.jPosZ;
    buffer[20 * i + 14] = coll.jMomX;
    buffer[20 * i + 15] = coll.jMomY;
    buffer[20 * i + 16] = coll.jMomZ;
    buffer[20 * i + 17] = coll.jvecX;
    buffer[20 * i + 18] = coll.jvecY;
    buffer[20 * i + 19] = coll.jvecZ;
  }
  MPI_Allreduce(MPI_IN_PLACE, buffer.data(), buffer.size(), MPI_Real, MPI_SUM,
                sim.chi->getWorldComm());
  for (size_t i = 0; i < N; i++) {
    auto &coll = collisions[i];
    coll.iM = buffer[20 * i];
    coll.iPosX = buffer[20 * i + 1];
    coll.iPosY = buffer[20 * i + 2];
    coll.iPosZ = buffer[20 * i + 3];
    coll.iMomX = buffer[20 * i + 4];
    coll.iMomY = buffer[20 * i + 5];
    coll.iMomZ = buffer[20 * i + 6];
    coll.ivecX = buffer[20 * i + 7];
    coll.ivecY = buffer[20 * i + 8];
    coll.ivecZ = buffer[20 * i + 9];
    coll.jM = buffer[20 * i + 10];
    coll.jPosX = buffer[20 * i + 11];
    coll.jPosY = buffer[20 * i + 12];
    coll.jPosZ = buffer[20 * i + 13];
    coll.jMomX = buffer[20 * i + 14];
    coll.jMomY = buffer[20 * i + 15];
    coll.jMomZ = buffer[20 * i + 16];
    coll.jvecX = buffer[20 * i + 17];
    coll.jvecY = buffer[20 * i + 18];
    coll.jvecZ = buffer[20 * i + 19];
  }

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i + 1; j < N; ++j) {
      if (i == j)
        continue;
      const Real m1 = shapes[i]->M;
      const Real m2 = shapes[j]->M;
      const Real v1[3] = {shapes[i]->u, shapes[i]->v, 0.0};
      const Real v2[3] = {shapes[j]->u, shapes[j]->v, 0.0};
      const Real o1[3] = {0, 0, shapes[i]->omega};
      const Real o2[3] = {0, 0, shapes[j]->omega};
      const Real C1[3] = {shapes[i]->centerOfMass[0],
                          shapes[i]->centerOfMass[1], 0};
      const Real C2[3] = {shapes[j]->centerOfMass[0],
                          shapes[j]->centerOfMass[1], 0};
      const Real I1[6] = {1.0, 0, 0, 0, 0, shapes[i]->J};
      const Real I2[6] = {1.0, 0, 0, 0, 0, shapes[j]->J};

      auto &coll = collisions[i];
      auto &coll_other = collisions[j];
      // less than one fluid element of overlap: wait to get closer. no hit
      if (coll.iM < 2.0 || coll.jM < 2.0)
        continue; // object i did not collide
      if (coll_other.iM < 2.0 || coll_other.jM < 2.0)
        continue; // object j did not collide

      if (std::fabs(coll.iPosX / coll.iM - coll_other.iPosX / coll_other.iM) >
              shapes[i]->getCharLength() ||
          std::fabs(coll.iPosY / coll.iM - coll_other.iPosY / coll_other.iM) >
              shapes[i]->getCharLength()) {
        continue; // then both objects i and j collided, but not with each
                  // other!
      }

      // A collision happened!
      sim.bCollision = true;
#pragma omp critical
      {
        sim.bCollisionID.push_back(i);
        sim.bCollisionID.push_back(j);
      }

      const bool iForced = shapes[i]->bForced;
      const bool jForced = shapes[j]->bForced;
      if (iForced || jForced) {
        std::cout
            << "[CUP2D] WARNING: Forced objects not supported for collision."
            << std::endl;
        // MPI_Abort(sim.chi->getWorldComm(),1);
      }

      Real ho1[3];
      Real ho2[3];
      Real hv1[3];
      Real hv2[3];

      // 1. Compute collision normal vector (NX,NY,NZ)
      const Real norm_i =
          std::sqrt(coll.ivecX * coll.ivecX + coll.ivecY * coll.ivecY +
                    coll.ivecZ * coll.ivecZ);
      const Real norm_j =
          std::sqrt(coll.jvecX * coll.jvecX + coll.jvecY * coll.jvecY +
                    coll.jvecZ * coll.jvecZ);
      const Real mX = coll.ivecX / norm_i - coll.jvecX / norm_j;
      const Real mY = coll.ivecY / norm_i - coll.jvecY / norm_j;
      const Real mZ = coll.ivecZ / norm_i - coll.jvecZ / norm_j;
      const Real inorm = 1.0 / std::sqrt(mX * mX + mY * mY + mZ * mZ);
      const Real NX = mX * inorm;
      const Real NY = mY * inorm;
      const Real NZ = mZ * inorm;

      // If objects are already moving away from each other, don't do anything
      // if( (v2[0]-v1[0])*NX + (v2[1]-v1[1])*NY + (v2[2]-v1[2])*NZ <= 0 )
      // continue;
      const Real hitVelX = coll.jMomX / coll.jM - coll.iMomX / coll.iM;
      const Real hitVelY = coll.jMomY / coll.jM - coll.iMomY / coll.iM;
      const Real hitVelZ = coll.jMomZ / coll.jM - coll.iMomZ / coll.iM;
      const Real projVel = hitVelX * NX + hitVelY * NY + hitVelZ * NZ;

      /*const*/ Real vc1[3] = {coll.iMomX / coll.iM, coll.iMomY / coll.iM,
                               coll.iMomZ / coll.iM};
      /*const*/ Real vc2[3] = {coll.jMomX / coll.jM, coll.jMomY / coll.jM,
                               coll.jMomZ / coll.jM};

      if (projVel <= 0)
        continue; // vel goes away from collision: no need to bounce

      // 2. Compute collision location
      const Real inv_iM = 1.0 / coll.iM;
      const Real inv_jM = 1.0 / coll.jM;
      const Real iPX = coll.iPosX * inv_iM; // object i collision location
      const Real iPY = coll.iPosY * inv_iM;
      const Real iPZ = coll.iPosZ * inv_iM;
      const Real jPX = coll.jPosX * inv_jM; // object j collision location
      const Real jPY = coll.jPosY * inv_jM;
      const Real jPZ = coll.jPosZ * inv_jM;
      const Real CX = 0.5 * (iPX + jPX);
      const Real CY = 0.5 * (iPY + jPY);
      const Real CZ = 0.5 * (iPZ + jPZ);

      // 3. Take care of the collision. Assume elastic collision (kinetic energy
      // is conserved)
      ElasticCollision(m1, m2, I1, I2, v1, v2, o1, o2, hv1, hv2, ho1, ho2, C1,
                       C2, NX, NY, NZ, CX, CY, CZ, vc1, vc2);
      shapes[i]->u = hv1[0];
      shapes[i]->v = hv1[1];
      // shapes[i]->transVel[2] = hv1[2];
      shapes[j]->u = hv2[0];
      shapes[j]->v = hv2[1];
      // shapes[j]->transVel[2] = hv2[2];
      // shapes[i]->angVel[0] = ho1[0];
      // shapes[i]->angVel[1] = ho1[1];
      shapes[i]->omega = ho1[2];
      // shapes[j]->angVel[0] = ho2[0];
      // shapes[j]->angVel[1] = ho2[1];
      shapes[j]->omega = ho2[2];

      if (sim.rank == 0) {
#pragma omp critical
        {
          std::cout << "Collision between objects " << i << " and " << j
                    << std::endl;
          std::cout << " iM   (0) = " << collisions[i].iM
                    << " jM   (1) = " << collisions[j].jM << std::endl;
          std::cout << " jM   (0) = " << collisions[i].jM
                    << " jM   (1) = " << collisions[j].iM << std::endl;
          std::cout << " Normal vector = (" << NX << "," << NY << "," << NZ
                    << std::endl;
          std::cout << " Location      = (" << CX << "," << CY << "," << CZ
                    << std::endl;
          std::cout << " Shape " << i << " before collision u    =(" << v1[0]
                    << "," << v1[1] << "," << v1[2] << ")" << std::endl;
          std::cout << " Shape " << i << " after  collision u    =(" << hv1[0]
                    << "," << hv1[1] << "," << hv1[2] << ")" << std::endl;
          std::cout << " Shape " << j << " before collision u    =(" << v2[0]
                    << "," << v2[1] << "," << v2[2] << ")" << std::endl;
          std::cout << " Shape " << j << " after  collision u    =(" << hv2[0]
                    << "," << hv2[1] << "," << hv2[2] << ")" << std::endl;
          std::cout << " Shape " << i << " before collision omega=(" << o1[0]
                    << "," << o1[1] << "," << o1[2] << ")" << std::endl;
          std::cout << " Shape " << i << " after  collision omega=(" << ho1[0]
                    << "," << ho1[1] << "," << ho1[2] << ")" << std::endl;
          std::cout << " Shape " << j << " before collision omega=(" << o2[0]
                    << "," << o2[1] << "," << o2[2] << ")" << std::endl;
          std::cout << " Shape " << j << " after  collision omega=(" << ho2[0]
                    << "," << ho2[1] << "," << ho2[2] << ")" << std::endl;
        }
      }
    }
}

void PressureSingle::operator()(const Real dt) {
  sim.startProfiler("Pressure");
  const size_t Nblocks = velInfo.size();

  // update velocity of obstacle
  for (const auto &shape : sim.shapes) {
    integrateMomenta(shape.get());
    shape->updateVelocity(dt);
  }
  // take care if two obstacles collide
  preventCollidingObstacles();

  // apply penalization force
  penalize(dt);

  // compute pressure RHS
  // first we put uDef to tmpV so that we can create a VectorLab to compute
  // div(uDef)
  const std::vector<cubism::BlockInfo> &tmpVInfo = sim.tmpV->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &chiInfo = sim.chi->getBlocksInfo();
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    ((VectorBlock *)tmpVInfo[i].ptrBlock)->clear();
  }
  for (const auto &shape : sim.shapes) {
    const std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
#pragma omp parallel for
    for (size_t i = 0; i < Nblocks; i++) {
      if (OBLOCK[tmpVInfo[i].blockID] == nullptr)
        continue; // obst not in block
      const UDEFMAT &__restrict__ udef = OBLOCK[tmpVInfo[i].blockID]->udef;
      const CHI_MAT &__restrict__ chi = OBLOCK[tmpVInfo[i].blockID]->chi;
      auto &__restrict__ UDEF = *(VectorBlock *)tmpVInfo[i].ptrBlock; // dest
      const ScalarBlock &__restrict__ CHI = *(ScalarBlock *)chiInfo[i].ptrBlock;
      for (int iy = 0; iy < VectorBlock::sizeY; iy++)
        for (int ix = 0; ix < VectorBlock::sizeX; ix++) {
          if (chi[iy][ix] < CHI(ix, iy).s)
            continue;
          Real p[2];
          tmpVInfo[i].pos(p, ix, iy);
          UDEF(ix, iy).u[0] += udef[iy][ix][0];
          UDEF(ix, iy).u[1] += udef[iy][ix][1];
        }
    }
  }
  updatePressureRHS K(sim);
  cubism::compute<updatePressureRHS, VectorGrid, VectorLab, VectorGrid,
                  VectorLab, ScalarGrid>(K, *sim.vel, *sim.tmpV, true, sim.tmp);

  // Add p_old (+dp/dt) to RHS
  const std::vector<cubism::BlockInfo> &presInfo = sim.pres->getBlocksInfo();
  const std::vector<cubism::BlockInfo> &poldInfo = sim.pold->getBlocksInfo();

// initial guess etc.
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    ScalarBlock &__restrict__ PRES = *(ScalarBlock *)presInfo[i].ptrBlock;
    ScalarBlock &__restrict__ POLD = *(ScalarBlock *)poldInfo[i].ptrBlock;
    for (int iy = 0; iy < VectorBlock::sizeY; ++iy)
      for (int ix = 0; ix < VectorBlock::sizeX; ++ix) {
        POLD(ix, iy).s = PRES(ix, iy).s;
        PRES(ix, iy).s = 0;
      }
  }
  updatePressureRHS1 K1(sim);
  cubism::compute<ScalarLab>(K1, sim.pold, sim.tmp);

  pressureSolver->solve(sim.tmp, sim.pres);

  Real avg = 0;
  Real avg1 = 0;
#pragma omp parallel for reduction(+ : avg, avg1)
  for (size_t i = 0; i < Nblocks; i++) {
    ScalarBlock &P = *(ScalarBlock *)presInfo[i].ptrBlock;
    const Real vv = presInfo[i].h * presInfo[i].h;
    for (int iy = 0; iy < VectorBlock::sizeY; iy++)
      for (int ix = 0; ix < VectorBlock::sizeX; ix++) {
        avg += P(ix, iy).s * vv;
        avg1 += vv;
      }
  }
  Real quantities[2] = {avg, avg1};
  MPI_Allreduce(MPI_IN_PLACE, &quantities, 2, MPI_Real, MPI_SUM, sim.comm);
  avg = quantities[0];
  avg1 = quantities[1];
  avg = avg / avg1;
#pragma omp parallel for
  for (size_t i = 0; i < Nblocks; i++) {
    ScalarBlock &P = *(ScalarBlock *)presInfo[i].ptrBlock;
    const ScalarBlock &__restrict__ POLD = *(ScalarBlock *)poldInfo[i].ptrBlock;
    for (int iy = 0; iy < VectorBlock::sizeY; iy++)
      for (int ix = 0; ix < VectorBlock::sizeX; ix++)
        P(ix, iy).s += POLD(ix, iy).s - avg;
  }

  // apply pressure correction
  pressureCorrection(dt);

  sim.stopProfiler();
}

PressureSingle::PressureSingle(SimulationData &s)
    : Operator{s}, pressureSolver{makePoissonSolver(s)} {}

PressureSingle::~PressureSingle() = default;
struct SimulationData;
struct FishSkin {
  const size_t Npoints;
  Real *const xSurf;
  Real *const ySurf;
  Real *const normXSurf;
  Real *const normYSurf;
  Real *const midX;
  Real *const midY;
  FishSkin(const FishSkin &c)
      : Npoints(c.Npoints), xSurf(new Real[Npoints]), ySurf(new Real[Npoints]),
        normXSurf(new Real[Npoints - 1]), normYSurf(new Real[Npoints - 1]),
        midX(new Real[Npoints - 1]), midY(new Real[Npoints - 1]) {}

  FishSkin(const size_t N)
      : Npoints(N), xSurf(new Real[Npoints]), ySurf(new Real[Npoints]),
        normXSurf(new Real[Npoints - 1]), normYSurf(new Real[Npoints - 1]),
        midX(new Real[Npoints - 1]), midY(new Real[Npoints - 1]) {}

  ~FishSkin() {
    delete[] xSurf;
    delete[] ySurf;
    delete[] normXSurf;
    delete[] normYSurf;
    delete[] midX;
    delete[] midY;
  }
};

struct FishData {
public:
  // Length and minimal gridspacing
  const Real length, h;

  // Midline is discretized by more points in first fraction and last fraction:
  const Real fracRefined = 0.1, fracMid = 1 - 2 * fracRefined;
  const Real dSmid_tgt = h / std::sqrt(2);
  const Real dSrefine_tgt = 0.125 * h;

  //// Nm should be divisible by 8, see Fish.cpp - 3)
  // thus Nmid enforced to be divisible by 8
  const int Nmid = (int)std::ceil(length * fracMid / dSmid_tgt / 8) * 8;
  const Real dSmid = length * fracMid / Nmid;

  // thus Nend enforced to be divisible by 4
  const int Nend =
      (int)std::ceil(fracRefined * length * 2 / (dSmid + dSrefine_tgt) / 4) * 4;
  const Real dSref = fracRefined * length * 2 / Nend - dSmid;

  const int Nm = Nmid + 2 * Nend + 1; // plus 1 because we contain 0 and L

  Real *const rS; // arclength discretization points
  Real *const rX; // coordinates of midline discretization points
  Real *const rY;
  Real *const vX; // midline discretization velocities
  Real *const vY;
  Real *const norX; // normal vector to the midline discretization points
  Real *const norY;
  Real *const vNorX;
  Real *const vNorY;
  Real *const width;

  Real linMom[2], area, J, angMom; // for diagnostics
  // start and end indices in the arrays where the fish starts and ends (to
  // ignore the extensions when interpolating the shapes)
  FishSkin upperSkin = FishSkin(Nm);
  FishSkin lowerSkin = FishSkin(Nm);
  virtual void resetAll();

protected:
  template <typename T>
  inline void _rotate2D(const Real Rmatrix2D[2][2], T &x, T &y) const {
    const T p[2] = {x, y};
    x = Rmatrix2D[0][0] * p[0] + Rmatrix2D[0][1] * p[1];
    y = Rmatrix2D[1][0] * p[0] + Rmatrix2D[1][1] * p[1];
  }
  template <typename T>
  inline void _translateAndRotate2D(const T pos[2], const Real Rmatrix2D[2][2],
                                    Real &x, Real &y) const {
    const Real p[2] = {x - pos[0], y - pos[1]};
    x = Rmatrix2D[0][0] * p[0] + Rmatrix2D[0][1] * p[1];
    y = Rmatrix2D[1][0] * p[0] + Rmatrix2D[1][1] * p[1];
  }

  static Real *_alloc(const int N) { return new Real[N]; }
  template <typename T> static void _dealloc(T *ptr) {
    if (ptr not_eq nullptr) {
      delete[] ptr;
      ptr = nullptr;
    }
  }

  inline Real _d_ds(const int idx, const Real *const vals,
                    const int maxidx) const {
    if (idx == 0)
      return (vals[idx + 1] - vals[idx]) / (rS[idx + 1] - rS[idx]);
    else if (idx == maxidx - 1)
      return (vals[idx] - vals[idx - 1]) / (rS[idx] - rS[idx - 1]);
    else
      return ((vals[idx + 1] - vals[idx]) / (rS[idx + 1] - rS[idx]) +
              (vals[idx] - vals[idx - 1]) / (rS[idx] - rS[idx - 1])) /
             2;
  }
  inline Real _integrationFac1(const int idx) const { return 2 * width[idx]; }
  inline Real _integrationFac2(const int idx) const {
    const Real dnorXi = _d_ds(idx, norX, Nm);
    const Real dnorYi = _d_ds(idx, norY, Nm);
    return 2 * std::pow(width[idx], 3) *
           (dnorXi * norY[idx] - dnorYi * norX[idx]) / 3;
  }
  inline Real _integrationFac3(const int idx) const {
    return 2 * std::pow(width[idx], 3) / 3;
  }

  virtual void _computeMidlineNormals() const;

  virtual Real _width(const Real s, const Real L) = 0;

  void _computeWidth() {
    for (int i = 0; i < Nm; ++i)
      width[i] = _width(rS[i], length);
  }

public:
  FishData(Real L, Real _h);
  virtual ~FishData();

  Real integrateLinearMomentum(Real CoM[2], Real vCoM[2]);
  Real integrateAngularMomentum(Real &angVel);

  void changeToCoMFrameLinear(const Real CoM_internal[2],
                              const Real vCoM_internal[2]) const;
  void changeToCoMFrameAngular(const Real theta_internal,
                               const Real angvel_internal) const;

  void computeSurface() const;
  void surfaceToCOMFrame(const Real theta_internal,
                         const Real CoM_internal[2]) const;
  void surfaceToComputationalFrame(const Real theta_comp,
                                   const Real CoM_interpolated[2]) const;
  void computeSkinNormals(const Real theta_comp, const Real CoM_comp[3]) const;
  void writeMidline2File(const int step_id, std::string filename);

  virtual void computeMidline(const Real time, const Real dt) = 0;
};

struct AreaSegment {
  const Real safe_distance;
  const std::pair<int, int> s_range;
  Real w[2], c[2];
  // should be normalized and >=0:
  Real normalI[2] = {(Real)1, (Real)0};
  Real normalJ[2] = {(Real)0, (Real)1};
  Real objBoxLabFr[2][2] = {{0, 0}, {0, 0}};
  Real objBoxObjFr[2][2] = {{0, 0}, {0, 0}};

  AreaSegment(std::pair<int, int> sr, const Real bb[2][2], const Real safe)
      : safe_distance(safe), s_range(sr),
        w{(bb[0][1] - bb[0][0]) / 2 + safe, (bb[1][1] - bb[1][0]) / 2 + safe},
        c{(bb[0][1] + bb[0][0]) / 2, (bb[1][1] + bb[1][0]) / 2} {
    assert(w[0] > 0);
    assert(w[1] > 0);
  }

  void changeToComputationalFrame(const Real position[2], const Real angle);
  bool isIntersectingWithAABB(const Real start[2], const Real end[2]) const;
};

struct PutFishOnBlocks {
  const FishData &cfish;
  const Real position[2];
  const Real angle;
  const Real Rmatrix2D[2][2] = {{std::cos(angle), -std::sin(angle)},
                                {std::sin(angle), std::cos(angle)}};
  static inline Real eulerDistSq2D(const Real a[2], const Real b[2]) {
    return std::pow(a[0] - b[0], 2) + std::pow(a[1] - b[1], 2);
  }
  void changeVelocityToComputationalFrame(Real x[2]) const {
    const Real p[2] = {x[0], x[1]};
    x[0] =
        Rmatrix2D[0][0] * p[0] + Rmatrix2D[0][1] * p[1]; // rotate (around CoM)
    x[1] = Rmatrix2D[1][0] * p[0] + Rmatrix2D[1][1] * p[1];
  }
  template <typename T> void changeToComputationalFrame(T x[2]) const {
    const T p[2] = {x[0], x[1]};
    x[0] = Rmatrix2D[0][0] * p[0] + Rmatrix2D[0][1] * p[1];
    x[1] = Rmatrix2D[1][0] * p[0] + Rmatrix2D[1][1] * p[1];
    x[0] += position[0]; // translate
    x[1] += position[1];
  }
  template <typename T> void changeFromComputationalFrame(T x[2]) const {
    const T p[2] = {x[0] - (T)position[0], x[1] - (T)position[1]};
    // rotate back around CoM
    x[0] = Rmatrix2D[0][0] * p[0] + Rmatrix2D[1][0] * p[1];
    x[1] = Rmatrix2D[0][1] * p[0] + Rmatrix2D[1][1] * p[1];
  }

  PutFishOnBlocks(const FishData &cf, const Real pos[2], const Real ang)
      : cfish(cf), position{(Real)pos[0], (Real)pos[1]}, angle(ang) {}
  virtual ~PutFishOnBlocks() {}

  void operator()(const cubism::BlockInfo &i, ScalarBlock &b,
                  ObstacleBlock *const o,
                  const std::vector<AreaSegment *> &v) const;
  virtual void constructSurface(const cubism::BlockInfo &i, ScalarBlock &b,
                                ObstacleBlock *const o,
                                const std::vector<AreaSegment *> &v) const;
  virtual void constructInternl(const cubism::BlockInfo &i, ScalarBlock &b,
                                ObstacleBlock *const o,
                                const std::vector<AreaSegment *> &v) const;
  virtual void signedDistanceSqrt(const cubism::BlockInfo &i, ScalarBlock &b,
                                  ObstacleBlock *const o,
                                  const std::vector<AreaSegment *> &v) const;
};

FishData::FishData(Real L, Real _h)
    : length(L), h(_h), rS(_alloc(Nm)), rX(_alloc(Nm)), rY(_alloc(Nm)),
      vX(_alloc(Nm)), vY(_alloc(Nm)), norX(_alloc(Nm)), norY(_alloc(Nm)),
      vNorX(_alloc(Nm)), vNorY(_alloc(Nm)), width(_alloc(Nm)) {
  if (dSref <= 0) {
    std::cout << "[CUP2D] dSref <= 0. Aborting..." << std::endl;
    fflush(0);
    abort();
  }

  rS[0] = 0;
  int k = 0;
  // extension head
  for (int i = 0; i < Nend; ++i, k++)
    rS[k + 1] = rS[k] + dSref + (dSmid - dSref) * i / ((Real)Nend - 1.);
  // interior points
  for (int i = 0; i < Nmid; ++i, k++)
    rS[k + 1] = rS[k] + dSmid;
  // extension tail
  for (int i = 0; i < Nend; ++i, k++)
    rS[k + 1] =
        rS[k] + dSref + (dSmid - dSref) * (Nend - i - 1) / ((Real)Nend - 1.);
  assert(k + 1 == Nm);
  // cout << "Discrepancy of midline length: " << std::fabs(rS[k]-L) << endl;
  rS[k] = std::min(rS[k], (Real)L);
  std::fill(rX, rX + Nm, 0);
  std::fill(rY, rY + Nm, 0);
  std::fill(vX, vX + Nm, 0);
  std::fill(vY, vY + Nm, 0);
}

FishData::~FishData() {
  _dealloc(rS);
  _dealloc(rX);
  _dealloc(rY);
  _dealloc(vX);
  _dealloc(vY);
  _dealloc(norX);
  _dealloc(norY);
  _dealloc(vNorX);
  _dealloc(vNorY);
  _dealloc(width);
  // if(upperSkin not_eq nullptr) { delete upperSkin; upperSkin=nullptr; }
  // if(lowerSkin not_eq nullptr) { delete lowerSkin; lowerSkin=nullptr; }
}

void FishData::resetAll() {}

void FishData::writeMidline2File(const int step_id, std::string filename) {
  char buf[500];
  sprintf(buf, "%s_midline_%07d.txt", filename.c_str(), step_id);
  FILE *f = fopen(buf, "a");
  fprintf(f, "s x y vX vY\n");
  for (int i = 0; i < Nm; i++) {
    // dummy.changeToComputationalFrame(temp);
    // dummy.changeVelocityToComputationalFrame(udef);
    fprintf(f, "%g %g %g %g %g %g\n", (double)rS[i], (double)rX[i],
            (double)rY[i], (double)vX[i], (double)vY[i], (double)width[i]);
  }
  fflush(0);
}

void FishData::_computeMidlineNormals() const {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < Nm - 1; i++) {
    const auto ds = rS[i + 1] - rS[i];
    const auto tX = rX[i + 1] - rX[i];
    const auto tY = rY[i + 1] - rY[i];
    const auto tVX = vX[i + 1] - vX[i];
    const auto tVY = vY[i + 1] - vY[i];
    norX[i] = -tY / ds;
    norY[i] = tX / ds;
    vNorX[i] = -tVY / ds;
    vNorY[i] = tVX / ds;
  }
  norX[Nm - 1] = norX[Nm - 2];
  norY[Nm - 1] = norY[Nm - 2];
  vNorX[Nm - 1] = vNorX[Nm - 2];
  vNorY[Nm - 1] = vNorY[Nm - 2];
}

Real FishData::integrateLinearMomentum(Real CoM[2], Real vCoM[2]) {
  // already worked out the integrals for r, theta on paper
  // remaining integral done with composite trapezoidal rule
  // minimize rhs evaluations --> do first and last point separately
  Real _area = 0, _cmx = 0, _cmy = 0, _lmx = 0, _lmy = 0;
#pragma omp parallel for schedule(static)                                      \
    reduction(+ : _area, _cmx, _cmy, _lmx, _lmy)
  for (int i = 0; i < Nm; ++i) {
    const Real ds = (i == 0) ? rS[1] - rS[0]
                             : ((i == Nm - 1) ? rS[Nm - 1] - rS[Nm - 2]
                                              : rS[i + 1] - rS[i - 1]);
    const Real fac1 = _integrationFac1(i);
    const Real fac2 = _integrationFac2(i);
    _area += fac1 * ds / 2;
    _cmx += (rX[i] * fac1 + norX[i] * fac2) * ds / 2;
    _cmy += (rY[i] * fac1 + norY[i] * fac2) * ds / 2;
    _lmx += (vX[i] * fac1 + vNorX[i] * fac2) * ds / 2;
    _lmy += (vY[i] * fac1 + vNorY[i] * fac2) * ds / 2;
  }
  area = _area;
  CoM[0] = _cmx;
  CoM[1] = _cmy;
  linMom[0] = _lmx;
  linMom[1] = _lmy;
  assert(area > std::numeric_limits<Real>::epsilon());
  CoM[0] /= area;
  CoM[1] /= area;
  vCoM[0] = linMom[0] / area;
  vCoM[1] = linMom[1] / area;
  // printf("%f %f %f %f %f\n",CoM[0],CoM[1],vCoM[0],vCoM[1], vol);
  return area;
}

Real FishData::integrateAngularMomentum(Real &angVel) {
  // assume we have already translated CoM and vCoM to nullify linear momentum
  // already worked out the integrals for r, theta on paper
  // remaining integral done with composite trapezoidal rule
  // minimize rhs evaluations --> do first and last point separately
  Real _J = 0, _am = 0;
#pragma omp parallel for reduction(+ : _J, _am) schedule(static)
  for (int i = 0; i < Nm; ++i) {
    const Real ds = (i == 0) ? rS[1] - rS[0]
                             : ((i == Nm - 1) ? rS[Nm - 1] - rS[Nm - 2]
                                              : rS[i + 1] - rS[i - 1]);
    const Real fac1 = _integrationFac1(i);
    const Real fac2 = _integrationFac2(i);
    const Real fac3 = _integrationFac3(i);
    const Real tmp_M = (rX[i] * vY[i] - rY[i] * vX[i]) * fac1 +
                       (rX[i] * vNorY[i] - rY[i] * vNorX[i] + vY[i] * norX[i] -
                        vX[i] * norY[i]) *
                           fac2 +
                       (norX[i] * vNorY[i] - norY[i] * vNorX[i]) * fac3;

    const Real tmp_J = (rX[i] * rX[i] + rY[i] * rY[i]) * fac1 +
                       2 * (rX[i] * norX[i] + rY[i] * norY[i]) * fac2 + fac3;

    _am += tmp_M * ds / 2;
    _J += tmp_J * ds / 2;
  }
  J = _J;
  angMom = _am;
  assert(J > std::numeric_limits<Real>::epsilon());
  angVel = angMom / J;
  return J;
}

void FishData::changeToCoMFrameLinear(const Real Cin[2],
                                      const Real vCin[2]) const {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < Nm; ++i) { // subtract internal CM and vCM
    rX[i] -= Cin[0];
    rY[i] -= Cin[1];
    vX[i] -= vCin[0];
    vY[i] -= vCin[1];
  }
}

void FishData::changeToCoMFrameAngular(const Real Ain, const Real vAin) const {
  const Real Rmatrix2D[2][2] = {{std::cos(Ain), -std::sin(Ain)},
                                {std::sin(Ain), std::cos(Ain)}};

#pragma omp parallel for schedule(static)
  for (int i = 0; i < Nm;
       ++i) { // subtract internal angvel and rotate position by ang
    vX[i] += vAin * rY[i];
    vY[i] -= vAin * rX[i];
    _rotate2D(Rmatrix2D, rX[i], rY[i]);
    _rotate2D(Rmatrix2D, vX[i], vY[i]);
  }
  _computeMidlineNormals();
}

void FishData::computeSurface() const {
// Compute surface points by adding width to the midline points
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < lowerSkin.Npoints; ++i) {
    Real norm[2] = {norX[i], norY[i]};
    Real const norm_mod1 = std::sqrt(norm[0] * norm[0] + norm[1] * norm[1]);
    norm[0] /= norm_mod1;
    norm[1] /= norm_mod1;
    assert(width[i] >= 0);
    lowerSkin.xSurf[i] = rX[i] - width[i] * norm[0];
    lowerSkin.ySurf[i] = rY[i] - width[i] * norm[1];
    upperSkin.xSurf[i] = rX[i] + width[i] * norm[0];
    upperSkin.ySurf[i] = rY[i] + width[i] * norm[1];
  }
}

void FishData::computeSkinNormals(const Real theta_comp,
                                  const Real CoM_comp[3]) const {
  const Real Rmatrix2D[2][2] = {{std::cos(theta_comp), -std::sin(theta_comp)},
                                {std::sin(theta_comp), std::cos(theta_comp)}};

  for (int i = 0; i < Nm; ++i) {
    _rotate2D(Rmatrix2D, rX[i], rY[i]);
    _rotate2D(Rmatrix2D, norX[i], norY[i]);
    rX[i] += CoM_comp[0];
    rY[i] += CoM_comp[1];
  }

// Compute midpoints as they will be pressure targets
#pragma omp parallel for
  for (size_t i = 0; i < lowerSkin.Npoints - 1; ++i) {
    lowerSkin.midX[i] = (lowerSkin.xSurf[i] + lowerSkin.xSurf[i + 1]) / 2;
    upperSkin.midX[i] = (upperSkin.xSurf[i] + upperSkin.xSurf[i + 1]) / 2;
    lowerSkin.midY[i] = (lowerSkin.ySurf[i] + lowerSkin.ySurf[i + 1]) / 2;
    upperSkin.midY[i] = (upperSkin.ySurf[i] + upperSkin.ySurf[i + 1]) / 2;

    lowerSkin.normXSurf[i] = (lowerSkin.ySurf[i + 1] - lowerSkin.ySurf[i]);
    upperSkin.normXSurf[i] = (upperSkin.ySurf[i + 1] - upperSkin.ySurf[i]);
    lowerSkin.normYSurf[i] = -(lowerSkin.xSurf[i + 1] - lowerSkin.xSurf[i]);
    upperSkin.normYSurf[i] = -(upperSkin.xSurf[i + 1] - upperSkin.xSurf[i]);

    const Real normL = std::sqrt(std::pow(lowerSkin.normXSurf[i], 2) +
                                 std::pow(lowerSkin.normYSurf[i], 2));
    const Real normU = std::sqrt(std::pow(upperSkin.normXSurf[i], 2) +
                                 std::pow(upperSkin.normYSurf[i], 2));

    lowerSkin.normXSurf[i] /= normL;
    upperSkin.normXSurf[i] /= normU;
    lowerSkin.normYSurf[i] /= normL;
    upperSkin.normYSurf[i] /= normU;

    // if too close to the head or tail, consider a point further in, so that we
    // are pointing out for sure
    const int ii =
        (i < 8) ? 8 : ((i > lowerSkin.Npoints - 9) ? lowerSkin.Npoints - 9 : i);

    const Real dirL = lowerSkin.normXSurf[i] * (lowerSkin.midX[i] - rX[ii]) +
                      lowerSkin.normYSurf[i] * (lowerSkin.midY[i] - rY[ii]);
    const Real dirU = upperSkin.normXSurf[i] * (upperSkin.midX[i] - rX[ii]) +
                      upperSkin.normYSurf[i] * (upperSkin.midY[i] - rY[ii]);

    if (dirL < 0) {
      lowerSkin.normXSurf[i] *= -1.0;
      lowerSkin.normYSurf[i] *= -1.0;
    }
    if (dirU < 0) {
      upperSkin.normXSurf[i] *= -1.0;
      upperSkin.normYSurf[i] *= -1.0;
    }
  }
}

void FishData::surfaceToCOMFrame(const Real theta_internal,
                                 const Real CoM_internal[2]) const {
  const Real Rmatrix2D[2][2] = {
      {std::cos(theta_internal), -std::sin(theta_internal)},
      {std::sin(theta_internal), std::cos(theta_internal)}};
  // Surface points rotation and translation

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < upperSkin.Npoints; ++i)
  // for(int i=0; i<upperSkin.Npoints-1; ++i)
  {
    upperSkin.xSurf[i] -= CoM_internal[0];
    upperSkin.ySurf[i] -= CoM_internal[1];
    _rotate2D(Rmatrix2D, upperSkin.xSurf[i], upperSkin.ySurf[i]);
    lowerSkin.xSurf[i] -= CoM_internal[0];
    lowerSkin.ySurf[i] -= CoM_internal[1];
    _rotate2D(Rmatrix2D, lowerSkin.xSurf[i], lowerSkin.ySurf[i]);
  }
}

void FishData::surfaceToComputationalFrame(
    const Real theta_comp, const Real CoM_interpolated[2]) const {
  const Real Rmatrix2D[2][2] = {{std::cos(theta_comp), -std::sin(theta_comp)},
                                {std::sin(theta_comp), std::cos(theta_comp)}};

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < upperSkin.Npoints; ++i) {
    _rotate2D(Rmatrix2D, upperSkin.xSurf[i], upperSkin.ySurf[i]);
    upperSkin.xSurf[i] += CoM_interpolated[0];
    upperSkin.ySurf[i] += CoM_interpolated[1];
    _rotate2D(Rmatrix2D, lowerSkin.xSurf[i], lowerSkin.ySurf[i]);
    lowerSkin.xSurf[i] += CoM_interpolated[0];
    lowerSkin.ySurf[i] += CoM_interpolated[1];
  }
}

void AreaSegment::changeToComputationalFrame(const Real pos[2],
                                             const Real angle) {
  // we are in CoM frame and change to comp frame --> first rotate around CoM
  // (which is at (0,0) in CoM frame), then update center
  const Real Rmatrix2D[2][2] = {{std::cos(angle), -std::sin(angle)},
                                {std::sin(angle), std::cos(angle)}};
  const Real p[2] = {c[0], c[1]};

  const Real nx[2] = {normalI[0], normalI[1]};
  const Real ny[2] = {normalJ[0], normalJ[1]};

  for (int i = 0; i < 2; ++i) {
    c[i] = Rmatrix2D[i][0] * p[0] + Rmatrix2D[i][1] * p[1];

    normalI[i] = Rmatrix2D[i][0] * nx[0] + Rmatrix2D[i][1] * nx[1];
    normalJ[i] = Rmatrix2D[i][0] * ny[0] + Rmatrix2D[i][1] * ny[1];
  }

  c[0] += pos[0];
  c[1] += pos[1];

  const Real magI =
      std::sqrt(normalI[0] * normalI[0] + normalI[1] * normalI[1]);
  const Real magJ =
      std::sqrt(normalJ[0] * normalJ[0] + normalJ[1] * normalJ[1]);
  assert(magI > std::numeric_limits<Real>::epsilon());
  assert(magJ > std::numeric_limits<Real>::epsilon());
  const Real invMagI = 1 / magI, invMagJ = 1 / magJ;

  for (int i = 0; i < 2; ++i) {
    // also take absolute value since thats what we need when doing intersection
    // checks later
    normalI[i] = std::fabs(normalI[i]) * invMagI;
    normalJ[i] = std::fabs(normalJ[i]) * invMagJ;
  }

  assert(normalI[0] >= 0 && normalI[1] >= 0);
  assert(normalJ[0] >= 0 && normalJ[1] >= 0);

  // Find the x,y,z max extents in lab frame ( exploit normal(I,J,K)[:] >=0 )
  const Real widthXvec[] = {w[0] * normalI[0], w[0] * normalI[1]};
  const Real widthYvec[] = {w[1] * normalJ[0], w[1] * normalJ[1]};

  for (int i = 0; i < 2; ++i) {
    objBoxLabFr[i][0] = c[i] - widthXvec[i] - widthYvec[i];
    objBoxLabFr[i][1] = c[i] + widthXvec[i] + widthYvec[i];
    objBoxObjFr[i][0] = c[i] - w[i];
    objBoxObjFr[i][1] = c[i] + w[i];
  }
}

bool AreaSegment::isIntersectingWithAABB(const Real start[2],
                                         const Real end[2]) const {
  // Remember: Incoming coordinates are cell centers, not cell faces
  // start and end are two diagonally opposed corners of grid block
  // GN halved the safety here but added it back to w[] in prepare
  const Real AABB_w[2] = {// half block width + safe distance
                          (end[0] - start[0]) / 2 + safe_distance,
                          (end[1] - start[1]) / 2 + safe_distance};

  const Real AABB_c[2] = {// block center
                          (end[0] + start[0]) / 2, (end[1] + start[1]) / 2};

  const Real AABB_box[2][2] = {{AABB_c[0] - AABB_w[0], AABB_c[0] + AABB_w[0]},
                               {AABB_c[1] - AABB_w[1], AABB_c[1] + AABB_w[1]}};

  assert(AABB_w[0] > 0 && AABB_w[1] > 0);

  // Now Identify the ones that do not intersect
  Real intersectionLabFrame[2][2] = {
      {std::max(objBoxLabFr[0][0], AABB_box[0][0]),
       std::min(objBoxLabFr[0][1], AABB_box[0][1])},
      {std::max(objBoxLabFr[1][0], AABB_box[1][0]),
       std::min(objBoxLabFr[1][1], AABB_box[1][1])}};

  if (intersectionLabFrame[0][1] - intersectionLabFrame[0][0] < 0 ||
      intersectionLabFrame[1][1] - intersectionLabFrame[1][0] < 0)
    return false;

  // This is x-width of box, expressed in fish frame
  const Real widthXbox[2] = {AABB_w[0] * normalI[0], AABB_w[0] * normalJ[0]};
  // This is y-width of box, expressed in fish frame
  const Real widthYbox[2] = {AABB_w[1] * normalI[1], AABB_w[1] * normalJ[1]};

  const Real boxBox[2][2] = {{AABB_c[0] - widthXbox[0] - widthYbox[0],
                              AABB_c[0] + widthXbox[0] + widthYbox[0]},
                             {AABB_c[1] - widthXbox[1] - widthYbox[1],
                              AABB_c[1] + widthXbox[1] + widthYbox[1]}};

  Real intersectionFishFrame[2][2] = {
      {std::max(boxBox[0][0], objBoxObjFr[0][0]),
       std::min(boxBox[0][1], objBoxObjFr[0][1])},
      {std::max(boxBox[1][0], objBoxObjFr[1][0]),
       std::min(boxBox[1][1], objBoxObjFr[1][1])}};

  if (intersectionFishFrame[0][1] - intersectionFishFrame[0][0] < 0 ||
      intersectionFishFrame[1][1] - intersectionFishFrame[1][0] < 0)
    return false;

  return true;
}

void PutFishOnBlocks::operator()(const cubism::BlockInfo &i, ScalarBlock &b,
                                 ObstacleBlock *const o,
                                 const std::vector<AreaSegment *> &v) const {
  // std::chrono::time_point<std::chrono::high_resolution_clock> t0, t1, t2, t3;
  // t0 = std::chrono::high_resolution_clock::now();
  constructSurface(i, b, o, v);
  // t1 = std::chrono::high_resolution_clock::now();
  constructInternl(i, b, o, v);
  // t2 = std::chrono::high_resolution_clock::now();
  signedDistanceSqrt(i, b, o, v);
  // t3 = std::chrono::high_resolution_clock::now();
  // printf("%g %g %g\n",std::chrono::duration<Real>(t1-t0).count(),
  //                     std::chrono::duration<Real>(t2-t1).count(),
  //                     std::chrono::duration<Real>(t3-t2).count());
}

void PutFishOnBlocks::signedDistanceSqrt(
    const cubism::BlockInfo &info, ScalarBlock &b, ObstacleBlock *const o,
    const std::vector<AreaSegment *> &vSegments) const {
  // finalize signed distance function in tmpU
  static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
  for (int iy = 0; iy < ScalarBlock::sizeY; iy++)
    for (int ix = 0; ix < ScalarBlock::sizeX; ix++) {
      const Real normfac = o->chi[iy][ix] > EPS ? o->chi[iy][ix] : 1;
      o->udef[iy][ix][0] /= normfac;
      o->udef[iy][ix][1] /= normfac;
      // change from signed squared distance function to normal sdf
      o->dist[iy][ix] = o->dist[iy][ix] >= 0 ? std::sqrt(o->dist[iy][ix])
                                             : -std::sqrt(-o->dist[iy][ix]);
      b(ix, iy).s = std::max(b(ix, iy).s, o->dist[iy][ix]);
      ;
    }
  static constexpr int BS[2] = {ScalarBlock::sizeX, ScalarBlock::sizeY};
  std::fill(o->chi[0], o->chi[0] + BS[1] * BS[0], 0);
}

void PutFishOnBlocks::constructSurface(
    const cubism::BlockInfo &info, ScalarBlock &b, ObstacleBlock *const o,
    const std::vector<AreaSegment *> &vSegments) const {
  Real org[2];
  info.pos(org, 0, 0);
#ifndef NDEBUG
  static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
#endif
  const Real h = info.h, invh = 1.0 / info.h;
  const Real *const rX = cfish.rX, *const norX = cfish.norX;
  const Real *const rY = cfish.rY, *const norY = cfish.norY;
  const Real *const vX = cfish.vX, *const vNorX = cfish.vNorX;
  const Real *const vY = cfish.vY, *const vNorY = cfish.vNorY;
  const Real *const width = cfish.width;
  static constexpr int BS[2] = {ScalarBlock::sizeX, ScalarBlock::sizeY};
  std::fill(o->dist[0], o->dist[0] + BS[1] * BS[0], -1);
  std::fill(o->chi[0], o->chi[0] + BS[1] * BS[0], 0);

  // construct the shape (P2M with min(distance) as kernel) onto ObstacleBlock
  for (int i = 0; i < (int)vSegments.size(); ++i) {
    // iterate over segments contained in the vSegm intersecting this block:
    const int firstSegm = std::max(vSegments[i]->s_range.first, 1);
    const int lastSegm = std::min(vSegments[i]->s_range.second, cfish.Nm - 2);
    for (int ss = firstSegm; ss <= lastSegm; ++ss) {
      assert(width[ss] > 0);
      // for each segment, we have one point to left and right of midl
      for (int signp = -1; signp <= 1; signp += 2) {
        // create a surface point
        // special treatment of tail (width = 0 --> no ellipse, just line)
        Real myP[2] = {rX[ss + 0] + width[ss + 0] * signp * norX[ss + 0],
                       rY[ss + 0] + width[ss + 0] * signp * norY[ss + 0]};
        changeToComputationalFrame(myP);
        const int iap[2] = {(int)std::floor((myP[0] - org[0]) * invh),
                            (int)std::floor((myP[1] - org[1]) * invh)};
        if (iap[0] + 3 <= 0 || iap[0] - 1 >= BS[0])
          continue; // NearNeigh loop
        if (iap[1] + 3 <= 0 || iap[1] - 1 >= BS[1])
          continue; // does not intersect

        Real pP[2] = {rX[ss + 1] + width[ss + 1] * signp * norX[ss + 1],
                      rY[ss + 1] + width[ss + 1] * signp * norY[ss + 1]};
        changeToComputationalFrame(pP);
        Real pM[2] = {rX[ss - 1] + width[ss - 1] * signp * norX[ss - 1],
                      rY[ss - 1] + width[ss - 1] * signp * norY[ss - 1]};
        changeToComputationalFrame(pM);
        Real udef[2] = {vX[ss + 0] + width[ss + 0] * signp * vNorX[ss + 0],
                        vY[ss + 0] + width[ss + 0] * signp * vNorY[ss + 0]};
        changeVelocityToComputationalFrame(udef);
        // support is two points left, two points right --> Towers Chi will
        // be one point left, one point right, but needs SDF wider
        for (int sy = std::max(0, iap[1] - 2); sy < std::min(iap[1] + 4, BS[1]);
             ++sy)
          for (int sx = std::max(0, iap[0] - 2);
               sx < std::min(iap[0] + 4, BS[0]); ++sx) {
            Real p[2];
            info.pos(p, sx, sy);
            const Real dist0 = eulerDistSq2D(p, myP);
            const Real distP = eulerDistSq2D(p, pP);
            const Real distM = eulerDistSq2D(p, pM);

            if (std::fabs(o->dist[sy][sx]) < std::min({dist0, distP, distM}))
              continue;

            changeFromComputationalFrame(p);
#ifndef NDEBUG // check that change of ref frame does not affect dist
            const Real p0[2] = {rX[ss] + width[ss] * signp * norX[ss],
                                rY[ss] + width[ss] * signp * norY[ss]};
            const Real distC = eulerDistSq2D(p, p0);
            assert(std::fabs(distC - dist0) < EPS);
#endif

            int close_s = ss, secnd_s = ss + (distP < distM ? 1 : -1);
            Real dist1 = dist0, dist2 = distP < distM ? distP : distM;
            if (distP < dist0 || distM < dist0) { // switch nearest surf point
              dist1 = dist2;
              dist2 = dist0;
              close_s = secnd_s;
              secnd_s = ss;
            }

            const Real dSsq = std::pow(rX[close_s] - rX[secnd_s], 2) +
                              std::pow(rY[close_s] - rY[secnd_s], 2);
            assert(dSsq > 2.2e-16);
            const Real cnt2ML = std::pow(width[close_s], 2);
            const Real nxt2ML = std::pow(width[secnd_s], 2);
            const Real safeW = std::max(width[close_s], width[secnd_s]) + 2 * h;
            const Real xMidl[2] = {rX[close_s], rY[close_s]};
            const Real grd2ML = eulerDistSq2D(p, xMidl);
            const Real diffH = std::fabs(width[close_s] - width[secnd_s]);
            Real sign2d = 0;
            // If width changes slowly or if point is very far away, this is
            // safer:
            if (dSsq > diffH * diffH ||
                grd2ML > safeW * safeW) { // if no abrupt changes in width we
                                          // use nearest neighbour
              sign2d = grd2ML > cnt2ML ? -1 : 1;
            } else {
              // else we model the span between ellipses as a spherical segment
              // http://mathworld.wolfram.com/SphericalSegment.html
              const Real corr = 2 * std::sqrt(cnt2ML * nxt2ML);
              const Real Rsq =
                  (cnt2ML + nxt2ML - corr + dSsq) // radius of the sphere
                  * (cnt2ML + nxt2ML + corr + dSsq) / 4 / dSsq;
              const Real maxAx = std::max(cnt2ML, nxt2ML);
              const int idAx1 = cnt2ML > nxt2ML ? close_s : secnd_s;
              const int idAx2 = idAx1 == close_s ? secnd_s : close_s;
              // 'submerged' fraction of radius:
              const Real d = std::sqrt((Rsq - maxAx) / dSsq); // (divided by ds)
              // position of the centre of the sphere:
              const Real xCentr[2] = {rX[idAx1] + (rX[idAx1] - rX[idAx2]) * d,
                                      rY[idAx1] + (rY[idAx1] - rY[idAx2]) * d};
              const Real grd2Core = eulerDistSq2D(p, xCentr);
              sign2d = grd2Core > Rsq ? -1 : 1; // as always, neg outside
            }

            if (std::fabs(o->dist[sy][sx]) > dist1) {
              const Real W =
                  1 - std::min((Real)1, std::sqrt(dist1) * (invh / 3));
              // W behaves like hat interpolation kernel that is used for
              // internal fish points. Introducing W (used to be W=1) smoothens
              // transition from surface to internal points. In fact, later we
              // plus equal udef*hat of internal points. If hat>0, point should
              // behave like internal point, meaning that fish-section udef
              // rotation should multiply distance from midline instead of
              // entire half-width. Remember that uder will become udef / chi,
              // so W simplifies out.
              assert(W >= 0);
              o->udef[sy][sx][0] = W * udef[0];
              o->udef[sy][sx][1] = W * udef[1];
              o->dist[sy][sx] = sign2d * dist1;
              o->chi[sy][sx] = W;
            }
            // Not chi yet, I stored squared distance from analytical boundary
            // distSq is updated only if curr value is smaller than the old one
          }
      }
    }
  }
}

void PutFishOnBlocks::constructInternl(
    const cubism::BlockInfo &info, ScalarBlock &b, ObstacleBlock *const o,
    const std::vector<AreaSegment *> &vSegments) const {
  Real org[2];
  info.pos(org, 0, 0);
  const Real h = info.h, invh = 1.0 / info.h;
  static constexpr int BS[2] = {ScalarBlock::sizeX, ScalarBlock::sizeY};
  // construct the deformation velocities (P2M with hat function as kernel)
  for (int i = 0; i < (int)vSegments.size(); ++i) {
    const int firstSegm = std::max(vSegments[i]->s_range.first, 1);
    const int lastSegm = std::min(vSegments[i]->s_range.second, cfish.Nm - 2);
    for (int ss = firstSegm; ss <= lastSegm; ++ss) {
      // P2M udef of a slice at this s
      const Real myWidth = cfish.width[ss];
      assert(myWidth > 0);
      // here we process also all inner points. Nw to the left and right of midl
      //  add xtension here to make sure we have it in each direction:
      const int Nw =
          std::floor(myWidth / h); // floor bcz we already did interior
      for (int iw = -Nw + 1; iw < Nw; ++iw) {
        const Real offsetW = iw * h;
        Real xp[2] = {cfish.rX[ss] + offsetW * cfish.norX[ss],
                      cfish.rY[ss] + offsetW * cfish.norY[ss]};
        changeToComputationalFrame(xp);
        xp[0] = (xp[0] - org[0]) * invh; // how many grid points from this block
        xp[1] = (xp[1] - org[1]) * invh; // origin is this fishpoint located at?
        const Real ap[2] = {std::floor(xp[0]), std::floor(xp[1])};
        const int iap[2] = {(int)ap[0], (int)ap[1]};
        if (iap[0] + 2 <= 0 || iap[0] >= BS[0])
          continue; // hatP2M loop
        if (iap[1] + 2 <= 0 || iap[1] >= BS[1])
          continue; // does not intersect

        Real udef[2] = {cfish.vX[ss] + offsetW * cfish.vNorX[ss],
                        cfish.vY[ss] + offsetW * cfish.vNorY[ss]};
        changeVelocityToComputationalFrame(udef);
        Real wghts[2][2]; // P2M weights
        for (int c = 0; c < 2; ++c) {
          const Real t[2] = {// we floored, hat between xp and grid point +-1
                             std::fabs(xp[c] - ap[c]),
                             std::fabs(xp[c] - (ap[c] + 1))};
          wghts[c][0] = 1 - t[0];
          wghts[c][1] = 1 - t[1];
        }

        for (int idy = std::max(0, iap[1]); idy < std::min(iap[1] + 2, BS[1]);
             ++idy)
          for (int idx = std::max(0, iap[0]); idx < std::min(iap[0] + 2, BS[0]);
               ++idx) {
            const int sx = idx - iap[0], sy = idy - iap[1];
            const Real wxwy = wghts[1][sy] * wghts[0][sx];
            assert(idx >= 0 && idx < ScalarBlock::sizeX && wxwy >= 0);
            assert(idy >= 0 && idy < ScalarBlock::sizeY && wxwy <= 1);
            o->udef[idy][idx][0] += wxwy * udef[0];
            o->udef[idy][idx][1] += wxwy * udef[1];
            o->chi[idy][idx] += wxwy;
            // set sign for all interior points
            static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
            if (std::fabs(o->dist[idy][idx] + 1) < EPS)
              o->dist[idy][idx] = 1;
          }
      }
    }
  }
}

class FactoryFileLineParser : public cubism::ArgumentParser {
protected:
  // from stackoverflow

  // trim from start
  inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(),
            std::find_if(s.begin(), s.end(),
                         std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
  }

  // trim from end
  inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
                         std::not1(std::ptr_fun<int, int>(std::isspace)))
                .base(),
            s.end());
    return s;
  }

  // trim from both ends
  inline std::string &trim(std::string &s) { return ltrim(rtrim(s)); }

public:
  FactoryFileLineParser(std::istringstream &is_line)
      : cubism::ArgumentParser(0, NULL, '#') // last char is comment leader
  {
    std::string key, value;
    while (std::getline(is_line, key, '=')) {
      if (std::getline(is_line, value, ' ')) {
        // add "-" because then we can use the same code for parsing factory as
        // command lines
        // mapArguments["-"+trim(key)] = Value(trim(value));
        mapArguments[trim(key)] = cubism::Value(trim(value));
      }
    }

    mute();
  }
};

struct FishData;
class Fish : public Shape {
public:
  const Real length, Tperiod, phaseShift;
  FishData *myFish = nullptr;

protected:
  Real area_internal = 0, J_internal = 0;
  Real CoM_internal[2] = {0, 0}, vCoM_internal[2] = {0, 0};
  Real theta_internal = 0, angvel_internal = 0, angvel_internal_prev = 0;

  Fish(SimulationData &s, cubism::ArgumentParser &p, Real C[2])
      : Shape(s, p, C), length(p("-L").asDouble(0.1)),
        Tperiod(p("-T").asDouble(1)), phaseShift(p("-phi").asDouble(0)) {}
  virtual ~Fish() override;

public:
  Real getCharLength() const override { return length; }
  void removeMoments(const std::vector<cubism::BlockInfo> &vInfo) override;
  virtual void resetAll() override;
  virtual void updatePosition(Real dt) override;
  virtual void create(const std::vector<cubism::BlockInfo> &vInfo) override;
  virtual void saveRestart(FILE *f) override;
  virtual void loadRestart(FILE *f) override;
};

void Fish::create(const std::vector<cubism::BlockInfo> &vInfo) {
  //// 0) clear obstacle blocks
  for (auto &entry : obstacleBlocks)
    delete entry;
  obstacleBlocks.clear();

  //// 1) Update Midline and compute surface
  assert(myFish != nullptr);
  profile(push_start("midline"));
  myFish->computeMidline(sim.time, sim.dt);
  myFish->computeSurface();
  profile(pop_stop());

  if (sim.rank == 0 && sim.bDump())
    myFish->writeMidline2File(0, "appending");

  //// 2) Integrate Linear and Angular Momentum and shift Fish accordingly
  profile(push_start("2dmoments"));
  // returns area, CoM_internal, vCoM_internal:
  area_internal = myFish->integrateLinearMomentum(CoM_internal, vCoM_internal);
  // takes CoM_internal, vCoM_internal, puts CoM in and nullifies  lin mom:
  myFish->changeToCoMFrameLinear(CoM_internal, vCoM_internal);
  angvel_internal_prev = angvel_internal;
  // returns mom of intertia and angvel:
  J_internal = myFish->integrateAngularMomentum(angvel_internal);
  // rotates fish midline to current angle and removes angular moment:
  myFish->changeToCoMFrameAngular(theta_internal, angvel_internal);
#if 0 // ndef NDEBUG
  {
    Real dummy_CoM_internal[2], dummy_vCoM_internal[2], dummy_angvel_internal;
    // check that things are zero
    const Real area_internal_check =
    myFish->integrateLinearMomentum(dummy_CoM_internal, dummy_vCoM_internal);
    myFish->integrateAngularMomentum(dummy_angvel_internal);
    const Real EPS = 10*std::numeric_limits<Real>::epsilon();
    assert(std::fabs(dummy_CoM_internal[0])<EPS);
    assert(std::fabs(dummy_CoM_internal[1])<EPS);
    assert(std::fabs(myFish->linMom[0])<EPS);
    assert(std::fabs(myFish->linMom[1])<EPS);
    assert(std::fabs(myFish->angMom)<EPS);
    assert(std::fabs(area_internal - area_internal_check) < EPS);
  }
#endif
  profile(pop_stop());
  myFish->surfaceToCOMFrame(theta_internal, CoM_internal);

  //// 3) Create Bounding Boxes around Fish
  //- performance of create seems to decrease if VolumeSegment_OBB are bigger
  //- this code groups segments together and finds a bounding box (maximal
  //  x and y coords) to then be able to check intersection with cartesian grid
  const int Nsegments = (myFish->Nm - 1) / 8, Nm = myFish->Nm;
  assert((Nm - 1) % Nsegments == 0);
  profile(push_start("boxes"));

  std::vector<AreaSegment *> vSegments(Nsegments, nullptr);
  const Real h = sim.getH();
#pragma omp parallel for schedule(static)
  for (int i = 0; i < Nsegments; ++i) {
    const int next_idx = (i + 1) * (Nm - 1) / Nsegments,
              idx = i * (Nm - 1) / Nsegments;
    // find bounding box based on this
    Real bbox[2][2] = {{1e9, -1e9}, {1e9, -1e9}};
    for (int ss = idx; ss <= next_idx; ++ss) {
      const Real xBnd[2] = {
          myFish->rX[ss] - myFish->norX[ss] * myFish->width[ss],
          myFish->rX[ss] + myFish->norX[ss] * myFish->width[ss]};
      const Real yBnd[2] = {
          myFish->rY[ss] - myFish->norY[ss] * myFish->width[ss],
          myFish->rY[ss] + myFish->norY[ss] * myFish->width[ss]};
      const Real maxX = std::max(xBnd[0], xBnd[1]),
                 minX = std::min(xBnd[0], xBnd[1]);
      const Real maxY = std::max(yBnd[0], yBnd[1]),
                 minY = std::min(yBnd[0], yBnd[1]);
      bbox[0][0] = std::min(bbox[0][0], minX);
      bbox[0][1] = std::max(bbox[0][1], maxX);
      bbox[1][0] = std::min(bbox[1][0], minY);
      bbox[1][1] = std::max(bbox[1][1], maxY);
    }
    const Real DD = 4 * h; // two points on each side
    // const Real safe_distance = info.h; // one point on each side
    AreaSegment *const tAS =
        new AreaSegment(std::make_pair(idx, next_idx), bbox, DD);
    tAS->changeToComputationalFrame(center, orientation);
    vSegments[i] = tAS;
  }
  profile(pop_stop());

  //// 4) Interpolate shape with computational grid
  profile(push_start("intersect"));
  const auto N = vInfo.size();
  std::vector<std::vector<AreaSegment *> *> segmentsPerBlock(N, nullptr);
  obstacleBlocks = std::vector<ObstacleBlock *>(N, nullptr);

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < vInfo.size(); ++i) {
    const cubism::BlockInfo &info = vInfo[i];
    Real pStart[2], pEnd[2];
    info.pos(pStart, 0, 0);
    info.pos(pEnd, ScalarBlock::sizeX - 1, ScalarBlock::sizeY - 1);

    for (size_t s = 0; s < vSegments.size(); ++s)
      if (vSegments[s]->isIntersectingWithAABB(pStart, pEnd)) {
        if (segmentsPerBlock[info.blockID] == nullptr)
          segmentsPerBlock[info.blockID] = new std::vector<AreaSegment *>(0);
        segmentsPerBlock[info.blockID]->push_back(vSegments[s]);
      }

    // allocate new blocks if necessary
    if (segmentsPerBlock[info.blockID] not_eq nullptr) {
      assert(obstacleBlocks[info.blockID] == nullptr);
      ObstacleBlock *const block = new ObstacleBlock();
      assert(block not_eq nullptr);
      obstacleBlocks[info.blockID] = block;
      block->clear();
    }
  }
  assert(not segmentsPerBlock.empty());
  assert(segmentsPerBlock.size() == obstacleBlocks.size());
  profile(pop_stop());

#pragma omp parallel
  {
    const PutFishOnBlocks putfish(*myFish, center, orientation);

#pragma omp for schedule(dynamic)
    for (size_t i = 0; i < vInfo.size(); i++) {
      const auto pos = segmentsPerBlock[vInfo[i].blockID];
      if (pos not_eq nullptr) {
        ObstacleBlock *const block = obstacleBlocks[vInfo[i].blockID];
        assert(block not_eq nullptr);
        putfish(vInfo[i], *(ScalarBlock *)vInfo[i].ptrBlock, block, *pos);
      }
    }
  }

  // clear vSegments
  for (auto &E : vSegments) {
    if (E not_eq nullptr)
      delete E;
  }
  for (auto &E : segmentsPerBlock) {
    if (E not_eq nullptr)
      delete E;
  }

  profile(pop_stop());
  if (sim.step % 100 == 0 && sim.verbose) {
    profile(printSummary());
    profile(reset());
  }
}

void Fish::updatePosition(Real dt) {
  // update position and angles
  Shape::updatePosition(dt);
  theta_internal -= dt * angvel_internal; // negative: we subtracted this angvel
}

void Fish::resetAll() {
  CoM_internal[0] = 0;
  CoM_internal[1] = 0;
  vCoM_internal[0] = 0;
  vCoM_internal[1] = 0;
  theta_internal = 0;
  angvel_internal = 0;
  angvel_internal_prev = 0;
  Shape::resetAll();
  myFish->resetAll();
}

Fish::~Fish() {
  if (myFish not_eq nullptr) {
    delete myFish;
    myFish = nullptr;
  }
}

void Fish::removeMoments(const std::vector<cubism::BlockInfo> &vInfo) {
  Shape::removeMoments(vInfo);
  myFish->surfaceToComputationalFrame(orientation, centerOfMass);
  myFish->computeSkinNormals(orientation, centerOfMass);
#if 0
  {
    std::stringstream ssF;
    ssF<<"skinPoints"<<std::setfill('0')<<std::setw(9)<<sim.step<<".dat";
    std::ofstream ofs (ssF.str().c_str(), std::ofstream::out);
    for(size_t i=0; i<myFish->upperSkin.Npoints; ++i)
      ofs<<myFish->upperSkin.xSurf[i]  <<" "<<myFish->upperSkin.ySurf[i]<<" " <<myFish->upperSkin.normXSurf[i]  <<" "<<myFish->upperSkin.normYSurf[i]  <<"\n";
    for(size_t i=myFish->lowerSkin.Npoints; i>0; --i)
      ofs<<myFish->lowerSkin.xSurf[i-1]<<" "<<myFish->lowerSkin.ySurf[i-1]<<" "<<myFish->lowerSkin.normXSurf[i-1]<<" "<<myFish->lowerSkin.normYSurf[i-1]<<"\n";
    ofs.flush();
    ofs.close();
  }
#endif
}

void Fish::saveRestart(FILE *f) {
  assert(f != NULL);
  Shape::saveRestart(f);
  fprintf(f, "theta_internal: %20.20e\n", (double)theta_internal);
  fprintf(f, "angvel_internal: %20.20e\n", (double)angvel_internal);
}

void Fish::loadRestart(FILE *f) {
  assert(f != NULL);
  Shape::loadRestart(f);
  bool ret = true;
  double in_theta_internal, in_angvel_internal;
  ret = ret && 1 == fscanf(f, "theta_internal: %le\n", &in_theta_internal);
  ret = ret && 1 == fscanf(f, "angvel_internal: %le\n", &in_angvel_internal);
  theta_internal = in_theta_internal;
  angvel_internal = in_angvel_internal;
  if ((not ret)) {
    printf("Error reading restart file. Aborting...\n");
    fflush(0);
    abort();
  }
}

class StefanFish : public Fish {
  const bool bCorrectTrajectory;
  const bool bCorrectPosition;

public:
  void act(const Real lTact, const std::vector<Real> &a) const;
  Real getLearnTPeriod() const;
  Real getPhase(const Real t) const;

  void resetAll() override;
  StefanFish(SimulationData &s, cubism::ArgumentParser &p, Real C[2]);
  void create(const std::vector<cubism::BlockInfo> &vInfo) override;

  // member functions for state in RL
  std::vector<Real> state(const std::vector<double> &origin) const;
  std::vector<Real> state3D() const;

  // Helpers for state function
  ssize_t holdingBlockID(const std::array<Real, 2> pos) const;
  std::array<Real, 2> getShear(const std::array<Real, 2> pSurf) const;

  // Old Helpers (here for backward compatibility)
  ssize_t holdingBlockID(const std::array<Real, 2> pos,
                         const std::vector<cubism::BlockInfo> &velInfo) const;
  std::array<int, 2> safeIdInBlock(const std::array<Real, 2> pos,
                                   const std::array<Real, 2> org,
                                   const Real invh) const;
  std::array<Real, 2>
  getShear(const std::array<Real, 2> pSurf, const std::array<Real, 2> normSurf,
           const std::vector<cubism::BlockInfo> &velInfo) const;

  // Helpers to restart simulation
  virtual void saveRestart(FILE *f) override;
  virtual void loadRestart(FILE *f) override;
};

class CurvatureFish : public FishData {
  const Real amplitudeFactor, phaseShift, Tperiod;

public:
  // PID controller of body curvature:
  Real curv_PID_fac = 0;
  Real curv_PID_dif = 0;
  // exponential averages:
  Real avgDeltaY = 0;
  Real avgDangle = 0;
  Real avgAngVel = 0;
  // stored past action for RL state:
  Real lastTact = 0;
  Real lastCurv = 0;
  Real oldrCurv = 0;
  // quantities needed to correctly control the speed of the midline maneuvers:
  Real periodPIDval = Tperiod;
  Real periodPIDdif = 0;
  bool TperiodPID = false;
  // quantities needed for rl:
  Real time0 = 0;
  Real timeshift = 0;
  // aux quantities for PID controllers:
  Real lastTime = 0;
  Real lastAvel = 0;

  // next scheduler is used to ramp-up the curvature from 0 during first period:
  Schedulers::ParameterSchedulerVector<6> curvatureScheduler;
  // next scheduler is used for midline-bending control points for RL:
  Schedulers::ParameterSchedulerLearnWave<7> rlBendingScheduler;

  // next scheduler is used to ramp-up the period
  Schedulers::ParameterSchedulerScalar periodScheduler;
  Real current_period = Tperiod;
  Real next_period = Tperiod;
  Real transition_start = 0.0;
  Real transition_duration = 0.1 * Tperiod;

protected:
  Real *const rK;
  Real *const vK;
  Real *const rC;
  Real *const vC;
  Real *const rB;
  Real *const vB;

public:
  CurvatureFish(Real L, Real T, Real phi, Real _h, Real _A)
      : FishData(L, _h), amplitudeFactor(_A), phaseShift(phi), Tperiod(T),
        rK(_alloc(Nm)), vK(_alloc(Nm)), rC(_alloc(Nm)), vC(_alloc(Nm)),
        rB(_alloc(Nm)), vB(_alloc(Nm)) {
    _computeWidth();
    writeMidline2File(0, "initialCheck");
  }

  void resetAll() override {
    curv_PID_fac = 0;
    curv_PID_dif = 0;
    avgDeltaY = 0;
    avgDangle = 0;
    avgAngVel = 0;
    lastTact = 0;
    lastCurv = 0;
    oldrCurv = 0;
    periodPIDval = Tperiod;
    periodPIDdif = 0;
    TperiodPID = false;
    time0 = 0;
    timeshift = 0;
    lastTime = 0;
    lastAvel = 0;
    curvatureScheduler.resetAll();
    periodScheduler.resetAll();
    rlBendingScheduler.resetAll();

    FishData::resetAll();
  }

  void correctTrajectory(const Real dtheta, const Real vtheta, const Real t,
                         const Real dt) {
    curv_PID_fac = dtheta;
    curv_PID_dif = vtheta;
  }

  void correctTailPeriod(const Real periodFac, const Real periodVel,
                         const Real t, const Real dt) {
    assert(periodFac > 0 && periodFac < 2); // would be crazy

    const Real lastArg = (lastTime - time0) / periodPIDval + timeshift;
    time0 = lastTime;
    timeshift = lastArg;
    // so that new arg is only constant (prev arg) + dt / periodPIDval
    // with the new l_Tp:
    periodPIDval = Tperiod * periodFac;
    periodPIDdif = Tperiod * periodVel;
    lastTime = t;
    TperiodPID = true;
  }

  // Execute takes as arguments the current simulation time and the time
  // the RL action should have actually started. This is important for the
  // midline bending because it relies on matching the splines with the half
  // period of the sinusoidal describing the swimming motion (in order to
  // exactly amplify or dampen the undulation). Therefore, for Tp=1, t_rlAction
  // might be K * 0.5 while t_current would be K * 0.5 plus a fraction of the
  // timestep. This because the new RL discrete step is detected as soon as
  // t_current>=t_rlAction
  void execute(const Real t_current, const Real t_rlAction,
               const std::vector<Real> &a) {
    assert(t_current >= t_rlAction);
    oldrCurv = lastCurv; // store action
    lastCurv = a[0];     // store action

    rlBendingScheduler.Turn(a[0], t_rlAction);

    if (a.size() > 1) // also modify the swimming period
    {
      if (TperiodPID)
        std::cout << "Warning: PID controller should not be used with RL."
                  << std::endl;
      lastTact = a[1]; // store action
      current_period = periodPIDval;
      next_period = Tperiod * (1 + a[1]);
      transition_start = t_rlAction;
    }
  }

  ~CurvatureFish() override {
    _dealloc(rK);
    _dealloc(vK);
    _dealloc(rC);
    _dealloc(vC);
    _dealloc(rB);
    _dealloc(vB);
  }

  void computeMidline(const Real time, const Real dt) override;
  Real _width(const Real s, const Real L) override {
    const Real sb = .04 * length, st = .95 * length, wt = .01 * length,
               wh = .04 * length;
    if (s < 0 or s > L)
      return 0;
    const Real w =
        (s < sb ? std::sqrt(2 * wh * s - s * s)
                : (s < st ? wh - (wh - wt) * std::pow((s - sb) / (st - sb), 1)
                          : // pow(.,2) is 3D
                       (wt * (L - s) / (L - st))));
    // std::cout << "s=" << s << ", w=" << w << std::endl;
    assert(w >= 0);
    return w;
  }
};

void StefanFish::resetAll() {
  CurvatureFish *const cFish = dynamic_cast<CurvatureFish *>(myFish);
  if (cFish == nullptr) {
    printf("Someone touched my fish\n");
    abort();
  }
  cFish->resetAll();
  Fish::resetAll();
}

void StefanFish::saveRestart(FILE *f) {
  assert(f != NULL);
  Fish::saveRestart(f);
  CurvatureFish *const cFish = dynamic_cast<CurvatureFish *>(myFish);
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(7) << "_" << obstacleID << "_";
  std::string filename = "Schedulers" + ss.str() + ".restart";
  {
    std::ofstream savestream;
    savestream.setf(std::ios::scientific);
    savestream.precision(std::numeric_limits<Real>::digits10 + 1);
    savestream.open(filename);
    {
      const auto &c = cFish->curvatureScheduler;
      savestream << c.t0 << "\t" << c.t1 << std::endl;
      for (int i = 0; i < c.npoints; ++i)
        savestream << c.parameters_t0[i] << "\t" << c.parameters_t1[i] << "\t"
                   << c.dparameters_t0[i] << std::endl;
    }
    {
      const auto &c = cFish->periodScheduler;
      savestream << c.t0 << "\t" << c.t1 << std::endl;
      for (int i = 0; i < c.npoints; ++i)
        savestream << c.parameters_t0[i] << "\t" << c.parameters_t1[i] << "\t"
                   << c.dparameters_t0[i] << std::endl;
    }
    {
      const auto &c = cFish->rlBendingScheduler;
      savestream << c.t0 << "\t" << c.t1 << std::endl;
      for (int i = 0; i < c.npoints; ++i)
        savestream << c.parameters_t0[i] << "\t" << c.parameters_t1[i] << "\t"
                   << c.dparameters_t0[i] << std::endl;
    }
    savestream.close();
  }

  // Save these numbers for PID controller and other stuff. Maybe not all of
  // them are needed but we don't care, it's only a few numbers.
  fprintf(f, "curv_PID_fac: %20.20e\n", (double)cFish->curv_PID_fac);
  fprintf(f, "curv_PID_dif: %20.20e\n", (double)cFish->curv_PID_dif);
  fprintf(f, "avgDeltaY   : %20.20e\n", (double)cFish->avgDeltaY);
  fprintf(f, "avgDangle   : %20.20e\n", (double)cFish->avgDangle);
  fprintf(f, "avgAngVel   : %20.20e\n", (double)cFish->avgAngVel);
  fprintf(f, "lastTact    : %20.20e\n", (double)cFish->lastTact);
  fprintf(f, "lastCurv    : %20.20e\n", (double)cFish->lastCurv);
  fprintf(f, "oldrCurv    : %20.20e\n", (double)cFish->oldrCurv);
  fprintf(f, "periodPIDval: %20.20e\n", (double)cFish->periodPIDval);
  fprintf(f, "periodPIDdif: %20.20e\n", (double)cFish->periodPIDdif);
  fprintf(f, "time0       : %20.20e\n", (double)cFish->time0);
  fprintf(f, "timeshift   : %20.20e\n", (double)cFish->timeshift);
  fprintf(f, "lastTime    : %20.20e\n", (double)cFish->lastTime);
  fprintf(f, "lastAvel    : %20.20e\n", (double)cFish->lastAvel);
}

void StefanFish::loadRestart(FILE *f) {
  assert(f != NULL);
  Fish::loadRestart(f);
  CurvatureFish *const cFish = dynamic_cast<CurvatureFish *>(myFish);
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(7) << "_" << obstacleID << "_";
  std::ifstream restartstream;
  std::string filename = "Schedulers" + ss.str() + ".restart";
  restartstream.open(filename);
  {
    auto &c = cFish->curvatureScheduler;
    restartstream >> c.t0 >> c.t1;
    for (int i = 0; i < c.npoints; ++i)
      restartstream >> c.parameters_t0[i] >> c.parameters_t1[i] >>
          c.dparameters_t0[i];
  }
  {
    auto &c = cFish->periodScheduler;
    restartstream >> c.t0 >> c.t1;
    for (int i = 0; i < c.npoints; ++i)
      restartstream >> c.parameters_t0[i] >> c.parameters_t1[i] >>
          c.dparameters_t0[i];
  }
  {
    auto &c = cFish->rlBendingScheduler;
    restartstream >> c.t0 >> c.t1;
    for (int i = 0; i < c.npoints; ++i)
      restartstream >> c.parameters_t0[i] >> c.parameters_t1[i] >>
          c.dparameters_t0[i];
  }
  restartstream.close();

  bool ret = true;
  double in_curv_PID_fac, in_curv_PID_dif, in_avgDeltaY, in_avgDangle,
      in_avgAngVel, in_lastTact, in_lastCurv, in_oldrCurv, in_periodPIDval,
      in_periodPIDdif, in_time0, in_timeshift, in_lastTime, in_lastAvel;
  ret = ret && 1 == fscanf(f, "curv_PID_fac: %le\n", &in_curv_PID_fac);
  ret = ret && 1 == fscanf(f, "curv_PID_dif: %le\n", &in_curv_PID_dif);
  ret = ret && 1 == fscanf(f, "avgDeltaY   : %le\n", &in_avgDeltaY);
  ret = ret && 1 == fscanf(f, "avgDangle   : %le\n", &in_avgDangle);
  ret = ret && 1 == fscanf(f, "avgAngVel   : %le\n", &in_avgAngVel);
  ret = ret && 1 == fscanf(f, "lastTact    : %le\n", &in_lastTact);
  ret = ret && 1 == fscanf(f, "lastCurv    : %le\n", &in_lastCurv);
  ret = ret && 1 == fscanf(f, "oldrCurv    : %le\n", &in_oldrCurv);
  ret = ret && 1 == fscanf(f, "periodPIDval: %le\n", &in_periodPIDval);
  ret = ret && 1 == fscanf(f, "periodPIDdif: %le\n", &in_periodPIDdif);
  ret = ret && 1 == fscanf(f, "time0       : %le\n", &in_time0);
  ret = ret && 1 == fscanf(f, "timeshift   : %le\n", &in_timeshift);
  ret = ret && 1 == fscanf(f, "lastTime    : %le\n", &in_lastTime);
  ret = ret && 1 == fscanf(f, "lastAvel    : %le\n", &in_lastAvel);
  cFish->curv_PID_fac = (Real)in_curv_PID_fac;
  cFish->curv_PID_dif = (Real)in_curv_PID_dif;
  cFish->avgDeltaY = (Real)in_avgDeltaY;
  cFish->avgDangle = (Real)in_avgDangle;
  cFish->avgAngVel = (Real)in_avgAngVel;
  cFish->lastTact = (Real)in_lastTact;
  cFish->lastCurv = (Real)in_lastCurv;
  cFish->oldrCurv = (Real)in_oldrCurv;
  cFish->periodPIDval = (Real)in_periodPIDval;
  cFish->periodPIDdif = (Real)in_periodPIDdif;
  cFish->time0 = (Real)in_time0;
  cFish->timeshift = (Real)in_timeshift;
  cFish->lastTime = (Real)in_lastTime;
  cFish->lastAvel = (Real)in_lastAvel;
  if ((not ret)) {
    printf("Error reading restart file. Aborting...\n");
    fflush(0);
    abort();
  }
}

StefanFish::StefanFish(SimulationData &s, cubism::ArgumentParser &p, Real C[2])
    : Fish(s, p, C), bCorrectTrajectory(p("-pid").asInt(0)),
      bCorrectPosition(p("-pidpos").asInt(0)) {
#if 0
  // parse tau
  tau = parser("-tau").asDouble(1.0);
  // parse curvature controlpoint values
  curvature_values[0] = parser("-k1").asDouble(0.82014);
  curvature_values[1] = parser("-k2").asDouble(1.46515);
  curvature_values[2] = parser("-k3").asDouble(2.57136);
  curvature_values[3] = parser("-k4").asDouble(3.75425);
  curvature_values[4] = parser("-k5").asDouble(5.09147);
  curvature_values[5] = parser("-k6").asDouble(5.70449);
  // if nonzero && Learnfreq<0 your fish is gonna keep turning
  baseline_values[0] = parser("-b1").asDouble(0.0);
  baseline_values[1] = parser("-b2").asDouble(0.0);
  baseline_values[2] = parser("-b3").asDouble(0.0);
  baseline_values[3] = parser("-b4").asDouble(0.0);
  baseline_values[4] = parser("-b5").asDouble(0.0);
  baseline_values[5] = parser("-b6").asDouble(0.0);
  // curvature points are distributed by default but can be overridden
  curvature_points[0] = parser("-pk1").asDouble(0.00)*length;
  curvature_points[1] = parser("-pk2").asDouble(0.15)*length;
  curvature_points[2] = parser("-pk3").asDouble(0.40)*length;
  curvature_points[3] = parser("-pk4").asDouble(0.65)*length;
  curvature_points[4] = parser("-pk5").asDouble(0.90)*length;
  curvature_points[5] = parser("-pk6").asDouble(1.00)*length;
  baseline_points[0] = parser("-pb1").asDouble(curvature_points[0]/length)*length;
  baseline_points[1] = parser("-pb2").asDouble(curvature_points[1]/length)*length;
  baseline_points[2] = parser("-pb3").asDouble(curvature_points[2]/length)*length;
  baseline_points[3] = parser("-pb4").asDouble(curvature_points[3]/length)*length;
  baseline_points[4] = parser("-pb5").asDouble(curvature_points[4]/length)*length;
  baseline_points[5] = parser("-pb6").asDouble(curvature_points[5]/length)*length;
  printf("created IF2D_StefanFish: xpos=%3.3f ypos=%3.3f angle=%3.3f L=%3.3f Tp=%3.3f tau=%3.3f phi=%3.3f\n",position[0],position[1],angle,length,Tperiod,tau,phaseShift);
  printf("curvature points: pk1=%3.3f pk2=%3.3f pk3=%3.3f pk4=%3.3f pk5=%3.3f pk6=%3.3f\n",curvature_points[0],curvature_points[1],curvature_points[2],curvature_points[3],curvature_points[4],curvature_points[5]);
  printf("curvature values (normalized to L=1): k1=%3.3f k2=%3.3f k3=%3.3f k4=%3.3f k5=%3.3f k6=%3.3f\n",curvature_values[0],curvature_values[1],curvature_values[2],curvature_values[3],curvature_values[4],curvature_values[5]);
  printf("baseline points: pb1=%3.3f pb2=%3.3f pb3=%3.3f pb4=%3.3f pb5=%3.3f pb6=%3.3f\n",baseline_points[0],baseline_points[1],baseline_points[2],baseline_points[3],baseline_points[4],baseline_points[5]);
  printf("baseline values (normalized to L=1): b1=%3.3f b2=%3.3f b3=%3.3f b4=%3.3f b5=%3.3f b6=%3.3f\n",baseline_values[0],baseline_values[1],baseline_values[2],baseline_values[3],baseline_values[4],baseline_values[5]);
  // make curvature dimensional for this length
  for(int i=0; i<6; ++i) curvature_values[i]/=length;
#endif

  const Real ampFac = p("-amplitudeFactor").asDouble(1.0);
  myFish = new CurvatureFish(length, Tperiod, phaseShift, sim.minH, ampFac);
  if (sim.rank == 0 && s.verbose)
    printf("[CUP2D] - CurvatureFish %d %f %f %f %f %f %f\n", myFish->Nm,
           (double)length, (double)myFish->dSref, (double)myFish->dSmid,
           (double)sim.minH, (double)Tperiod, (double)phaseShift);
}

// static inline Real sgn(const Real val) { return (0 < val) - (val < 0); }
void StefanFish::create(const std::vector<cubism::BlockInfo> &vInfo) {
  // If PID controller to keep position or swim straight enabled
  if (bCorrectPosition || bCorrectTrajectory) {
    CurvatureFish *const cFish = dynamic_cast<CurvatureFish *>(myFish);
    if (cFish == nullptr) {
      printf("Someone touched my fish\n");
      abort();
    }
    const Real DT = sim.dt / Tperiod; //, time = sim.time;
    // Control pos diffs
    const Real xDiff = (centerOfMass[0] - origC[0]) / length;
    const Real yDiff = (centerOfMass[1] - origC[1]) / length;
    const Real angDiff = orientation - origAng;
    const Real relU = (u + sim.uinfx) / length;
    const Real relV = (v + sim.uinfy) / length;
    const Real angVel = omega, lastAngVel = cFish->lastAvel;
    // compute ang vel at t - 1/2 dt such that we have a better derivative:
    const Real aVelMidP = (angVel + lastAngVel) * Tperiod / 2;
    const Real aVelDiff = (angVel - lastAngVel) * Tperiod / sim.dt;
    cFish->lastAvel = angVel; // store for next time

    // derivatives of following 2 exponential averages:
    const Real velDAavg = (angDiff - cFish->avgDangle) / Tperiod + DT * angVel;
    const Real velDYavg = (yDiff - cFish->avgDeltaY) / Tperiod + DT * relV;
    const Real velAVavg =
        10 * ((aVelMidP - cFish->avgAngVel) / Tperiod + DT * aVelDiff);
    // exponential averages
    cFish->avgDangle = (1.0 - DT) * cFish->avgDangle + DT * angDiff;
    cFish->avgDeltaY = (1.0 - DT) * cFish->avgDeltaY + DT * yDiff;
    // faster average:
    cFish->avgAngVel = (1 - 10 * DT) * cFish->avgAngVel + 10 * DT * aVelMidP;
    const Real avgDangle = cFish->avgDangle, avgDeltaY = cFish->avgDeltaY;

    // integral (averaged) and proportional absolute DY and their derivative
    const Real absPy = std::fabs(yDiff), absIy = std::fabs(avgDeltaY);
    const Real velAbsPy = yDiff > 0 ? relV : -relV;
    const Real velAbsIy = avgDeltaY > 0 ? velDYavg : -velDYavg;
    assert(origAng < 2e-16 && "TODO: rotate pos and vel to fish POV to enable \
                             PID to work even for non-zero angles");

    if (bCorrectPosition && sim.dt > 0) {
      // If angle is positive: positive curvature only if Dy<0 (must go up)
      // If angle is negative: negative curvature only if Dy>0 (must go down)
      const Real IangPdy = (avgDangle * yDiff < 0) ? avgDangle * absPy : 0;
      const Real PangIdy = (angDiff * avgDeltaY < 0) ? angDiff * absIy : 0;
      const Real IangIdy = (avgDangle * avgDeltaY < 0) ? avgDangle * absIy : 0;

      // derivatives multiplied by 0 when term is inactive later:
      const Real velIangPdy = velAbsPy * avgDangle + absPy * velDAavg;
      const Real velPangIdy = velAbsIy * angDiff + absIy * angVel;
      const Real velIangIdy = velAbsIy * avgDangle + absIy * velDAavg;

      // zero also the derivatives when appropriate
      const Real coefIangPdy = avgDangle * yDiff < 0 ? 1 : 0;
      const Real coefPangIdy = angDiff * avgDeltaY < 0 ? 1 : 0;
      const Real coefIangIdy = avgDangle * avgDeltaY < 0 ? 1 : 0;

      const Real valIangPdy = coefIangPdy * IangPdy;
      const Real difIangPdy = coefIangPdy * velIangPdy;
      const Real valPangIdy = coefPangIdy * PangIdy;
      const Real difPangIdy = coefPangIdy * velPangIdy;
      const Real valIangIdy = coefIangIdy * IangIdy;
      const Real difIangIdy = coefIangIdy * velIangIdy;
      const Real periodFac = 1.0 - xDiff;
      const Real periodVel = -relU;
#if 0
      if(not sim.muteAll) {
        std::ofstream filePID;
        std::stringstream ssF;
        ssF<<sim.path2file<<"/PID_"<<obstacleID<<".dat";
        filePID.open(ssF.str().c_str(), std::ios::app);
        filePID<<time<<" "<<valIangPdy<<" "<<difIangPdy
                     <<" "<<valPangIdy<<" "<<difPangIdy
                     <<" "<<valIangIdy<<" "<<difIangIdy
                     <<" "<<periodFac <<" "<<periodVel <<"\n";
      }
#endif
      const Real totalTerm = valIangPdy + valPangIdy + valIangIdy;
      const Real totalDiff = difIangPdy + difPangIdy + difIangIdy;
      cFish->correctTrajectory(totalTerm, totalDiff, sim.time, sim.dt);
      cFish->correctTailPeriod(periodFac, periodVel, sim.time, sim.dt);
    }
    // if absIy<EPS then we have just one fish that the simulation box follows
    // therefore we control the average angle but not the Y disp (which is 0)
    else if (bCorrectTrajectory && sim.dt > 0) {
      const Real avgAngVel = cFish->avgAngVel, absAngVel = std::fabs(avgAngVel);
      const Real absAvelDiff = avgAngVel > 0 ? velAVavg : -velAVavg;
      const Real coefInst = angDiff * avgAngVel > 0 ? 0.01 : 1, coefAvg = 0.1;
      const Real termInst = angDiff * absAngVel;
      const Real diffInst = angDiff * absAvelDiff + angVel * absAngVel;
      const Real totalTerm = coefInst * termInst + coefAvg * avgDangle;
      const Real totalDiff = coefInst * diffInst + coefAvg * velDAavg;

#if 0
      if(not sim.muteAll) {
        std::ofstream filePID;
        std::stringstream ssF;
        ssF<<sim.path2file<<"/PID_"<<obstacleID<<".dat";
        filePID.open(ssF.str().c_str(), std::ios::app);
        filePID<<time<<" "<<coefInst*termInst<<" "<<coefInst*diffInst
                     <<" "<<coefAvg*avgDangle<<" "<<coefAvg*velDAavg<<"\n";
      }
#endif
      cFish->correctTrajectory(totalTerm, totalDiff, sim.time, sim.dt);
    }
  }
  Fish::create(vInfo);
}

void StefanFish::act(const Real t_rlAction, const std::vector<Real> &a) const {
  CurvatureFish *const cFish = dynamic_cast<CurvatureFish *>(myFish);
  cFish->execute(sim.time, t_rlAction, a);
}

Real StefanFish::getLearnTPeriod() const {
  const CurvatureFish *const cFish = dynamic_cast<CurvatureFish *>(myFish);
  // return cFish->periodPIDval;
  return cFish->next_period;
}

Real StefanFish::getPhase(const Real t) const {
  const CurvatureFish *const cFish = dynamic_cast<CurvatureFish *>(myFish);
  const Real T0 = cFish->time0;
  const Real Ts = cFish->timeshift;
  const Real Tp = cFish->periodPIDval;
  const Real arg = 2 * M_PI * ((t - T0) / Tp + Ts) + M_PI * phaseShift;
  const Real phase = std::fmod(arg, 2 * M_PI);
  return (phase < 0) ? 2 * M_PI + phase : phase;
}

std::vector<Real> StefanFish::state(const std::vector<double> &origin) const {
  const CurvatureFish *const cFish = dynamic_cast<CurvatureFish *>(myFish);
  std::vector<Real> S(16, 0);
  S[0] = (center[0] - origin[0]) / length;
  S[1] = (center[1] - origin[1]) / length;
  S[2] = getOrientation();
  S[3] = getPhase(sim.time);
  S[4] = getU() * Tperiod / length;
  S[5] = getV() * Tperiod / length;
  S[6] = getW() * Tperiod;
  S[7] = cFish->lastTact;
  S[8] = cFish->lastCurv;
  S[9] = cFish->oldrCurv;

  // Shear stress computation at three sensors
  //******************************************
  //  Get fish skin
  const auto &DU = myFish->upperSkin;
  const auto &DL = myFish->lowerSkin;

  // index for sensors on the side of head
  int iHeadSide = 0;
  for (int i = 0; i < myFish->Nm - 1; ++i)
    if (myFish->rS[i] <= 0.04 * length && myFish->rS[i + 1] > 0.04 * length)
      iHeadSide = i;
  assert(iHeadSide > 0);

  // sensor locations
  const std::array<Real, 2> locFront = {DU.xSurf[0], DU.ySurf[0]};
  const std::array<Real, 2> locUpper = {DU.midX[iHeadSide], DU.midY[iHeadSide]};
  const std::array<Real, 2> locLower = {DL.midX[iHeadSide], DL.midY[iHeadSide]};

  // compute shear stress force (x,y) components
  std::array<Real, 2> shearFront = getShear(locFront);
  std::array<Real, 2> shearUpper = getShear(locLower);
  std::array<Real, 2> shearLower = getShear(locUpper);

  // normal vectors at sensor locations (these vectors already have unit length)
  //  first point of the two skins is the same normal should be almost the same:
  //  take the mean
  const std::array<Real, 2> norFront = {
      0.5 * (DU.normXSurf[0] + DL.normXSurf[0]),
      0.5 * (DU.normYSurf[0] + DL.normYSurf[0])};
  const std::array<Real, 2> norUpper = {DU.normXSurf[iHeadSide],
                                        DU.normYSurf[iHeadSide]};
  const std::array<Real, 2> norLower = {DL.normXSurf[iHeadSide],
                                        DL.normYSurf[iHeadSide]};

  // tangent vectors at sensor locations (these vectors already have unit
  // length) signs alternate so that both upper and lower tangent vectors point
  // towards fish tail
  const std::array<Real, 2> tanFront = {norFront[1], -norFront[0]};
  const std::array<Real, 2> tanUpper = {-norUpper[1], norUpper[0]};
  const std::array<Real, 2> tanLower = {norLower[1], -norLower[0]};

  // project three stresses to normal and tangent directions
  const double shearFront_n =
      shearFront[0] * norFront[0] + shearFront[1] * norFront[1];
  const double shearUpper_n =
      shearUpper[0] * norUpper[0] + shearUpper[1] * norUpper[1];
  const double shearLower_n =
      shearLower[0] * norLower[0] + shearLower[1] * norLower[1];
  const double shearFront_t =
      shearFront[0] * tanFront[0] + shearFront[1] * tanFront[1];
  const double shearUpper_t =
      shearUpper[0] * tanUpper[0] + shearUpper[1] * tanUpper[1];
  const double shearLower_t =
      shearLower[0] * tanLower[0] + shearLower[1] * tanLower[1];

  // put non-dimensional results into state into state
  S[10] = shearFront_n * Tperiod / length;
  S[11] = shearFront_t * Tperiod / length;
  S[12] = shearLower_n * Tperiod / length;
  S[13] = shearLower_t * Tperiod / length;
  S[14] = shearUpper_n * Tperiod / length;
  S[15] = shearUpper_t * Tperiod / length;

  return S;
}

std::vector<Real> StefanFish::state3D() const {
  const CurvatureFish *const cFish = dynamic_cast<CurvatureFish *>(myFish);
  std::vector<Real> S(25);
  S[0] = center[0];
  S[1] = center[1];
  S[2] = 1.0;

  // convert angle to quaternion
  S[3] = cos(0.5 * getOrientation());
  S[4] = 0.0;
  S[5] = 0.0;
  S[6] = sin(0.5 * getOrientation());

  S[7] = getPhase(sim.time);

  S[8] = getU() * Tperiod / length;
  S[9] = getV() * Tperiod / length;
  S[10] = 0.0;

  S[11] = 0.0;
  S[12] = 0.0;
  S[13] = getW() * Tperiod;

  S[14] = cFish->lastCurv;
  S[15] = cFish->oldrCurv;

  // Shear stress computation at three sensors
  //******************************************
  //  Get fish skin
  const auto &DU = myFish->upperSkin;
  const auto &DL = myFish->lowerSkin;

  // index for sensors on the side of head
  int iHeadSide = 0;
  for (int i = 0; i < myFish->Nm - 1; ++i)
    if (myFish->rS[i] <= 0.04 * length && myFish->rS[i + 1] > 0.04 * length)
      iHeadSide = i;
  assert(iHeadSide > 0);

  // sensor locations
  const std::array<Real, 2> locFront = {DU.xSurf[0], DU.ySurf[0]};
  const std::array<Real, 2> locUpper = {DU.midX[iHeadSide], DU.midY[iHeadSide]};
  const std::array<Real, 2> locLower = {DL.midX[iHeadSide], DL.midY[iHeadSide]};

  // compute shear stress force (x,y) components
  std::array<Real, 2> shearFront = getShear(locFront);
  std::array<Real, 2> shearUpper = getShear(locLower);
  std::array<Real, 2> shearLower = getShear(locUpper);
  S[16] = shearFront[0] * Tperiod / length;
  S[17] = shearFront[1] * Tperiod / length;
  S[18] = 0.0;
  S[19] = shearLower[0] * Tperiod / length;
  S[20] = shearLower[1] * Tperiod / length;
  S[21] = 0.0;
  S[22] = shearUpper[0] * Tperiod / length;
  S[23] = shearUpper[1] * Tperiod / length;
  S[24] = 0.0;
#if 0
  //normal vectors at sensor locations (these vectors already have unit length)
  // first point of the two skins is the same normal should be almost the same: take the mean
  const std::array<Real,2> norFront = {0.5*(DU.normXSurf[0] + DL.normXSurf[0]), 0.5*(DU.normYSurf[0] + DL.normYSurf[0]) };
  const std::array<Real,2> norUpper = { DU.normXSurf[iHeadSide], DU.normYSurf[iHeadSide]};
  const std::array<Real,2> norLower = { DL.normXSurf[iHeadSide], DL.normYSurf[iHeadSide]};

  //tangent vectors at sensor locations (these vectors already have unit length)
  //signs alternate so that both upper and lower tangent vectors point towards fish tail
  const std::array<Real,2> tanFront = { norFront[1],-norFront[0]};
  const std::array<Real,2> tanUpper = {-norUpper[1], norUpper[0]};
  const std::array<Real,2> tanLower = { norLower[1],-norLower[0]};

  // project three stresses to normal and tangent directions
  const double shearFront_n = shearFront[0]*norFront[0]+shearFront[1]*norFront[1];
  const double shearUpper_n = shearUpper[0]*norUpper[0]+shearUpper[1]*norUpper[1];
  const double shearLower_n = shearLower[0]*norLower[0]+shearLower[1]*norLower[1];
  const double shearFront_t = shearFront[0]*tanFront[0]+shearFront[1]*tanFront[1];
  const double shearUpper_t = shearUpper[0]*tanUpper[0]+shearUpper[1]*tanUpper[1];
  const double shearLower_t = shearLower[0]*tanLower[0]+shearLower[1]*tanLower[1];

  // put non-dimensional results into state into state
  S[10] = shearFront_n * Tperiod / length;
  S[11] = shearFront_t * Tperiod / length;
  S[12] = shearLower_n * Tperiod / length;
  S[13] = shearLower_t * Tperiod / length;
  S[14] = shearUpper_n * Tperiod / length;
  S[15] = shearUpper_t * Tperiod / length;
#endif

  return S;
}

/* helpers to compute sensor information */

// function that finds block id of block containing pos (x,y)
ssize_t StefanFish::holdingBlockID(const std::array<Real, 2> pos) const {
  const std::vector<cubism::BlockInfo> &velInfo = sim.vel->getBlocksInfo();
  for (size_t i = 0; i < velInfo.size(); ++i) {
    // compute lower left and top right corners of block (+- 0.5 h because pos
    // returns cell centers)
    std::array<Real, 2> MIN = velInfo[i].pos<Real>(0, 0);
    std::array<Real, 2> MAX =
        velInfo[i].pos<Real>(VectorBlock::sizeX - 1, VectorBlock::sizeY - 1);
    MIN[0] -= 0.5 * velInfo[i].h;
    MIN[1] -= 0.5 * velInfo[i].h;
    MAX[0] += 0.5 * velInfo[i].h;
    MAX[1] += 0.5 * velInfo[i].h;

    // check whether point is inside block
    if (pos[0] >= MIN[0] && pos[1] >= MIN[1] && pos[0] <= MAX[0] &&
        pos[1] <= MAX[1]) {
      return i;
    }
  }
  return -1; // rank does not contain point
};

// returns shear at given surface location
std::array<Real, 2>
StefanFish::getShear(const std::array<Real, 2> pSurf) const {
  const std::vector<cubism::BlockInfo> &velInfo = sim.vel->getBlocksInfo();

  Real myF[2] = {0, 0};

  // Get blockId of block that contains point pSurf.
  ssize_t blockIdSurf = holdingBlockID(pSurf);
  char error = false;
  if (blockIdSurf >= 0) {
    const auto &skinBinfo = velInfo[blockIdSurf];

    // check whether obstacle block exists
    if (obstacleBlocks[blockIdSurf] == nullptr) {
      printf("[CUP2D, rank %u] velInfo[%lu] contains point (%f,%f), but "
             "obstacleBlocks[%lu] is a nullptr! obstacleBlocks.size()=%lu\n",
             sim.rank, blockIdSurf, (double)pSurf[0], (double)pSurf[1],
             blockIdSurf, obstacleBlocks.size());
      const std::vector<cubism::BlockInfo> &chiInfo = sim.chi->getBlocksInfo();
      const auto &chiBlock = chiInfo[blockIdSurf];
      ScalarBlock &__restrict__ CHI = *(ScalarBlock *)chiBlock.ptrBlock;
      for (size_t i = 0; i < ScalarBlock::sizeX; i++)
        for (size_t j = 0; j < ScalarBlock::sizeY; j++) {
          const auto pos = chiBlock.pos<Real>(i, j);
          printf("i,j=%ld,%ld: pos=(%f,%f) with chi=%f\n", i, j, (double)pos[0],
                 (double)pos[1], (double)CHI(i, j).s);
        }
      fflush(0);
      error = true;
    } else {
      Real dmin = 1e10;
      ObstacleBlock *const O = obstacleBlocks[blockIdSurf];
      for (size_t k = 0; k < O->n_surfPoints; ++k) {
        const int ix = O->surface[k]->ix;
        const int iy = O->surface[k]->iy;
        const std::array<Real, 2> p = skinBinfo.pos<Real>(ix, iy);
        const Real d = (p[0] - pSurf[0]) * (p[0] - pSurf[0]) +
                       (p[1] - pSurf[1]) * (p[1] - pSurf[1]);
        if (d < dmin) {
          dmin = d;
          myF[0] = O->fXv_s[k];
          myF[1] = O->fYv_s[k];
        }
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, myF, 2, MPI_Real, MPI_SUM,
                sim.chi->getWorldComm());

// DEBUG purposes
#if 1
  MPI_Allreduce(MPI_IN_PLACE, &blockIdSurf, 1, MPI_INT64_T, MPI_MAX,
                sim.chi->getWorldComm());
  if (sim.rank == 0 && blockIdSurf == -1) {
    printf("ABORT: coordinate (%g,%g) could not be associated to ANY obstacle "
           "block\n",
           (double)pSurf[0], (double)pSurf[1]);
    fflush(0);
    abort();
  }
  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_CHAR, MPI_LOR,
                sim.chi->getWorldComm());
  if (error) {
    sim.dumpAll("failed");
    abort();
  }
#endif

  // return shear
  return std::array<Real, 2>{{myF[0], myF[1]}};
};

void CurvatureFish::computeMidline(const Real t, const Real dt) {
  periodScheduler.transition(t, transition_start,
                             transition_start + transition_duration,
                             current_period, next_period);
  periodScheduler.gimmeValues(t, periodPIDval, periodPIDdif);
  if (transition_start < t &&
      t < transition_start + transition_duration) // timeshift also rampedup
  {
    timeshift = (t - time0) / periodPIDval + timeshift;
    time0 = t;
  }

  // define interpolation points on midline
  const std::array<Real, 6> curvaturePoints = {
      (Real)0,           (Real).15 * length,
      (Real).4 * length, (Real).65 * length,
      (Real).9 * length, length};
  // define values of curvature at interpolation points
  const std::array<Real, 6> curvatureValues = {
      (Real)0.82014 / length, (Real)1.46515 / length, (Real)2.57136 / length,
      (Real)3.75425 / length, (Real)5.09147 / length, (Real)5.70449 / length};
  // define interpolation points for RL action
  const std::array<Real, 7> bendPoints = {
      (Real)-.5, (Real)-.25, (Real)0, (Real).25, (Real).5, (Real).75, (Real)1};

// transition curvature from 0 to target values
#if 1 // ramp-up over Tperiod
  // Set 0.01*curvatureValues as initial values (not zeros).
  // This prevents the Poisson solver from exploding in some cases, when
  // starting from zero residuals.
  const std::array<Real, 6> curvatureZeros = {
      0.01 * curvatureValues[0], 0.01 * curvatureValues[1],
      0.01 * curvatureValues[2], 0.01 * curvatureValues[3],
      0.01 * curvatureValues[4], 0.01 * curvatureValues[5],
  };
  curvatureScheduler.transition(0, 0, Tperiod, curvatureZeros, curvatureValues);
#else // no rampup for debug
  curvatureScheduler.transition(t, 0, Tperiod, curvatureValues,
                                curvatureValues);
#endif

  // write curvature values
  curvatureScheduler.gimmeValues(t, curvaturePoints, Nm, rS, rC, vC);
  rlBendingScheduler.gimmeValues(t, periodPIDval, length, bendPoints, Nm, rS,
                                 rB, vB);

  // next term takes into account the derivative of periodPIDval in darg:
  const Real diffT = 1 - (t - time0) * periodPIDdif / periodPIDval;
  // time derivative of arg:
  const Real darg = 2 * M_PI / periodPIDval * diffT;
  const Real arg0 =
      2 * M_PI * ((t - time0) / periodPIDval + timeshift) + M_PI * phaseShift;

#pragma omp parallel for schedule(static)
  for (int i = 0; i < Nm; ++i) {
    const Real arg = arg0 - 2 * M_PI * rS[i] / length;
    rK[i] = amplitudeFactor * rC[i] * (std::sin(arg) + rB[i] + curv_PID_fac);
    vK[i] = amplitudeFactor *
            (vC[i] * (std::sin(arg) + rB[i] + curv_PID_fac) +
             rC[i] * (std::cos(arg) * darg + vB[i] + curv_PID_dif));
    assert(not std::isnan(rK[i]));
    assert(not std::isinf(rK[i]));
    assert(not std::isnan(vK[i]));
    assert(not std::isinf(vK[i]));
  }

  // solve frenet to compute midline parameters
  IF2D_Frenet2D::solve(Nm, rS, rK, vK, rX, rY, vX, vY, norX, norY, vNorX,
                       vNorY);
#if 0
   {
    FILE * f = fopen("stefan_profile","w");
    for(int i=0;i<Nm;++i)
      fprintf(f,"%d %g %g %g %g %g %g %g %g %g\n",
        i,rS[i],rX[i],rY[i],vX[i],vY[i],
        vNorX[i],vNorY[i],width[i],height[i]);
    fclose(f);
   }
#endif
}

/***** Old Helpers (here for backward compatibility) ******/

// function that finds block id of block containing pos (x,y)
ssize_t StefanFish::holdingBlockID(
    const std::array<Real, 2> pos,
    const std::vector<cubism::BlockInfo> &velInfo) const {
  for (size_t i = 0; i < velInfo.size(); ++i) {
    // get gridspacing in block
    const Real h = velInfo[i].h;

    // compute lower left corner of block
    std::array<Real, 2> MIN = velInfo[i].pos<Real>(0, 0);
    for (int j = 0; j < 2; ++j)
      MIN[j] -= 0.5 * h; // pos returns cell centers

    // compute top right corner of block
    std::array<Real, 2> MAX =
        velInfo[i].pos<Real>(VectorBlock::sizeX - 1, VectorBlock::sizeY - 1);
    for (int j = 0; j < 2; ++j)
      MAX[j] += 0.5 * h; // pos returns cell centers

    // check whether point is inside block
    if (pos[0] >= MIN[0] && pos[1] >= MIN[1] && pos[0] <= MAX[0] &&
        pos[1] <= MAX[1]) {
      // point lies inside this block
      return i;
    }
  }
  // rank does not contain point
  return -1;
};

// function that gives indice of point in block
std::array<int, 2> StefanFish::safeIdInBlock(const std::array<Real, 2> pos,
                                             const std::array<Real, 2> org,
                                             const Real invh) const {
  const int indx = (int)std::round((pos[0] - org[0]) * invh);
  const int indy = (int)std::round((pos[1] - org[1]) * invh);
  const int ix = std::min(std::max(0, indx), VectorBlock::sizeX - 1);
  const int iy = std::min(std::max(0, indy), VectorBlock::sizeY - 1);
  return std::array<int, 2>{{ix, iy}};
};

// returns shear at given surface location
std::array<Real, 2>
StefanFish::getShear(const std::array<Real, 2> pSurf,
                     const std::array<Real, 2> normSurf,
                     const std::vector<cubism::BlockInfo> &velInfo) const {
  // Buffer to broadcast velcities and gridspacing
  Real velocityH[3] = {0.0, 0.0, 0.0};

  // 1. Compute surface velocity on surface
  // get blockId of surface
  ssize_t blockIdSurf = holdingBlockID(pSurf, velInfo);

  // get surface velocity if block containing point found
  char error = false;
  if (blockIdSurf >= 0) {
    // get block
    const auto &skinBinfo = velInfo[blockIdSurf];

    // check whether obstacle block exists
    if (obstacleBlocks[blockIdSurf] == nullptr) {
      printf("[CUP2D, rank %u] velInfo[%lu] contains point (%f,%f), but "
             "obstacleBlocks[%lu] is a nullptr! obstacleBlocks.size()=%lu\n",
             sim.rank, blockIdSurf, pSurf[0], pSurf[1], blockIdSurf,
             obstacleBlocks.size());
      const std::vector<cubism::BlockInfo> &chiInfo = sim.chi->getBlocksInfo();
      const auto &chiBlock = chiInfo[blockIdSurf];
      ScalarBlock &__restrict__ CHI = *(ScalarBlock *)chiBlock.ptrBlock;
      for (size_t i = 0; i < ScalarBlock::sizeX; i++)
        for (size_t j = 0; j < ScalarBlock::sizeY; j++) {
          const auto pos = chiBlock.pos<Real>(i, j);
          printf("i,j=%ld,%ld: pos=(%f,%f) with chi=%f\n", i, j, pos[0], pos[1],
                 CHI(i, j).s);
        }
      fflush(0);
      error = true;
      // abort();
    } else {
      // get origin of block
      const std::array<Real, 2> oBlockSkin = skinBinfo.pos<Real>(0, 0);

      // get gridspacing on this block
      velocityH[2] = velInfo[blockIdSurf].h;

      // get index of point in block
      const std::array<int, 2> iSkin =
          safeIdInBlock(pSurf, oBlockSkin, 1 / velocityH[2]);

      // get deformation velocity
      const Real udefX =
          obstacleBlocks[blockIdSurf]->udef[iSkin[1]][iSkin[0]][0];
      const Real udefY =
          obstacleBlocks[blockIdSurf]->udef[iSkin[1]][iSkin[0]][1];

      // compute velocity of skin point
      velocityH[0] = u - omega * (pSurf[1] - centerOfMass[1]) + udefX;
      velocityH[1] = v + omega * (pSurf[0] - centerOfMass[0]) + udefY;
    }
  }

// DEBUG purposes
#if 1
  MPI_Allreduce(MPI_IN_PLACE, &blockIdSurf, 1, MPI_INT64_T, MPI_MAX,
                sim.chi->getWorldComm());
  if (sim.rank == 0 && blockIdSurf == -1) {
    printf("ABORT: coordinate (%g,%g) could not be associated to ANY obstacle "
           "block\n",
           (double)pSurf[0], (double)pSurf[1]);
    fflush(0);
    abort();
  }

  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_CHAR, MPI_LOR,
                sim.chi->getWorldComm());
  if (error) {
    sim.dumpAll("failed");
    abort();
  }
#endif

  // Allreduce to Bcast surface velocity
  MPI_Allreduce(MPI_IN_PLACE, velocityH, 3, MPI_Real, MPI_SUM,
                sim.chi->getWorldComm());

  // Assign skin velocities and grid-spacing
  const Real uSkin = velocityH[0];
  const Real vSkin = velocityH[1];
  const Real h = velocityH[2];
  const Real invh = 1 / h;

  // Reset buffer to 0
  velocityH[0] = 0.0;
  velocityH[1] = 0.0;
  velocityH[2] = 0.0;

  // 2. Compute flow velocity away from surface
  // compute point on lifted surface
  const std::array<Real, 2> pLiftedSurf = {pSurf[0] + h * normSurf[0],
                                           pSurf[1] + h * normSurf[1]};

  // get blockId of lifted surface
  const ssize_t blockIdLifted = holdingBlockID(pLiftedSurf, velInfo);

  // get surface velocity if block containing point found
  if (blockIdLifted >= 0) {
    // get block
    const auto &liftedBinfo = velInfo[blockIdLifted];

    // get origin of block
    const std::array<Real, 2> oBlockLifted = liftedBinfo.pos<Real>(0, 0);

    // get inverse gridspacing in block
    const Real invhLifted = 1 / velInfo[blockIdLifted].h;

    // get index for sensor
    const std::array<int, 2> iSens =
        safeIdInBlock(pLiftedSurf, oBlockLifted, invhLifted);

    // get velocity field at point
    const VectorBlock &b = *(const VectorBlock *)liftedBinfo.ptrBlock;
    velocityH[0] = b(iSens[0], iSens[1]).u[0];
    velocityH[1] = b(iSens[0], iSens[1]).u[1];
  }

  // Allreduce to Bcast flow velocity
  MPI_Allreduce(MPI_IN_PLACE, velocityH, 3, MPI_Real, MPI_SUM,
                sim.chi->getWorldComm());

  // Assign lifted skin velocities
  const Real uLifted = velocityH[0];
  const Real vLifted = velocityH[1];

  // return shear
  return std::array<Real, 2>{
      {(uLifted - uSkin) * invh, (vLifted - vSkin) * invh}};
};

class Simulation {
public:
  SimulationData sim;
  std::vector<std::shared_ptr<Operator>> pipeline;

protected:
  cubism::ArgumentParser parser;

  void createShapes();
  void parseRuntime();

public:
  Simulation(int argc, char **argv, MPI_Comm comm);
  ~Simulation();

  /// Find the first operator in the pipeline that matches the given type.
  /// Returns `nullptr` if nothing was found.
  template <typename Op> Op *findOperator() const {
    for (const auto &ptr : pipeline) {
      Op *out = dynamic_cast<Op *>(ptr.get());
      if (out != nullptr)
        return out;
    }
    return nullptr;
  }

  /// Insert the operator at the end of the pipeline.
  void insertOperator(std::shared_ptr<Operator> op);

  /// Insert an operator after the operator of the given name.
  /// Throws an exception if the name is not found.
  void insertOperatorAfter(std::shared_ptr<Operator> op,
                           const std::string &name);

  void reset();
  void resetRL();
  void init();
  void startObstacles();
  void simulate();
  Real calcMaxTimestep();
  void advance(const Real dt);

  const std::vector<std::shared_ptr<Shape>> &getShapes() { return sim.shapes; }
};

BCflag cubismBCX;
BCflag cubismBCY;

static const char kHorLine[] =
    "=======================================================================\n";

static inline std::vector<std::string> split(const std::string &s,
                                             const char dlm) {
  std::stringstream ss(s);
  std::string item;
  std::vector<std::string> tokens;
  while (std::getline(ss, item, dlm))
    tokens.push_back(item);
  return tokens;
}

Simulation::Simulation(int argc, char **argv, MPI_Comm comm)
    : parser(argc, argv) {
  sim.comm = comm;
  int size;
  MPI_Comm_size(sim.comm, &size);
  MPI_Comm_rank(sim.comm, &sim.rank);
  if (sim.rank == 0) {
    std::cout << "============================================================="
                 "==========\n";
    std::cout << "    CubismUP 2D (velocity-pressure 2D incompressible "
                 "Navier-Stokes)    \n";
    std::cout << "============================================================="
                 "==========\n";
    parser.print_args();
#pragma omp parallel
    {
      int numThreads = omp_get_num_threads();
#pragma omp master
      printf("[CUP2D] Running with %d rank(s) and %d thread(s).\n", size,
             numThreads);
    }
  }
}

Simulation::~Simulation() = default;

void Simulation::insertOperator(std::shared_ptr<Operator> op) {
  pipeline.push_back(std::move(op));
}
void Simulation::insertOperatorAfter(std::shared_ptr<Operator> op,
                                     const std::string &name) {
  for (size_t i = 0; i < pipeline.size(); ++i) {
    if (pipeline[i]->getName() == name) {
      pipeline.insert(pipeline.begin() + i + 1, std::move(op));
      return;
    }
  }
  std::string msg;
  msg.reserve(300);
  msg += "operator '";
  msg += name;
  msg += "' not found, available: ";
  for (size_t i = 0; i < pipeline.size(); ++i) {
    if (i > 0)
      msg += ", ";
    msg += pipeline[i]->getName();
  }
  msg += " (ensure that init() is called before inserting custom operators)";
  throw std::runtime_error(std::move(msg));
}

void Simulation::init() {
  // parse field variables
  if (sim.rank == 0 && sim.verbose)
    std::cout << "[CUP2D] Parsing Simulation Configuration..." << std::endl;
  parseRuntime();
  // allocate the grid
  if (sim.rank == 0 && sim.verbose)
    std::cout << "[CUP2D] Allocating Grid..." << std::endl;
  sim.allocateGrid();
  // create shapes
  if (sim.rank == 0 && sim.verbose)
    std::cout << "[CUP2D] Creating Shapes..." << std::endl;
  createShapes();
  // impose field initial condition
  if (sim.rank == 0 && sim.verbose)
    std::cout << "[CUP2D] Imposing Initial Conditions..." << std::endl;
  if (sim.ic == "random") {
    randomIC ic(sim);
    ic(0);
  } else {
    IC ic(sim);
    ic(0);
  }
  // create compute pipeline
  if (sim.rank == 0 && sim.verbose)
    std::cout << "[CUP2D] Creating Computational Pipeline..." << std::endl;

  pipeline.push_back(std::make_shared<AdaptTheMesh>(sim));
  pipeline.push_back(std::make_shared<PutObjectsOnGrid>(sim));
  pipeline.push_back(std::make_shared<advDiff>(sim));
  pipeline.push_back(std::make_shared<PressureSingle>(sim));
  pipeline.push_back(std::make_shared<ComputeForces>(sim));

  if (sim.rank == 0 && sim.verbose) {
    std::cout << "[CUP2D] Operator ordering:\n";
    for (size_t c = 0; c < pipeline.size(); c++)
      std::cout << "[CUP2D] - " << pipeline[c]->getName() << "\n";
  }

  // Put Object on Intially defined Mesh and impose obstacle velocities
  startObstacles();
}

void Simulation::parseRuntime() {
  // restart the simulation?
  sim.bRestart = parser("-restart").asBool(false);

  /* parameters that have to be given */
  /************************************/
  parser.set_strict_mode();

  // set initial number of blocks
  sim.bpdx = parser("-bpdx").asInt();
  sim.bpdy = parser("-bpdy").asInt();

  // maximal number of refinement levels
  sim.levelMax = parser("-levelMax").asInt();

  // refinement/compression tolerance for vorticity magnitude
  sim.Rtol = parser("-Rtol").asDouble();
  sim.Ctol = parser("-Ctol").asDouble();

  parser.unset_strict_mode();
  /************************************/
  /************************************/

  // refiment according to Qcriterion instead of |omega|
  sim.Qcriterion = parser("-Qcriterion").asBool(false);

  // check for refinement every this many timesteps
  sim.AdaptSteps = parser("-AdaptSteps").asInt(20);

  // boolean to switch between refinement according to chi or grad(chi)
  sim.bAdaptChiGradient = parser("-bAdaptChiGradient").asInt(1);

  // initial level of refinement
  sim.levelStart = parser("-levelStart").asInt(-1);
  if (sim.levelStart == -1)
    sim.levelStart = sim.levelMax - 1;

  // simulation extent
  sim.extent = parser("-extent").asDouble(1);

  // timestep / CFL number
  sim.dt = parser("-dt").asDouble(0);
  sim.CFL = parser("-CFL").asDouble(0.2);
  sim.rampup = parser("-rampup").asInt(0);

  // simulation ending parameters
  sim.nsteps = parser("-nsteps").asInt(0);
  sim.endTime = parser("-tend").asDouble(0);

  // penalisation coefficient
  sim.lambda = parser("-lambda").asDouble(1e7);

  // constant for explicit penalisation lambda=dlm/dt
  sim.dlm = parser("-dlm").asDouble(0);

  // kinematic viscocity
  sim.nu = parser("-nu").asDouble(1e-2);

  // forcing
  sim.bForcing = parser("-bForcing").asInt(0);
  sim.forcingWavenumber = parser("-forcingWavenumber").asDouble(4);
  sim.forcingCoefficient = parser("-forcingCoefficient").asDouble(4);

  // Smagorinsky Model
  sim.smagorinskyCoeff = parser("-smagorinskyCoeff").asDouble(0);
  sim.bDumpCs = parser("-dumpCs").asInt(0);

  // Flag for initial condition
  sim.ic = parser("-ic").asString("");

  // Boundary conditions (freespace or periodic)
  std::string BC_x = parser("-BC_x").asString("freespace");
  std::string BC_y = parser("-BC_y").asString("freespace");
  cubismBCX = string2BCflag(BC_x);
  cubismBCY = string2BCflag(BC_y);

  // poisson solver parameters
  sim.poissonSolver = parser("-poissonSolver").asString("iterative");
  sim.PoissonTol = parser("-poissonTol").asDouble(1e-6);
  sim.PoissonTolRel = parser("-poissonTolRel").asDouble(0);
  sim.maxPoissonRestarts = parser("-maxPoissonRestarts").asInt(30);
  sim.maxPoissonIterations = parser("-maxPoissonIterations").asInt(1000);
  sim.bMeanConstraint = parser("-bMeanConstraint").asInt(0);

  // output parameters
  sim.profilerFreq = parser("-profilerFreq").asInt(0);
  sim.dumpFreq = parser("-fdump").asInt(0);
  sim.dumpTime = parser("-tdump").asDouble(0);
  sim.path2file = parser("-file").asString("./");
  sim.path4serialization = parser("-serialization").asString(sim.path2file);
  sim.verbose = parser("-verbose").asInt(1);
  sim.muteAll = parser("-muteAll").asInt(0);
  sim.DumpUniform = parser("-DumpUniform").asBool(false);
  if (sim.muteAll)
    sim.verbose = 0;
}

void Simulation::createShapes() {
  const std::string shapeArg = parser("-shapes").asString("");
  std::stringstream descriptors(shapeArg);
  std::string lines;

  while (std::getline(descriptors, lines)) {
    std::replace(lines.begin(), lines.end(), '_', ' ');
    const std::vector<std::string> vlines = split(lines, ',');

    for (const auto &line : vlines) {
      std::istringstream line_stream(line);
      std::string objectName;
      if (sim.rank == 0 && sim.verbose)
        std::cout << "[CUP2D] " << line << std::endl;
      line_stream >> objectName;
      // Comments and empty lines ignored:
      if (objectName.empty() or objectName[0] == '#')
        continue;
      FactoryFileLineParser ffparser(line_stream);
      Real center[2] = {ffparser("-xpos").asDouble(.5 * sim.extents[0]),
                        ffparser("-ypos").asDouble(.5 * sim.extents[1])};
      // ffparser.print_args();
      Shape *shape = nullptr;
      if (objectName == "stefanfish")
        shape = new StefanFish(sim, ffparser, center);
      else
        throw std::invalid_argument("unrecognized shape: " + objectName);
      sim.addShape(std::shared_ptr<Shape>{shape});
    }
  }

  if (sim.shapes.size() == 0 && sim.rank == 0)
    std::cout << "Did not create any obstacles." << std::endl;
}

void Simulation::reset() {
  // reset field variables and shapes
  if (sim.rank == 0 && sim.verbose)
    std::cout << "[CUP2D] Resetting Simulation..." << std::endl;
  sim.resetAll();
  // impose field initial condition
  if (sim.rank == 0 && sim.verbose)
    std::cout << "[CUP2D] Imposing Initial Conditions..." << std::endl;
  IC ic(sim);
  ic(0);
  // Put Object on Intially defined Mesh and impose obstacle velocities
  startObstacles();
}

void Simulation::resetRL() {
  // reset simulation (not shape)
  if (sim.rank == 0 && sim.verbose)
    std::cout << "[CUP2D] Resetting Simulation..." << std::endl;
  sim.resetAll();
  // impose field initial condition
  if (sim.rank == 0 && sim.verbose)
    std::cout << "[CUP2D] Imposing Initial Conditions..." << std::endl;
  IC ic(sim);
  ic(0);
}

void Simulation::startObstacles() {
  Checker check(sim);

  // put obstacles to grid and compress
  if (sim.rank == 0 && sim.verbose && !sim.bRestart)
    std::cout << "[CUP2D] Initial PutObjectsOnGrid and Compression of Grid\n";
  PutObjectsOnGrid *const putObjectsOnGrid = findOperator<PutObjectsOnGrid>();
  AdaptTheMesh *const adaptTheMesh = findOperator<AdaptTheMesh>();
  assert(putObjectsOnGrid != nullptr && adaptTheMesh != nullptr);
  if (not sim.bRestart)
    for (int i = 0; i < sim.levelMax; i++) {
      (*putObjectsOnGrid)(0.0);
      (*adaptTheMesh)(0.0);
    }
  (*putObjectsOnGrid)(0.0);

  // impose velocity of obstacles
  if (not sim.bRestart) {
    if (sim.rank == 0 && sim.verbose)
      std::cout << "[CUP2D] Imposing Initial Velocity of Objects on field\n";
    ApplyObjVel initVel(sim);
    initVel(0);
  }
}

void Simulation::simulate() {
  if (sim.rank == 0 && !sim.muteAll)
    std::cout << kHorLine << "[CUP2D] Starting Simulation...\n" << std::flush;

  while (1) {
    Real dt = calcMaxTimestep();

    bool done = false;

    // Ignore the final time step if `dt` is way too small.
    if (!done || dt > 2e-16)
      advance(dt);

    if (!done)
      done = sim.bOver();

    if (sim.rank == 0 && sim.profilerFreq > 0 &&
        sim.step % sim.profilerFreq == 0)
      sim.printResetProfiler();

    if (done) {
      const bool bDump = sim.bDump();
      if (bDump) {
        if (sim.rank == 0 && sim.verbose)
          std::cout << "[CUP2D] dumping field...\n";
        sim.registerDump();
        sim.dumpAll("_");
      }
      if (sim.rank == 0 && !sim.muteAll) {
        std::cout << kHorLine
                  << "[CUP2D] Simulation Over... Profiling information:\n";
        sim.printResetProfiler();
        std::cout << kHorLine;
      }
      break;
    }
  }
}

Real Simulation::calcMaxTimestep() {
  sim.dt_old2 = sim.dt_old;
  sim.dt_old = sim.dt;
  Real CFL = sim.CFL;
  const Real h = sim.getH();
  const auto findMaxU_op = findMaxU(sim);
  sim.uMax_measured = findMaxU_op.run();

  if (CFL > 0) {
    const Real dtDiffusion =
        0.25 * h * h / (sim.nu + 0.25 * h * sim.uMax_measured);
    const Real dtAdvection = h / (sim.uMax_measured + 1e-8);

    // non-constant timestep introduces a source term = (1-dt_new/dt_old)
    // \nabla^2 P_{old} in the Poisson equation. Thus, we try to modify the
    // timestep less often
    if (sim.step < sim.rampup) {
      const Real x = (sim.step + 1.0) / sim.rampup;
      const Real rampupFactor = std::exp(std::log(1e-3) * (1 - x));
      sim.dt = rampupFactor * std::min({dtDiffusion, CFL * dtAdvection});
    } else {
      sim.dt = std::min({dtDiffusion, CFL * dtAdvection});
    }
  }

  if (sim.dt <= 0) {
    std::cout << "[CUP2D] dt <= 0. Aborting..." << std::endl;
    fflush(0);
    abort();
  }

  if (sim.dlm > 0)
    sim.lambda = sim.dlm / sim.dt;
  return sim.dt;
}

void Simulation::advance(const Real dt) {

  const Real CFL = (sim.uMax_measured + 1e-8) * sim.dt / sim.getH();
  if (sim.rank == 0 && !sim.muteAll) {
    std::cout << kHorLine;
    printf("[CUP2D] step:%d, blocks:%zu, time:%f, dt=%f, uinf:[%f %f], "
           "maxU:%f, CFL:%f\n",
           sim.step, sim.chi->getBlocksInfo().size(), (double)sim.time,
           (double)dt, (double)sim.uinfx, (double)sim.uinfy,
           (double)sim.uMax_measured, (double)CFL);
  }

  // dump field
  const bool bDump = sim.bDump();
  if (bDump) {
    if (sim.rank == 0 && sim.verbose)
      std::cout << "[CUP2D] dumping field...\n";
    sim.registerDump();
    sim.dumpAll("_");
  }

  for (size_t c = 0; c < pipeline.size(); c++) {
    if (sim.rank == 0 && sim.verbose)
      std::cout << "[CUP2D] running " << pipeline[c]->getName() << "...\n";
    (*pipeline[c])(dt);
  }
  sim.time += dt;
  sim.step++;
}

int main(int argc, char **argv) {
  int threadSafety;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadSafety);

  double time = -MPI_Wtime();

  Simulation *sim = new Simulation(argc, argv, MPI_COMM_WORLD);
  sim->init();
  sim->simulate();
  time += MPI_Wtime();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
    std::cout << "Runtime = " << time << std::endl;
  delete sim;
  MPI_Finalize();
  return 0;
}
