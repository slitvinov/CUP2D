template <typename ElementType> struct GridMPI {
  typedef ElementType Block[_BS_][_BS_];
  std::unordered_map<long long, BlockInfo *> BlockInfoAll;
  std::unordered_map<long long, TreePosition> Octree;
  std::vector<BlockInfo> m_vInfo;
  int NX;
  int NY;
  int NZ;
  double maxextent;
  int levelMax;
  int levelStart;
  bool xperiodic;
  bool yperiodic;
  bool zperiodic;
  std::vector<long long> level_base;
  bool UpdateFluxCorrection{true};
  bool UpdateGroups{true};
  bool FiniteDifferences{true};
  FluxCorrection<GridMPI, ElementType> CorrectorGrid;
  typedef SynchronizerMPI_AMR<GridMPI<ElementType>> SynchronizerMPIType;
  size_t timestamp;
  std::map<StencilInfo, SynchronizerMPIType *> SynchronizerMPIs;
  FluxCorrectionMPI<FluxCorrection<GridMPI<ElementType>, ElementType>,
                    ElementType>
      Corrector;
  std::vector<BlockInfo *> boundary;
  GridMPI(int nX, int nY, int nZ, double a_maxextent, int a_levelStart,
          int a_levelMax, bool a_xperiodic, bool a_yperiodic, bool a_zperiodic)
      : NX(nX), NY(nY), NZ(nZ), maxextent(a_maxextent), levelMax(a_levelMax),
        levelStart(a_levelStart), xperiodic(a_xperiodic),
        yperiodic(a_yperiodic), zperiodic(a_zperiodic), timestamp(0) {
    BlockInfo dummy;
    const int nx = dummy.blocks_per_dim(0, NX, NY);
    const int ny = dummy.blocks_per_dim(1, NX, NY);
    const int nz = 1;
    const int lvlMax = dummy.levelMax(levelMax);
    for (int m = 0; m < lvlMax; m++) {
      const int TwoPower = 1 << m;
      const long long Ntot = nx * ny * nz * pow(TwoPower, (Real)DIMENSION);
      if (m == 0)
        level_base.push_back(Ntot);
      if (m > 0)
        level_base.push_back(level_base[m - 1] + Ntot);
    }

    const long long total_blocks =
        nX * nY * nZ * pow(pow(2, a_levelStart), DIMENSION);
    long long my_blocks = total_blocks / sim.size;
    if ((long long)sim.rank < total_blocks % sim.size)
      my_blocks++;
    long long n_start = sim.rank * (total_blocks / sim.size);
    if (total_blocks % sim.size > 0) {
      if ((long long)sim.rank < total_blocks % sim.size)
        n_start += sim.rank;
      else
        n_start += total_blocks % sim.size;
    }
    std::vector<short int> levels(my_blocks, a_levelStart);
    std::vector<long long> Zs(my_blocks);
    for (long long n = n_start; n < n_start + my_blocks; n++)
      Zs[n - n_start] = n;
    initialize_blocks(Zs, levels);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  Block *avail(const int m, const long long n) {
    return (Tree(m, n).rank() == sim.rank)
               ? (Block *)getBlockInfoAll(m, n).ptrBlock
               : nullptr;
  }
  virtual void UpdateBoundary(bool clean = false) {
    const auto blocksPerDim = getMaxBlocks();
    std::vector<std::vector<long long>> send_buffer(sim.size);
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
        if (!xperiodic && code[0] == xskip && xskin)
          continue;
        if (!yperiodic && code[1] == yskip && yskin)
          continue;
        if (!zperiodic && code[2] == zskip && zskin)
          continue;
        if (code[2] != 0)
          continue;
        BlockInfo &infoNei =
            getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));
        const TreePosition &infoNeiTree = Tree(infoNei.level, infoNei.Z);
        if (infoNeiTree.Exists() && infoNeiTree.rank() != sim.rank) {
          if (infoNei.state != Refine || clean)
            infoNei.state = Leave;
          receivers.insert(infoNeiTree.rank());
          Neighbors.insert(infoNeiTree.rank());
        } else if (infoNeiTree.CheckCoarser()) {
          const long long nCoarse = infoNei.Zparent;
          BlockInfo &infoNeiCoarser =
              getBlockInfoAll(infoNei.level - 1, nCoarse);
          const int infoNeiCoarserrank =
              Tree(infoNei.level - 1, nCoarse).rank();
          if (infoNeiCoarserrank != sim.rank) {
            assert(infoNeiCoarserrank >= 0);
            if (infoNeiCoarser.state != Refine || clean)
              infoNeiCoarser.state = Leave;
            receivers.insert(infoNeiCoarserrank);
            Neighbors.insert(infoNeiCoarserrank);
          }
        } else if (infoNeiTree.CheckFiner()) {
          int Bstep = 1;
          if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))
            Bstep = 3;
          else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
            Bstep = 4;
          for (int B = 0; B <= 1; B += Bstep) {
            const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
            const long long nFine =
                infoNei.Zchild[std::max(-code[0], 0) +
                               (B % 2) * std::max(0, 1 - abs(code[0]))]
                              [std::max(-code[1], 0) +
                               temp * std::max(0, 1 - abs(code[1]))]
                              [std::max(-code[2], 0) +
                               (B / 2) * std::max(0, 1 - abs(code[2]))];
            BlockInfo &infoNeiFiner = getBlockInfoAll(infoNei.level + 1, nFine);
            const int infoNeiFinerrank = Tree(infoNei.level + 1, nFine).rank();
            if (infoNeiFinerrank != sim.rank) {
              if (infoNeiFiner.state != Refine || clean)
                infoNeiFiner.state = Leave;
              receivers.insert(infoNeiFinerrank);
              Neighbors.insert(infoNeiFinerrank);
            }
          }
        }
      }
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
      if (r != sim.rank) {
        requests.resize(requests.size() + 1);
        if (send_buffer[r].size() != 0)
          MPI_Isend(&send_buffer[r][0], send_buffer[r].size(), MPI_LONG_LONG, r,
                    123, MPI_COMM_WORLD, &requests[requests.size() - 1]);
        else {
          MPI_Isend(&dummy, 1, MPI_LONG_LONG, r, 123, MPI_COMM_WORLD,
                    &requests[requests.size() - 1]);
        }
      }
    std::vector<std::vector<long long>> recv_buffer(sim.size);
    for (int r : Neighbors)
      if (r != sim.rank) {
        int recv_size;
        MPI_Status status;
        MPI_Probe(r, 123, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_LONG_LONG, &recv_size);
        if (recv_size > 0) {
          recv_buffer[r].resize(recv_size);
          requests.resize(requests.size() + 1);
          MPI_Irecv(&recv_buffer[r][0], recv_buffer[r].size(), MPI_LONG_LONG, r,
                    123, MPI_COMM_WORLD, &requests[requests.size() - 1]);
        }
      }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    for (int r = 0; r < sim.size; r++)
      if (recv_buffer[r].size() > 1)
        for (int index = 0; index < (int)recv_buffer[r].size(); index += 3) {
          int level = recv_buffer[r][index];
          long long Z = recv_buffer[r][index + 1];
          getBlockInfoAll(level, Z).state =
              (recv_buffer[r][index + 2] == 1) ? Compress : Refine;
        }
  };
  void UpdateBlockInfoAll_States(bool UpdateIDs = false) {
    std::vector<int> myNeighbors = FindMyNeighbors();
    const auto blocksPerDim = getMaxBlocks();
    std::vector<long long> myData;
    for (auto &info : m_vInfo) {
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
        if (!xperiodic && code[0] == xskip && xskin)
          continue;
        if (!yperiodic && code[1] == yskip && yskin)
          continue;
        if (!zperiodic && code[2] == zskip && zskin)
          continue;
        if (code[2] != 0)
          continue;
        BlockInfo &infoNei =
            getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));
        const TreePosition &infoNeiTree = Tree(infoNei.level, infoNei.Z);
        if (infoNeiTree.Exists() && infoNeiTree.rank() != sim.rank) {
          myflag = true;
          break;
        } else if (infoNeiTree.CheckCoarser()) {
          long long nCoarse = infoNei.Zparent;
          const int infoNeiCoarserrank =
              Tree(infoNei.level - 1, nCoarse).rank();
          if (infoNeiCoarserrank != sim.rank) {
            myflag = true;
            break;
          }
        } else if (infoNeiTree.CheckFiner()) {
          int Bstep = 1;
          if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))
            Bstep = 3;
          else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
            Bstep = 4;
          for (int B = 0; B <= 3; B += Bstep) {
            const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
            const long long nFine =
                infoNei.Zchild[std::max(-code[0], 0) +
                               (B % 2) * std::max(0, 1 - abs(code[0]))]
                              [std::max(-code[1], 0) +
                               temp * std::max(0, 1 - abs(code[1]))]
                              [std::max(-code[2], 0) +
                               (B / 2) * std::max(0, 1 - abs(code[2]))];
            const int infoNeiFinerrank = Tree(infoNei.level + 1, nFine).rank();
            if (infoNeiFinerrank != sim.rank) {
              myflag = true;
              break;
            }
          }
        } else if (infoNeiTree.rank() < 0) {
          myflag = true;
          break;
        }
      }
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
      MPI_Irecv(&recv_size[kk], 1, MPI_INT, r, timestamp, MPI_COMM_WORLD,
                &size_requests[2 * kk]);
      MPI_Isend(&mysize, 1, MPI_INT, r, timestamp, MPI_COMM_WORLD,
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
                r, timestamp, MPI_COMM_WORLD, &requests[2 * kk]);
      MPI_Isend(send_buffer[kk].data(), send_buffer[kk].size(), MPI_LONG_LONG,
                r, timestamp, MPI_COMM_WORLD, &requests[2 * kk + 1]);
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
        Tree(level, Z).setrank(r);
        if (UpdateIDs)
          getBlockInfoAll(level, Z).blockID = recv_buffer[kk][index__ + 2];
        int p[2];
        BlockInfo::inverse(Z, level, p[0], p[1]);
        if (level < levelMax - 1)
          for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++) {
              const long long nc =
                  getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
              Tree(level + 1, nc).setCheckCoarser();
            }
        if (level > 0) {
          const long long nf = getZforward(level - 1, p[0] / 2, p[1] / 2);
          Tree(level - 1, nf).setCheckFiner();
        }
      }
    }
  }
  std::vector<int> FindMyNeighbors() {
    std::vector<int> myNeighbors;
    double low[3] = {+1e20, +1e20, +1e20};
    double high[3] = {-1e20, -1e20, -1e20};
    double p_low[3];
    double p_high[3];
    for (auto &info : m_vInfo) {
      const double h = 2 * info.h;
      info.pos(p_low, 0, 0);
      info.pos(p_high, _BS_ - 1, _BS_ - 1);
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
    }
    std::vector<double> all_boxes(sim.size * 6);
    double my_box[6] = {low[0], low[1], low[2], high[0], high[1], high[2]};
    MPI_Allgather(my_box, 6, MPI_DOUBLE, all_boxes.data(), 6, MPI_DOUBLE,
                  MPI_COMM_WORLD);
    for (int i = 0; i < sim.size; i++) {
      if (i == sim.rank)
        continue;
      if (Intersect(low, high, &all_boxes[i * 6], &all_boxes[i * 6 + 3]))
        myNeighbors.push_back(i);
    }
    return myNeighbors;
  }
  bool Intersect(double *l1, double *h1, double *l2, double *h2) {
    const double h0 =
        (maxextent / std::max(NX * _BS_, std::max(NY * _BS_, NZ * 1)));
    const double extent[3] = {NX * _BS_ * h0, NY * _BS_ * h0, NZ * 1 * h0};
    const Real intersect[3][2] = {
        {std::max(l1[0], l2[0]), std::min(h1[0], h2[0])},
        {std::max(l1[1], l2[1]), std::min(h1[1], h2[1])},
        {std::max(l1[2], l2[2]), std::min(h1[2], h2[2])}};
    bool intersection[3];
    intersection[0] = intersect[0][1] - intersect[0][0] > 0.0;
    intersection[1] = intersect[1][1] - intersect[1][0] > 0.0;
    intersection[2] =
        DIMENSION == 3 ? (intersect[2][1] - intersect[2][0] > 0.0) : true;
    const bool isperiodic[3] = {xperiodic, yperiodic, zperiodic};
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
  SynchronizerMPIType *sync(const StencilInfo &stencil) {
    assert(stencil.isvalid());
    StencilInfo Cstencil(-1, -1, DIMENSION == 3 ? -1 : 0, 2, 2,
                         DIMENSION == 3 ? 2 : 1, true, stencil.selcomponents);
    SynchronizerMPIType *queryresult = nullptr;
    typename std::map<StencilInfo, SynchronizerMPIType *>::iterator
        itSynchronizerMPI = SynchronizerMPIs.find(stencil);
    if (itSynchronizerMPI == SynchronizerMPIs.end()) {
      queryresult = new SynchronizerMPIType(stencil, Cstencil, this,
                                            sizeof(ElementType) / sizeof(Real));
      queryresult->_Setup();
      SynchronizerMPIs[stencil] = queryresult;
    } else {
      queryresult = itSynchronizerMPI->second;
    }
    queryresult->sync();
    timestamp = (timestamp + 1) % 32768;
    return queryresult;
  }
  void initialize_blocks0(const std::vector<long long> &blocksZ,
                          const std::vector<short int> &blockslevel) {
    for (size_t i = 0; i < m_vInfo.size(); i++) {
      const int m = m_vInfo[i].level;
      const long long n = m_vInfo[i].Z;
      delete[](Block *) getBlockInfoAll(m, n).ptrBlock;
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
    for (size_t i = 0; i < blocksZ.size(); i++) {
      const int level = blockslevel[i];
      const long long Z = blocksZ[i];
      _alloc(level, Z);
      Tree(level, Z).setrank(rank());
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
    }
    FillPos();
    UpdateFluxCorrection = true;
    UpdateGroups = true;
  }
  virtual void initialize_blocks(const std::vector<long long> &blocksZ,
                                 const std::vector<short int> &blockslevel) {
    initialize_blocks0(blocksZ, blockslevel);
    UpdateBlockInfoAll_States(false);
    for (auto it = SynchronizerMPIs.begin(); it != SynchronizerMPIs.end(); ++it)
      (*it->second)._Setup();
  }
  int rank() const { return sim.rank; }
  int get_world_size() const { return sim.size; }
  TreePosition &Tree(const int m, const long long n) {
    const long long aux = level_base[m] + n;
    const auto retval = Octree.find(aux);
    if (retval == Octree.end()) {
#pragma omp critical
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
  TreePosition &Tree(BlockInfo &info) { return Tree(info.level, info.Z); }
  TreePosition &Tree(const BlockInfo &info) { return Tree(info.level, info.Z); }
  void _alloc(const int m, const long long n) {
    BlockInfo &new_info = getBlockInfoAll(m, n);
    new_info.ptrBlock = new Block;
#pragma omp critical
    { m_vInfo.push_back(new_info); }
    Tree(m, n).setrank(rank());
  }
  void _dealloc(const int m, const long long n) {
    delete[](Block *) getBlockInfoAll(m, n).ptrBlock;
    for (size_t j = 0; j < m_vInfo.size(); j++) {
      if (m_vInfo[j].level == m && m_vInfo[j].Z == n) {
        m_vInfo.erase(m_vInfo.begin() + j);
        return;
      }
    }
  }
  void dealloc_many(const std::vector<long long> &dealloc_IDs) {
    for (size_t j = 0; j < m_vInfo.size(); j++)
      m_vInfo[j].changed2 = false;
    for (size_t i = 0; i < dealloc_IDs.size(); i++)
      for (size_t j = 0; j < m_vInfo.size(); j++) {
        if (m_vInfo[j].blockID_2 == dealloc_IDs[i]) {
          const int m = m_vInfo[j].level;
          const long long n = m_vInfo[j].Z;
          m_vInfo[j].changed2 = true;
          delete[](Block *) getBlockInfoAll(m, n).ptrBlock;
          break;
        }
      }
    m_vInfo.erase(std::remove_if(m_vInfo.begin(), m_vInfo.end(),
                                 [](const BlockInfo &x) { return x.changed2; }),
                  m_vInfo.end());
  }
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
  void FillPos(bool CopyInfos = true) {
    std::sort(m_vInfo.begin(), m_vInfo.end());
    Octree.reserve(Octree.size() + m_vInfo.size() / 8);
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
  long long getZforward(const int level, const int i, const int j) const {
    const int TwoPower = 1 << level;
    const int ix = (i + TwoPower * NX) % (NX * TwoPower);
    const int iy = (j + TwoPower * NY) % (NY * TwoPower);
    return BlockInfo::forward(level, ix, iy);
  }
  Block *avail1(const int ix, const int iy, const int m) {
    const long long n = getZforward(m, ix, iy);
    return avail(m, n);
  }
  Block &operator()(const long long ID) {
    return *(Block *)m_vInfo[ID].ptrBlock;
  }
  std::array<int, 3> getMaxBlocks() const { return {NX, NY, NZ}; }
  std::array<int, 3> getMaxMostRefinedBlocks() const {
    return {
        NX << (levelMax - 1),
        NY << (levelMax - 1),
        1,
    };
  }
  std::array<int, 3> getMaxMostRefinedCells() const {
    const auto b = getMaxMostRefinedBlocks();
    return {b[0] * _BS_, b[1] * _BS_, b[2] * 1};
  }
  int getlevelMax() const { return levelMax; }
  BlockInfo &getBlockInfoAll(const int m, const long long n) {
    const long long aux = level_base[m] + n;
    const auto retval = BlockInfoAll.find(aux);
    if (retval != BlockInfoAll.end()) {
      return *retval->second;
    } else {
#pragma omp critical
      {
        const auto retval1 = BlockInfoAll.find(aux);
        if (retval1 == BlockInfoAll.end()) {
          BlockInfo *dumm = new BlockInfo();
          const int TwoPower = 1 << m;
          const double h0 =
              (maxextent / std::max(NX * _BS_, std::max(NY * _BS_, NZ * 1)));
          const double h = h0 / TwoPower;
          double origin[3];
          int i, j, k;
          BlockInfo::inverse(n, m, i, j);
          k = 0;
          origin[0] = i * _BS_ * h;
          origin[1] = j * _BS_ * h;
          origin[2] = k * 1 * h;
          dumm->setup(m, h, origin, n);
          BlockInfoAll[aux] = dumm;
        }
      }
      return getBlockInfoAll(m, n);
    }
  }
  /*
  virtual Block *avail(const int m, const long long n) {
    return (Block *)getBlockInfoAll(m, n).ptrBlock;
    } */
  /* virtual int rank() const { return 0; } */
};
