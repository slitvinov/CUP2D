template <typename ElementType> struct Grid {
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
  FluxCorrection<Grid, ElementType> CorrectorGrid;
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
  virtual Block *avail(const int m, const long long n) {
    return (Block *)getBlockInfoAll(m, n).ptrBlock;
  }
  virtual int rank() const { return 0; }
  virtual void initialize_blocks(const std::vector<long long> &blocksZ,
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
};
