class SpRowInfo {
public:
  const int rank_;
  const long long idx_;
  std::map<long long, double> loc_colval_;
  std::map<long long, double> bd_colval_;
  std::vector<std::pair<int, long long>> neirank_cols_;
  SpRowInfo(const int rank, const long long row_idx, const int neirank_max)
      : rank_(rank), idx_(row_idx) {
    neirank_cols_.reserve(neirank_max);
  }
  ~SpRowInfo() = default;
  void mapColVal(const long long col_idx, const double val) {
    loc_colval_[col_idx] += val;
  }
  void mapColVal(const int rank, const long long col_idx, const double val) {
    if (rank == rank_)
      mapColVal(col_idx, val);
    else {
      bd_colval_[col_idx] += val;
      neirank_cols_.push_back({rank, col_idx});
    }
  }
};
class BiCGSTABSolver;
struct LocalSpMatDnVec {
  LocalSpMatDnVec(const int BLEN, const bool bMeanConstraint,
                  const std::vector<double> &P_inv);
  ~LocalSpMatDnVec();
  void reserve(const int N);
  void cooPushBackVal(const double val, const long long row,
                      const long long col);
  void cooPushBackRow(const SpRowInfo &row);
  void make(const std::vector<long long> &Nrows_xcumsum);
  void solveWithUpdate(const double max_error, const double max_rel_error,
                       const int max_restarts);
  void solveNoUpdate(const double max_error, const double max_rel_error,
                     const int max_restarts);
  friend class BiCGSTABSolver;
  int rank_;
  int comm_size_;
  const int BLEN_;
  int m_;
  int halo_;
  int loc_nnz_;
  int bd_nnz_;
  int bMeanRow_;
  std::vector<double> loc_cooValA_;
  std::vector<long long> loc_cooRowA_long_;
  std::vector<long long> loc_cooColA_long_;
  std::vector<double> x_;
  std::vector<double> b_;
  std::vector<double> h2_;
  std::vector<double> bd_cooValA_;
  std::vector<long long> bd_cooRowA_long_;
  std::vector<long long> bd_cooColA_long_;
  std::vector<int> loc_cooRowA_int_;
  std::vector<int> loc_cooColA_int_;
  std::vector<int> bd_cooRowA_int_;
  std::vector<int> bd_cooColA_int_;
  std::vector<std::set<long long>> bd_recv_set_;
  std::vector<std::vector<long long>> bd_recv_vec_;
  std::vector<int> recv_ranks_;
  std::vector<int> recv_offset_;
  std::vector<int> recv_sz_;
  std::vector<int> send_ranks_;
  std::vector<int> send_offset_;
  std::vector<int> send_sz_;
  std::vector<int> send_pack_idx_;
  std::unique_ptr<BiCGSTABSolver> solver_;
};
