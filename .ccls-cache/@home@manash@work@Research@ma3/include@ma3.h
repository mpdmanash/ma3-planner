#ifndef MA3_H
#define MA3_H

#include <algorithm>
#include <functional> // for hash
#include <vector> // for successors
#include <unordered_map> // for G values
#include <unordered_set> // for E_conf
#include <map>
#include <memory>
#include <tuple>
#include <queue> // for priority_queue
#include <thread>             // std::thread, std::this_thread::yield
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable
#include "lib_wastar.h"
#include "utils.h"

namespace ma3{
template <typename T>
struct p_path{
    double p; std::shared_ptr<T> path_shptr;
    bool operator<(const p_path<T>& rhs) const
    {return (-p < -rhs.p);}
};

template <class stateT, class actionT>
using PathType = std::vector<was::ActionState<actionT, stateT> >;

template <class stateT, class actionT>
using SharedPathType = std::shared_ptr<PathType<stateT, actionT> >;

typedef std::chrono::_V2::system_clock::time_point time_ds;

template <class stateT, class actionT, class stateHasher = std::hash<stateT> >
class MA3{
public:
    MA3(std::function<double(const stateT &state, const stateT &goal)> GetH,
           std::function<PathType<stateT, actionT>(const stateT &state, bool successor,
                const std::shared_ptr<std::unordered_map<was::ActionState<actionT, stateT> , double, ActionStateHasher> >)> GetNextStates,
           std::function<bool(const stateT &state, const stateT &goal)> SatisfiesGoal,
           std::function<double(const was::ActionState<actionT, stateT> &edge)> QuerySimModel,
           std::function<double(const was::ActionState<actionT, stateT> &edge)> QueryOptimisticModel,
           std::function<std::tuple<bool,bool>(const was::ActionState<actionT, stateT> &edge)> QueryMLModel,
           double suboptimality_gap, bool anytime_mode, bool use_pessimistic, int sim_wait_time);
    ~MA3();
    double Plan(const stateT &s_start, const stateT &s_goal);
    PathType<stateT, actionT> GetPlannedPath();
    void Reset();
private:
    std::unordered_set<was::ActionState<actionT, stateT> , ActionStateHasher> E_conf_;
    std::shared_ptr<std::unordered_map<was::ActionState<actionT, stateT> , double, ActionStateHasher> > W_current_;
    std::function<double(const stateT &state, const stateT &goal)> GetH_;
    std::function<PathType<stateT, actionT>(const stateT &state, bool successor,
                const std::shared_ptr<std::unordered_map<was::ActionState<actionT, stateT> , double, ActionStateHasher> >)> GetNextStates_;
    std::function<bool(const stateT &state, const stateT &goal)> SatisfiesGoal_;
    std::function<double(const was::ActionState<actionT, stateT> &edge)> QuerySimModel_;
    std::function<double(const was::ActionState<actionT, stateT> &edge)> QueryOptimisticModel_;
    std::function<std::tuple<bool,bool>(const was::ActionState<actionT, stateT> &edge)> QueryMLModel_;
    std::unique_ptr<was::WAstar<stateT, actionT, stateHasher> > sp_planner_;
    std::priority_queue<p_path<PathType<stateT, actionT> > > P_sim_, P_cand_, P_valid_;
    std::vector<was::ActionState<actionT, stateT> > O_e_;
    std::vector<p_path<PathType<stateT, actionT> > > P_sim_results_, P_sim_results_swap_;
    std::unordered_map<size_t, double> LB_;
    bool is_W_new_, run_sim_thread_, anytime_mode_, sim_busy_, sog_used_, use_pessimistic_;
    std::mutex mtx_sim_, mtx_sim_results_;
    std::condition_variable cv_sim_, cv_sim_results_;
    double min_cost_valid_path_, path_cost_lowerbound_, suboptimality_gap_;
    time_ds planning_started_at_;
    int sim_runs_, sim_wait_time_;
    size_t path_id_ = 0;
    double path_cost_multiplier_ = 10000;

    size_t SelectNextEdge(const PathType<stateT, actionT>* path, size_t start_id=0);
    bool IsInEconf(const was::ActionState<actionT, stateT> &edge);
    std::tuple<double, PathType<stateT, actionT> > ShortestPath(const stateT &s_start, const stateT &s_goal);
    double GetW(const was::ActionState<actionT, stateT> &edge, double default_value=INF);
    inline void Overwrite(const was::ActionState<actionT, stateT> &edge, double value, unsigned short model);
    inline void OverwriteModel(const was::ActionState<actionT, stateT> &edge, unsigned short model);
    std::pair<unsigned short, double> GetWModel(const was::ActionState<actionT, stateT> &edge);
    void GetSimDataThreaded();
    void SimulationWorker();
    bool CheckGap();
    bool IsSimRequired(){return !P_sim_.empty() || !run_sim_thread_;}
    bool IsSimDataAvailable(){return !P_sim_results_.empty();}
    bool IsFullSimDataAvailable(){return P_sim_.empty() && !sim_busy_;}
    void RemoveLB_(size_t path_id);
};

#include "lazyascp.tpp"
}

#endif