#pragma once

using namespace std;
using namespace std::placeholders;

template <class stateT, class actionT, class stateHasher>
MA3<stateT, actionT, stateHasher>::MA3(
           std::function<double(const stateT &state, const stateT &goal)> GetH,
           std::function<PathType<stateT, actionT>(const stateT &state, bool successor,
               const std::shared_ptr<std::unordered_map<was::ActionState<actionT, stateT> , double, ActionStateHasher> >)> GetNextStates,
           std::function<bool(const stateT &state, const stateT &goal)> SatisfiesGoal,
           std::function<double(const was::ActionState<actionT, stateT> &edge)> QuerySimModel,
           std::function<double(const was::ActionState<actionT, stateT> &edge)> QueryOptimisticModel,
           std::function<std::tuple<bool,bool>(const was::ActionState<actionT, stateT> &edge)> QueryMLModel,
           double suboptimality_gap, bool anytime_mode, bool use_pessimistic, int sim_wait_time){
    W_current_ = make_shared<unordered_map<was::ActionState<actionT, stateT> , double, ActionStateHasher> >();
    GetH_ = GetH;
    GetNextStates_ = GetNextStates;
    SatisfiesGoal_ = SatisfiesGoal;
    QuerySimModel_ = QuerySimModel;
    QueryMLModel_ = QueryMLModel;
    QueryOptimisticModel_ = QueryOptimisticModel;
    run_sim_thread_ = true;
    min_cost_valid_path_ = INF;
    path_cost_lowerbound_ = -INF;
    anytime_mode_ = anytime_mode;
    sim_busy_ = false;
    suboptimality_gap_ = suboptimality_gap;
    sim_runs_ = 0;
    sim_wait_time_ = sim_wait_time;
    sog_used_ = false;
    use_pessimistic_ = use_pessimistic;
}

template <class stateT, class actionT, class stateHasher>
MA3<stateT, actionT, stateHasher>::~MA3(){
    run_sim_thread_ = false;
}

template <class stateT, class actionT, class stateHasher>
double MA3<stateT, actionT, stateHasher>::Plan(const stateT &s_start, const stateT &s_goal){
    P_sim_results_.clear();
    const std::chrono::milliseconds timeout( 3000 );
    std::unique_ptr<std::thread> sim_worker;
    sim_worker = std::make_unique<std::thread>(std::thread(&MA3<stateT, actionT, stateHasher>::SimulationWorker, this));

    sp_planner_ = std::make_unique<was::WAstar<stateT, actionT, stateHasher> >(was::WAstar<stateT, actionT, stateHasher> (1.0,
                                                                                std::bind(GetH_, _1, _2),
                                                                                std::bind(GetNextStates_, _1, true, W_current_),
                                                                                std::bind(SatisfiesGoal_, _1, _2) ));
    P_sim_ = std::priority_queue<p_path<PathType<stateT, actionT> > >();
    P_valid_ = std::priority_queue<p_path<PathType<stateT, actionT> > >();
    P_cand_ = std::priority_queue<p_path<PathType<stateT, actionT> > >();
    O_e_ = std::vector<was::ActionState<actionT, stateT> >();
    is_W_new_ = false;

    auto [ path_cost, path_v ] = ShortestPath(s_start, s_goal);
    path_cost_lowerbound_ = path_cost;
    SharedPathType<stateT, actionT> path = std::make_shared<PathType<stateT, actionT> >(path_v);
    
    if(path_cost != INF)
        P_cand_.push(p_path<PathType<stateT, actionT> >{path_cost, path});
    while(P_cand_.size()>0 or P_sim_.size()>0 or sim_busy_){
        Log("Main Thread", "Loop Start");

        if(P_cand_.size()>0){
            p_path<PathType<stateT, actionT> > pcand = P_cand_.top();
            P_cand_.pop();
            size_t selected = SelectNextEdge(pcand.path_shptr.get(), 0);
            if (selected == pcand.path_shptr.get()->size()){
                // for(auto& edge: (*pcand.path_shptr)){
                //     auto [qmodel, qcost] = GetWModel(edge);
                //     edge.model = qmodel;
                // }
                // LogPath(*pcand.path_shptr, "MA3 for validation");
                // std::cout << "A path of cost " << pcand.p << " was sent for verification\n";
                {
                    std::unique_lock<std::mutex> lck(mtx_sim_);
                    P_sim_.push(p_path<PathType<stateT, actionT> >{pcand.p/path_cost_multiplier_, pcand.path_shptr});
                }
                cv_sim_.notify_all();
            }
            else{
                std::vector<SharedPathType<stateT, actionT> > edges_to_push;
                while(selected != pcand.path_shptr.get()->size()){
                    double ce = pcand.path_shptr.get()->at(selected).cost;
                    auto [valid, confident] = QueryMLModel_(pcand.path_shptr.get()->at(selected));
                    if(!valid) ce = INF;
                    if(confident){
                        Overwrite(pcand.path_shptr.get()->at(selected),ce,was::MODEL_ML);
                        E_conf_.insert(pcand.path_shptr.get()->at(selected));
                        if(ce == INF){
                            auto edge = pcand.path_shptr.get()->at(selected);
                            edge.cost = pcand.p;
                            if(use_pessimistic_) O_e_.push_back(edge);
                            break;
                        }
                    }
                    else{
                        /*** TODO: Optimize this ***/ 
                        PathType<stateT, actionT> edges_v; edges_v.push_back(pcand.path_shptr.get()->at(selected));
                        SharedPathType<stateT, actionT> edges = std::make_shared<PathType<stateT, actionT> >(edges_v);
                        edges_to_push.push_back(edges);

                        Overwrite(pcand.path_shptr.get()->at(selected),INF,was::MODEL_DIS);
                        break;
                    }
                    selected = SelectNextEdge(pcand.path_shptr.get(), selected+1);
                }
                if(edges_to_push.size()>0){
                    {
                        std::unique_lock<std::mutex> lck(mtx_sim_);
                        for(const auto& edges: edges_to_push){
                            P_sim_.push(p_path<PathType<stateT, actionT> >{2.1 + pcand.p/path_cost_multiplier_, edges});
                        }
                    }
                    cv_sim_.notify_all();
                }
            }
        }
        else if(anytime_mode_){
            Log("Main Thread", "Waiting One Data Start");
            std::unique_lock<std::mutex> lck(mtx_sim_results_);
            cv_sim_results_.wait(lck,[this]{return IsSimDataAvailable();});
            Log("Main Thread", "Waiting One Data End");
        }

        if (!anytime_mode_){ /* While this involved a lot of back and forth, but it is easier to guarantee */
            Log("Main Thread", "Waiting Full Data Start");
            std::unique_lock<std::mutex> lck(mtx_sim_results_);
            cv_sim_results_.wait(lck,[this]{return IsFullSimDataAvailable();});
            Log("Main Thread", "Waiting Full Data End");
        }

        GetSimDataThreaded();

        if(is_W_new_){
            is_W_new_ = false;
            auto [ _path_cost, _path_v ] = ShortestPath(s_start, s_goal);
            SharedPathType<stateT, actionT> _path = std::make_shared<PathType<stateT, actionT> >(_path_v);
            
            if(_path_cost != INF){
                P_cand_.push(p_path<PathType<stateT, actionT> >{_path_cost, _path});
            }
        }

        bool is_gap_satisfied = CheckGap();
        Log("Main Thread", "Loop End");
        if(is_gap_satisfied && anytime_mode_){
            break;
        }
    }

    run_sim_thread_ = false;
    cv_sim_.notify_all();
    sim_worker->join();
    std::cout << "SOG USED: " << sog_used_ << std::endl;
    if(P_valid_.empty()) return INF;
    else{
        // for(auto& edge: *(P_valid_.top().path_shptr)){
        //     auto [qmodel, qcost] = GetWModel(edge);
        //     edge.model = qmodel;
        // }
        // LogPath(*(P_valid_.top().path_shptr), "MA3 validated");
        std::cout << "ASCP lb=" << path_cost_lowerbound_ << " ub=" << min_cost_valid_path_ << std::endl;
        return P_valid_.top().p;
    }
}

template <class stateT, class actionT, class stateHasher>
bool MA3<stateT, actionT, stateHasher>::IsInEconf(const was::ActionState<actionT, stateT> &edge){
    if (E_conf_.find(edge) != E_conf_.end()) return true;
    return false;
}

template <class stateT, class actionT, class stateHasher>
size_t MA3<stateT, actionT, stateHasher>::SelectNextEdge(const PathType<stateT, actionT>* path, size_t start_id){
    for(size_t i=start_id; i<path->size(); i++) if(!IsInEconf(path->at(i))) return i;
    return path->size();
}

template <class stateT, class actionT, class stateHasher>
std::tuple<double, 
           PathType<stateT, actionT> > MA3<stateT, actionT, stateHasher>::ShortestPath(const stateT &s_start, const stateT &s_goal){
    path_id_++;
    Log("Main Thread", "AStar Start");
    sp_planner_.get()->Reset();
    double path_cost = sp_planner_.get()->Plan(s_start, s_goal);
    auto path = sp_planner_.get()->BackTrackPath(s_goal, path_id_);
    Log("Main Thread", "AStar End");
    //LB_[path_id_] = path_cost;
    return std::make_tuple(path_cost, path);
}

template <class stateT, class actionT, class stateHasher>
inline double MA3<stateT, actionT, stateHasher>::GetW(const was::ActionState<actionT, stateT> &edge, double default_value){
    if(!W_current_) return default_value;
    auto it = W_current_.get()->find(edge);
    if (it != W_current_.get()->end()) return it->second;
    else return default_value;
}

template <class stateT, class actionT, class stateHasher>
inline void MA3<stateT, actionT, stateHasher>::Overwrite(const was::ActionState<actionT, stateT> &edge, double value, unsigned short model){
    auto it = W_current_.get()->find(edge);
    if (it != W_current_.get()->end()){ /* If edge exists in W_current */
        if (it->second != value){ /* If the value is not new */
            it->second = value;
            is_W_new_ = true;
        }
        if(it->first.model != model){ /* Regardless, update the model source */
            auto new_edge = it->first;
            new_edge.model = model;
            W_current_.get()->erase(it);
            (*W_current_)[new_edge] = it->second;
        }
    }
    else{ /* Add edge to W_current */
        auto new_edge = edge;
        new_edge.model = model;
        (*W_current_)[new_edge] = value;
        is_W_new_ = true;
    }
}

template <class stateT, class actionT, class stateHasher>
inline void MA3<stateT, actionT, stateHasher>::OverwriteModel(const was::ActionState<actionT, stateT> &edge, unsigned short model){
    auto it = W_current_.get()->find(edge);
    if (it != W_current_.get()->end()){ /* If edge exists in W_current */
        if(it->first.model != model){ /* Regardless, update the model source */
            auto new_edge = it->first;
            new_edge.model = model;
            W_current_.get()->erase(it);
            (*W_current_)[new_edge] = it->second;
        }
    }
}

template <class stateT, class actionT, class stateHasher>
std::pair<unsigned short, double> MA3<stateT, actionT, stateHasher>::GetWModel(const was::ActionState<actionT, stateT> &edge){
    auto it = W_current_.get()->find(edge);
    if (it != W_current_.get()->end()){
        return std::make_pair(it->first.model, it->second);
    }
    return std::make_pair(was::MODEL_INIT, QueryOptimisticModel_(edge));
}

template <class stateT, class actionT, class stateHasher>
void MA3<stateT, actionT, stateHasher>::GetSimDataThreaded(){
    {
        std::unique_lock<std::mutex> lck(mtx_sim_results_);
        P_sim_results_swap_.clear();
        P_sim_results_.swap(P_sim_results_swap_);
        
    }
    for(const auto& path: P_sim_results_swap_){
        if(path.p < 2.0){
            bool path_validated = true;
            for(const auto& edge: (*path.path_shptr)){
                if(edge.cost == INF){
                    path_validated = false;
                    //RemoveLB_(edge.path_id);
                }
                Overwrite(edge,edge.cost,was::MODEL_SIM);
                E_conf_.insert(edge);
            }
            if(path_validated){
                P_valid_.push(p_path<PathType<stateT, actionT> >{path.p*path_cost_multiplier_, path.path_shptr});
                min_cost_valid_path_ = min(min_cost_valid_path_, path.p*path_cost_multiplier_);
            }
        }
        else{
            for(const auto& edge: (*path.path_shptr)){
                //if(edge.cost == INF) RemoveLB_(edge.path_id);
                Overwrite(edge,edge.cost,was::MODEL_SIM);
                E_conf_.insert(edge);
            }
        }
    }
}

template <class stateT, class actionT, class stateHasher>
void MA3<stateT, actionT, stateHasher>::SimulationWorker(){
    while(run_sim_thread_){
        Log("Sim Thread", "Loop Start");
        p_path<PathType<stateT, actionT> > path;
        {
            Log("Sim Thread", "Wait Start");
            std::unique_lock<std::mutex> lck(mtx_sim_);
            cv_sim_.wait(lck,[this]{return IsSimRequired();});
            Log("Sim Thread", "Wait Stop" + std::to_string(P_sim_.size()));
            if(!run_sim_thread_) break;
            sim_busy_ = true;
            path = P_sim_.top();
            P_sim_.pop();
            if(path.p < 2.0 && path.p*path_cost_multiplier_ > min_cost_valid_path_){
                /*** TODO: Verify that this filtering is correct.
                 * You can potentially also filter edges with the min_cost_valid_path_ upper-bound ***/
                continue;
            }
        }
        for(auto& edge: (*path.path_shptr)){
            auto [qmodel, qcost] = GetWModel(edge);
            if(qmodel == was::MODEL_SIM){
                edge.cost = qcost;
                continue;
            }
            if(!run_sim_thread_) break;
            edge.cost = QuerySimModel_(edge);
            sim_runs_++;
            if(sim_runs_%50 == 0) std::cout << sim_runs_ << " : " << O_e_.size() << " sim runs\n";
            if(anytime_mode_) std::this_thread::sleep_for(std::chrono::milliseconds(sim_wait_time_));
        }
        {
            std::unique_lock<std::mutex> lck(mtx_sim_results_);
            P_sim_results_.push_back(path);
        }
        sim_busy_ = false;
        cv_sim_results_.notify_all();
        Log("Sim Thread", "Loop End");
    }
}

template <class stateT, class actionT, class stateHasher>
bool MA3<stateT, actionT, stateHasher>::CheckGap(){
    auto lb = path_cost_lowerbound_;
    // if(!O_e_.empty()) lb = std::max(lb, O_e_.top().cost);

    auto cp = INF;
    if(!P_valid_.empty()) cp = std::min(cp, P_valid_.top().p);

    if(cp/lb > suboptimality_gap_ && !O_e_.empty() && use_pessimistic_){
        if(cp!=INF || (P_cand_.empty() && P_sim_.empty() && !sim_busy_)){
            std::make_heap(O_e_.begin(), O_e_.end());
            while(!O_e_.empty() && (O_e_[0].cost < lb)){ std::pop_heap(O_e_.begin(), O_e_.end()); O_e_.pop_back(); }
            if(!O_e_.empty()){
                auto edge = O_e_[0];
                std::pop_heap(O_e_.begin(), O_e_.end()); O_e_.pop_back();
                OverwriteModel(edge, was::MODEL_SOG);
                sog_used_ = true;
                PathType<stateT, actionT> edges_v; edges_v.push_back(edge);
                SharedPathType<stateT, actionT> edges = std::make_shared<PathType<stateT, actionT> >(edges_v);
                
                {
                    std::unique_lock<std::mutex> lck(mtx_sim_);
                    P_sim_.push(p_path<PathType<stateT, actionT> >{2.1 + edge.cost/path_cost_multiplier_, edges});
                }
                cv_sim_.notify_all();
            }
        }
    }
    if(cp/lb <= suboptimality_gap_ && !P_valid_.empty()) {
        return true;
    }
    else return false;
}

template <class stateT, class actionT, class stateHasher>
void MA3<stateT, actionT, stateHasher>::Reset(){
    // Things reset in Plan:  sp_planner_, P_sim_, P_cand_, P_valid_, O_e_, is_W_new_
    E_conf_.clear(); 
    W_current_ = make_shared<unordered_map<was::ActionState<actionT, stateT> , double, ActionStateHasher> >();
    P_sim_results_swap_.clear();
    run_sim_thread_ = true; sim_busy_ = false; sog_used_ = false; 
    min_cost_valid_path_ = INF;
    path_cost_lowerbound_ = -INF;
    sim_runs_ = 0;
    path_id_ = 0;
    LB_.clear();
}

template <class stateT, class actionT, class stateHasher>
void MA3<stateT, actionT, stateHasher>::RemoveLB_(size_t path_id){
    auto it = LB_.find(path_id);
    if(it != LB_.end()){
        LB_.erase(it);
    }

    auto new_lowerbound_it = min_element(LB_.begin(), LB_.end(),
            [](const auto& l, const auto& r) { return l.second < r.second; });
    
    double new_lowerbound = new_lowerbound_it->second;

    if(path_cost_lowerbound_ != new_lowerbound){
        std::cout << "Lower Bound updated to: " << new_lowerbound << " from: " << path_cost_lowerbound_ << std::endl;
        path_cost_lowerbound_ = new_lowerbound;
    }
}

template <class stateT, class actionT, class stateHasher>
PathType<stateT, actionT> MA3<stateT, actionT, stateHasher>::GetPlannedPath(){
    return *(P_valid_.top().path_shptr);
}