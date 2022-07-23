#pragma once

using namespace std;
using namespace std::placeholders;

template <class stateT, class actionT, class stateHasher>
LazyASCP<stateT, actionT, stateHasher>::LazyASCP(
           std::function<double(const stateT &state, const stateT &goal)> GetH,
           std::function<PathType<stateT, actionT>(const stateT &state, bool successor,
               const std::shared_ptr<std::unordered_map<was::ActionState<actionT, stateT> , double, ActionStateHasher> >)> GetNextStates,
           std::function<bool(const stateT &state, const stateT &goal)> SatisfiesGoal,
           std::function<double(const was::ActionState<actionT, stateT> &edge)> QuerySimModel,
           std::function<double(const was::ActionState<actionT, stateT> &edge)> QueryOptimisticModel,
           std::function<std::tuple<bool,bool>(const was::ActionState<actionT, stateT> &edge)> QueryMLModel,
           double suboptimality_gap, bool anytime_mode, int sim_wait_time){
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
}

template <class stateT, class actionT, class stateHasher>
LazyASCP<stateT, actionT, stateHasher>::~LazyASCP(){
    run_sim_thread_ = false;
}

template <class stateT, class actionT, class stateHasher>
double LazyASCP<stateT, actionT, stateHasher>::Plan(const stateT &s_start, const stateT &s_goal){
    P_sim_ = std::priority_queue<p_path<PathType<stateT, actionT> > >();
    P_sim_results_.clear();
    const std::chrono::milliseconds timeout( 3000 );
    std::unique_ptr<std::thread> sim_worker;
    sim_worker = std::make_unique<std::thread>(std::thread(&LazyASCP<stateT, actionT, stateHasher>::SimulationWorker, this));

    sp_planner_ = std::make_unique<was::WAstar<stateT, actionT, stateHasher> >(was::WAstar<stateT, actionT, stateHasher> (1.0,
                                                                                std::bind(GetH_, _1, _2),
                                                                                std::bind(GetNextStates_, _1, true, W_current_),
                                                                                std::bind(SatisfiesGoal_, _1, _2) ));
    P_sim_ = std::priority_queue<p_path<PathType<stateT, actionT> > >();
    P_valid_ = std::priority_queue<p_path<PathType<stateT, actionT> > >();
    P_cand_ = std::priority_queue<p_path<PathType<stateT, actionT> > >();
    O_e_ = std::priority_queue<was::ActionState<actionT, stateT> >();
    is_W_new_ = false;

    auto [ path_cost, path_v ] = ShortestPath(s_start, s_goal);
    path_cost_lowerbound_ = path_cost;
    SharedPathType<stateT, actionT> path = std::make_shared<PathType<stateT, actionT> >(path_v);
    
    if(path_cost != INF)
        P_cand_.push(p_path<PathType<stateT, actionT> >{path_cost, path});
    sent_v_ = false; failed_v_ = false;
    while(P_cand_.size()>0 or P_sim_.size()>0 or sim_busy_){
        // cout  << P_cand_.size() << " " << P_sim_.size() << " ====== sizes\n";

        if(P_cand_.size()>0){
            p_path<PathType<stateT, actionT> > pcand = P_cand_.top();
            P_cand_.pop();
            size_t selected = SelectNextEdge(pcand.path_shptr.get(), 0);
            if (selected == pcand.path_shptr.get()->size()){
                std::unique_lock<std::mutex> lck(mtx_sim_);
                for(auto& edge: (*pcand.path_shptr)){
                    auto [qmodel, qcost] = GetWModel(edge);
                    edge.model = qmodel;
                }
                sent_v_ = true;
                // LogPath(*pcand.path_shptr, "LazyASCP for validation");
                // std::cout << "A path of cost " << pcand.p << " was sent for verification\n";
                P_sim_.push(p_path<PathType<stateT, actionT> >{pcand.p/path_cost_multiplier_, pcand.path_shptr});
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
                    std::unique_lock<std::mutex> lck(mtx_sim_);
                    for(const auto& edges: edges_to_push){
                        P_sim_.push(p_path<PathType<stateT, actionT> >{2.1 + pcand.p/path_cost_multiplier_, edges});
                    }
                    cv_sim_.notify_all();
                }
            }
        }
        else{
            std::unique_lock<std::mutex> lck(mtx_sim_results_);
            cv_sim_results_.wait_for(lck,timeout,[this]{return IsSimDataAvailable();});
        }

        if (!anytime_mode_){ /* While this involved a lot of back and forth, but it is easier to guarantee */
            {
                std::unique_lock<std::mutex> lck(mtx_sim_results_);
                cv_sim_results_.wait_for(lck,timeout,[this]{return IsFullSimDataAvailable();});
            }
        }

        GetSimDataThreaded();

        if(sent_v_ && failed_v_) break; // Only for baseline

        if(is_W_new_){
            is_W_new_ = false;
            auto [ _path_cost, _path_v ] = ShortestPath(s_start, s_goal);
            SharedPathType<stateT, actionT> _path = std::make_shared<PathType<stateT, actionT> >(_path_v);
            
            if(_path_cost != INF){
                P_cand_.push(p_path<PathType<stateT, actionT> >{_path_cost, _path});
            }
        }

        bool is_gap_satisfied = CheckGap();
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
        // LogPath(*(P_valid_.top().path_shptr), "LazyASCP validated");
        std::cout << "ASCP lb=" << path_cost_lowerbound_ << " ub=" << min_cost_valid_path_ << std::endl;
        return P_valid_.top().p;
    }
}

template <class stateT, class actionT, class stateHasher>
bool LazyASCP<stateT, actionT, stateHasher>::IsInEconf(const was::ActionState<actionT, stateT> &edge){
    if (E_conf_.find(edge) != E_conf_.end()) return true;
    return false;
}

template <class stateT, class actionT, class stateHasher>
size_t LazyASCP<stateT, actionT, stateHasher>::SelectNextEdge(const PathType<stateT, actionT>* path, size_t start_id){
    for(size_t i=start_id; i<path->size(); i++) if(!IsInEconf(path->at(i))) return i;
    return path->size();
}

template <class stateT, class actionT, class stateHasher>
std::tuple<double, 
           PathType<stateT, actionT> > LazyASCP<stateT, actionT, stateHasher>::ShortestPath(const stateT &s_start, const stateT &s_goal){
    sp_planner_.get()->Reset();
    double path_cost = sp_planner_.get()->Plan(s_start, s_goal);
    auto path = sp_planner_.get()->BackTrackPath(s_goal);
    return std::make_tuple(path_cost, path);
}

template <class stateT, class actionT, class stateHasher>
inline double LazyASCP<stateT, actionT, stateHasher>::GetW(const was::ActionState<actionT, stateT> &edge, double default_value){
    if(!W_current_) return default_value;
    auto it = W_current_.get()->find(edge);
    if (it != W_current_.get()->end()) return it->second;
    else return default_value;
}

template <class stateT, class actionT, class stateHasher>
inline void LazyASCP<stateT, actionT, stateHasher>::Overwrite(const was::ActionState<actionT, stateT> &edge, double value, unsigned short model){
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
inline void LazyASCP<stateT, actionT, stateHasher>::OverwriteModel(const was::ActionState<actionT, stateT> &edge, unsigned short model){
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
std::pair<unsigned short, double> LazyASCP<stateT, actionT, stateHasher>::GetWModel(const was::ActionState<actionT, stateT> &edge){
    auto it = W_current_.get()->find(edge);
    if (it != W_current_.get()->end()){
        return std::make_pair(it->first.model, it->second);
    }
    return std::make_pair(was::MODEL_INIT, QueryOptimisticModel_(edge));
}

template <class stateT, class actionT, class stateHasher>
void LazyASCP<stateT, actionT, stateHasher>::GetSimDataThreaded(){
    {
        std::unique_lock<std::mutex> lck(mtx_sim_results_);
        P_sim_results_swap_.clear();
        P_sim_results_.swap(P_sim_results_swap_);
        
    }
    for(const auto& path: P_sim_results_swap_){
        if(path.p < 2.0){
            bool path_validated = true;
            for(const auto& edge: (*path.path_shptr)){
                if(edge.cost == INF) path_validated = false;
                Overwrite(edge,edge.cost,was::MODEL_SIM);
                E_conf_.insert(edge);
            }
            if(path_validated){
                P_valid_.push(p_path<PathType<stateT, actionT> >{path.p*path_cost_multiplier_, path.path_shptr});
                min_cost_valid_path_ = min(min_cost_valid_path_, path.p*path_cost_multiplier_);
                failed_v_ = false;
            }
            else{failed_v_ = true;}
        }
        else{
            for(const auto& edge: (*path.path_shptr)){
                Overwrite(edge,edge.cost,was::MODEL_SIM);
                E_conf_.insert(edge);
            }
        }
    }
}

template <class stateT, class actionT, class stateHasher>
void LazyASCP<stateT, actionT, stateHasher>::SimulationWorker(){
    while(run_sim_thread_){
        p_path<PathType<stateT, actionT> > path;
        {
            std::unique_lock<std::mutex> lck(mtx_sim_);
            cv_sim_.wait(lck,[this]{return IsSimRequired();});
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

        std::unique_lock<std::mutex> lck(mtx_sim_results_);
        P_sim_results_.push_back(path);
        sim_busy_ = false;
        cv_sim_results_.notify_all();
    }
}

template <class stateT, class actionT, class stateHasher>
bool LazyASCP<stateT, actionT, stateHasher>::CheckGap(){
    auto lb = path_cost_lowerbound_;
    if(!O_e_.empty()) lb = std::max(lb, O_e_.top().cost);

    auto cp = INF;
    if(!P_valid_.empty()) cp = std::min(cp, P_valid_.top().p);

    if(cp/lb > suboptimality_gap_ && !O_e_.empty()){
        if(cp!=INF || (P_cand_.empty() && P_sim_.empty() && !sim_busy_)){
            while(!O_e_.empty() && (O_e_.top().cost < lb)) O_e_.pop();
            if(!O_e_.empty()){
                auto edge = O_e_.top();
                O_e_.pop();
                OverwriteModel(edge, was::MODEL_SOG);
                sog_used_ = true;
                PathType<stateT, actionT> edges_v; edges_v.push_back(edge);
                SharedPathType<stateT, actionT> edges = std::make_shared<PathType<stateT, actionT> >(edges_v);
                
                std::unique_lock<std::mutex> lck(mtx_sim_);
                P_sim_.push(p_path<PathType<stateT, actionT> >{2.1 + edge.cost/path_cost_multiplier_, edges});
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
void LazyASCP<stateT, actionT, stateHasher>::Reset(){
    // Things reset in Plan:  sp_planner_, P_sim_, P_cand_, P_valid_, O_e_, is_W_new_, failed_v_, sent_v_
    E_conf_.clear(); 
    W_current_ = make_shared<unordered_map<was::ActionState<actionT, stateT> , double, ActionStateHasher> >();
    P_sim_results_swap_.clear();
    run_sim_thread_ = true; sim_busy_ = false; sog_used_ = false; 
    min_cost_valid_path_ = INF;
    path_cost_lowerbound_ = -INF;
    sim_runs_ = 0;
}