#pragma once

using namespace std;
using namespace std::placeholders;

template <class stateT, class actionT, class stateHasher>
LazySP<stateT, actionT, stateHasher>::LazySP(
           std::function<double(const stateT &state, const stateT &goal)> getH,
           std::function<std::vector<was::ActionState<actionT, stateT> >(const stateT &state, bool successor,
               const std::shared_ptr<std::unordered_map<was::ActionState<actionT, stateT> , double, ActionStateHasher> >)> getNextStates,
           std::function<bool(const stateT &state, const stateT &goal)> satisfiesGoal,
           std::function<double(const was::ActionState<actionT, stateT> &edge)> evaluateEdge,
           std::function<double(const was::ActionState<actionT, stateT> &edge)> QueryOptimisticModel){
    m_W_current = make_shared<unordered_map<was::ActionState<actionT, stateT> , double, ActionStateHasher> >();
    m_getH = getH;
    m_getNextStates = getNextStates;
    m_satisfiesGoal = satisfiesGoal;
    m_evaluateEdge = evaluateEdge;
    m_QueryOptimisticModel = QueryOptimisticModel;
}

template <class stateT, class actionT, class stateHasher>
double LazySP<stateT, actionT, stateHasher>::Plan(const stateT &s_start, const stateT &s_goal){

    was::WAstar<stateT, actionT, stateHasher> m_h_planner(1.0,
                                                   std::bind(m_getH, _1, _2),
                                                   std::bind(m_getNextStates, _1, true, m_W_current),
                                                   std::bind(m_satisfiesGoal, _1, _2) );

    while(true){
        double path_cost = m_h_planner.Plan(s_start, s_goal);
        if(path_cost == std::numeric_limits<double>::infinity()) break; // No path candidate found
        vector<was::ActionState<actionT, stateT> > path = m_h_planner.BackTrackPath(s_goal);
        // LogPath(path, "LazySP for validation");
        size_t selected = selectNextEdge(path, 0);
        if (selected == path.size()){
            // LogPath(path, "LazySP");
            cout << path_cost << " Path Found by LazySP\n";
            return path_cost;
        }
        // else cout << path_cost << " " << selected << " edge selected by LazySP\n";
        while(selected != path.size()){
            double edge_cost = m_evaluateEdge(path[selected]);
            (*m_W_current)[path[selected]] = edge_cost;
            m_E_conf.insert(path[selected]);
            selected = selectNextEdge(path, selected+1);
            if(edge_cost==INF) break;
        }
        m_h_planner.Reset();
    }
    return std::numeric_limits<double>::infinity();
}

template <class stateT, class actionT, class stateHasher>
bool LazySP<stateT, actionT, stateHasher>::isInEconf(const was::ActionState<actionT, stateT> &edge){
    if (m_E_conf.find(edge) != m_E_conf.end()) return true;
    return false;
}

template <class stateT, class actionT, class stateHasher>
size_t LazySP<stateT, actionT, stateHasher>::selectNextEdge(const std::vector<was::ActionState<actionT, stateT> > &path, size_t start_id){
    for(size_t i=start_id; i<path.size(); i++) if(!isInEconf(path[i])) return i;
    return path.size();
}

template <class stateT, class actionT, class stateHasher>
void LazySP<stateT, actionT, stateHasher>::Reset(){
    m_E_conf.clear();
    m_W_current = make_shared<unordered_map<was::ActionState<actionT, stateT> , double, ActionStateHasher> >();
}
