#pragma once

namespace was{
template <class stateT, class actionT, class stateHasher>
WAstar<stateT, actionT, stateHasher>::WAstar(double epsilon, 
        std::function<double(const stateT &state, const stateT &goal)> getH,
        std::function<std::vector<ActionState<actionT, stateT> >(const stateT &state)> getSuccessors,
        std::function<bool(const stateT &state, const stateT &goal)> satisfiesGoal){
    m_epsilon = epsilon;
    m_getH = getH;
    m_getSuccessors = getSuccessors;
    m_satisfiesGoal = satisfiesGoal;
};

template <class stateT, class actionT, class stateHasher>
double WAstar<stateT, actionT, stateHasher>::Plan(const stateT &s_start, const stateT &s_goal,
                                                  std::priority_queue<ActionState<actionT, stateT> >* O_e){
    m_g[s_start] = 0.0;
    m_OPEN.push(f_State<stateT>{0.0, s_start});
    bool goal_expanded = false;
    while(!goal_expanded && m_OPEN.size() != 0){
        auto this_state = m_OPEN.top().state;
        m_OPEN.pop();
        m_CLOSED.insert(this_state);
        if(m_satisfiesGoal(this_state, s_goal)){
            goal_expanded=true; 
            backtrackState=this_state; 
            return this->getG(this_state);
        }
        double g_this_state = this->getG(this_state);
        auto successors = m_getSuccessors(this_state);        
        for(auto &successor: successors){
            if(!successor.valid) continue;
            if(!this->isInCLOSED(successor.state,m_CLOSED)){
                double h_successor = m_getH(successor.state, s_goal);
                double g_successor_old = this->getG(successor.state);
                double g_successor_new = g_this_state + successor.cost;
                if (g_successor_old > g_successor_new){
                    m_g[successor.state] = g_successor_new;
                    m_backward_transitions[successor.state] = ActionState<actionT, stateT>{successor.action, this_state, successor.cost};
                    m_OPEN.push(f_State<stateT>{g_successor_new + m_epsilon*h_successor, successor.state});
                }
            }
        }
    }
    return std::numeric_limits<double>::infinity(); // infinity if path to goal is not found
};

template <class stateT, class actionT, class stateHasher>
void WAstar<stateT, actionT, stateHasher>::Reset(){
    m_g.clear(); m_OPEN = std::priority_queue<f_State<stateT> >(); m_backward_transitions.clear(); m_CLOSED.clear();
};

template <class stateT, class actionT, class stateHasher>
std::vector<ActionState<actionT, stateT> > WAstar<stateT, actionT, stateHasher>::BackTrackPath(const stateT &s_goal, size_t path_id){
    std::vector<ActionState<actionT, stateT> > path;
    stateT prev_state = backtrackState;
    while(true){
        auto it = m_backward_transitions.find(prev_state);
        if (it != m_backward_transitions.end()) {
            path.insert(path.begin(), it->second);
            path[0].path_id = path_id;
            prev_state = it->second.state;
        }
        else break;
    }
    return path;
};

template <class stateT, class actionT, class stateHasher>
inline double WAstar<stateT, actionT, stateHasher>::getGHeuristic(const stateT &s, const stateT &s2){
    auto it = m_g.find(s);
    if (it != m_g.end()) return it->second;
    else return std::numeric_limits<double>::infinity();
};

template <class stateT, class actionT, class stateHasher>
inline bool WAstar<stateT, actionT, stateHasher>::isInCLOSED(const stateT &s, 
        const std::unordered_set<stateT, stateHasher> &CLOSED){
    return (CLOSED.find(s) != CLOSED.end());
};

template <class stateT, class actionT, class stateHasher>
inline double WAstar<stateT, actionT, stateHasher>::getG(const stateT &s){
    auto it = m_g.find(s);
    if (it != m_g.end()) return it->second;
    else return std::numeric_limits<double>::infinity();
};

}