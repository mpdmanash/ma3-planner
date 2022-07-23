#ifndef LIB_WASTAR_H
#define LIB_WASTAR_H

#include <functional> // for hash
#include <vector> // for successors
#include <unordered_map> // for G values
#include <unordered_set> // for CLOSED
#include <queue> // for OPEN

namespace was{
template <typename T>
struct f_State{
    double f; T state;
    bool operator<(const f_State<T>& rhs) const
    {return (-f < -rhs.f);}
};

const unsigned short MODEL_INIT=1;
const unsigned short MODEL_ML=2;
const unsigned short MODEL_DIS=3;
const unsigned short MODEL_SIM=4;
const unsigned short MODEL_SOG=5;

template<typename actionT, typename stateT>
struct ActionState{
    actionT action;
    stateT state;
    double cost = 0;
    bool valid = true;
    size_t path_id = 0;
    unsigned short model = MODEL_INIT;
    bool operator==(const ActionState& rhs) const {return (action.id==rhs.action.id) && (state.id==rhs.state.id);}
    bool operator< (const ActionState& rhs) const {return (-cost < -rhs.cost);}
    friend std::ostream& operator<<(std::ostream& os, const ActionState& s)
    {
        os << s.state << " " << s.action << " &" << s.model;
        return os;
    }
};

template <class stateT, class actionT, class stateHasher = std::hash<stateT> >
class WAstar{
public:
    WAstar(double epsilon, 
           std::function<double(const stateT &state, const stateT &goal)> getH, 
           std::function<std::vector<ActionState<actionT, stateT> >(const stateT &state)> getSuccessors,
           std::function<bool(const stateT &state, const stateT &goal)> satisfiesGoal);
    double Plan(const stateT &s_start,const stateT &s_goal,
                std::priority_queue<ActionState<actionT, stateT> >* O_e = NULL);
    std::vector<ActionState<actionT, stateT> > BackTrackPath(const stateT &s_goal, size_t path_id=0);
    inline size_t getNumStatesExpanded(){return m_CLOSED.size();};
    double getGHeuristic(const stateT &state, const stateT &s2);
    void Reset();
private:
    double m_epsilon;
    std::function<double(const stateT &state, const stateT &goal)> m_getH;
    std::function<std::vector<ActionState<actionT, stateT> >(const stateT &state)> m_getSuccessors;
    std::function<bool(const stateT &state, const stateT &goal)> m_satisfiesGoal;
    std::unordered_map<stateT, double, stateHasher> m_g;
    std::priority_queue<f_State<stateT> > m_OPEN;
    std::unordered_set<stateT, stateHasher> m_CLOSED;
    std::unordered_map<stateT, ActionState<actionT, stateT>, stateHasher> m_backward_transitions;
    stateT backtrackState;

    double getG(const stateT &state);
    bool isInCLOSED(const stateT &s, const std::unordered_set<stateT, stateHasher> &CLOSED);
};
}

#include "lib_wastar.tpp"

#endif
