#ifndef UTILS_H
#define UTILS_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <functional> // for std::bind
#include <memory> // for unique_ptr
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <sstream>
#include "lib_wastar.h"

using namespace std;

const double INF = std::numeric_limits<double>::infinity();

struct State{
    int x; int y; int t;
    size_t id;

    bool operator==(const State& rhs) const {return id == rhs.id;}

    friend ostream& operator<<(ostream& os, const State& s)
    {
        os << "State: (" << s.id << ") " << s.x << "," << s.y << "," << s.t;
        return os;
    }
};

struct Action{
    float dx, dy, dt, velocity, goal_threshold, timeout, cost;
    size_t id;
    friend ostream& operator<<(ostream& os, const Action& a)
    {
        os << "Action: (" << a.id << ") " << a.dx << "," << a.dy << "," << a.dt;
        return os;
    }
};

inline double getZeroH(State state, State goal){return 0;}
inline double getH(State s, State g){return sqrt((s.x-g.x)*(s.x-g.x) + (s.y-g.y)*(s.y-g.y)) ;}
inline bool satisfiesGoal(State state, State goal){return (state.x==goal.x && state.y==goal.y);}
inline bool satisfiesDJGoal(State state, State goal){return false;}
struct StateHasher
{
    size_t operator()(const State& state) const{return state.id;}
};
struct ActionStateHasher
{
    size_t operator()(const was::ActionState<Action, State>& edge) const{
        return (((uint)edge.action.id * 73856093u) ^ ((uint)edge.state.id * 19349669u));
    }
};

inline void LogPath(const std::vector<was::ActionState<Action, State> >& path, std::string source){
    std::stringstream log_out;
    for(const auto &edge: path) log_out << edge << " | ";
    log_out << "<= Path\n";
    //spdlog::get("lazy-ascp-logger")->info("{} path {}", source, log_out.str());
}

inline void Log(std::string source, std::string message){
    //spdlog::get("lazy-ascp-logger")->info("{}: {}", source, message);
}


class Graph{
public:
    Graph();
    void readGraphFile(string filename);
    vector<was::ActionState<Action, State> > getNextStates(const State s, bool successor,
        const shared_ptr<unordered_map<was::ActionState<Action, State>, double, ActionStateHasher> > W_current={});
    inline State getStateByID(size_t id){return m_vertices[id];}
    inline State getRandomState(){
        size_t id = rand() % m_vertices.size();
        return m_vertices[id];
    }

    State getStateNearGuess(int x, int y, int t);
    double getDefaultCost(const was::ActionState<Action, State> &edge);

private:
    size_t m_max_transitions;
    vector<State> m_vertices;
    vector<Action> m_actions;
    vector<vector<pair<size_t, size_t> > > m_forward_edges;
    vector<vector<pair<size_t, size_t> > > m_backward_edges;
    vector<was::ActionState<Action, State> > m_successors_container;
    vector<was::ActionState<Action, State> > getSuccessors(const State s,
        const shared_ptr<unordered_map<was::ActionState<Action, State>, double, ActionStateHasher> > W_current={});
    vector<was::ActionState<Action, State> > getPredecessor(const State sp,
        const shared_ptr<unordered_map<was::ActionState<Action, State>, double, ActionStateHasher> > W_current={});
    double getW(const was::ActionState<Action, State> &edge,
                const shared_ptr<unordered_map<was::ActionState<Action, State>, double, ActionStateHasher> > W_current,
                unsigned short *model);
};

#endif