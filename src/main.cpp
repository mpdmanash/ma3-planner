#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <utility> // for std::pair
#include <fstream>
#include <functional> // for std::bind
#include <memory> // for unique_ptr
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <set>
#include <random>
#include "utils.h"
#include "ma3.h"

using namespace std;

size_t sim_call = 0;
std::mt19937_64 generator_;
std::uniform_real_distribution<double> distribution_ = std::uniform_real_distribution<double>(0.0,1.0);

double proxy_sim_model(const was::ActionState<Action, State> &edge, Graph &g){
    sim_call++;
    return g.getDefaultCost(edge);
}

double proxy_ml_model(const was::ActionState<Action, State> &edge, Graph &g, double error_rate=0.1){
    auto sim_out = proxy_sim_model(edge, g);
    double rv = distribution_(generator_);
    bool valid = sim_out!=INF;
    if(rv < error_rate)
        return std::make_tuple(!valid, true);
    else return std::make_tuple(valid, true);
}

using namespace std::placeholders;
int main(int argc, char *argv[]){
   // Configs
   double suboptimality_gap = -1e6;
   if(suboptimality_gap < 0.0) suboptimality_gap = INF;
   bool anytime_mode = true;
   double ml_conf_threshold = 0.6;
   int sim_wait_time = 100; // Inactivate for Real Robot Expts
   double error_rate = 0.2; // Inactivate for Real Robot Expts

   std::cout << "Configs: " << suboptimality_gap << "," << anytime_mode << "," << ml_conf_threshold << "," << sim_wait_time << "," << error_rate << std::endl;

   Graph g;
   g.readGraphFile("./data/full_map.txt");

   const State start = g.getStateByID(100);
   const State goal  = g.getStateByID(7547);
   cout << "Start: " << start << " " << "Goal:" << goal << endl; 
   Action action;
   was::ActionState<Action, State> edge;
   edge_eval(edge);
   delete publisher;
   return 0;

   was::WAstar<State, Action, StateHasher> m_dj_planner(1.0, 
                                                  std::bind(getZeroH, _1, _2),
                                                  std::bind(&Graph::getNextStates, g, _1, false, nullptr),
                                                  std::bind(satisfiesDJGoal, _1, _2) );
   auto timer_start = std::chrono::high_resolution_clock::now();
   m_dj_planner.Plan(goal, start);
   auto timer_end = std::chrono::high_resolution_clock::now();
   cout << "Planning Time: " << std::chrono::duration<double, std::milli>(timer_end-timer_start).count() << " ms\n";
   cout << "Planner Expanded: " << m_dj_planner.getNumStatesExpanded() << " states\n";

   // Method: MA3
   ma3::MA3<State, Action, StateHasher> ma3(
       std::bind(getH, _1, _2),
       std::bind(&Graph::getNextStates, g, _1, _2, _3),
       std::bind(satisfiesGoal, _1, _2),
       std::bind(proxy_sim_model, _1, g),
       std::bind(&Graph::getDefaultCost, g, _1),
       std::bind(proxy_ml_model, _1, g, error_rate),
       1000000, true, true, sim_wait_time
   );
   return 0;
}
