#include "utils.h"

Graph::Graph(){}
void Graph::readGraphFile(string filename){
    ifstream myfile (filename);
    size_t sid, spid, mpid, num_vertices, max_actions, num_edges;
    int sx, sy, st, spx, spy, spz;
    float mdx, mdy, mdt, mdv, mgc, mto, mc;
    myfile >> num_vertices >> max_actions;
    m_max_transitions = 0;
    m_vertices.resize(num_vertices);
    m_forward_edges.resize(num_vertices);
    m_backward_edges.resize(num_vertices);
    m_actions.resize(max_actions);
    cout << num_vertices << " Num Verts\n";
    for(size_t i = 0; i < num_vertices; i++){
        myfile >> sid >> sx >> sy >> st;
        myfile >> num_edges;
        m_vertices[sid].x = sx; m_vertices[sid].y = sy; m_vertices[sid].t = st; m_vertices[sid].id = sid; 
        for(size_t j=0; j<num_edges; j++){
            myfile >> mpid >> mdx >> mdy >> mdt >> mdv >> mgc >> mto >> mc >> spid >> spx >> spy >> spz;
            m_actions[mpid].dx=mdx; m_actions[mpid].dy=mdy; m_actions[mpid].dt=mdt; m_actions[mpid].velocity=mdv;
            m_actions[mpid].goal_threshold=mgc; m_actions[mpid].timeout=mto; m_actions[mpid].cost=mc; m_actions[mpid].id=mpid;
            m_forward_edges[sid].push_back(make_pair(mpid,spid));
            m_backward_edges[spid].push_back(make_pair(mpid,sid));
            m_max_transitions = max(m_max_transitions,m_backward_edges[spid].size());
        }
    }
    m_max_transitions = max(m_max_transitions,max_actions);
    m_successors_container.resize(m_max_transitions);
    cout << "Max transitions="<<m_max_transitions<< ". Done creating graph\n";
}

vector<was::ActionState<Action, State> > Graph::getNextStates(const State s, bool successor,
    const shared_ptr<unordered_map<was::ActionState<Action, State>, double, ActionStateHasher> > W_current){
    if(successor) return this->getSuccessors(s,W_current);
    else return this->getPredecessor(s,W_current);
}

vector<was::ActionState<Action, State> > Graph::getSuccessors(const State s,
    const shared_ptr<unordered_map<was::ActionState<Action, State>, double, ActionStateHasher> > W_current){
    size_t num_s = m_forward_edges[s.id].size();
    unsigned short model;
    for(size_t i=0; i<num_s; i++){
        size_t mpid = m_forward_edges[s.id][i].first;
        size_t spid = m_forward_edges[s.id][i].second;
        m_successors_container[i].action = m_actions[mpid];
        m_successors_container[i].state = m_vertices[spid];
        m_successors_container[i].cost = this->getW(was::ActionState<Action, State>{m_actions[mpid], s}, 
                                                    W_current, &model);
        m_successors_container[i].valid = true;
        m_successors_container[i].model = model;
    }
    for(size_t i=num_s; i<m_max_transitions; i++) m_successors_container[i].valid = false;
    return m_successors_container; // This will be copied out. Can't do RVO
}

vector<was::ActionState<Action, State> > Graph::getPredecessor(const State sp,
    const shared_ptr<unordered_map<was::ActionState<Action, State>, double, ActionStateHasher> > W_current){
    size_t num_s = m_backward_edges[sp.id].size();
    unsigned short model;
    for(size_t i=0; i<num_s; i++){
        size_t mpid = m_backward_edges[sp.id][i].first;
        size_t sid = m_backward_edges[sp.id][i].second;
        m_successors_container[i].action = m_actions[mpid];
        m_successors_container[i].state = m_vertices[sid];
        m_successors_container[i].cost = this->getW(m_successors_container[i],
                                                    W_current, &model);
        m_successors_container[i].valid = true;
        m_successors_container[i].model = model;
    }
    for(size_t i=num_s; i<m_max_transitions; i++) m_successors_container[i].valid = false;
    return m_successors_container;
}

double Graph::getW(const was::ActionState<Action, State> &edge, 
                   const shared_ptr<unordered_map<was::ActionState<Action, State>, double, ActionStateHasher> > W_current,
                   unsigned short *model){
    *model = was::MODEL_INIT;
    if(!W_current) return this->getDefaultCost(edge);
    auto it = W_current.get()->find(edge);
    if (it != W_current.get()->end()){
        *model = it->first.model;
        return it->second;
    }
    else return this->getDefaultCost(edge);
}

double Graph::getDefaultCost(const was::ActionState<Action, State> &edge){
    return edge.action.cost;
}

State Graph::getStateNearGuess(int x, int y, int t){
    size_t best_id = 0;
    double best_distance = INF;
    for(size_t i=0; i<m_vertices.size(); i++){
        double distance = std::pow((double)(x-m_vertices[i].x),2) + std::pow((double)(y-m_vertices[i].y),2) + std::pow((double)(t-m_vertices[i].t),2);
        if (distance < best_distance){
            best_distance = distance;
            best_id = i;
        }
    }
    return m_vertices[best_id];
}