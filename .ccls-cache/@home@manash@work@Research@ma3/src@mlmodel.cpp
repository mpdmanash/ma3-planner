#include "mlmodel.h"


MLModel::MLModel(c10::DeviceType device){
    m_device = device;
    m_options = torch::TensorOptions().dtype(torch::kHalf).device(m_device, 0);
    m_true_ref = torch::ones({1}, m_options);
    distribution_ = std::uniform_real_distribution<double>(0.0,1.0);
}

bool MLModel::loadModel(string filename, size_t max_batch_size){
    try {
        m_module = torch::jit::load(filename);
        m_module.eval();
        m_module.to(m_device);
        m_max_batch_size = max_batch_size;
        return true;
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return false;
    }
}

bool MLModel::loadEdgeFeatures(string feature_data_filename, string edge_association_filename){
    try{
        ifstream eaf (edge_association_filename);
        size_t num_vertices, max_actions;
        eaf >> num_vertices >> max_actions;
        m_stateid_features.resize(num_vertices);
        m_mprimid_features.resize(max_actions);
        torch::jit::script::Module container = torch::jit::load(feature_data_filename);
        for(size_t i=0; i<num_vertices; i++){
            m_stateid_features[i] = container.attr(to_string(i)).toTensor().toType(at::kHalf).to(m_device);
        }
        for(size_t i=0; i<max_actions; i++){
            m_mprimid_features[i] = container.attr("m"+to_string(i)).toTensor().toType(at::kHalf).to(m_device);
        }
        cout << "Features loaded\n";

        m_stateid_associations.resize(num_vertices);
        size_t sid, num_associations, said;
        int sx, sy, st, spx, spy, spz;
        float mdx, mdy, mdt, mdv, mgc, mto, mc;
        for(size_t i=0; i<num_vertices; i++){
            eaf >> sid >> num_associations;
            m_stateid_associations[i].resize(num_associations+1);
            m_stateid_associations[i][0] = i;
            for(size_t j=0; j<num_associations; j++){
                eaf >> said;
                m_stateid_associations[i][j+1] = said;
            }
        }
        cout << "Associations loaded\n";
        return true;
    }
    catch(const c10::Error& e){
        std::cerr << "error loading the data\n";
        return false;
    }
}

void MLModel::printExecutionTime(){
    cout << torch::cuda::is_available() << " cuda\n";

    auto timer_start = std::chrono::high_resolution_clock::now();
    auto timer_end = std::chrono::high_resolution_clock::now();

    std::vector<torch::jit::IValue> inputs2;
    auto aa2 = torch::randn({9, 2, 21, 23}, m_options);
    inputs2.push_back(aa2);

    // Warming Up
    for(size_t i=0; i<10; i++){
        m_module.forward(inputs2).toTensor();
    }

    double ts_time = 0;
    size_t num = 50;
    for(size_t i=0; i<num; i++){
        timer_start = std::chrono::high_resolution_clock::now();
        m_module.forward(inputs2).toTensor();
        timer_end = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration<double, std::milli>(timer_end-timer_start).count() << " ms\n";
        ts_time+=std::chrono::duration<double, std::milli>(timer_end-timer_start).count();
    }
    cout << " ======================= \n" << ts_time/num << "\n";
}

tuple<bool, bool> MLModel::isEdgeValidID(size_t stateid, size_t mprimid, double conf_threshold, double error_rate){
    assert(stateid<m_stateid_features.size() && mprimid<m_mprimid_features.size());
    vector<torch::Tensor> input_v;
    size_t batch_size = m_stateid_associations[stateid].size();
    for(size_t i=0; i<batch_size; i++){
        input_v.push_back(torch::concat({m_stateid_features[stateid].view({1,1,21,23}),
                                         m_mprimid_features[mprimid].view({1,1,21,23})},1));
    }
    if(batch_size<m_max_batch_size) input_v.push_back(torch::zeros({(signed long)((int)m_max_batch_size-batch_size), 2, 21, 23}, m_options));
    auto edge_features = torch::concat(input_v,0);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(edge_features);
    auto output = m_module.forward(inputs).toTensor().slice(1,0,batch_size).argmax(2).toType(at::kHalf).mean(0);
    bool valid = (output.mean()>=0.5).equal(m_true_ref[0]);
    bool confident = valid? (output.max()>=conf_threshold).equal(m_true_ref[0]):((1.0-output.min())>=conf_threshold).equal(m_true_ref[0]);
    double rv = distribution_(generator_);
    if(error_rate<1.0 && rv < error_rate){valid = !valid;}
    return std::make_tuple(valid, confident);
}

tuple<bool, bool> MLModel::isEdgeValid(const was::ActionState<Action, State> &edge, double conf_threshold, double error_rate){
    return isEdgeValidID(edge.state.id, edge.action.id, conf_threshold, error_rate);
}