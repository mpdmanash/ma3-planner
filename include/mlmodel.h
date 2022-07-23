#ifndef MLMODEL_H
#define MLMODEL_H

#include <cstdlib>
#include <torch/torch.h>
#include <torch/script.h>
#include "torch_tensorrt/torch_tensorrt.h"
#include <string>
#include <tuple>
#include <fstream>
#include <vector>
#include <random>
#include "utils.h"

using namespace std;

class MLModel{
public:
    MLModel(c10::DeviceType device=at::kCUDA);
    bool loadModel(string filename, size_t m_max_batch_size=9);
    bool loadEdgeFeatures(string feature_data_filename, string edge_association_filename);
    tuple<bool, bool> isEdgeValidID(size_t stateid, size_t mprimid, double conf_threshold, double error_rate=-1.0);
    tuple<bool, bool> isEdgeValid(const was::ActionState<Action, State> &edge, double conf_threshold, double error_rate=-1.0);
    void printExecutionTime();
private:
    size_t m_max_batch_size=0;
    at::Tensor m_true_ref;
    c10::TensorOptions m_options;
    c10::DeviceType m_device;
    torch::jit::script::Module m_module;
    vector<torch::Tensor> m_stateid_features;
    vector<torch::Tensor> m_mprimid_features;
    vector<vector<size_t> > m_stateid_associations;
    std::mt19937_64 generator_;
    std::uniform_real_distribution<double> distribution_;
};

#endif