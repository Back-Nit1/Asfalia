#include "policy_runner.h"

#include <cstring>
#include <stdexcept>

PolicyRunner::PolicyRunner(const std::string& model_path, int threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "hospital_policy") {
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(threads);
  opts.SetInterOpNumThreads(threads);
  session_ = Ort::Session(env_, model_path.c_str(), opts);  // POSIX: ORTCHAR_T is char
  Ort::AllocatorWithDefaultOptions alloc;
  auto meta = session_.GetModelMetadata();
  auto v = meta.LookupCustomMetadataMapAllocated("hospital_rl", alloc);
  if (!v) throw std::runtime_error("model has no hospital_rl metadata; refusing to run an unidentified model");
  metadata_ = v.get();
}

void PolicyRunner::run(const PolicyInputs& in, float* logits, float* value) {
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const int64_t b = in.batch, m = in.max_nodes;
  std::vector<int64_t> s_nf{b, m, in.node_dim}, s_nm{b, m}, s_adj{b, m, m}, s_h{b, in.hist_len, in.hist_dim},
      s_g{b, in.glob_dim}, s_am{b, in.n_actions};
  auto f = [&](const float* p, const std::vector<int64_t>& s) {
    size_t n = 1; for (auto d : s) n *= d;
    return Ort::Value::CreateTensor<float>(mem, const_cast<float*>(p), n, s.data(), s.size());
  };
  auto bl = [&](const bool* p, const std::vector<int64_t>& s) {
    size_t n = 1; for (auto d : s) n *= d;
    return Ort::Value::CreateTensor<bool>(mem, const_cast<bool*>(p), n, s.data(), s.size());
  };
  std::vector<Ort::Value> inputs;
  inputs.push_back(f(in.node_features, s_nf)); inputs.push_back(bl(in.node_mask, s_nm));
  inputs.push_back(bl(in.adjacency, s_adj));   inputs.push_back(f(in.history, s_h));
  inputs.push_back(f(in.globals, s_g));        inputs.push_back(bl(in.action_mask, s_am));
  const char* in_names[] = {"node_features", "node_mask", "adjacency", "history", "globals", "action_mask"};
  const char* out_names[] = {"logits", "value"};
  auto out = session_.Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), inputs.size(), out_names, 2);
  std::memcpy(logits, out[0].GetTensorData<float>(), sizeof(float) * b * in.n_actions);
  std::memcpy(value, out[1].GetTensorData<float>(), sizeof(float) * b);
}
