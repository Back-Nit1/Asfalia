// Long-lived ONNX Runtime session for the exported feed-forward policy. No subprocess, no network.
#pragma once
#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <string>
#include <vector>

struct PolicyInputs {  // borrowed, C-contiguous; the caller keeps ownership
  const float* node_features; const bool* node_mask; const bool* adjacency;
  const float* history; const float* globals; const bool* action_mask;
  int64_t batch, max_nodes, node_dim, hist_len, hist_dim, glob_dim, n_actions;
};

class PolicyRunner {
 public:
  PolicyRunner(const std::string& model_path, int threads);
  // Writes logits[batch * n_actions] and value[batch]; outputs are owned by the caller.
  void run(const PolicyInputs& in, float* logits, float* value);
  std::string metadata_json() const { return metadata_; }

 private:
  Ort::Env env_;
  Ort::Session session_{nullptr};
  std::string metadata_;
};
