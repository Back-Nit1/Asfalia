#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "policy_runner.h"

namespace py = pybind11;
template <class T> using Arr = py::array_t<T, py::array::c_style>;  // noconvert on the call: wrong dtype/layout is an error

PYBIND11_MODULE(hospital_policy_runner, m) {
  m.doc() = "ONNX Runtime C++ policy runner (feed-forward)";
  py::class_<PolicyRunner>(m, "PolicyRunner")
      .def(py::init<const std::string&, int>(), py::arg("model_path"), py::arg("threads") = 1)
      .def("metadata_json", &PolicyRunner::metadata_json)
      .def("run", [](PolicyRunner& r, Arr<float> nf, Arr<bool> nm, Arr<bool> adj, Arr<float> hist, Arr<float> glob, Arr<bool> am) {
        if (nf.ndim() != 3 || hist.ndim() != 3 || am.ndim() != 2) throw std::invalid_argument("bad rank");
        PolicyInputs in{nf.data(), nm.data(), adj.data(), hist.data(), glob.data(), am.data(),
                        nf.shape(0), nf.shape(1), nf.shape(2), hist.shape(1), hist.shape(2), glob.shape(1), am.shape(1)};
        py::array_t<float> logits({in.batch, in.n_actions}), value(in.batch);
        {
          py::gil_scoped_release nogil;
          r.run(in, logits.mutable_data(), value.mutable_data());
        }
        return py::make_tuple(logits, value);
      }, py::arg("node_features").noconvert(), py::arg("node_mask").noconvert(), py::arg("adjacency").noconvert(),
         py::arg("history").noconvert(), py::arg("globals").noconvert(), py::arg("action_mask").noconvert());
}
