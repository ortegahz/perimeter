#include <pybind11/embed.h>   // 用于在 C++ 中调用 Python
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <iostream>

namespace py = pybind11;

struct TrackInput {
    std::string stream_id;
    int tid;
    int fid;
    std::vector<float> body_feat;
    float body_score;
    std::vector<float> face_feat;
    bool has_face;
};

// 调用 Python 函数的辅助函数
std::string call_python_greet(const std::string &name) {
    try {
        // 导入 py_helper.py
        py::module_ helper = py::module_::import("py_helper");
        // 调用 greet_from_py
        py::object result = helper.attr("greet_from_py")(name);
        return result.cast<std::string>();
    } catch (py::error_already_set &e) {
        // 捕获 Python 抛出的异常
        std::cerr << "Python error: " << e.what() << std::endl;
        return "";
    }
}

std::unordered_map<std::string, std::unordered_map<int, std::tuple<std::string, float, int>>>
process_tracks(const std::vector<TrackInput> &inputs) {
    std::unordered_map<std::string, std::unordered_map<int, std::tuple<std::string, float, int>>> result;

    for (auto &inp: inputs) {
        // 调用 Python 函数
        std::string greet_msg = call_python_greet(inp.stream_id);
        std::cout << "[C++] Received from Python: " << greet_msg << std::endl;

        std::string gid = (inp.tid % 2 == 0) ? "G00001" : "-unknown-";
        float score = inp.body_score;
        int n_tid = 1;
        result[inp.stream_id][inp.tid] = {gid, score, n_tid};
    }
    return result;
}

PYBIND11_MODULE(feature_logic, m) {
    py::class_<TrackInput>(m, "TrackInput")
            .def(py::init<>())
            .def_readwrite("stream_id", &TrackInput::stream_id)
            .def_readwrite("tid", &TrackInput::tid)
            .def_readwrite("fid", &TrackInput::fid)
            .def_readwrite("body_feat", &TrackInput::body_feat)
            .def_readwrite("body_score", &TrackInput::body_score)
            .def_readwrite("face_feat", &TrackInput::face_feat)
            .def_readwrite("has_face", &TrackInput::has_face);

    m.def("process_tracks", &process_tracks, "Process track logic demo");
}