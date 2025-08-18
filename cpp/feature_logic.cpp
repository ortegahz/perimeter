#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <tuple>

namespace py = pybind11;

// 定义一个输入数据结构
struct TrackInput {
    std::string stream_id;
    int tid;
    int fid;
    std::vector<float> body_feat;
    float body_score;
    std::vector<float> face_feat;
    bool has_face;
};

// 一个极简的逻辑函数：仅用 C++ 接收输入，然后输出一个 map
std::unordered_map<std::string, std::unordered_map<int, std::tuple<std::string, float, int>>>

process_tracks(const std::vector<TrackInput> &inputs) {
    std::unordered_map<std::string, std::unordered_map<int, std::tuple<std::string, float, int>>> result;

    for (auto &inp: inputs) {
        std::string gid = (inp.tid % 2 == 0) ? "G00001" : "-unknown-"; // 假逻辑
        float score = inp.body_score;  // 直接返回 body_score
        int n_tid = 1;
        result[inp.stream_id][inp.tid] = {gid, score, n_tid};
    }
    return result;
}

// pybind11 模块绑定
PYBIND11_MODULE(feature_logic, m
) {
    py::class_<TrackInput>(m,
                           "TrackInput")
            .

                    def(py::init<>())

            .def_readwrite("stream_id", &TrackInput::stream_id)
            .def_readwrite("tid", &TrackInput::tid)
            .def_readwrite("fid", &TrackInput::fid)
            .def_readwrite("body_feat", &TrackInput::body_feat)
            .def_readwrite("body_score", &TrackInput::body_score)
            .def_readwrite("face_feat", &TrackInput::face_feat)
            .def_readwrite("has_face", &TrackInput::has_face);

    m.def("process_tracks", &process_tracks, "Process track logic demo");
}