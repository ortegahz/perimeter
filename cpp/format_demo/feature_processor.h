#pragma once

#include <string>
#include <map>
#include <deque>
#include <vector>
#include <tuple>
#include <nlohmann/json.hpp>

// 常量（与 Python 保持一致）
const int MIN_BODY4GID = 8;
const int MIN_FACE4GID = 8;
const float MATCH_THR = 0.5f;
const float FACE_THR_STRICT = 0.5f;
const float BODY_THR_STRICT = 0.4f;
const float W_FACE = 0.6f;
const float W_BODY = 0.4f;

// 工具
float sim_vec(const std::vector<float> &a, const std::vector<float> &b);

std::vector<float> avg_feats(const std::vector<std::vector<float>> &feats);

// TrackAgg: 存储一条轨迹的 body / face 特征
struct TrackAgg {
    std::deque<std::tuple<std::vector<float>, float>> body;
    std::deque<std::vector<float>> face;

    void add_body(const std::vector<float> &feat, float score);

    void add_face(const std::vector<float> &feat);

    std::vector<float> main_body_feat() const;

    std::vector<float> main_face_feat() const;
};

// GlobalID: 全局身份管理
struct GlobalID {
    int gid_next = 1;
    std::map<std::string, std::vector<std::vector<float>>> bank_faces;
    std::map<std::string, std::vector<std::vector<float>>> bank_bodies;
    std::map<std::string, std::vector<std::string>> tid_hist;

    std::string new_gid();

    bool can_update_proto(const std::string &gid, const std::vector<float> &face_f, const std::vector<float> &body_f);

    void bind(const std::string &gid, const std::vector<float> &face_f, const std::vector<float> &body_f,
              const std::string &tid);

    std::pair<std::string, float> probe(const std::vector<float> &face_f, const std::vector<float> &body_f);
};

// FeatureProcessor: load 模式
class FeatureProcessor {
public:
    FeatureProcessor(const std::string &cache_path);

    // 返回: cam_id -> tid -> (gid, score, n_tid)
    std::map<std::string, std::map<int, std::tuple<std::string, float, int>>>
    process_packet(const std::string &stream_id, int fid);

private:
    nlohmann::json features_cache;
    GlobalID gid_mgr;
    std::map<std::string, TrackAgg> agg_pool;
    std::map<std::string, std::string> tid2gid;
};