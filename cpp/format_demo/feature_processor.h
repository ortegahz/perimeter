#pragma once

#include <deque>
#include <map>
#include <unordered_map>
#include <tuple>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>

/* ---------- 与 python 保持一致的常量 ---------- */
constexpr int MIN_BODY4GID = 8;
constexpr int MIN_FACE4GID = 8;
constexpr float W_FACE = 0.6f;
constexpr float W_BODY = 0.4f;
constexpr float MATCH_THR = 0.5f;
constexpr float THR_NEW_GID = 0.3f;
constexpr float UPDATE_THR = 0.65f;
constexpr float FACE_THR_STRICT = 0.5f;
constexpr float BODY_THR_STRICT = 0.4f;
constexpr int MAX_PROTO_PER_TYPE = 8;
constexpr int MAX_TID_IDLE_FRAMES = 256;
constexpr int CANDIDATE_FRAMES = 2;

/* ------------------------------------------------ */

struct TrackAgg {
    void add_body(const std::vector<float> &feat, float score);

    void add_face(const std::vector<float> &feat);

    std::vector<float> main_body_feat() const;

    std::vector<float> main_face_feat() const;

    static bool check_consistency(const std::deque<std::vector<float>> &feats,
                                  float thr = 0.5f);

    std::deque<std::tuple<std::vector<float>, float>> body;   // (feat,score)
    std::deque<std::vector<float>> face;   // feat
};

/* ---------------- GlobalID ---------------- */
struct GlobalID {
    std::string new_gid();

    /* 0 = ok,  -1 / -2 = face / body 不兼容 */
    int can_update_proto(const std::string &gid,
                         const std::vector<float> &face_f,
                         const std::vector<float> &body_f);

    void bind(const std::string &gid,
              const std::vector<float> &face_f,
              const std::vector<float> &body_f,
              const std::string &tid);

    std::pair<std::string, float>
    probe(const std::vector<float> &face_f,
          const std::vector<float> &body_f);

    int gid_next = 1;
    std::map<std::string, std::vector<std::vector<float>>> bank_faces;
    std::map<std::string, std::vector<std::vector<float>>> bank_bodies;
    std::map<std::string, std::vector<std::string>> tid_hist;
};

/* --------------- FeatureProcessor (load) --------------- */
class FeatureProcessor {
public:
    explicit FeatureProcessor(const std::string &cache_path);

    /* realtime_map[cam_id][tid] = (gid/flag , score , n_tid) */
    std::map<std::string,
            std::map<int, std::tuple<std::string, float, int>>>
    process_packet(const std::string &stream_id, int fid);

private:
    nlohmann::json features_cache;
    std::map<std::string, TrackAgg> agg_pool;
    GlobalID gid_mgr;
    std::map<std::string, std::string> tid2gid;
    std::unordered_map<std::string, int> last_seen;

    /* 候选状态： tid -> (cand_gid , 连续命中次数) */
    std::unordered_map<std::string, std::pair<std::string, int>> candidate_state;
};