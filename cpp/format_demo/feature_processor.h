#pragma once

#include <deque>
#include <map>
#include <tuple>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>

constexpr int MIN_BODY4GID = 8;
constexpr int MIN_FACE4GID = 8;
constexpr float W_FACE = 0.6f;
constexpr float W_BODY = 0.4f;
constexpr float MATCH_THR = 0.5f;
constexpr float FACE_THR_STRICT = 0.5f;
constexpr float BODY_THR_STRICT = 0.4f;

struct TrackAgg {
    void add_body(const std::vector<float> &feat, float score);

    void add_face(const std::vector<float> &feat);

    std::vector<float> main_body_feat() const;

    std::vector<float> main_face_feat() const;

    std::deque<std::tuple<std::vector<float>, float>> body;
    std::deque<std::vector<float>> face;
};

struct GlobalID {
    std::string new_gid();

    /* 0 = ok；-1 = face 不一致；-2 = body 不一致 */
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

class FeatureProcessor {
public:
    explicit FeatureProcessor(const std::string &cache_path);

    /* realtime_map[stream_id][tid] = {gid , score , n_tid} */
    std::map<std::string, std::map<int, std::tuple<std::string, float, int>>>
    process_packet(const std::string &stream_id, int fid);

private:
    nlohmann::json features_cache;
    std::map<std::string, TrackAgg> agg_pool;
    GlobalID gid_mgr;
    std::map<std::string, std::string> tid2gid;
};