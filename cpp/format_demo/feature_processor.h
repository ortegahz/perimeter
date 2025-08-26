#pragma once

#include <deque>
#include <map>
#include <unordered_map>
#include <tuple>
#include <vector>
#include <string>
#include <set> // === NEW ===
#include <filesystem> // === NEW ===
#include <nlohmann/json.hpp>

/* ---------- 与 python 保持一致的常量 ---------- */
// === MODIFIED: Added all constants from Python ===
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

// --- GID 更新 / 新建参数 ---
constexpr int NEW_GID_MIN_FRAMES = 3;
constexpr int NEW_GID_TIME_WINDOW = 50;
constexpr int BIND_LOCK_FRAMES = 15;
constexpr int CANDIDATE_FRAMES = 2;
constexpr int MAX_TID_IDLE_FRAMES = 256;
constexpr int GID_MAX_IDLE_FRAMES = 25 * 60; // 简化版: 25fps * 60s
constexpr int WAIT_FRAMES_AMBIGUOUS = 10;

// --- 报警相关 ---
constexpr int ALARM_CNT_TH = 8;
constexpr float ALARM_DUP_THR = 0.4f;
constexpr float FUSE_W_FACE = 0.6f;
constexpr float FUSE_W_BODY = 0.4f;
constexpr int EMB_FACE_DIM = 512;
constexpr int EMB_BODY_DIM = 2048;

// --- 路径 (可根据需要修改)
const std::string SAVE_DIR = "/home/manu/tmp/perimeter_cpp";
const std::string ALARM_DIR = "/home/manu/tmp/perimeter_alarm_cpp";

/* ------------------------------------------------ */

struct TrackAgg {
    void add_body(const std::vector<float> &feat, float score);

    void add_face(const std::vector<float> &feat);

    std::vector<float> main_body_feat() const;

    std::vector<float> main_face_feat() const;

    static bool check_consistency(const std::deque<std::vector<float>> &feats, float thr = 0.5f);

    std::deque<std::tuple<std::vector<float>, float>> body;
    std::deque<std::vector<float>> face;
};

struct GlobalID {
    std::string new_gid();

    int can_update_proto(const std::string &gid, const std::vector<float> &face_f, const std::vector<float> &body_f);

    void bind(const std::string &gid, const std::vector<float> &face_f, const std::vector<float> &body_f,
              const std::string &tid, int current_ts);

    std::pair<std::string, float> probe(const std::vector<float> &face_f, const std::vector<float> &body_f);

    // === MODIFIED: Added last_update ===
    int gid_next = 1;
    std::map<std::string, std::vector<std::vector<float>>> bank_faces;
    std::map<std::string, std::vector<std::vector<float>>> bank_bodies;
    std::map<std::string, std::vector<std::string>> tid_hist;
    std::map<std::string, int> last_update; // GID -> last_update_fid
};

// === NEW: State structs for clarity ===
struct CandidateState {
    std::string cand_gid = "";
    int count = 0;
    int last_bind_fid = 0;
};

struct NewGidState {
    int count = 0;
    int last_new_fid = -NEW_GID_TIME_WINDOW; // Initialize to allow immediate creation
    int ambig_count = 0;
};

class FeatureProcessor {
public:
    explicit FeatureProcessor(const std::string &cache_path);

    std::map<std::string, std::map<int, std::tuple<std::string, float, int>>>
    process_packet(const std::string &stream_id, int fid);

private:
    std::vector<float> _fuse_feat(const std::vector<float> &face_f, const std::vector<float> &body_f);

    std::vector<float> _gid_fused_rep(const std::string &gid);

    void trigger_alarm(const std::string &gid);

    nlohmann::json features_cache;
    std::map<std::string, TrackAgg> agg_pool;
    GlobalID gid_mgr;
    std::map<std::string, std::string> tid2gid; // tid_str -> gid
    std::unordered_map<std::string, int> last_seen; // tid_str -> fid

    // === MODIFIED: Replaced with more detailed state structs ===
    std::unordered_map<std::string, CandidateState> candidate_state;
    std::unordered_map<std::string, NewGidState> new_gid_state;

    // === NEW: Alarm state ===
    std::set<std::string> alarmed;
    std::map<std::string, std::vector<float>> alarm_reprs;
};