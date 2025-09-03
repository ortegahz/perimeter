#pragma once

#include <deque>
#include <map>
#include <unordered_map>
#include <tuple>
#include <vector>
#include <string>
#include <set>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

// 包含模型头文件
#include "cores/personReid/PersonReid.hpp"
#include "cores/face/FaceAnalyzer.hpp"

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

constexpr int NEW_GID_MIN_FRAMES = 3;
constexpr int NEW_GID_TIME_WINDOW = 50;
constexpr int BIND_LOCK_FRAMES = 15;
constexpr int CANDIDATE_FRAMES = 2;
constexpr int MAX_TID_IDLE_FRAMES = 256;
constexpr int GID_MAX_IDLE_FRAMES = 1080000;
constexpr int WAIT_FRAMES_AMBIGUOUS = 10;

constexpr int ALARM_CNT_TH = 2;
constexpr float ALARM_DUP_THR = 0.4f;
constexpr float FUSE_W_FACE = 0.6f;
constexpr float FUSE_W_BODY = 0.4f;
constexpr int EMB_FACE_DIM = 512;
constexpr int EMB_BODY_DIM = 2048;
constexpr int BEHAVIOR_ALARM_DURATION_FRAMES = 256;

constexpr float MIN_HW_RATIO = 1.5f;
constexpr float FACE_DET_MIN_SCORE = 0.60f;

// ======================= 【MODIFIED】 =======================
// 路径修改为与 Python 版本在同一目录下，以方便对比
const std::string SAVE_DIR = "/mnt/nfs/perimeter_cpp";
const std::string ALARM_DIR = "/mnt/nfs/perimeter_alarm_cpp";
// ======================= 【修改结束】 =======================

/* ---------- 数据结构定义 ---------- */
struct Detection {
    cv::Rect2f tlwh;
    float score;
    int id;
    int class_id = 0;
};

struct Packet {
    std::string cam_id;
    int fid;
    std::vector<cv::Mat> patches;
    std::vector<Detection> dets;
};

/* ===== 入侵和穿越检测模块 (辅助类) ===== */
static cv::Point2f get_foot_point(const cv::Rect2f &tlwh) {
    return cv::Point2f(tlwh.x + tlwh.width / 2.0f, tlwh.y + tlwh.height);
}

class IntrusionDetector {
public:
    explicit IntrusionDetector(const std::vector<cv::Point> &boundary_poly) {
        if (boundary_poly.size() >= 3) {
            boundary_ = boundary_poly;
        }
    }

    std::set<int> check(const std::vector<Detection> &dets, const std::string &stream_id) {
        if (boundary_.empty()) return {};
        std::set<int> newly_alarmed_tids;
        std::set<int> current_tids;
        for (const auto &d: dets) {
            current_tids.insert(d.id);
            if (alarmed_tids_.count(d.id)) continue;
            cv::Point2f current_point = get_foot_point(d.tlwh);
            auto it = track_history_.find(d.id);
            if (it != track_history_.end()) {
                cv::Point2f last_point = it->second;
                if (cv::pointPolygonTest(boundary_, last_point, false) < 0 &&
                    cv::pointPolygonTest(boundary_, current_point, false) >= 0) {
                    alarmed_tids_.insert(d.id);
                    newly_alarmed_tids.insert(d.id);
                }
            }
            track_history_[d.id] = current_point;
        }
        std::vector<int> disappeared_tids;
        for (const auto &pair: track_history_) {
            if (current_tids.find(pair.first) == current_tids.end()) {
                disappeared_tids.push_back(pair.first);
            }
        }
        for (int tid: disappeared_tids) {
            track_history_.erase(tid);
            alarmed_tids_.erase(tid);
        }
        return newly_alarmed_tids;
    }

private:
    std::vector<cv::Point> boundary_;
    std::map<int, cv::Point2f> track_history_;
    std::set<int> alarmed_tids_;
};

static int get_point_side(const cv::Point2f &p, const cv::Point2f &a, const cv::Point2f &b) {
    float val = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
    if (val > 0) return 1;
    if (val < 0) return -1;
    return 0;
}

class LineCrossingDetector {
public:
    LineCrossingDetector(const cv::Point &start, const cv::Point &end, const std::string &direction = "any")
            : line_start_(start), line_end_(end), direction_(direction) {}

    std::set<int> check(const std::vector<Detection> &dets, const std::string &stream_id) {
        std::set<int> newly_alarmed_tids;
        std::set<int> current_tids;
        for (const auto &d: dets) {
            current_tids.insert(d.id);
            if (alarmed_tids_.count(d.id)) continue;
            cv::Point2f current_point = get_foot_point(d.tlwh);
            int current_side = get_point_side(current_point, line_start_, line_end_);
            auto it = track_side_history_.find(d.id);
            if (it != track_side_history_.end()) {
                int last_side = it->second;
                if (last_side != 0 && current_side != 0 && current_side != last_side) {
                    bool crossed = (direction_ == "any") ||
                                   (direction_ == "in" && last_side < 0 && current_side > 0) ||
                                   (direction_ == "out" && last_side > 0 && current_side < 0);
                    if (crossed) {
                        alarmed_tids_.insert(d.id);
                        newly_alarmed_tids.insert(d.id);
                    }
                }
            }
            track_side_history_[d.id] = current_side;
        }
        std::vector<int> disappeared_tids;
        for (const auto &pair: track_side_history_) {
            if (current_tids.find(pair.first) == current_tids.end()) {
                disappeared_tids.push_back(pair.first);
            }
        }
        for (int tid: disappeared_tids) {
            track_side_history_.erase(tid);
            alarmed_tids_.erase(tid);
        }
        return newly_alarmed_tids;
    }

private:
    cv::Point2f line_start_, line_end_;
    std::string direction_;
    std::map<int, int> track_side_history_;
    std::set<int> alarmed_tids_;
};

struct TrackAgg {
    // ======================= 【MODIFIED】 =======================
    void add_body(const std::vector<float> &feat, float score, const cv::Mat &patch);

    void add_face(const std::vector<float> &feat, const cv::Mat &patch);

    std::pair<std::vector<float>, cv::Mat> main_body_feat_and_patch() const;

    std::pair<std::vector<float>, cv::Mat> main_face_feat_and_patch() const;

    std::vector<cv::Mat> body_patches() const;

    std::vector<cv::Mat> face_patches() const;

    static bool check_consistency(const std::deque<std::vector<float>> &feats, float thr = 0.5f);

    std::deque<std::tuple<std::vector<float>, float, cv::Mat>> body;
    std::deque<std::tuple<std::vector<float>, cv::Mat>> face;
    // ======================= 【修改结束】 =======================
};

struct GlobalID {
    std::string new_gid();

    int can_update_proto(const std::string &gid, const std::vector<float> &face_f, const std::vector<float> &body_f);

    // ======================= 【MODIFIED】 =======================
    void bind(const std::string &gid, const std::string &tid, int current_ts, const TrackAgg &agg);
    // ======================= 【修改结束】 =======================

    std::pair<std::string, float> probe(const std::vector<float> &face_f, const std::vector<float> &body_f);

    int gid_next = 1;
    std::map<std::string, std::vector<std::vector<float>>> bank_faces;
    std::map<std::string, std::vector<std::vector<float>>> bank_bodies;
    std::map<std::string, std::vector<std::string>> tid_hist;
    std::map<std::string, int> last_update;
};

struct CandidateState {
    std::string cand_gid = "";
    int count = 0;
    int last_bind_fid = 0;
};

struct NewGidState {
    int count = 0;
    int last_new_fid = -NEW_GID_TIME_WINDOW;
    int ambig_count = 0;
};

class FeatureProcessor {
public:
    explicit FeatureProcessor(const std::string &mode,
                              const std::string &device = "cuda",
                              const std::string &feature_cache_path = "",
                              const nlohmann::json &boundary_config = {});

    ~FeatureProcessor();

    std::map<std::string, std::map<int, std::tuple<std::string, float, int>>>
    process_packet(const Packet &pkt, const std::vector<Face> &face_info);

private:
    void _extract_features_realtime(const Packet &pkt, const std::vector<Face> &face_info);

    void _load_features_from_cache(const Packet &pkt);

    std::vector<float> _fuse_feat(const std::vector<float> &face_f, const std::vector<float> &body_f);

    std::vector<float> _gid_fused_rep(const std::string &gid);

    // ======================= 【MODIFIED】 =======================
    void trigger_alarm(const std::string &gid, const TrackAgg &agg);
    // ======================= 【修改结束】 =======================

    std::string mode_;
    std::string device_;
    std::string feature_cache_path_;
    nlohmann::json features_cache_;
    nlohmann::json features_to_save_;

    std::unique_ptr<PersonReid> reid_model_;
    std::unique_ptr<FaceAnalyzer> face_analyzer_;

    std::map<std::string, TrackAgg> agg_pool;
    GlobalID gid_mgr;
    std::map<std::string, std::string> tid2gid;
    std::unordered_map<std::string, int> last_seen;
    std::unordered_map<std::string, CandidateState> candidate_state;
    std::unordered_map<std::string, NewGidState> new_gid_state;
    std::set<std::string> alarmed;
    std::map<std::string, std::vector<float>> alarm_reprs;
    std::map<std::string, std::tuple<int, std::string>> behavior_alarm_state;
    std::map<std::string, IntrusionDetector> intrusion_detectors;
    std::map<std::string, LineCrossingDetector> line_crossing_detectors;
};