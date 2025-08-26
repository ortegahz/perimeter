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
constexpr int GID_MAX_IDLE_FRAMES = 25 * 60; // 简化版: 25fps * 60s
constexpr int WAIT_FRAMES_AMBIGUOUS = 10;

constexpr int ALARM_CNT_TH = 8;
constexpr float ALARM_DUP_THR = 0.4f;
constexpr float FUSE_W_FACE = 0.6f;
constexpr float FUSE_W_BODY = 0.4f;
constexpr int EMB_FACE_DIM = 512;
constexpr int EMB_BODY_DIM = 2048;

const std::string SAVE_DIR = "/home/manu/tmp/perimeter_cpp";
const std::string ALARM_DIR = "/home/manu/tmp/perimeter_alarm_cpp";
/* ------------------------------------------------ */

/* ---------- 新增：数据结构定义 ---------- */
struct Detection {
    cv::Rect2f tlwh;
    float score;
    int id;
};

struct Packet {
    std::string cam_id;
    int fid;
    std::vector<cv::Mat> patches;
    std::vector<Detection> dets;
};

/* ---------- 占位符：模拟模型封装 (用您的真实模型类替换) ---------- */
struct PersonReid {
    // 示例接口：输入一批图像，返回一批特征
    std::vector<std::vector<float>> infer(const std::vector<cv::Mat> &patches) {
        // TODO: 在此实现您的 ReID 模型推理
        std::vector<std::vector<float>> dummy_feats;
        for (const auto &p: patches) {
            dummy_feats.push_back(std::vector<float>(EMB_BODY_DIM, 0.1f));
        }
        return dummy_feats;
    }
};

struct FaceSearcher {
    struct FaceObject {
        std::vector<float> embedding;
        float det_score = 1.0f;
    };

    // 示例接口：输入单张图，返回检测到的人脸
    std::vector<FaceObject> get(const cv::Mat &patch) {
        // TODO: 在此实现您的人脸检测和特征提取模型推理
        return {};
    }
};

/* ---------- TrackAgg, GlobalID 等结构体 (保持不变) ---------- */
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

/* ---------- FeatureProcessor 类定义 ---------- */
class FeatureProcessor {
public:
    explicit FeatureProcessor(const std::string &mode,
                              const std::string &device = "cuda",
                              const std::string &feature_cache_path = "");

    ~FeatureProcessor();

    std::map<std::string, std::map<int, std::tuple<std::string, float, int>>>
    process_packet(const Packet &pkt);

private:
    void _extract_features_realtime(const Packet &pkt);

    void _load_features_from_cache(const Packet &pkt);

    std::vector<float> _fuse_feat(const std::vector<float> &face_f, const std::vector<float> &body_f);

    std::vector<float> _gid_fused_rep(const std::string &gid);

    void trigger_alarm(const std::string &gid);

    // --- 成员变量 ---
    std::string mode_;
    std::string device_;
    std::string feature_cache_path_;

    nlohmann::json features_cache_;      // 用于 load 模式
    nlohmann::json features_to_save_;    // 用于 realtime 模式

    std::unique_ptr<PersonReid> reid_model_;
    std::unique_ptr<FaceSearcher> face_app_;

    std::map<std::string, TrackAgg> agg_pool;
    GlobalID gid_mgr;
    std::map<std::string, std::string> tid2gid;
    std::unordered_map<std::string, int> last_seen;
    std::unordered_map<std::string, CandidateState> candidate_state;
    std::unordered_map<std::string, NewGidState> new_gid_state;
    std::set<std::string> alarmed;
    std::map<std::string, std::vector<float>> alarm_reprs;
};