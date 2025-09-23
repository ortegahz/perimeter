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
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <queue>
#include <optional>

#include <sqlite3.h>
// MODIFIED HERE: Include the correct header
#include "cores/personReid/PersonReid_dla.hpp"
#include <cstdint> // For uint64_t

// ======================= 【MODIFIED】 =======================
// 定义 `GstClockTime` 类型别名，使其在模块内自包含
using GstClockTime = uint64_t;
// ======================= 【修改结束】 =======================
#include "cores/face/FaceAnalyzer_dla.hpp"

/* ---------- 常量定义 ---------- */
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
constexpr int GID_MAX_IDLE_FRAMES = 25 / 2 * 60 * 60 * 24;
constexpr int WAIT_FRAMES_AMBIGUOUS = 10;

constexpr double FPS_ESTIMATE = 25.0 / 2.;
constexpr double MAX_TID_IDLE_SEC = MAX_TID_IDLE_FRAMES / FPS_ESTIMATE;
constexpr double GID_MAX_IDLE_SEC = GID_MAX_IDLE_FRAMES / FPS_ESTIMATE;

constexpr int ALARM_CNT_TH = 2;
constexpr float ALARM_DUP_THR = 0.4f;
constexpr float FUSE_W_FACE = 0.6f;
constexpr float FUSE_W_BODY = 0.4f;
constexpr int EMB_FACE_DIM = 512;
constexpr int EMB_BODY_DIM = 2048;
constexpr int BEHAVIOR_ALARM_DURATION_FRAMES = 256;

constexpr float MIN_HW_RATIO = 1.5f;
constexpr float FACE_DET_MIN_SCORE = 0.60f;

//const std::string SAVE_DIR = "/mnt/nfs/perimeter_cpp";
//const std::string ALARM_DIR = "/mnt/nfs/perimeter_alarm_cpp";
//const std::string DB_PATH = "/mnt/nfs/perimeter_data.db";

#define SAVE_DIR "/mnt/nfs/perimeter_cpp"
#define ALARM_DIR "/mnt/nfs/perimeter_alarm_cpp"
#define DB_PATH "/mnt/nfs/perimeter_data.db"

//#define SAVE_DIR "/home/nvidia/perimeter_cpp"
//#define ALARM_DIR "/home/nvidia/perimeter_alarm_cpp"
//#define DB_PATH "/home/nvidia/perimeter_data.db"

/* ---------- 可调参数结构体 ---------- */
struct ProcessConfig {
    // Key: cam_id, Value: match_thr for that camera.
    std::map<std::string, float> match_thr_by_cam;
    // Global alarm count threshold.
    int alarm_cnt_th = ALARM_CNT_TH;
};

/* ---------- 数据结构定义 ---------- */
struct Detection {
    cv::Rect2f tlwh;
    float score;
    int id;
    int class_id = 0;
};

struct Packet {
    std::string cam_id;
    uint64_t fid;
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

/**
 * @brief 包含一次告警触发时的详细上下文信息。
 */
struct AlarmTriggerInfo {
    std::string gid;                // 触发告警的GID
    double first_seen_timestamp = 0.0; // 新增：GID首次识别的时间戳(秒)
    cv::Rect2f person_bbox;         // 当前帧中该GID关联的行人框
    cv::Rect2d face_bbox;           // 当前帧中该GID关联的人脸框 (如果找到)
    cv::Mat latest_body_patch;      // GID库中最新的（或最具代表性的）行人图块
    cv::Mat latest_face_patch;      // GID库中最新的（或最具代表性的）人脸图块
};

struct TrackAgg {
    void add_body(const std::vector<float> &feat, float score, const cv::Mat &patch);

    void add_face(const std::vector<float> &feat, const cv::Mat &patch);

    std::pair<std::vector<float>, cv::Mat> main_body_feat_and_patch() const;

    std::pair<std::vector<float>, cv::Mat> main_face_feat_and_patch() const;

    std::vector<cv::Mat> body_patches() const;

    std::vector<cv::Mat> face_patches() const;

    static bool check_consistency(const std::deque<std::vector<float>> &feats, float thr = 0.5f);

    std::deque<std::tuple<std::vector<float>, float, cv::Mat>> body;
    std::deque<std::tuple<std::vector<float>, cv::Mat>> face;
};

struct GlobalID {
    std::string new_gid();

    int can_update_proto(const std::string &gid, const std::vector<float> &face_f, const std::vector<float> &body_f);

    void bind(const std::string &gid, const std::string &tid, double current_ts, const TrackAgg &agg,
              class FeatureProcessor *fp);

    std::pair<std::string, float> probe(const std::vector<float> &face_f, const std::vector<float> &body_f);

    int gid_next = 1;
    std::map<std::string, std::vector<std::vector<float>>> bank_faces;
    std::map<std::string, std::vector<std::vector<float>>> bank_bodies;
    std::map<std::string, std::vector<std::string>> tid_hist;
    std::map<std::string, double> last_update;
    std::map<std::string, double> first_seen_ts; // 新增：记录每个 GID 首次出现的时间戳
};

struct CandidateState {
    std::string cand_gid = "";
    int count = 0;
    uint64_t last_bind_fid = 0;
};

struct NewGidState {
    int count = 0;
    uint64_t last_new_fid = 0;
    int ambig_count = 0;
};

enum class IoTaskType {
    CREATE_DIRS,             // 初始化环境
    SAVE_PROTOTYPE,          // 保存一个新的原型
    UPDATE_PROTOTYPE,        // 更新已有原型
    REMOVE_FILES,
    BACKUP_ALARM,
    CLEANUP_GID_DIR,
    SAVE_ALARM_INFO,         // 保存报警的文本信息
    SAVE_ALARM_CONTEXT_IMAGES // 保存报警的相关图片
};

struct IoTask {
    IoTaskType type;
    std::string gid;

    // 新增: 用于 SAVE_ALARM_CONTEXT_IMAGES
    cv::Mat full_frame_bgr;
    cv::Mat latest_body_patch_rgb;
    cv::Mat latest_face_patch_rgb;
    cv::Rect2f person_bbox;
    cv::Rect2d face_bbox;

    // 用于 SAVE_PROTOTYPE/UPDATE_PROTOTYPE
    cv::Mat image;
    std::vector<float> feature;
    std::string path_suffix;
    std::vector<std::string> files_to_remove;

    // 通用字段
    std::string timestamp;

    // 用于 SAVE_ALARM_INFO
    std::string alarm_info_content;

    // 用于 BACKUP_ALARM
    std::vector<cv::Mat> face_patches_backup;
    std::vector<cv::Mat> body_patches_backup;
};

/**
 * @brief Holds the result of a single person Re-ID extraction from the worker thread.
 */
struct ReidResult {
    std::string tid_str;
    std::vector<float> body_feat;
    cv::Mat patch;
    float det_score;
};

/**
 * @brief Task structure for the Re-ID worker thread.
 */
using ReidResults = std::vector<ReidResult>;
struct ReidTask {
    const cv::cuda::GpuMat *full_frame;
    const std::vector<Detection> *dets;
    std::string cam_id;
    std::promise<ReidResults> promise; // Requires <future>
};

// ======================= 【MODIFIED】 =======================
// 新增: 为 process_packet 定义统一的输入结构体
struct ProcessInput {
    const std::string &cam_id;
    uint64_t fid;
    GstClockTime timestamp; // 修改为 GstClockTime (uint64_t)
    const cv::cuda::GpuMat &full_frame;
    const std::vector<Detection> &dets;
    const ProcessConfig &config;
};
// ======================= 【修改结束】 =======================

struct ProcessOutput {
    // 原有的实时匹配结果
    std::map<std::string, std::map<int, std::tuple<std::string, float, int>>> mp;
    // 新增的告警信息列表 (通常为空，仅在触发新告警的帧中包含元素)
    std::vector<AlarmTriggerInfo> alarms;
};

class FeatureProcessor {
public:
    // ======================= 【MODIFIED】 =======================
    // 修改构造函数：模型路径为必要参数，其他参数为可选
    explicit FeatureProcessor(const std::string &reid_model_path,
                              const std::string &face_det_model_path,
                              const std::string &face_rec_model_path,
                              const std::string &mode = "realtime",
                              const std::string &device = "dla",
                              const std::string &feature_cache_path = "",
                              const nlohmann::json &boundary_config = {},
                              bool use_fid_time = false,
                              bool enable_alarm_saving = true,
                              bool processing_enabled = true);
    // ======================= 【修改结束】 =======================

    ~FeatureProcessor();

    // 修改: 使用新的 ProcessInput 结构体作为唯一参数
    ProcessOutput process_packet(const ProcessInput &input);

    void submit_io_task(IoTask task);

    // ======================= 【MODIFIED】 =======================
    // 新增：用于调试和验证的函数
    void save_final_state_to_file(const std::string &filepath);
    // ======================= 【修改结束】 =======================

private:

    void _load_features_from_cache(const std::string &cam_id, uint64_t fid, const cv::cuda::GpuMat &full_frame,
                                   const std::vector<Detection> &dets, double now_stamp);

    std::vector<float> _fuse_feat(const std::vector<float> &face_f, const std::vector<float> &body_f);

    std::vector<float> _gid_fused_rep(const std::string &gid);

    std::optional<std::string> trigger_alarm(const std::string &gid, const TrackAgg &agg, double frame_timestamp);

    // Re-ID worker thread
    void _reid_worker();

    // I/O线程相关
    void _io_worker();

    void _init_db();

    void _close_db();

    std::thread io_thread_;
    std::queue<IoTask> io_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cond_;
    std::atomic<bool> stop_io_thread_{false};

    // Re-ID线程相关
    std::thread reid_thread_;
    std::queue<ReidTask> reid_queue_;
    std::mutex reid_queue_mutex_;
    std::condition_variable reid_queue_cond_;
    std::atomic<bool> stop_reid_thread_{false};

    // 数据库句柄
    sqlite3 *db_ = nullptr;

    std::string mode_;
    bool use_fid_time_;
    std::string device_;
    std::string feature_cache_path_;
    nlohmann::json features_cache_;
    nlohmann::json features_to_save_;

    // 新增: 存储模型路径的成员变量
    std::string m_reid_model_path;
    std::string m_face_det_model_path;
    std::string m_face_rec_model_path;
    bool m_enable_alarm_saving;
    bool m_processing_enabled;

    // MODIFIED HERE: Changed from PersonReid to PersonReidDLA
    std::unique_ptr<PersonReidDLA> reid_model_;
    std::unique_ptr<PersonReidDLA> reid_model_dla1_; // For the worker thread on DLA1
    std::unique_ptr<FaceAnalyzer> face_analyzer_;

    std::map<std::string, TrackAgg> agg_pool;
    GlobalID gid_mgr;
    std::map<std::string, std::string> tid2gid;
    std::unordered_map<std::string, double> last_seen;
    std::unordered_map<std::string, CandidateState> candidate_state;
    std::unordered_map<std::string, NewGidState> new_gid_state;
    std::set<std::string> alarmed;
    std::map<std::string, std::vector<float>> alarm_reprs;
    std::map<std::string, std::tuple<uint64_t, std::string>> behavior_alarm_state;
    std::map<std::string, std::unique_ptr<IntrusionDetector>> intrusion_detectors;
    std::map<std::string, std::unique_ptr<LineCrossingDetector>> line_crossing_detectors;
    std::map<std::string, cv::Rect2d> current_frame_face_boxes_; // 临时存储TID-FaceBbox映射
};