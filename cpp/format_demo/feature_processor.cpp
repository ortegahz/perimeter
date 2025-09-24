#include "feature_processor.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <iostream>
#include <chrono>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iomanip> // For sprintf
#include <optional>
#include <sstream>

// ======================= 【MODIFIED】 =======================
// 添加一个宏来控制所有磁盘I/O操作
// 如果要关闭所有耗时的文件写入和删除，请注释掉下面这行
//#define ENABLE_DISK_IO
// ======================= 【修改结束】 =======================

const int REID_INPUT_WIDTH = 128;
const int REID_INPUT_HEIGHT = 256;

static bool is_long_patch(const cv::Mat &patch, float thr = MIN_HW_RATIO) {
    if (patch.empty()) return false;
    return (float) patch.rows / ((float) patch.cols + 1e-9f) >= thr;
}

static float sim_vec(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.empty() || b.empty() || a.size() != b.size()) return 0.f;
    float s = 0.f;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

static std::vector<float> avg_feats(const std::vector<std::vector<float>> &feats) {
    if (feats.empty()) return {};
    std::vector<float> mean(feats[0].size(), 0.f);
    for (const auto &f: feats) for (size_t i = 0; i < f.size(); ++i) mean[i] += f[i];
    float num_feats = static_cast<float>(feats.size());
    for (float &v: mean) v /= num_feats;
    float n = 1e-9f;
    for (float v: mean) n += v * v;
    n = std::sqrt(n);
    if (n > 1e-9f) for (float &v: mean) v /= n;
    return mean;
}

static std::vector<float> blend(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) return {};
    std::vector<float> r(a.size());
    float n = 1e-9f;
    for (size_t i = 0; i < a.size(); ++i) {
        r[i] = 0.7f * a[i] + 0.3f * b[i];
        n += r[i] * r[i];
    }
    n = std::sqrt(n);
    if (n > 1e-9f) for (float &v: r) v /= n;
    return r;
}

static float vec_dist(const std::vector<float> &a, const std::vector<float> &b) {
    float d_sq = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        d_sq += diff * diff;
    }
    return std::sqrt(d_sq);
}

static std::pair<float, float> mean_stddev(const std::vector<float> &data) {
    if (data.size() < 2) return {data.empty() ? 0.f : data[0], 0.f};
    if (data.empty()) return {0.f, 0.f};
    float sum = std::accumulate(data.begin(), data.end(), 0.0f);
    float mean = sum / data.size();
    float sq_sum = 0.f;
    for (const auto &val: data) { sq_sum += (val - mean) * (val - mean); }
    float stddev = std::sqrt(sq_sum / data.size());
    return {mean, stddev};
}

static float calculate_ioa(const cv::Rect2f &person_tlwh, const cv::Rect2d &face_xyxy) {
    cv::Rect2f person_rect = person_tlwh;
    cv::Rect2f face_rect(face_xyxy);
    cv::Rect2f intersection = person_rect & face_rect;
    if (intersection.area() <= 0 || face_rect.area() <= 0) return 0.0f;
    return intersection.area() / face_rect.area();
}

static std::pair<std::vector<std::vector<float>>, std::vector<bool>>
remove_outliers_cpp(const std::vector<std::vector<float>> &embeddings, float thresh) {
    size_t n = embeddings.size();
    if (n < 3) return {embeddings, std::vector<bool>(n, true)};
    auto mean_vec = avg_feats(embeddings);
    std::vector<float> dists;
    dists.reserve(n);
    for (const auto &emb: embeddings) dists.push_back(vec_dist(emb, mean_vec));
    auto [dist_mean, dist_std] = mean_stddev(dists);
    std::vector<bool> keep_mask(n, true);
    std::vector<std::vector<float>> new_list;
    float std_dev_safe = dist_std + 1e-8f;

    for (size_t i = 0; i < n; ++i) {
        float z_score = (dists[i] - dist_mean) / std_dev_safe;
        if (std::abs(z_score) < thresh) {
            new_list.push_back(embeddings[i]);
        } else {
            keep_mask[i] = false;
        }
    }
    if (new_list.empty()) return {embeddings, std::vector<bool>(n, true)};
    return {new_list, keep_mask};
}

template<typename T>
static std::pair<std::vector<float>, cv::Mat>
main_representation_with_patch_cpp(const T &data_deque, float outlier_thr = 1.5f) {
    if (data_deque.empty())
        return {{},
                {}};

    std::vector<std::vector<float>> feats;
    std::vector<cv::Mat> patches;
    for (const auto &item: data_deque) {
        feats.push_back(std::get<0>(item));
        if constexpr (std::tuple_size_v<typename T::value_type> == 3) {
            patches.push_back(std::get<2>(item));
        } else {
            patches.push_back(std::get<1>(item));
        }
    }

    auto initial_mean = avg_feats(feats);
    std::vector<float> dists;
    dists.reserve(feats.size());
    for (const auto &f: feats) { dists.push_back(vec_dist(f, initial_mean)); }
    auto [dist_mean, dist_std] = mean_stddev(dists);
    float keep_thresh = dist_mean + outlier_thr * dist_std;

    std::vector<std::vector<float>> kept_feats;
    std::vector<cv::Mat> kept_patches;
    for (size_t i = 0; i < feats.size(); ++i) {
        if (dists[i] < keep_thresh) {
            kept_feats.push_back(feats[i]);
            kept_patches.push_back(patches[i]);
        }
    }

    if (kept_feats.empty()) {
        kept_feats = feats;
        kept_patches = patches;
    }

    auto final_mean = avg_feats(kept_feats);
    int best_idx = -1;
    float best_sim = -2.0f;
    for (size_t i = 0; i < kept_feats.size(); ++i) {
        float s = sim_vec(kept_feats[i], final_mean);
        if (s > best_sim) {
            best_sim = s;
            best_idx = i;
        }
    }

    if (best_idx != -1) {
        return {kept_feats[best_idx], kept_patches[best_idx]};
    }
    return {{},
            {}};
}

/* ================= TrackAgg ================= */
bool TrackAgg::check_consistency(const std::deque<std::vector<float>> &feats, float thr) {
    if (feats.size() < 2) return true;
    std::vector<float> sims;
    for (size_t i = 0; i < feats.size(); ++i)
        for (size_t j = i + 1; j < feats.size(); ++j)
            sims.push_back(sim_vec(feats[i], feats[j]));
    if (sims.empty()) return true;
    float m = std::accumulate(sims.begin(), sims.end(), 0.0f) / sims.size();
    return (1.f - m) <= thr;
}

void TrackAgg::add_body(const std::vector<float> &feat, float score, const cv::Mat &patch) {
    if (patch.empty()) return;
    body.emplace_back(feat, score, patch.clone());
    if (body.size() > MIN_BODY4GID) body.pop_front();
}

void TrackAgg::add_face(const std::vector<float> &feat, const cv::Mat &patch) {
    if (patch.empty()) return;
    face.emplace_back(feat, patch.clone());
    if (face.size() > MIN_FACE4GID) face.pop_front();
}

std::pair<std::vector<float>, cv::Mat> TrackAgg::main_body_feat_and_patch() const {
    if (body.empty())
        return {{},
                {}};
    std::deque<std::vector<float>> feats;
    for (const auto &t: body) feats.push_back(std::get<0>(t));
    if (!check_consistency(feats, 0.5f))
        return {{},
                {}};
    return main_representation_with_patch_cpp(body);
}

std::pair<std::vector<float>, cv::Mat> TrackAgg::main_face_feat_and_patch() const {
    if (face.empty())
        return {{},
                {}};
    std::deque<std::vector<float>> feats;
    for (const auto &t: face) feats.push_back(std::get<0>(t));
    if (!check_consistency(feats))
        return {{},
                {}};
    return main_representation_with_patch_cpp(face);
}

std::vector<cv::Mat> TrackAgg::body_patches() const {
    std::vector<cv::Mat> patches;
    for (const auto &item: body) patches.push_back(std::get<2>(item).clone());
    return patches;
}

std::vector<cv::Mat> TrackAgg::face_patches() const {
    std::vector<cv::Mat> patches;
    for (const auto &item: face) patches.push_back(std::get<1>(item).clone());
    return patches;
}

/* ================= GlobalID ================= */
static void _add_or_update_prototype(
        std::vector<std::vector<float>> &feat_list,
        const std::vector<float> &new_feat,
        const cv::Mat &new_patch,
        const std::string &gid,
        const std::string &type, // "faces" or "bodies"
        FeatureProcessor *fp) {

    if (new_feat.empty() || new_patch.empty()) return;

    if (!feat_list.empty()) {
        float max_sim = -1.f;
        for (const auto &x: feat_list) max_sim = std::max(max_sim, sim_vec(new_feat, x));
        if (max_sim < UPDATE_THR) return;
    }

    bool is_new_proto = false;
    int idx_to_replace = -1;
    if ((int) feat_list.size() < MAX_PROTO_PER_TYPE) {
        idx_to_replace = feat_list.size();
        feat_list.push_back(new_feat);
    } else {
        float best_sim = -1.f;
        for (int i = 0; i < (int) feat_list.size(); ++i) {
            float s = sim_vec(new_feat, feat_list[i]);
            if (s > best_sim) {
                best_sim = s;
                idx_to_replace = i;
            }
        }
        if (idx_to_replace != -1) {
            feat_list[idx_to_replace] = blend(feat_list[idx_to_replace], new_feat);
        }
    }

    if (idx_to_replace != -1) {
        char filename[32];
        sprintf(filename, "%02d.jpg", idx_to_replace);
        IoTask task;
        task.type = is_new_proto ? IoTaskType::SAVE_PROTOTYPE
                                 : IoTaskType::UPDATE_PROTOTYPE; // is_new_proto is now defined
        task.gid = gid;
        task.feature = feat_list[idx_to_replace]; // The feature is already updated/added at this index
        task.path_suffix = std::string(type) + "/" + filename;
        task.image = new_patch.clone(); // Deep copy for thread safety
        fp->submit_io_task(task);
    }

    // 保留了离群点检测，但只提交删除任务
    auto original_size = feat_list.size();
    auto [new_lst, keep_mask] = remove_outliers_cpp(feat_list, 3.0f);
    if (new_lst.size() != original_size) {
        std::cout << "[GlobalID] Outlier detected: " << original_size - new_lst.size() << " from GID " << gid << "/"
                  << type << std::endl;

        std::vector<std::string> files_to_del;
        // 注意：这里的索引逻辑需要基于原始列表，而不是已删除的列表
        for (size_t i = 0; i < keep_mask.size(); ++i) {
            if (!keep_mask[i]) {
                char filename[32];
                sprintf(filename, "%02zu.jpg", i);
                files_to_del.push_back((std::filesystem::path(SAVE_DIR) / gid / type / filename).string());
            }
        }
        if (!files_to_del.empty()) {
            IoTask task;
            task.type = IoTaskType::REMOVE_FILES;
            task.files_to_remove = files_to_del;
            fp->submit_io_task(task);
        }
        feat_list = new_lst;
    }
}

std::string GlobalID::new_gid() {
    char buf[32];
    sprintf(buf, "G%05d", gid_next++);
    std::string gid(buf);
    bank_faces[gid] = {};
    bank_bodies[gid] = {};
    tid_hist[gid] = {};
    last_update[gid] = 0;
    std::cout << "[GlobalID] new " << gid << std::endl;
    return gid;
}

int
GlobalID::can_update_proto(const std::string &gid, const std::vector<float> &face_f, const std::vector<float> &body_f) {
    if (!bank_faces.count(gid)) return 0;
    if (!bank_faces[gid].empty() && !face_f.empty() && sim_vec(face_f, avg_feats(bank_faces[gid])) < FACE_THR_STRICT)
        return -1;
    if (!bank_bodies.count(gid) || (bank_bodies.count(gid) && !bank_bodies[gid].empty() && !body_f.empty() &&
                                    sim_vec(body_f, avg_feats(bank_bodies[gid])) < BODY_THR_STRICT))
        return -2;
    return 0;
}

void GlobalID::bind(const std::string &gid, const std::string &tid, double current_ts, GstClockTime current_ts_gst,
                    const TrackAgg &agg, FeatureProcessor *fp) {
    auto [face_f, face_p] = agg.main_face_feat_and_patch();
    auto [body_f, body_p] = agg.main_body_feat_and_patch();

    _add_or_update_prototype(bank_faces[gid], face_f, face_p, gid, "faces", fp);
    _add_or_update_prototype(bank_bodies[gid], body_f, body_p, gid, "bodies", fp);

    auto &v = tid_hist[gid];
    if (std::find(v.begin(), v.end(), tid) == v.end()) v.push_back(tid);
    last_update[gid] = current_ts;

    // 新增：如果这是第一次绑定该 GID，记录其首次出现的时间戳
    if (first_seen_ts.find(gid) == first_seen_ts.end()) {
        first_seen_ts[gid] = current_ts_gst;
    }
}

std::pair<std::string, float> GlobalID::probe(const std::vector<float> &face_f, const std::vector<float> &body_f,
                                              float w_face, float w_body) {
    std::string best_gid;
    float best_score = -1.f;
    for (auto const &[gid, face_pool]: bank_faces) {
        if (!bank_bodies.count(gid) || face_pool.empty() || bank_bodies.at(gid).empty()) continue;
        float face_sim = face_f.empty() ? 0.f : sim_vec(face_f, avg_feats(face_pool));
        float body_sim = body_f.empty() ? 0.f : sim_vec(body_f, avg_feats(bank_bodies.at(gid)));
        float sc = w_face * face_sim + w_body * body_sim;
        if (sc > best_score) {
            best_score = sc;
            best_gid = gid;
        }
    }
    return {best_gid, best_score};
}

// 修改: 更新构造函数以匹配新的签名，并使用成员变量存储路径
FeatureProcessor::FeatureProcessor(const std::string &reid_model_path,
                                   const std::string &face_det_model_path,
                                   const std::string &face_rec_model_path,
                                   const std::string &mode,
                                   const std::string &device,
                                   const std::string &feature_cache_path,
                                   const nlohmann::json &boundary_config,
                                   bool use_fid_time,
                                   bool enable_alarm_saving,
                                   bool processing_enabled)
        : m_reid_model_path(reid_model_path),
          m_face_det_model_path(face_det_model_path),
          m_face_rec_model_path(face_rec_model_path),
          mode_(mode), use_fid_time_(use_fid_time), device_(device),
          feature_cache_path_(feature_cache_path),
          m_enable_alarm_saving(enable_alarm_saving),
          m_processing_enabled(processing_enabled) {
    std::cout << "FeatureProcessor initialized in '" << mode_ << "' mode. Alarm saving is "
              << (m_enable_alarm_saving ? "ENABLED" : "DISABLED") << "." << std::endl;

    if (mode_ == "realtime") {
        std::cout << "Loading ReID and Face models for feature extraction..." << std::endl;
        bool use_gpu = (device == "cuda"); // Note: DLA is Jetson specific, this flag is less relevant here.
        try {
            // 初始化将在主线程中使用的 ReID 模型，分配给 DLA Core 1
            reid_model_ = std::make_unique<PersonReidDLA>(m_reid_model_path, REID_INPUT_WIDTH, REID_INPUT_HEIGHT,
                                                          1, "/home/nvidia/VSCodeProject/smartboxcore/models/reid_model.dla.engine");
            std::cout << "Initialized Re-ID model on DLA 1 for main thread." << std::endl;
            face_analyzer_ = std::make_unique<FaceAnalyzer>(m_face_det_model_path, m_face_rec_model_path);
            std::string provider = use_gpu ? "GPU" : "DLA";
            // Dedicate DLA core 0 to the FaceAnalyzer.
            face_analyzer_->prepare(provider, FACE_DET_MIN_SCORE, cv::Size(640, 640));
        } catch (const std::exception &e) {
            std::cerr << "[FATAL] Failed to load models in realtime mode: " << e.what() << std::endl;
            throw;
        }

        // The feature cache can optionally be written in realtime mode.
        if (!feature_cache_path_.empty()) {
            auto parent_path = std::filesystem::path(feature_cache_path_).parent_path();
            if (!parent_path.empty()) std::filesystem::create_directories(parent_path);
        }
    } else if (mode_ == "load") {
        // In 'load' mode, no models are needed. We just load features from the cache.
        if (feature_cache_path_.empty() || !std::filesystem::exists(feature_cache_path_)) {
            throw std::runtime_error("In 'load' mode, a valid feature_cache_path is required: " + feature_cache_path_);
        }
        std::ifstream jf(feature_cache_path_);
        jf >> features_cache_;
    } else {
        throw std::invalid_argument("Invalid mode: " + mode_ + ". Choose 'realtime' or 'load'.");
    }

    // ... (boundary config and other setup remains the same)
    if (!boundary_config.is_null()) {
        for (auto const &[stream_id, config]: boundary_config.items()) {
            if (config.contains("intrusion_poly")) {
                std::vector<cv::Point> poly;
                for (const auto &pt_json: config["intrusion_poly"]) {
                    poly.emplace_back(pt_json[0].get<int>(), pt_json[1].get<int>());
                }
                intrusion_detectors[stream_id] = std::make_unique<IntrusionDetector>(poly);
                std::cout << "Initialized IntrusionDetector for stream '" << stream_id << "'." << std::endl;
            }
            if (config.contains("crossing_line")) {
                const auto &line_cfg = config["crossing_line"];
                cv::Point start(line_cfg["start"][0].get<int>(), line_cfg["start"][1].get<int>());
                cv::Point end(line_cfg["end"][0].get<int>(), line_cfg["end"][1].get<int>());
                std::string direction = line_cfg.value("direction", "any");
                line_crossing_detectors[stream_id] = std::make_unique<LineCrossingDetector>(start, end, direction);
                std::cout << "Initialized LineCrossingDetector for stream '" << stream_id << "'." << std::endl;
            }
        }
    }

    // ======================= 【MODIFIED: 初始化逻辑变更】 =======================
    bool db_existed_before_init = std::filesystem::exists(DB_PATH);
    _init_or_load_db();

    // 新增: 如果数据库存在并已加载，则立即保存内存状态以供验证
    if (db_existed_before_init) {
        save_final_state_to_file("/mnt/nfs/state_after_load_in_processor.txt");
    }

    submit_io_task({IoTaskType::CREATE_DIRS}); // 移到DB初始化后
    // ======================= 【修改结束】 =======================

    // Start worker threads
    io_thread_ = std::thread(&FeatureProcessor::_io_worker, this);
}
// ======================= 【修改结束】 =======================

// ======================= 【MODIFIED】 =======================
FeatureProcessor::~FeatureProcessor() {
    // 停止IO线程，Re-ID线程已被移除
    // 确保析构时，I/O线程被正确停止，数据库被安全关闭
    if (!stop_io_thread_.load()) {
        stop_io_thread_ = true;
        queue_cond_.notify_all(); // 唤醒可能正在等待的I/O线程
        if (io_thread_.joinable()) {
            io_thread_.join();
        }
        std::cout << "I/O thread finished." << std::endl;
        _close_db();
        std::cout << "Database connection closed." << std::endl;
    }
}
// ======================= 【修改结束】 =======================

void FeatureProcessor::submit_io_task(IoTask task) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        io_queue_.push(std::move(task));
    }
    queue_cond_.notify_one();
}

// ======================= 【MODIFIED: 数据库函数重构】 =======================
void FeatureProcessor::_create_db_schema() {
    char *zErrMsg = nullptr;
    const char *schema = R"(
        PRAGMA foreign_keys = ON;

        CREATE TABLE prototypes (
            gid TEXT NOT NULL,
            type TEXT NOT NULL,
            idx INTEGER NOT NULL,
            feature BLOB,
            image BLOB,
            PRIMARY KEY (gid, type, idx)
        );

        CREATE TABLE alarms (
            alarm_id INTEGER PRIMARY KEY AUTOINCREMENT,
            gid TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE alarm_patches (
            patch_id INTEGER PRIMARY KEY AUTOINCREMENT,
            alarm_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            image_data BLOB NOT NULL,
            FOREIGN KEY (alarm_id) REFERENCES alarms(alarm_id) ON DELETE CASCADE
        );
    )";

    if (sqlite3_exec(db_, schema, nullptr, nullptr, &zErrMsg) != SQLITE_OK) {
        std::string err_msg = "SQL error during schema creation: " + std::string(zErrMsg);
        sqlite3_free(zErrMsg);
        sqlite3_close(db_);
        db_ = nullptr;
        throw std::runtime_error(err_msg);
    }
    std::cout << "New database schema created successfully at " << DB_PATH << std::endl;
}

void FeatureProcessor::_init_or_load_db() {
    bool db_exists = std::filesystem::exists(DB_PATH);

    if (sqlite3_open(DB_PATH, &db_) != SQLITE_OK) {
        std::string errmsg = db_ ? sqlite3_errmsg(db_) : "Unknown SQLite error";
        throw std::runtime_error("Can't open database: " + errmsg);
    }

    if (db_exists) {
        std::cout << "Existing database found. Loading state..." << std::endl;
        _load_state_from_db();
    } else {
        std::cout << "No existing database found. Creating a new one..." << std::endl;
        _create_db_schema();
    }
}
// ======================= 【修改结束】 =======================

void FeatureProcessor::_close_db() {
    if (db_) sqlite3_close(db_);
    db_ = nullptr;
}

void FeatureProcessor::_io_worker() {
    while (true) {
        IoTask task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cond_.wait(lock, [this] { return !io_queue_.empty() || stop_io_thread_; });
            if (stop_io_thread_ && io_queue_.empty()) {
                return;
            }
            task = std::move(io_queue_.front());
            io_queue_.pop();
        }

        try {
            switch (task.type) {
                case IoTaskType::CREATE_DIRS: {
                    // ======================= 【MODIFIED: 不再删除SAVE_DIR】 =======================
                    // 不再删除 SAVE_DIR，因为可能包含从数据库加载的 GID 对应的持久化原型图像
                    if (std::filesystem::exists(ALARM_DIR)) std::filesystem::remove_all(ALARM_DIR); // 报警目录可以每次安全地清理
                    std::filesystem::create_directories(SAVE_DIR);
                    std::filesystem::create_directories(ALARM_DIR);
                    // ======================= 【修改结束】 =======================
                    break;
                }
                case IoTaskType::SAVE_PROTOTYPE:
                case IoTaskType::UPDATE_PROTOTYPE: {
                    auto full_path = std::filesystem::path(SAVE_DIR) / task.gid / task.path_suffix;
                    std::filesystem::create_directories(full_path.parent_path());

                    // 新增: task.image 是 RGB 格式, imwrite/imencode 需要 BGR 格式
                    cv::Mat bgr_image;
                    if (!task.image.empty()) {
                        cv::cvtColor(task.image, bgr_image, cv::COLOR_RGB2BGR);
                    }
                    if (!bgr_image.empty()) cv::imwrite(full_path.string(), bgr_image);
                    // --- 数据库操作 ---
                    if (db_) {
                        std::string proto_type;
                        int proto_idx = -1;
                        size_t slash_pos = task.path_suffix.find('/');
                        if (slash_pos != std::string::npos) {
                            proto_type = task.path_suffix.substr(0, slash_pos);
                            std::string idx_str = task.path_suffix.substr(slash_pos + 1);
                            try { proto_idx = std::stoi(idx_str.substr(0, idx_str.find('.'))); } catch (...) {}
                        }

                        if (proto_idx != -1) {
                            const char *sql = "INSERT OR REPLACE INTO prototypes (gid, type, idx, feature, image) VALUES (?, ?, ?, ?, ?);";
                            sqlite3_stmt *stmt;
                            if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) == SQLITE_OK) {
                                std::vector<uchar> img_buf;
                                // 使用上面转换后的 bgr_image
                                if (!bgr_image.empty()) {
                                    cv::imencode(".jpg", bgr_image, img_buf);
                                }
                                sqlite3_bind_text(stmt, 1, task.gid.c_str(), -1, SQLITE_STATIC);
                                sqlite3_bind_text(stmt, 2, proto_type.c_str(), -1, SQLITE_STATIC);
                                sqlite3_bind_int(stmt, 3, proto_idx);
                                sqlite3_bind_blob(stmt, 4, task.feature.data(), task.feature.size() * sizeof(float),
                                                  SQLITE_STATIC);
                                sqlite3_bind_blob(stmt, 5, img_buf.data(), img_buf.size(), SQLITE_STATIC);

                                if (sqlite3_step(stmt) != SQLITE_DONE) {
                                    std::cerr << "\nDB Error (SAVE/UPDATE_PROTOTYPE): " << sqlite3_errmsg(db_)
                                              << std::endl;
                                }
                                sqlite3_finalize(stmt);
                            }
                        }
                    }
                    break;
                }
                case IoTaskType::REMOVE_FILES: {
                    for (const auto &file_path_str: task.files_to_remove) {
                        std::filesystem::path file_path(file_path_str);
                        if (std::filesystem::exists(file_path)) {
                            std::filesystem::remove(file_path);
                        }

                        if (db_) {
                            try {
                                std::string idx_str = file_path.stem().string();
                                std::string type_str = file_path.parent_path().filename().string();
                                std::string gid_str = file_path.parent_path().parent_path().filename().string();
                                int proto_idx = std::stoi(idx_str);

                                const char *sql = "DELETE FROM prototypes WHERE gid = ? AND type = ? AND idx = ?;";
                                sqlite3_stmt *stmt;
                                if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) == SQLITE_OK) {
                                    sqlite3_bind_text(stmt, 1, gid_str.c_str(), -1, SQLITE_STATIC);
                                    sqlite3_bind_text(stmt, 2, type_str.c_str(), -1, SQLITE_STATIC);
                                    sqlite3_bind_int(stmt, 3, proto_idx);
                                    if (sqlite3_step(stmt) != SQLITE_DONE) {
                                        std::cerr << "\nDB Error (REMOVE_FILES): " << sqlite3_errmsg(db_) << std::endl;
                                    }
                                    sqlite3_finalize(stmt);
                                }
                            } catch (const std::exception &e) {
                                std::cerr << "\nDB Error parsing path to delete: " << file_path_str << " - " << e.what()
                                          << std::endl;
                            }
                        }
                    }
                    break;
                }
                case IoTaskType::BACKUP_ALARM: {
                    // --- 文件系统备份 ---
                    std::string dir_name = task.gid + "_" + task.tid_str + "_n" + std::to_string(task.n) + "_" + task.timestamp;
                    std::filesystem::path dst_dir = std::filesystem::path(ALARM_DIR) / dir_name;

                    std::filesystem::path src_dir = std::filesystem::path(SAVE_DIR) / task.gid;
                    if (std::filesystem::exists(src_dir))
                        std::filesystem::copy(src_dir, dst_dir, std::filesystem::copy_options::recursive |
                                                                std::filesystem::copy_options::overwrite_existing);

                    std::filesystem::path seq_face_dir = dst_dir / "agg_sequence/face";
                    std::filesystem::path seq_body_dir = dst_dir / "agg_sequence/body";
                    std::filesystem::create_directories(seq_face_dir);
                    std::filesystem::create_directories(seq_body_dir);

                    for (size_t i = 0; i < task.face_patches_backup.size(); ++i) {
                        char fname[16];
                        sprintf(fname, "%03zu.jpg", i);
                        // 新增: patch 是 RGB 格式, imwrite 需要 BGR 格式
                        cv::Mat bgr_patch;
                        cv::cvtColor(task.face_patches_backup[i], bgr_patch, cv::COLOR_RGB2BGR);
                        cv::imwrite((seq_face_dir / fname).string(), bgr_patch);
                    }
                    for (size_t i = 0; i < task.body_patches_backup.size(); ++i) {
                        char fname[16];
                        sprintf(fname, "%03zu.jpg", i);
                        // 新增: patch 是 RGB 格式, imwrite 需要 BGR 格式
                        cv::Mat bgr_patch;
                        cv::cvtColor(task.body_patches_backup[i], bgr_patch, cv::COLOR_RGB2BGR);
                        cv::imwrite((seq_body_dir / fname).string(), bgr_patch);
                    }

                    // --- 数据库备份 ---
                    if (db_) {
                        sqlite3_stmt *alarm_stmt;
                        const char *sql_alarm = "INSERT INTO alarms (gid, timestamp) VALUES (?, ?);";
                        long long alarm_db_id = -1;

                        if (sqlite3_prepare_v2(db_, sql_alarm, -1, &alarm_stmt, nullptr) == SQLITE_OK) {
                            sqlite3_bind_text(alarm_stmt, 1, task.gid.c_str(), -1, SQLITE_STATIC);
                            sqlite3_bind_text(alarm_stmt, 2, task.timestamp.c_str(), -1, SQLITE_STATIC);
                            if (sqlite3_step(alarm_stmt) == SQLITE_DONE) {
                                alarm_db_id = sqlite3_last_insert_rowid(db_);
                            } else {
                                std::cerr << "\nDB Error (BACKUP_ALARM - insert alarm): " << sqlite3_errmsg(db_)
                                          << std::endl;
                            }
                            sqlite3_finalize(alarm_stmt);
                        }

                        if (alarm_db_id != -1) {
                            const char *sql_patch = "INSERT INTO alarm_patches (alarm_id, type, image_data) VALUES (?, ?, ?);";
                            sqlite3_stmt *patch_stmt;
                            if (sqlite3_prepare_v2(db_, sql_patch, -1, &patch_stmt, nullptr) == SQLITE_OK) {
                                auto process_patches = [&](const std::vector<cv::Mat> &patches, const char *type) {
                                    for (const auto &patch: patches) {
                                        // 新增: patch 是 RGB 格式, imencode 需要 BGR 格式
                                        cv::Mat bgr_patch;
                                        std::vector<uchar> img_buf;
                                        cv::cvtColor(patch, bgr_patch, cv::COLOR_RGB2BGR);
                                        cv::imencode(".jpg", bgr_patch, img_buf);
                                        sqlite3_bind_int64(patch_stmt, 1, alarm_db_id);
                                        sqlite3_bind_text(patch_stmt, 2, type, -1, SQLITE_STATIC);
                                        sqlite3_bind_blob(patch_stmt, 3, img_buf.data(), img_buf.size(), SQLITE_STATIC);
                                        if (sqlite3_step(patch_stmt) != SQLITE_DONE) {
                                            std::cerr << "\nDB Error (BACKUP_ALARM - insert patch): "
                                                      << sqlite3_errmsg(db_) << std::endl;
                                        }
                                        sqlite3_reset(patch_stmt);
                                    }
                                };
                                process_patches(task.face_patches_backup, "face");
                                process_patches(task.body_patches_backup, "body");
                                sqlite3_finalize(patch_stmt);
                            }
                        }
                    }

                    std::cout << "\n[ALARM] GID " << task.gid << " triggered and backed up to " << dst_dir.string()
                              << std::endl;
                    break;
                }
                case IoTaskType::CLEANUP_GID_DIR: {
                    auto dir_to_del = std::filesystem::path(SAVE_DIR) / task.gid;
                    if (std::filesystem::exists(dir_to_del)) {
                        std::filesystem::remove_all(dir_to_del);

                        if (db_) {
                            sqlite3_stmt *stmt;
                            const char *sql = "DELETE FROM prototypes WHERE gid = ?;";
                            if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) == SQLITE_OK) {
                                sqlite3_bind_text(stmt, 1, task.gid.c_str(), -1, SQLITE_STATIC);
                                if (sqlite3_step(stmt) != SQLITE_DONE) {
                                    std::cerr << "\nDB Error (CLEANUP_GID_DIR): " << sqlite3_errmsg(db_) << std::endl;
                                }
                                sqlite3_finalize(stmt);
                            }
                        }
                    }
                    break;
                }
                case IoTaskType::SAVE_ALARM_INFO: {
                    std::string dir_name = task.gid + "_" + task.tid_str + "_n" + std::to_string(task.n) + "_" + task.timestamp;
                    auto dst_path = std::filesystem::path(ALARM_DIR) / dir_name / "frame_info.txt";

                    // BACKUP_ALARM 任务会创建目录，但为了安全起见，这里也确保一下
                    if (!dst_path.parent_path().empty()) {
                        std::filesystem::create_directories(dst_path.parent_path());
                    }
                    std::ofstream ofs(dst_path.string());
                    if (ofs.is_open()) {
                        ofs << task.alarm_info_content;
                    } else {
                        std::cerr << "\nI/O worker error: Could not open " << dst_path.string() << " for writing."
                                  << std::endl;
                    }
                    break;
                }
                case IoTaskType::SAVE_ALARM_CONTEXT_IMAGES: {
                    std::string dir_name = task.gid + "_" + task.tid_str + "_n" + std::to_string(task.n) + "_" + task.timestamp;
                    auto base_path = std::filesystem::path(ALARM_DIR) / dir_name;

                    std::filesystem::create_directories(base_path);

                    // 保存原始帧
                    if (!task.full_frame_bgr.empty()) {
                        cv::imwrite((base_path / "frame.jpg").string(), task.full_frame_bgr);
                    }

                    // 保存最新的行人图块
                    if (!task.latest_body_patch_rgb.empty()) {
                        cv::Mat bgr_patch;
                        cv::cvtColor(task.latest_body_patch_rgb, bgr_patch, cv::COLOR_RGB2BGR);
                        cv::imwrite((base_path / "latest_body_patch.jpg").string(), bgr_patch);
                    }

                    // 保存最新的人脸图块
                    if (!task.latest_face_patch_rgb.empty()) {
                        cv::Mat bgr_patch;
                        cv::cvtColor(task.latest_face_patch_rgb, bgr_patch, cv::COLOR_RGB2BGR);
                        cv::imwrite((base_path / "latest_face_patch.jpg").string(), bgr_patch);
                    }

                    // 创建并保存在告警帧上绘制边界框的标注图
                    if (!task.full_frame_bgr.empty()) {
                        cv::Mat alarm_vis = task.full_frame_bgr.clone();
                        cv::rectangle(alarm_vis, task.person_bbox, cv::Scalar(0, 0, 255), 3); // 红色粗框标出行人
                        if (task.face_bbox.area() > 0) {
                            cv::rectangle(alarm_vis, task.face_bbox, cv::Scalar(0, 255, 255), 2); // 黄色框标出人脸
                        }
                        cv::imwrite((base_path / "annotated_frame.jpg").string(), alarm_vis);
                    }
                    break;
                }
            } // end of switch
        } catch (const std::exception &e) {
            std::cerr << "I/O worker error: " << e.what() << std::endl;
        }
    }
}

// ======================= 【MODIFIED: 新增的DB加载函数】 =======================
void FeatureProcessor::_load_state_from_db() {
    // 在从数据库加载状态前，先清空并重建文件系统缓存，以确保两者同步
    if (std::filesystem::exists(SAVE_DIR)) {
        std::filesystem::remove_all(SAVE_DIR);
    }
    std::filesystem::create_directories(SAVE_DIR);
    std::cout << "Filesystem cache directory '" << SAVE_DIR << "' has been reset for DB synchronization." << std::endl;

    // 修改SQL语句，同时选择image列
    const char *sql = "SELECT gid, type, idx, feature, image FROM prototypes ORDER BY gid, type, idx;";
    sqlite3_stmt *stmt;

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "Error: Failed to prepare statement for DB loading: " << sqlite3_errmsg(db_) << std::endl;
        return;
    }

    int loaded_prototypes = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        std::string gid = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0));
        std::string type = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
        int idx = sqlite3_column_int(stmt, 2);

        const void *feature_blob = sqlite3_column_blob(stmt, 3);
        int feature_bytes = sqlite3_column_bytes(stmt, 3);
        const void *image_blob = sqlite3_column_blob(stmt, 4);
        int image_bytes = sqlite3_column_bytes(stmt, 4);

        if (feature_blob && feature_bytes > 0) {
            int num_floats = feature_bytes / sizeof(float);
            std::vector<float> feature(static_cast<const float *>(feature_blob),
                                       static_cast<const float *>(feature_blob) + num_floats);

            bool feature_matched = false;
            if (type == "faces" && num_floats == EMB_FACE_DIM) {
                gid_mgr.bank_faces[gid].push_back(std::move(feature));
                feature_matched = true;
            } else if (type == "bodies" && num_floats == 512) {  // TODO 512 / EMB_BODY_DIM ?
                gid_mgr.bank_bodies[gid].push_back(std::move(feature));
                feature_matched = true;
            }

            if (feature_matched) {
                loaded_prototypes++;

                // 如果存在关联的图像，则将其写入文件系统
                if (image_blob && image_bytes > 0) {
                    try {
                        std::vector<uchar> img_data(static_cast<const uchar *>(image_blob),
                                                    static_cast<const uchar *>(image_blob) + image_bytes);
                        cv::Mat image = cv::imdecode(img_data, cv::IMREAD_COLOR);
                        if (!image.empty()) {
                            std::filesystem::path out_dir = std::filesystem::path(SAVE_DIR) / gid / type;
                            std::filesystem::create_directories(out_dir);

                            char filename[32];
                            sprintf(filename, "%02d.jpg", idx);
                            std::filesystem::path out_path = out_dir / filename;
                            cv::imwrite(out_path.string(), image);
                        }
                    } catch (const std::exception &e) {
                        std::cerr << "Warning: Failed to decode/save image for GID " << gid << "/" << type << "/" << idx
                                  << ". Error: " << e.what() << std::endl;
                    }
                }
            }
        }
    }
    sqlite3_finalize(stmt);

    // 从加载的 GID 中推断下一个 GID 序号
    int max_gid_num = 0;
    std::set<std::string> all_gids;
    for (const auto &pair: gid_mgr.bank_faces) all_gids.insert(pair.first);
    for (const auto &pair: gid_mgr.bank_bodies) all_gids.insert(pair.first);

    for (const auto &gid: all_gids) {
        if (gid.rfind("G", 0) == 0 && gid.length() > 1) {
            try {
                max_gid_num = std::max(max_gid_num, std::stoi(gid.substr(1)));
            } catch (const std::exception &) { /* Ignore parsing errors */ }
        }
    }
    gid_mgr.gid_next = max_gid_num + 1;

    std::cout << "Successfully loaded " << loaded_prototypes << " prototypes for " << all_gids.size()
              << " GIDs from DB and re-created filesystem cache. Next GID is set to: " << gid_mgr.gid_next << std::endl;
}
// ======================= 【修改结束】 =======================

// MODIFIED HERE: 整个函数被重构以使用 GpuMat
void
FeatureProcessor::_load_features_from_cache(const std::string &cam_id, uint64_t fid, const cv::cuda::GpuMat &full_frame,
                                            const std::vector<Detection> &dets, double now_stamp) {
    const auto &stream_id = cam_id;
    const int H = full_frame.rows;
    const int W = full_frame.cols;
    std::string fid_str = std::to_string(fid);
    if (!features_cache_.contains(fid_str)) return;

    for (const auto &det: dets) {
        std::string tid_str_json = stream_id + "_" + std::to_string(det.id);
        first_seen_tid.try_emplace(tid_str_json, now_stamp);
    }

    const auto &frame_features = features_cache_.at(fid_str);

    for (const auto &det: dets) {
        std::string tid_str_json = stream_id + "_" + std::to_string(det.id);

        if (!frame_features.contains(tid_str_json)) continue;

        const nlohmann::json &fd = frame_features.at(tid_str_json);

        // 内部裁剪 patch
        cv::Rect roi = cv::Rect(det.tlwh) & cv::Rect(0, 0, W, H);
        if (roi.width <= 0 || roi.height <= 0) continue;
        cv::cuda::GpuMat gpu_patch = full_frame(roi);
        cv::Mat patch;
        gpu_patch.download(patch);

        auto &agg = agg_pool[tid_str_json];

        if (fd.contains("body_feat") && !fd["body_feat"].is_null()) {
            auto bf = fd["body_feat"].get<std::vector<float>>();
            if (!bf.empty()) agg.add_body(bf, det.score, patch.clone());
        }
        if (fd.contains("face_feat") && !fd["face_feat"].is_null()) {
            auto ff = fd["face_feat"].get<std::vector<float>>();
            if (!ff.empty()) agg.add_face(ff, patch.clone());
        } // 注意：在load模式下，`last_seen`的更新依赖于`use_fid_time`的设置。
        last_seen[tid_str_json] = now_stamp;
    }
}

std::vector<float> FeatureProcessor::_fuse_feat(const std::vector<float> &face_f, const std::vector<float> &body_f) {
    std::vector<float> face_part = face_f.empty() ? std::vector<float>(EMB_FACE_DIM, 0.f) : face_f;
    std::vector<float> body_part = body_f.empty() ? std::vector<float>(EMB_BODY_DIM, 0.f) : body_f;
    for (float &v: face_part) v *= FUSE_W_FACE;
    for (float &v: body_part) v *= FUSE_W_BODY;
    std::vector<float> combo;
    combo.insert(combo.end(), face_part.begin(), face_part.end());
    combo.insert(combo.end(), body_part.begin(), body_part.end());
    float n = 1e-9f;
    for (float v: combo) n += v * v;
    n = std::sqrt(n);
    if (n > 1e-9f) for (float &v: combo) v /= n;
    return combo;
}

std::vector<float> FeatureProcessor::_gid_fused_rep(const std::string &gid) {
    if (!gid_mgr.bank_faces.count(gid) || !gid_mgr.bank_bodies.count(gid)) return {};
    const auto &face_pool = gid_mgr.bank_faces.at(gid);
    const auto &body_pool = gid_mgr.bank_bodies.at(gid);
    auto face_f = face_pool.empty() ? std::vector<float>() : avg_feats(face_pool);
    auto body_f = body_pool.empty() ? std::vector<float>() : avg_feats(body_pool);
    return _fuse_feat(face_f, body_f);
}

// ======================= 【MODIFIED: 函数逻辑和签名变更】 =======================
std::optional<std::tuple<std::string, std::string, bool>>
FeatureProcessor::trigger_alarm(const std::string &tid_str, const std::string &gid, int n, const TrackAgg &agg,
                                double frame_timestamp) {
    auto cur_rep = _gid_fused_rep(gid);
    if (cur_rep.empty()) return std::nullopt;

    std::string gid_to_report = gid;
    bool is_new_alarm_gid = true;

    // 检查当前 GID 是否与已有的“原始报警GID”相似
    for (const auto &[ogid, rep]: alarm_reprs) {
        if (sim_vec(cur_rep, rep) >= ALARM_DUP_THR) {
            gid_to_report = ogid;
            is_new_alarm_gid = false;
            // 如果相似，则无论当前 gid 是谁，都将此次报警归属于原始的 ogid
            if (gid != ogid) {
                std::cout << "[ALARM] GID " << gid << " is similar to original alarmer " << ogid << ". Reporting as "
                          << ogid << "." << std::endl;
            } else {
                std::cout << "[ALARM] Re-triggering for original GID " << gid << "." << std::endl;
            }
            break; // 找到匹配项，中断循环
        }
    }

    // 如果遍历完所有原始报警者都不相似，则此 gid 成为一个新的“原始报警GID”
    if (is_new_alarm_gid) {
        std::cout << "[ALARM] New original alarmer: " << gid << "." << std::endl;
        alarmed.insert(gid);
        alarm_reprs[gid] = cur_rep;
    }

    // --- 新增：检查此TID是否已保存过报警 ---
    bool was_newly_saved = false;
    if (saved_alarm_tids_.find(tid_str) == saved_alarm_tids_.end()) {
        was_newly_saved = true;
        saved_alarm_tids_.insert(tid_str);

        // 仅在首次保存此TID的报警时，提交原型备份任务 (Back up GID prototypes)
        time_t seconds_for_task = static_cast<time_t>(frame_timestamp);
        char buf_for_task[80];
        struct tm broken_down_time_for_task;
        localtime_r(&seconds_for_task, &broken_down_time_for_task);
        strftime(buf_for_task, sizeof(buf_for_task), "%Y%m%d_%H%M%S", &broken_down_time_for_task);

        IoTask task;
        task.type = IoTaskType::BACKUP_ALARM;
        task.gid = gid_to_report;
        task.tid_str = tid_str;
        task.n = n;
        task.timestamp = std::string(buf_for_task);
        task.face_patches_backup = agg.face_patches();
        task.body_patches_backup = agg.body_patches();
        submit_io_task(task);
    }

    // --- 无论是否保存，都生成时间戳并返回报警信号 ---
    time_t seconds = static_cast<time_t>(frame_timestamp);
    char buf[80];
    struct tm broken_down_time;
    localtime_r(&seconds, &broken_down_time);
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &broken_down_time);
    std::string timestamp_str(buf);

    return std::make_tuple(gid_to_report, timestamp_str, was_newly_saved);
}
// ======================= 【修改结束】 =======================

// ======================= 【NEW】 =======================
// 新增的私有辅助函数，用于封装重复的报警逻辑
void FeatureProcessor::_check_and_process_alarm(
        ProcessOutput &output,
        const ProcessConfig &config,
        const std::string &tid_str,
        const std::string &gid,
        const TrackAgg &agg,
        double now_stamp,
        GstClockTime now_stamp_gst,
        const std::vector<Detection> &dets,
        const std::string &stream_id,
        const cv::Mat &body_p,
        const cv::Mat &face_p,
        std::vector<std::tuple<std::string, std::string, std::string, int, bool>> &triggered_alarms_this_frame) {

    int n = gid_mgr.tid_hist.count(gid) ? (int) gid_mgr.tid_hist.at(gid).size() : 0;

    if (n >= config.alarm_cnt_th) {
        if (auto alarm_data_opt = trigger_alarm(tid_str, gid, n, agg, now_stamp)) {
            auto &[gid_to_alarm, timestamp, was_newly_saved] = *alarm_data_opt;

            AlarmTriggerInfo alarm_info;
            alarm_info.gid = gid_to_alarm;
            alarm_info.tid_str = tid_str;
            alarm_info.first_seen_timestamp = gid_mgr.first_seen_ts.count(gid_to_alarm)
                                              ? gid_mgr.first_seen_ts.at(gid_to_alarm)
                                              : now_stamp_gst;
            for (const auto &det: dets) {
                if (stream_id + "_" + std::to_string(det.id) == tid_str) {
                    alarm_info.person_bbox = det.tlwh;
                    break;
                }
            }
            // 新增：将识别次数 n 赋值给告警信息
            alarm_info.n = n;

            if (alarm_info.person_bbox.area() > 0) {
                triggered_alarms_this_frame.emplace_back(gid_to_alarm, tid_str, timestamp, n, was_newly_saved);
                if (current_frame_face_boxes_.count(tid_str)) {
                    alarm_info.face_bbox = current_frame_face_boxes_.at(tid_str);
                }
                alarm_info.latest_body_patch = body_p.clone();
                alarm_info.latest_face_patch = face_p.clone();
                output.alarms.push_back(alarm_info);
            }
        }
    }
}
// ======================= 【修改结束】 =======================

// ======================= 【MODIFIED】 =======================
// 修改: 函数签名以接收 ProcessInput 结构体，并返回 ProcessOutput
ProcessOutput FeatureProcessor::process_packet(const ProcessInput &input) {
    // 新增：算法总开关
    if (!m_processing_enabled) {
        return {}; // 如果禁用，直接返回空结果
    }

    // 在函数入口处解包，保持函数体内部逻辑不变，减少出错风险
    const std::string &cam_id = input.cam_id;
    uint64_t fid = input.fid;
    const cv::cuda::GpuMat &full_frame = input.full_frame;
    const std::vector<Detection> &dets = input.dets;
    const ProcessConfig &config = input.config;
    // ======================= 【修改结束】 =======================

    // 根据配置确定当前相机是否启用人脸处理
    bool face_enabled = true; // 默认为启用
    if (config.face_switch_by_cam.count(cam_id)) {
        face_enabled = config.face_switch_by_cam.at(cam_id);
    }

    ProcessOutput output;
    std::vector<std::tuple<std::string, std::string, std::string, int, bool>> triggered_alarms_this_frame; // <gid, tid_str, timestamp, n, was_newly_saved>
    current_frame_face_boxes_.clear(); // 每帧开始时清空

    const auto &stream_id = cam_id;

    // --- 确定当前帧的匹配阈值 ---
    // 1. 根据灵敏度设置一个基础阈值
    int sensitivity = 2; // 默认中等灵敏度
    if (config.sensitivity_by_cam.count(stream_id)) {
        sensitivity = config.sensitivity_by_cam.at(stream_id);
    }
    float base_match_thr;
    switch (sensitivity) {
        case 1: // 低灵敏度 -> 高阈值，更难匹配
            base_match_thr = 0.6f;
            break;
        case 3: // 高灵敏度 -> 低阈值，更容易匹配
            base_match_thr = 0.4f;
            break;
        default: // case 2 或其他值
            base_match_thr = MATCH_THR; // 中等灵敏度，使用默认阈值 0.5f
    }
    // 2. 允许使用具体的浮点数值覆盖基于灵敏度的设置，提供更精细的控制
    float current_match_thr = config.match_thr_by_cam.count(stream_id) ? config.match_thr_by_cam.at(stream_id)
                                                                       : base_match_thr;

    if (intrusion_detectors.count(stream_id)) {
        for (int tid: intrusion_detectors.at(stream_id)->check(dets, stream_id))
            behavior_alarm_state[stream_id + "_" + std::to_string(tid)] = {fid, "_AA"};
    }
    if (line_crossing_detectors.count(stream_id)) {
        for (int tid: line_crossing_detectors.at(stream_id)->check(dets, stream_id))
            behavior_alarm_state[stream_id + "_" + std::to_string(tid)] = {fid, "_AL"};
    }

    // --- 核心修改：超时逻辑 ---
    double now_stamp; // 内部统一使用 double 秒级时间戳
    GstClockTime now_stamp_gst; // GStreamer 时间戳 (ns), 用于精确存储
    double max_tid_idle, gid_max_idle;
    if (use_fid_time_) {
        // 如果是基于帧号计时（例如 'load' 模式），则 fid 是时间源
        now_stamp = static_cast<double>(fid);
        // 在 'load' 模式下没有真实的 GstClockTime，用0作为无效值
        now_stamp_gst = 0;
        max_tid_idle = MAX_TID_IDLE_FRAMES;
        gid_max_idle = GID_MAX_IDLE_FRAMES;
    } else {
        // 如果是实时模式，将传入的 GstClockTime (纳秒) 转换为 double (秒)
        now_stamp = static_cast<double>(input.timestamp) / 1000000000.0;
        now_stamp_gst = input.timestamp;
        max_tid_idle = MAX_TID_IDLE_SEC;
        gid_max_idle = GID_MAX_IDLE_SEC;
    }

    // 【修改】获取该摄像头的徘徊时间配置，如果未设置则默认为0
    long long current_alarm_duration_ms = 0;
    if (config.alarmDuration_ms_by_cam.count(cam_id)) {
        current_alarm_duration_ms = config.alarmDuration_ms_by_cam.at(cam_id);
    }

    // 计算徘徊时间阈值 (单位与 now_stamp 一致)
    double alarmDuration_threshold = 0.0;
    if (current_alarm_duration_ms > 0) {
        if (use_fid_time_) {
            alarmDuration_threshold = (double) current_alarm_duration_ms / 1000.0 * FPS_ESTIMATE;
        } else {
            alarmDuration_threshold = (double) current_alarm_duration_ms / 1000.0;
        }
    }

    // ======================= 【MODIFIED】 =======================
    // Reworked feature extraction pipeline for parallelism
    if (mode_ == "realtime") {
        nlohmann::json extracted_features_for_this_frame;
        const int H = full_frame.rows;
        const int W = full_frame.cols;

        // --- DEADLOCK FIX: 串行执行DLA任务 ---
        // 1. 首先在主线程中执行Re-ID特征提取 (使用DLA Core 1)
        for (const auto &det : dets) {
            if (det.class_id != 0) continue;

            cv::Rect roi = cv::Rect(det.tlwh) & cv::Rect(0, 0, W, H);
            if (roi.width <= 0 || roi.height <= 0) continue;

            cv::cuda::GpuMat gpu_patch = full_frame(roi);
            cv::Mat patch;
            gpu_patch.download(patch);

            if (!is_long_patch(patch)) continue;

            cv::cuda::GpuMat feat_mat_gpu = reid_model_->extract_feat(gpu_patch);
            if (feat_mat_gpu.empty()) continue;

            cv::Mat feat_mat_cpu;
            feat_mat_gpu.download(feat_mat_cpu);
            std::vector<float> body_feat(feat_mat_cpu.begin<float>(), feat_mat_cpu.end<float>());
            std::string tid_str = stream_id + "_" + std::to_string(det.id);
            first_seen_tid.try_emplace(tid_str, now_stamp);
            agg_pool[tid_str].add_body(body_feat, det.score, patch.clone());
            last_seen[tid_str] = now_stamp;
            if (!feature_cache_path_.empty()) extracted_features_for_this_frame[tid_str]["body_feat"] = body_feat;
        }

        // 2. 然后在主线程中执行人脸分析 (使用 DLA Core 0)
        std::vector<Face> internal_face_info;
        if (face_enabled && face_analyzer_) {
            internal_face_info = face_analyzer_->detect(full_frame);
            if (!internal_face_info.empty()) {
                const int H = full_frame.rows;
                const int W = full_frame.cols;
                std::set<size_t> used_face_indices;

                for (const auto &det: dets) {
                    if (det.class_id != 0) continue;

                    std::vector<size_t> matching_face_indices;
                    for (size_t j = 0; j < internal_face_info.size(); ++j) {
                        if (used_face_indices.count(j)) continue;
                        if (calculate_ioa(det.tlwh, internal_face_info[j].bbox) > 0.8) {
                            matching_face_indices.push_back(j);
                        }
                    }
                    if (matching_face_indices.size() != 1) continue;

                    size_t unique_face_idx = matching_face_indices[0];
                    Face &face_global_coords = internal_face_info[unique_face_idx]; // Use reference to update
                    if (face_global_coords.det_score < FACE_DET_MIN_SCORE) continue;

                    cv::Rect face_roi(face_global_coords.bbox);
                    face_roi &= cv::Rect(0, 0, W, H);
                    if (face_roi.width < 32 || face_roi.height < 32) continue;

                    try {
                        face_analyzer_->get_embedding(full_frame, face_global_coords);
                        if (face_global_coords.embedding.empty()) continue;
                        used_face_indices.insert(unique_face_idx);

                        cv::Mat normalized_emb;
                        cv::normalize(face_global_coords.embedding, normalized_emb, 1.0, 0.0, cv::NORM_L2);
                        std::vector<float> f_emb(normalized_emb.begin<float>(), normalized_emb.end<float>());

                        // ======================= 【MODIFIED】 =======================
                        // Crop the real face patch from the GPU frame instead of using a dummy one.
                        cv::cuda::GpuMat gpu_face_patch = full_frame(face_roi);
                        cv::Mat face_patch;
                        std::string tid_str = stream_id + "_" + std::to_string(det.id);
                        // 新增：如果TID首次出现，记录其时间戳
                        first_seen_tid.try_emplace(tid_str, now_stamp);
                        current_frame_face_boxes_[tid_str] = face_global_coords.bbox;
                        gpu_face_patch.download(face_patch);
                        agg_pool[tid_str].add_face(f_emb, face_patch.clone());
                        // ======================= 【修改结束】 =======================
                        last_seen[tid_str] = now_stamp;
                        if (!feature_cache_path_.empty()) extracted_features_for_this_frame[tid_str]["face_feat"] = f_emb;
                    } catch (const std::exception &) { continue; }
                }
            }
        }

        // 3. 保存所有本帧提取的特征到缓存文件
        if (!feature_cache_path_.empty() && !extracted_features_for_this_frame.is_null()) {
            features_to_save_[std::to_string(fid)] = extracted_features_for_this_frame;
        }

    } else if (mode_ == "load") { // 注意：这里的 now_stamp 是基于 fid 的
        _load_features_from_cache(cam_id, fid, full_frame, dets, now_stamp);
    }
    // ======================= 【修改结束】=======================

    // ======================= 【FIXED】 =======================
    // 关键修复：在主循环入口处增加过滤器，确保只处理属于当前摄像头(stream_id)的轨迹。
    // 这从根本上解决了跨摄像头上下文污染的问题，且改动极小。
    for (auto const &[tid_str, agg]: agg_pool) {
        if (tid_str.rfind(stream_id, 0) != 0) continue;

        // 新增：计算可见时长并填充输出，以便在UI上显示
        double duration = 0.0;
        if (first_seen_tid.count(tid_str)) {
            duration = now_stamp - first_seen_tid.at(tid_str);
        }
        output.tid_durations_sec[tid_str] = use_fid_time_ ? (duration / FPS_ESTIMATE) : duration;

        size_t last_underscore = tid_str.find_last_of('_');
        std::string s_id = tid_str.substr(0, last_underscore);
        int tid_num = std::stoi(tid_str.substr(last_underscore + 1));

        // 【修改】检查是否满足徘徊时间
        if (current_alarm_duration_ms > 0 && duration < alarmDuration_threshold) {
            output.mp[s_id][tid_num] = {tid_str + "_-8_wait_d", -1.f, 0};
            continue;
        }

        if ((int) agg.body.size() < MIN_BODY4GID) {
            std::string gid_str = tid_str + "_-1_b_" + std::to_string(agg.body.size());
            output.mp[s_id][tid_num] = {gid_str, -1.f, 0};
            continue;
        }
        if ((int) agg.face.size() < MIN_FACE4GID) {
            std::string gid_str = tid_str + "_-1_f_" + std::to_string(agg.face.size());
            output.mp[s_id][tid_num] = {gid_str, -1.f, 0};
            continue;
        }

        auto [face_f, face_p] = agg.main_face_feat_and_patch();
        auto [body_f, body_p] = agg.main_body_feat_and_patch();
        // 如果启用了人脸但特征为空，则认为是不稳定状态
        if (face_enabled && face_f.empty()) {
            output.mp[s_id][tid_num] = {tid_str + "_-2_f", -1.f, 0};
            continue;
        }
        // ReID特征是必须的
        if (body_f.empty()) {
            output.mp[s_id][tid_num] = {tid_str + "_-2_b", -1.f, 0};
            continue;
        }

        // 根据配置确定用于探测的融合权重
        float w_face = W_FACE;
        float w_body = W_BODY;
        if (!face_enabled) {
            w_face = 0.0f;
            w_body = 1.0f;
        } else {
            if (config.face_weight_by_cam.count(stream_id)) {
                w_face = config.face_weight_by_cam.at(stream_id);
            }
            if (config.reid_weight_by_cam.count(stream_id)) {
                w_body = config.reid_weight_by_cam.at(stream_id);
            }
        }

        auto [cand_gid, score] = gid_mgr.probe(face_f, body_f, w_face, w_body);

        auto &state = candidate_state[tid_str];
        auto &ng_state = new_gid_state[tid_str];
        uint64_t time_since_last_new = fid - ng_state.last_new_fid;

        if (tid2gid.count(tid_str)) {
            std::string bound_gid = tid2gid.at(tid_str);
            if (!cand_gid.empty() && cand_gid != bound_gid && score >= current_match_thr &&
                (fid - state.last_bind_fid) < BIND_LOCK_FRAMES) {
                int n_tid = gid_mgr.tid_hist.count(bound_gid) ? (int) gid_mgr.tid_hist.at(bound_gid).size() : 0;
                output.mp[s_id][tid_num] = {tid_str + "_-3", score, n_tid};
                continue;
            }
        }

        if (!cand_gid.empty() && score >= current_match_thr) {
            ng_state.ambig_count = 0;
            state.count = (state.cand_gid == cand_gid) ? state.count + 1 : 1;
            state.cand_gid = cand_gid;
            int flag_code = gid_mgr.can_update_proto(cand_gid, face_f, body_f);
            if (state.count >= CANDIDATE_FRAMES && flag_code == 0) {
                gid_mgr.bind(cand_gid, tid_str, now_stamp, now_stamp_gst, agg, this);
                tid2gid[tid_str] = cand_gid;
                state.last_bind_fid = fid;
                int n = gid_mgr.tid_hist.count(cand_gid) ? (int) gid_mgr.tid_hist[cand_gid].size() : 0;
                _check_and_process_alarm(output, config, tid_str, cand_gid, agg, now_stamp, now_stamp_gst, dets, stream_id, body_p,
                                         face_p, triggered_alarms_this_frame);

                output.mp[s_id][tid_num] = {s_id + "_" + std::to_string(tid_num) + "_" + cand_gid, score, n};
            } else {
                std::string flag = (flag_code == -1) ? "_-4_ud_f" : (flag_code == -2) ? "_-4_ud_b" : "_-4_c";
                output.mp[s_id][tid_num] = {tid_str + flag, -1.0f, 0};
            }
        } else if (gid_mgr.bank_faces.empty()) {
            std::string new_gid = gid_mgr.new_gid();
            gid_mgr.bind(new_gid, tid_str, now_stamp, now_stamp_gst, agg, this);
            tid2gid[tid_str] = new_gid;
            candidate_state[tid_str] = {new_gid, CANDIDATE_FRAMES, fid};
            new_gid_state[tid_str].last_new_fid = fid;
            int n = gid_mgr.tid_hist.count(new_gid) ? (int) gid_mgr.tid_hist[new_gid].size() : 0;
            _check_and_process_alarm(output, config, tid_str, new_gid, agg, now_stamp, now_stamp_gst, dets, stream_id, body_p,
                                     face_p, triggered_alarms_this_frame);

            output.mp[s_id][tid_num] = {s_id + "_" + std::to_string(tid_num) + "_" + new_gid, score, n};
        } else if (!cand_gid.empty() && score >= THR_NEW_GID) {
            ng_state.ambig_count++;
            if (ng_state.ambig_count >= WAIT_FRAMES_AMBIGUOUS && time_since_last_new >= NEW_GID_TIME_WINDOW) {
                std::string new_gid = gid_mgr.new_gid();
                gid_mgr.bind(new_gid, tid_str, now_stamp, now_stamp_gst, agg, this);
                tid2gid[tid_str] = new_gid;
                candidate_state[tid_str] = {new_gid, CANDIDATE_FRAMES, fid};
                new_gid_state[tid_str] = {0, fid, 0};
                int n = gid_mgr.tid_hist.count(new_gid) ? (int) gid_mgr.tid_hist[new_gid].size() : 0;
                _check_and_process_alarm(output, config, tid_str, new_gid, agg, now_stamp, now_stamp_gst, dets, stream_id, body_p,
                                         face_p, triggered_alarms_this_frame);

                output.mp[s_id][tid_num] = {s_id + "_" + std::to_string(tid_num) + "_" + new_gid, score, n};
            } else {
                output.mp[s_id][tid_num] = {tid_str + "_-7", score, 0};
            }
        } else { // score < THR_NEW_GID
            ng_state.ambig_count = 0;
            if (time_since_last_new >= NEW_GID_TIME_WINDOW) {
                ng_state.count++;
                if (ng_state.count >= NEW_GID_MIN_FRAMES) {
                    std::string new_gid = gid_mgr.new_gid();
                    gid_mgr.bind(new_gid, tid_str, now_stamp, now_stamp_gst, agg, this);
                    tid2gid[tid_str] = new_gid;
                    candidate_state[tid_str] = {new_gid, CANDIDATE_FRAMES, fid};
                    new_gid_state[tid_str] = {0, fid, 0};
                    int n = gid_mgr.tid_hist.count(new_gid) ? (int) gid_mgr.tid_hist[new_gid].size() : 0;
                    _check_and_process_alarm(output, config, tid_str, new_gid, agg, now_stamp, now_stamp_gst, dets, stream_id, body_p,
                                             face_p, triggered_alarms_this_frame);

                    output.mp[s_id][tid_num] = {s_id + "_" + std::to_string(tid_num) + "_" + new_gid, score, n};
                } else {
                    output.mp[s_id][tid_num] = {tid_str + "_-5", -1.0f, 0};
                }
            } else {
                output.mp[s_id][tid_num] = {tid_str + "_-6", -1.0f, 0};
            }
        }
    }

    std::map<std::string, std::tuple<uint64_t, std::string>> active_alarms;
    for (const auto &[full_tid, state_tuple]: behavior_alarm_state) {
        if (fid - std::get<0>(state_tuple) <= BEHAVIOR_ALARM_DURATION_FRAMES) {
            active_alarms[full_tid] = state_tuple;
            size_t last_underscore = full_tid.find_last_of('_');
            std::string s_id = full_tid.substr(0, last_underscore);
            int t_id_int = std::stoi(full_tid.substr(last_underscore + 1));
            std::string bound_gid = tid2gid.count(full_tid) ? tid2gid.at(full_tid) : "";
            int n_tid = 0;
            if (!bound_gid.empty() && gid_mgr.tid_hist.count(bound_gid)) {
                n_tid = gid_mgr.tid_hist.at(bound_gid).size();
            }
            std::string info_str = !bound_gid.empty() ? (full_tid + "_" + bound_gid + std::get<1>(state_tuple)) : (
                    full_tid + "_-1" + std::get<1>(state_tuple));
            output.mp[s_id][t_id_int] = {info_str, 1.0f, n_tid};
        }
    }
    behavior_alarm_state = active_alarms;

    for (auto it = last_seen.cbegin(); it != last_seen.cend();) {
        if (now_stamp - it->second >= max_tid_idle) {
            agg_pool.erase(it->first);
            saved_alarm_tids_.erase(it->first); // 清理已保存报警的TID记录
            first_seen_tid.erase(it->first);
            tid2gid.erase(it->first);
            candidate_state.erase(it->first);
            new_gid_state.erase(it->first);
            behavior_alarm_state.erase(it->first);
            it = last_seen.erase(it);
        } else { ++it; }
    }

    std::vector<std::string> gids_to_del;
    for (auto const &[gid, last_ts]: gid_mgr.last_update) {
        if (now_stamp - last_ts >= gid_max_idle) gids_to_del.push_back(gid);
    }
    for (const auto &gid_del: gids_to_del) {
        std::vector<std::string> tids_to_clean;
        for (auto const &[tid_str, g]: tid2gid) { if (g == gid_del) tids_to_clean.push_back(tid_str); }
        for (const auto &tid_str: tids_to_clean) {
            agg_pool.erase(tid_str);
            first_seen_tid.erase(tid_str);
            tid2gid.erase(tid_str);
            candidate_state.erase(tid_str);
            new_gid_state.erase(tid_str);
            last_seen.erase(tid_str);
            behavior_alarm_state.erase(tid_str);
        }
        gid_mgr.bank_faces.erase(gid_del);
        gid_mgr.bank_bodies.erase(gid_del);
        gid_mgr.tid_hist.erase(gid_del);
        gid_mgr.last_update.erase(gid_del);
        gid_mgr.first_seen_ts.erase(gid_del);
        alarmed.erase(gid_del);
        alarm_reprs.erase(gid_del);

#ifdef ENABLE_DISK_IO
        IoTask task;
        task.type = IoTaskType::CLEANUP_GID_DIR;
        task.gid = gid_del;
        submit_io_task(task);
#endif
    }

    // 仅当功能开关打开且确实有报警时才执行保存逻辑
    if (m_enable_alarm_saving && !triggered_alarms_this_frame.empty()) {
        // --- 新增：识别出本帧中首次需要保存上下文的报警集合 ---
        // The old set is not enough. We need to iterate the full tuple vector.
        std::vector<std::tuple<std::string, std::string, std::string, int>> unique_new_alarms_to_save; // <gid, tid_str, timestamp, n>
        std::set<std::string> seen_gids_for_saving; // To ensure we only save context once per GID per frame
        for (const auto &[gid, tid_str, timestamp, n, was_newly_saved]: triggered_alarms_this_frame) {
            if (was_newly_saved && seen_gids_for_saving.find(gid) == seen_gids_for_saving.end()) {
                unique_new_alarms_to_save.emplace_back(gid, tid_str, timestamp, n);
                seen_gids_for_saving.insert(gid);
            }
        }

        // 只有在存在需要保存上下文的新报警时，才执行耗时的文件I/O准备
        if (!unique_new_alarms_to_save.empty()) {
            // ======================= 【NEW LOGIC】 =======================
            // 为本帧触发的每个报警，提交保存上下文信息（图片和文本）的异步任务

            // 1. 如果需要保存图片，提前将GPU帧下载并转换为BGR格式
            cv::Mat frame_bgr_for_saving;
            cv::cuda::GpuMat temp_gpu_bgr;
            cv::cuda::cvtColor(full_frame, temp_gpu_bgr, cv::COLOR_RGB2BGR);
            temp_gpu_bgr.download(frame_bgr_for_saving);

            // 1. 准备 frame_info.txt 的内容
            // 为确保输出一致，对TID进行排序
            std::map<std::string, std::vector<int>> sorted_tids_by_cam;
            for (const auto &[cam, tids_map]: output.mp) {
                for (const auto &[tid, result_tuple]: tids_map) {
                    sorted_tids_by_cam[cam].push_back(tid);
                }
                if (sorted_tids_by_cam.count(cam)) {
                    std::sort(sorted_tids_by_cam.at(cam).begin(), sorted_tids_by_cam.at(cam).end());
                }
            }

            std::stringstream ss;
            ss << std::fixed << std::setprecision(4);
            ss << "frame_id,cam_id,tid,gid,score,n_tid\n";
            for (const auto &[cam, sorted_tids]: sorted_tids_by_cam) {
                if (output.mp.count(cam)) {
                    for (int tid: sorted_tids) {
                        if (output.mp.at(cam).count(tid)) {
                            const auto &tpl = output.mp.at(cam).at(tid);
                            ss << fid << ',' << cam << ',' << tid << ','
                               << std::get<0>(tpl) << ',' << std::get<1>(tpl) << ',' << std::get<2>(tpl) << "\n";
                        }
                    }
                }
            }
            std::string content = ss.str();

            // 2. 遍历本帧所有新触发的报警，为每个报警创建并提交任务
            for (const auto &[gid, tid_str, timestamp, n]: unique_new_alarms_to_save) {
                // 2a. 提交保存 frame_info.txt 的任务
                IoTask txt_task;
                txt_task.type = IoTaskType::SAVE_ALARM_INFO;
                txt_task.gid = gid;
                txt_task.tid_str = tid_str;
                txt_task.n = n;
                txt_task.timestamp = timestamp;
                txt_task.alarm_info_content = content;
                submit_io_task(std::move(txt_task));

                // 2b. 提交保存相关图片的任务
                auto it = std::find_if(output.alarms.begin(), output.alarms.end(),
                                       [&](const AlarmTriggerInfo &a) { return a.gid == gid && a.tid_str == tid_str; });
                if (it != output.alarms.end()) {
                    IoTask img_task;
                    img_task.type = IoTaskType::SAVE_ALARM_CONTEXT_IMAGES;
                    img_task.gid = gid;
                    img_task.tid_str = tid_str;
                    img_task.n = n;
                    img_task.timestamp = timestamp;
                    img_task.full_frame_bgr = frame_bgr_for_saving.clone();
                    img_task.latest_body_patch_rgb = it->latest_body_patch.clone();
                    img_task.latest_face_patch_rgb = it->latest_face_patch.clone();
                    img_task.person_bbox = it->person_bbox;
                    img_task.face_bbox = it->face_bbox;
                    submit_io_task(std::move(img_task));
                }
            }
            // ======================= 【END NEW LOGIC】 =======================
        }
    }

    return output;
}

// ======================= 【MODIFIED】 =======================
// 新增：用于调试和验证的函数，将内存中的 GID 状态写入文件
void FeatureProcessor::save_final_state_to_file(const std::string &filepath) {
    std::ofstream out(filepath);
    out << "Next GID: " << gid_mgr.gid_next << "\n\n";

    std::set<std::string> all_gids;
    for (const auto &pair: gid_mgr.bank_faces) all_gids.insert(pair.first);
    for (const auto &pair: gid_mgr.bank_bodies) all_gids.insert(pair.first);

    // std::set 保证了 GID 的遍历顺序是字母序，与 load_and_dump.cpp 中 ORDER BY gid 的行为一致
    for (const auto &gid: all_gids) {
        out << "--- GID: " << gid << " ---\n";

        if (gid_mgr.bank_faces.count(gid)) {
            const auto &face_feats = gid_mgr.bank_faces.at(gid);
            out << "Faces: " << face_feats.size() << "\n";
            // 假设 bank_faces 中特征的顺序与数据库中的 idx 顺序一致
            for (size_t i = 0; i < face_feats.size(); ++i) {
                out << "  Face[" << i << "]: ";
                for (float val: face_feats[i]) {
                    out << std::fixed << std::setprecision(8) << val << " ";
                }
                out << "\n";
            }
        }

        if (gid_mgr.bank_bodies.count(gid)) {
            const auto &body_feats = gid_mgr.bank_bodies.at(gid);
            out << "Bodies: " << body_feats.size() << "\n";
            // 假设 bank_bodies 中特征的顺序与数据库中的 idx 顺序一致
            for (size_t i = 0; i < body_feats.size(); ++i) {
                out << "  Body[" << i << "]: ";
                for (float val: body_feats[i]) {
                    out << std::fixed << std::setprecision(8) << val << " ";
                }
                out << "\n";
            }
        }
        out << "\n";
    }
    out.close();
    std::cout << "In-memory state successfully written to " << filepath << std::endl;
}
// ======================= 【修改结束】 =======================