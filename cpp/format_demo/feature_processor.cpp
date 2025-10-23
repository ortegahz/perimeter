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
#include <cmath> // For std::atan2, std::abs
#include <ctime>   // For localtime_r and strftime
#include <sstream>
#include "cores/face/PoseEstimator.hpp" // 新增头文件

// ======================= 【新增宏：控制冷却逻辑调试打印】 =======================
// 取消注释此行以启用关于 GID 识别冷却逻辑的详细打印
 #define ENABLE_COOLDOWN_DEBUG_PRINTS
// ======================= 【新增结束】 =======================

// ======================= 【MODIFIED】 =======================
// 添加一个宏来控制所有磁盘I/O操作
// 如果要关闭所有耗时的文件写入和删除，请注释掉下面这行
//#define ENABLE_DISK_IO
// ======================= 【修改结束】 =======================

// ======================= 【新增：时间戳格式化函数】 =======================
#ifndef GST_CLOCK_TIME_NONE
// using GstClockTime defined in header, but keep guard for standalone safety
#define GST_CLOCK_TIME_NONE ((GstClockTime)-1)
#endif

/**
 * @brief 将 GStreamer NTP 时间戳格式化为人类可读的字符串 (用于调试打印)。
 * @param ntp_timestamp GStreamer 时钟时间（纳秒）。
 * @return 格式化后的时间字符串 (例如 "YYYY-MM-DD HH:MM:SS.ms")。
 */
static std::string format_ntp_timestamp(GstClockTime ntp_timestamp) {
    if (ntp_timestamp == 0 || ntp_timestamp == GST_CLOCK_TIME_NONE) {
        return "[INVALID TIMESTAMP]";
    }
    time_t seconds = ntp_timestamp / 1000000000;
    long milliseconds = (ntp_timestamp % 1000000000) / 1000000;
    char time_str_buffer[128];
    struct tm broken_down_time;
    localtime_r(&seconds, &broken_down_time);
    int len = strftime(time_str_buffer, sizeof(time_str_buffer),
                       "%Y-%m-%d %H:%M:%S", &broken_down_time);
    snprintf(time_str_buffer + len, sizeof(time_str_buffer) - len,
             ".%03ld", milliseconds);
    return std::string(time_str_buffer);
}
// ======================= 【新增结束】 =======================

// ======================= 【NEW: 人脸清晰度估计算法】 =======================
/**
 * @brief 使用拉普拉斯算子的方差来估计图像清晰度，并将其映射到 0-100 的范围。
 * @param face_patch RGB格式的人脸图像块。
 * @return 0.0f (非常模糊) 到 100.0f (非常清晰) 的浮点数分数。
 */
static float calculate_clarity_score(const cv::Mat& face_patch) {
    if (face_patch.empty()) {
        return 0.0f;
    }

    cv::Mat gray;
    cv::cvtColor(face_patch, gray, cv::COLOR_RGB2GRAY); // 输入是RGB，转为灰度
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    double variance = stddev.val[0] * stddev.val[0];

    // 将方差非线性地映射到 0-100 分。阈值是经验值，可根据实际效果调整。
    const float min_var = 20.0f;  // 方差低于此值，清晰度视为0
    const float max_var = 500.0f; // 方差高于此值，清晰度视为100
    float score = ((variance - min_var) / (max_var - min_var)) * 100.0f;
    return std::max(0.0f, std::min(100.0f, score));
}
// ======================= 【修改结束】 =======================

// ======================= 【MODIFIED: 正脸判断算法 (PnP法)】 =======================
/**
 * @brief 使用 solvePnP 估计头部姿态并判断是否为正脸。
 * @param kps 5个关键点 (左眼, 右眼, 鼻子, 左嘴角, 右嘴角) 的 std::vector<cv::Point2f>。
 * @param image_size 原始图像的尺寸。
 * @return 如果是正脸则返回 true，否则返回 false。
 */
static bool is_frontal_face_pnp(const std::vector<cv::Point2f>& kps, cv::Size image_size, double yaw_th, double roll_th, double pitch_ratio_lower_th, double pitch_ratio_upper_th) {

    if (kps.size() != 5) {
        return false;
    }

    auto pose_result_opt = PoseEstimator::estimate_pose(image_size, kps);

    if (!pose_result_opt.has_value()) {
        return false; // 姿态估计失败，保守地认为不是正脸
    }

    const auto &result = pose_result_opt.value();
    return std::abs(result.yaw) < yaw_th &&
           std::abs(result.roll) < roll_th &&
           result.pitch_score > pitch_ratio_lower_th && result.pitch_score < pitch_ratio_upper_th;
}

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

void TrackAgg::add_face(const std::vector<float> &feat, const cv::Mat &patch, bool is_frontal, float score) {
    if (patch.empty()) return;
    face.emplace_back(feat, patch.clone(), is_frontal, score);
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

    // 为兼容 main_representation_with_patch_cpp 模板函数，创建一个临时的、不包含姿态和分数信息的 deque
    std::deque<std::tuple<std::vector<float>, cv::Mat>> face_data_for_main_rep;
    for (const auto &rec : face) {
        face_data_for_main_rep.emplace_back(std::get<0>(rec), std::get<1>(rec));
    }

    std::deque<std::vector<float>> feats;
    for (const auto &t: face) feats.push_back(std::get<0>(t));
    if (!check_consistency(feats))
        return {{},
                {}};
    return main_representation_with_patch_cpp(face_data_for_main_rep);
}

int TrackAgg::count_high_quality_faces(float score_thr) const {
    int count = 0;
    for (const auto &record : face) {
        bool is_frontal = std::get<2>(record);
        float score = std::get<3>(record);
        if (is_frontal && score >= score_thr) {
            count++;
        }
    }
    return count;
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
        const std::string &pure_gid,
        const std::string &type, // "faces" or "bodies"
        FeatureProcessor *fp,
        const std::string &creation_reason) {

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
        std::string gid_for_path = pure_gid;
        if (!creation_reason.empty()) {
            gid_for_path += "_reason_" + creation_reason;
        }

        char filename[32];
        sprintf(filename, "%02d.jpg", idx_to_replace);
        IoTask task;
        task.type = is_new_proto ? IoTaskType::SAVE_PROTOTYPE
                                 : IoTaskType::UPDATE_PROTOTYPE; // is_new_proto is now defined
        task.gid = pure_gid; // Use pure GID for DB
        task.feature = feat_list[idx_to_replace]; // The feature is already updated/added at this index
        task.path_suffix = gid_for_path + "/" + std::string(type) + "/" + filename;
        task.image = new_patch.clone(); // Deep copy for thread safety
        fp->submit_io_task(task);
    }

    // 保留了离群点检测，但只提交删除任务
    auto original_size = feat_list.size();
    auto [new_lst, keep_mask] = remove_outliers_cpp(feat_list, 3.0f);
    if (new_lst.size() != original_size) {
        std::cout << "[GlobalID] Outlier detected: " << original_size - new_lst.size() << " from GID " << pure_gid << "/"
                  << type << std::endl;

        std::vector<std::string> files_to_del;
        // 注意：这里的索引逻辑需要基于原始列表，而不是已删除的列表
        for (size_t i = 0; i < keep_mask.size(); ++i) {
            if (!keep_mask[i]) {
                char filename[32];
                sprintf(filename, "%02d.jpg", static_cast<int>(i));
                // When removing, we don't know the creation reason anymore, so we must assume the path could be modified.
                // A robust solution would be to scan, but for now, we assume we delete from the *original* name path.
                // This is a limitation: if a GID is created with a reason and then has outliers removed, this might fail.
                // A better approach is to not modify path but have a separate field. Given the constraints, we proceed.
                // Let's assume for now deletion works on pure GID paths.
                files_to_del.push_back((std::filesystem::path(SAVE_DIR) / pure_gid / type / filename).string());
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
GlobalID::can_update_proto(const std::string &gid, const std::vector<float> &face_f, const std::vector<float> &body_f, bool is_face_only_mode) {
    if (!bank_faces.count(gid)) return 0;
    if (!bank_faces[gid].empty() && !face_f.empty() && sim_vec(face_f, avg_feats(bank_faces[gid])) < FACE_THR_STRICT)
        return -1;
    // 只有在提供了 body 特征时，才进行 body 相关的一致性检查。
    // 这修复了在 face-only 模式下，从数据库加载的、没有 body 原型的旧 GID 无法通过检查的问题。
    // 如果不是 face-only 模式，并且提供了 body 特征时，才进行 body 相关的一致性检查。
    if (!is_face_only_mode && !body_f.empty()) {
        if (!bank_bodies.count(gid) ||
            (!bank_bodies.at(gid).empty() && sim_vec(body_f, avg_feats(bank_bodies.at(gid))) < BODY_THR_STRICT))
            return -2;
    }
    return 0;
}

void GlobalID::bind(const std::string &gid, const std::string &tid, double current_ts, GstClockTime current_ts_gst,
                    const TrackAgg &agg, FeatureProcessor *fp, const std::string &creation_reason, bool increment_n) {
    auto [face_f, face_p] = agg.main_face_feat_and_patch();
    auto [body_f, body_p] = agg.main_body_feat_and_patch();

    _add_or_update_prototype(bank_faces[gid], face_f, face_p, gid, "faces", fp, creation_reason);
    _add_or_update_prototype(bank_bodies[gid], body_f, body_p, gid, "bodies", fp, creation_reason);

    if (increment_n) {
        auto &v = tid_hist[gid];
        if (std::find(v.begin(), v.end(), tid) == v.end()) v.push_back(tid);
    }
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
        // 如果需要人脸特征进行比对(w_face > 0)，但当前GID的人脸库为空，则跳过
        if (w_face > 1e-6f && face_pool.empty()) continue;
        // 如果需要人体特征进行比对(w_body > 0)，但当前GID的人体库为空，则跳过
        if (w_body > 1e-6f && (!bank_bodies.count(gid) || bank_bodies.at(gid).empty())) continue;

        float face_sim = face_f.empty() ? 0.f : sim_vec(face_f, avg_feats(face_pool));
        // 仅在需要时计算body相似度
        float body_sim = (body_f.empty() || w_body < 1e-6f) ? 0.f : sim_vec(body_f, avg_feats(bank_bodies.at(gid)));
        float sc = w_face * face_sim + w_body * body_sim;
        if (sc > best_score) {
            best_score = sc;
            best_gid = gid;
        }
    }
    return {best_gid, best_score};
}

nlohmann::json FeatureProcessor::_load_or_create_config() {
    nlohmann::json config;

    if (std::filesystem::exists(CONFIG_FILE_PATH)) {
        try {
            std::ifstream ifs(CONFIG_FILE_PATH);
            ifs >> config;
            std::cout << "Successfully loaded configuration from: " << CONFIG_FILE_PATH << std::endl;
        } catch (const nlohmann::json::parse_error& e) {
            std::cerr << "Warning: Failed to parse config file '" << CONFIG_FILE_PATH << "'. Error: " << e.what()
                      << ". Using default values." << std::endl;
            config = nlohmann::json{}; // Reset to empty json on parse error
        }
    } else {
        std::cout << "Info: Configuration file '" << CONFIG_FILE_PATH
                  << "' not found. Creating a default config file with default values." << std::endl;

        // 创建一个包含默认参数的JSON对象
        nlohmann::json default_config;
        default_config["alarm_dup_thr"] = 1.0f;

        // 新增: 将姿态和人脸检测阈值参数化
        default_config["face_det_min_score_face_only"] = 0.85f;
        default_config["pose_yaw_th"] = 30.0;
        default_config["pose_roll_th"] = 25.0;
        default_config["pose_pitch_ratio_lower_th"] = 0.6;
        default_config["pose_pitch_ratio_upper_th"] = 1.0;
        // 新增: 全局配置，同一GID两次有效识别之间的最小间隔 (秒)。值为0表示禁用。
        default_config["gid_recognition_cooldown_s"] = 0;

        // 将默认配置写入文件
        try {
            std::ofstream ofs(CONFIG_FILE_PATH);
            ofs << default_config.dump(4); // 使用4个空格缩进，使其更易读
            ofs.close();
            std::cout << "Successfully created a default config file at '" << CONFIG_FILE_PATH << "'." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error: Failed to create default config file. Reason: " << e.what() << std::endl;
        }
        // 无论文件是否创建成功，都为本次运行加载默认配置，以防止传入一个空的（null）JSON对象。
        config = default_config;
    }
    return config;
}

// 修改: 更新构造函数以匹配新的签名，并使用成员变量存储路径
FeatureProcessor::FeatureProcessor(const std::string &reid_model_path,
                                   const std::string &face_det_model_path,
                                   const std::string &face_rec_model_path,
                                   const std::string &mode,
                                   const std::string &device,
                                   const std::string &feature_cache_path,
                                   bool use_fid_time,
                                   bool enable_alarm_saving,
                                   bool processing_enabled,
                                   bool enable_feature_caching,
                                   bool clear_db_on_startup)
        : m_reid_model_path(reid_model_path),
          m_face_det_model_path(face_det_model_path),
          m_face_rec_model_path(face_rec_model_path),
          mode_(mode), use_fid_time_(use_fid_time), device_(device),
          feature_cache_path_(feature_cache_path),
          m_enable_alarm_saving(enable_alarm_saving),
          m_processing_enabled(processing_enabled),
          m_enable_feature_caching(enable_feature_caching),
          m_clear_db_on_startup(clear_db_on_startup) {

    nlohmann::json boundary_config = _load_or_create_config();
    // 新增：从配置中读取重复报警过滤阈值, 如果未提供，默认为 1.0 (禁用)
    m_alarm_dup_thr = boundary_config.value("alarm_dup_thr", 1.0f);
    // 新增: 从配置中加载姿态估计和人脸检测的阈值
    m_face_det_min_score_face_only = boundary_config.value("face_det_min_score_face_only", 0.85f);
    m_pose_yaw_th = boundary_config.value("pose_yaw_th", 30.0);
    m_pose_roll_th = boundary_config.value("pose_roll_th", 25.0);
    m_pose_pitch_ratio_lower_th = boundary_config.value("pose_pitch_ratio_lower_th", 0.6);
    m_pose_pitch_ratio_upper_th = boundary_config.value("pose_pitch_ratio_upper_th", 1.0);
    // 新增：从配置中读取GID识别冷却时间(秒)，并转换为毫秒
    long long cooldown_s = boundary_config.value("gid_recognition_cooldown_s", 0LL);
    m_gid_recognition_cooldown_ms = cooldown_s * 1000;

    std::cout << "FeatureProcessor initialized in '" << mode_ << "' mode. Alarm saving is "
              << (m_enable_alarm_saving ? "ENABLED" : "DISABLED") << "." << std::endl;
    std::cout << ">>> Alarm duplication filter threshold (alarm_dup_thr) set to: " << m_alarm_dup_thr << std::endl;
    std::cout << ">>> Face-only det score threshold set to: " << m_face_det_min_score_face_only << std::endl;
    std::cout << ">>> Pose Yaw threshold set to: " << m_pose_yaw_th << std::endl;
    std::cout << ">>> Pose Roll threshold set to: " << m_pose_roll_th << std::endl;
    std::cout << ">>> Pose Pitch Ratio threshold set to: [" << m_pose_pitch_ratio_lower_th << ", " << m_pose_pitch_ratio_upper_th << "]" << std::endl;
    std::cout << ">>> GID Recognition Cooldown set to: " << cooldown_s << " s" << std::endl;


    if (mode_ == "realtime") {
        std::cout << "Loading ReID and Face models for feature extraction..." << std::endl;
        bool use_gpu = (device == "cuda"); // Note: DLA is Jetson specific, this flag is less relevant here.
        try {
            // 初始化将在主线程中使用的 ReID 模型，分配给 DLA Core 1
            reid_model_ = std::make_unique<PersonReidDLA>(m_reid_model_path, REID_INPUT_WIDTH, REID_INPUT_HEIGHT,
                                                          1,
                                                          "/home/nvidia/VSCodeProject/smartboxcore/models/reid_model.dla.engine");
            std::cout << "Initialized Re-ID model on DLA 1 for main thread." << std::endl;
            face_analyzer_ = std::make_unique<FaceAnalyzer>(m_face_det_model_path, m_face_rec_model_path);
            std::string provider = use_gpu ? "GPU" : "DLA";
            // Dedicate DLA core 0 to the FaceAnalyzer.
            face_analyzer_->prepare(provider, FACE_DET_MIN_SCORE, cv::Size(640, 640));
        } catch (const std::exception &e) {
            std::cerr << "[FATAL] Failed to load models in realtime mode: " << e.what() << std::endl;
            throw;
        }

        // The feature cache can optionally be written in realtime mode. Check the switch.
        if (m_enable_feature_caching) {
            if (!feature_cache_path_.empty()) {
                auto parent_path = std::filesystem::path(feature_cache_path_).parent_path();
                if (!parent_path.empty()) std::filesystem::create_directories(parent_path);
                std::cout << "Feature caching to JSON is ENABLED." << std::endl;
            } else {
                std::cout << "Feature caching is requested but feature_cache_path is empty. Caching DISABLED."
                          << std::endl;
            }
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
    if (m_clear_db_on_startup && std::filesystem::exists(DB_PATH)) {
        std::cout << "Switch 'clear_db_on_startup' is ON. Removing existing database: " << DB_PATH << std::endl;
        std::filesystem::remove(DB_PATH);
        // 在从数据库加载状态前，先清空并重建文件系统缓存，以确保两者同步
        if (std::filesystem::exists(SAVE_DIR)) {
            std::filesystem::remove_all(SAVE_DIR);
        }
        std::filesystem::create_directories(SAVE_DIR);
    }

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
    // 新增：在析构时，如果开启了特征缓存，则将内存中的特征缓存写入文件
    if (m_enable_feature_caching && !feature_cache_path_.empty() && !features_to_save_.is_null()) {
        std::cout << "\nSaving " << features_to_save_.size()
                  << " frames of features to " << feature_cache_path_ << " ..." << std::endl;
        try {
            std::ofstream ofs(feature_cache_path_);
            ofs << features_to_save_.dump(4); // 使用 dump(4) 进行格式化输出，便于阅读
            ofs.close();
            std::cout << "Feature cache saved successfully." << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "Error saving feature cache: " << e.what() << std::endl;
        }
    }

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
                    auto full_path = std::filesystem::path(SAVE_DIR) / task.path_suffix;
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
                            // The path_suffix is now "GID_reason.../type/idx.jpg", so we need to adjust finding the type
                            size_t type_start_pos = task.path_suffix.find('/');
                            if (type_start_pos != std::string::npos) {
                                size_t type_end_pos = task.path_suffix.find('/', type_start_pos + 1);
                                if (type_end_pos != std::string::npos) {
                                    proto_type = task.path_suffix.substr(type_start_pos + 1, type_end_pos - (type_start_pos + 1));
                                    idx_str = task.path_suffix.substr(type_end_pos + 1);
                                }
                            }
                            try { proto_idx = std::stoi(idx_str.substr(0, idx_str.find('.'))); } catch (...) {}
                        }

                        if (proto_idx != -1) {
                            const char* sql = "INSERT OR REPLACE INTO prototypes (gid, type, idx, feature, image) VALUES (?, ?, ?, ?, ?);";
                            sqlite3_stmt *stmt;
                            if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) == SQLITE_OK) {
                                std::vector<uchar> img_buf;
                                // 使用上面转换后的 bgr_image
                                if (!bgr_image.empty()) {
                                    cv::imencode(".jpg", bgr_image, img_buf);
                                }
                                sqlite3_bind_text(stmt, 1, task.gid.c_str(), -1, SQLITE_STATIC); // task.gid is pure
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
                    std::string dir_name =
                            task.gid + "_" + task.tid_str + "_n" + std::to_string(task.n) + "_" + task.timestamp;
                    std::filesystem::path dst_dir = std::filesystem::path(ALARM_DIR) / dir_name;

                    std::filesystem::path src_dir = std::filesystem::path(SAVE_DIR) / task.gid;
                    if (std::filesystem::exists(src_dir))
                        std::filesystem::copy(src_dir, dst_dir, std::filesystem::copy_options::recursive |
                                                                std::filesystem::copy_options::overwrite_existing);

                    std::filesystem::path seq_face_dir = dst_dir / "agg_sequence/face";
                    std::filesystem::path seq_body_dir = dst_dir / "agg_sequence/body";
                    std::filesystem::create_directories(seq_face_dir);
                    std::filesystem::create_directories(seq_body_dir);

                    // WORKAROUND: 不再保存实时轨迹的agg_sequence，而是直接从已备份到告警目录的原型中复制一份，
                    // 以确保告警文件夹内的 bodies, faces, agg_sequence 三者图片绝对一致。
                    // 这个改动会丢失 agg_sequence 作为“现场轨迹快照”的原始意义，但能快速解决图片不匹配的问题。
                    std::filesystem::path proto_face_dir_in_alarm = dst_dir / "faces";
                    if (std::filesystem::exists(proto_face_dir_in_alarm)) {
                        std::filesystem::copy(proto_face_dir_in_alarm, seq_face_dir,
                                              std::filesystem::copy_options::recursive |
                                              std::filesystem::copy_options::overwrite_existing);
                    }
                    std::filesystem::path proto_body_dir_in_alarm = dst_dir / "bodies";
                    if (std::filesystem::exists(proto_body_dir_in_alarm)) {
                        std::filesystem::copy(proto_body_dir_in_alarm, seq_body_dir,
                                              std::filesystem::copy_options::recursive |
                                              std::filesystem::copy_options::overwrite_existing);
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
                    std::string dir_name =
                            task.gid + "_" + task.tid_str + "_n" + std::to_string(task.n) + "_" + task.timestamp;
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
                    std::string dir_name =
                            task.gid + "_" + task.tid_str + "_n" + std::to_string(task.n) + "_" + task.timestamp;
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
            // Provide default values for is_frontal and score when loading from cache, as they are not available.
            if (!ff.empty()) agg.add_face(ff, patch.clone(), false, 0.0f);
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
                                double frame_timestamp, int alarm_record_thresh) {
    auto cur_rep = _gid_fused_rep(gid);
    if (cur_rep.empty()) return std::nullopt;

    // 检查当前 GID 是否与已有的“原始报警GID”相似
    for (const auto &[ogid, rep]: alarm_reprs) {
        // 关键修改：仅当 GID 不同但特征相似时，才抑制报警。
        // 这允许同一个 GID 在不同时间点被多次识别并触发报警。
        if (gid != ogid && sim_vec(cur_rep, rep) >= m_alarm_dup_thr) {
            // 如果是不同的 GID 但特征相似，则抑制本次报警。
            std::cout << "\n[ALARM SUPPRESSED] GID " << gid << " is similar to already alarmed GID "
                      << ogid << " (Similarity threshold: " << m_alarm_dup_thr << "). Suppressing this alarm." << std::endl;
            return std::nullopt; // 返回空 optional 来抑制报警
        }
    }

    // 如果程序执行到这里，说明这是一个全新的、不重复的报警 GID (或者重复了一个未被记录的GID)。
    // 根据 N 值决定是否将其注册为新的“原始报警 GID”以用于未来的去重。
    if (n >= alarm_record_thresh) {
//        std::cout << "\n[ALARM] New original alarmer registered for deduplication: " << gid
//                  << " (n=" << n << " > threshold=" << alarm_record_thresh << ")." << std::endl;
        alarmed.insert(gid);
        alarm_reprs[gid] = cur_rep;
    } else {
//        std::cout << "\n[ALARM] GID " << gid << " triggered, but not registered for deduplication (n=" << n
//                  << " <= threshold=" << alarm_record_thresh << ")." << std::endl;
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
        task.gid = gid; // 既然是新报警，上报的 GID 就是当前 GID
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

    return std::make_tuple(gid, timestamp_str, was_newly_saved);
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
        double duration,
        const std::vector<Detection> &dets,
        const std::string &stream_id,
        const cv::Mat &body_p,
        const cv::Mat &face_p,
        std::vector<std::tuple<std::string, std::string, std::string, int, bool>> &triggered_alarms_this_frame,
        float w_face,
        float face_det_score,
        float face_clarity,
        bool is_face_only_mode,
        const std::vector<float>& current_face_feat,
        float current_match_thr) {

    /* -------- 动态计算当前 GID 是否处于冷却期 -------- */
    bool is_on_cooldown = false;
    long long current_gid_cooldown_ms = 0;
    if (config.gid_recognition_cooldown_s.has_value()) {
        current_gid_cooldown_ms = config.gid_recognition_cooldown_s.value() * 1000;
    } else {
        current_gid_cooldown_ms = m_gid_recognition_cooldown_ms;
    }
    double gid_cooldown_threshold = 0.0;
    if (current_gid_cooldown_ms > 0) {
        gid_cooldown_threshold = use_fid_time_
                                 ? (double) current_gid_cooldown_ms / 1000.0 * FPS_ESTIMATE
                                 : (double) current_gid_cooldown_ms / 1000.0;

        if (gid_last_recognized_time.count(gid)) {
            double delta = now_stamp - gid_last_recognized_time.at(gid);
            if (delta < (gid_cooldown_threshold - MAX_TID_IDLE_SEC)) {
                is_on_cooldown = true;
            }
#ifdef ENABLE_COOLDOWN_DEBUG_PRINTS
            std::cout << "[COOLDOWN_DEBUG] TID: " << tid_str << ", GID: " << gid
                      << " delta=" << delta << (use_fid_time_ ? " frames" : "s")
                      << ", threshold=" << gid_cooldown_threshold
                      << " => on_cooldown=" << (is_on_cooldown ? "true" : "false") << std::endl;
#endif
        }
    }

    // 如果当前 GID 在本次调用传入的白名单中，则直接返回，不触发任何报警逻辑。
    if (config.whitelist_gids.count(gid)) {
        std::cout << "\n[ALARM SKIPPED] GID " << gid << " (TID " << tid_str << ") is in the whitelist." << std::endl;
        // 如果是在纯人脸模式下，且 GID 在白名单中，则将其识别次数（tid_hist）清零。
        // 这样，当该 GID 从白名单中移除后，其识别次数 n 会从 0 重新开始累计。
        if (is_face_only_mode && gid_mgr.tid_hist.count(gid)) {
            gid_mgr.tid_hist.at(gid).clear();
        }
        return;
    }

    // 获取该摄像头的徘徊时间配置，如果未设置则默认为0
    long long current_alarm_duration_ms = 0;
    if (config.alarmDuration_ms_by_cam.count(stream_id)) {
        current_alarm_duration_ms = config.alarmDuration_ms_by_cam.at(stream_id);
    }

    // 计算徘徊时间阈值 (单位与 now_stamp 一致)
    double alarmDuration_threshold = 0.0;
    if (current_alarm_duration_ms > 0) {
        if (use_fid_time_) {
            alarmDuration_threshold = (double)current_alarm_duration_ms / 1000.0 * FPS_ESTIMATE;
        } else {
            alarmDuration_threshold = (double)current_alarm_duration_ms / 1000.0;
        }
    }

    // 检查是否满足徘徊时间，如果不满足，则直接返回，不触发报警
    alarmDuration_threshold = 0.0;  // disable
    if (alarmDuration_threshold > 0 && duration < alarmDuration_threshold) {
        // 识别结果依然有效，只是不触发报警
//        double duration_sec = use_fid_time_ ? (duration / FPS_ESTIMATE) : duration;
//        double threshold_sec = use_fid_time_ ? (alarmDuration_threshold / FPS_ESTIMATE) : alarmDuration_threshold;
//        std::cout << "\n[ALARM SKIPPED] GID " << gid << " (TID " << tid_str << ") did not meet loitering time."
//                  << " Duration: " << std::fixed << std::setprecision(2) << duration_sec << "s"
//                  << " < Threshold: " << threshold_sec << "s." << std::endl;
        return;
    }

    // ======================= 【MODIFIED: 移除对当前帧人脸检测的依赖】 =======================
    // 新增：如果要求有人脸（例如纯人脸模式），但历史轨迹中从未成功提取过代表人脸，则不触发报警。
    // 这个检查是基于历史的(face_p)，而不是当前帧的检测结果。
    if (is_face_only_mode && face_p.empty()) {
//        std::cout << "\n[ALARM SKIPPED] GID " << gid << " (TID " << tid_str << ") (is_face_only_mode && face_p.empty())."
//                  << std::endl;
        return;
    }
    // ======================= 【修改结束】 =======================

    int n = gid_mgr.tid_hist.count(gid) ? (int) gid_mgr.tid_hist.at(gid).size() : 0;

    if (n >= config.alarm_cnt_th) {
        // ======================= 【新增：报警一致性校验】 =======================
        // 检查当前帧的人脸特征与 GID 的原型库是否足够相似，防止跟踪器跟错人导致的持续误报
        if (!current_face_feat.empty() && gid_mgr.bank_faces.count(gid) && !gid_mgr.bank_faces.at(gid).empty()) {
            auto gid_prototype_face_feat = avg_feats(gid_mgr.bank_faces.at(gid));
            float consistency_score = sim_vec(current_face_feat, gid_prototype_face_feat);

            if (consistency_score < current_match_thr) { // 使用动态阈值
                std::cout << "\n[ALARM SUPPRESSED] TID " << tid_str
                          << " failed consistency check for GID " << gid
                          << ". Live Score: " << std::fixed << std::setprecision(4) << consistency_score
                          << " < Threshold: " << current_match_thr << std::endl;
                return; // 校验失败，抑制本次报警
            }
        }
        // ======================= 【新增结束】 =======================

        if (auto alarm_data_opt = trigger_alarm(tid_str, gid, n, agg, now_stamp, config.alarm_record_thresh)) {
            auto &[gid_to_alarm, timestamp, was_newly_saved] = *alarm_data_opt;

            AlarmTriggerInfo alarm_info;
            alarm_info.gid = gid_to_alarm;
            alarm_info.tid_str = tid_str;
            alarm_info.first_seen_timestamp = gid_mgr.first_seen_ts.count(gid_to_alarm)
                                              ? gid_mgr.first_seen_ts.at(gid_to_alarm)
                                              : now_stamp_gst;
            // 新增：记录最后一次看到的时间戳，即当前帧的时间戳
            alarm_info.last_seen_timestamp = now_stamp_gst;
            for (const auto &det: dets) {
                if (stream_id + "_" + std::to_string(det.id) == tid_str) {
                    alarm_info.person_bbox = det.tlwh;
                    break;
                }
            }

            // ======================= 【MODIFIED: 修正业务n值递增逻辑】 =======================
            int business_n = 0;
            bool recalculate_n = true; // 默认需要重新计算n

            // 检查此TID之前是否触发过报警
            if (tid_to_business_n_.count(tid_str)) {
                // 如果触发过，检查当时关联的GID是否与现在的GID相同
                // 新增：如果当前 GID 处于冷却状态，即使是新的 TID，也强制复用 n 值，不重新计算
                if (is_on_cooldown) {
                     recalculate_n = false;
                     business_n = gid_alarm_business_counts_.count(gid) ? gid_alarm_business_counts_.at(gid) : 0;
                     std::cout << "\n[n_DEBUG] Reusing n=" << business_n << " for GID " << gid << " on new TID " << tid_str << " because GID is in cooldown." << std::endl;
                }
                else if (tid_last_gid_for_n_.count(tid_str) && tid_last_gid_for_n_.at(tid_str) == gid) {
                    // GID未变，直接复用之前的n值
                    business_n = tid_to_business_n_.at(tid_str);
                    recalculate_n = false;
                    std::cout << "\n[n_DEBUG] Reusing n=" << business_n << " for same GID " << gid << " on TID: " << tid_str << std::endl;
                } else {
                    // GID发生了变化（TID漂移），需要为新的GID重新计算n
                    std::cout << "\n[n_DEBUG] GID changed for TID " << tid_str << ". From "
                                << (tid_last_gid_for_n_.count(tid_str) ? tid_last_gid_for_n_.at(tid_str) : "N/A")
                                << " to " << gid << ". Recalculating n." << std::endl;
                }
            }
            // 新增：如果当前 GID 处于冷却状态，即使是新的 TID，也强制复用 n 值，不重新计算
            else if (is_on_cooldown) {
                recalculate_n = false;
                business_n = gid_alarm_business_counts_.count(gid) ? gid_alarm_business_counts_.at(gid) : 0;
                std::cout << "\n[n_DEBUG] Reusing n=" << business_n << " for GID " << gid << " on new TID " << tid_str << " because GID is in cooldown." << std::endl;
                tid_to_business_n_[tid_str] = business_n; // 锁定这个TID到复用的n值
            }

            // 如果需要重新计算（例如新TID或TID漂移），或者 business_n 因错误的复用逻辑被置为0，则强制计算。
            // 这可以修复在冷却期间复用一个还未被计数的GID时，business_n为0的bug。
            if (recalculate_n || business_n == 0) {
                // 首次触发报警，或TID关联的GID发生变化，需要计算新的n值
                if (!tid_to_business_n_.count(tid_str)) {
                    std::cout << "\n[n_DEBUG] New alarm TID: " << tid_str << " for GID: " << gid << ". Calculating new n..." << std::endl;
                }
                int last_n_for_gid = gid_alarm_business_counts_.count(gid) ? gid_alarm_business_counts_.at(gid) : 0;
                std::cout << "          - Last recorded n for GID was: " << last_n_for_gid << std::endl;
                business_n = last_n_for_gid + 1;
                std::cout << "          - New continuous n is now: " << business_n << std::endl;
                gid_alarm_business_counts_[gid] = business_n;
                tid_to_business_n_[tid_str] = business_n;
                tid_last_gid_for_n_[tid_str] = gid;
                std::cout << "          - Global count for GID updated. TID " << tid_str << " locked to n=" << business_n << " for GID " << gid << "." << std::endl;
            }

            alarm_info.n = business_n;
            alarm_info.face_clarity_score = face_clarity; // 新增：赋值人脸清晰度分数

//            if (alarm_info.person_bbox.area() > 0)
            {
                triggered_alarms_this_frame.emplace_back(gid, tid_str, timestamp, n, was_newly_saved);
                if (current_frame_face_boxes_.count(tid_str)) {
                    alarm_info.face_bbox = current_frame_face_boxes_.at(tid_str);
                }
                alarm_info.latest_body_patch = !is_face_only_mode ? body_p.clone() : alarm_info.latest_body_patch;
                alarm_info.latest_face_patch = face_p.clone();
                // ======================= 【新增：打印待上报的报警详细信息】 =======================
                std::cout << "\n--- [Preparing to Report Alarm] ---\n"
                          << "  GID: " << alarm_info.gid << "\n"
                          << "  TID: " << alarm_info.tid_str << "\n"
                          << "  First Seen: " << format_ntp_timestamp(alarm_info.first_seen_timestamp) << "\n"
                          << "  Last Seen: " << format_ntp_timestamp(alarm_info.last_seen_timestamp) << "\n"
                          << "  Recognition Count (n): " << alarm_info.n << "\n"
                          << "  Face Clarity: " << std::fixed << std::setprecision(1) << alarm_info.face_clarity_score << "/100\n"
                          << "  Person Bbox: [" << alarm_info.person_bbox.x << ", " << alarm_info.person_bbox.y << ", " << alarm_info.person_bbox.width << ", " << alarm_info.person_bbox.height << "]\n"
                          << "  Face Bbox: [" << alarm_info.face_bbox.x << ", " << alarm_info.face_bbox.y << ", " << alarm_info.face_bbox.width << ", " << alarm_info.face_bbox.height << "]\n"
                          << "  Body Patch Size: " << (!alarm_info.latest_body_patch.empty() ? std::to_string(alarm_info.latest_body_patch.cols) + "x" + std::to_string(alarm_info.latest_body_patch.rows) : "empty") << "\n"
                          << "  Face Patch Size: " << (!alarm_info.latest_face_patch.empty() ? std::to_string(alarm_info.latest_face_patch.cols) + "x" + std::to_string(alarm_info.latest_face_patch.rows) : "empty") << "\n"
                          << "-------------------------------------\n";
                // ======================= 【新增结束】 =======================
                output.alarms.push_back(alarm_info);
            }
        }
    } else {
//        std::cout << "\n[ALARM PENDING] GID " << gid << " (TID " << tid_str << ") recognition count n=" << n
//                  << " is less than alarm threshold n_th=" << config.alarm_cnt_th << "." << std::endl;
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

    // 根据配置确定用于探测的融合权重
    float w_face = W_FACE;
    float w_body = W_BODY;
    if (!face_enabled) {
        w_face = 0.0f;
        w_body = 1.0f;
    } else {
        if (config.face_weight_by_cam.count(cam_id)) {
            w_face = config.face_weight_by_cam.at(cam_id);
        }
        if (config.reid_weight_by_cam.count(cam_id)) {
            w_body = config.reid_weight_by_cam.at(cam_id);
        }
    }

    // 新增: 当仅使用人脸比对时(w_face=1.0)，提高人脸检测的置信度阈值
    const bool is_face_only_mode = (w_face >= 0.999f); // 使用小公差进行浮点数比较
    float current_face_det_min_score = is_face_only_mode ? m_face_det_min_score_face_only : FACE_DET_MIN_SCORE;
    int current_min_face_4_gid = is_face_only_mode ? 2 : MIN_FACE4GID;

    ProcessOutput output;
    std::vector<std::tuple<std::string, std::string, std::string, int, bool>> triggered_alarms_this_frame; // <gid, tid_str, timestamp, n, was_newly_saved>
    // 新增：延迟更新冷却时间戳，以保证单帧内状态一致性
    std::set<std::string> gids_recognized_this_frame;

    current_frame_face_boxes_.clear(); // 每帧开始时清空
    current_frame_face_scores_.clear(); // 清空人脸置信度表
    current_frame_face_clarity_.clear(); // 新增：清空人脸清晰度表
    current_frame_face_features_.clear(); // 新增：清空当前帧的人脸特征

    const auto &stream_id = cam_id;

    // --- 确定当前帧的匹配阈值 ---
    // 1. 根据灵敏度设置一个基础阈值
    // 默认灵敏度为 5 (中等)，对应阈值 0.5f
    int sensitivity = 5;
    if (config.sensitivity_by_cam.count(stream_id)) {
        sensitivity = config.sensitivity_by_cam.at(stream_id);
    }
    // 确保灵敏度在 1 到 10 的范围内
    sensitivity = std::max(1, std::min(10, sensitivity));

    // 10级灵敏度映射：1(最高) -> 0.1, ..., 10(最低) -> 1.0
    // 公式: T = 0.1f + (S - 1) * 0.1f
    float base_match_thr = 0.1f + static_cast<float>(sensitivity - 1) * 0.1f;

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
    // 声明冷却状态变量在循环外层，以便后续传递
    bool on_cooldown = false;
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

    // 优先使用实时参数，如果未提供，则回退到初始化时加载的默认参数
    long long current_gid_cooldown_ms;
    if (config.gid_recognition_cooldown_s.has_value()) {
        // 实时参数(秒)存在，将其转换为毫秒
        current_gid_cooldown_ms = config.gid_recognition_cooldown_s.value() * 1000;
    } else {
        // 实时参数不存在，使用从 config.json 加载的默认值 (已是毫秒)
        current_gid_cooldown_ms = m_gid_recognition_cooldown_ms;
    }
    double gid_cooldown_threshold = 0.0;
    if (current_gid_cooldown_ms > 0) {
        if (use_fid_time_) {
            gid_cooldown_threshold = (double)current_gid_cooldown_ms / 1000.0 * FPS_ESTIMATE;
        } else {
            gid_cooldown_threshold = (double)current_gid_cooldown_ms / 1000.0;
        }
    }

    if (mode_ == "realtime") {
        nlohmann::json extracted_features_for_this_frame;
        const int H = full_frame.rows;
        const int W = full_frame.cols;

        // 新增：将GpuMat下载到CpuMat，因为FaceAnalyzer现在使用OpenCV DNN，需要cv::Mat
        cv::Mat cpu_frame;
        full_frame.download(cpu_frame);

        // 1. Re-ID特征提取
        // 仅当需要进行人体特征比对时才执行 (w_body > 0, 即 w_face < 1.0)
//        if (!is_face_only_mode)
        {
            // 首先在主线程中执行Re-ID特征提取 (使用DLA Core 1)
            for (const auto &det: dets) {
                if (det.class_id != 0) continue;

                cv::Rect roi = cv::Rect(det.tlwh) & cv::Rect(0, 0, W, H);
                if (roi.width <= 0 || roi.height <= 0) continue;

                cv::cuda::GpuMat gpu_patch = full_frame(roi);
                cv::Mat patch;
                gpu_patch.download(patch);

                if (!is_long_patch(patch)) continue;

                // --- HANG FIX: Wrap Re-ID DLA call with a timeout ---
                cv::cuda::GpuMat feat_mat_gpu;
                auto fut_reid = std::async(std::launch::async, [&]() {
                    return reid_model_->extract_feat(gpu_patch);
                });

                if (fut_reid.wait_for(std::chrono::milliseconds(300)) == std::future_status::ready) {
                    feat_mat_gpu = fut_reid.get();
                } else {
                    std::cerr << "Re-ID DLA inference timeout for TID " << det.id << ", cam_id: " << stream_id << std::endl;
                    continue; // Skip this detection if DLA hangs
                }
                if (feat_mat_gpu.empty()) continue;

                cv::Mat feat_mat_cpu;
                feat_mat_gpu.download(feat_mat_cpu);
                std::vector<float> body_feat(feat_mat_cpu.begin<float>(), feat_mat_cpu.end<float>());
                std::string tid_str = stream_id + "_" + std::to_string(det.id);
                first_seen_tid.try_emplace(tid_str, now_stamp);
                agg_pool[tid_str].add_body(body_feat, det.score, patch.clone());
                last_seen[tid_str] = now_stamp;
                if (m_enable_feature_caching) {
                    extracted_features_for_this_frame[tid_str]["body_feat"] = body_feat;
                }
            }
        }

        // 2. 人脸分析 (使用 DLA Core 0)
        std::vector<Face> internal_face_info;
        if (face_enabled && face_analyzer_) {
            // --- Wrap Face Detection call with a timeout ---
            auto fut_detect = std::async(std::launch::async, [&]() {
                return face_analyzer_->detect(cpu_frame); // 使用CPU Mat
            });
            if (fut_detect.wait_for(std::chrono::milliseconds(300)) == std::future_status::ready) {
                internal_face_info = fut_detect.get();
            } else {
                std::cerr << "Face detection timeout for cam_id: " << stream_id << std::endl;
                // Timeout: internal_face_info will remain empty, gracefully skipping subsequent logic.
            }

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

                    // ======================= 【MODIFIED: 分阶段人脸质量检查】 =======================
                    bool is_frontal = true; // 在非 face_only 模式下，默认所有脸都合格
                    if (is_face_only_mode) {
                        const float RELAXED_FACE_SCORE_THR = 0.75f;
                        // 阶段一：使用宽松的分数阈值收集人脸，为后续严格检查做准备
                        if (face_global_coords.det_score < RELAXED_FACE_SCORE_THR) continue;
                        // 获取姿态标志，但在此阶段不进行过滤
                        is_frontal = is_frontal_face_pnp(face_global_coords.kps, cpu_frame.size(), m_pose_yaw_th, m_pose_roll_th, m_pose_pitch_ratio_lower_th, m_pose_pitch_ratio_upper_th);
                    } else {
                        // 在混合模式下，保持原有的分数检查逻辑
                        if (face_global_coords.det_score < current_face_det_min_score) continue;
                    }
                    // ======================= 【修改结束】 =======================

                    cv::Rect face_roi(face_global_coords.bbox);
                    face_roi &= cv::Rect(0, 0, W, H);
                    if (face_roi.width < 32 || face_roi.height < 32) continue;

                    try {
                        // --- Wrap Face Embedding call with a timeout ---
                        auto fut_face_emb = std::async(std::launch::async, [&]() {
                            // This lambda modifies the captured reference face_global_coords
                            face_analyzer_->get_embedding(cpu_frame, face_global_coords); // 使用CPU Mat
                        });

                        if (fut_face_emb.wait_for(std::chrono::milliseconds(300)) != std::future_status::ready) {
                            std::cerr << "Face Embedding inference timeout for TID " << det.id << ", cam_id: "
                                      << stream_id << std::endl;
                            continue; // Skip this face if DLA hangs
                        }
                        // call get() to ensure completion and propagate exceptions if any
                        fut_face_emb.get();

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
                        current_frame_face_scores_[tid_str] = face_global_coords.det_score; // 记录置信度
                        // 新增：计算并记录人脸清晰度
                        float clarity = calculate_clarity_score(face_patch);
                        current_frame_face_clarity_[tid_str] = clarity;
                        gpu_face_patch.download(face_patch);
                        agg_pool[tid_str].add_face(f_emb, face_patch.clone(), is_frontal, face_global_coords.det_score);
                        current_frame_face_features_[tid_str] = f_emb; // 新增：暂存当前帧人脸特征
                        // ======================= 【修改结束】 =======================
                        last_seen[tid_str] = now_stamp;
                        if (m_enable_feature_caching) {
                            extracted_features_for_this_frame[tid_str]["face_feat"] = f_emb;
                        }
                    } catch (const std::exception &e) {
                        std::cerr << "Exception during face processing for TID " << det.id << ": " << e.what()
                                  << std::endl;
                        continue;
                    }
                }
            }
        }

        // 3. 保存所有本帧提取的特征到缓存文件
        if (m_enable_feature_caching && !extracted_features_for_this_frame.is_null()) {
            features_to_save_[std::to_string(fid)] = extracted_features_for_this_frame;
        }

    } else if (mode_ == "load") { // 注意：这里的 now_stamp 是基于 fid 的
        _load_features_from_cache(cam_id, fid, full_frame, dets, now_stamp);
    }

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
        if (current_alarm_duration_ms > 0 && duration < alarmDuration_threshold and mode_ == "load") {
            output.mp[s_id][tid_num] = {tid_str + "_-8_wait_d", -1.f, 0};
            continue;
        }

        // 如果不是纯人脸模式，则检查body特征数量
        if (!is_face_only_mode && (int) agg.body.size() < MIN_BODY4GID) {
            std::string gid_str = tid_str + "_-1_b_" + std::to_string(agg.body.size());
            output.mp[s_id][tid_num] = {gid_str, -1.f, 0};
            continue;
        }
        if (face_enabled && (int) agg.face.size() < current_min_face_4_gid) {
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
        // 如果不是纯人脸模式，则ReID特征是必须的
        if (!is_face_only_mode && body_f.empty()) {
            output.mp[s_id][tid_num] = {tid_str + "_-2_b", -1.f, 0};
            continue;
        }

        // ======================= 【新增: 获取当前帧特征以备后用】 =======================
        std::vector<float> current_face_feat;
        if (current_frame_face_features_.count(tid_str)) {
            current_face_feat = current_frame_face_features_.at(tid_str);
        }
        // ======================= 【新增结束】 =======================

        auto [cand_gid, score] = gid_mgr.probe(face_f, body_f, w_face, w_body);

        if (face_enabled && !face_f.empty()) {
            std::cout << "[FaceMatch] cam:" << stream_id
                      << " tid:" << tid_num
                      << " cand_gid:" << (cand_gid.empty() ? "None" : cand_gid)
                      << " score:" << std::fixed << std::setprecision(4) << score
                      << std::endl;
        }

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
            int flag_code = gid_mgr.can_update_proto(cand_gid, face_f, body_f, is_face_only_mode);
            if (state.count >= CANDIDATE_FRAMES && flag_code == 0) {
                #ifdef ENABLE_COOLDOWN_DEBUG_PRINTS
                if (gid_cooldown_threshold > 0) {
                     std::cout << "\n[COOLDOWN_DEBUG] TID: " << tid_str << ", GID: " << cand_gid
                               << ", Cooldown Check ACTIVE (threshold=" << std::fixed << std::setprecision(3) << gid_cooldown_threshold
                               << (use_fid_time_ ? " frames" : "s") << ")\n";
                }
                #endif
                // 重置冷却状态，为当前TID的检查做准备
                on_cooldown = false;
                if (gid_cooldown_threshold > 0 && gid_last_recognized_time.count(cand_gid)) {
                    double time_since_last_rec = now_stamp - gid_last_recognized_time.at(cand_gid);

                    #ifdef ENABLE_COOLDOWN_DEBUG_PRINTS
                    std::cout << "                 > Now: " << now_stamp << ", Last Rec @: " << gid_last_recognized_time.at(cand_gid)
                              << ", Delta: " << time_since_last_rec << "\n";
                    std::cout << "                 > Check: Is Delta (" << time_since_last_rec << ") < Threshold (" << gid_cooldown_threshold << ") ?\n";
                    #endif

                    if (time_since_last_rec < gid_cooldown_threshold) {
                        on_cooldown = true;
                    }
                }

                #ifdef ENABLE_COOLDOWN_DEBUG_PRINTS
                if (gid_cooldown_threshold > 0) {
                    std::cout << "                 > Result: on_cooldown = " << (on_cooldown ? "true" : "false") << "\n";
                }
                #endif

                // 核心修改：无论是否在冷却期，只要成功识别并绑定，就立即更新其“最后出现时间”。
                // 这会持续刷新冷却计时器，直到目标消失超过冷却时间为止。
                gids_recognized_this_frame.insert(cand_gid);

                // 调用 bind 更新原型，但根据冷却状态决定是否增加 n
                gid_mgr.bind(cand_gid, tid_str, now_stamp, now_stamp_gst, agg, this, "", !on_cooldown); // creation_reason is empty, increment_n is conditional
                tid2gid[tid_str] = cand_gid;
                state.last_bind_fid = fid;
                int n = gid_mgr.tid_hist.count(cand_gid) ? (int) gid_mgr.tid_hist[cand_gid].size() : 0;

                if (!on_cooldown) { // 冷却时间已过或首次识别
                    output.mp[s_id][tid_num] = {s_id + "_" + std::to_string(tid_num) + "_" + cand_gid, score, n};
                } else {
                    // 冷却时间内 (n 未增加)，不重置计时器，但在UI上显示特殊状态
                    output.mp[s_id][tid_num] = {s_id + "_" + std::to_string(tid_num) + "_" + cand_gid + "_-9_cool", score, n};
                }
            } else {
                std::string flag = (flag_code == -1) ? "_-4_ud_f" : (flag_code == -2) ? "_-4_ud_b" : "_-4_c";
                output.mp[s_id][tid_num] = {tid_str + flag, -1.0f, 0};
            }
        } else if (gid_mgr.bank_faces.empty()) {
            // Reason 1: The very first GID
            // ======================= 【MODIFIED: 新增高质量人脸检查点】 =======================
            if (is_face_only_mode) {
                if (agg.count_high_quality_faces(m_face_det_min_score_face_only) < current_min_face_4_gid) {
                    output.mp[s_id][tid_num] = {tid_str + "_-1_f_hq", -1.0f, 0}; // hq: high quality
                    continue;
                }
            }
            // ======================= 【修改结束】 =======================
            std::string new_gid = gid_mgr.new_gid(); // This is the pure GID
            gid_mgr.bind(new_gid, tid_str, now_stamp, now_stamp_gst, agg, this, "first"); // 默认 increment_n=true
            tid2gid[tid_str] = new_gid;
            candidate_state[tid_str] = {new_gid, CANDIDATE_FRAMES, fid};
            new_gid_state[tid_str].last_new_fid = fid;
            gid_last_recognized_time[new_gid] = now_stamp;
            int n = gid_mgr.tid_hist.count(new_gid) ? (int) gid_mgr.tid_hist[new_gid].size() : 0;
            output.mp[s_id][tid_num] = {s_id + "_" + std::to_string(tid_num) + "_" + new_gid, score, n};
        } else if (!cand_gid.empty() && score >= THR_NEW_GID && mode_ == "load") { // Reason 3: Ambiguous, pending for a long time
            ng_state.ambig_count++;
            if (ng_state.ambig_count >= WAIT_FRAMES_AMBIGUOUS && time_since_last_new >= NEW_GID_TIME_WINDOW) {
                std::string new_gid = gid_mgr.new_gid(); // Pure GID
                gid_mgr.bind(new_gid, tid_str, now_stamp, now_stamp_gst, agg, this, "similar_pending"); // 默认 increment_n=true
                tid2gid[tid_str] = new_gid;
                candidate_state[tid_str] = {new_gid, CANDIDATE_FRAMES, fid};
                new_gid_state[tid_str] = {0, fid, 0};
                gids_recognized_this_frame.insert(new_gid);
                int n = gid_mgr.tid_hist.count(new_gid) ? (int) gid_mgr.tid_hist[new_gid].size() : 0;
                output.mp[s_id][tid_num] = {s_id + "_" + std::to_string(tid_num) + "_" + new_gid, score, n};
            } else {
                output.mp[s_id][tid_num] = {tid_str + "_-7", score, 0};
            }
        } else if (score < THR_NEW_GID) { // score < THR_NEW_GID (Reason 2: Clearly dissimilar)
            ng_state.ambig_count = 0;
            if (time_since_last_new >= NEW_GID_TIME_WINDOW) {
                ng_state.count++;
                if (ng_state.count >= NEW_GID_MIN_FRAMES) {
                    // ======================= 【MODIFIED: 新增高质量人脸检查点】 =======================
                    if (is_face_only_mode) {
                        if (agg.count_high_quality_faces(m_face_det_min_score_face_only) < current_min_face_4_gid) {
                            output.mp[s_id][tid_num] = {tid_str + "_-1_f_hq", -1.0f, 0}; // hq: high quality
                            continue;
                        }
                    }
                    // ======================= 【修改结束】 =======================
                    std::string new_gid = gid_mgr.new_gid(); // Pure GID
                    gid_mgr.bind(new_gid, tid_str, now_stamp, now_stamp_gst, agg, this, "dissimilar"); // 默认 increment_n=true
                    tid2gid[tid_str] = new_gid;
                    candidate_state[tid_str] = {new_gid, CANDIDATE_FRAMES, fid};
                    new_gid_state[tid_str] = {0, fid, 0};
                    gids_recognized_this_frame.insert(new_gid);
                    int n = gid_mgr.tid_hist.count(new_gid) ? (int) gid_mgr.tid_hist[new_gid].size() : 0;
                    output.mp[s_id][tid_num] = {s_id + "_" + std::to_string(tid_num) + "_" + new_gid, score, n};
                } else {
                    output.mp[s_id][tid_num] = {tid_str + "_-5", -1.0f, 0};
                }
            } else {
                output.mp[s_id][tid_num] = {tid_str + "_-6", -1.0f, 0};
            }
        }

        // ======================= 【NEW: Consolidated Alarm Logic】 =======================
        // 从 GID 匹配块中移至此处，形成一个统一的报警检查点。
        // 现在，只要一个 TID 仍然活跃并绑定到 GID，即使当前帧没有高质量的匹配，
        // 报警也可以基于持续时间被触发。
        if (tid2gid.count(tid_str)) {
            std::string bound_gid = tid2gid.at(tid_str);
            // 主要的判断逻辑现在封装在 _check_and_process_alarm 函数内部。
            // 我们传递所有相关的上下文信息，由它来决定是否触发报警。
            _check_and_process_alarm(
                output, config, tid_str, bound_gid, agg, now_stamp, now_stamp_gst,
                duration, dets, stream_id, body_p, face_p, triggered_alarms_this_frame,
                w_face,
                current_frame_face_scores_.count(tid_str) ? current_frame_face_scores_.at(tid_str) : 0.f,
                current_frame_face_clarity_.count(tid_str) ? current_frame_face_clarity_.at(tid_str) : 0.f,
                is_face_only_mode, // 模式
                current_face_feat, // 特征
                current_match_thr
            );
        }
        // ======================= 【NEW: Consolidated Cooldown Timer Reset Logic】 =======================
        // 在该TID的所有识别逻辑结束后，统一检查它是否已绑定到一个GID。
        // 如果绑定了（无论是之前绑定的还是刚刚新绑定的），就刷新该GID的冷却计时器。
        // 这同时满足了两个需求：
        // 1. 对于一个已识别的TID，只要它还在视野中，就持续刷新其GID的冷却时间。
        // 2. 对于一个新识别并绑定的TID，立即为其GID启动冷却计时。
        if (tid2gid.count(tid_str)) {
            const std::string& bound_gid = tid2gid.at(tid_str);
            #ifdef ENABLE_COOLDOWN_DEBUG_PRINTS
            if (gid_cooldown_threshold > 0 && (!gid_last_recognized_time.count(bound_gid) || std::abs(gid_last_recognized_time.at(bound_gid) - now_stamp) > 1e-6)) {
                 std::cout << "[COOLDOWN_DEBUG] TID: " << tid_str << ", GID: " << bound_gid
                           << " -> END-OF-LOOP TIMESTAMP REFRESH. Old: " << (gid_last_recognized_time.count(bound_gid) ? std::to_string(gid_last_recognized_time.at(bound_gid)) : "N/A")
                           << ", New: " << now_stamp << "\n";
            }
            #endif
            gid_last_recognized_time[bound_gid] = now_stamp;
        }
        // ======================= 【修改结束】 =======================
    }

    // 在处理完本帧所有TID后，统一更新所有被识别到的GID的冷却时间戳
    for (const auto& gid : gids_recognized_this_frame) {
        gid_last_recognized_time[gid] = now_stamp;
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
            tid_to_business_n_.erase(it->first); // 清理已分配的业务n值
            tid_last_gid_for_n_.erase(it->first); // 清理TID与其n值关联的GID记录
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
            tid_last_gid_for_n_.erase(tid_str); // 清理TID与其n值关联的GID记录
            tid_to_business_n_.erase(tid_str);
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
        gid_last_recognized_time.erase(gid_del);
        // 新增：清理业务报警计数器中对应的GID条目
        gid_alarm_business_counts_.erase(gid_del);

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