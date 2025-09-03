#include "feature_processor.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <iostream>
#include <chrono>
#include <iomanip> // For sprintf

// ======================= 【MODIFIED】 =======================
// 添加一个宏来控制所有磁盘I/O操作
// 如果要关闭所有耗时的文件写入和删除，请注释掉下面这行
//#define ENABLE_DISK_IO
// ======================= 【修改结束】 =======================

const std::string REID_MODEL_PATH = "/mnt/nfs/reid_model.onnx";
const std::string FACE_DET_MODEL_PATH = "/mnt/nfs/det_10g_simplified.onnx";
const std::string FACE_REC_MODEL_PATH = "/mnt/nfs/w600k_r50_simplified.onnx";
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
    for (const auto &item: body) patches.push_back(std::get<2>(item));
    return patches;
}

std::vector<cv::Mat> TrackAgg::face_patches() const {
    std::vector<cv::Mat> patches;
    for (const auto &item: face) patches.push_back(std::get<1>(item));
    return patches;
}

/* ================= GlobalID ================= */
static void _add_proto_with_patch(
        std::vector<std::vector<float>> &feat_list,
        const std::vector<float> &new_feat,
        const cv::Mat &new_patch,
        const std::string &dir_path) {
    if (new_feat.empty() || new_patch.empty()) return;

    if (!feat_list.empty()) {
        float max_sim = -1.f;
        for (const auto &x: feat_list) max_sim = std::max(max_sim, sim_vec(new_feat, x));
        if (max_sim < UPDATE_THR) return;
    }

#ifdef ENABLE_DISK_IO
    std::filesystem::create_directories(dir_path);
#endif

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
        feat_list[idx_to_replace] = blend(feat_list[idx_to_replace], new_feat);
    }

#ifdef ENABLE_DISK_IO
    if (idx_to_replace != -1) {
        char filename[32];
        sprintf(filename, "%02d.jpg", idx_to_replace);
        cv::imwrite((std::filesystem::path(dir_path) / filename).string(), new_patch);
    }
#endif

    auto [new_lst, keep_mask] = remove_outliers_cpp(feat_list, 3.0f);
    if (new_lst.size() != feat_list.size()) {
        std::cout << "[GlobalID] Outlier removed: " << feat_list.size() - new_lst.size() << " from " << dir_path
                  << std::endl;

#ifdef ENABLE_DISK_IO
        std::vector<std::string> img_paths;
        try {
            if (std::filesystem::exists(dir_path)) {
                for (const auto &entry: std::filesystem::directory_iterator(dir_path)) {
                    if (entry.path().extension() == ".jpg") img_paths.push_back(entry.path().string());
                }
            }
        } catch (...) {}

        std::sort(img_paths.begin(), img_paths.end());

        for (size_t i = 0; i < img_paths.size(); ++i) {
            if (i >= keep_mask.size() || !keep_mask[i]) {
                try {
                    std::filesystem::remove(img_paths[i]);
                } catch (const std::exception &e) {
                    std::cerr << "Failed to remove outlier image: " << e.what() << std::endl;
                }
            }
        }
#endif
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
#ifdef ENABLE_DISK_IO
    try {
        std::filesystem::create_directories(std::filesystem::path(SAVE_DIR) / gid / "faces");
        std::filesystem::create_directories(std::filesystem::path(SAVE_DIR) / gid / "bodies");
    } catch (...) {}
#endif
    std::cout << "[GlobalID] new " << gid << std::endl;
    return gid;
}

int
GlobalID::can_update_proto(const std::string &gid, const std::vector<float> &face_f, const std::vector<float> &body_f) {
    if (bank_faces.find(gid) == bank_faces.end()) return 0;
    if (!bank_faces[gid].empty() && !face_f.empty() &&
        sim_vec(face_f, avg_feats(bank_faces[gid])) < FACE_THR_STRICT)
        return -1;
    if (!bank_bodies[gid].empty() && !body_f.empty() &&
        sim_vec(body_f, avg_feats(bank_bodies[gid])) < BODY_THR_STRICT)
        return -2;
    return 0;
}

void GlobalID::bind(const std::string &gid, const std::string &tid, int current_ts, const TrackAgg &agg) {
    auto [face_f, face_p] = agg.main_face_feat_and_patch();
    auto [body_f, body_p] = agg.main_body_feat_and_patch();

    _add_proto_with_patch(bank_faces[gid], face_f, face_p, (std::filesystem::path(SAVE_DIR) / gid / "faces").string());
    _add_proto_with_patch(bank_bodies[gid], body_f, body_p,
                          (std::filesystem::path(SAVE_DIR) / gid / "bodies").string());

    auto &v = tid_hist[gid];
    if (std::find(v.begin(), v.end(), tid) == v.end()) v.push_back(tid);
    last_update[gid] = current_ts;
}

std::pair<std::string, float> GlobalID::probe(const std::vector<float> &face_f, const std::vector<float> &body_f) {
    std::string best_gid;
    float best_score = -1.f;
    for (auto const &[gid, face_pool]: bank_faces) {
        if (bank_bodies.count(gid) == 0 || face_pool.empty() || bank_bodies.at(gid).empty()) continue;
        float face_sim = face_f.empty() ? 0.f : sim_vec(face_f, avg_feats(face_pool));
        float body_sim = body_f.empty() ? 0.f : sim_vec(body_f, avg_feats(bank_bodies.at(gid)));
        float sc = W_FACE * face_sim + W_BODY * body_sim;
        if (sc > best_score) {
            best_score = sc;
            best_gid = gid;
        }
    }
    return {best_gid, best_score};
}

/* =============== FeatureProcessor =============== */
FeatureProcessor::FeatureProcessor(const std::string &mode, const std::string &device,
                                   const std::string &feature_cache_path,
                                   const nlohmann::json &boundary_config)
        : mode_(mode), device_(device), feature_cache_path_(feature_cache_path) {
    std::cout << "FeatureProcessor initialized in '" << mode_ << "' mode." << std::endl;
    if (mode_ == "realtime") {
        std::cout << "Loading ReID and Face models for feature extraction..." << std::endl;
        bool use_gpu = (device == "cuda");
        try {
            reid_model_ = std::make_unique<PersonReid>(REID_MODEL_PATH, REID_INPUT_WIDTH, REID_INPUT_HEIGHT, use_gpu);
            face_analyzer_ = std::make_unique<FaceAnalyzer>(FACE_DET_MODEL_PATH, FACE_REC_MODEL_PATH);
            std::string provider = use_gpu ? "CUDAExecutionProvider" : "CPUExecutionProvider";
            face_analyzer_->prepare(provider, FACE_DET_MIN_SCORE, cv::Size(640, 640));
        } catch (const std::exception &e) {
            std::cerr << "[FATAL] Failed to load models in realtime mode: " << e.what() << std::endl;
            throw;
        }
        if (!feature_cache_path_.empty()) {
            auto parent_path = std::filesystem::path(feature_cache_path_).parent_path();
            if (!parent_path.empty()) std::filesystem::create_directories(parent_path);
        }
    } else if (mode_ == "load") {
        if (feature_cache_path_.empty() || !std::filesystem::exists(feature_cache_path_)) {
            throw std::runtime_error("In 'load' mode, a valid feature_cache_path is required: " + feature_cache_path_);
        }
        std::ifstream jf(feature_cache_path_);
        jf >> features_cache_;
    } else {
        throw std::invalid_argument("Invalid mode: " + mode_ + ". Choose 'realtime' or 'load'.");
    }

    if (!boundary_config.is_null()) {
        for (auto const &[stream_id, config]: boundary_config.items()) {
            if (config.contains("intrusion_poly")) {
                std::vector<cv::Point> poly;
                for (const auto &pt_json: config["intrusion_poly"]) {
                    poly.emplace_back(pt_json[0].get<int>(), pt_json[1].get<int>());
                }
                intrusion_detectors.emplace(stream_id, IntrusionDetector(poly));
                std::cout << "Initialized IntrusionDetector for stream '" << stream_id << "'." << std::endl;
            }
            if (config.contains("crossing_line")) {
                const auto &line_cfg = config["crossing_line"];
                cv::Point start(line_cfg["start"][0].get<int>(), line_cfg["start"][1].get<int>());
                cv::Point end(line_cfg["end"][0].get<int>(), line_cfg["end"][1].get<int>());
                std::string direction = line_cfg.value("direction", "any");
                line_crossing_detectors.emplace(stream_id, LineCrossingDetector(start, end, direction));
                std::cout << "Initialized LineCrossingDetector for stream '" << stream_id << "'." << std::endl;
            }
        }
    }

#ifdef ENABLE_DISK_IO
    try {
        if (std::filesystem::exists(SAVE_DIR)) std::filesystem::remove_all(SAVE_DIR);
        if (std::filesystem::exists(ALARM_DIR)) std::filesystem::remove_all(ALARM_DIR);
        std::filesystem::create_directories(SAVE_DIR);
        std::filesystem::create_directories(ALARM_DIR);
    } catch (const std::exception &e) { std::cerr << "Error creating base directories: " << e.what() << std::endl; }
#endif
}

FeatureProcessor::~FeatureProcessor() {
#ifdef ENABLE_DISK_IO
    if (mode_ == "realtime" && !feature_cache_path_.empty() && !features_to_save_.is_null()) {
        std::cout << "\nSaving " << features_to_save_.size() << " frames of features to '" << feature_cache_path_
                  << "'..." << std::endl;
        try {
            std::ofstream of(feature_cache_path_);
            of << features_to_save_.dump(4);
        } catch (const std::exception &e) {
            std::cerr << "Failed to save features to '" << feature_cache_path_ << "': " << e.what() << std::endl;
        }
    }
#endif
}

void FeatureProcessor::_extract_features_realtime(const Packet &pkt, const std::vector<Face> &face_info) {
    const auto &[stream_id, fid, patches, dets] = pkt;
    nlohmann::json extracted_features_for_this_frame;

    for (size_t i = 0; i < dets.size(); ++i) {
        const auto &det = dets[i];
        if (i >= patches.size()) continue;
        const auto &patch = patches[i];
        if (det.class_id != 0 || !is_long_patch(patch)) continue;

        cv::Mat resized_patch;
        cv::resize(patch, resized_patch, cv::Size(REID_INPUT_WIDTH, REID_INPUT_HEIGHT));
        cv::Mat feat_mat = reid_model_->extract_feat(resized_patch);
        if (feat_mat.empty()) continue;

        std::vector<float> feat_vec(feat_mat.begin<float>(), feat_mat.end<float>());
        std::string tid_str = stream_id + "_" + std::to_string(det.id);

        agg_pool[tid_str].add_body(feat_vec, det.score, patch);
        last_seen[tid_str] = fid;
        if (!feature_cache_path_.empty()) extracted_features_for_this_frame[tid_str]["body_feat"] = feat_vec;
    }

    if (!face_info.empty() && !patches.empty() && face_analyzer_) {
        std::set<size_t> used_face_indices;
        for (size_t i = 0; i < dets.size(); ++i) {
            const auto &det = dets[i];
            if (det.class_id != 0 || i >= patches.size()) continue;
            const auto &person_patch = patches[i];

            std::vector<size_t> matching_face_indices;
            for (size_t j = 0; j < face_info.size(); ++j) {
                if (used_face_indices.count(j)) continue;
                if (calculate_ioa(det.tlwh, face_info[j].bbox) > 0.8) matching_face_indices.push_back(j);
            }
            if (matching_face_indices.size() != 1) continue;

            size_t unique_face_idx = matching_face_indices[0];
            const Face &face_global_coords = face_info[unique_face_idx];
            if (face_global_coords.det_score < FACE_DET_MIN_SCORE) continue;
            used_face_indices.insert(unique_face_idx);

            Face face_relative_coords = face_global_coords;
            face_relative_coords.bbox.x -= det.tlwh.x;
            face_relative_coords.bbox.y -= det.tlwh.y;
            for (auto &kp: face_relative_coords.kps) {
                kp.x -= det.tlwh.x;
                kp.y -= det.tlwh.y;
            }

            cv::Rect face_bbox_in_patch_safe =
                    cv::Rect(face_relative_coords.bbox) & cv::Rect(0, 0, person_patch.cols, person_patch.rows);
            if (face_bbox_in_patch_safe.width < 32 || face_bbox_in_patch_safe.height < 32) continue;

            try {
                face_analyzer_->get_embedding(person_patch, face_relative_coords);
                if (face_relative_coords.embedding.empty()) continue;

                cv::Mat normalized_emb;
                cv::normalize(face_relative_coords.embedding, normalized_emb, 1.0, 0.0, cv::NORM_L2);
                std::vector<float> f_emb(normalized_emb.begin<float>(), normalized_emb.end<float>());

                std::string tid_str = stream_id + "_" + std::to_string(det.id);
                agg_pool[tid_str].add_face(f_emb, person_patch);
                last_seen[tid_str] = fid;
                if (!feature_cache_path_.empty()) extracted_features_for_this_frame[tid_str]["face_feat"] = f_emb;
            } catch (const std::exception &e) { continue; }
        }
    }
#ifdef ENABLE_DISK_IO
    if (!feature_cache_path_.empty() && !extracted_features_for_this_frame.is_null()) {
        features_to_save_[std::to_string(fid)] = extracted_features_for_this_frame;
    }
#endif
}

void FeatureProcessor::_load_features_from_cache(const Packet &pkt) {
    const auto &[stream_id, fid, patches, dets] = pkt;
    std::string fid_str = std::to_string(fid);
    if (!features_cache_.contains(fid_str)) return;

    for (size_t i = 0; i < dets.size(); ++i) {
        const auto &det = dets[i];
        if (i >= patches.size()) continue;

        std::string tid_str_json = stream_id + "_" + std::to_string(det.id);

        if (!features_cache_.at(fid_str).contains(tid_str_json)) continue;

        const nlohmann::json &fd = features_cache_.at(fid_str).at(tid_str_json);
        const auto &patch = patches[i];

        auto &agg = agg_pool[tid_str_json];

        if (fd.contains("body_feat") && !fd["body_feat"].is_null()) {
            auto bf = fd["body_feat"].get<std::vector<float>>();
            if (!bf.empty()) agg.add_body(bf, det.score, patch);
        }
        if (fd.contains("face_feat") && !fd["face_feat"].is_null()) {
            auto ff = fd["face_feat"].get<std::vector<float>>();
            if (!ff.empty()) agg.add_face(ff, patch);
        }
        last_seen[tid_str_json] = fid;
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
    if (gid_mgr.bank_faces.find(gid) == gid_mgr.bank_faces.end()) return {};
    auto face_pool = gid_mgr.bank_faces.at(gid);
    auto body_pool = gid_mgr.bank_bodies.at(gid);
    auto face_f = face_pool.empty() ? std::vector<float>() : avg_feats(face_pool);
    auto body_f = body_pool.empty() ? std::vector<float>() : avg_feats(body_pool);
    return _fuse_feat(face_f, body_f);
}

void FeatureProcessor::trigger_alarm(const std::string &gid, const TrackAgg &agg) {
    if (alarmed.count(gid)) return;
    auto cur_rep = _gid_fused_rep(gid);
    if (cur_rep.empty()) return;

    for (const auto &[ogid, rep]: alarm_reprs) {
        if (sim_vec(cur_rep, rep) >= ALARM_DUP_THR) {
            std::cout << "[ALARM] Skip " << gid << " (similar to " << ogid << ")" << std::endl;
            return;
        }
    }

#ifdef ENABLE_DISK_IO
    time_t now_time = time(0);
    char buf[80];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", localtime(&now_time));
    std::filesystem::path dst_dir = std::filesystem::path(ALARM_DIR) / (gid + "_" + buf);

    try {
        std::filesystem::path src_dir = std::filesystem::path(SAVE_DIR) / gid;
        if (std::filesystem::exists(src_dir))
            std::filesystem::copy(src_dir, dst_dir, std::filesystem::copy_options::recursive |
                                                    std::filesystem::copy_options::overwrite_existing);

        std::filesystem::path seq_face_dir = dst_dir / "agg_sequence/face";
        std::filesystem::path seq_body_dir = dst_dir / "agg_sequence/body";
        std::filesystem::create_directories(seq_face_dir);
        std::filesystem::create_directories(seq_body_dir);

        int i = 0;
        for (const auto &img: agg.face_patches()) {
            char fname[16];
            sprintf(fname, "%03d.jpg", i++);
            cv::imwrite((seq_face_dir / fname).string(), img);
        }

        i = 0;
        for (const auto &img: agg.body_patches()) {
            char fname[16];
            sprintf(fname, "%03d.jpg", i++);
            cv::imwrite((seq_body_dir / fname).string(), img);
        }
    } catch (const std::exception &e) {
        std::cerr << "\n[ALARM] Failed copy/write for " << gid << ": " << e.what() << std::endl;
    }
#endif

    alarmed.insert(gid);
    alarm_reprs[gid] = cur_rep;
    std::cout << "\n[ALARM] GID " << gid << " triggered and backed up." << std::endl;
}

auto FeatureProcessor::process_packet(const Packet &pkt, const std::vector<Face> &face_info)
-> std::map<std::string, std::map<int, std::tuple<std::string, float, int>>> {
    const auto &[stream_id, fid, patches, dets] = pkt;

    if (intrusion_detectors.count(stream_id)) {
        for (int tid: intrusion_detectors.at(stream_id).check(dets, stream_id))
            behavior_alarm_state[stream_id + "_" + std::to_string(tid)] = {fid, "_AA"};
    }
    if (line_crossing_detectors.count(stream_id)) {
        for (int tid: line_crossing_detectors.at(stream_id).check(dets, stream_id))
            behavior_alarm_state[stream_id + "_" + std::to_string(tid)] = {fid, "_AL"};
    }

    if (mode_ == "realtime") _extract_features_realtime(pkt, face_info);
    else if (mode_ == "load") _load_features_from_cache(pkt);

    std::map<std::string, std::map<int, std::tuple<std::string, float, int>>> realtime_map;
    for (auto const &[tid_str, agg]: agg_pool) {
        size_t last_underscore = tid_str.find_last_of('_');
        std::string s_id = tid_str.substr(0, last_underscore);
        int tid_num = std::stoi(tid_str.substr(last_underscore + 1));

        if ((int) agg.body.size() < MIN_BODY4GID) {
            std::string gid_str = tid_str + "_-1_b_" + std::to_string(agg.body.size());
            realtime_map[s_id][tid_num] = {gid_str, -1.f, 0};
            continue;
        }
        if ((int) agg.face.size() < MIN_FACE4GID) {
            std::string gid_str = tid_str + "_-1_f_" + std::to_string(agg.face.size());
            realtime_map[s_id][tid_num] = {gid_str, -1.f, 0};
            continue;
        }

        auto [face_f, face_p] = agg.main_face_feat_and_patch();
        auto [body_f, body_p] = agg.main_body_feat_and_patch();
        if (face_f.empty()) {
            realtime_map[s_id][tid_num] = {tid_str + "_-2_f", -1.f, 0};
            continue;
        }
        if (body_f.empty()) {
            realtime_map[s_id][tid_num] = {tid_str + "_-2_b", -1.f, 0};
            continue;
        }

        auto [cand_gid, score] = gid_mgr.probe(face_f, body_f);

        auto &state = candidate_state[tid_str];
        auto &ng_state = new_gid_state[tid_str];
        int time_since_last_new = fid - ng_state.last_new_fid;

        if (tid2gid.count(tid_str)) {
            std::string bound_gid = tid2gid.at(tid_str);
            if (!cand_gid.empty() && cand_gid != bound_gid && (fid - state.last_bind_fid) < BIND_LOCK_FRAMES) {
                int n_tid = gid_mgr.tid_hist.count(bound_gid) ? (int) gid_mgr.tid_hist.at(bound_gid).size() : 0;
                realtime_map[s_id][tid_num] = {tid_str + "_-3", score, n_tid};
                continue;
            }
        }

        if (!cand_gid.empty() && score >= MATCH_THR) {
            ng_state.ambig_count = 0;
            state.count = (state.cand_gid == cand_gid) ? state.count + 1 : 1;
            state.cand_gid = cand_gid;
            int flag_code = gid_mgr.can_update_proto(cand_gid, face_f, body_f);
            if (state.count >= CANDIDATE_FRAMES && flag_code == 0) {
                gid_mgr.bind(cand_gid, tid_str, fid, agg);
                tid2gid[tid_str] = cand_gid;
                state.last_bind_fid = fid;
                int n = (int) gid_mgr.tid_hist[cand_gid].size();
                if (n >= ALARM_CNT_TH) trigger_alarm(cand_gid, agg);
                realtime_map[s_id][tid_num] = {s_id + "_" + std::to_string(tid_num) + "_" + cand_gid, score, n};
            } else {
                std::string flag = (flag_code == -1) ? "_-4_ud_f" : (flag_code == -2) ? "_-4_ud_b" : "_-4_c";
                realtime_map[s_id][tid_num] = {tid_str + flag, -1.0f, 0};
            }
        } else if (gid_mgr.bank_faces.empty()) {
            std::string new_gid = gid_mgr.new_gid();
            gid_mgr.bind(new_gid, tid_str, fid, agg);
            tid2gid[tid_str] = new_gid;
            state = {new_gid, CANDIDATE_FRAMES, fid};
            ng_state.last_new_fid = fid;
            int n = (int) gid_mgr.tid_hist[new_gid].size();
            if (n >= ALARM_CNT_TH) trigger_alarm(new_gid, agg);
            realtime_map[s_id][tid_num] = {s_id + "_" + std::to_string(tid_num) + "_" + new_gid, score, n};
        } else if (!cand_gid.empty() && score >= THR_NEW_GID) {
            ng_state.ambig_count++;
            if (ng_state.ambig_count >= WAIT_FRAMES_AMBIGUOUS && time_since_last_new >= NEW_GID_TIME_WINDOW) {
                std::string new_gid = gid_mgr.new_gid();
                gid_mgr.bind(new_gid, tid_str, fid, agg);
                tid2gid[tid_str] = new_gid;
                state = {new_gid, CANDIDATE_FRAMES, fid};
                ng_state = {0, fid, 0};
                int n = (int) gid_mgr.tid_hist[new_gid].size();
                if (n >= ALARM_CNT_TH) trigger_alarm(new_gid, agg);
                realtime_map[s_id][tid_num] = {s_id + "_" + std::to_string(tid_num) + "_" + new_gid, score, n};
            } else {
                realtime_map[s_id][tid_num] = {tid_str + "_-7", score, 0};
            }
        } else { // score < THR_NEW_GID
            ng_state.ambig_count = 0;
            if (time_since_last_new >= NEW_GID_TIME_WINDOW) {
                ng_state.count++;
                if (ng_state.count >= NEW_GID_MIN_FRAMES) {
                    std::string new_gid = gid_mgr.new_gid();
                    gid_mgr.bind(new_gid, tid_str, fid, agg);
                    tid2gid[tid_str] = new_gid;
                    state = {new_gid, CANDIDATE_FRAMES, fid};
                    ng_state = {0, fid, 0};
                    int n = (int) gid_mgr.tid_hist[new_gid].size();
                    if (n >= ALARM_CNT_TH) trigger_alarm(new_gid, agg);
                    realtime_map[s_id][tid_num] = {s_id + "_" + std::to_string(tid_num) + "_" + new_gid, score, n};
                } else {
                    realtime_map[s_id][tid_num] = {tid_str + "_-5", -1.0f, 0};
                }
            } else {
                realtime_map[s_id][tid_num] = {tid_str + "_-6", -1.0f, 0};
            }
        }
    }

    std::map<std::string, std::tuple<int, std::string>> active_alarms;
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
            realtime_map[s_id][t_id_int] = {info_str, 1.0f, n_tid};
        }
    }
    behavior_alarm_state = active_alarms;

    for (auto it = last_seen.cbegin(); it != last_seen.cend();) {
        if (fid - it->second >= MAX_TID_IDLE_FRAMES) {
            agg_pool.erase(it->first);
            tid2gid.erase(it->first);
            candidate_state.erase(it->first);
            new_gid_state.erase(it->first);
            behavior_alarm_state.erase(it->first);
            it = last_seen.erase(it);
        } else { ++it; }
    }

    std::vector<std::string> gids_to_del;
    for (auto const &[gid, last_fid]: gid_mgr.last_update) {
        if (fid - last_fid >= GID_MAX_IDLE_FRAMES) gids_to_del.push_back(gid);
    }
    for (const auto &gid_del: gids_to_del) {
        std::vector<std::string> tids_to_clean;
        for (auto const &[tid_str, g]: tid2gid) { if (g == gid_del) tids_to_clean.push_back(tid_str); }
        for (const auto &tid_str: tids_to_clean) {
            agg_pool.erase(tid_str);
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
        alarmed.erase(gid_del);
        alarm_reprs.erase(gid_del);
#ifdef ENABLE_DISK_IO
        try {
            if (std::filesystem::exists(std::filesystem::path(SAVE_DIR) / gid_del)) {
                std::filesystem::remove_all(std::filesystem::path(SAVE_DIR) / gid_del);
            }
        } catch (const std::exception &e) {
            std::cerr << "Failed to remove GID directory: " << gid_del << " " << e.what() << std::endl;
        }
#endif
    }

    return realtime_map;
}