#include "feature_processor.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <iostream> // For logging/debug

/* ----------------- 工具函数 (与之前一致) ----------------- */
// ... (sim_vec, avg_feats, blend, vec_dist, mean_stddev, etc. - 保持不变)
// copy all helper functions from your existing file here...
/* ----------------- 工具函数 ----------------- */
static float sim_vec(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.empty() || b.empty() || a.size() != b.size()) return 0.f;
    float s = 0.f;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

static std::vector<float> avg_feats(const std::vector<std::vector<float>> &feats) {
    if (feats.empty()) return {};
    std::vector<float> mean(feats[0].size(), 0.f);
    for (auto &f: feats) for (size_t i = 0; i < f.size(); ++i) mean[i] += f[i];
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

static std::vector<float>
main_representation_cpp(const std::deque<std::vector<float>> &feats_dq, float outlier_thr = 1.5f) {
    if (feats_dq.empty()) return {};
    const std::vector<std::vector<float>> feats(feats_dq.begin(), feats_dq.end());
    auto initial_mean = avg_feats(feats);
    std::vector<float> dists;
    dists.reserve(feats.size());
    for (const auto &f: feats) { dists.push_back(vec_dist(f, initial_mean)); }
    auto [dist_mean, dist_std] = mean_stddev(dists);
    float keep_thresh = dist_mean + outlier_thr * dist_std;
    std::vector<std::vector<float>> kept_feats;
    for (size_t i = 0; i < feats.size(); ++i) { if (dists[i] < keep_thresh) { kept_feats.push_back(feats[i]); }}
    if (kept_feats.empty()) { kept_feats = feats; }
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
    return best_idx != -1 ? kept_feats[best_idx] : std::vector<float>{};
}
/* ================= TrackAgg (与之前一致) ================= */
// ... (TrackAgg implementation - 保持不变)
// copy all TrackAgg implementation from your existing file here...
bool TrackAgg::check_consistency(const std::deque<std::vector<float>> &feats, float thr) {
    if (feats.size() < 2) return true;
    std::vector<float> sims;
    for (size_t i = 0; i < feats.size(); ++i)
        for (size_t j = i + 1; j < feats.size(); ++j)
            sims.push_back(sim_vec(feats[i], feats[j]));
    float m = 0.f;
    for (float v: sims) m += v;
    m /= sims.size();
    return (1.f - m) <= thr;
}

void TrackAgg::add_body(const std::vector<float> &feat, float score) {
    body.emplace_back(feat, score);
    if ((int) body.size() > MIN_BODY4GID) body.pop_front();
}

void TrackAgg::add_face(const std::vector<float> &feat) {
    face.push_back(feat);
    if ((int) face.size() > MIN_FACE4GID) face.pop_front();
}

std::vector<float> TrackAgg::main_body_feat() const {
    if (body.empty()) return {};
    std::deque<std::vector<float>> feats;
    for (auto &t: body) feats.push_back(std::get<0>(t));
    if (!check_consistency(feats, 0.5f)) return {};
    return main_representation_cpp(feats);
}

std::vector<float> TrackAgg::main_face_feat() const {
    if (face.empty()) return {};
    if (!check_consistency(face)) return {};
    return main_representation_cpp(face);
}

/* ================= GlobalID ================= */

static void add_proto(std::vector<std::vector<float>> &lst, const std::vector<float> &feat) {
    if (feat.empty()) return;
    if (!lst.empty()) {
        float best = -1.f;
        for (auto &x: lst) best = std::max(best, sim_vec(feat, x));
        if (best < UPDATE_THR) return;
    }
    if ((int) lst.size() < MAX_PROTO_PER_TYPE) {
        lst.push_back(feat);
    } else {
        int idx = 0;
        float best = -1.f;
        for (int i = 0; i < (int) lst.size(); ++i) {
            float s = sim_vec(feat, lst[i]);
            if (s > best) {
                best = s;
                idx = i;
            }
        }
        lst[idx] = blend(lst[idx], feat);
    }
}

// === MODIFIED: Create directories when a new GID is made ===
std::string GlobalID::new_gid() {
    char buf[32];
    sprintf(buf, "G%05d", gid_next++);
    std::string gid(buf);
    bank_faces[gid] = {};
    bank_bodies[gid] = {};
    tid_hist[gid] = {};
    last_update[gid] = 0; // Initialize last_update

    // Create directories
    try {
        std::filesystem::create_directories(std::filesystem::path(SAVE_DIR) / gid);
    } catch (const std::filesystem::filesystem_error &e) {
        std::cerr << "Error creating directory for GID " << gid << ": " << e.what() << std::endl;
    }

    std::cout << "[GlobalID] new " << gid << std::endl;
    return gid;
}

int
GlobalID::can_update_proto(const std::string &gid, const std::vector<float> &face_f, const std::vector<float> &body_f) {
    if (bank_faces.find(gid) == bank_faces.end()) return 0; // Gid does not exist, can't check
    if (!bank_faces[gid].empty() && !face_f.empty() &&
        sim_vec(face_f, avg_feats(bank_faces[gid])) < FACE_THR_STRICT)
        return -1;
    if (!bank_bodies[gid].empty() && !body_f.empty() &&
        sim_vec(body_f, avg_feats(bank_bodies[gid])) < BODY_THR_STRICT)
        return -2;
    return 0;
}

// === MODIFIED: Added current_ts parameter for timeout management ===
void GlobalID::bind(const std::string &gid, const std::vector<float> &face_f,
                    const std::vector<float> &body_f, const std::string &tid, int current_ts) {
    add_proto(bank_faces[gid], face_f);
    add_proto(bank_bodies[gid], body_f);
    auto &v = tid_hist[gid];
    if (std::find(v.begin(), v.end(), tid) == v.end()) v.push_back(tid);
    last_update[gid] = current_ts;
}

std::pair<std::string, float> GlobalID::probe(const std::vector<float> &face_f, const std::vector<float> &body_f) {
    std::string best_gid;
    float best_score = -1.f;
    for (auto &kv: bank_faces) {
        const std::string &gid = kv.first;
        if (bank_faces[gid].empty() || bank_bodies[gid].empty()) continue;
        float face_sim = face_f.empty() ? 0.f : sim_vec(face_f, avg_feats(bank_faces[gid]));
        float body_sim = body_f.empty() ? 0.f : sim_vec(body_f, avg_feats(bank_bodies[gid]));
        float sc = W_FACE * face_sim + W_BODY * body_sim;
        if (sc > best_score) {
            best_score = sc;
            best_gid = gid;
        }
    }
    return {best_gid, best_score};
}

/* =============== FeatureProcessor =============== */
FeatureProcessor::FeatureProcessor(const std::string &cache_path) {
    std::ifstream jf(cache_path);
    if (!jf.is_open()) throw std::runtime_error("open cache fail: " + cache_path);
    jf >> features_cache;

    // === NEW: Create base directories ===
    try {
        std::filesystem::create_directories(SAVE_DIR);
        std::filesystem::create_directories(ALARM_DIR);
    } catch (const std::filesystem::filesystem_error &e) {
        std::cerr << "Error creating base directories: " << e.what() << std::endl;
    }
}

// === NEW: Alarm helper functions ===
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

void FeatureProcessor::trigger_alarm(const std::string &gid) {
    if (alarmed.count(gid)) return;

    auto cur_rep = _gid_fused_rep(gid);
    if (cur_rep.empty()) {
        std::cerr << "[ALARM] Failed to generate fused representation for GID " << gid << std::endl;
        return;
    }

    for (const auto &[ogid, rep]: alarm_reprs) {
        if (sim_vec(cur_rep, rep) >= ALARM_DUP_THR) {
            std::cout << "[ALARM] Skip " << gid << " (similar to " << ogid << ")" << std::endl;
            return;
        }
    }

    std::string src_dir = std::filesystem::path(SAVE_DIR) / gid;
    time_t now = time(0);
    char buf[80];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", localtime(&now));
    std::string dst_dir = std::filesystem::path(ALARM_DIR) / (gid + "_" + buf);

    try {
        if (std::filesystem::exists(src_dir)) {
            std::filesystem::copy(src_dir, dst_dir, std::filesystem::copy_options::recursive);
            alarmed.insert(gid);
            alarm_reprs[gid] = cur_rep;
            std::cout << "[ALARM] GID " << gid << " alarmed and backed up to " << dst_dir << std::endl;
        }
    } catch (const std::filesystem::filesystem_error &e) {
        std::cerr << "[ALARM] Failed to process alarm for " << gid << ": " << e.what() << std::endl;
    }
}

auto FeatureProcessor::process_packet(const std::string &stream_id, int fid)
-> std::map<std::string, std::map<int, std::tuple<std::string, float, int>>> {
    std::map<std::string, std::map<int, std::tuple<std::string, float, int>>> realtime_map;
    if (!features_cache.contains(std::to_string(fid))) return realtime_map;

    /* ---------- 1. 更新 agg_pool (不变) ---------- */
    for (auto &[tid_str, fd]: features_cache[std::to_string(fid)].items()) {
        std::vector<float> bf, ff;
        if (fd.contains("body_feat") && fd["body_feat"].is_array())
            for (auto &v: fd["body_feat"])
                bf.push_back(v.get<float>());
        if (fd.contains("face_feat") && fd["face_feat"].is_array())
            for (auto &v: fd["face_feat"])
                ff.push_back(v.get<float>());
        auto &agg = agg_pool[tid_str];
        if (!bf.empty()) agg.add_body(bf, 1.f);
        if (!ff.empty()) agg.add_face(ff);
        last_seen[tid_str] = fid;
    }

    /* ---------- 2. 遍历每个 tid (=== 全面重写此部分逻辑 ===) ---------- */
    for (auto &[tid_str, agg]: agg_pool) {
        int tid_num = std::stoi(tid_str.substr(tid_str.find('_') + 1));

        // 2-a 数量/质量检查
        if ((int) agg.body.size() < MIN_BODY4GID) {
            realtime_map[stream_id][tid_num] = {std::to_string(tid_num) + "_-1_b_" + std::to_string(agg.body.size()),
                                                -1.f, 0};
            continue;
        }
        if ((int) agg.face.size() < MIN_FACE4GID) {
            realtime_map[stream_id][tid_num] = {std::to_string(tid_num) + "_-1_f_" + std::to_string(agg.face.size()),
                                                -1.f, 0};
            continue;
        }

        auto face_f = agg.main_face_feat();
        auto body_f = agg.main_body_feat();
        if (face_f.empty()) {
            realtime_map[stream_id][tid_num] = {"-2_f", -1.f, 0};
            continue;
        }
        if (body_f.empty()) {
            realtime_map[stream_id][tid_num] = {"-2_b", -1.f, 0};
            continue;
        }

        // 2-c probe
        auto [cand_gid, score] = gid_mgr.probe(face_f, body_f);

        // --- 引入完整的 FSM ---
        auto &state = candidate_state[tid_str];
        auto &ng_state = new_gid_state[tid_str];
        int time_since_last_new = fid - ng_state.last_new_fid;

        // 已有 GID 的锁定逻辑
        if (tid2gid.count(tid_str)) {
            std::string bound_gid = tid2gid[tid_str];
            int lock_elapsed = fid - state.last_bind_fid;
            if (!cand_gid.empty() && cand_gid != bound_gid && lock_elapsed < BIND_LOCK_FRAMES) {
                int n = gid_mgr.tid_hist.count(bound_gid) ? (int) gid_mgr.tid_hist[bound_gid].size() : 0;
                realtime_map[stream_id][tid_num] = {"-3", score, n};
                continue;
            }
        }

        // 2-d-1: 直接匹配成功
        if (!cand_gid.empty() && score >= MATCH_THR) {
            ng_state.ambig_count = 0;
            state.count = (state.cand_gid == cand_gid) ? state.count + 1 : 1;
            state.cand_gid = cand_gid;

            if (state.count >= CANDIDATE_FRAMES && gid_mgr.can_update_proto(cand_gid, face_f, body_f) == 0) {
                gid_mgr.bind(cand_gid, face_f, body_f, tid_str, fid);
                tid2gid[tid_str] = cand_gid;
                state.last_bind_fid = fid;
                int n = (int) gid_mgr.tid_hist[cand_gid].size();
                if (n >= ALARM_CNT_TH) trigger_alarm(cand_gid);
                realtime_map[stream_id][tid_num] = {cand_gid, score, n};
            } else {
                int flag_code = gid_mgr.can_update_proto(cand_gid, face_f, body_f);
                std::string flag = (flag_code == -1) ? "-4_ud_f" : (flag_code == -2) ? "-4_ud_b" : "-4_c";
                realtime_map[stream_id][tid_num] = {flag, -1.0f, 0};
            }
        }
            // 2-d-2: 库为空，新建
        else if (gid_mgr.bank_faces.empty()) {
            std::string new_gid = gid_mgr.new_gid();
            gid_mgr.bind(new_gid, face_f, body_f, tid_str, fid);
            tid2gid[tid_str] = new_gid;
            state = {new_gid, CANDIDATE_FRAMES, fid};
            ng_state.last_new_fid = fid;
            int n = (int) gid_mgr.tid_hist[new_gid].size();
            if (n >= ALARM_CNT_TH) trigger_alarm(new_gid);
            realtime_map[stream_id][tid_num] = {new_gid, score, n};
        }
            // 2-d-3: 模糊匹配
        else if (!cand_gid.empty() && score >= THR_NEW_GID) {
            ng_state.ambig_count++;
            if (ng_state.ambig_count >= WAIT_FRAMES_AMBIGUOUS && time_since_last_new >= NEW_GID_TIME_WINDOW) {
                std::string new_gid = gid_mgr.new_gid();
                gid_mgr.bind(new_gid, face_f, body_f, tid_str, fid);
                tid2gid[tid_str] = new_gid;
                state = {new_gid, CANDIDATE_FRAMES, fid};
                ng_state = {0, fid, 0};
                int n = (int) gid_mgr.tid_hist[new_gid].size();
                if (n >= ALARM_CNT_TH) trigger_alarm(new_gid);
                realtime_map[stream_id][tid_num] = {new_gid, score, n};
            } else {
                realtime_map[stream_id][tid_num] = {"-7", score, 0};
            }
        }
            // 2-d-4: 完全不匹配
        else {
            ng_state.ambig_count = 0;
            if (time_since_last_new >= NEW_GID_TIME_WINDOW) {
                ng_state.count++;
                if (ng_state.count >= NEW_GID_MIN_FRAMES) {
                    std::string new_gid = gid_mgr.new_gid();
                    gid_mgr.bind(new_gid, face_f, body_f, tid_str, fid);
                    tid2gid[tid_str] = new_gid;
                    state = {new_gid, CANDIDATE_FRAMES, fid};
                    ng_state = {0, fid, 0};
                    int n = (int) gid_mgr.tid_hist[new_gid].size();
                    if (n >= ALARM_CNT_TH) trigger_alarm(new_gid);
                    realtime_map[stream_id][tid_num] = {new_gid, score, n};
                } else {
                    realtime_map[stream_id][tid_num] = {"-5", -1.0f, 0};
                }
            } else {
                realtime_map[stream_id][tid_num] = {"-6", -1.0f, 0};
            }
        }
    }

    /* ---------- 3. 清理 idle tid (不变) ---------- */
    for (auto it = last_seen.begin(); it != last_seen.end();) {
        if (fid - it->second >= MAX_TID_IDLE_FRAMES) {
            agg_pool.erase(it->first);
            tid2gid.erase(it->first);
            candidate_state.erase(it->first);
            new_gid_state.erase(it->first);
            it = last_seen.erase(it);
        } else ++it;
    }

    /* ---------- 4. === NEW: 清理 idle gid === ---------- */
    std::vector<std::string> gids_to_del;
    for (auto const &[gid, last_fid]: gid_mgr.last_update) {
        if (fid - last_fid >= GID_MAX_IDLE_FRAMES) {
            gids_to_del.push_back(gid);
        }
    }

    for (const auto &gid: gids_to_del) {
        std::cout << "[Cleanup] GID " << gid << " timed out." << std::endl;

        // Find and remove linked TIDs
        std::vector<std::string> tids_to_clean;
        for (auto const &[tid_str, g]: tid2gid) {
            if (g == gid) {
                tids_to_clean.push_back(tid_str);
            }
        }
        for (const auto &tid_str: tids_to_clean) {
            agg_pool.erase(tid_str);
            tid2gid.erase(tid_str);
            candidate_state.erase(tid_str);
            new_gid_state.erase(tid_str);
            last_seen.erase(tid_str);
        }

        // Remove GID from all managers
        gid_mgr.bank_faces.erase(gid);
        gid_mgr.bank_bodies.erase(gid);
        gid_mgr.tid_hist.erase(gid);
        gid_mgr.last_update.erase(gid);
        alarmed.erase(gid);
        alarm_reprs.erase(gid);

        // Remove from disk
        try {
            std::filesystem::remove_all(std::filesystem::path(SAVE_DIR) / gid);
        } catch (const std::filesystem::filesystem_error &e) {
            std::cerr << "Failed to delete GID directory " << gid << ": " << e.what() << std::endl;
        }
    }

    return realtime_map;
}