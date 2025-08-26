#include "feature_processor.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <numeric> // 新增：为了 std::accumulate

/* ----------------- 工具函数 ----------------- */
static float sim_vec(const std::vector<float> &a, const std::vector<float> &b) {
    float s = 0.f;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

static std::vector<float>
avg_feats(const std::vector<std::vector<float>> &feats) {
    if (feats.empty()) return {};
    std::vector<float> mean(feats[0].size(), 0.f);
    for (auto &f: feats)
        for (size_t i = 0; i < f.size(); ++i) mean[i] += f[i];
    float num_feats = static_cast<float>(feats.size());
    for (float &v: mean) v /= num_feats;

    float n = 1e-9f;
    for (float v: mean) n += v * v;
    n = std::sqrt(n);
    if (n > 1e-9f) {
        for (float &v: mean) v /= n;
    }
    return mean;
}

static std::vector<float> blend(const std::vector<float> &a,
                                const std::vector<float> &b) {
    std::vector<float> r(a.size());
    float n = 1e-9f;
    for (size_t i = 0; i < a.size(); ++i) {
        r[i] = 0.7f * a[i] + 0.3f * b[i];
        n += r[i] * r[i];
    }
    n = std::sqrt(n);
    for (float &v: r) v /= n;
    return r;
}

// ------ 新增：与 Python 的 numpy 对齐的数学辅助函数 ------
// 计算两个向量的欧式距离
static float vec_dist(const std::vector<float> &a, const std::vector<float> &b) {
    float d_sq = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        d_sq += diff * diff;
    }
    return std::sqrt(d_sq);
}

// 计算一组浮点数的均值和标准差
static std::pair<float, float> mean_stddev(const std::vector<float> &data) {
    if (data.empty()) return {0.f, 0.f};

    float sum = std::accumulate(data.begin(), data.end(), 0.0f);
    float mean = sum / data.size();

    float sq_sum = 0.f;
    for (const auto &val: data) {
        sq_sum += (val - mean) * (val - mean);
    }
    // 注意：Python的np.std默认除以N，这里也保持一致
    float stddev = std::sqrt(sq_sum / data.size());

    return {mean, stddev};
}
// --------------------------------------------------------

/* ================= TrackAgg ================= */

// ------ 新增：复刻 Python 的 _main_representation 核心逻辑 ------
static std::vector<float>
main_representation_cpp(const std::deque<std::vector<float>> &feats_dq, float outlier_thr = 1.5f) {
    if (feats_dq.empty()) return {};

    const std::vector<std::vector<float>> feats(feats_dq.begin(), feats_dq.end());

    // 1. 计算初始均值和离群点
    auto initial_mean = avg_feats(feats);
    std::vector<float> dists;
    dists.reserve(feats.size());
    for (const auto &f: feats) {
        dists.push_back(vec_dist(f, initial_mean));
    }

    auto [dist_mean, dist_std] = mean_stddev(dists);
    float keep_thresh = dist_mean + outlier_thr * dist_std;

    std::vector<std::vector<float>> kept_feats;
    for (size_t i = 0; i < feats.size(); ++i) {
        if (dists[i] < keep_thresh) {
            kept_feats.push_back(feats[i]);
        }
    }

    // 如果所有点都被剔除，则保留所有原始点 (与Python的 if keep.any() else arr 对齐)
    if (kept_feats.empty()) {
        kept_feats = feats;
    }

    // 2. 用清理过的数据重新计算均值
    auto final_mean = avg_feats(kept_feats);

    // 3. 在清理过的数据中，找到与新均值最相似的那个原始特征向量并返回
    int best_idx = -1;
    float best_sim = -2.0f; // 初始化为一个很小的值
    for (size_t i = 0; i < kept_feats.size(); ++i) {
        float s = sim_vec(kept_feats[i], final_mean);
        if (s > best_sim) {
            best_sim = s;
            best_idx = i;
        }
    }

    return best_idx != -1 ? kept_feats[best_idx] : std::vector<float>{};
}
// -------------------------------------------------------------

bool TrackAgg::check_consistency(const std::deque<std::vector<float>> &feats,
                                 float thr) {
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

// ------ 修改：调用新的、更精确的特征计算方法 ------
std::vector<float> TrackAgg::main_body_feat() const {
    if (body.empty()) return {};
    std::deque<std::vector<float>> feats;
    for (auto &t: body) feats.push_back(std::get<0>(t));

    if (!check_consistency(feats, 0.5f)) return {}; // 保持一致性检查

    // 从调用简单平均，改为调用复杂的代表性样本选择逻辑
    return main_representation_cpp(feats);
}

std::vector<float> TrackAgg::main_face_feat() const {
    if (face.empty()) return {};

    if (!check_consistency(face)) return {}; // 保持一致性检查

    // 从调用简单平均，改为调用复杂的代表性样本选择逻辑
    return main_representation_cpp(face);
}
// ------------------------------------------------------

/* ================= GlobalID (此部分保持不变) ================= */
static void add_proto(std::vector<std::vector<float>> &lst,
                      const std::vector<float> &feat) {
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

std::string GlobalID::new_gid() {
    char buf[32];
    sprintf(buf, "G%05d", gid_next++);
    std::string gid(buf);
    bank_faces[gid] = {};
    bank_bodies[gid] = {};
    tid_hist[gid] = {};
    return gid;
}

int GlobalID::can_update_proto(const std::string &gid,
                               const std::vector<float> &face_f,
                               const std::vector<float> &body_f) {
    if (bank_faces[gid].empty() || face_f.empty()) return 0; // 如果库或探针为空，则不检查
    if (sim_vec(face_f, avg_feats(bank_faces[gid])) < FACE_THR_STRICT)
        return -1;

    if (bank_bodies[gid].empty() || body_f.empty()) return 0; // 如果库或探针为空，则不检查
    if (sim_vec(body_f, avg_feats(bank_bodies[gid])) < BODY_THR_STRICT)
        return -2;

    return 0;
}

void GlobalID::bind(const std::string &gid, const std::vector<float> &face_f,
                    const std::vector<float> &body_f, const std::string &tid) {
    add_proto(bank_faces[gid], face_f);
    add_proto(bank_bodies[gid], body_f);
    auto &v = tid_hist[gid];
    if (std::find(v.begin(), v.end(), tid) == v.end()) v.push_back(tid);
}

std::pair<std::string, float>
GlobalID::probe(const std::vector<float> &face_f, const std::vector<float> &body_f) {
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

/* =============== FeatureProcessor (此部分保持不变) =============== */
FeatureProcessor::FeatureProcessor(const std::string &cache_path) {
    std::ifstream jf(cache_path);
    if (!jf.is_open()) throw std::runtime_error("open cache fail: " + cache_path);
    jf >> features_cache;
}

auto FeatureProcessor::process_packet(const std::string &stream_id, int fid)
-> std::map<std::string, std::map<int, std::tuple<std::string, float, int>>> {
    std::map<std::string, std::map<int, std::tuple<std::string, float, int>>> realtime_map;
    if (!features_cache.contains(std::to_string(fid))) return realtime_map;

    /* ---------- 1. 更新 agg_pool ---------- */
    for (auto &[tid_str, fd]: features_cache[std::to_string(fid)].items()) {
        std::vector<float> bf, ff;
        if (fd["body_feat"].is_array()) for (auto &v: fd["body_feat"]) bf.push_back(v.get<float>());
        if (fd["face_feat"].is_array()) for (auto &v: fd["face_feat"]) ff.push_back(v.get<float>());
        auto &agg = agg_pool[tid_str];
        if (!bf.empty()) agg.add_body(bf, 1.f);
        if (!ff.empty()) agg.add_face(ff);
        last_seen[tid_str] = fid;
    }

    /* ---------- 2. 遍历每个 tid ---------- */
    for (auto &[tid_str, agg]: agg_pool) {
        int tid_num = std::stoi(tid_str.substr(tid_str.find('_') + 1));

        /* 2-a 数量不足 */
        if ((int) agg.body.size() < MIN_BODY4GID || (int) agg.face.size() < MIN_FACE4GID) {
            std::string flag = (int) agg.body.size() < MIN_BODY4GID ?
                               (std::to_string(tid_num) + "_-1_b_" + std::to_string((int) agg.body.size())) :
                               (std::to_string(tid_num) + "_-1_f_" + std::to_string((int) agg.face.size()));
            realtime_map[stream_id][tid_num] = {flag, -1.f, 0};
            continue;
        }

        /* 2-b 主特征 */
        auto face_f = agg.main_face_feat();
        auto body_f = agg.main_body_feat();
        if (face_f.empty() || body_f.empty()) {
            realtime_map[stream_id][tid_num] = {face_f.empty() ? "-2_f" : "-2_b", -1.f, 0};
            continue;
        }

        /* 2-c probe */
        auto [cand_gid, score] = gid_mgr.probe(face_f, body_f);

        auto &cs = candidate_state[tid_str];

        if (!cand_gid.empty() && score >= MATCH_THR) {
            if (cand_gid == cs.first) {
                cs.second++;
            } else {
                cs.first = cand_gid;
                cs.second = 1;
            }

            int proto_check = gid_mgr.can_update_proto(cand_gid, face_f, body_f);
            if (cs.second >= CANDIDATE_FRAMES && proto_check == 0) {
                gid_mgr.bind(cand_gid, face_f, body_f, tid_str);
                tid2gid[tid_str] = cand_gid;
                int n = (int) gid_mgr.tid_hist[cand_gid].size();
                realtime_map[stream_id][tid_num] = {cand_gid, score, n};
            } else {
                std::string flag;
                if (proto_check == -1) flag = "-4_ud_f";
                else if (proto_check == -2) flag = "-4_ud_b";
                else flag = "-4_c";
                realtime_map[stream_id][tid_num] = {flag, -1.0f, 0};
            }
        } else if (gid_mgr.bank_faces.empty()) {
            std::string new_gid = gid_mgr.new_gid();
            gid_mgr.bind(new_gid, face_f, body_f, tid_str);
            tid2gid[tid_str] = new_gid;
            int n = (int) gid_mgr.tid_hist[new_gid].size();
            realtime_map[stream_id][tid_num] = {new_gid, score, n};
            cs.first = new_gid;
            cs.second = CANDIDATE_FRAMES;
        } else if (!cand_gid.empty() && score >= THR_NEW_GID) {
            realtime_map[stream_id][tid_num] = {"-7", score, 0};
        } else {
            realtime_map[stream_id][tid_num] = {"-5", -1.f, 0};
        }
    }

    /* ---------- 3. 清理 idle tid ---------- */
    for (auto it = last_seen.begin(); it != last_seen.end();) {
        if (fid - it->second >= MAX_TID_IDLE_FRAMES) {
            agg_pool.erase(it->first);
            tid2gid.erase(it->first);
            candidate_state.erase(it->first);
            it = last_seen.erase(it);
        } else ++it;
    }
    return realtime_map;
}