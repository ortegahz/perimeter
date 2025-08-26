#include "feature_processor.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <stdexcept>

/* ----------------- 工具函数 ----------------- */
static float sim_vec(const std::vector<float> &a, const std::vector<float> &b) {
    float s = 0.f;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

static std::vector<float>
avg_feats(const std::vector<std::vector<float>> &feats) {
    std::vector<float> mean(feats[0].size(), 0.f);
    for (auto &f: feats)
        for (size_t i = 0; i < f.size(); ++i) mean[i] += f[i];
    for (float &v: mean) v /= feats.size();
    float n = 1e-9f;
    for (float v: mean) n += v * v;
    n = std::sqrt(n);
    for (float &v: mean) v /= n;
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

/* ================= TrackAgg ================= */
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

std::vector<float> TrackAgg::main_body_feat() const {
    if (body.empty()) return {};
    std::deque<std::vector<float>> feats;
    for (auto &t: body) feats.push_back(std::get<0>(t));
    if (!check_consistency(feats)) return {};
    return avg_feats(std::vector<std::vector<float>>(feats.begin(), feats.end()));
}

std::vector<float> TrackAgg::main_face_feat() const {
    if (face.empty()) return {};
    if (!check_consistency(face)) return {};
    return avg_feats(std::vector<std::vector<float>>(face.begin(), face.end()));
}

/* ================= GlobalID ================= */
static void add_proto(std::vector<std::vector<float>> &lst,
                      const std::vector<float> &feat) {
    if (feat.empty()) return;
    if (!lst.empty()) {
        float best = -1.f;
        for (auto &x: lst) best = std::max(best, sim_vec(feat, x));
        if (best < UPDATE_THR) return;                    // 不达阈值→丢弃
    }
    if ((int) lst.size() < MAX_PROTO_PER_TYPE) {
        lst.push_back(feat);
    } else {  /* 更新最相似 proto */
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
    if (!bank_faces[gid].empty() &&
        sim_vec(face_f, avg_feats(bank_faces[gid])) < FACE_THR_STRICT)
        return -1;
    if (!bank_bodies[gid].empty() &&
        sim_vec(body_f, avg_feats(bank_bodies[gid])) < BODY_THR_STRICT)
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
        float sc = W_FACE * sim_vec(face_f, avg_feats(bank_faces[gid])) +
                   W_BODY * sim_vec(body_f, avg_feats(bank_bodies[gid]));
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

        /* 模糊区输出 -7 */
        if (!cand_gid.empty() && score >= THR_NEW_GID && score < MATCH_THR) {
            realtime_map[stream_id][tid_num] = {"-7", score, 0};
            continue;                                     // 不动 candidate_state
        }

        /* 候选累积 (保持 python 逻辑) */
        auto &cs = candidate_state[tid_str]; // pair<gid,cnt>
        bool ok = (!cand_gid.empty() && score >= MATCH_THR &&
                   gid_mgr.can_update_proto(cand_gid, face_f, body_f) == 0);
        if (ok) {
            if (cand_gid == cs.first) ++cs.second;
            else {
                cs.first = cand_gid;
                cs.second = 1;
            }
        }
        /* 没命中时保持计数，不清零 */

        /* 绑定成功 */
        if (cs.second >= CANDIDATE_FRAMES) {
            gid_mgr.bind(cand_gid, face_f, body_f, tid_str);
            tid2gid[tid_str] = cand_gid;
            int n = (int) gid_mgr.tid_hist[cand_gid].size();
            realtime_map[stream_id][tid_num] = {cand_gid, score, n};
        }
            /* 候选中 */
        else if (cs.second > 0) {
            realtime_map[stream_id][tid_num] = {"-4_c", -1.f, 0};
        }
            /* 身份库空，新建 */
        else if (gid_mgr.bank_faces.empty()) {
            std::string new_gid = gid_mgr.new_gid();
            gid_mgr.bind(new_gid, face_f, body_f, tid_str);
            tid2gid[tid_str] = new_gid;
            int n = (int) gid_mgr.tid_hist[new_gid].size();
            realtime_map[stream_id][tid_num] = {new_gid, score, n};
        }
            /* 不能更新 proto */
        else if (!cand_gid.empty()) {
            int chk = gid_mgr.can_update_proto(cand_gid, face_f, body_f);
            std::string flag = chk == -1 ? "-4_ud_f" : "-4_ud_b";
            realtime_map[stream_id][tid_num] = {flag, -1.f, 0};
        }
            /* 无匹配 */
        else {
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