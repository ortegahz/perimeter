#include "feature_processor.h"
#include <fstream>
#include <algorithm>
#include <iomanip>          // **** NEW ****
#include <cmath>
#include <stdexcept>

using json = nlohmann::json;

/* ---------- 工具 ---------- */
float sim_vec(const std::vector<float> &a, const std::vector<float> &b) {
    float s = 0.f;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

std::vector<float> avg_feats(const std::vector<std::vector<float>> &feats) {
    std::vector<float> mean(feats[0].size(), 0.f);
    for (auto &f: feats)
        for (size_t i = 0; i < f.size(); ++i) mean[i] += f[i];
    for (float &v: mean) v /= feats.size();
    float norm = 1e-9f;
    for (float v: mean) norm += v * v;
    norm = std::sqrt(norm);
    for (float &v: mean) v /= norm;
    return mean;
}

/* ---------- TrackAgg ---------- */
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
    std::vector<std::vector<float>> feats;
    feats.reserve(body.size());
    for (auto &b: body) feats.push_back(std::get<0>(b));
    return avg_feats(feats);
}

std::vector<float> TrackAgg::main_face_feat() const {
    if (face.empty()) return {};
    std::vector<std::vector<float>> feats(face.begin(), face.end());
    return avg_feats(feats);
}

/* ---------- GlobalID ---------- */
std::string GlobalID::new_gid() {
    char buf[32];
    sprintf(buf, "G%05d", gid_next++);
    std::string gid(buf);
    bank_faces[gid] = {};
    bank_bodies[gid] = {};
    tid_hist[gid] = {};
    return gid;
}

/* ---- 修改为 0/-1/-2 ---- */
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
    if (!face_f.empty()) bank_faces[gid].push_back(face_f);
    if (!body_f.empty()) bank_bodies[gid].push_back(body_f);
    if (std::find(tid_hist[gid].begin(), tid_hist[gid].end(), tid) == tid_hist[gid].end())
        tid_hist[gid].push_back(tid);
}

std::pair<std::string, float> GlobalID::probe(const std::vector<float> &face_f,
                                              const std::vector<float> &body_f) {
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

/* ---------- FeatureProcessor ---------- */
FeatureProcessor::FeatureProcessor(const std::string &cache_path) {
    std::ifstream jf(cache_path);
    if (!jf.is_open()) throw std::runtime_error("Failed to open cache: " + cache_path);
    jf >> features_cache;
}

std::map<std::string, std::map<int, std::tuple<std::string, float, int>>>
FeatureProcessor::process_packet(const std::string &stream_id, int fid) {
    std::map<std::string, std::map<int, std::tuple<std::string, float, int>>> realtime_map;

    /* 1. 读取缓存 */
    if (!features_cache.contains(std::to_string(fid))) return realtime_map;
    auto pre_f = features_cache[std::to_string(fid)];

    /* 2. 更新 agg_pool */
    for (auto &[tid_str, feat_dict]: pre_f.items()) {
        std::vector<float> body_feat, face_feat;
        if (feat_dict["body_feat"].is_array())
            for (auto &v: feat_dict["body_feat"]) body_feat.push_back(v.get<float>());
        if (feat_dict["face_feat"].is_array())
            for (auto &v: feat_dict["face_feat"]) face_feat.push_back(v.get<float>());

        auto &agg = agg_pool[tid_str];
        if (!body_feat.empty()) agg.add_body(body_feat, 1.f);
        if (!face_feat.empty()) agg.add_face(face_feat);
    }

    /* 3. 绑定 / 新建 GID，与 python-load 逻辑保持一致 */
    for (auto &[tid_str, agg]: agg_pool) {
        const int tid_num = std::stoi(tid_str.substr(tid_str.find("_") + 1));

        /* 3-a 样本数量不足 */
        if ((int) agg.body.size() < MIN_BODY4GID || (int) agg.face.size() < MIN_FACE4GID) {
            realtime_map[stream_id][tid_num] = {"-1", -1.f, 0};
            continue;
        }

        auto face_f = agg.main_face_feat();
        auto body_f = agg.main_body_feat();
        if (face_f.empty() || body_f.empty()) {
            realtime_map[stream_id][tid_num] = {"-2", -1.f, 0};
            continue;
        }

        auto [cand_gid, score] = gid_mgr.probe(face_f, body_f);

        /* 3-b 匹配成功 */
        if (!cand_gid.empty() && score >= MATCH_THR) {
            int flag = gid_mgr.can_update_proto(cand_gid, face_f, body_f);
            if (flag == 0) {                                  // 可以更新
                gid_mgr.bind(cand_gid, face_f, body_f, tid_str);
                tid2gid[tid_str] = cand_gid;
                int n_tid = gid_mgr.tid_hist[cand_gid].size();
                realtime_map[stream_id][tid_num] = {cand_gid, score, n_tid};
            } else {                                        // 不能更新
                realtime_map[stream_id][tid_num] = {"-4", -1.f, 0};
            }
        }
            /* 3-c 库为空 → 新建 */
        else if (gid_mgr.bank_faces.empty()) {
            std::string new_gid = gid_mgr.new_gid();
            gid_mgr.bind(new_gid, face_f, body_f, tid_str);
            tid2gid[tid_str] = new_gid;
            int n_tid = gid_mgr.tid_hist[new_gid].size();
            realtime_map[stream_id][tid_num] = {new_gid, score, n_tid};
        }
            /* 3-d 其他情况 */
        else {
            realtime_map[stream_id][tid_num] = {"-5", -1.f, 0};
        }
    }
    return realtime_map;
}