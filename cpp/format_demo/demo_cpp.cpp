#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include "feature_processor.h" // Includes all necessary headers like opencv and json

// 工具函数：从缓存目录加载一个 packet
Packet load_packet_from_cache(const std::string &cam_id, int fid, const std::string &root_dir) {
    Packet pkt;
    pkt.cam_id = cam_id;
    pkt.fid = fid;

    char fid_str[7];
    sprintf(fid_str, "%06d", fid);
    std::filesystem::path frame_dir = std::filesystem::path(root_dir) / cam_id / fid_str;

    if (!std::filesystem::exists(frame_dir)) {
        throw std::runtime_error("Cache directory not found for fid " + std::to_string(fid));
    }
    std::ifstream ifs(frame_dir / "meta.json");
    if (!ifs.is_open()) {
        throw std::runtime_error("meta.json not found in " + frame_dir.string());
    }

    nlohmann::json meta;
    ifs >> meta;

    // 按 patch 名称排序以保证顺序
    std::vector<std::string> patch_names = meta["patches"].get<std::vector<std::string>>();
    std::sort(patch_names.begin(), patch_names.end());

    std::map<int, cv::Mat> patch_map_by_idx;
    for (const auto &name: patch_names) {
        int patch_idx = std::stoi(name.substr(name.find('_') + 1, 2));
        patch_map_by_idx[patch_idx] = cv::imread((frame_dir / name).string());
    }

    for (const auto &d_json: meta["dets"]) {
        Detection det;
        auto tlwh_vec = d_json["tlwh"].get<std::vector<float>>();
        det.tlwh = cv::Rect2f(tlwh_vec[0], tlwh_vec[1], tlwh_vec[2], tlwh_vec[3]);
        det.score = d_json.value("score", 0.0f);
        det.id = d_json.value("id", -1);
        pkt.dets.push_back(det);
    }

    // 重要：确保 dets 和 patches 按照 Python 中的 sort_dets_and_patches 逻辑对齐
    std::vector<size_t> indices(pkt.dets.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        const auto &det_a = pkt.dets[a];
        const auto &det_b = pkt.dets[b];
        if (det_a.id != det_b.id) return det_a.id < det_b.id;
        if (std::abs(det_a.score - det_b.score) > 1e-5) return det_a.score > det_b.score; // 降序
        if (std::abs(det_a.tlwh.x - det_b.tlwh.x) > 1e-5) return det_a.tlwh.x < det_b.tlwh.x;
        if (std::abs(det_a.tlwh.y - det_b.tlwh.y) > 1e-5) return det_a.tlwh.y < det_b.tlwh.y;
        if (std::abs(det_a.tlwh.width - det_b.tlwh.width) > 1e-5) return det_a.tlwh.width < det_b.tlwh.width;
        return det_a.tlwh.height < det_b.tlwh.height;
    });

    std::vector<Detection> sorted_dets;
    for (size_t original_idx: indices) {
        sorted_dets.push_back(pkt.dets[original_idx]);
        pkt.patches.push_back(patch_map_by_idx[original_idx]);
    }
    pkt.dets = sorted_dets;

    return pkt;
}

int main(int argc, char **argv) {
    // --- 可调参数 ---
    std::string VIDEO_PATH = "/home/manu/tmp/64.mp4";
    std::string RAW_DIR = "/home/manu/tmp/cache_v0";
    std::string FEATURE_CACHE_JSON = "/home/manu/tmp/features_cache_v0.json";
    std::string OUTPUT_TXT = "/home/manu/tmp/output_result_cpp.txt";
    std::string CAM_ID = "cam1";
    int SKIP = 50;

    std::string MODE = "load"; // "load" or "realtime"
    if (argc > 1) {
        MODE = argv[1];
    }

    try {
        FeatureProcessor processor(MODE, "cuda", FEATURE_CACHE_JSON);

        cv::VideoCapture cap(VIDEO_PATH);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open video to get frame count: " << VIDEO_PATH << std::endl;
            return -1;
        }
        int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
        cap.release();

        std::ofstream fout(OUTPUT_TXT);
        fout << "frame_id,cam_id,tid,gid,score,n_tid\n";
        fout << std::fixed << std::setprecision(4);

        for (int fid = 1; fid <= total_frames; ++fid) {
            if (fid % SKIP != 0) continue;

            std::cout << "\rProcessing frame " << fid << "/" << total_frames << std::flush;

            try {
                Packet packet = load_packet_from_cache(CAM_ID, fid, RAW_DIR);

                auto mp = processor.process_packet(packet);
                if (mp.find(CAM_ID) == mp.end() || mp.at(CAM_ID).empty()) continue;

                std::vector<int> tids;
                for (auto const &[tid, val]: mp.at(CAM_ID)) tids.push_back(tid);
                std::sort(tids.begin(), tids.end());

                for (int tid: tids) {
                    auto const &tpl = mp.at(CAM_ID).at(tid);
                    fout << fid << ',' << CAM_ID << ',' << tid << ','
                         << std::get<0>(tpl) << ',' << std::get<1>(tpl) << ',' << std::get<2>(tpl) << "\n";
                }
            } catch (const std::runtime_error &e) {
                // 如果某个帧的缓存不存在，则跳过
                // std::cerr << "\nSkipping frame " << fid << ": " << e.what() << std::endl;
                continue;
            }
        }
        std::cout << "\nDONE -> " << OUTPUT_TXT << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "\nAn unrecoverable error occurred: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}