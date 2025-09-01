#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include "feature_processor.h" // Includes all necessary headers like opencv and json

// ======================= 【新增的部分在此】 =======================
// 辅助函数：从完整的信息字符串中解析出用于显示的 GID 或状态
std::string parse_gid_for_display(const std::string &info_str) {
    size_t first_underscore = info_str.find('_');
    if (first_underscore == std::string::npos) return info_str; //格式不符

    size_t second_underscore = info_str.find('_', first_underscore + 1);
    if (second_underscore == std::string::npos) return info_str; //格式不符

    return info_str.substr(second_underscore + 1);
}

// 辅助函数：为不同的 GID 生成一个相对稳定的颜色
cv::Scalar get_color_for_gid(const std::string &gid_str) {
    if (gid_str.rfind("G", 0) != 0) { // 不是 'G' 开头，是状态码
        return cv::Scalar(0, 0, 255); // 红色表示状态/错误
    }
    try {
        int gid_num = std::stoi(gid_str.substr(1));
        // 一个简单的哈希函数来生成颜色
        unsigned int hash = (gid_num * 2654435761);
        unsigned char r = (hash & 0xFF0000) >> 16;
        unsigned char g = (hash & 0x00FF00) >> 8;
        unsigned char b = (hash & 0x0000FF);
        return cv::Scalar(b, g, r);
    } catch (...) {
        return cv::Scalar(255, 255, 255); // 白色作为备用
    }
}
// ======================= 【新增结束】 =======================

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

    std::map<int, cv::Mat> patch_map_by_idx;
    if (meta.contains("patches")) {
        std::vector<std::string> patch_names = meta["patches"].get<std::vector<std::string>>();
        for (const auto &name: patch_names) {
            int patch_idx = std::stoi(name.substr(name.find('_') + 1, 2));
            patch_map_by_idx[patch_idx] = cv::imread((frame_dir / name).string());
        }
    }

    for (const auto &d_json: meta["dets"]) {
        Detection det;
        auto tlwh_vec = d_json["tlwh"].get<std::vector<float>>();
        det.tlwh = cv::Rect2f(tlwh_vec[0], tlwh_vec[1], tlwh_vec[2], tlwh_vec[3]);
        det.score = d_json.value("score", 0.0f);
        det.id = d_json.value("id", -1);
        pkt.dets.push_back(det);
    }

    std::vector<size_t> indices(pkt.dets.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        const auto &det_a = pkt.dets[a];
        const auto &det_b = pkt.dets[b];
        if (det_a.id != det_b.id) return det_a.id < det_b.id;
        if (std::abs(det_a.score - det_b.score) > 1e-5) return det_a.score > det_b.score;
        const auto &tlwh_a = det_a.tlwh;
        const auto &tlwh_b = det_b.tlwh;
        if (std::abs(tlwh_a.x - tlwh_b.x) > 1e-5) return tlwh_a.x < tlwh_b.x;
        if (std::abs(tlwh_a.y - tlwh_b.y) > 1e-5) return tlwh_a.y < tlwh_b.y;
        if (std::abs(tlwh_a.width - tlwh_b.width) > 1e-5) return tlwh_a.width < tlwh_b.width;
        return tlwh_a.height < tlwh_b.height;
    });

    std::vector<Detection> sorted_dets;
    std::vector<cv::Mat> sorted_patches;
    for (size_t original_idx: indices) {
        sorted_dets.push_back(pkt.dets[original_idx]);
        if (patch_map_by_idx.count(original_idx)) {
            sorted_patches.push_back(patch_map_by_idx[original_idx]);
        }
    }
    pkt.dets = sorted_dets;
    pkt.patches = sorted_patches;

    return pkt;
}

int main(int argc, char **argv) {
    // --- 可调参数 ---
    std::string VIDEO_PATH = "/home/manu/tmp/64.mp4";
    std::string RAW_DIR = "/home/manu/tmp/cache_v1";
    std::string CAM_ID = "cam1";
    int SKIP = 2;

    std::string MODE = "realtime";
    std::string FEATURE_CACHE_JSON = "/home/manu/tmp/features_cache_realtime_output.json";
    std::string OUTPUT_TXT = "/home/manu/tmp/output_result_cpp_realtime.txt";

    // ======================= 【修改的部分在此】 =======================
    // 1. 增加输出视频路径
    std::string OUTPUT_VIDEO_PATH = "/home/manu/tmp/output_video_cpp_realtime.mp4";
    // ======================= 【修改结束】 =======================

    if (argc > 1) {
        MODE = argv[1];
    }

    nlohmann::json boundary_config;

    try {
        FeatureProcessor processor(MODE, "cuda", FEATURE_CACHE_JSON, boundary_config);

        // ======================= 【修改的部分在此】 =======================
        // 2. 打开视频文件用于读取帧和写入
        cv::VideoCapture cap(VIDEO_PATH);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open video for reading: " << VIDEO_PATH << std::endl;
            return -1;
        }
        int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
        cv::Size frame_size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);

        cv::VideoWriter writer;
        writer.open(OUTPUT_VIDEO_PATH, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frame_size, true);
        if (!writer.isOpened()) {
            std::cerr << "Cannot open video for writing: " << OUTPUT_VIDEO_PATH << std::endl;
            return -1;
        }
        // ======================= 【修改结束】 =======================

        std::ofstream fout(OUTPUT_TXT);
        fout << "frame_id,cam_id,tid,gid,score,n_tid\n";
        fout << std::fixed << std::setprecision(4);

        int fid = 0;
        cv::Mat frame;

        // ======================= 【修改的部分在此】 =======================
        // 3. 将主循环改为读取视频帧的 while 循环
        while (cap.read(frame)) {
            fid++; // 帧号从1开始

            if (fid % SKIP != 0) {
                writer.write(frame); // 未处理的帧直接写入
                continue;
            }

            int processed_frames_count = fid / SKIP;
            int total_to_process = total_frames / SKIP;
            std::cout << "\rProcessing frame " << fid << "/" << total_frames
                      << " (" << processed_frames_count << "/" << total_to_process << ")" << std::flush;

            try {
                Packet packet = load_packet_from_cache(CAM_ID, fid, RAW_DIR);
                auto mp = processor.process_packet(packet);

                // --- 绘制标注 ---
                if (mp.count(CAM_ID)) {
                    const auto &results_for_cam = mp.at(CAM_ID);
                    // 遍历当前帧的所有检测框
                    for (const auto &det: packet.dets) {
                        std::string label;
                        cv::Scalar color(0, 0, 255); // 默认红色

                        // 在处理结果中查找当前 tid
                        if (results_for_cam.count(det.id)) {
                            const auto &tpl = results_for_cam.at(det.id);
                            std::string info_str = std::get<0>(tpl);
                            std::string gid_display = parse_gid_for_display(info_str);
                            int n_tid = std::get<2>(tpl);
                            label = "ID:" + std::to_string(det.id) + " GID:" + gid_display + " N:" +
                                    std::to_string(n_tid);
                            color = get_color_for_gid(gid_display);
                        } else {
                            // 如果结果中没有，可能它被过滤了，但也显示出来
                            label = "ID:" + std::to_string(det.id);
                        }

                        // 绘制矩形框
                        cv::rectangle(frame, det.tlwh, color, 2);
                        // 绘制文字
                        cv::Point text_origin(det.tlwh.x, det.tlwh.y - 10);
                        if (text_origin.y < 10) text_origin.y = det.tlwh.y + 20; // 防止文字超出画面
                        cv::putText(frame, label, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
                    }
                }

                // --- 写入文本结果到文件 (逻辑不变) ---
                if (mp.count(CAM_ID) && !mp.at(CAM_ID).empty()) {
                    std::vector<int> tids;
                    for (auto const &[tid, val]: mp.at(CAM_ID)) tids.push_back(tid);
                    std::sort(tids.begin(), tids.end());

                    for (int tid: tids) {
                        auto const &tpl = mp.at(CAM_ID).at(tid);
                        fout << fid << ',' << CAM_ID << ',' << tid << ','
                             << std::get<0>(tpl) << ',' << std::get<1>(tpl) << ',' << std::get<2>(tpl) << "\n";
                    }
                }
            } catch (const std::runtime_error &e) {
                // 如果某个帧的缓存不存在，则跳过处理，但帧还是要写入
            }

            writer.write(frame); // 写入处理并标注过的帧
        }
        // ======================= 【修改结束】 =======================

        std::cout << "\nDONE -> " << OUTPUT_TXT << " and " << OUTPUT_VIDEO_PATH << std::endl;

        // 释放资源
        cap.release();
        writer.release();
        fout.close();

    } catch (const std::exception &e) {
        std::cerr << "\nAn unrecoverable error occurred: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}