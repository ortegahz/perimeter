#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sstream> // 用于 std::ostringstream
#include "feature_processor.h" // Includes all necessary headers like opencv and json

// 工具函数：从缓存目录加载一个 packet (保持不变)
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
            // patch_00.bmp -> 0
            int patch_idx = std::stoi(name.substr(name.find('_') + 1, 2));
            patch_map_by_idx[patch_idx] = cv::imread((frame_dir / name).string());
        }
    }

    std::vector<Detection> temp_dets;
    std::vector<int> original_indices; // 记录每个det在json中的原始索引
    int current_idx = 0;
    for (const auto &d_json: meta["dets"]) {
        Detection det;
        auto tlwh_vec = d_json["tlwh"].get<std::vector<float>>();
        det.tlwh = cv::Rect2f(tlwh_vec[0], tlwh_vec[1], tlwh_vec[2], tlwh_vec[3]);
        det.score = d_json.value("score", 0.0f);
        det.id = d_json.value("id", -1);
        temp_dets.push_back(det);
        original_indices.push_back(current_idx++);
    }

    // 按照 python 的 sort_dets_and_patches 逻辑排序
    std::sort(original_indices.begin(), original_indices.end(), [&](int a, int b) {
        const auto &det_a = temp_dets[a];
        const auto &det_b = temp_dets[b];
        if (det_a.id != det_b.id) return det_a.id < det_b.id;
        if (std::abs(det_a.score - det_b.score) > 1e-5) return det_a.score > det_b.score;
        const auto &tlwh_a = det_a.tlwh;
        const auto &tlwh_b = det_b.tlwh;
        if (std::abs(tlwh_a.x - tlwh_b.x) > 1e-5) return tlwh_a.x < tlwh_b.x;
        if (std::abs(tlwh_a.y - tlwh_b.y) > 1e-5) return tlwh_a.y < tlwh_b.y;
        if (std::abs(tlwh_a.width - tlwh_b.width) > 1e-5) return tlwh_a.width < tlwh_b.width;
        return tlwh_a.height < tlwh_b.height;
    });

    for (int original_idx: original_indices) {
        pkt.dets.push_back(temp_dets[original_idx]);
        if (patch_map_by_idx.count(original_idx)) {
            pkt.patches.push_back(patch_map_by_idx[original_idx]);
        }
    }

    return pkt;
}

int main(int argc, char **argv) {
    // --- 可调参数 ---
    std::string VIDEO_PATH = "/home/manu/tmp/64.mp4";
    std::string RAW_DIR = "/home/manu/tmp/cache_v1";
    std::string CAM_ID = "cam1";
    int SKIP = 2;
    float SHOW_SCALE = 0.5; // <-- 新增：与Python对齐

    // 根据模式选择不同的输入/输出
    std::string MODE = "load"; // 默认为 "load" 模式
    if (argc > 1) {
        MODE = argv[1];
    }

    std::string FEATURE_CACHE_JSON, OUTPUT_TXT, OUTPUT_VIDEO_PATH;
    if (MODE == "load") {
        FEATURE_CACHE_JSON = "/home/manu/tmp/features_cache_v1.json";
        OUTPUT_TXT = "/home/manu/tmp/output_result_cpp_load.txt";
        OUTPUT_VIDEO_PATH = "/home/manu/tmp/output_video_cpp_load.mp4";
    } else { // realtime
        FEATURE_CACHE_JSON = "/home/manu/tmp/features_cache_realtime_output.json";
        OUTPUT_TXT = "/home/manu/tmp/output_result_cpp_realtime.txt";
        OUTPUT_VIDEO_PATH = "/home/manu/tmp/output_video_cpp_realtime.mp4";
    }

    nlohmann::json boundary_config; // 留空

    try {
        FeatureProcessor processor(MODE, "cuda", FEATURE_CACHE_JSON, boundary_config);

        cv::VideoCapture cap(VIDEO_PATH);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open video for reading: " << VIDEO_PATH << std::endl;
            return -1;
        }
        int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
        double fps = cap.get(cv::CAP_PROP_FPS);
        int ori_W = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int ori_H = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        cv::Size vis_size(static_cast<int>(ori_W * SHOW_SCALE), static_cast<int>(ori_H * SHOW_SCALE));

        cv::VideoWriter writer;
        writer.open(OUTPUT_VIDEO_PATH, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, vis_size, true);
        if (!writer.isOpened()) {
            std::cerr << "Cannot open video for writing: " << OUTPUT_VIDEO_PATH << std::endl;
            return -1;
        }

        std::ofstream fout(OUTPUT_TXT);
        fout << "frame_id,cam_id,tid,gid,score,n_tid\n";
        fout << std::fixed << std::setprecision(4);

        int fid = 0;
        cv::Mat frame;

        while (cap.read(frame)) {
            fid++;
            if (fid % SKIP != 0) {
                // 对于跳过的帧，直接缩放后写入，以保证视频时间线正确
//                cv::Mat vis;
//                cv::resize(frame, vis, vis_size);
//                writer.write(vis);
                continue;
            }

            int processed_frames_count = fid / SKIP;
            int total_to_process = total_frames / SKIP;
            std::cout << "\rProcessing frame " << fid << "/" << total_frames
                      << " (" << processed_frames_count << "/" << total_to_process << ")" << std::flush;

            try {
                Packet packet = load_packet_from_cache(CAM_ID, fid, RAW_DIR);
                auto mp = processor.process_packet(packet);

                // --- 可视化 ---
                cv::Mat vis;
                cv::resize(frame, vis, vis_size); // 先缩放画布

                const auto &cam_map = mp.count(CAM_ID) ? mp.at(CAM_ID)
                                                       : std::map<int, std::tuple<std::string, float, int>>{};

                for (const auto &det: packet.dets) {
                    // 1. 获取 GID, score, n_tid (如果不存在则使用默认值)
                    std::string gid_str = "-1";
                    float score = -1.0f;
                    int n_tid = 0;

                    if (cam_map.count(det.id)) {
                        const auto &tpl = cam_map.at(det.id);
                        gid_str = std::get<0>(tpl);
                        score = std::get<1>(tpl);
                        n_tid = std::get<2>(tpl);
                    }

                    // 2. 根据 n_tid 决定颜色
                    cv::Scalar color = (n_tid < 2) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255); // Green vs Red

                    // 3. 计算缩放后的坐标
                    int x = static_cast<int>(det.tlwh.x * SHOW_SCALE);
                    int y = static_cast<int>(det.tlwh.y * SHOW_SCALE);
                    int w = static_cast<int>(det.tlwh.width * SHOW_SCALE);
                    int h = static_cast<int>(det.tlwh.height * SHOW_SCALE);

                    // 4. 绘制矩形框
                    cv::rectangle(vis, cv::Rect(x, y, w, h), color, 2);

                    // 5. 绘制两行文本
                    std::string text1 = "G:" + gid_str;
                    cv::putText(vis, text1, cv::Point(x, std::max(y + 15, 15)), cv::FONT_HERSHEY_SIMPLEX, 0.5, color,
                                1);

                    std::ostringstream ss;
                    ss << "n=" << n_tid << " s=" << std::fixed << std::setprecision(2) << score;
                    std::string text2 = ss.str();
                    cv::putText(vis, text2, cv::Point(x, std::max(y + 30, 30)), cv::FONT_HERSHEY_SIMPLEX, 0.4, color,
                                1);
                }

                writer.write(vis); // 将绘制好的帧写入视频

                // --- 写入文本结果到文件 (和以前一样，但需要排序) ---
                if (mp.count(CAM_ID)) {
                    std::vector<int> sorted_tids;
                    for (const auto &pair: mp.at(CAM_ID)) {
                        sorted_tids.push_back(pair.first);
                    }
                    std::sort(sorted_tids.begin(), sorted_tids.end());

                    for (int tid: sorted_tids) {
                        const auto &tpl = mp.at(CAM_ID).at(tid);
                        fout << fid << ',' << CAM_ID << ',' << tid << ','
                             << std::get<0>(tpl) << ',' << std::get<1>(tpl) << ',' << std::get<2>(tpl) << "\n";
                    }
                }

            } catch (const std::runtime_error &e) {
                // 加载 packet 失败，写入原始缩放帧
                std::cout << "\nWarning: Failed to process frame " << fid << ". " << e.what() << std::endl;
                cv::Mat vis;
                cv::resize(frame, vis, vis_size);
                writer.write(vis);
            }
        }

        std::cout << "\nDONE -> " << OUTPUT_TXT << " and " << OUTPUT_VIDEO_PATH << std::endl;
        cap.release();
        writer.release();
        fout.close();

    } catch (const std::exception &e) {
        std::cerr << "\nAn unrecoverable error occurred: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}