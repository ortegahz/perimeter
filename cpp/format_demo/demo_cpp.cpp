#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <chrono>
#include "feature_processor.h" // Includes all necessary headers like opencv and json

// -------- 新增：模型路径常量，与 feature_processor.cpp 保持一致 --------
const std::string FACE_DET_MODEL_PATH = "/mnt/nfs/det_10g_simplified.onnx";
const std::string FACE_REC_MODEL_PATH = "/mnt/nfs/w600k_r50_simplified.onnx";
// ----------------------------------------------------------------------

// -------- 修改：load_packet_from_cache 返回一个包含 packet 和 face_info 的结构体 --------
struct LoadedData {
    Packet packet;
    std::vector<Face> face_info;
};

LoadedData load_packet_from_cache(const std::string &cam_id, int fid, const std::string &root_dir) {
    LoadedData data;
    data.packet.cam_id = cam_id;
    data.packet.fid = fid;

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

    // 加载 Patches (为兼容旧的Packet结构，但Processor不再直接使用)
    std::map<int, cv::Mat> patch_map_by_idx;
    if (meta.contains("patches")) {
        std::vector<std::string> patch_names = meta["patches"].get<std::vector<std::string>>();
        for (const auto &name: patch_names) {
            int patch_idx = std::stoi(name.substr(name.find('_') + 1, 2));
            patch_map_by_idx[patch_idx] = cv::imread((frame_dir / name).string());
        }
    }

    // 加载并排序 Dets
    std::vector<Detection> temp_dets;
    std::vector<int> original_indices;
    int current_idx = 0;
    for (const auto &d_json: meta["dets"]) {
        Detection det;
        auto tlwh_vec = d_json["tlwh"].get<std::vector<float>>();
        det.tlwh = cv::Rect2f(tlwh_vec[0], tlwh_vec[1], tlwh_vec[2], tlwh_vec[3]);
        det.score = d_json.value("score", 0.0f);
        det.id = d_json.value("id", -1);
        // 新增: 从json中读取class_id，以匹配Python过滤逻辑
        det.class_id = d_json.value("class_id", 0);
        temp_dets.push_back(det);
        original_indices.push_back(current_idx++);
    }

    std::sort(original_indices.begin(), original_indices.end(), [&](int a, int b) {
        const auto &det_a = temp_dets[a];
        const auto &det_b = temp_dets[b];
        if (det_a.id != det_b.id) return det_a.id < det_a.id;
        if (std::abs(det_a.score - det_b.score) > 1e-5) return det_a.score > det_b.score;
        const auto &tlwh_a = det_a.tlwh;
        const auto &tlwh_b = det_b.tlwh;
        if (std::abs(tlwh_a.x - tlwh_b.x) > 1e-5) return tlwh_a.x < tlwh_b.x;
        if (std::abs(tlwh_a.y - tlwh_b.y) > 1e-5) return tlwh_a.y < tlwh_b.y;
        if (std::abs(tlwh_a.width - tlwh_b.width) > 1e-5) return tlwh_a.width < tlwh_b.width;
        return tlwh_a.height < tlwh_b.height;
    });

    for (int original_idx: original_indices) {
        data.packet.dets.push_back(temp_dets[original_idx]);
        if (patch_map_by_idx.count(original_idx)) {
            data.packet.patches.push_back(patch_map_by_idx[original_idx]);
        }
    }

    // 新增：加载 face_info (仅用于可视化)
    if (meta.contains("face_info") && meta["face_info"].is_array()) {
        for (const auto &face_json: meta["face_info"]) {
            Face face;
            auto bbox_vec = face_json["bbox"].get<std::vector<float>>();
            face.bbox = cv::Rect2d(bbox_vec[0], bbox_vec[1], bbox_vec[2] - bbox_vec[0], bbox_vec[3] - bbox_vec[1]);
            face.det_score = face_json.value("score", 0.0f);
            if (face_json.contains("kps") && !face_json["kps"].is_null()) {
                for (const auto &kp_json: face_json["kps"]) {
                    face.kps.emplace_back(kp_json[0].get<float>(), kp_json[1].get<float>());
                }
            }
            data.face_info.push_back(face);
        }
    }
    return data;
}

int main(int argc, char **argv) {
    // --- 可调参数 ---
    std::string VIDEO_PATH = "/mnt/nfs/64.mp4";
    // 注意：请确保 RAW_DIR 指向与 Python 端一致的 v2 版本缓存（包含face_info）
    std::string RAW_DIR = "/mnt/nfs/cache_v2";
    std::string CAM_ID = "cam1";
    int SKIP = 2;
    float SHOW_SCALE = 0.5;

    std::string MODE = "load"; // 默认为 "load" 模式
    if (argc > 1) {
        MODE = argv[1];
    }

    std::cout << "Running in " << MODE << " mode." << std::endl;

    std::string FEATURE_CACHE_JSON, OUTPUT_TXT, OUTPUT_VIDEO_PATH;
    if (MODE == "load") {
        FEATURE_CACHE_JSON = "/mnt/nfs/features_cache_v2.json";
        OUTPUT_TXT = "/mnt/nfs/output_result_cpp_load.txt";
        OUTPUT_VIDEO_PATH = "/mnt/nfs/output_video_cpp_load.mp4";
    } else { // realtime
        // 注意：realtime模式的C++版本依赖ByteTrack的C++实现，此处未提供
        // 为演示逻辑，假设ByteTrack已运行并将结果存入cache_v2
        std::cout << "Warning: C++ realtime mode assumes trackers (like ByteTrack) are implemented separately."
                  << std::endl;
        std::cout << "This example will read from RAW_DIR even in realtime mode, but will re-extract features."
                  << std::endl;
        RAW_DIR = "/mnt/nfs/cache_v2"; // realtime模式也从这里读检测结果
        FEATURE_CACHE_JSON = "/mnt/nfs/features_cache_cpp_realtime_output.json";
        OUTPUT_TXT = "/mnt/nfs/output_result_cpp_realtime.txt";
        OUTPUT_VIDEO_PATH = "/mnt/nfs/output_video_cpp_realtime.mp4";
    }

    nlohmann::json boundary_config; // 留空

    try {
        FeatureProcessor processor(MODE, "dla", FEATURE_CACHE_JSON, boundary_config);

        // 注意：在realtime模式下，FeatureProcessor会创建自己的FaceAnalyzer实例。
        // 此处的face_analyzer仅用于演示，实际处理在processor内部完成。

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
        cv::cuda::GpuMat gpu_frame;

        double total_proc_time = 0.0;
        int proc_count = 0;

        while (cap.read(frame)) {
            fid++;
            if (fid % SKIP != 0) {
                continue;
            }

            int processed_frames_count = fid / SKIP;
            int total_to_process = total_frames / SKIP;
            std::cout << "\rProcessing frame " << fid << "/" << total_frames
                      << " (" << processed_frames_count << "/" << total_to_process << ")";

            try {
                gpu_frame.upload(frame);
                // 修改：统一从缓存加载基础检测信息 (dets) 和可视化信息 (face_info)
                LoadedData loaded_data = load_packet_from_cache(CAM_ID, fid, RAW_DIR);

                // --- 新增：定义本次处理的配置参数 ---
                ProcessConfig proc_config;
                proc_config.alarm_cnt_th = 2;  // 示例：将全局告警计数阈值改为3
                proc_config.match_thr_by_cam[CAM_ID] = 0.5f; // 示例：为当前相机"cam1"设置特定的匹配阈值

                // ---- 计时 ----
                auto t1 = std::chrono::high_resolution_clock::now();
                // MODIFIED HERE: 调用新的 process_packet 接口
                auto mp = processor.process_packet(CAM_ID, fid, gpu_frame, loaded_data.packet.dets, proc_config);
                auto t2 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> proc_time = t2 - t1;
                std::cout << "  [proc_packet took " << proc_time.count() << " ms]" << std::endl;

                total_proc_time += proc_time.count();
                proc_count++;

                // --- 可视化 ---
                cv::Mat vis;
                cv::resize(frame, vis, vis_size);

                const auto &cam_map = mp.count(CAM_ID) ? mp.at(CAM_ID)
                                                       : std::map<int, std::tuple<std::string, float, int>>{};

                // ======================= 【可视化部分保持不变】 =======================
                for (const auto &det: loaded_data.packet.dets) {
                    // --- 默认值 ---
                    float score = -1.0f;
                    int n_tid = 0;
                    std::string display_status = "G:-?";
                    int color_id = det.id;
                    cv::Scalar color_override(-1, -1, -1); // 用于强制指定颜色的哨兵值

                    // --- 检查此 TID 是否有处理器返回的结果 ---
                    if (cam_map.count(det.id)) {
                        const auto &tpl = cam_map.at(det.id);
                        const std::string &full_gid_str = std::get<0>(tpl);
                        score = std::get<1>(tpl);
                        n_tid = std::get<2>(tpl);

                        // --- 解析 full_gid_str 以生成用于显示的状态文本 ---
                        bool gid_found = false;
                        size_t g_pos = full_gid_str.find("_G");
                        if (g_pos != std::string::npos) {
                            std::string gid_part = full_gid_str.substr(g_pos + 1);
                            // GID 后面可能跟着告警信息, 例如 G00001_AA
                            size_t next_underscore = gid_part.find('_');
                            if (next_underscore != std::string::npos) {
                                gid_part = gid_part.substr(0, next_underscore);
                            }

                            if (gid_part.rfind("G", 0) == 0) {
                                display_status = gid_part;
                                try { color_id = std::stoi(gid_part.substr(1)); } catch (...) {}
                                gid_found = true;
                            }
                        }

                        // 检查行为告警，它可能附加在 GID 之后
                        if (full_gid_str.find("_AA") != std::string::npos) {
                            display_status = (gid_found ? display_status + " " : "") + "Intrusion!";
                            color_override = cv::Scalar(0, 0, 255); // 红色
                        } else if (full_gid_str.find("_AL") != std::string::npos) {
                            display_status = (gid_found ? display_status + " " : "") + "Crossing!";
                            color_override = cv::Scalar(0, 0, 255); // 红色
                        }
                            // 如果没找到 GID 且没有告警，则说明是调试状态
                        else if (!gid_found) {
                            if (full_gid_str.find("_-1_b_") != std::string::npos) {
                                size_t pos = full_gid_str.find("_-1_b_");
                                display_status =
                                        "body<" + std::to_string(MIN_BODY4GID) + " (" + full_gid_str.substr(pos + 6) +
                                        ")";
                            } else if (full_gid_str.find("_-1_f_") != std::string::npos) {
                                size_t pos = full_gid_str.find("_-1_f_");
                                display_status =
                                        "face<" + std::to_string(MIN_FACE4GID) + " (" + full_gid_str.substr(pos + 6) +
                                        ")";
                            } else if (full_gid_str.find("_-2_f") != std::string::npos) {
                                display_status = "Face Inconsist";
                            } else if (full_gid_str.find("_-2_b") != std::string::npos) {
                                display_status = "Body Inconsist";
                            } else if (full_gid_str.find("_-3") != std::string::npos) {
                                display_status = "Bind Lock";
                            } else if (full_gid_str.find("_-4_ud_f") != std::string::npos) {
                                display_status = "Update Face No";
                            } else if (full_gid_str.find("_-4_ud_b") != std::string::npos) {
                                display_status = "Update Body No";
                            } else if (full_gid_str.find("_-4_c") != std::string::npos) {
                                display_status = "Cand Wait";
                            } else if (full_gid_str.find("_-5") != std::string::npos) {
                                display_status = "New GID Wait";
                            } else if (full_gid_str.find("_-6") != std::string::npos) {
                                display_status = "New GID Cool";
                            } else if (full_gid_str.find("_-7") != std::string::npos) {
                                display_status = "Ambig Wait";
                            }
                        }
                    }

                    // --- 确定边界框颜色 ---
                    const cv::Scalar colors[] = {{0,   0,   255},
                                                 {0,   255, 0},
                                                 {255, 255, 0},
                                                 {0,   255, 255},
                                                 {255, 0,   255},
                                                 {128, 0,   0},
                                                 {0,   128, 0},
                                                 {0,   0,   128}};
                    cv::Scalar color = (color_override[0] != -1)
                                       ? color_override
                                       : colors[color_id % (sizeof(colors) / sizeof(cv::Scalar))];

                    // --- 获取框的坐标 ---
                    int x = static_cast<int>(det.tlwh.x * SHOW_SCALE);
                    int y = static_cast<int>(det.tlwh.y * SHOW_SCALE);
                    int w = static_cast<int>(det.tlwh.width * SHOW_SCALE);
                    int h = static_cast<int>(det.tlwh.height * SHOW_SCALE);

                    // --- 绘制框和文本 ---
                    cv::rectangle(vis, cv::Rect(x, y, w, h), color, 2);

                    // 顶部文本: Track ID 和 GID/状态
                    std::string display_text_top = "T:" + std::to_string(det.id) + " " + display_status;
                    cv::putText(vis, display_text_top, cv::Point(x, std::max(y - 5, 10)), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 1);

                    // 底部文本: n_tid 和 score
                    std::ostringstream ss_bottom;
                    ss_bottom << "n=" << n_tid << " s=" << std::fixed << std::setprecision(2) << score;
                    cv::putText(vis, ss_bottom.str(), cv::Point(x, std::max(y + h + 15, 15)), cv::FONT_HERSHEY_SIMPLEX,
                                0.4,
                                color, 1);
                }

                // 新增：可视化从缓存加载的人脸框 (face_info)
                for (const auto &face: loaded_data.face_info) {
                    int x1 = static_cast<int>(face.bbox.x * SHOW_SCALE);
                    int y1 = static_cast<int>(face.bbox.y * SHOW_SCALE);
                    int x2 = static_cast<int>((face.bbox.x + face.bbox.width) * SHOW_SCALE);
                    int y2 = static_cast<int>((face.bbox.y + face.bbox.height) * SHOW_SCALE);
                    cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2); // 红色

                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(2) << face.det_score;
                    cv::putText(vis, ss.str(), cv::Point(x1, std::max(y1 - 2, 0)), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                                cv::Scalar(0, 128, 255), 1);

                    for (const auto &kp: face.kps) {
                        cv::circle(vis, cv::Point(kp.x * SHOW_SCALE, kp.y * SHOW_SCALE), 1, cv::Scalar(255, 0, 0),
                                   2); // 蓝色
                    }
                }
                writer.write(vis);

                // 写入结果文件
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
                std::cout << "\nWarning: Failed to process frame " << fid << ". " << e.what() << std::endl;
                cv::Mat vis;
                cv::resize(frame, vis, vis_size);
                writer.write(vis);
            }
        }

        if (proc_count > 0) {
            std::cout << "\nAverage process_packet time = "
                      << (total_proc_time / proc_count) << " ms over "
                      << proc_count << " frames." << std::endl;
        }

        std::cout << "\nDONE -> " << OUTPUT_TXT << " and " << OUTPUT_VIDEO_PATH << std::endl;
        cap.release();
        writer.release();
        fout.close();

        // ======================= 【MODIFIED】 =======================
        // 新增：在程序结束前，保存内存状态以供验证。
        // processor的析构函数将在此代码块结束时被调用，它会等待所有I/O完成并关闭数据库。
        std::cout << "\n--- Final State Verification ---" << std::endl;
        processor.save_final_state_to_file("/mnt/nfs/state_before_shutdown.txt");
        // ======================= 【修改结束】 =======================

    } catch (const std::exception &e) {
        std::cerr << "\nAn unrecoverable error occurred: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}