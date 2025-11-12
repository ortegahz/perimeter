#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <chrono>
#include "feature_processor.h" // Includes all necessary headers like opencv and json
#include <opencv2/cudaimgproc.hpp>

// ======================= 【新增】 =======================
// GStreamer 相关的时间戳类型和处理函数
#ifndef GST_CLOCK_TIME_NONE
using GstClockTime = uint64_t;
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

/**
 * @brief 一个占位函数，用于模拟从 GStreamer 流中获取当前帧的 NTP 时间戳。
 * @param frame_id 当前帧号。
 * @param fps 视频的帧率。
 * @return 模拟的 GstClockTime 时间戳。
 */
GstClockTime get_current_frame_ntp_timestamp(uint64_t frame_id, double fps) {
    // 这是一个模拟实现。在您的 GStreamer 应用中，您应该从 GstBuffer 中提取真实的时间戳。
    // 例如：buffer->pts 或 buffer->dts

    // 为了演示，我们基于系统时钟模拟一个连续的时间戳
    auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()
    ).count();
    return static_cast<GstClockTime>(now_ns);
}
// ======================= 【新增结束】 =======================

// ======================= 【MODIFIED】 =======================
// 调整：将所有模型路径常量集中在此处，用于传递给构造函数
const std::string REID_MODEL_PATH = "/home/nvidia/VSCodeProject/smartboxcore/models/reid_model.onnx";
const std::string FACE_DET_MODEL_PATH = "/home/nvidia/VSCodeProject/smartboxcore/models/det_10g_simplified.onnx";
const std::string FACE_REC_MODEL_PATH = "/home/nvidia/VSCodeProject/smartboxcore/models/w600k_r50_simplified.onnx";
// ======================= 【修改结束】 =======================

// -------- 修改：load_packet_from_cache 返回一个包含 packet 和 face_info 的结构体 --------
struct LoadedData {
    Packet packet;
    std::vector<Face> face_info;
};

LoadedData load_packet_from_cache(const std::string &cam_id, uint64_t fid, const std::string &root_dir) {
    LoadedData data;
    data.packet.cam_id = cam_id;
    data.packet.fid = fid;

    std::stringstream ss_fid;
    ss_fid << std::setw(6) << std::setfill('0') << fid;
    std::filesystem::path frame_dir = std::filesystem::path(root_dir) / cam_id / ss_fid.str();

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
    std::string RAW_DIR = "/mnt/nfs/cache_v4";
    std::string CAM_ID = "cam1";
    int SKIP = 2;
    float SHOW_SCALE = 0.5;
    bool RUN_IN_LOOP = false; // 【修改】控制视频是否循环播放。true: 循环, false: 播放一次
    // 新增：是否在实时模式下启用特征缓存写入的功能。关闭可防止内存泄漏。
    bool ENABLE_FEATURE_CACHING = true;
    // 新增：是否在启动时清除现有的数据库。true: 清除, false: 加载
    bool CLEAR_DB_ON_STARTUP = true;

    std::string MODE = "load"; // realtime or load
    if (argc > 1) {
        MODE = argv[1];
    }

    std::cout << "Running in " << MODE << " mode." << std::endl;

    std::string FEATURE_CACHE_JSON, OUTPUT_TXT, OUTPUT_VIDEO_PATH;
    if (MODE == "load") {
        FEATURE_CACHE_JSON = "/mnt/nfs/features_cache_v4.json";
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

    // 新增：在主循环外加载一次配置文件，以供后续使用
    nlohmann::json config_json;
    if (std::filesystem::exists(CONFIG_FILE_PATH)) {
        std::ifstream ifs(CONFIG_FILE_PATH);
        try {
            config_json = nlohmann::json::parse(ifs);
            std::cout << "Successfully loaded settings from " << CONFIG_FILE_PATH << std::endl;
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "Warning: Could not parse config.json for settings. Using defaults. Error: " << e.what()
                      << std::endl;
        }
    }

    try {
        bool _use_fid_time = (MODE == "load");
        // ======================= 【MODIFIED】 =======================
        // 修改: 使用新的构造函数实例化 FeatureProcessor
        FeatureProcessor processor(
                REID_MODEL_PATH,
                FACE_DET_MODEL_PATH,
                FACE_REC_MODEL_PATH,
                MODE,                     // 明确传递，覆盖默认值
                "cuda",                    // 明确传递
                FEATURE_CACHE_JSON,       // 明确传递，覆盖默认值
                _use_fid_time,
                true,                     // enable_alarm_saving
                true,                     // processing_enabled: 新增算法总开关, 设置为 false 可禁用所有处理
                ENABLE_FEATURE_CACHING,   // 新增: 是否启用特征缓存写入的开关
                CLEAR_DB_ON_STARTUP);     // 新增: 是否在启动时清除数据库
        // ======================= 【修改结束】 =======================

        cv::VideoCapture cap(VIDEO_PATH, cv::CAP_FFMPEG);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open video for reading: " << VIDEO_PATH << std::endl;
            return -1;
        }
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        int ori_W = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int ori_H = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        cv::Size vis_size(static_cast<int>(ori_W * SHOW_SCALE), static_cast<int>(ori_H * SHOW_SCALE));

        cv::VideoWriter writer;
        writer.open(OUTPUT_VIDEO_PATH, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, vis_size, true);
        if (!writer.isOpened()) {
            std::cerr << "Cannot open video for writing: " << OUTPUT_VIDEO_PATH << std::endl;
            return -1;
        }

        std::ofstream fout(OUTPUT_TXT);
        fout
                << "frame_id,cam_id,tid,gid,score,n_tid,alarm_type,alarm_direction,alarm_distance,alarm_ratio,alarm_area\n";
        fout << std::fixed << std::setprecision(4);

        uint64_t fid = 0;          // 全局递增帧号（无限循环或单次）
        cv::Mat frame;
        cv::cuda::GpuMat gpu_frame;

        double total_proc_time = 0.0;
        int proc_count = 0;

        // ------------- 【修改】无限循环或单次运行 -------------
        while (true) {
            if (!cap.read(frame)) { // 读到结尾或异常
                if (RUN_IN_LOOP) {
                    std::cout << "\nVideo ended. Resetting to the beginning." << std::endl;
                    cap.set(cv::CAP_PROP_POS_FRAMES, 0); // 重置到首帧
                    if (!cap.read(frame)) {             // 复位后依旧失败，直接退出
                        std::cerr << "Failed to read frame even after reset. Exiting.\n";
                        break;
                    }
                } else {
                    // 非循环模式，视频播放完毕，退出循环
                    break;
                }
            }

            fid++;
            if (fid % SKIP != 0) {
                continue;
            }

            // 在循环内使用 cache_fid 对应首轮帧号，保证缓存目录存在
            uint64_t cache_fid = ((fid - 1) % total_frames) + 1;

            std::cout << "\rProcessing global_fid=" << fid
                      << " (cache_fid=" << cache_fid << '/' << total_frames << ")" << std::flush;

            try {
                gpu_frame.upload(frame);
                // 新增：将 BGR 帧转换为 RGB 帧，因为模型需要 RGB 输入
                cv::cuda::GpuMat gpu_frame_rgb;
                cv::cuda::cvtColor(gpu_frame, gpu_frame_rgb, cv::COLOR_BGR2RGB);
                // 修改：统一从缓存加载基础检测信息 (dets) 和可视化信息 (face_info)
                LoadedData loaded_data = load_packet_from_cache(CAM_ID, cache_fid, RAW_DIR);

                // --- 新增：定义本次处理的配置参数 ---
                ProcessConfig proc_config;
                proc_config.alarm_cnt_th = 2;  // 示例：将全局告警计数阈值改为2
                proc_config.sensitivity_by_cam[CAM_ID] = 5; // 示例: 设置为较高灵敏度 (1-10级, 1为最高, 对应阈值0.3)
                proc_config.alarm_record_thresh = 3; // 新增：设置报警记录阈值, n>3时才记录GID用于去重

                // 新增：人脸/ReID权重配置
                proc_config.face_switch_by_cam[CAM_ID] = true;
                proc_config.face_weight_by_cam[CAM_ID] = 0.6f;
                proc_config.reid_weight_by_cam[CAM_ID] = 0.4f;
                proc_config.alarmDuration_ms_by_cam[CAM_ID] = 0;

                proc_config.new_gid_time_window = 50;

                proc_config.gid_recognition_cooldown_s = 0;

                // ======================= 【新增：实时配置白名单，用于测试】 =======================
                // 在实际应用中，这个列表可以从外部实时获取 (例如，通过网络接口)
                // 这里我们硬编码一个示例，将 "G00001" 和 "G00005" 加入白名单
                proc_config.whitelist_gids = {"G00001", "G00005"};
                // ======================= 【新增结束】 =======================

                // ---- 计时 ----
                auto t1 = std::chrono::high_resolution_clock::now();
                // ======================= 【MODIFIED】 =======================
                // 创建新的输入结构体并调用修改后的 process_packet 接口
                GstClockTime timestamp;
                if (_use_fid_time) {
                    timestamp = 0; // load 模式占位
                } else {
                    timestamp = get_current_frame_ntp_timestamp(fid, fps);
                }

                ProcessInput proc_input = {
                        CAM_ID,
                        fid,
                        timestamp,
                        gpu_frame_rgb,
                        loaded_data.packet.dets,
                        proc_config
                };
                auto proc_output = processor.process_packet(proc_input);
                auto t2 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> proc_time = t2 - t1;
                std::cout << "  [proc_packet took " << proc_time.count() << " ms]" << std::endl;

                total_proc_time += proc_time.count();
                proc_count++;

                // ======================= 【MODIFIED】 =======================
                // 新增：处理返回的告警信息
                if (!proc_output.alarms.empty()) {
                    std::cout << "\n\n*** ALARM TRIGGERED! Frame: " << fid << " ***\n";
                    for (const auto &alarm: proc_output.alarms) {
                        std::cout << "  - GID: " << alarm.gid << "\n";
                        std::cout << "  - Recognition Count (n): " << alarm.n << "\n";
                        std::cout << "  - First Seen Time: " << format_ntp_timestamp(alarm.first_seen_timestamp)
                                  << "\n";
                        std::cout << "  - Last Seen Time: " << format_ntp_timestamp(alarm.last_seen_timestamp)
                                  << "\n";
                        std::cout << "  - Face Clarity: " << std::fixed << std::setprecision(1)
                                  << alarm.face_clarity_score << "/100\n"
                                  << "\n";
                    }
                }
                // ======================= 【修改结束】 =======================
                // --- 可视化 ---
                cv::Mat vis;
                cv::resize(frame, vis, vis_size);

                const auto &cam_map = proc_output.mp.count(CAM_ID) ? proc_output.mp.at(CAM_ID)
                                                                   : std::map<int, std::tuple<std::string, float, int, std::optional<AlarmGeometry>>>{};

                // ======================= 【可视化部分保持不变】 =======================
                for (const auto &det: loaded_data.packet.dets) {
                    // --- 默认值 ---
                    float score = -1.0f;
                    int n_tid = 0;
                    std::string display_status = "G:-?";
                    std::string full_gid_str = ""; // For logic that needs the full string
                    int color_id = det.id;
                    cv::Scalar color_override(-1, -1, -1); // 用于强制指定颜色的哨兵值

                    // --- 新增：获取TID的可见时长 ---
                    std::string tid_str = CAM_ID + "_" + std::to_string(det.id);
                    double duration_sec = proc_output.tid_durations_sec.count(tid_str)
                                          ? proc_output.tid_durations_sec.at(tid_str) : 0.0;

                    // --- 检查此 TID 是否有处理器返回的结果 ---
                    if (cam_map.count(det.id)) {
                        const auto &tpl = cam_map.at(det.id);
                        full_gid_str = std::get<0>(tpl);
                        score = std::get<1>(tpl);
                        n_tid = std::get<2>(tpl);
                        const auto &geom_opt = std::get<3>(tpl); // Get optional geometry

                        bool is_alarm = (full_gid_str.find("_AA") != std::string::npos ||
                                         full_gid_str.find("_AL") != std::string::npos);

                        if (is_alarm) {
                            color_override = cv::Scalar(0, 255, 255); // Yellow for alarm

                            if (geom_opt.has_value()) {
                                const auto &geom = geom_opt.value();
                                cv::Mat overlay = vis.clone();
                                double alpha = 0.3;

                                // Draw crossing zone
                                if (!geom.crossing_zone_poly.empty()) {
                                    std::vector<cv::Point> poly_scaled;
                                    for (const auto &pt: geom.crossing_zone_poly) {
                                        poly_scaled.emplace_back(pt.x * SHOW_SCALE, pt.y * SHOW_SCALE);
                                    }
                                    cv::fillConvexPoly(overlay, poly_scaled, cv::Scalar(0, 255, 255), cv::LINE_AA);
                                }

                                // Draw intersection
                                if (!geom.intersection_poly.empty()) {
                                    std::vector<cv::Point> poly_scaled;
                                    for (const auto &pt: geom.intersection_poly) {
                                        poly_scaled.emplace_back(pt.x * SHOW_SCALE, pt.y * SHOW_SCALE);
                                    }
                                    cv::fillConvexPoly(overlay, poly_scaled, cv::Scalar(0, 0, 255), cv::LINE_AA);
                                }

                                cv::addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis);

                                // Draw projection vector
                                if (geom.line_start != cv::Point2f(0, 0) && geom.line_end != cv::Point2f(0, 0)) {
                                    cv::Point2f line_center = (geom.line_start + geom.line_end) * 0.5f;
                                    cv::Point arr_start(line_center.x * SHOW_SCALE, line_center.y * SHOW_SCALE);
                                    cv::Point2f proj_end_pt = line_center + geom.projection_vector * 50.f;
                                    cv::Point arr_end(proj_end_pt.x * SHOW_SCALE, proj_end_pt.y * SHOW_SCALE);
                                    cv::arrowedLine(vis, arr_start, arr_end, cv::Scalar(255, 255, 0), 2, cv::LINE_8, 0,
                                                    0.3);
                                }
                            }
                        }

                        // --- 解析 full_gid_str 以生成用于显示的状态文本 ---
                        bool gid_found = false;
                        size_t g_pos = full_gid_str.find("_G");
                        if (g_pos != std::string::npos) {
                            std::string gid_part = full_gid_str.substr(g_pos + 1);
                            size_t next_underscore = gid_part.find('_');
                            // Strip alarm details for display
                            if (gid_part.find("_AL") != std::string::npos) {
                                gid_part = gid_part.substr(0, gid_part.find("_AL"));
                            }

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
                            display_status = (gid_found ? display_status + " " : "") + "Intrusion";
                        } else if (full_gid_str.find("_AL") != std::string::npos) {
                            display_status = (gid_found ? display_status + " " : "") + "Crossing";
                        } else if (full_gid_str.find("_-9_cool") != std::string::npos) {
                            // 如果 GID 在冷却状态，也附加状态文本
                            display_status = (gid_found ? display_status + " " : "") + "Cooldown";
                        } else if (!gid_found) {
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
                            } else if (full_gid_str.find("_-8_wait_d") != std::string::npos) {
                                display_status = "Wait Duration";
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
                                       ? color_override : cv::Scalar(0, 0, 0);
                    if (color_override[0] == -1) {
                        // 在单路视频中，为不同 GID 使用调色板以区分
                        size_t g_pos = display_status.find('G');
                        int color_idx_val = det.id;
                        if (g_pos == 0) {
                            try { color_idx_val = std::stoi(display_status.substr(1)); } catch (...) {}
                        }
                        color = colors[color_idx_val % (sizeof(colors) / sizeof(cv::Scalar))];
                    }
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

                    //【修改】底部文本: n_tid, score 和 duration
                    std::ostringstream ss_bottom;
                    ss_bottom << "n=" << n_tid << " s=" << std::fixed << std::setprecision(2) << score
                              << " d=" << std::fixed << std::setprecision(1) << duration_sec << "s";
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
                if (proc_output.mp.count(CAM_ID)) {
                    std::vector<int> sorted_tids;
                    for (const auto &pair: proc_output.mp.at(CAM_ID)) {
                        sorted_tids.push_back(pair.first);
                    }
                    std::sort(sorted_tids.begin(), sorted_tids.end());

                    for (int tid: sorted_tids) {
                        const auto &tpl = proc_output.mp.at(CAM_ID).at(tid);
                        const std::string &full_str = std::get<0>(tpl);
                        float score = std::get<1>(tpl);
                        int n_tid = std::get<2>(tpl);
                        const auto &geom_opt = std::get<3>(tpl); // Get the optional geometry

                        // --- NEW LOGIC TO MATCH PYTHON ---
                        std::string gid_to_write = full_str; // Step 1: Initialize with full string
                        std::string alarm_type = "";
                        std::string alarm_direction = "";
                        float alarm_distance = 0.0f;
                        float alarm_ratio = 0.0f;
                        float alarm_area = 0.0f;

                        size_t details_pos = gid_to_write.find("_D");
                        if (details_pos != std::string::npos && gid_to_write.find("_AL_") != std::string::npos) {
                            // Line crossing alarm with details
                            std::string base_gid_str = gid_to_write.substr(0, details_pos);
                            std::string details_str = gid_to_write.substr(details_pos);
                            gid_to_write = base_gid_str; // Step 2: Re-assign, stripping details

                            // Extract alarm sub-type (line name)
                            size_t al_pos = base_gid_str.find("_AL_");
                            if (al_pos != std::string::npos) {
                                alarm_type = "crossing_" + base_gid_str.substr(al_pos + 4);
                            } else {
                                alarm_type = "crossing";
                            }

                            // Parse details string like _D123_R0.45_A6789
                            std::replace(details_str.begin(), details_str.end(), '_', ' ');
                            std::stringstream ss(details_str);
                            std::string part;
                            while (ss >> part) {
                                try {
                                    if (part.rfind('D', 0) == 0) alarm_distance = std::stof(part.substr(1));
                                    else if (part.rfind('R', 0) == 0) alarm_ratio = std::stof(part.substr(1));
                                    else if (part.rfind('A', 0) == 0) alarm_area = std::stof(part.substr(1));
                                } catch (...) { /* ignore parse errors */ }
                            }

                            if (geom_opt.has_value()) {
                                alarm_direction = geom_opt->direction;
                            }
                        } else if (gid_to_write.find("_AA") != std::string::npos) {
                            alarm_type = "intrusion";
                            // For intrusion, gid_to_write is NOT modified, which matches python.
                        }

                        fout << fid << ',' << CAM_ID << ',' << tid << ','
                             << gid_to_write << ',' << std::fixed << std::setprecision(4) << score << ',' << n_tid
                             << ','
                             << alarm_type << ',' << alarm_direction << ',' << std::setprecision(2) << alarm_distance
                             << ','
                             << std::setprecision(4) << alarm_ratio << ',' << std::setprecision(2) << alarm_area
                             << "\n";
                    }
                }

            } catch (const std::runtime_error &e) {
                std::cout << "\nWarning: Failed to process frame " << fid << ". " << e.what() << std::endl;
                cv::Mat vis;
                cv::resize(frame, vis, vis_size);
                writer.write(vis);
            }
        } // while(true)

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
        processor.save_final_state_to_file("/mnt/nfs/state_before_shutdown.txt");
        // ======================= 【修改结束】 =======================

    } catch (const std::exception &e) {
        std::cerr << "\nAn unrecoverable error occurred: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}