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
// 我们需要在这里定义它，以便创建要传递的变量
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
const std::string FACE_DET_MODEL_PATH = "/home/nvidia/VSCodeProject/smartboxcore/models/mobilenet0.25_Final.onnx";
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
    std::string RAW_DIR = "/mnt/nfs/cache_v2";
    std::string CAM_ID = "cam1";
    int SKIP = 2;
    float SHOW_SCALE = 0.5;

    std::string MODE = "load"; // realtime or load
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
        bool _use_fid_time = (MODE == "load");
        // ======================= 【MODIFIED】 =======================
        // 修改: 使用新的构造函数实例化 FeatureProcessor
        FeatureProcessor processor(
                REID_MODEL_PATH,
                FACE_DET_MODEL_PATH,
                FACE_REC_MODEL_PATH,
                MODE,                     // 明确传递，覆盖默认值
                "dla",                    // 明确传递
                FEATURE_CACHE_JSON,       // 明确传递，覆盖默认值
                boundary_config,          // 明确传递，覆盖默认值
                _use_fid_time,
                true,                     // enable_alarm_saving
                true);                    // processing_enabled: 新增算法总开关, 设置为 false 可禁用所有处理
        // ======================= 【修改结束】 =======================

        // 注意：在realtime模式下，FeatureProcessor会创建自己的FaceAnalyzer实例。
        // 此处的face_analyzer仅用于演示，实际处理在processor内部完成。

        cv::VideoCapture cap(VIDEO_PATH, cv::CAP_FFMPEG);
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

        uint64_t fid = 0;
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
                // 新增：将 BGR 帧转换为 RGB 帧，因为模型需要 RGB 输入
                cv::cuda::GpuMat gpu_frame_rgb;
                cv::cuda::cvtColor(gpu_frame, gpu_frame_rgb, cv::COLOR_BGR2RGB);
                // 修改：统一从缓存加载基础检测信息 (dets) 和可视化信息 (face_info)
                LoadedData loaded_data = load_packet_from_cache(CAM_ID, fid, RAW_DIR);

                // --- 新增：定义本次处理的配置参数 ---
                ProcessConfig proc_config;
                proc_config.alarm_cnt_th = 2;  // 示例：将全局告警计数阈值改为2
                //proc_config.match_thr_by_cam[CAM_ID] = 0.5f; // 示例：为当前相机"cam1"设置特定的浮点数匹配阈值，会覆盖下面的灵敏度设置
                // 新增：设置相机 "cam1" 的匹配灵敏度
                // 灵敏度级别: 1 (低), 2 (中), 3 (高)。
                // 高灵敏度意味着使用较低的匹配分数阈值，更容易匹配上 (例如_ 阈值=0.4)。
                proc_config.sensitivity_by_cam[CAM_ID] = 2; // 示例: 设置为高灵敏度

                // 新增：人脸/ReID权重配置
                // 为 "cam1" 启用人脸处理，并设置权重为 70% 人脸 + 30% ReID。
                proc_config.face_switch_by_cam[CAM_ID] = true;
                proc_config.face_weight_by_cam[CAM_ID] = 0.6f;
                proc_config.reid_weight_by_cam[CAM_ID] = 0.4f;
                //【修改】为 "cam1" 设置3秒的徘徊时间
                proc_config.alarmDuration_ms_by_cam[CAM_ID] = 0;
                // 示例：若有另一路 "cam2"，可禁用人脸处理 (权重将自动变为 0% 人脸 + 100% ReID)。
                // proc_config.face_switch_by_cam["cam2"] = false;
                // 示例：可以为 "cam2" 设置不同的徘徊时间，或者不设置（默认为0，即禁用）
                // proc_config.alarmDuration_ms_by_cam["cam2"] = 5000; // 5秒

                // ---- 计时 ----
                auto t1 = std::chrono::high_resolution_clock::now();
                // ======================= 【MODIFIED】 =======================
                // 创建新的输入结构体并调用修改后的 process_packet 接口
                GstClockTime timestamp;
                if (_use_fid_time) {
                    // 在 'load' 模式下，时间戳由内部基于 fid 生成，此处传入的值会被忽略。
                    // 传入 0 作为占位符。
                    timestamp = 0;
                } else {
                    // 在 'realtime' 模式下，获取原始的 GstClockTime 时间戳。
                    timestamp = get_current_frame_ntp_timestamp(fid, fps);
                }

                ProcessInput proc_input = {
                        CAM_ID,
                        fid,
                        timestamp, // 直接传入原始的 GstClockTime
                        gpu_frame_rgb,
                        loaded_data.packet.dets,
                        proc_config
                };
                auto proc_output = processor.process_packet(proc_input);
                // ======================= 【修改结束】 =======================
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
                        // 新增：打印 GID 首次出现的时间戳 (已是GstClockTime格式)
                        std::cout << "  - First Seen Time: " << format_ntp_timestamp(alarm.first_seen_timestamp) << "\n";

                        std::string base_path = "/mnt/nfs/alarm_" + alarm.gid + "_fid" + std::to_string(fid);

                        // 保存告警帧、最新的行人/人脸图块以供查验
                        cv::imwrite(base_path + "_frame.jpg", frame);
                        // 新增：由于 patch 是 RGB 格式，保存前需转换为 BGR
                        if (!alarm.latest_body_patch.empty()) {
                            cv::Mat bgr_patch;
                            cv::cvtColor(alarm.latest_body_patch, bgr_patch, cv::COLOR_RGB2BGR);
                            cv::imwrite(base_path + "_body_patch.jpg", bgr_patch);
                        }
                        if (!alarm.latest_face_patch.empty()) {
                            cv::Mat bgr_patch;
                            cv::cvtColor(alarm.latest_face_patch, bgr_patch, cv::COLOR_RGB2BGR);
                            cv::imwrite(base_path + "_face_patch.jpg", bgr_patch);
                        }

                        // 在告警帧上绘制边界框并保存
                        cv::Mat alarm_vis = frame.clone();
                        cv::rectangle(alarm_vis, alarm.person_bbox, cv::Scalar(0, 0, 255), 3); // 红色粗框标出行人
                        if (alarm.face_bbox.area() > 0)
                            cv::rectangle(alarm_vis, alarm.face_bbox, cv::Scalar(0, 255, 255), 2); // 黄色框标出人脸
                        cv::imwrite(base_path + "_annotated.jpg", alarm_vis);
                    }
                }
                // ======================= 【修改结束】 =======================
                // --- 可视化 ---
                cv::Mat vis;
                cv::resize(frame, vis, vis_size);

                const auto &cam_map = proc_output.mp.count(CAM_ID) ? proc_output.mp.at(CAM_ID)
                                                                   : std::map<int, std::tuple<std::string, float, int>>{};

                // ======================= 【可视化部分保持不变】 =======================
                for (const auto &det: loaded_data.packet.dets) {
                    // --- 默认值 ---
                    float score = -1.0f;
                    int n_tid = 0;
                    std::string display_status = "G:-?";
                    int color_id = det.id;
                    cv::Scalar color_override(-1, -1, -1); // 用于强制指定颜色的哨兵值

                    // --- 新增：获取TID的可见时长 ---
                    std::string tid_str = CAM_ID + "_" + std::to_string(det.id);
                    double duration_sec = proc_output.tid_durations_sec.count(tid_str)
                                          ? proc_output.tid_durations_sec.at(tid_str) : 0.0;

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