#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>

#include "FaceAnalyzer_dla.hpp"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

// 辅助函数
double cosine_similarity(const cv::Mat &vec1, const cv::Mat &vec2);

std::pair<std::vector<cv::Mat>, std::vector<bool>>
remove_outliers(const std::vector<cv::Mat> &embeddings, double thresh);

int main(int argc, char *argv[]) {
    // [MODIFICATION] All original parameters restored
    std::string folder = "/mnt/nfs/perimeter_v1/G00002/faces/";
    std::string provider = "DLA"; // 使用TensorRT DLA后端
    int det_size = 640;
    double outlier_thresh = 1.2;
    std::string output_json = "/mnt/nfs/embeddings.json";
    std::string output_aligned_dir = "/mnt/nfs/aligned_faces_cpp/";
    std::string output_detection_txt = "/mnt/nfs/detections_cpp_arm.txt";
    std::string output_embedding_txt = "/mnt/nfs/embeddings_cpp_arm.txt";
    std::string det_model = "/mnt/nfs/det_10g_simplified.onnx";
    std::string rec_model = "/mnt/nfs/w600k_r50_simplified.onnx";

    std::cout << "[INFO] folder: " << folder << std::endl;

    try {
        // [MODIFICATION] Initializing with both models again
        FaceAnalyzer face_app(det_model, rec_model);

        // 准备模型
        face_app.prepare(provider, 0.5f, cv::Size(det_size, det_size));
        std::cout << "[INFO] Face analyzer ready with DLA backend" << std::endl;

        if (!fs::exists(folder)) {
            std::cerr << "[ERROR] 文件夹不存在: " << folder << std::endl;
            return -1;
        }

        // [MODIFICATION] Preparation for aligned faces and embedding files restored
        if (!output_aligned_dir.empty()) {
            fs::create_directories(output_aligned_dir);
            std::cout << "[INFO] 对齐后的人脸将保存到: " << output_aligned_dir << std::endl;
        }

        std::ofstream detection_txt_file;
        if (!output_detection_txt.empty()) {
            detection_txt_file.open(output_detection_txt, std::ios::out | std::ios::trunc);
            if (detection_txt_file.is_open()) {
                detection_txt_file
                        << "filename,bbox_x,bbox_y,bbox_width,bbox_height,score,kps0_x,kps0_y,kps1_x,kps1_y,kps2_x,kps2_y,kps3_x,kps3_y,kps4_x,kps4_y\n";
                std::cout << "[INFO] 人脸检测结果将保存到: " << output_detection_txt << std::endl;
            }
        }

        std::ofstream embedding_txt_file;
        if (!output_embedding_txt.empty()) {
            embedding_txt_file.open(output_embedding_txt, std::ios::out | std::ios::trunc);
            if (embedding_txt_file.is_open()) {
                std::cout << "[INFO] 人脸特征向量将保存到: " << output_embedding_txt << std::endl;
            }
        }

        // [MODIFICATION] Vectors for storing embeddings restored
        std::vector<cv::Mat> embeddings;
        std::vector<std::string> paths_list;

        std::vector<fs::path> image_paths;
        for (const auto &entry: fs::directory_iterator(folder)) {
            image_paths.push_back(entry.path());
        }
        std::sort(image_paths.begin(), image_paths.end());

        for (const auto &entry_path: image_paths) {
            std::string ext = entry_path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext == ".jpg" || ext == ".png" || ext == ".jpeg") {
                cv::Mat img = cv::imread(entry_path.string());
                if (img.empty()) {
                    std::cerr << "[WARNING] 读取失败: " << entry_path << std::endl;
                    continue;
                }
                
                int _n = 1000;
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < _n; i++) {
                    auto faces = face_app.get(img);
                }
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                std::cout << "总耗时: " << elapsed.count() << " ms" << std::endl;
                std::cout << "平均耗时: " << elapsed.count() / _n << " ms" << std::endl;

                // 如果测试后仍需处理，可以复用最后一次的结果，但更好的方式是单独获取
                auto faces = face_app.get(img);
                if (faces.empty()) {
                    std::cout << "[WARNING] 未检测到人脸: " << entry_path << std::endl;
                    continue;
                }

                int face_idx = 0;
                fs::path p(entry_path);

                for (auto &f: faces) {
                    // [MODIFICATION] Logic for saving aligned faces restored
                    if (!output_aligned_dir.empty() && !f.aligned_face.empty()) {
                        cv::Mat aligned_to_save = f.aligned_face.clone();
                        std::vector<cv::Point2f> arcface_template = {
                                {38.2946f, 51.6963f},
                                {73.5318f, 51.5014f},
                                {56.0252f, 71.7366f},
                                {41.5493f, 92.3655f},
                                {70.7299f, 92.2041f}
                        };
                        for (const auto &pt: arcface_template) {
                            cv::circle(aligned_to_save, pt, 2, cv::Scalar(0, 255, 0), -1);
                        }
                        std::string stem = p.stem().string();
                        std::string save_path_str = (fs::path(output_aligned_dir) /
                                                     (stem + "_face" + std::to_string(face_idx) + ".jpg")).string();
                        cv::imwrite(save_path_str, aligned_to_save);
                    }

                    // 保存检测结果
                    if (detection_txt_file.is_open()) {
                        detection_txt_file << std::fixed << std::setprecision(4)
                                           << entry_path.filename().string() << ","
                                           << f.bbox.x << "," << f.bbox.y << ","
                                           << f.bbox.width << "," << f.bbox.height << ","
                                           << std::setprecision(6) << f.det_score << std::setprecision(4);
                        for (const auto &kps_pt: f.kps) {
                            detection_txt_file << "," << kps_pt.x << "," << kps_pt.y;
                        }
                        detection_txt_file << "\n";
                    }

                    // [MODIFICATION] Logic for saving feature vectors restored
                    if (!f.embedding.empty()) {
                        cv::Mat emb;
                        cv::normalize(f.embedding, emb, 1.0, 0.0, cv::NORM_L2);
                        embeddings.push_back(emb);
                        paths_list.push_back(entry_path.filename().string());

                        if (embedding_txt_file.is_open()) {
                            std::string face_id = p.stem().string() + "_face" + std::to_string(face_idx);
                            embedding_txt_file << face_id;
                            const float *emb_data = emb.ptr<float>(0);
                            for (int j = 0; j < emb.cols; ++j) {
                                embedding_txt_file << "," << std::fixed << std::setprecision(8) << emb_data[j];
                            }
                            embedding_txt_file << "\n";
                        }
                    }
                    face_idx++;
                }
            }
        }

        if (detection_txt_file.is_open()) {
            detection_txt_file.close();
        }
        if (embedding_txt_file.is_open()) {
            embedding_txt_file.close();
        }

        // [MODIFICATION] Post-processing logic for embeddings restored
        if (embeddings.empty()) {
            if (!output_detection_txt.empty()) {
                std::cout << "[INFO] 仅执行了人脸检测/特征提取，结果已保存。" << std::endl;
                return 0;
            }
            throw std::runtime_error("未获取到任何人脸特征");
        }

        json output_data;
        for (size_t i = 0; i < paths_list.size(); ++i) {
            std::vector<float> emb_vec(embeddings[i].begin<float>(), embeddings[i].end<float>());
            output_data[paths_list[i]] = emb_vec;
        }
        std::ofstream out_file(output_json);
        out_file << std::setw(4) << output_data << std::endl;
        std::cout << "[INFO] 所有提取到的人脸特征已保存到: " << output_json << std::endl;

        std::cout << "[INFO] 共获取 " << embeddings.size() << " 张人脸特征" << std::endl;

        auto [clean_embeddings, keep_mask] = remove_outliers(embeddings, outlier_thresh);

        std::vector<std::string> kept_paths, removed_paths;
        for (size_t i = 0; i < paths_list.size(); ++i) {
            if (keep_mask[i]) {
                kept_paths.push_back(paths_list[i]);
            } else {
                removed_paths.push_back(paths_list[i]);
            }
        }

        std::cout << "[INFO] 去除异常值后剩余 " << clean_embeddings.size()
                  << " 张 (剔除 " << (embeddings.size() - clean_embeddings.size()) << ")" << std::endl;

        if (!removed_paths.empty()) {
            std::cout << "[INFO] 被剔除的图片有：" << std::endl;
            for (const auto &name: removed_paths) {
                std::cout << " - " << name << std::endl;
            }
        }

        if (clean_embeddings.empty()) {
            throw std::runtime_error("去除异常值后无剩余特征。");
        }

        cv::Mat mean_vec = cv::Mat::zeros(1, clean_embeddings[0].cols, clean_embeddings[0].type());
        for (const auto &emb: clean_embeddings) {
            mean_vec += emb;
        }
        mean_vec /= static_cast<double>(clean_embeddings.size());
        cv::normalize(mean_vec, mean_vec, 1.0, 0.0, cv::NORM_L2);

        std::cout << "[INFO] 与均值向量相似度:" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        for (size_t i = 0; i < clean_embeddings.size(); ++i) {
            double sim = cosine_similarity(clean_embeddings[i], mean_vec);
            std::cout << kept_paths[i] << "  相似度: " << sim << std::endl;
        }

    } catch (const std::exception &e) {
        std::cerr << "[FATAL ERROR] " << e.what() << '\n';
        return -1;
    }

    std::cout << "[INFO] 计算完成" << std::endl;
    return 0;
}

// [MODIFICATION] Helper functions implementation restored
double cosine_similarity(const cv::Mat &vec1, const cv::Mat &vec2) {
    double dot = vec1.dot(vec2);
    double norm1 = cv::norm(vec1);
    double norm2 = cv::norm(vec2);
    if (norm1 == 0 || norm2 == 0) { return 0.0; }
    return dot / (norm1 * norm2);
}

std::pair<std::vector<cv::Mat>, std::vector<bool>> remove_outliers(
        const std::vector<cv::Mat> &embeddings, double thresh) {
    if (embeddings.empty()) {
        return {{},
                {}};
    }

    cv::Mat mean_vec = cv::Mat::zeros(1, embeddings[0].cols, embeddings[0].type());
    for (const auto &emb: embeddings) {
        mean_vec += emb;
    }
    mean_vec /= static_cast<double>(embeddings.size());

    std::vector<double> dists;
    dists.reserve(embeddings.size());
    for (const auto &emb: embeddings) {
        dists.push_back(cv::norm(emb - mean_vec));
    }

    double dist_sum = std::accumulate(dists.begin(), dists.end(), 0.0);
    double dist_mean = dist_sum / dists.size();
    double sq_sum = std::inner_product(dists.begin(), dists.end(), dists.begin(), 0.0);
    double dist_std = std::sqrt(sq_sum / dists.size() - dist_mean * dist_mean);

    std::vector<cv::Mat> clean_embeddings;
    std::vector<bool> keep_mask(embeddings.size(), false);
    clean_embeddings.reserve(embeddings.size());

    for (size_t i = 0; i < embeddings.size(); ++i) {
        double z_score = (dists[i] - dist_mean) / (dist_std + 1e-8);
        if (std::abs(z_score) < thresh) {
            clean_embeddings.push_back(embeddings[i]);
            keep_mask[i] = true;
        }
    }
    return {clean_embeddings, keep_mask};
}