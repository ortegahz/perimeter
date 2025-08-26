#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include "feature_processor.h"

int main() {
    std::string VIDEO_PATH = "/home/manu/tmp/64.mp4";
    std::string CACHE_JSON = "/home/manu/tmp/features_cache.json";
    std::string OUTPUT_TXT = "/home/manu/tmp/output_result.txt";
    std::string CAM_ID = "cam1";
    int SKIP = 50;

    FeatureProcessor processor(CACHE_JSON);

    cv::VideoCapture cap(VIDEO_PATH);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video!" << std::endl;
        return -1;
    }

    std::ofstream fout(OUTPUT_TXT);
    fout << "frame_id,cam_id,tid,gid,score,n_tid\n";

    int fid = 0;
    cv::Mat frame;
    while (true) {
        if (!cap.read(frame)) break;
        fid++;
        if (fid % SKIP != 0) continue;

        auto realtime_map = processor.process_packet(CAM_ID, fid);
        if (realtime_map.find(CAM_ID) != realtime_map.end()) {
            auto &cam_map = realtime_map[CAM_ID];
            std::vector<int> tids;
            for (auto &kv: cam_map) tids.push_back(kv.first);
            std::sort(tids.begin(), tids.end());
            for (int tid: tids) {
                auto &[gid, score, n_tid] = cam_map[tid];
                fout << fid << "," << CAM_ID << "," << tid << "," << gid << "," << score << "," << n_tid << "\n";
            }
        }
    }
    return 0;
}