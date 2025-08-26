#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>        // **** NEW ****
#include <algorithm>
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
        std::cerr << "Cannot open video!\n";
        return -1;
    }
    std::ofstream fout(OUTPUT_TXT);
    /* 与 python 保持相同精度 & 表头 */
    fout << "frame_id,cam_id,tid,gid,score,n_tid\n";
    fout << std::fixed << std::setprecision(4);

    int fid = 0;
    cv::Mat frame;
    while (cap.read(frame)) {
        ++fid;
        if (fid % SKIP != 0) continue;

        auto realtime_map = processor.process_packet(CAM_ID, fid);
        if (realtime_map.find(CAM_ID) == realtime_map.end()) continue;

        /* 输出前对 tid 做升序排序，保持一致 */
        std::vector<int> tids;
        for (auto &kv: realtime_map[CAM_ID]) tids.push_back(kv.first);
        std::sort(tids.begin(), tids.end());

        for (int tid: tids) {
            auto &tpl = realtime_map[CAM_ID][tid];
            const std::string &gid = std::get<0>(tpl);
            float score = std::get<1>(tpl);
            int n_tid = std::get<2>(tpl);
            fout << fid << "," << CAM_ID << "," << tid << ","
                 << gid << "," << score << "," << n_tid << "\n";
        }
    }
    return 0;
}