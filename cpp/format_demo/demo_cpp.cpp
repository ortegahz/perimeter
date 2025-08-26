#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
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
    fout << "frame_id,cam_id,tid,gid,score,n_tid\n";
    fout.setf(std::ios::fixed);
    fout << std::setprecision(4);

    int fid = 0;
    cv::Mat frame;
    while (cap.read(frame)) {
        ++fid;
        if (fid % SKIP != 0) continue;

        auto mp = processor.process_packet(CAM_ID, fid);
        if (mp.find(CAM_ID) == mp.end()) continue;

        std::vector<int> tids;
        for (auto &kv: mp[CAM_ID]) tids.push_back(kv.first);
        std::sort(tids.begin(), tids.end());

        for (int tid: tids) {
            auto &tpl = mp[CAM_ID][tid];
            fout << fid << ',' << CAM_ID << ',' << tid << ','
                 << std::get<0>(tpl) << ',' << std::get<1>(tpl) << ',' << std::get<2>(tpl) << "\n";
        }
    }
    std::cout << "DONE -> " << OUTPUT_TXT << "\n";
    return 0;
}