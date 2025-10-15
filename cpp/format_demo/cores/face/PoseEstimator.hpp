#ifndef POSE_ESTIMATOR_HPP
#define POSE_ESTIMATOR_HPP

#include <vector>
#include <optional>
#include <opencv2/core.hpp>

/**
 * @brief Structure to hold calculated pose results. Yaw and Roll are in degrees.
 */
struct PoseResult {
    double yaw, pitch_score, roll;
};

/**
 * @brief A class to encapsulate head pose estimation logic.
 */
class PoseEstimator {
public:
    /**
     * @brief Estimates head pose using solvePnP from 5 facial keypoints.
     * @param image_size The size of the input image.
     * @param image_pts A vector of 5 2D keypoints (from ArcFace model).
     * @return An optional containing PoseResult. Contains yaw, roll (in degrees), and a calculated pitch_score. Returns std::nullopt on failure.
     */
    static std::optional<PoseResult> estimate_pose(cv::Size image_size, const std::vector<cv::Point2f> &image_pts);

private:
    struct EulerAngles {
        double pitch, yaw, roll;
    };

    static const std::vector<cv::Point3f> OBJECT_POINTS_3D;
    static EulerAngles rotationMatrixToEulerAngles(const cv::Mat &R);
};

#endif // POSE_ESTIMATOR_HPP
