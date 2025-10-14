#include "PoseEstimator.hpp"

#include <opencv2/calib3d.hpp> // For solvePnP and Rodrigues
#include <cmath>

// 3D model points corresponding to ArcFace 5-point landmarks:
// [left_eye, right_eye, nose_tip, left_mouth_corner, right_mouth_corner]
const std::vector<cv::Point3f> PoseEstimator::OBJECT_POINTS_3D = {
        {-30.0f, 40.0f,  0.0f},   // Left eye
        {30.0f,  40.0f,  0.0f},    // Right eye
        {0.0f,   20.0f,  30.0f},   // Nose tip
        {-25.0f, -20.0f, 0.0f},  // Left mouth corner
        {25.0f,  -20.0f, 0.0f}    // Right mouth corner
};

PoseAngles PoseEstimator::rotationMatrixToEulerAngles(const cv::Mat &R) {
    double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
    bool singular = sy < 1e-6;

    double x, y, z;
    if (!singular) {
        x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2)); // Pitch
        y = std::atan2(-R.at<double>(2, 0), sy);                // Yaw
        z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0)); // Roll
    } else {
        x = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }

    // Return pitch, yaw, roll in degrees
    return {x * 180.0 / CV_PI, y * 180.0 / CV_PI, z * 180.0 / CV_PI};
}

std::optional<PoseAngles> PoseEstimator::estimate_pose(cv::Size image_size, const std::vector<cv::Point2f> &image_pts) {
    // This estimator is specifically for 5-point landmarks from an ArcFace-style model.
    if (image_pts.size() != 5) {
        return std::nullopt;
    }

    double focal_length = image_size.width;
    cv::Point2d center(image_size.width / 2.0, image_size.height / 2.0);

    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x,
            0, focal_length, center.y,
            0, 0, 1);
    // Assume no lens distortion
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

    cv::Mat rvec, tvec;

    // Use EPNP for this task. The original code in the test app had branching logic
    // that is not applicable to a fixed 5-point system.
    bool success = cv::solvePnP(OBJECT_POINTS_3D, image_pts, camera_matrix, dist_coeffs, rvec, tvec, false,
                                cv::SOLVEPNP_EPNP);

    if (!success) {
        return std::nullopt;
    }

    // Refine with Levenberg-Marquardt for better accuracy
    cv::solvePnPRefineLM(OBJECT_POINTS_3D, image_pts, camera_matrix, dist_coeffs, rvec, tvec);

    cv::Mat rot_mat;
    cv::Rodrigues(rvec, rot_mat);

    // The original test app code had a bug that swapped pitch and yaw upon return.
    // This implementation is correct: it returns a PoseAngles struct where
    // .pitch is pitch, .yaw is yaw, and .roll is roll.
    return rotationMatrixToEulerAngles(rot_mat);
}

