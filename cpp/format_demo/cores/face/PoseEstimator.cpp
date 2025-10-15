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

PoseEstimator::EulerAngles PoseEstimator::rotationMatrixToEulerAngles(const cv::Mat &R) {
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

std::optional<PoseResult> PoseEstimator::estimate_pose(cv::Size image_size, const std::vector<cv::Point2f> &image_pts) {
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

    EulerAngles angles = rotationMatrixToEulerAngles(rot_mat);
    double yaw = angles.yaw;
    double roll = angles.roll;

    // --- New: Calculate pitch score based on keypoint ratio ---
    // Keypoint order: [left_eye, right_eye, nose_tip, left_mouth_corner, right_mouth_corner]
    cv::Point2f left_eye = image_pts[0];
    cv::Point2f right_eye = image_pts[1];
    cv::Point2f nose = image_pts[2];
    cv::Point2f left_mouth = image_pts[3];
    cv::Point2f right_mouth = image_pts[4];

    cv::Point2f eye_center = (left_eye + right_eye) * 0.5f;
    cv::Point2f mouth_center = (left_mouth + right_mouth) * 0.5f;

    double eye_to_nose = cv::norm(nose - eye_center);
    double pitch_score;
    if (eye_to_nose < 1e-6) {
        pitch_score = 1.0; // Avoid division by zero, return neutral value
    } else {
        double nose_to_mouth = cv::norm(mouth_center - nose);
        pitch_score = nose_to_mouth / eye_to_nose;
    }
    return PoseResult{yaw, pitch_score, roll};
}

