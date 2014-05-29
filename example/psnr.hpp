#ifndef _PSNR_HPP_
#define _PSNR_HPP_

#include <opencv2/core/core.hpp>

std::pair<double, double> psnrMse(const cv::Mat& image, const cv::Mat& orig, double bias = 0.0);

#endif
