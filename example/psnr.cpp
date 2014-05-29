#include "psnr.hpp"

std::pair<double, double> psnrMse(const cv::Mat& image, const cv::Mat& orig, double bias) {
    assert(image.rows == orig.rows && image.cols == orig.cols);
    assert(image.type() == CV_8U);

    cv::Mat tmp;
    cv::absdiff(image + bias, orig, tmp);
    tmp.convertTo(tmp, CV_32F);
    tmp = tmp.mul(tmp);

    double sumSquare = sum(tmp)[0];

    double mse = static_cast<double>(sumSquare)/static_cast<double>(image.rows*image.cols);
    double psnr = 20*log10(255) - 10*log10(mse);

    return std::make_pair(psnr, mse);
}

