#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/program_options.hpp>

std::pair<double, double> psnrMse(const cv::Mat& image, const cv::Mat& orig) {
    assert(image.rows == orig.rows && image.cols == orig.cols);
    assert(image.type() == CV_8U);

    cv::Mat tmp;
    cv::absdiff(image, orig, tmp);
    tmp.convertTo(tmp, CV_32F);
    tmp = tmp.mul(tmp);

    double sumSquare = sum(tmp)[0];

    double mse = static_cast<double>(sumSquare)/static_cast<double>(image.rows*image.cols);
    double psnr = 20*log10(255) - 10*log10(mse);

    return std::make_pair(psnr, mse);
}

int main(int argc, char **argv) {
    namespace po = boost::program_options;
    // Variables set by program options
    std::string fname1;
    std::string fname2;

    po::options_description options_desc("Deconvolve arguments");
    options_desc.add_options()
        ("help", "Display this help message")
        ("image", po::value<std::string>(&fname1)->required(), "image")
        ("orig", po::value<std::string>(&fname2)->required(), "original-image")
    ;

    po::positional_options_description popts_desc;
    popts_desc.add("image", 1);
    popts_desc.add("orig", 2);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
            options(options_desc).positional(popts_desc).run(), vm);

    if (vm.count("help") != 0) {
        std::cout << "Usage: image-deconvolve [options] image original-image\n";
        std::cout << options_desc;
        exit(0);
    }
    try {
        po::notify(vm);
    } catch (std::exception& e) {
        std::cout << "Parsing error: " << e.what() << "\n";
        std::cout << "Usage: image-deconvolve [options] basename\n";
        std::cout << options_desc;
        exit(-1);
    }

    cv::Mat image = cv::imread(fname1.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat orig = cv::imread(fname2.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

    auto p = psnrMse(image, orig);

    std::cout << "PSNR: " << p.first << "\n";
    std::cout << "MSE:  " << p.second << "\n";
    return 0;
}
