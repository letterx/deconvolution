#include <iostream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/program_options.hpp>
#include "deconvolve.hpp"
#include "convolve.hpp"

int main(int argc, char **argv) {
    namespace po = boost::program_options;
    // Variables set by program options
    std::string basename;
    std::string infilename;
    std::string outfilename;

    po::options_description options_desc("Deconvolve arguments");
    options_desc.add_options()
        ("help", "Display this help message")
        ("image", po::value<std::string>(&basename)->required(), "Name of image (without extension)")
    ;

    po::positional_options_description popts_desc;
    popts_desc.add("image", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
            options(options_desc).positional(popts_desc).run(), vm);

    if (vm.count("help") != 0) {
        std::cout << "Usage: image-deconvolve [options] basename\n";
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
    infilename = basename + ".pgm";
    outfilename = basename + "-out.pgm";

    cv::Mat image = cv::imread(infilename.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    if (!image.data) {
        std::cout << "Could not load image: " << infilename << "\n";
        exit(-1);
    }

    auto width = image.cols;
    auto height = image.rows;
    assert(image.type() == CV_8UC1);

    deconvolution::Array<2> y{boost::extents[width][height]};

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            y[i][j] = image.at<unsigned char>(j,i);
        }
    }

    int kerSize = 11;
    deconvolution::Array<2> ker{boost::extents[kerSize][kerSize]};
    for (int i = 0; i < kerSize; ++i) {
        ker[i][i] += 0.5/kerSize;
        ker[i][kerSize-i-1] += 0.5/kerSize;
    }
    std::vector<int> kerBase{-5, -5};
    ker.reindex(kerBase);

    auto blur = convolve(y, ker);

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            image.at<unsigned char>(j,i) = blur[i][j];
        }
    }

    cv::namedWindow("Display Window", CV_WINDOW_AUTOSIZE);
    cv::imshow("Display Window", image);
    cv::waitKey(0);


    deconvolution::LinearSystem<2> H = 
        [&](const deconvolution::Array<2>& x) -> deconvolution::Array<2> {
            return convolve(x, ker);
        };

    auto deblur = deconvolution::Deconvolve<2>(y, H, H, deconvolution::Regularizer<2>{});

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            image.at<unsigned char>(j,i) = deblur[i][j];
        }
    }

    cv::namedWindow("Display Window", CV_WINDOW_AUTOSIZE);
    cv::imshow("Display Window", image);
    cv::waitKey(0);

    auto reblur = convolve(deblur, ker);

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            image.at<unsigned char>(j,i) = reblur[i][j];
        }
    }

    cv::namedWindow("Display Window", CV_WINDOW_AUTOSIZE);
    cv::imshow("Display Window", image);
    cv::waitKey(0);

    cv::imwrite(outfilename.c_str(), image); 

    return 0;
}
