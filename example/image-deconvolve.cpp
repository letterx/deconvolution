#include <iostream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/program_options.hpp>
#include "deconvolve.hpp"
#include "regularizer.hpp"
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

    int kerSize = 21;
    double sigma = 3.0;
    deconvolution::Array<2> ker{boost::extents[kerSize][kerSize]};
    std::vector<int> kerBase{-10, -10};
    ker.reindex(kerBase);
    for (int i = ker.index_bases()[0];
            i != ker.index_bases()[0] + int(ker.shape()[0]);
            ++i) {
        for (int j = ker.index_bases()[1];
                j != ker.index_bases()[1] + int(ker.shape()[1]);
                ++j) {
            ker[i][j] = exp(-(i*i + j*j)/(2*sigma*sigma))/(2*3.14159*sigma*sigma);
        }
    }

    std::cout << "Convolving\n";
    auto blur = convolve(y, ker);

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            image.at<unsigned char>(j,i) = blur[i][j];
        }
    }

    /*
    cv::namedWindow("Display Window", CV_WINDOW_AUTOSIZE);
    cv::imshow("Display Window", image);
    cv::waitKey(0);
    */


    deconvolution::LinearSystem<2> H = 
        [&](const deconvolution::Array<2>& x) -> deconvolution::Array<2> {
            return convolve(x, ker);
        };

    constexpr int nLabels = 32;
    constexpr double labelScale = 255.0/(nLabels-1);
    constexpr double smoothMax = 100.0;
    constexpr double regularizerWeight = 100.0;
    auto R = deconvolution::GridRegularizer<2>{
        std::vector<int>{width, height}, 
        nLabels, labelScale, 
        [=](int l1, int l2)->double {
            return regularizerWeight*std::min(smoothMax, fabs(labelScale*(l1 - l2)));
        } 
    };

    std::cout << "Deconvolving\n";
    auto deblur = deconvolution::Deconvolve<2>(y, H, H, R);
    std::cout << "Done\n";

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
