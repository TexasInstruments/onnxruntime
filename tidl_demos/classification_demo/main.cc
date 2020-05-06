#include <getopt.h>
#include <iostream>
#include <cstdarg>
#include <cstdio>
#include <fstream>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/tidl/tidl_provider_factory.h>
#include <onnxruntime/core/providers/dnnl/dnnl_provider_factory.h>

#include "validator.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

bool SetupInput(std::string input_path, cv::Mat& input_image)
{
    // Read image input
    input_image = cv::imread(input_path, CV_LOAD_IMAGE_COLOR);
    if(! input_image.data )                              // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl;
        return false;
    }
    std::cout << "Image input: " << input_path.c_str() << std::endl;
    return true;
}

/*
 * Retrieve frame, resize, and record in NCHW format
 */
void CollectFrames(std::vector<uint8_t> &output,
                   cv::Mat &in_image,
                   int width, int height, int channels)
{
    cv::Mat image;
    cv::resize(in_image, image, cv::Size(width, height));
    cv::Mat *spl = new cv::Mat[channels];
    split(image,spl);
    
    // Read the frame in NCHW format
    output.resize(height * width * channels);
    int idx = 0;
    for(int c = 0; c < channels; c++)
    {
        const unsigned char* data = image.ptr();
        for(int x = 0; x < width; x++)
        {
            for(int y = 0; y < height; y++)
            {
                output[idx++] =
                    (uint8_t)data[(channels) * (y + x*width) + (channels - 1) - c];
            }
        }
    }
}

int main(int argc, char* argv[])
{
    std::string model_path = "";
    std::string image_path = "";
    std::string labels_path = "";
    
    int tidl_flag = 0;
    int index;
    int c;

    opterr = 0;

    const char* help_str =
        "Usage: classification_demo <image_path> <model_path> <labels_path> [-t] [-h]\n"
        "Options:\n"
        "    image_path\tPath to the input image to classify\n"
        "    model_path\tPath to the ONNX model\n"
        "    labels_path\tPath to the labels txt file\n"
        "    -t\t\tUse the TIDL execution provider (default DNNL)\n"
        "    -h\t\tDisplay this help text"
        "\n";

    while ((c = getopt (argc, argv, "toh")) != -1)
        switch (c)
        {
        case 't':
            tidl_flag = 1;
            break;
        case 'h':
            fprintf (stdout, help_str, optopt);
            return 0;
        case '?':
            if (isprint (optopt))
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf (stderr,
                         "Unknown option character `\\x%x'.\n",
                         optopt);
            return 1;
        default:
            abort ();
        }

    if ((argc - optind) < 3) {
        fprintf (stderr, help_str, optopt);
        return 1;
    }

    std::cout << argc - optind << std::endl;

    image_path = std::string(argv[optind]);
    model_path = std::string(argv[optind+1]);
    labels_path = std::string(argv[optind+2]);

    std::cout << image_path << std::endl;
    std::cout << model_path << std::endl;
    std::cout << labels_path << std::endl;

    for (index = optind + 2; index < argc; index++)
    {
        printf ("!!! Ignoring argument %s\n", argv[index]);
    }

    
    OrtStatus *status;
    
    // Initialize  enviroment, maintains thread pools and state info
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    
    // Initialize session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    c_api_tidl_options * options = (c_api_tidl_options *)malloc(sizeof(c_api_tidl_options));
    strcpy(options->import, "no");
    options->debug_level = 0;
    options->tidl_tensor_bits = 8;
    strcpy(options->tidl_tools_path, "../../../tidl/c7x-mma-tidl/tidl_tools/");
    strcpy(options->artifacts_folder, "../../../onnxrt-artifacts/");

    if (tidl_flag)
    {
        status = OrtSessionOptionsAppendExecutionProvider_Tidl(session_options, options);
    } else
    {
        status = OrtSessionOptionsAppendExecutionProvider_Dnnl(session_options, 1);
    }
    
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Create Validator
    cv::Mat input_image;
    std::vector<uint8_t> image_data;

    // Process the input image
    SetupInput(image_path, input_image);
    CollectFrames(image_data, input_image, 224, 224, 3);

    // Do the thing
    Validator validator(env, model_path, labels_path, session_options, image_data);

    printf("Done!\n");
    return 0;
}

