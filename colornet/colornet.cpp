#include "net.h"
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "Sig17Slice.h"
#include "colornet.h"

// load_model() loads a model params and model bin
void* load_model(const char* model_path, const char* param_path) {
    // allocate memory for a new ncnn::Net instance
    auto* net = new ncnn::Net();

    // use vulkan compute
    net->opt.use_vulkan_compute = true;
    net->register_custom_layer("Sig17Slice", Sig17Slice_layer_creator);
    if (net->load_param(param_path)) {
        std::cerr << "Failed to load model params" << std::endl;
        delete net; // deallocate on failure.
        return nullptr; // return a nullptr on failure.
    }

    if (net->load_model(model_path)) {
        std::cerr << "Failed to load model" << std::endl;
        delete net;
        return nullptr;
    }

    return net;
}

// unload_model() deallocates the memory occupied by the loaded model
void unload_model(void* net_ptr) {
    // deallocate memory
    delete static_cast<ncnn::Net*>(net_ptr);
}

// infer() runs an inference on the model
Buffer* infer(Buffer* input_buffer, void* net_ptr, const char* format) {
    if (net_ptr == nullptr) {
        std::cerr << "Please load model before inference" << std::endl;
        return {};
    }

    if (format == nullptr) {
        std::cerr << "format string is required and must be one of png, jpg" << std::endl;
        return {};
    }

    // cast the void* net_ptr address to ncnn::Net* type
    auto* net = static_cast<ncnn::Net*>(net_ptr);

    // convert Buffer* object to a std::vector<unsigned char> type object
    const std::vector<unsigned char> data(input_buffer->data, input_buffer->data + input_buffer->size);

    // Decode the image from the byte array
    const cv::Mat input_mat = cv::imdecode(data, cv::IMREAD_COLOR);
    if (input_mat.empty()) {
        std::cerr << "Failed to decode image!" << std::endl;
        return {};
    }

    // Fixed input size for the pretrained network
    constexpr int W_in = 256;
    constexpr int H_in = 256;

    cv::Mat Base_img, lab, L, input_img;
    Base_img = input_mat.clone();

    // Normalize levels
    Base_img.convertTo(Base_img, CV_32F, 1.0 / 255);

    // Convert BGR to LAB color space format
    cv::cvtColor(Base_img, lab, cv::COLOR_BGR2Lab);
    cv::extractChannel(lab, L, 0); // Extract L channel
    cv::resize(L, input_img, cv::Size(W_in, H_in)); // Resize to input shape

    // Convert to ncnn::Mat
    ncnn::Mat in_LAB_L(input_img.cols, input_img.rows, 1, (void*)input_img.data);
    in_LAB_L = in_LAB_L.clone();

    ncnn::Extractor ex = net->create_extractor();
    ex.input("input", in_LAB_L); // Set input

    ncnn::Mat out;
    ex.extract("out_ab", out); // Inference network

    // Check if output is valid
    if (out.empty()) {
        std::cerr << "Inference output is empty!" << std::endl;
        return {};
    }

    // Create LAB material
    cv::Mat colored_LAB(out.h, out.w, CV_32FC2);
    memcpy(colored_LAB.data, out.data, out.w * out.h * 2 * sizeof(float));

    std::vector<cv::Mat> lab_channels(3);
    lab_channels[0] = L;                                                                            // Use the L channel from the input
    lab_channels[1] = cv::Mat(out.h, out.w, CV_32F, out.data);                                      // 'a' channel
    lab_channels[2] = cv::Mat(out.h, out.w, CV_32F, static_cast<float*>(out.data) + out.w * out.h); // 'b' channel

    // Resize a and b channels to match the original image size
    cv::resize(lab_channels[1], lab_channels[1], Base_img.size());
    cv::resize(lab_channels[2], lab_channels[2], Base_img.size());

    // Merge the LAB channels back
    cv::merge(lab_channels, lab);
    cv::cvtColor(lab, Base_img, cv::COLOR_Lab2BGR);
    Base_img.convertTo(Base_img, CV_8UC3, 255); // Normalize values to 0->255

    // Check if the Base_img is empty before encoding
    if (Base_img.empty()) {
        std::cerr << "Base_img is empty before encoding!" << std::endl;
        return {};
    }


    // encode the cv::Mat to byte vector
    std::vector<unsigned char> buffer;

    // Prefix format with a dot
    std::string dot_prefixed_format = "." + std::string(format);

    if (!cv::imencode(dot_prefixed_format, Base_img, buffer)) {
        std::cerr << "Failed to encode image! Format: " << format << std::endl;
        return {};
    }

    // Check if buffer is empty after encoding
    if (buffer.empty()) {
        std::cerr << "Encoded buffer is empty!" << std::endl;
        return {};
    }

    // convert the byte vector to pointer to a Buffer* object
    auto* output = new Buffer();
    output->data = static_cast<unsigned char*>(malloc(buffer.size()));
    if (!output->data) {
        std::cerr << "Failed to allocate memory for output buffer" << std::endl;
        delete output; // Clean up the allocated Buffer structure
        return nullptr; // Return nullptr on allocation failure
    }

    // cast const unsigned char* to void*
    memcpy(const_cast<void*>(static_cast<const void*>(output->data)), buffer.data(), buffer.size());
    output->size = buffer.size();
    return output;
}