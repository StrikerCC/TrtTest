//
// Created by cheng.chen05 on 6/21/2024.
//
#include <iostream>
#include <sstream>
#include <fstream>
#include <numeric>

//#include "cuda_utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "NvInfer.h"
#include "NvOnnxParser.h"

using namespace nvinfer1;
using namespace nvonnxparser;

int kInputH = 384;
int kInputW = 640;

static uint8_t* img_buffer_host = nullptr;
static uint8_t* img_buffer_device = nullptr;


int mask_2_img(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& out) {
    auto img2_copy = cv::Mat();

    return 0;
}

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

auto GetMemorySize = [](const nvinfer1::Dims &dims,
                        const int32_t elem_size) -> int32_t {
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1,
                           std::multiplies<int64_t>()) * elem_size;
};

int serializedAModel() {
    std::string modelFile = "./../segformer.onnx";

    /// init logger
    std::cout << "init logger" << std::endl;
    IBuilder* builder = createInferBuilder(logger);

    /// Creating a Network Definition
    std::cout << "def net" << std::endl;
    uint32_t flag = 1U <<static_cast<uint32_t> (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(flag);

    /// Importing a Model using the ONNX Parser
    std::cout << "read model" << std::endl;
    IParser*  parser = createParser(*network, logger);
    parser->parseFromFile(modelFile.data(), static_cast<int32_t>(ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    /// Building an Engine
    std::cout << "build model" << std::endl;
    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20);
    IHostMemory*  serializedModel = builder->buildSerializedNetwork(*network, *config);
    std::cout << "model built" << std::endl;

//    serializedModel->data();
    return 1;
}

int runAModel() {
    std::string modelFile = "./../segformer.onnx";

    /// init logger
    std::cout << "init logger" << std::endl;
    IBuilder* builder = createInferBuilder(logger);

    /// Creating a Network Definition
    std::cout << "def net" << std::endl;
    uint32_t flag = 1U <<static_cast<uint32_t> (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(flag);

    /// Importing a Model using the ONNX Parser
    std::cout << "read model" << std::endl;
    IParser*  parser = createParser(*network, logger);
    parser->parseFromFile(modelFile.data(), static_cast<int32_t>(ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    /// Building an Engine
    std::cout << "build model" << std::endl;
    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20);
    IHostMemory*  serializedModel = builder->buildSerializedNetwork(*network, *config);
    std::cout << "model built" << std::endl;

//    serializedModel->data();
    auto modelData = serializedModel->data();
    auto modelSize = serializedModel->size();

    std::string INPUT_NAME = "images";
    std::string OUTPUT_NAME = "seg";

    /// init logger
    IRuntime* runtime = createInferRuntime(logger);

    /// init engine
    ICudaEngine* engine = runtime->deserializeCudaEngine(modelData, modelSize);

    /// activate engine
    IExecutionContext *context = engine->createExecutionContext();

    ///
    cudaStream_t stream;
//    CUDA_CHECK(cudaStreamCreate(&stream));

    /// set io name
    int32_t inputIndex = engine->getBindingIndex(INPUT_NAME.data());
    int32_t outputIndex = engine->getBindingIndex(OUTPUT_NAME.data());

    /// set io buffer
    void* buffers[2];

    cudaMalloc((void**)buffers[0], 1 * 3 * kInputH * kInputW * sizeof(float));
    cudaMalloc((void**)buffers[1], 1 * 15120 * 8 * sizeof(float));

//    buffers[inputIndex] = inputBuffer;
//    buffers[outputIndex] = outputBuffer;

    /// read images
    std::vector<cv::Mat> img_batch;
    {
        cv::Mat img = cv::imread("./../0_camera0_RGB.png");
        img_batch.push_back(img);
    }

    /// prepare input
    {
        auto img = img_batch.at(0);
        int img_size = img.cols * img.rows * 3;
        int dst_size = kInputW * kInputH * 3;

        memcpy(img_buffer_host, img.data, img_size);
        cudaMemcpyAsync(img_buffer_device, img_buffer_host, img_size, cudaMemcpyHostToDevice, stream);
        buffers[0] = img_buffer_host;
    }

    /// inference
    context->enqueueV2(buffers, stream, nullptr);

    return 1;
}


int runTRT() {
//    std::string input_filename = "./../data/city/demo.png";
    std::string input_filename = "./../data/Cable/18_camera1_RGB.png";
//    std::string input_filename = "./../data/ICCV/iccv09Data/images/9005245.jpg";

    IRuntime* m_runtime;
    nvinfer1::ICudaEngine* m_engine;
    nvinfer1::IExecutionContext* m_context;
    int m_inputIndex0, m_outputIndex0, m_nbBindings;



    std::stringstream gieModelStream;
//    std::string enginePath = "./../data/Cable/segformer.b1.1280x720.cable07032024.160k.trt";
    std::string enginePath = "./../data/Cable/segformer.b1.720x720.cable07032024.160k.trt";
//    std::string enginePath = "./../data/city/segformer.b1.1024x1024.city.160k.trt";
//    std::string enginePath = "./../data/ICCV/segformer.b1.512x512.ade.160k.trt";
//    std::string enginePath = "./../data/Cable/segformer.trt";
//    std::string enginePath = "./../resnet_engine8.6.trt";

    gieModelStream.seekg(0, gieModelStream.beg);
    std::ifstream cache( enginePath, std::ios::in | std::ios::binary);
    gieModelStream << cache.rdbuf();
    cache.close();
    m_runtime = createInferRuntime(logger);
//    m_runtime = createInferRuntime(sample::gLogger.getTRTLogger());
    gieModelStream.seekg(0, std::ios::end);
    const int modelSize = gieModelStream.tellg();
    gieModelStream.seekg(0, std::ios::beg);
    void* modelMem = malloc(modelSize);
    gieModelStream.read((char*)modelMem, modelSize);

    std::cout << "model read" << std::endl;

    m_engine = m_runtime->deserializeCudaEngine(modelMem, modelSize);
    if (m_engine == nullptr) {
        free(modelMem);
        throw std::invalid_argument("Invalid Path Received.");
        return 0;
    }
    std::cout << "engine read" << std::endl;

//    m_inputIndex0 = m_engine->getBindingIndex(INPUT_BLOB_NAME);
//    m_outputIndex0 = m_engine->getBindingIndex(OUTPUT_BLOB_NAME);
//    m_nbBindings = m_engine->getNbBindings();
    m_context = m_engine->createExecutionContext();
//    cudaStream_t stream;

    /// check binding
    auto nb = m_engine->getNbBindings();
    for (int32_t i = 0; i < nb; i++) {
        auto is_input = m_engine->bindingIsInput(i);
        auto name = m_engine->getBindingName(i);
        auto dims = m_engine->getBindingDimensions(i);
        auto datatype = m_engine->getBindingDataType(i);
        static auto datatype_names = std::map<nvinfer1::DataType, std::string>{
                {nvinfer1::DataType::kFLOAT, "FLOAT"},
                {nvinfer1::DataType::kHALF,  "HALF"},
                {nvinfer1::DataType::kINT8,  "INT8"},
                {nvinfer1::DataType::kINT32, "INT32"},
                {nvinfer1::DataType::kBOOL,  "BOOL"},
        };

        std::cout << " " << (is_input ? "Input[" : "Output[") << i << "]"
                  << " name=" << name << " dims=[";
        for (int32_t j = 0; j < dims.nbDims; j++) {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1) std::cout << ",";
        }
        std::cout << "] datatype=" << datatype_names[datatype] << std::endl;

        // ...
    }


    /// allocate memory
//    auto nb = engine->getNbBindings();
    std::vector<void *> bindings(nb, nullptr);
    std::vector<int32_t> bindings_size(nb, 0);
    for (int32_t i = 0; i < nb; i++) {
        auto dims = m_engine->getBindingDimensions(i);
        auto size = GetMemorySize(dims, sizeof(float));
        if (cudaMalloc(&bindings[i], size) != cudaSuccess) {
            std::cerr << "ERROR: cuda memory allocation failed, size = " << size
                      << " bytes" << std::endl;
            return false;
        }
        bindings_size[i] = size;

        std::cout << "bind: " << i << " nbDims: " << dims.nbDims << " size: " << size << std::endl;
    }

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "ERROR: cuda stream creation failed" << std::endl;
        return false;
    }

    for (int i=1; i<100000000; i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        /// get input
        const std::vector<float> mean = {123.675, 116.28, 103.53};
        const std::vector<float> std = {58.395, 57.12, 57.375};

//        cv::Size input_size = {1280, 720};
        cv::Size input_size = {720, 720};
//        cv::Size input_size = {1024, 1024};
//        cv::Size input_size = {1240, 1240};
//        cv::Size input_size = {512, 512};
        auto img = cv::imread(input_filename, cv::IMREAD_ANYCOLOR);
        auto input = cv::Mat();
        input = img(cv::Range(0, 720), cv::Range(1280-720, 1280));
//        cv::resize(img, input, input_size);
        cv::imwrite("./input.png", input);
//        cv::imshow("input", input);
//        cv::waitKey(0);

        auto src = cv::Mat(input.rows, input.cols, CV_32FC3);
        int src_h = input.rows;
        int src_w = input.cols;
        {
            int src_n = src_h * src_w;
            auto src_data = (float *) (src.data);
            for (int y = 0; y < src_h; ++y) {
                for (int x = 0; x < src_w; ++x) {
                    auto &&bgr = input.at<cv::Vec3b>(y, x);
                    /*r*/ *(src_data + y * src_w + x) = (bgr[2] - mean[0]) / std[0];
                    /*g*/ *(src_data + src_n + y * src_w + x) = (bgr[1] - mean[1]) / std[1];
                    /*b*/ *(src_data + src_n * 2 + y * src_w + x) = (bgr[0] - mean[2]) / std[2];
//                    /*r*/ *(src_data + y * src_w + x) = bgr[2] / 255.;
//                    /*g*/ *(src_data + src_n + y * src_w + x) = bgr[1] / 255.;
//                    /*b*/ *(src_data + src_n * 2 + y * src_w + x) = bgr[0] / 255.;
                }
            }
        }
        std::cout << "img read: rows=" << img.rows << " cols=" << img.cols << " chs=" << img.channels() << " dims="
                  << img.dims << std::endl;
        std::cout << "src read: rows=" << src.rows << " cols=" << src.cols << " chs=" << src.channels() << " dims="
                  << src.dims << std::endl;

        if (cudaMemcpyAsync(bindings[0], src.data, bindings_size[0], cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            std::cerr << "ERROR: CUDA memory copy of src failed, size = "
                      << bindings_size[0] << " bytes" << std::endl;
            return false;
        }

        /// inference
        bool status = m_context->enqueueV2(bindings.data(), stream, nullptr);
        if (!status) {
            std::cout << "ERROR: TensorRT inference failed" << std::endl;
            return false;
        }

        /// Copy data from output binding memory
        auto res = cv::Mat(src_h, src_w, CV_32SC1);  // BCHW RGB [0,1] fp32
        if (cudaMemcpyAsync(res.data, bindings[1], bindings_size[1],
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
            std::cerr << "ERROR: CUDA memory copy of output failed, size = "
                      << bindings_size[1] << " bytes" << std::endl;
            return false;
        }
        cudaStreamSynchronize(stream);

//        for (int32_t i_nb = 0; i_nb < nb; i_nb++) {
//            bool freed = cudaFree(bindings[i]);
//            if (!freed) {
//                std::cerr << "ERROR: CUDA free memory failed, binding #"
//                << i_nb << " is_null = " << (bindings[i_nb] == nullptr) << " bytes"
//                << std::endl;
//                return false;
//            }
//        }
        auto t2 = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
        std::cout << "TensorRT inference done, take " << d.count() << " ms" << std::endl;

        auto res8bit = cv::Mat(src_h, src_w, CV_8UC1);  // BCHW RGB [0,1] fp32
        res.convertTo(res8bit, CV_8UC1, 70, 0);

        auto seg = cv::Mat(src_h, src_w, CV_8UC1);
        cv::resize(res8bit, seg, {input.cols, input.rows});
//        cv::resize(res8bit, seg, {img.cols, img.rows});
        cv::imwrite("./seg.png", seg);

    }
    return 1;
}