#include <iostream>
#include <chrono>
#include <cmath>
#include "tensorrtx/cuda_utils.h"
#include "tensorrtx/logging.h"
#include "tensorrtx/common.hpp"
#include "tensorrtx/utils.h"
#include "tensorrtx/calibrator.h"
#include "tensorrtx/preprocess.h"

#include "ros/ros.h"
#include "image_transport/image_transport.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CompressedImage.h"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "fsd_common_msgs/YoloCone.h"
#include "fsd_common_msgs/YoloConeDetections.h"
#include "std_msgs/Float32.h"

#include <pthread.h>

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.3
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 // ensure it exceed the maximum size in the input images !

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

bool g_pub_detect_image = false;
// cv::Mat g_img;

// pthread_mutex_t mutex;
// pthread_t thread_id;

static int get_width(int x, float gw, int divisor = 8) {
    return int(ceil((x * gw) / divisor)) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1) return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
        --r;
    }
    return std::max<int>(r, 1);
}

ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);
    /* ------ yolov5 backbone------ */
    auto conv0 = convBlock(network, weightMap, *data,  get_width(64, gw), 6, 2, 1,  "model.0");
    assert(conv0);
    auto conv1 = convBlock(network, weightMap, *conv0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(6, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
    auto bottleneck_csp8 = C3(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
    auto spp9 = SPPF(network, weightMap, *bottleneck_csp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, "model.9");
    /* ------ yolov5 head ------ */
    auto conv10 = convBlock(network, weightMap, *spp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

    auto upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(ResizeMode::kNEAREST);
    upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());

    ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

    auto upsample15 = network->addResize(*conv14->getOutput(0));
    assert(upsample15);
    upsample15->setResizeMode(ResizeMode::kNEAREST);
    upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());

    ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

    /* ------ detect ------ */
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer*>{det0, det1, det2});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));
    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

ICudaEngine* build_engine_p6(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);
    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);
    
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    /* ------ yolov5 backbone------ */
    auto conv0 = convBlock(network, weightMap, *data,  get_width(64, gw), 6, 2, 1,  "model.0");
    auto conv1 = convBlock(network, weightMap, *conv0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto c3_2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *c3_2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto c3_4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(6, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *c3_4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto c3_6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *c3_6->getOutput(0), get_width(768, gw), 3, 2, 1, "model.7");
    auto c3_8 = C3(network, weightMap, *conv7->getOutput(0), get_width(768, gw), get_width(768, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
    auto conv9 = convBlock(network, weightMap, *c3_8->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.9");
    auto c3_10 = C3(network, weightMap, *conv9->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), true, 1, 0.5, "model.10");
    auto sppf11 = SPPF(network, weightMap, *c3_10->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, "model.11");

    /* ------ yolov5 head ------ */
    auto conv12 = convBlock(network, weightMap, *sppf11->getOutput(0), get_width(768, gw), 1, 1, 1, "model.12");
    auto upsample13 = network->addResize(*conv12->getOutput(0));
    assert(upsample13);
    upsample13->setResizeMode(ResizeMode::kNEAREST);
    upsample13->setOutputDimensions(c3_8->getOutput(0)->getDimensions());
    ITensor* inputTensors14[] = { upsample13->getOutput(0), c3_8->getOutput(0) };
    auto cat14 = network->addConcatenation(inputTensors14, 2);
    auto c3_15 = C3(network, weightMap, *cat14->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.15");

    auto conv16 = convBlock(network, weightMap, *c3_15->getOutput(0), get_width(512, gw), 1, 1, 1, "model.16");
    auto upsample17 = network->addResize(*conv16->getOutput(0));
    assert(upsample17);
    upsample17->setResizeMode(ResizeMode::kNEAREST);
    upsample17->setOutputDimensions(c3_6->getOutput(0)->getDimensions());
    ITensor* inputTensors18[] = { upsample17->getOutput(0), c3_6->getOutput(0) };
    auto cat18 = network->addConcatenation(inputTensors18, 2);
    auto c3_19 = C3(network, weightMap, *cat18->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.19");

    auto conv20 = convBlock(network, weightMap, *c3_19->getOutput(0), get_width(256, gw), 1, 1, 1, "model.20");
    auto upsample21 = network->addResize(*conv20->getOutput(0));
    assert(upsample21);
    upsample21->setResizeMode(ResizeMode::kNEAREST);
    upsample21->setOutputDimensions(c3_4->getOutput(0)->getDimensions());
    ITensor* inputTensors21[] = { upsample21->getOutput(0), c3_4->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors21, 2);
    auto c3_23 = C3(network, weightMap, *cat22->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.23");

    auto conv24 = convBlock(network, weightMap, *c3_23->getOutput(0), get_width(256, gw), 3, 2, 1, "model.24");
    ITensor* inputTensors25[] = { conv24->getOutput(0), conv20->getOutput(0) };
    auto cat25 = network->addConcatenation(inputTensors25, 2);
    auto c3_26 = C3(network, weightMap, *cat25->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.26");

    auto conv27 = convBlock(network, weightMap, *c3_26->getOutput(0), get_width(512, gw), 3, 2, 1, "model.27");
    ITensor* inputTensors28[] = { conv27->getOutput(0), conv16->getOutput(0) };
    auto cat28 = network->addConcatenation(inputTensors28, 2);
    auto c3_29 = C3(network, weightMap, *cat28->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.29");

    auto conv30 = convBlock(network, weightMap, *c3_29->getOutput(0), get_width(768, gw), 3, 2, 1, "model.30");
    ITensor* inputTensors31[] = { conv30->getOutput(0), conv12->getOutput(0) };
    auto cat31 = network->addConcatenation(inputTensors31, 2);
    auto c3_32 = C3(network, weightMap, *cat31->getOutput(0), get_width(2048, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.32");

    /* ------ detect ------ */
    IConvolutionLayer* det0 = network->addConvolutionNd(*c3_23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.0.weight"], weightMap["model.33.m.0.bias"]);
    IConvolutionLayer* det1 = network->addConvolutionNd(*c3_26->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.1.weight"], weightMap["model.33.m.1.bias"]);
    IConvolutionLayer* det2 = network->addConvolutionNd(*c3_29->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.2.weight"], weightMap["model.33.m.2.bias"]);
    IConvolutionLayer* det3 = network->addConvolutionNd(*c3_32->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.3.weight"], weightMap["model.33.m.3.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, "model.33", std::vector<IConvolutionLayer*>{det0, det1, det2, det3});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, bool& is_p6, float& gd, float& gw, std::string& wts_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = nullptr;
    if (is_p6) {
        engine = build_engine_p6(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    } else {
        engine = build_engine(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    }
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize) {
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

// bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& img_dir) {
//     if (argc < 4) return false;
//     if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
//         wts = std::string(argv[2]);
//         engine = std::string(argv[3]);
//         auto net = std::string(argv[4]);
//         if (net[0] == 'n') {
//             gd = 0.33;
//             gw = 0.25;
//         } else if (net[0] == 's') {
//             gd = 0.33;
//             gw = 0.50;
//         } else if (net[0] == 'm') {
//             gd = 0.67;
//             gw = 0.75;
//         } else if (net[0] == 'l') {
//             gd = 1.0;
//             gw = 1.0;
//         } else if (net[0] == 'x') {
//             gd = 1.33;
//             gw = 1.25;
//         } else if (net[0] == 'c' && argc == 7) {
//             gd = atof(argv[5]);
//             gw = atof(argv[6]);
//         } else {
//             return false;
//         }
//         if (net.size() == 2 && net[1] == '6') {
//             is_p6 = true;
//         }
//     } else if (std::string(argv[1]) == "-d" && argc == 4) {
//         engine = std::string(argv[2]);
//         img_dir = std::string(argv[3]);
//     } else {
//         return false;
//     }
//     return true;
// }

// void* detected_image_publish_thread(void* image_pub) {
//     pthread_mutex_lock(&mutex);

//     auto start = std::chrono::system_clock::now();

//     sensor_msgs::ImagePtr g_img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", g_img).toImageMsg();
//     *(image_transport::Publisher*)image_pub.publish(g_img_msg);

//     auto end = std::chrono::system_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

//     ROS_INFO("Detected image publish use time: %dms", (int)duration);

//     pthread_mutex_unlock(&mutex);
// }

void Callback(const sensor_msgs::ImageConstPtr &msg, uint8_t* img_host, uint8_t* img_device, 
                    cudaStream_t& stream, float** buffers, IExecutionContext* context, float* prob,
                    ros::NodeHandle& nh) {
    // Detect part.

    static ros::Publisher result_pub = nh.advertise<fsd_common_msgs::YoloConeDetections>("/perception/camera/cone_detections", 1);
    static ros::Publisher inference_time_pub = nh.advertise<std_msgs::Float32>("/camera_detection_compute_time", 1);

    static cv::Mat img; // Why using static?

    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    img = cv_ptr->image;
 
    // bgr 2 rgb
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    //auto start = std::chrono::system_clock::now();
    float* buffer_idx = (float*)buffers[0];

    size_t  size_image = img.cols * img.rows * 3;
    // size_t  size_image_dst = INPUT_H * INPUT_W * 3;

    //copy data to pinned memory
    memcpy(img_host,img.data,size_image);
    //copy data to device memory
    CUDA_CHECK(cudaMemcpyAsync(img_device,img_host,size_image,cudaMemcpyHostToDevice,stream));

    preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);

    // Run inference
    auto start = std::chrono::system_clock::now();
    doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    //std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    ROS_INFO("Inference time: %dms", (int)duration);
    std_msgs::Float32 inference_time;
    inference_time.data = (float)duration;
    inference_time_pub.publish(inference_time);

    std::vector<std::vector<Yolo::Detection>> batch_res(1);

    /**
     * @brief Structure of res
     *  struct alignas(float) Detection {
     *      //center_x center_y w h
     *      float bbox[LOCATIONS];
     *      float conf;  // bbox_conf * cls_conf
     *      float class_id;
     *  };
     */
    auto& res1 = batch_res[0];
    nms(res1, &prob[0], CONF_THRESH, NMS_THRESH);

    auto& res2 = batch_res[0];
    // cv::Mat predict_img = img;
    fsd_common_msgs::YoloConeDetections predict_result_msgs;
    predict_result_msgs.image_header = msg->header;

    for (size_t j = 0; j < res2.size(); j++) {
        // cv::Rect r = get_rect(predict_img, res2[j].bbox);
        fsd_common_msgs::YoloCone result;
        // 发布消息时，以实际比例发出，而不是640*640的比例
        result.x.data       =   res2[j].bbox[0] / INPUT_W * img.rows;
        result.y.data       =   res2[j].bbox[1] / INPUT_H * img.cols;
        result.width.data   =   res2[j].bbox[2] / INPUT_W * img.rows;
        result.height.data  =   res2[j].bbox[3] / INPUT_H * img.cols;
        result.colorConfidence.data = res2[j].conf;
        // 0: red, 1: blue, 2: yellow
        // Drawing under bgr8
        if((int)res2[j].class_id == 0) {
            // cv::rectangle(predict_img, r, cv::Scalar(0x00, 0x00, 0xFF), 2);
            // cv::putText(predict_img, "red", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0x00, 0x00, 0xFF), 2);
            result.color.data = "r";
        }
        else if((int)res2[j].class_id == 1) {
            // cv::rectangle(predict_img, r, cv::Scalar(0xFF, 0x00, 0x00), 2);
            // cv::putText(predict_img, "blue", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0x00, 0x00), 2);
            result.color.data = "b";
        }
        else if((int)res2[j].class_id == 2) {
            // cv::rectangle(predict_img, r, cv::Scalar(0x00, 0xFF, 0xFF), 2);
            // cv::putText(predict_img, "yellow", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0x00, 0xFF, 0xFF), 2);
            result.color.data = "y";
        }
        predict_result_msgs.cone_detections.push_back(result);
    }


    // Output as files.
    // cv::imwrite("_" + file_names[f - fcount + 1 + b], img);

    // Publish predict image.
    // cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

    // sensor_msgs::ImagePtr predict_img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", predict_img).toImageMsg();
    // image_pub.publish(predict_img_msg);
    // if(g_pub_detect_image) {
    //     g_img = predict_img;
    //     pthread_create(&thread_id, NULL, detected_image_publish_thread, image_pub);
    // }

    result_pub.publish(predict_result_msgs);
}

int main(int argc, char** argv) {

    ros::init(argc, argv, "yolov5_tensorrtx");
    ros::NodeHandle nh;

    cudaSetDevice(DEVICE);

    // Setup the parameters.
    std::string wts_name = "";
    static std::string engine_name = "/home/nvidia/catkin_ws/src/camera_perception/yolov5_tensorrtx/model/cone_20220819.engine";
    std::string input_image_topic_name = "/hikrobot_camera/rgb";
    if(nh.hasParam("yolov5_model/path")) {
        std::string path, name;
        nh.param("yolov5_model/path", path, std::string("/home/nvidia/catkin_ws/src/camera_perception/yolov5_tensorrtx/model/"));
        nh.param("yolov5_model/name", name, std::string("cone_20220818.engine"));
        nh.param("pub_predicted_image", g_pub_detect_image, false);
        nh.param("input_image_topic_name", input_image_topic_name, std::string("/hikrobot_camera/rgb"));
        engine_name = path + name;
    }
    ROS_INFO("Using model: %s", engine_name.c_str());
    // if(nh.hasParam("pub_detected_image")) {
    //     nh.param("pub_detected_image", g_pub_detect_image, false);
    // }
    // if(g_pub_detect_image){
    //     pthread_mutex_init(&mutex, NULL);
    // }
    bool is_p6 = false;
    float gd = 0.0f, gw = 0.0f;
    // yolov5s
    gd = 0.33;
    gw = 0.50;
    // std::string img_dir;
    // if (!parse_args(argc, argv, wts_name, engine_name, is_p6, gd, gw, img_dir)) {
    //     std::cerr << "arguments not right!" << std::endl;
    //     std::cerr << "./yolov5 -s [.wts] [.engine] [n/s/m/l/x/n6/s6/m6/l6/x6 or c/c6 gd gw]  // serialize model to plan file" << std::endl;
    //     std::cerr << "./yolov5 -d [.engine] ../samples  // deserialize plan file and run inference" << std::endl;
    //     return -1;
    // }

    // create a model using the API directly and serialize it to a stream
    if (!wts_name.empty()) {
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream, is_p6, gd, gw, wts_name);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }

    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    // std::vector<std::string> file_names;
    // if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
    //     std::cerr << "read_files_in_dir failed." << std::endl;
    //     return -1;
    // }

    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    float* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;
    // prepare input data cache in pinned memory 
    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));

    image_transport::ImageTransport it(nh);

    // image_transport::Publisher image_pub = it.advertise("/perception/camera/image_detected", 1);

    // image_transport::Subscriber sub = it.subscribe("/pylon_camera_node/image_raw", 1, 
    //                                     boost::bind(&Callback, _1, img_host, img_device, 
    //                                     stream, buffers, context, prob, image_pub, result_pub));

    image_transport::Subscriber sub = it.subscribe("/hikrobot_camera/rgb", 1, 
                                        boost::bind(&Callback, _1, img_host, img_device, 
                                        stream, buffers, context, prob, nh));

    ros::spin();

    // Synchron thread
    // if(g_pub_detect_image) {
    //     pthread_join(thread_id, NULL);
    //     pthread_mutex_destroy(&mutex);
    // }
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(img_device));
    CUDA_CHECK(cudaFreeHost(img_host));
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();


    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}
