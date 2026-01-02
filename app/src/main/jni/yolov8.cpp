// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "yolov8.h"

YOLOv8::~YOLOv8()
{
    det_target_size = 320;
}

int YOLOv8::load(const char* parampath, const char* modelpath, bool use_gpu)
{
    yolov8.clear();

    yolov8.opt = ncnn::Option();

#if NCNN_VULKAN
    yolov8.opt.use_vulkan_compute = use_gpu;
#endif

    yolov8.load_param(parampath);
    yolov8.load_model(modelpath);

    return 0;
}

int YOLOv8::load(AAssetManager* mgr, const char* parampath, const char* modelpath, bool use_gpu)
{
    yolov8.clear();

    yolov8.opt = ncnn::Option();

#if NCNN_VULKAN
    yolov8.opt.use_vulkan_compute = use_gpu;
#endif

    yolov8.load_param(mgr, parampath);
    yolov8.load_model(mgr, modelpath);

    return 0;
}

void YOLOv8::set_det_target_size(int target_size)
{
    det_target_size = target_size;
}

//wlzc 1
// YOLOv8_wlzc_fruit 的 load 方法：模型+标签一起加载
/*
int YOLOv8_wlzc_fruit::load(AAssetManager* mgr, const char* parampath, const char* modelpath, bool use_gpu)
{
    // 1. 加载模型
    int ret_param = fruit_net.load_param_asset(mgr, parampath);
    int ret_bin = fruit_net.load_model_asset(mgr, modelpath);
    if (ret_param != 0 || ret_bin != 0)
    {
        LOGE("Load fruit model failed: %s %s", parampath, modelpath);
        return -1;
    }
    fruit_net.opt.use_vulkan_compute = use_gpu;

    // 2. 加载标签（路径写死在类内部，外部无需传参）
    return load_labels(mgr);
}

// 内部工具函数：加载标签
int YOLOv8_wlzc_fruit::load_labels(AAssetManager* mgr)
{
    AAsset* asset = AAssetManager_open(mgr, label_path.c_str(), AASSET_MODE_BUFFER);
    if (!asset)
    {
        LOGE("Open label file failed: %s", label_path.c_str());
        return -1;
    }

    const char* data = (const char*)AAsset_getBuffer(asset);
    int len = AAsset_getLength(asset);
    std::string content(data, len);

    size_t pos = 0;
    while ((pos = content.find_first_of("\r\n")) != std::string::npos)
    {
        std::string line = content.substr(0, pos);
        if (!line.empty()) class_names.push_back(line);
        content.erase(0, pos + 1);
    }
    if (!content.empty()) class_names.push_back(content);

    AAsset_close(asset);
    LOGD("Loaded %d fruit classes", (int)class_names.size());
    return 0;
}
*/
//2026-01-02 6
// YOLOv8_wlzc_fruit::load 实现：参数为 const char*，和头文件声明一致
int YOLOv8_wlzc_fruit::load(AAssetManager* mgr, const char* parampath, const char* modelpath, bool use_gpu)
{
    // 1. 加载模型（参数 const char* 直接使用，和声明一致）
    int ret_param = fruit_net.load_param_asset(mgr, parampath);
    int ret_bin = fruit_net.load_model_asset(mgr, modelpath);
    if (ret_param != 0 || ret_bin != 0)
    {
        LOGE("Load fruit model failed: %s %s", parampath, modelpath);
        return -1;
    }
    fruit_net.opt.use_vulkan_compute = use_gpu;

    // 2. 加载标签（内部路径，无需外部传参，解决你说的标签配置问题）
    return load_labels(mgr);
}

// 内部工具函数：加载标签（使用类内部写死的 label_path）
int YOLOv8_wlzc_fruit::load_labels(AAssetManager* mgr)
{
    AAsset* asset = AAssetManager_open(mgr, label_path.c_str(), AASSET_MODE_BUFFER);
    if (!asset)
    {
        LOGE("Open label file failed: %s", label_path.c_str());
        return -1;
    }

    const char* data = (const char*)AAsset_getBuffer(asset);
    int len = AAsset_getLength(asset);
    std::string content(data, len);

    size_t pos = 0;
    while ((pos = content.find_first_of("\r\n")) != std::string::npos)
    {
        std::string line = content.substr(0, pos);
        if (!line.empty()) class_names.push_back(line);
        content.erase(0, pos + 1);
    }
    if (!content.empty()) class_names.push_back(content);

    AAsset_close(asset);
    LOGD("Loaded %d fruit classes", (int)class_names.size());
    return 0;
}

// 补全 detect 方法（和训练参数对齐，固定不变）
int YOLOv8_wlzc_fruit::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    // 复用统一设置的 target_size（即 det_target_size，父类成员变量）
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(det_target_size, det_target_size)); // 用统一的尺寸

    ncnn::Mat in = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_RGB, resized.cols, resized.rows);
    const float norm_vals[3] = {1.0f/255.0f, 1.0f/255.0f, 1.0f/255.0f};
    in.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = fruit_net.create_extractor();
    ex.set_light_mode(true);
    ex.input("input", in);

    ncnn::Mat out;
    ex.extract("output", out);

    // 解析最大概率类别
    float max_conf = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < out.h; i++)
    {
        float conf = out[i];
        if (conf > max_conf)
        {
            max_conf = conf;
            max_idx = i;
        }
    }

    // 保存分类结果
    result_conf = max_conf;
    if (max_conf >= conf_threshold && max_idx < class_names.size())
    {
        result_class = class_names[max_idx];
    }
    else
    {
        result_class = "Unknown";
    }

    return 0;
}

// 补全 draw 方法（固定不变）
int YOLOv8_wlzc_fruit::draw(cv::Mat& rgb, std::vector<Object>& objects)
{
    std::string text = "Fruit: " + result_class + " | Conf: " + std::to_string(result_conf);
    int baseLine = 0;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseLine);

    // 绘制文本背景
    cv::rectangle(rgb, cv::Point(10, 10), cv::Point(10 + text_size.width, 10 + text_size.height + baseLine), cv::Scalar(0, 255, 0), -1);
    // 绘制文本
    cv::putText(rgb, text, cv::Point(10, 10 + text_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);

    return 0;
}
