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
// 实现自己的 load 方法，加载模型+标签
int YOLOv8_wlzc_fruit::load(AAssetManager* mgr, const std::string& param_path, const std::string& bin_path, const std::string& label_path, bool use_gpu)
{
    // 加载模型
    int ret_param = fruit_net.load_param_asset(mgr, param_path.c_str());
    int ret_bin = fruit_net.load_model_asset(mgr, bin_path.c_str());
    if (ret_param != 0 || ret_bin != 0)
    {
        LOGE("Load fruit model failed");
        return -1;
    }
    fruit_net.opt.use_vulkan_compute = use_gpu;

    // 加载标签（和之前的 load_labels 逻辑一样）
    AAsset* asset = AAssetManager_open(mgr, label_path.c_str(), AASSET_MODE_BUFFER);
    if (!asset) return -1;
    // ... 省略标签加载的重复代码 ...
    AAsset_close(asset);
    return 0;
}
