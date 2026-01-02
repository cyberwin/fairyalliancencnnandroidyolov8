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

#ifndef YOLOV8_H
#define YOLOV8_H

#include <opencv2/core/core.hpp>

#include <net.h>

struct KeyPoint
{
    cv::Point2f p;
    float prob;
};

struct Object
{
    cv::Rect_<float> rect;
    cv::RotatedRect rrect;
    int label;
    float prob;
    int gindex;
    cv::Mat mask;
    std::vector<KeyPoint> keypoints;
};

class YOLOv8
{
public:
    virtual ~YOLOv8();

    int load(const char* parampath, const char* modelpath, bool use_gpu = false);
    int load(AAssetManager* mgr, const char* parampath, const char* modelpath, bool use_gpu = false);

    void set_det_target_size(int target_size);

    virtual int detect(const cv::Mat& rgb, std::vector<Object>& objects) = 0;
    virtual int draw(cv::Mat& rgb, const std::vector<Object>& objects) = 0;

protected:
    ncnn::Net yolov8;
    int det_target_size;
};

class YOLOv8_det : public YOLOv8
{
public:
    virtual int detect(const cv::Mat& rgb, std::vector<Object>& objects);
};

class YOLOv8_det_coco : public YOLOv8_det
{
public:
    virtual int draw(cv::Mat& rgb, const std::vector<Object>& objects);
};

class YOLOv8_det_oiv7 : public YOLOv8_det
{
public:
    virtual int draw(cv::Mat& rgb, const std::vector<Object>& objects);
};

class YOLOv8_seg : public YOLOv8
{
public:
    virtual int detect(const cv::Mat& rgb, std::vector<Object>& objects);
    virtual int draw(cv::Mat& rgb, const std::vector<Object>& objects);
};

class YOLOv8_pose : public YOLOv8
{
public:
    virtual int detect(const cv::Mat& rgb, std::vector<Object>& objects);
    virtual int draw(cv::Mat& rgb, const std::vector<Object>& objects);
};

class YOLOv8_cls : public YOLOv8
{
public:
    virtual int detect(const cv::Mat& rgb, std::vector<Object>& objects);
    virtual int draw(cv::Mat& rgb, const std::vector<Object>& objects);
};

class YOLOv8_obb : public YOLOv8
{
public:
    virtual int detect(const cv::Mat& rgb, std::vector<Object>& objects);
    virtual int draw(cv::Mat& rgb, const std::vector<Object>& objects);
};

// 在 yolov8.h 
// 在 yolov8.h 末尾添加
/*
class YOLOv8_wlzc_fruit : public YOLOv8
{
public:
    // 自己实现 load 方法（因为不继承 YOLOv8_cls 了）
    int load(AAssetManager* mgr, const std::string& param_path, const std::string& bin_path, const std::string& label_path, bool use_gpu);
    // 重写必须实现的虚函数
    virtual int detect(const cv::Mat& rgb, std::vector<Object>& objects) override;
    virtual int draw(cv::Mat& rgb, std::vector<Object>& objects) override;
    // 自定义方法
    void set_target_size(int w, int h) { target_w = w; target_h = h; }

private:
    std::vector<std::string> class_names;
    int target_w = 100;
    int target_h = 100;
    float conf_threshold = 0.5f;
    mutable std::string result_class;
    mutable float result_conf;
    // 自己定义 Net，不再复用父类的 yolov8 成员
    ncnn::Net fruit_net;
};
*/
// yolov8.h 末尾添加 YOLOv8_wlzc_fruit 类（参数完全对齐）
/*
class YOLOv8_wlzc_fruit : public YOLOv8
{
public:
    // 声明：参数为 const char*（和父类一致，和调用/实现匹配），无外部传入label_path
    virtual int load(AAssetManager* mgr, const char* parampath, const char* modelpath, bool use_gpu = false) override;
    // 重写必须实现的虚函数（固定不变）
    virtual int detect(const cv::Mat& rgb, std::vector<Object>& objects) override;
    virtual int draw(cv::Mat& rgb, std::vector<Object>& objects) override;

private:
    ncnn::Net fruit_net;
    std::vector<std::string> class_names;
    float conf_threshold = 0.5f;
    mutable std::string result_class;
    mutable float result_conf;
    // 标签路径内部写死（无需外部传递，解决路径配置问题）
    const std::string label_path = "fruit_labels.txt";
    // 内部工具函数：加载标签（无需外部传参）
    int load_labels(AAssetManager* mgr);
};
*/
// 先看父类 YOLOv8 的虚函数签名（原文件里的）


// 修正 YOLOv8_wlzc_fruit 子类：override 方法必须和父类完全一致
class YOLOv8_wlzc_fruit : public YOLOv8
{
public:
    // 1. 重写 带 AAssetManager* 的 load 方法：和父类签名完全一致
    virtual int load(AAssetManager* mgr, const char* parampath, const char* modelpath, bool use_gpu = false) override;

    // 2. 重写 detect：和父类签名完全一致
    virtual int detect(const cv::Mat& rgb, std::vector<Object>& objects) override;

    // 3. 重写 draw：和父类签名完全一致
    virtual int draw(cv::Mat& rgb, std::vector<Object>& objects) override;

private:
    ncnn::Net fruit_net;
    std::vector<std::string> class_names;
    float conf_threshold = 0.5f;
    mutable std::string result_class;
    mutable float result_conf;
    const std::string label_path = "fruit_labels.txt";

    // 内部工具函数：加载标签（不加 override）
    int load_labels(AAssetManager* mgr);
};

#endif // YOLOV8_H
