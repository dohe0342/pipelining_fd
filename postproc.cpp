//
//  postproc.cpp
//  pipelining_ex
//
//  Created by 김도희 on 2020/02/25.
//  Copyright © 2020 김도희. All rights reserved.
//
#include <stdio.h>
#include <algorithm>
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;

struct anchor_win
{
    float x_ctr;
    float y_ctr;
    float w;
    float h;
};

struct anchor_box
{
    float x1;
    float y1;
    float x2;
    float y2;
};

struct FacePts
{
    float x[5];
    float y[5];
};

struct FaceDetectInfo
{
    float score;
    anchor_box rect;
    FacePts pts;
};

struct anchor_cfg
{
    public:
        int STRIDE;
        vector<int> SCALES;
        int BASE_SIZE;
        vector<float> RATIOS;
        int ALLOWED_BORDER;

        anchor_cfg()
        {
            STRIDE = 0;
            SCALES.clear();
            BASE_SIZE = 0;
            RATIOS.clear();
            ALLOWED_BORDER = 0;
        }
};

anchor_win  _whctrs(anchor_box anchor)
{
    //Return width, height, x center, and y center for an anchor (window).
    anchor_win win;
    win.w = anchor.x2 - anchor.x1 + 1;
    win.h = anchor.y2 - anchor.y1 + 1;
    win.x_ctr = anchor.x1 + 0.5 * (win.w - 1);
    win.y_ctr = anchor.y1 + 0.5 * (win.h - 1);

    return win;
}

anchor_box _mkanchors(anchor_win win)
{
    //Given a vector of widths (ws) and heights (hs) around a center
    //(x_ctr, y_ctr), output a set of anchors (windows).
    anchor_box anchor;
    anchor.x1 = win.x_ctr - 0.5 * (win.w - 1);
    anchor.y1 = win.y_ctr - 0.5 * (win.h - 1);
    anchor.x2 = win.x_ctr + 0.5 * (win.w - 1);
    anchor.y2 = win.y_ctr + 0.5 * (win.h - 1);

    return anchor;
}

vector<anchor_box> _ratio_enum(anchor_box anchor, vector<float> ratios)
{
    //Enumerate a set of anchors for each aspect ratio wrt an anchor.
    vector<anchor_box> anchors;
    for (size_t i = 0; i < ratios.size(); i++) {
        anchor_win win = _whctrs(anchor);
        float size = win.w * win.h;
        float scale = size / ratios[i];

        win.w = round(sqrt(scale));
        win.h = round(win.w * ratios[i]);

        anchor_box tmp = _mkanchors(win);
        anchors.push_back(tmp);
    }

    return anchors;
}

vector<anchor_box> _scale_enum(anchor_box anchor, vector<int> scales)
{
    //Enumerate a set of anchors for each scale wrt an anchor.
    vector<anchor_box> anchors;
    for (size_t i = 0; i < scales.size(); i++) {
        anchor_win win = _whctrs(anchor);

        win.w = win.w * scales[i];
        win.h = win.h * scales[i];

        anchor_box tmp = _mkanchors(win);
        anchors.push_back(tmp);
    }

    return anchors;
}

vector<anchor_box> generate_anchors(int base_size = 16, vector<float> ratios = { 0.5, 1, 2 },
        vector<int> scales = { 8, 64 }, int stride = 16, bool dense_anchor = false)
{
    //Generate anchor (reference) windows by enumerating aspect ratios X
    //scales wrt a reference (0, 0, 15, 15) window.

    anchor_box base_anchor;
    base_anchor.x1 = 0;
    base_anchor.y1 = 0;
    base_anchor.x2 = base_size - 1;
    base_anchor.y2 = base_size - 1;

    vector<anchor_box> ratio_anchors;
    ratio_anchors = _ratio_enum(base_anchor, ratios);

    vector<anchor_box> anchors;
    for (size_t i = 0; i < ratio_anchors.size(); i++) {
        vector<anchor_box> tmp = _scale_enum(ratio_anchors[i], scales);
        anchors.insert(anchors.end(), tmp.begin(), tmp.end());
    }

    if (dense_anchor) {
        assert(stride % 2 == 0);
        vector<anchor_box> anchors2 = anchors;
        for (size_t i = 0; i < anchors2.size(); i++) {
            anchors2[i].x1 += stride / 2;
            anchors2[i].y1 += stride / 2;
            anchors2[i].x2 += stride / 2;
            anchors2[i].y2 += stride / 2;
        }
        anchors.insert(anchors.end(), anchors2.begin(), anchors2.end());
    }

    return anchors;
}

vector<vector<anchor_box> > generate_anchors_fpn(bool dense_anchor = false, vector<anchor_cfg> cfg = {})
{
    //Generate anchor (reference) windows by enumerating aspect ratios X
    //scales wrt a reference (0, 0, 15, 15) window.

    vector<vector<anchor_box> > anchors;
    for (size_t i = 0; i < cfg.size(); i++) {
        anchor_cfg tmp = cfg[i];
        int bs = tmp.BASE_SIZE;
        vector<float> ratios = tmp.RATIOS;
        vector<int> scales = tmp.SCALES;
        int stride = tmp.STRIDE;

        vector<anchor_box> r = generate_anchors(bs, ratios, scales, stride, dense_anchor);
        anchors.push_back(r);
    }

    return anchors;
}

vector<anchor_box> anchors_plane(int height, int width, int stride, vector<anchor_box> base_anchors)
{
    /*
height: height of plane
width:  width of plane
stride: stride ot the original image
anchors_base: a base set of anchors
*/

    vector<anchor_box> all_anchors;
    for (size_t k = 0; k < base_anchors.size(); k++) {
        for (int ih = 0; ih < height; ih++) {
            int sh = ih * stride;
            for (int iw = 0; iw < width; iw++) {
                int sw = iw * stride;

                anchor_box tmp;
                tmp.x1 = base_anchors[k].x1 + sw;
                tmp.y1 = base_anchors[k].y1 + sh;
                tmp.x2 = base_anchors[k].x2 + sw;
                tmp.y2 = base_anchors[k].y2 + sh;
                all_anchors.push_back(tmp);
            }
        }
    }

    return all_anchors;
}

void clip_boxes(vector<anchor_box> & boxes, int width, int height)
{
    //Clip boxes to image boundaries.
    for (size_t i = 0; i < boxes.size(); i++) {
        if (boxes[i].x1 < 0) {
            boxes[i].x1 = 0;
        }
        if (boxes[i].y1 < 0) {
            boxes[i].y1 = 0;
        }
        if (boxes[i].x2 > width - 1) {
            boxes[i].x2 = width - 1;
        }
        if (boxes[i].y2 > height - 1) {
            boxes[i].y2 = height - 1;
        }
        boxes[i].x1 = std::max<float>(std::min<float>(boxes[i].x1, width - 1), 0);
        boxes[i].y1 = std::max<float>(std::min<float>(boxes[i].y1, height - 1), 0);
        boxes[i].x2 = std::max<float>(std::min<float>(boxes[i].x2, width - 1), 0);
        boxes[i].y2 = std::max<float>(std::min<float>(boxes[i].y2, height - 1), 0);
    }
}

void clip_boxes(anchor_box & box, int width, int height)
{
    //Clip boxes to image boundaries.
    if (box.x1 < 0) {
        box.x1 = 0;
    }
    if (box.y1 < 0) {
        box.y1 = 0;
    }
    if (box.x2 > width - 1) {
        box.x2 = width - 1;
    }
    if (box.y2 > height - 1) {
        box.y2 = height - 1;
    }
    //    boxes[i].x1 = std::max<float>(std::min<float>(boxes[i].x1, width - 1), 0);
    //    boxes[i].y1 = std::max<float>(std::min<float>(boxes[i].y1, height - 1), 0);
    //    boxes[i].x2 = std::max<float>(std::min<float>(boxes[i].x2, width - 1), 0);
    //    boxes[i].y2 = std::max<float>(std::min<float>(boxes[i].y2, height - 1), 0);

}



vector<anchor_box> bbox_pred(vector<anchor_box> anchors, vector<Vec4f> regress)
{
    //"""
    //  Transform the set of class-agnostic boxes into class-specific boxes
    //  by applying the predicted offsets (box_deltas)
    //  :param boxes: !important [N 4]
    //  :param box_deltas: [N, 4 * num_classes]
    //  :return: [N 4 * num_classes]
    //  """

    vector<anchor_box> rects(anchors.size());
    for (size_t i = 0; i < anchors.size(); i++) {
        float width = anchors[i].x2 - anchors[i].x1 + 1;
        float height = anchors[i].y2 - anchors[i].y1 + 1;
        float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
        float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

        float pred_ctr_x = regress[i][0] * width + ctr_x;
        float pred_ctr_y = regress[i][1] * height + ctr_y;
        float pred_w = exp(regress[i][2]) * width;
        float pred_h = exp(regress[i][3]) * height;

        rects[i].x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
        rects[i].y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
        rects[i].x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
        rects[i].y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);
    }

    return rects;
}

anchor_box bbox_pred(anchor_box anchor, Vec4f regress)
{
    anchor_box rect;

    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
    float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

    float pred_ctr_x = regress[0] * width + ctr_x;
    float pred_ctr_y = regress[1] * height + ctr_y;
    float pred_w = exp(regress[2]) * width;
    float pred_h = exp(regress[3]) * height;

    rect.x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
    rect.y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
    rect.x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
    rect.y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);

    return rect;
}

vector<FacePts> landmark_pred(vector<anchor_box> anchors, vector<FacePts> facePts)
{
    vector<FacePts> pts(anchors.size());
    for (size_t i = 0; i < anchors.size(); i++) {
        float width = anchors[i].x2 - anchors[i].x1 + 1;
        float height = anchors[i].y2 - anchors[i].y1 + 1;
        float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
        float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

        for (size_t j = 0; j < 5; j++) {
            pts[i].x[j] = facePts[i].x[j] * width + ctr_x;
            pts[i].y[j] = facePts[i].y[j] * height + ctr_y;
        }
    }

    return pts;
}

FacePts landmark_pred(anchor_box anchor, FacePts facePt)
{
    FacePts pt;
    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
    float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

    for (size_t j = 0; j < 5; j++) {
        pt.x[j] = facePt.x[j] * width + ctr_x;
        pt.y[j] = facePt.y[j] * height + ctr_y;
    }

    return pt;
}

bool CompareBBox(const FaceDetectInfo & a, const FaceDetectInfo & b)
{
    return a.score > b.score;
}

vector<FaceDetectInfo> nms(vector<FaceDetectInfo> & bboxes, float threshold)
{
    vector<FaceDetectInfo> bboxes_nms;
    sort(bboxes.begin(), bboxes.end(), CompareBBox);

    int32_t select_idx = 0;
    int32_t num_bbox = static_cast<int32_t>(bboxes.size());
    vector<int32_t> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged) {
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        if (select_idx == num_bbox) {
            all_merged = true;
            continue;
        }

        bboxes_nms.push_back(bboxes[select_idx]);
        mask_merged[select_idx] = 1;

        anchor_box select_bbox = bboxes[select_idx].rect;
        float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) * (select_bbox.y2 - select_bbox.y1 + 1));
        float x1 = static_cast<float>(select_bbox.x1);
        float y1 = static_cast<float>(select_bbox.y1);
        float x2 = static_cast<float>(select_bbox.x2);
        float y2 = static_cast<float>(select_bbox.y2);

        select_idx++;
        for (int32_t i = select_idx; i < num_bbox; i++) {
            if (mask_merged[i] == 1)
                continue;

            anchor_box & bbox_i = bboxes[i].rect;
            float x = max<float>(x1, static_cast<float>(bbox_i.x1));
            float y = max<float>(y1, static_cast<float>(bbox_i.y1));
            float w = min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;   //<- float 型不加1
            float h = min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
            if (w <= 0 || h <= 0)
                continue;

            float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1));
            float area_intersect = w * h;


            if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > threshold) {
                mask_merged[i] = 1;
            }
        }
    }

    return bboxes_nms;
}

static void softmax(float *input, int input_len) {
    assert (input != NULL);
    assert (input_len != 0);

    int count = 0;

    float m = 0, sum = 0;
    while(count < input_len/2) {
        m = max(input[count], input[input_len/2+count]);
        //printf("%f\n", m);
        sum = expf(input[count]-m) + expf(input[input_len/2+count]-m);
        //printf("sum = %f\n", sum);
        input[count] = expf(input[count]-m-log(sum));
        input[input_len/2+count] = expf(input[input_len/2+count]-m-log(sum));
        count++;
    }
}


void retina_preprocessing(Mat& img, float *img_1d) {
    img.convertTo(img, CV_32FC3);

    cvtColor(img, img, CV_BGR2RGB);

    vector<Mat> input_channels;

    int width = 320;
    int height = 320;

    split(img, input_channels);

    //    load_input_lbl(img, input_size, "./input/input.txt");

    memcpy(img_1d,               input_channels[0].data, 320*320*sizeof(float));
    memcpy(img_1d + 320*320,     input_channels[1].data, 320*320*sizeof(float));
    memcpy(img_1d + 320*320*2,   input_channels[2].data, 320*320*sizeof(float));
}


int retina_postprocessing(float *face_rpn[9], float width_scale, float height_scale, int ws, int hs, Mat& src)
{
    ///////////////////////for caffe/////////////////////////////
    /*
       Caffe::set_mode(Caffe::CPU);
       boost::shared_ptr<Net<float> > Net_;
       Net_.reset(new Net<float>(("/home/dohe0342/extract-caffe-params/retina_model/mnet-deconv-0517.prototxt"), TEST));
       Net_->CopyTrainedLayersFrom(("/home/dohe0342/extract-caffe-params/retina_model//mnet-deconv-0517.caffemodel"));
       */
    /////////////////////////////////////////////////////////////

    ///////////////////////initialize////////////////////////////
    float threshold=0.30;
    float nms_threshold = 0.20;

    vector<int> _feat_stride_fpn;
    map<string, int> _num_anchors;
    bool dense_anchor = false;
    anchor_cfg tmp;
    vector<anchor_cfg> cfg;
    vector<float> _ratio;
    map<string, vector<anchor_box> > _anchors_fpn;
    _feat_stride_fpn = { 32, 16, 8 };
    _ratio = { 1.0 };
    tmp.SCALES = { 32, 16 };
    tmp.BASE_SIZE = 16;
    tmp.RATIOS = _ratio;
    tmp.ALLOWED_BORDER = 9999;
    tmp.STRIDE = 32;
    cfg.push_back(tmp);

    tmp.SCALES = { 8, 4 };
    tmp.BASE_SIZE = 16;
    tmp.RATIOS = _ratio;
    tmp.ALLOWED_BORDER = 9999;
    tmp.STRIDE = 16;
    cfg.push_back(tmp);

    tmp.SCALES = { 2, 1 };
    tmp.BASE_SIZE = 16;
    tmp.RATIOS = _ratio;
    tmp.ALLOWED_BORDER = 9999;
    tmp.STRIDE = 8;
    cfg.push_back(tmp);
    vector<vector<anchor_box> > anchors_fpn = generate_anchors_fpn(dense_anchor, cfg);

    for (size_t i = 0; i < anchors_fpn.size(); i++) {
        string key = "stride" + to_string(_feat_stride_fpn[i]);
        _anchors_fpn[key] = anchors_fpn[i];
        _num_anchors[key] = anchors_fpn[i].size();
    }
    //////////////////////////////////////////init done/////////////////////////////////////////////////////


    //////////////////////////////////////start preprocess/////////////////////////////////////////////////

    vector<FaceDetectInfo> faceInfo;

    for (size_t i = 0; i < _feat_stride_fpn.size(); i++) {
        //        inference output name is output.
        //i is stride, stride order is 32, 16, 8
        //output[3*i] is socre output
        //output[3*i+1] is bbox output
        //output[3*i+2] is landmark output

        string key = "stride" + to_string(_feat_stride_fpn[i]);
        int stride = _feat_stride_fpn[i];
        int score_count = 400*(int)pow(4,i);
        int bbox_count = 800*(int)pow(4,i);
        int landmark_count = 2000*(int)pow(4,i);

        softmax(face_rpn[3*i], score_count);
        const float* scoreB = face_rpn[3*i] + score_count / 2;
        const float* scoreE = scoreB + score_count / 2;
        vector<float> score = vector<float>(scoreB, scoreE);

        const float* bboxB = face_rpn[3*i + 1];
        const float* bboxE = bboxB + bbox_count;
        vector<float> bbox_delta = vector<float>(bboxB, bboxE);

        const float* landmarkB = face_rpn[3*i + 2];
        const float* landmarkE = landmarkB + landmark_count;
        vector<float> landmark_delta = vector<float>(landmarkB, landmarkE);

        int width = 10*(int)pow(2,i);
        int height = 10*(int)pow(2,i);
        size_t count = width * height;
        size_t num_anchor = _num_anchors[key];

        vector<anchor_box> anchors = anchors_plane(height, width, stride, _anchors_fpn[key]);

        for (size_t num = 0; num < num_anchor; num++) {
            for (size_t j = 0; j < count; j++) {
                float conf = score[j + count * num];
                if (conf <= threshold) {
                    continue;
                }

                Vec4f regress;
                float dx = bbox_delta[j + count * (0 + num * 4)];
                float dy = bbox_delta[j + count * (1 + num * 4)];
                float dw = bbox_delta[j + count * (2 + num * 4)];
                float dh = bbox_delta[j + count * (3 + num * 4)];
                regress = Vec4f(dx, dy, dw, dh);


                anchor_box rect = bbox_pred(anchors[j + count * num], regress);

                //clip_boxes(rect, ws, hs);

                FacePts pts;
                for (size_t k = 0; k < 5; k++) {
                    pts.x[k] = landmark_delta[j + count * (num * 10 + k * 2)];
                    pts.y[k] = landmark_delta[j + count * (num * 10 + k * 2 + 1)];
                }

                FacePts landmarks = landmark_pred(anchors[j + count * num], pts);

                FaceDetectInfo tmp;
                tmp.score = conf;
                tmp.rect = rect;
                tmp.pts = landmarks;
                faceInfo.push_back(tmp);
            }
        }
    }

    faceInfo = nms(faceInfo, nms_threshold);

    // OpenCV
    for (int u =0; u < faceInfo.size(); u++) {
        faceInfo[u].rect.x1 *= width_scale;
        faceInfo[u].rect.y1 *= height_scale;
        faceInfo[u].rect.x2 *= width_scale;
        faceInfo[u].rect.y2 *= height_scale;

        //cout << faceInfo[u].rect.x1 << " " << faceInfo[u].rect.y1 << " " << faceInfo[u].rect.x2 << " " << faceInfo[u].rect.y2 << endl;

        //rectangle(src, Point(faceInfo[u].rect.x1, faceInfo[u].rect.y1), Point(faceInfo[u].rect.x2, faceInfo[u].rect.y2), Scalar(0, 0, 255), 3);
        //cout << "face bbox xmin: " << faceInfo[u].rect.x1 << endl;
        //cout << "face bbox ymin: " << faceInfo[u].rect.y1 << endl;
        //cout << "face bbox xmax: " << faceInfo[u].rect.x2 << endl;
        //cout << "face bbox ymax: " << faceInfo[u].rect.y2 << endl;
        /*
        for (int v = 0; v < 5; v++) {
            faceInfo[u].pts.x[v] += cols_num;
            faceInfo[u].pts.y[v] += rows_num;

            line(src, Point(faceInfo[u].pts.x[v], faceInfo[u].pts.y[v]), Point(faceInfo[u].pts.x[v], faceInfo[u].pts.y[v]), Scalar(0, 0, 255), 3);
            //cout << "landmark x : " << faceInfo[u].pts.x[v] << endl;
            //cout << "landmark y : " << faceInfo[u].pts.y[v] << endl;
        }
        //printf("\n");
        */
    }
    //cout << "face num: " << faceInfo.size() << endl;
    /*
    for (int u = 0; u < faceInfo.size(); u++) {
        char xmin[100];
        char ymin[100];
        char xmax[100];
        char ymax[100];
        sprintf(xmin, "%f", faceInfo[u].rect.x1);
        sprintf(ymin, "%f", faceInfo[u].rect.y1);
        sprintf(xmax, "%f", faceInfo[u].rect.x2);
        sprintf(ymax, "%f", faceInfo[u].rect.y2);
        
        fprintf(fp, "%s %s %s %s\n", xmin, ymin, xmax, ymax);
    }
     */
    //imwrite("./output/result_img.jpg", src);
    return 0;
}

