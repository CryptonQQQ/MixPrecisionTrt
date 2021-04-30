import sys
import onnx
import os
import argparse
import time
import numpy as np
import cv2
import onnxruntime
import tensorrt as trt
import torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tensorrt_engine import allocate_buffers, to_numpy, do_inference, postprocess_the_outputs
from utils.general import non_max_suppression

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) 

def w_bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def w_non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
   
    # box_corner = prediction.new(prediction.shape)
    box_corner = torch.FloatTensor(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
    
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]

        if not image_pred.size(0):
            continue

 
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()

        for c in unique_labels:
       
            detections_class = detections[detections[:, -1] == c]
         
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
        
            max_detections = []
            while detections_class.size(0):
              
                max_detections.append(detections_class[0].unsqueeze(0))
                if len(detections_class) == 1:
                    break
                ious = w_bbox_iou(max_detections[-1], detections_class[1:])
                detections_class = detections_class[1:][ious < nms_thres]
          
            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output


# class names
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear',
               'hair drier', 'toothbrush']


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def detect(official=True, image_path='',engine_model_path='',onnx_model_path=''):
    num_classes = 80
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    # input
    image_src = Image.open(image_path)
    resized = letterbox_image(image_src, (480, 640))
    img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    # print("Shape of the image input shape: ", img_in.shape)

    # inference
    if os.path.exists(onnx_model_path):
        session = onnxruntime.InferenceSession(onnx_model_path)
        # print("The model expects input shape: ", session.get_inputs()[0].shape)
        #batch_size = session.get_inputs()[0].shape[0]
        #img_size_h = session.get_inputs()[0].shape[2]
        #img_size_w = session.get_inputs()[0].shape[3]
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_in})
    elif os.path.exists(engine_model_path):
        with open(engine_model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine_fixed=runtime.deserialize_cuda_engine(f.read())
        context = engine_fixed.create_execution_context()
        inputs, outputs, bindings, stream = allocate_buffers(engine_fixed)
        inputs[0].host = img_in.reshape(-1)
        
        t1 = time.time()
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs,
                                   stream=stream)  # numpy data
        t2 = time.time()
        outputs[0] = postprocess_the_outputs(trt_outputs[0], (batch_size, 3, 80, 60, 85))
        outputs[1] = postprocess_the_outputs(trt_outputs[1], (batch_size, 3, 40, 30, 85))
        outputs[2] = postprocess_the_outputs(trt_outputs[2], (batch_size, 3, 20, 15, 85))
    batch_detections = []
    if official and len(outputs) == 4:   # model.model[-1].export = boolean ---> True:3 False:4
        # model.model[-1].export = False ---> outputs[0] (1, xxxx, 85)

        batch_detections = torch.from_numpy(np.array(outputs[0]))
        batch_detections = non_max_suppression(batch_detections, conf_thres=0.4, iou_thres=0.5, agnostic=False)
    else:
        # model.model[-1].export = False ---> outputs[1]/outputs[2]/outputs[2]
        # model.model[-1].export = True  ---> outputs
        # (1, 3, 20, 20, 85)
        # (1, 3, 40, 40, 85)
        # (1, 3, 80, 80, 85)

        boxs = []
        a = torch.tensor(anchors).float().view(3, -1, 2)
        anchor_grid = a.clone().view(3, 1, -1, 1, 1, 2)
        if len(outputs) == 4:
            outputs = [outputs[1], outputs[2], outputs[3]]
        for index, out in enumerate(outputs):
            out = torch.from_numpy(out)
            batch = out.shape[1]
            feature_w = out.shape[3]
            feature_h = out.shape[2]

            # Feature map corresponds to the original image zoom factor
            stride_w = int(480 / feature_w)
            stride_h = int(640 / feature_h)

            grid_x, grid_y = np.meshgrid(np.arange(feature_w), np.arange(feature_h))

            # cx, cy, w, h
            pred_boxes = torch.FloatTensor(out[..., :4].shape)
            pred_boxes[..., 0] = (torch.sigmoid(out[..., 0]) * 2.0 - 0.5 + grid_x) * stride_w  # cx
            pred_boxes[..., 1] = (torch.sigmoid(out[..., 1]) * 2.0 - 0.5 + grid_y) * stride_h  # cy
            pred_boxes[..., 2:4] = (torch.sigmoid(out[..., 2:4]) * 2) ** 2 * anchor_grid[index]  # wh

            conf = torch.sigmoid(out[..., 4])
            pred_cls = torch.sigmoid(out[..., 5:])

            output = torch.cat((pred_boxes.view(batch_size, -1, 4),
                                conf.view(batch_size, -1, 1),
                                pred_cls.view(batch_size, -1, num_classes)),
                               -1)
            boxs.append(output)

        outputx = torch.cat(boxs, 1)
        # NMS
        batch_detections = w_non_max_suppression(outputx, num_classes, conf_thres=opt.conf_thres, nms_thres=0.3)

    return batch_detections


def display(detections=None, image_path=None, line_thickness=None, text_bg_alpha=0.0):
    labels = detections[..., -1]
    boxs = detections[..., :4]
    confs = detections[..., 4]

    image_src = Image.open(image_path)
    w, h = image_src.size
    # resized = np.array(image_src)
    # resized = letterbox_image(image_src, (640, 640))
    # resized = np.array(resized)
    image_src = np.array(image_src)

    boxs[:, :] = scale_coords((640, 480), boxs[:, :], (h, w)).round()

    tl = line_thickness or round(0.002 * (w + h) / 2) + 1
    for i, box in enumerate(boxs):
        x1, y1, x2, y2 = box
        x1=int(x1)
        y1=int(y1)
        x2=int(x2)
        y2=int(y2)


        np.random.seed(int(labels[i].numpy()) + 2020)
        color = [np.random.randint(0, 255), 0, np.random.randint(0, 255)]
        cv2.rectangle(img=image_src, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=color, thickness=max(int((w + h) / 600), 1), lineType=cv2.LINE_AA)
        label = '%s %.2f' % (class_names[int(labels[i].numpy())], confs[i])
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=1)[0]
        c2 = x1 + t_size[0] + 3, y1 - t_size[1] - 5
        if text_bg_alpha == 0.0:
            cv2.rectangle(image_src, (x1 - 1, y1), c2, color, cv2.FILLED, cv2.LINE_AA)
        else:

            alphaReserve = text_bg_alpha  
            BChannel, GChannel, RChannel = color
            xMin, yMin = int(x1 - 1), int(y1 - t_size[1] - 3)
            xMax, yMax = int(x1 + t_size[0]), int(y1)
            image_src[yMin:yMax, xMin:xMax, 0] = image_src[yMin:yMax, xMin:xMax, 0] * alphaReserve + BChannel * (1 - alphaReserve)
            image_src[yMin:yMax, xMin:xMax, 1] = image_src[yMin:yMax, xMin:xMax, 1] * alphaReserve + GChannel * (1 - alphaReserve)
            image_src[yMin:yMax, xMin:xMax, 2] = image_src[yMin:yMax, xMin:xMax, 2] * alphaReserve + RChannel * (1 - alphaReserve)
        cv2.putText(image_src, label, (x1 + 3, y1 - 4), 0, tl / 3, [255, 255, 255],
                    thickness=1, lineType=cv2.LINE_AA)
        print(box.numpy(), confs[i].numpy(), class_names[int(labels[i].numpy())])

    plt.imshow(image_src)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_model_path', type=str, default='', help='model.trt path(s),model_save/yolov5s.trt')
    parser.add_argument('--img_path', type=str, default='test_image/bus.jpg', help='source') 
    parser.add_argument('--onnx_model_path', type=str, default='', help='model.onnx path(s),model_save/yolov5s.trt')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--batch-size', type=int, default=1, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)
    matplotlib.use('TKAgg')
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.style.use(['fast'])
    plt.rcParams['figure.facecolor'] = 'gray'
    batch_size=opt.batch_size
    with torch.no_grad():
        detections = detect(official=False, image_path=opt.img_path,engine_model_path=opt.engine_model_path,onnx_model_path=opt.onnx_model_path)
        if detections[0] is not None:
            display(detections[0], opt.img_path, text_bg_alpha=0.6)
