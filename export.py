"""Exports a YOLOv5 *.pt model to ONNX and test the result

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ../model_save/yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys
import time

import cv2
import numpy as np
import onnxruntime

sys.path.append('./')
sys.path.append('./yolov5/')# to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

from models import experimental
from models import common
from utils import torch_utils
from utils import general
from utils import activations


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='model_save/yolov5s.pt', help='weights path')
    parser.add_argument('--img_path', type=str, default='test_image/bus.jpg', help='source')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=24, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--hyp', type=str, default='../yolov5/data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    general.set_logging()
    t = time.time()

    # Load PyTorch model
    device = torch_utils.select_device(opt.device)
    model = experimental.attempt_load(opt.weights, map_location=device)  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [general.check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input,picture
    img = cv2.imread(opt.img_path)
    img = letterbox(img, 640, stride=1)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # RGB transpose BRG
    img = np.ascontiguousarray(img)  # memery continued
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:  # add a dim in dim0
        img = img.unsqueeze(0)
    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = activations.Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = activations.SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)

    t1 = time.time()
    y = model(img)  # dry run
    t2 = time.time()
    model.model[-1].export = not opt.grid  # set Detect() layer grid export

    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=True, opset_version=11, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                                        'output': {0: 'batch', 2: 'y', 3: 'x'}} if opt.dynamic else None)

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # compute ONNX Runtime output prediction
    ort_session = onnxruntime.InferenceSession(f)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    t3 = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    t4 = time.time()
    # compare ONNX Runtime and PyTorch results

    # print(to_numpy(y[1][0][0][0][0][0]))
    # print(ort_outs[0][0][0][0][0])
    np.testing.assert_allclose(to_numpy(y[1][0]), ort_outs[0], rtol=1e-03, atol=1e-05)
    mse = np.sqrt(np.mean((to_numpy(y[1][0]) - ort_outs[0]) ** 2))
    print("Inference time with the PyTorch model: {}".format(t2 - t1))
    print("Inference time with the ONNX    model: {}".format(t4 - t3))
    print('MSE Error = {}'.format(mse))
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
