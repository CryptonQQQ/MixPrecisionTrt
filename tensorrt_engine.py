# -*-coding:utf-8-*-
"""
input:a strategy,trt model,image for train
use trt model and strategy to make an engine

"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import onnxruntime
import pycuda.driver as cuda
import tensorrt as trt
import torch
import Logger
import util_trt
import util_trt_modify
from export import letterbox, to_numpy

BATCH_SIZE = 1
BATCH = 100
height = 640
width = 480

#calculate on GPU about the Specifications,such as the shape of input
def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

#calculate on GPU about making inference
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

#preprecross the images for inference
def preprocess_v1(image_raw):
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

    r_w = width / w
    r_h = height / h
    if r_h > r_w:
        tw = width
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((height - th) / 2)
        ty2 = height - th - ty1
    else:
        tw = int(r_h * w)
        th = height
        tx1 = int((width - tw) / 2)
        tx2 = width - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
    )
    image = image.astype(np.float32)
    image /= 255.0
    # HWC to CHW format
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    # image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    # image = np.ascontiguousarray(image)
    return image


def preprocess(img):
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """
        host_mem: cpu memory
        device_mem: gpu memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

#preprecross the images for generating the engine
class DataLoader:
    def __init__(self):
        self.index = 0
        self.length = BATCH
        self.batch_size = BATCH_SIZE
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = glob.glob(os.path.join(opt.CALIB_IMG_DIR, "*.jpg"))
        assert len(self.img_list) > self.batch_size * self.length, '{} must contains more than '.format(
            opt.CALIB_IMG_DIR) + str(self.batch_size * self.length) + ' images to calib'
        log.logger.info('found all {} images to calib.'.format(len(self.img_list)))
        self.calibration_data = np.zeros((self.batch_size, 3, height, width), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = preprocess_v1(img)
                self.calibration_data[i] = img

            self.index += 1

            # example only
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length


def main():

    strategy = [4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 8, 4, 4, 8, 4, 4, 8, 4, 4, 8, 4, 4, 4, 8, 8, 4, 4, 4, 8, 4, 8, 4, 4, 4,
                4, 8, 4, 6, 4, 8, 4, 4, 8, 6, 4, 4, 8, 4, 8, 8, 6, 4, 4, 8, 4, 4, 8, 4, 8, 4, 4]

    # onnx2trt
    fp16_mode = False
    int8_mode = False
    fp32_mode = False
    mix_mode = False
    if opt.quantize == 'mix':
        mix_mode = True
    if opt.quantize == 'int8':
        int8_mode = True
    elif opt.quantize == 'fp16':
        fp16_mode = True
    elif opt.quantize == 'fp32':
        fp32_mode = True
    else:
        log.logger.error('please set appropriate mode for quantification.(--quantize fp32)')

    log.logger.info('*** onnx to tensorrt begin ***')
    # calibration
    calibration_stream = DataLoader()
    if mix_mode:
        engine_fixed = util_trt_modify.get_engine(BATCH_SIZE, onnx_file_path=opt.onnx_model_path,
                                                  engine_file_path=opt.engine_model_path, fp32_mode=fp32_mode,
                                                  fp16_mode=fp16_mode,
                                                  mix_mode=mix_mode, calibration_stream=calibration_stream,
                                                  calibration_table_path=opt.calibration_table, save_engine=True,
                                                  strategy=strategy,log=log)
    else:
        engine_fixed = util_trt.get_engine(BATCH_SIZE, onnx_file_path=opt.onnx_model_path,
                                           engine_file_path=opt.engine_model_path, fp32_mode=fp32_mode,
                                           fp16_mode=fp16_mode,
                                           int8_mode=int8_mode, calibration_stream=calibration_stream,
                                           calibration_table_path=opt.calibration_table, save_engine=True,log=log)
    assert engine_fixed, 'Broken engine_fixed'
    log.logger.info('*** onnx to tensorrt completed ***\n')

    # --------------------inference------------------
    # Input,picture
    img = cv2.imread(opt.img_path)
    img = letterbox(img, 640, stride=1)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # do tensorrt inference
    context = engine_fixed.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine_fixed)
    shape_of_output = (BATCH_SIZE, 3, 80, 60, 85)
    inputs[0].host = to_numpy(img).reshape(-1)
    t1 = time.time()
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)  # numpy data
    t2 = time.time()
    feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)


    # do onnx inference
    ort_session = onnxruntime.InferenceSession(opt.onnx_model_path)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    t3 = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    t4 = time.time()

    mse = np.sqrt(np.mean((feat[0] - ort_outs[0]) ** 2))
    log.logger.info("Inference time with the TensorRT engine: {}".format(t2 - t1))
    log.logger.info("Inference time with the ONNX      model: {}".format(t4 - t3))
    log.logger.info('MSE Error = {}'.format(mse))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantize', type=str, default='fp32', help='mix,int8,fp16,fp32')
    parser.add_argument('--engine_model_path', type=str, default='model_save/yolov5s.trt', help='model.trt path(s)')
    parser.add_argument('--img_path', type=str, default='test_image/bus.jpg', help='source')
    parser.add_argument('--onnx_model_path', type=str, default='model_save/yolov5s.trt', help='model.onnx path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--CALIB_IMG_DIR', type=str, default='../tensorrt2/coco/images/train2017', help='coco images path(s)')
    parser.add_argument('--calibration_table', type=str, default='model_save/yolov5s_calibration.cache', help='yolov5s_calibration path(s)')
    opt = parser.parse_args()
    log = Logger.Logger('tensorrt_engine.log', 'info')
    log.logger.info(opt)
    main()
