# tensorrt-lib

import os
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from tensorrt.tensorrt import BuilderFlag

from calibrator import Calibrator
from torch.autograd import Variable
import torch
import numpy as np
import time

# add verbose
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)  # ** engineø… ”ªØ **
f_layer = open('log.txt','w')

# create tensorrt-engine
# fixed and dynamic
def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="", \
               fp32_mode=False, fp16_mode=False, int4_mode=False, calibration_stream=None, calibration_table_path="", save_engine=False,strategy=None):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        # 1 << NetworkDefinitionCreationFlag.EXPLICIT_BATCH
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:

            # parse onnx model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
                #modify the orignal onnx network
                count=0;
                for i in range(network.num_layers):
                     net=network.get_layer(i)
                     if (net.type == trt.LayerType.CONVOLUTION and 'Conv' in net.name):
                         count=count+1;
                         #network.mark_output(net.get_output(0))
                         f_layer.write('----1----network.num_layers:' + str(network.num_layers) + '  .the number of the conv is:'+str(count)+'\n')
                         f_layer.write(net.name + '\n')
                         f_layer.write(str(net.type) + '\n')
                         #network.unmark_output(net.get_output(0))
                         activate = network.add_activation(input=net.get_output(0),
                                                           type=trt.ActivationType.CLIP)  # return a layer
                         if(strategy==None):
                           print('strategy error!')
                         activate.beta = pow(2,strategy[count-1]-1)-1
                         activate.alpha = -pow(2,strategy[count-1]-1)
                         activate.name = 'CLIP_%d' % i
                         f_layer.write(activate.name +' beta is '+ str(activate.beta)+' and alpha is '+str(activate.alpha) +'\n')
                         #get the layer next to conv,and input the output of the activation layer.
                         net_next= network.get_layer(i+1)
                         net_next.set_input(0,activate.get_output(0))
                         f_layer.write(net_next.name + '\n')
                         f_layer.write(str(net_next.type) + '\n')

                         net_next2 = network.get_layer(i + 2)
                         net_next2.set_input(0,net_next.get_output(0))
                         net_next2.set_input(1,activate.get_output(0))
                         f_layer.write(net_next2.name + '\n')
                         f_layer.write(str(net_next2.type) + '\n')
                f_layer.write('----2----network.num_layers:' + str(network.num_layers) + '\n')
                assert network.num_layers > 0, 'Failed to parse ONNX model. \
                            Please check if the ONNX model is compatible '
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

            # build trt engine
            if int4_mode:
                builder.max_batch_size = max_batch_size
                builder.int8_mode = int4_mode
                builder.max_workspace_size = 1 << 30  # 1GB
                assert calibration_stream, 'Error: a calibration_stream should be provided for int8 mode'
                builder.int8_calibrator = Calibrator(calibration_stream, calibration_table_path)
                engine = builder.build_cuda_engine(network)
                # config=builder.create_builder_config()
                # config.max_workspace_size = 1 << 30  #
                # config.flags = 1 << int(trt.BuilderFlag.INT8)  #
                # assert calibration_stream, 'Error: a calibration_stream should be provided for int8 mode'
                # config.int8_calibrator = Calibrator(calibration_stream, calibration_table_path)
                # engine = builder.build_engine(network, config)
                print('Int4 mode enabled')
            if fp16_mode:
                builder.strict_type_constraints=True
                builder.max_batch_size = max_batch_size
                builder.max_workspace_size = 1 << 30  # 1GB
                builder.fp16_mode = fp16_mode
                engine = builder.build_cuda_engine(network)
                print('fp16 modify mode enabled')
            if fp32_mode:
                builder.max_batch_size = max_batch_size
                builder.max_workspace_size = 1 << 30  # 1GB
                engine = builder.build_cuda_engine(network)
                print('fp32 modify mode enabled')
            if engine is None:
                print('Failed to create the engine')
                return None
            print("Completed creating the engine")
            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)
