# MixPrecisionTrt  TRT2021



## 一·环境搭建

环境：ubuntu：18.04

cuda：11.1

cudnn：8.0

tensorrt：7.2.3.4

OpenCV：3.4.2

python3.8（使用其他版本可能会出现问题）

Conda

## 二.准备训练好的pytorch-yolov5模型,保存到model_save目录下       

```
git clone https://github.com.cnpmjs.org/ultralytics/yolov5.git #下载yolov5
```

2.1 yolov5s检测图形结果



## 三.导出export.py                                

### 3.1加载测试图片img，test_image/bus.jpg

--img_path test_image/bus.jpg

### 3.2使用torch.onnx.export导出yolov5s.onnx

​    onnx模型保存到pt模型的同一目录下，可以使用netron工具，查看图形化onnx模型

```
#在netron_yolov5s.py中修改

netron.start('此处填充简化后的onnx模型路径')
python netron_yolov5s.py                      #即可查看模型输出名
```



### 3.3使用yolov5s.pt计算预测

 --weights

```
y=model(img) 
```

### 3.4使用ONNX Runtime计算预测

```
ort_session = onnxruntime.InferenceSession(f)
ort_outs = ort_session.run(None, ort_inputs)
```



### 3.5比较精度，计算mse

```
np.testing.assert_allclose(to_numpy(y[1][0]), ort_outs[0], rtol=1e-03, atol=1e-05)
mse = np.sqrt(np.mean((to_numpy(y[1][0]) - ort_outs[0]) ** 2))
```

### 3.6结果

<img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210430100723048.png" alt="image-20210430100723048" style="zoom:80%;" />

### 示例

```
`python export.py --weights  model_save/yolov5s.pt --img_path test_image/bus.jpg --img-size 640 --batch-size 1`
```

##   四.生成tensorrt engine，精度转换                           

tensorrt_engine.py

util_trt_modify.py

calibrator.py

### 4.1准备COCO数据集,onnx模型，测试图片，engine保存位置

CALIB_IMG_DIR = 'coco/images/train2017'

--onnx_model_path = 'model_save/yolov5s.onnx'

--img_path = 'test_image/bus.jpg'

--engine_model_path = "model_save/yolov5s.trt"

###   4.2确定精度

--quantize fp32/fp16/int8/int4

###   4.3使用engine预测

`outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)`

### 4.4比较精度，计算mse

```
mse = np.sqrt(np.mean((feat[0] - ort_outs[0]) ** 2))
```

### 4.5 结果

fp32

<img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210430104502387.png" alt="image-20210430104502387" style="zoom:80%;" />



fp32 modify

<img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210430101038644.png" alt="image-20210430101038644" style="zoom:80%;" />

fp16

<img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210430105954134.png" alt="image-20210430105954134" style="zoom:80%;" />



fp16 modify

<img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210430102213229.png" alt="image-20210430102213229" style="zoom:80%;" />

int8

<img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210430110342714.png" alt="image-20210430110342714" style="zoom:80%;" />

int4

<img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210430102821459.png" alt="image-20210430102821459" style="zoom:80%;" />



### 示例

```
python tensorrt_engine.py  --quantize fp32 --onnx_model_path model_save/yolov5s.onnx --img_path test_image/bus.jpg --engine_model_path  model_save/yolov5s.trt
```

 

 

##  五.图像检测

detect.py

### 5.1选择检测使用的模型

​    --engine_model_path

​    --onnx_model_path

###  5.2选择需要检测的图片

--img_path test_image/bus.jpg

### 5.3结果

yolov5s.trt

![image-20210430111200036](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210430111200036.png)

yolov5s_fp16.trt

<img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210430112108415.png" alt="image-20210430112108415" style="zoom:80%;" />

yolov5s_int8.trt



### 示例

```
python detect.py --engine_model_path model_save/yolov5s.trt --img_path test_image/bus.jpg   #使用engine检测

python detect.py --onnx_model_path model_save/yolov5s.trt  --img_path test_image/bus.jpg    #使用onnx检测
```

