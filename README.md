# FreeYOLO-onnx
Infer FreeYOLO for images and videos with onnx model.

![Image Test PNG](/outputs/test.png)

## 1.pip install
```bash
pip install -U pip && pip install opencv-python onnxruntime
```
## 2.pretraind-model
Taken from the official Github repository:
[FreeYOLO/deployment/ONNXRuntime](https://github.com/yjh0410/FreeYOLO/tree/master/deployment/ONNXRuntime)

## 3.Inference
### 3.1 Image
```bash
python onnx_inference.py --mode image --model model/yolo_free_tiny_opset_11.onnx -i test.jpg -s 0.3 --img_size 640
```
### 3.2Video
```bash
python onnx_inference.py --mode video --model model/yolo_free_tiny_opset_11.onnx -i sample.mp4 -s 0.3 --img_size 640
```
## 4.Reference
* [FreeYOLO](https://github.com/yjh0410/FreeYOLO)
* [FreeYOLO/deployment/ONNXRuntime](https://github.com/yjh0410/FreeYOLO/tree/master/deployment/ONNXRuntime)
