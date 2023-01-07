# FreeYOLO-onnx
Infer FreeYOLO for images and videos with onnx model.

# pip install
```bash
pip install -U pip
pip install opencv-python onnxruntime
```

# Inference
## Image
```bash
python onnx_inference.py --mode image --model model/yolo_free_tiny_opset_11.onnx -i test.jpg -s 0.3 --img_size 640
```
## Video
```bash
python onnx_inference.py --mode video --model model/yolo_free_tiny_opset_11.onnx -i sample.mp4 -s 0.3 --img_size 640
```
