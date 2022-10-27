# YOLOv7-Pose
Implementation of "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"

Pose estimation implimentation is based on [YOLO-Pose](https://arxiv.org/abs/2204.06806). 

## Dataset preparison

[[Keypoints Labels of MS COCO 2017]](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-keypoints.zip)

## Training

[yolov7-w6-person.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-person.pt)

``` shell
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train.py --data data/coco_kpts.yaml --cfg cfg/yolov7-w6-pose.yaml --weights weights/yolov7-w6-person.pt --batch-size 128 --img 960 --kpt-label --sync-bn --device 0,1,2,3,4,5,6,7 --name yolov7-w6-pose --hyp data/hyp.pose.yaml
```

## Deploy
### 1. export onnx
Running the following command will generate the onnx model and engine model in the same directory as the pt model
```bash
python models/export_onnx.py \
    --weights weights/yolov7-w6-pose.pt \
    --img-size 832 \
    --device 0 \
    --batch-size 1 \
    --simplify \
```
### 2.export TensorRT
```bash
# using scripts
python models/export_TRT.py \
    --onnx weights/yolov7-w6-pose.onnx \
    --batch-size 1 \
    --device 1 \
    --fp16

# using trtexec
trtexec \
    --onnx=weights/yolov7-w6-pose.onnx \
    --workspace=4096 \
    --saveEngine=weights/yolov7-w6-pose-FP16.engine \
    --fp16

```

## Inference
### 1. inference pytorch model
```bash
python detect_multi_backend.py \
    --weights weights/yolov7-w6-pose.pt \
    --source data/images \
    --device 0 \
    --img-size 832 \
    --kpt-label
```
### 2. inference onnx model
```bash
# test ONNX model
python detect_multi_backend.py \
    --weights weights/yolov7-w6-pose.onnx \
    --source data/images \
    --device 0 \
    --img-size 832 \
    --kpt-label
```
### 3. inference TensorRT model
```bash
# test Pytorch model
python detect_multi_backend.py \
    --weights weights/yolov7-w6-pose.engine \
    --source data/images \
    --device 0 \
    --img-size 832 \
    --kpt-label
```

## Testing
&emsp;The official YOLOv7-pose and YOLO-Pose code just calculate the detection mAP in test.py, if you want to calculate the keypoint mAP, you need to use the COCO API for calculation, but its oks_iou calculation is very slow, when calculate keypoints mAP in validation during the process of training will slow down the training process, so i implement the calculation of oks_iou with numpy matrix calculation in this repo, which speeds up the calculation of oks_iou when calculate keypoints mAP.   
&emsp;Note that the area calculation in the oks_iou implementation uses the ground truth box width and height product instead of the ground truth area of each object, because custom datasets often do not label the area of each object. See more detail in the [code](https://github.com/Gwencong/yolov7-pose-tensorrt/blob/main/utils/general.py#L537-L603).  
&emsp;When testing the key point mAP, the OKS area is set to `0.6 * ground truth box area`, so the keypoints mAP displayed by the terminal may be higher than the mAP calculated using the COCO API
### 1. test pytorch model
```bash
python test_multi_backend.py \
    --weights weights/yolov7-w6-pose.pt \
    --data data/coco_kpts.yaml \
    --img-size 832 \
    --conf-thres 0.001 \
    --iou-thres 0.6 \
    --task val \
    --device 0 \
    --kpt-label
```
### 2. test onnx model
```bash
# test ONNX model
python test_multi_backend.py \
    --weights weights/yolov7-w6-pose.onnx \
    --data data/custom_kpts.yaml \
    --img-size 832 \
    --conf-thres 0.001 \
    --iou-thres 0.6 \
    --task val \
    --device 0 \
    --kpt-label
```
### 3. test TensorRT model
```bash
# test Pytorch model
python test_multi_backend.py \
    --weights weights/yolov7-w6-pose-FP16.engine \
    --data data/coco_kpts.yaml \
    --img-size 832 \
    --conf-thres 0.001 \
    --iou-thres 0.6 \
    --task val \
    --device 0 \
    --kpt-label
```

## INT8 Calibration
```bash
python models/export_TRT.py \
    --onnx weights/yolov7-w6-pose.onnx \
    --batch-size 1 \
    --device 0 \
    --int8 \
    --calib_path data/coco_kpts/images \
    --calib_num 1024 \
    --calib_batch 128 \
    --calib_imgsz 832 \
    --cache_dir caches \
    --calib_method MinMax \
    --calib_letterbox
```

## Citation

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)
* [https://github.com/nanmi/yolov7-pose](https://github.com/nanmi/yolov7-pose)

</details>
