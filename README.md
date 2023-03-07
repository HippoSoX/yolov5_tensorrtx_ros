# yolov5_tensorrtx

此包为目标检测yolov5算法通过TensorRT技术在Nidia AGX Xavier上部署的推理功能包，修改自开源项目[tensorrtx-yolov5](https://github.com/wang-xinyu/tensorrtx.git)。此包依赖于tensorrt库，使用前请确保当前环境与Nvidia显卡支持并安装tensorrt。否则请将此包加入编译黑名单中。

```
catkin_make -DCATKIN_BLACKLIST_PACKAGES="yolov5_tesnsorrtx"
```

# config

## model.yaml

用于修改模型路径而不用重新编译。模型需要用原生[tensorrtx-yolov5](https://github.com/wang-xinyu/tensorrtx.git)转换为.engine文件后再使用。此包修改后取消了转换模型的功能。

## yolov5_inference.yaml

目前只开放了是否publish推理后绘制了检测框的图像的接口。

# msgs

## fsd_common_msgs::YoloConeDetections & fsd_common_msgs::YoloCone

用于发布yolov5算法的检测结果。检测结果为像素坐标系下矩形框x,y,w,h的形式，同时发布颜色与颜色的置信度。

```
std_msgs/Float32 x                  # center x
std_msgs/Float32 y                  # center y
std_msgs/Float32 width              # width
std_msgs/Float32 height             # height

std_msgs/String color                 # color of cone, 'r' = red, 'b' = blue, 'y' = yellow
std_msgs/Float32 colorConfidence   	      # confidence of cone detect
```

# topics

此包订阅的话题有:

```
/hikrobot_camera/rgb                    type: sensor_msgs::Image
```

此包发布的话题有

```
/perception/camera/cone_detections      type: fsd_common_msgs::YoloConeDetections
/camera_detection_compute_time          type: std_msgs::Float32
/perception/camera/image_detected       type: sensor_msgs::Image
```

# run & launch

```
rosrun yolov5_tensorrtx yolov5
or
roslaunch yolov5_tensorrtx yolov5.launch
```