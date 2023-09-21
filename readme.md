### 运行

直接运行track.py
结果在output文件夹中

### version:3.0 修改说明

    “占用”判断采用该视频帧前后多帧的IOU和box距离综合判断。

    “空位”状态在"驶出"状态出现后，逐秒判断之后满足条件的帧。
    发现输出的视频比原视频短，原因是未检测到的视频帧没输出，已修改。
    空位状态只会输出一次。
    输出的照片添加上了car_boundingbox。

    中间结果用pkl文件保存，加快第二次检测的速度。

### 待改进的点

（1）汽车车牌被遮挡，会空缺一段数据。

（2）即使未被遮挡，汽车也未被检测出来，可能是目标检测模型的问题。


### 添加了frame_gap

    （1）每隔 frame_gap 帧进行视频车辆检测, 可在 handlar/track.py line 104 修改， 默认 gap 数值 为 3

```
dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit, frame_gap=3)
```

### 添加低光判定逻辑 

    （1）在低光照时对图像补光， 在 car_plot/util/common.py line 132 增加了图像在暗光场景中的判断逻辑

    （2）增加暗光照图像自适应增强逻辑，参考 car_plot/util/hdr.py


### TODO 

    （1）基于 box 的轨迹估计修改为基于 中心点 的估计，


## 0920 TODO

    (1)  视频帧 -> [box(x1, y1, x2, y2) cls(5 or 6), track_id]
    (2) Set(
        {track_车牌号（可缺失）: 
            [box box(x1, y1, x2, y2)]
            [box box(x1, y1, x2, y2)]
            [box box(x1, y1, x2, y2)]
            }
    )

    (3)  判断车是否停进车库且被遮挡
    Set_1（停车前）(
        {track_车牌号（A12345）: 
            [box box(x1, y1, x2, y2)]
            [box box(x1, y1, x2, y2)]
            [box box(x1, y1, x2, y2)]
            }
    )

    Set_2 （驶出）(
        {track_车牌号（A12345）: 
            [box box(x1, y1, x2, y2)]
            [box box(x1, y1, x2, y2)]
            [box box(x1, y1, x2, y2)]
            }
    )

    (4) 删除 Tracks 中 Truck 类别的轨迹

    (5) 可选择的模型列表
        - https://github.com/ultralytics/yolov5/blob/master/README.zh-CN.md
        - 经过测试，可用的模型包含 YOLOv5n， YOLOv5s， YOLOv5m， YOLOv5l， YOLOv5x
        - 下载模型后，在setting/config.yml 中替换checkpoint 路径 即可
