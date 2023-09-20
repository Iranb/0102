# limit the number of cpus used by high performance libraries

import os
import warnings
import copy
import glob
import pathlib
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import math
import torchvision
# sys.path.insert(0, './yolov5')
lib_path = os.path.abspath(os.path.join('infrastructure', 'yolov5'))
sys.path.append(lib_path)
sys.path.append('task_02\Vehicle-tracking\car_plot')
from infrastructure.yolov5.models.experimental import attempt_load
from infrastructure.yolov5.utils.downloads import attempt_download
from infrastructure.yolov5.models.common import DetectMultiBackend
from infrastructure.yolov5.utils.datasets import LoadImages, LoadStreams
from infrastructure.yolov5.utils.general import LOGGER, check_img_size, increment_path, non_max_suppression, scale_coords, check_imshow, xyxy2xywh, increment_path
from infrastructure.yolov5.utils.torch_utils import select_device, time_sync
from infrastructure.yolov5.utils.plots import Annotator, colors
from infrastructure.deep_sort_pytorch.utils.parser import get_config
from infrastructure.deep_sort_pytorch.deep_sort import DeepSort

from util.common import  *
from util.OPT_config import OPT
from datetime import timedelta, datetime
from pathlib import Path
import cv2
import copy
import numpy as np
import time
import torch
import hyperlpr3 as lpr3


class Tracker:
    def __init__(self, config_path:str) -> None:

        config = read_yml(config_path)
        self.opt = OPT(config=config)
        self.opt.imgsz *= 2 if len(self.opt.imgsz) == 1 else 1  # expand
        self.catcher = lpr3.LicensePlateCatcher(detect_level=1)
        self.exit_frames = 175

    # TODO 
    def detect(self, set_source='/home/hyq/code/comp/dataset/prestige/H8V-陕AE8R40.avi'):
        out_path = Path('./output/') / Path(set_source).stem
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if os.path.isfile(str(out_path / 'frame_list.pkl')):
            frame_list = load_list_from_file(str(out_path / 'frame_list.pkl'))
            self.generate_results(frame_list, set_source)
            return
        opt = self.opt
        out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, save_csv, imgsz, evaluate, half = \
            opt.output, set_source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
                opt.save_txt, opt.save_csv, opt.imgsz, opt.evaluate, opt.half
        upper_ratio = opt.upper_ratio
        lower_ratio = opt.lower_ratio

        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(opt.config_deepsort)
        attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        # Initialize
        device = select_device(opt.device)
        half = True  # half precision only suvehiclesorted on CUDA

        if not evaluate:
            if os.path.exists(out):
                pass
                # shutil.rmtree(out)  # delete output folder
            else:
                os.makedirs(out)  # make new output folder

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(opt.yolo_weights, device=device, dnn=opt.dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)  # check image size


        model.model.half()

        # Set Dataloader
        vid_path, vid_writer = None, None
        # Check if environment suvehiclesorts image displays
        if show_vid:
            show_vid = check_imshow()

        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit, frame_gap=1)
        bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs


        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        save_path = str(Path(out))
        # extract what is in between the last '/' and last '.'
        txt_file_name = source.split('/')[-1].split('.')[0]

        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0

        previous_frame, current_frame = [-1, -1]
        frame_list = []
        vehicle_infos = {} # id:{start in view, exit view, type }
        list_vehicles = set()  #LIST CONTAIN vehicles HAS APPEARED, IF THAT VEHICLE HAD BEEN UPLOADED TO DB, REMOVE THAT VEHICLE
        
        site_last = [0,0,0,0,0] # 最后一个是帧位
        for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
            # if frame_idx % 100 == 0:
            #     print(source[-11:-4]+": "+ str(frame_idx)+"/"+ str(dataset.frames))
            t1 = time_sync()
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            frame_height = im0s.shape[0]
            frame_width = im0s.shape[1]

            # # resize 
            # print(frame_height)
            # print(frame_width)
            # exit()
            print(img.size())
            height = 1080
            width = 1920
            upper_line = int(frame_height*upper_ratio)
            lower_line = int(frame_height*lower_ratio)
            middle_line = frame_width//2

            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_path / Path(path).stem, mkdir=True) if opt.visualize else False
            pred = model(img, augment=opt.augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # Avehiclesly NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
            dt[2] += time_sync() - t3

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                seen += 1

                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                # s += '%gx%g ' % img.shape[2:]  # print string
                save_path = str(Path(out) / Path(p).name)

                annotator = Annotator(im0, line_width=2, pil=not ascii)
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    # for c in det[:, -1].unique():
                    #     n = (det[:, -1] == c).sum()  # detections per class
                    #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]                   

                    # # pass detections to deepsort, only objects in used zone                     
                    xywhs = np.asarray(xywhs.cpu())
                    confs = np.asarray(confs.cpu())
                    clss = np.asarray(clss.cpu())

                    row_indexes_delete = []
                    for i, cord in enumerate(xywhs):
                        # if (cord[1]+cord[3])>lower_line or cord[1]<upper_line:
                        if (cord[1]+cord[3])<upper_line or cord[1]>lower_line:
                            row_indexes_delete.append(i)
                    xywhs = np.delete(xywhs, row_indexes_delete, axis=0)
                    confs = np.delete(confs, row_indexes_delete)
                    clss = np.delete(clss, row_indexes_delete)

                    xywhs = torch.tensor(xywhs)
                    confs = torch.tensor(confs)
                    clss = torch.tensor(clss)

                    # NOTE DeepSort 模型推理
                    # im0 每帧原始图像
                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    # print(outputs)
                    # NOTE 检测车牌信息
                    licenses = self.detect_car_license(im0s)
                    # print(licenses)
                    # NOTE 车与车牌对应起来
                    if len(outputs) != 0:
                        car_and_licenses = self.assign_license_to_vehicle(licenses, outputs)
                        # print(car_and_licenses)
                        
                        for item in car_and_licenses:
                            if len(item) == 2: # 意味着空车牌
                                continue
                            site_car, plate = item[0], item[1]
                            if plate != '贵A1ZM03':
                                continue
                            elif sum(site_last) == 0:
                                site_car.append(frame_idx)
                                site_last = site_car
                            else:
                                IOU = get_IOU_xyxy(site_last, site_car)
                                site_car.append(frame_idx)
                                print("----------------------------------------------")
                                print(site_last[4],end=" ")
                                print(site_car[4],end=" ")
                                site_last = site_car
                                print(IOU)
                    else:
                        car_and_licenses = None
                    current_frame = {}
                    current_frame['time'] = datetime.now()
                    current_frame['frame'] = frame_idx
                    current_frame['n_vehicles_at_time'] = len(outputs)    
                    current_frame['IDs_vehicles'] = []
                    current_frame['result'] = car_and_licenses               

                    if len(outputs)>0:
                        current_frame['IDs_vehicles'] = list(outputs[:, 4])
                        # current_frame['bb_vehicles'] = list(outputs[:, :4])
                    
                    if (current_frame != -1) and (previous_frame != -1):
                        previous_IDs = previous_frame['IDs_vehicles']
                        current_IDs = current_frame['IDs_vehicles']
    
                        for ID in current_IDs:
                            # neu id khong co trong khung hinh truoc va chua tung xuat hien
                            if (ID not in previous_IDs) and (ID not in list_vehicles):
                                vehicle_infos[ID] = {}
                                vehicle_infos[ID]['in_time'] = datetime.now()
                                vehicle_infos[ID]['exit_time'] = datetime.max
                                vehicle_infos[ID]['type_vehicle'] = 'vehicle' 
                                vehicle_infos[ID]['lane'] = 'lane'                     
                                vehicle_infos[ID]['temporarily_disappear'] = 0

                                
                        # for ID in previous_IDs:
                        for ID in copy.deepcopy(list_vehicles):
                            if (ID not in current_IDs):
                                vehicle_infos[ID]['exit_time'] = datetime.now()
                                vehicle_infos[ID]['temporarily_disappear'] += 1
                                #25 frame ~ 1 seconds
                                if (vehicle_infos[ID]['temporarily_disappear'] > self.exit_frames) and \
                                    (vehicle_infos[ID]['exit_time'] - vehicle_infos[ID]['in_time']) > timedelta(seconds= self.exit_frames / fps): # 15 frame/s 3060 GPU

                                    str_ID = str(ID) + "-" +str(time.time()).replace(".", "")     
                                    
                                    list_vehicles.discard(ID)
                                    # vehicle_infos.pop(ID)
                    
                    
                    # Visualize deepsort outputs
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)): 
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]                            
                            c = int(cls)  # integer class
                            # label = f'{id} {names[c]} {conf:.2f}'
                            label = f'{names[c]}- id {id}'
                            
                            bbox_left, bbox_top, bbox_right, bbox_bottom = bboxes 
                            if bbox_right < middle_line:
                                vehicle_infos[id]['lane'] = 'left'
                            if bbox_left > middle_line:
                                vehicle_infos[id]['lane'] = 'right'

                            annotator.box_label(bboxes, label, color=colors(c, True))
                            vehicle_infos[id]['type_vehicle'] = names[c]                            
            

                        vehicles_count, IDs_vehicles = current_frame['n_vehicles_at_time'], current_frame['IDs_vehicles']                            
                        # LOGGER.info("{}: {} vehicles".format(s, vehicles_count))

                        if not np.isnan(np.sum(IDs_vehicles)):
                            list_vehicles.update(list(IDs_vehicles)) 

                else:
                    deepsort.increment_ages()

                # Stream results
                im0 = annotator.result()
                if show_vid:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_vid:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'

                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
            if len(pred) == 0:  # 没检测到物体的帧也不忽略
                if save_vid:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0s.shape[1], im0s.shape[0]
                            save_path += '.mp4'

                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0s)

            previous_frame = current_frame
            frame_list.append(current_frame)
            # if frame_idx > 300:
            #     break
        vid_writer.release()    # 不运行这一句则输出视频未关闭，后面也不能正常使用
        save_list_to_file(frame_list, out_path / 'frame_list.pkl')
        self.generate_results(frame_list, source)


        return  list_vehicles


        # # Print results
        # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    def assign_license_to_vehicle(self, licenses, cars):
        # 思路：遍历车牌，找出距离车牌最近的bbox中心点，若有剩余bbox，按照空牌照处理
        sets = set()
        car_and_licenses = []
        for license in licenses:
            x0 = (license[3][0] + license[3][2]) / 2
            y0 = (license[3][1] + license[3][3]) / 2
            d0 = 1e8
            flag = -1
            for i, car in enumerate(cars):
                x1 = (car[0] + car[2]) / 2
                y1 = (car[1] + car[3] ) / 2
                d1 = math.sqrt((x1 - x0)**2 + (y1 - y0)**2) # 计算车bbox中心点和车牌距离
                if d1 < d0:
                    d0 = d1
                    flag = i
            sets.add(tuple(cars[flag]))
            # 要求车牌必须在车检测框内
            if ((license[3][0]+license[3][2]) / 2 < cars[flag][0] or (license[3][0]+license[3][2]) / 2 > cars[flag][2] ):
                continue
            if ((license[3][1]+license[3][3]) / 2 < cars[flag][1] or (license[3][1]+license[3][3]) / 2 > cars[flag][3] ):
                continue
            car_and_license = []
            car_and_license.append(cars[flag][0:4].tolist())    # 车的位置
            car_and_license.append(license[0])                  # 车牌号
            car_and_license.append(license[3])                  # 车牌位置
            car_and_licenses.append(car_and_license)

        for car in cars:
            if tuple(car) not in sets:
                car_and_license = [car[0:4].copy().tolist()]
                car_and_license.extend(['空车牌'])
                car_and_licenses.append(car_and_license)
        
        
        return car_and_licenses

    def detect_car_license(self, img):
        """
            TODO 车牌检测结果去除重复项
            例如：
            [['贵A1ZM03', 0.9223048, 0, [1441, 1420, 1671, 1502]]]
            [['贵A1ZM03', 0.9999353, 0, [1441, 1420, 1671, 1502]]]
            [['贵A1ZM03', 0.99993336, 0, [1441, 1420, 1671, 1502]]]
            [['贵A1ZM03', 0.9999213, 0, [1440, 1420, 1671, 1502]]]
            [['贵A1ZM03', 0.9999129, 0, [1440, 1420, 1671, 1502]]]
            [['贵A1ZM03', 0.9999221, 0, [1441, 1420, 1670, 1501]]]
            [['贵A1ZM03', 0.99642044, 0, [1440, 1419, 1671, 1501]]]
            [['贵A1ZM03', 0.9983447, 0, [1440, 1419, 1671, 1501]]]
            [['贵A1ZM03', 0.999922, 0, [1440, 1419, 1671, 1501]]]
            [['贵A1ZM03', 0.9999205, 0, [1440, 1419, 1670, 1501]]]
            [['贵A1ZM03', 0.9998342, 0, [1440, 1418, 1671, 1501]]]
            [['贵A1ZM03', 0.9999061, 0, [1440, 1419, 1670, 1502]]]
            [['贵A919BU', 0.9999205, 0, [1440, 1419, 1670, 1501]]]
            [['贵A919BU', 0.9998342, 0, [1440, 1418, 1671, 1501]]]
            [['贵A919BU', 0.9999061, 0, [1440, 1419, 1670, 1502]]] ->  [
                                                                        ['贵A1ZM03', 0.9223048, 0, [1441, 1420, 1671, 1502] 
                                                                        ['贵A919BU', 0.9999205, 0, [1440, 1419, 1670, 1501]
                                                                    ]
        """
        license_pred = self.catcher(img) # [['贵A1ZM03', 0.9376292, 0, [1440, 1419, 1671, 1503]]]
        result = []
        seen_plates = set()

        for item in license_pred:
            plate = item[0]
            if plate not in seen_plates:
                result.append(item)
                seen_plates.add(plate)

        return result


    def generate_results(self, list_vehicles, source):
        """
            TODO  生成： 车牌号码（如是无牌车，采用“无牌车”字符）、车辆状态（驶入/占用/驶出/空位）

        """
        """
        数据形式list_vehicles = 
        [
        [  # 第一帧
            [[1193, 110, 2138, 878], '贵AE9M02', [1424, 706, 1612, 774]], 
            [[21, 1248, 857, 2145], '贵A919BU', [1440, 1419, 1670, 1501]]
        ], 
        [  # 第二帧
            [[1194, 111, 2139, 879], '贵AE9M02', [1425, 706, 1612, 774]],
            [[22, 1249, 858, 2146], '贵A919BU', [1441, 1419, 1670, 1501]]
        ]
        ]
        
        """
        video_path = './output/'+ os.path.basename(source)
        video_capture = cv2.VideoCapture(video_path)
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        video_capture.release()
        out_path = './output/'+ os.path.basename(source)[:-4] + '/'
        f = open(out_path + 'text.txt', "w")
        check = 10  # 被对比的两帧之间相差的帧数
        occupy_area = 0.97  # 判断是否占用时的阈值
        out_area = 0.30  # 判断是否驶出的阈值
        empty_area = 0.05  # 判断是否是空位的阈值

        license_list = []  # 已经出现的车牌列表
        occupy_list = []  # 占用时对应的车牌和car_box
        leaving_list = [] # 进入驶出状态的车的车牌号和占用状态的car_box
        out_list = []   # 进入空位状态的车


        res_pic_tosave = [] # [帧位，车牌号，状态]

        for i in range(check, len(list_vehicles), check):
            if list_vehicles[i-check] == -1 or list_vehicles[i-check]['result'] == None:
                continue
            if list_vehicles[i] == -1 or list_vehicles[i]['result'] == None:
                continue
            # previous_frame = list_vehicles[i - check]
            current_frame = list_vehicles[i]
            for item in current_frame['result']:
                # 驶出和空位
                for box, license in occupy_list:
                    if item[1] == license:
                        iou = get_IOU_xyxy(box, item[0]) 
                        # if iou < empty_area:  # 空位
                        #     if [box, license] not in leaving_list: # 必须先有驶出状态才有空位状态
                        #         continue
                        #     if license in out_list: # 已经有空位状态了
                        #         continue
                        #     out_list.append(license)
                        #     res_pic_tosave.append([i,item[1],'空位'])
                        # elif  iou < out_area:  # 驶出
                        #     if [box, license] in leaving_list:  #正处于驶出状态但又未达到空位状态
                        #         continue
                        #     leaving_list.append([box, license])
                        #     res_pic_tosave.append([i,item[1],'驶出'])
                        if  iou < out_area:  # 驶出和空位
                            if license not in leaving_list:
                                leaving_list.append(license)
                                res_pic_tosave.append([i,item[1],'驶出'])
                                temp = max(i + fps*6+1, len(list_vehicles)-fps)
                                for j in range(i, temp, fps): #在驶出状态的后六秒检测汽车是否达到空位状态
                                    out_frame = list_vehicles[j]
                                    if out_frame == -1 or out_frame['result'] is None:
                                        continue
                                    for out_item in out_frame['result']:
                                        if out_item[1] == license and get_IOU_xyxy(box, out_item[0]) < empty_area:
                                            out_list.append(license)
                                            res_pic_tosave.append([j,item[1],'空位'])
                                    if license in out_list:
                                        break
                                if license not in out_list:
                                    out_list.append([license])
                                    res_pic_tosave.append([temp,item[1],'空位'])
                                
                # 占用
                count = 0
                for j in range(i - 6*check, i + 7*check, check):    # 以前后各6帧来判断
                    if item[1] not in license_list: # 保证在占用前必须有驶入
                        break
                    if any(item[1] == row[1] for row in occupy_list):
                           break
                    if 0 <= j <= len(list_vehicles):
                        nearby_frame = list_vehicles[j]
                        if nearby_frame == -1 or nearby_frame['result'] is None:
                            continue
                        for nearby_item in nearby_frame['result']:
                            if item[1] not in license_list:
                                continue
                            if item[1] == nearby_item[1]:
                                iou = get_IOU_xyxy(nearby_item[0], item[0])
                                relative_dist = get_dist_coefficient(nearby_item[0], item[0])
                                if iou > occupy_area and relative_dist < 0.5:
                                    count += 1
                                    print("rel_dist:", relative_dist)
                                    print("count: ", count,"         ",i)
                if count > 6:
                    occupy_list.append([item[0], item[1]])
                    res_pic_tosave.append([i,item[1],'占用'])

                # 驶入
                if item[1] not in license_list:
                    license_list.append(item[1])
                    # self.save_image(source, i, out_path + str(i)+'_驶入_' + item[1] + '.jpg')
                    # f.write('帧位'+ str(i)+'\tplate:' + item[1] + '\tstatus:驶入' + '\tcaptime:' + str(i/25) + 's\n')
                    res_pic_tosave.append([i,item[1],'驶入'])
        # 后处理
        plates_detect = set([_[1] for _ in res_pic_tosave]) # 检测出的车牌
        plates_output = []  # 筛选后的车牌
        for plate in plates_detect:
            flag=[0,0,0,0]
            for _, plate2, state in res_pic_tosave:
                if plate2 != plate:
                    continue
                elif state=='驶入':
                    flag[0] = 1
                elif state=='占用':
                    flag[1] = 1
                elif state=='驶出':
                    flag[2] = 1
                # else:   # 空位
                #     flag[3] = 1
            if sum(flag) >= 3:
                plates_output.append(plate)
        print('初步检测出的车:')
        print(plates_detect)
        print('最终选定输出的车:')
        print(plates_output)
        for frame, plate, state in res_pic_tosave:
            if plate not in plates_output:
                continue
            else:
                f.write('帧位'+str(frame)+'\tplate:'+plate+'\tstatus:'+state+'\tcaptime:'+str(frame/fps) + 's\n')
                self.save_image(video_path, frame, out_path+str(frame)+'_'+state+'_'+plate+'.jpg')            
        f.close()
        return None

    def save_image(self, source, frame_number, output_path):
        # 打开视频文件
        video = cv2.VideoCapture(source)
        if not video.isOpened():
            print("无法打开视频文件")
            return

        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # 设置帧位置
        ret, frame = video.read()  # 读取指定帧
        if not ret:
            print("无法读取帧")
            return

        cv2.imencode('.jpg', frame)[1].tofile(output_path)  # 保存帧为图片
        video.release()  # 关闭视频文件
        return None

if __name__ == '__main__':

    tracker = Tracker(config_path='task_02/Vehicle-tracking/settings/config.yml')
    
    with torch.no_grad():
        # list_vehicles =  tracker.detect()
        dataset_path = "dataset/2"
        file_paths = glob.glob(os.path.join(dataset_path, "*"))
        for video_path in file_paths:
            print("*********  ",video_path,"  **********")
            list_vehicles =  tracker.detect(video_path)
"""
0:'dataset/2\\H3V-贵A919BU.avi'
1:'dataset/2\\H4V-贵A1ZM03.avi'
2:'dataset/2\\H5V-贵AM6Y69.avi'
3:'dataset/2\\H6V-贵A9QJ51.avi'
4:'dataset/2\\H8V-陕AE8R40.avi'
5:'dataset/2\\S10V-粤HBZ093.avi'
6:'dataset/2\\S1V-京N7SG10.avi'
7:'dataset/2\\S4V-粤B0L1E6.avi'
8:'dataset/2\\S5V-京QWA550.avi'
9:'dataset/2\\S7V-粤TA733W.avi'
"""
