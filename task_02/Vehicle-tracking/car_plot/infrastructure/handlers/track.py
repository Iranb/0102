# limit the number of cpus used by high performance libraries

import os
import warnings
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
# sys.path.insert(0, './yolov5')
lib_path = os.path.abspath(os.path.join('infrastructure', 'yolov5'))
sys.path.append(lib_path)

from infrastructure.yolov5.models.experimental import attempt_load
from infrastructure.yolov5.utils.downloads import attempt_download
from infrastructure.yolov5.models.common import DetectMultiBackend
from infrastructure.yolov5.utils.datasets import LoadImages, LoadStreams
from infrastructure.yolov5.utils.general import LOGGER, check_img_size, increment_path, non_max_suppression, scale_coords, check_imshow, xyxy2xywh, increment_path
from infrastructure.yolov5.utils.torch_utils import select_device, time_sync
from infrastructure.yolov5.utils.plots import Annotator, colors
from infrastructure.deep_sort_pytorch.utils.parser import get_config
from infrastructure.deep_sort_pytorch.deep_sort import DeepSort

from util.common import  read_yml, extract_xywh_hog
from util.OPT_config import OPT
from infrastructure.helper.zone_drawer_helper import ZoneDrawerHelper
from threading import Thread
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

    # TODO 这里source 路径需要修改
    def detect(self, source='/home/hyq/code/comp/dataset/prestige/H3V-贵A919BU.avi'):
        opt = self.opt
        out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, save_csv, imgsz, evaluate, half = \
            opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
                opt.save_txt, opt.save_csv, opt.imgsz, opt.evaluate, opt.half
        zone_drawer = ZoneDrawerHelper()
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

        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs


        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        save_path = str(Path(out))
        # extract what is in between the last '/' and last '.'
        txt_file_name = source.split('/')[-1].split('.')[0]
        txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'
        csv_path = str(Path(out)) + '/' + txt_file_name + '.csv'        
        

        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0

        previous_frame, current_frame = [-1, -1]
        vehicle_infos = {} # id:{start in view, exit view, type }
        list_vehicles = set()  #LIST CONTAIN vehicles HAS APPEARED, IF THAT VEHICLE HAD BEEN UPLOADED TO DB, REMOVE THAT VEHICLE
        

        for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
            t1 = time_sync()
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            frame_height = im0s.shape[0]
            frame_width = im0s.shape[1]
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
                    print(outputs)
                    # NOTE 检测车牌信息
                    licenses = self.detect_car_license(im0s)
                    print(licenses)

                    current_frame = {}
                    current_frame['time'] = datetime.now()
                    current_frame['frame'] = frame_idx
                    current_frame['n_vehicles_at_time'] = len(outputs)    
                    current_frame['IDs_vehicles'] = []                    
                   
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
                                    (vehicle_infos[ID]['exit_time'] - vehicle_infos[ID]['in_time']) > timedelta(seconds= self.exit_frames / 25): # 15 frame/s 3060 GPU

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
                        LOGGER.info("{}: {} vehicles".format(s, vehicles_count))

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

            previous_frame = current_frame

        print(vehicle_infos)
        print(list_vehicles)

        return  list_vehicles


        # # Print results
        # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    
    def assign_license_to_vehicle(self, license, cars):
        """
            TODO
        """

        return None

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
        return license_pred

    def generate_results(self, list_vehicles):
        """
            TODO  生成： 车牌号码（如是无牌车，采用“无牌车”字符）、车辆状态（驶入/占用/驶出/空位）

        """
        return None

if __name__ == '__main__':

    tracker = Tracker(config_path='../settings/config.yml')
    
    with torch.no_grad():
       list_vehicles =  tracker.detect()

