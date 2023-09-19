import yaml
import numpy as np
from typing import Union
import pandas as pd
import math
import pickle

def str_array (array: np.ndarray):
  output_string = ""
  for i in range(len(array)):
    output_string = output_string + str(list(array[i]))+","
  return output_string[:-1]

def str_to_list(str_list):
  result = str_list.split(" ")
  result[0] = result[0][1:]
  result[-1] = result[-1][:-1]
  result = list(map(int, result))
  return result


def read_yml(path:str):
  with open(path, 'r', encoding='utf-8') as file:
      text = yaml.safe_load(file)
  return text

def update_config(config_path:str, new_config:dict):
   config_text = read_yml(config_path)
   for setting in new_config.keys():
      config_text[setting] = new_config[setting]
   with open(config_path, "w") as f:
         yaml.dump(config_text, f)

def extract_xywh_hog(face):
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y
    return (x, y, w, h)

def read_db_config(config_path:str):
  config = read_yml(config_path)
  dialect = config['dialect']
  driver = config['driver']
  user = config['user']
  password = config['password']
  host = config['host']
  port = config['port']
  database = config['database']
  connection_string = f"{dialect}+{driver}://{user}:{password}@{host}:{port}/{database}"
  return connection_string

def write_csv(csv_path:str, list_ouputs:dict, list_frontal_faces:dict, fps:int) -> None:

  for fi in list(list_ouputs.keys()):
    if fi not in list(list_frontal_faces.keys()):
        list_frontal_faces[fi] = np.asarray([])

  fi_list = list(list_ouputs.keys())
  fi_list.sort()
  with open(csv_path, 'a') as f:
      for fi in fi_list:
          pp_component = list_ouputs[fi]
          face_component = list_frontal_faces[fi]

          pp_count = len(pp_component)
          face_count = len(face_component)

          if len(pp_component)>0:
              IDs_pp = list(pp_component[:,4])
              bb_pp = pp_component[:,:4]
          else:
              IDs_pp = ""
              bb_pp = ""

          if len(face_component)>0:
              IDs_face = list(face_component[:,4])      
              bb_face = face_component[:,:4]
          else:
              IDs_face = ""
              bb_face = ""

          f.write(str(fi/fps).replace(".", ":")+";")
          f.write(str(pp_count)+";")
          f.write(str(face_count)+";")
          f.write(str(IDs_pp)+";")
          f.write(str(IDs_face)+";")
          f.write(str_array(bb_pp)+";")
          f.write(str_array(bb_face)+"\n")

def get_IOU_xyxy(site1, site2):
    # 计算框一的面积
    box1_area = (site1[2] - site1[0]) * (site1[3] - site1[1])
    # 计算框二的面积
    box2_area = (site2[2] - site2[0]) * (site2[3] - site2[1])
    # 计算交集框的左上角和右下角坐标
    intersection_left = max(site1[0], site2[0])
    intersection_top = max(site1[1], site2[1])
    intersection_right = min(site1[2], site2[2])
    intersection_bottom = min(site1[3], site2[3])
    # 计算交集框的面积
    intersection_area = max(0, intersection_right - intersection_left) * max(0, intersection_bottom - intersection_top)
    # 计算并集框的面积
    union_area = box1_area + box2_area - intersection_area
    # 计算IoU
    iou = intersection_area / union_area
    # 打印IoU
    return iou

def get_dist_coefficient(site1, site2):
   x1 = (site1[0] + site1[2]) / 2
   y1 = (site1[1] + site1[3]) / 2
   x2 = (site2[0] + site2[2]) / 2
   y2 = (site2[1] + site2[3]) / 2
   dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
   box1_area = (site1[2] - site1[0]) * (site1[3] - site1[1])
   box2_area = (site2[2] - site2[0]) * (site2[3] - site2[1])
   len_aver = math.sqrt((box1_area + box2_area)/2)
   return dist / len_aver

def save_list_to_file(data_list, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data_list, file)

def load_list_from_file(file_path):
    with open(file_path, 'rb') as file:
        data_list = pickle.load(file)
        return data_list