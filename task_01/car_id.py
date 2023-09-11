import cv2
import regex
import hyperlpr3 as lpr3
from tqdm import tqdm
from pathlib import Path
from more_itertools import first_true

pattern_str = '([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领]{1}[A-Z]{1}(([A-HJ-NP-Z0-9]{5}[DF]{1})|([DF]{1}[A-HJ-NP-Z0-9]{5})))|([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领]{1}[A-Z]{1}[A-HJ-NP-Z0-9]{4}[A-HJ-NP-Z0-9挂学警港澳]{1})'

def compute_acc(predcition, label):
    """
        计算识别准确率，并返回识别错误的结果
    """
    sum = 0.
    correct = 0.
    error_list = []
    for index, pred in enumerate(predcition):
        if pred == label[index]:
            correct += 1
        else:
            error_list.append(label[index])
        sum +=1
    return correct / sum,  error_list

def get_label(img_path, pattern_str):
    """
        get car id from file name
    """
    label = regex.findall(pattern_str, str(img_path))
    if not label: return '';
    while type(label) != str:
        label = first_true(label)
    return label

def get_prediction(img_data, catcher):
    prediction = catcher(img_data)
    if not prediction:
        return None
    else:
        while type(prediction) != str:
            prediction = first_true(prediction)
        return prediction



if __name__ == "__main__":

    data_root = '/home/hyq/code/comp/dataset/base'
    prediction_all = []
    label_all = []
    # 初始化识别模型
    catcher = lpr3.LicensePlateCatcher(detect_level=1)

    for img_path in tqdm(
            Path(data_root).rglob('*.jpg')
        ):
        label = get_label(
            img_path=img_path, 
            pattern_str=pattern_str
        )
        img_data = cv2.imread(str(img_path))
        prediction = get_prediction(img_data=img_data, catcher=catcher)

        prediction_all.append(prediction)
        label_all.append(label)

    acc, error_list = compute_acc(predcition=prediction_all, label=label_all)
    print(acc) # 0.953
    print(error_list) 



