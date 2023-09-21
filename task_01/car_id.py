import cv2
import numpy as np
import regex
import hyperlpr3 as lpr3
from tqdm import tqdm
from pathlib import Path
from more_itertools import first_true
from PIL import ImageFont, ImageDraw, Image

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
        sum += 1
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


def get_prediction(image_data, img_data, catcher):
    prediction = catcher(img_data)
    if not prediction:
        return None
    else:
        while type(prediction) != str:
            box = prediction[-1]
            prediction = first_true(prediction)
        prediction = post_process(img_data, prediction)
        save_image(image_data, img_data, prediction, box)
        return prediction


def post_process(img_data, prediction):
    """
        后处理流程
    """
    if len(prediction) > 7:
        gray_img = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        prediction = catcher(gray_img)
        if not prediction:
            return None
        else:
            while type(prediction) != str:
                prediction = first_true(prediction)
    if len(prediction) > 8:
        count = 0
        index = -1
        for char in prediction:
            if count >= 2:
                break
            if ord(char) > 127:
                count += 1
            index += 1
        prediction = prediction[:index]
    return prediction


def save_image(image_data, img_data, prediction, box):
    out_path = 'output/'
    Path(out_path).mkdir(parents=True, exist_ok=True)
    if prediction:
        image_name = out_path + '_' + prediction + '_' + image_data.name

        size = 50
        img = Image.fromarray(img_data)
        font = ImageFont.truetype("simfang.ttf", size, encoding="utf-8")
        draw = ImageDraw.Draw(img)
        width = draw.textlength(prediction, font=font)
        draw.rectangle(((box[0], box[1] - size), (box[0] + width, box[1])), fill=(0, 0, 0))
        draw.text((box[0], box[1] - size), prediction, font=font, fill=(255, 255, 255))
        img1 = np.array(img)
        cv2.imencode('.jpg', img1)[1].tofile(image_name)

    return None


if __name__ == "__main__":

    data_root = '../../dataset/base/'
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
        file = str(img_path).encode('utf-8')
        img_data = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
        prediction = get_prediction(image_data=img_path, img_data=img_data, catcher=catcher)

        prediction_all.append(prediction)
        label_all.append(label)

    acc, error_list = compute_acc(predcition=prediction_all, label=label_all)
    print(acc)  # 0.953
    print(error_list) 
