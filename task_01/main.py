import cv2
import numpy as np
import regex
import hyperlpr3 as lpr3
from tqdm import tqdm
from pathlib import Path
from more_itertools import first_true
from PIL import ImageFont, ImageDraw, Image


class CarLicenseDetector(object):

    def __init__(self) -> None:
        self.catcher = lpr3.LicensePlateCatcher(detect_level=1)

    def get_prediction(self, image_data, img_data):
        prediction = self.catcher(img_data)
        if not prediction:
            return None
        else:
            while type(prediction) != str:
                box = prediction[-1]
                prediction = first_true(prediction)
            prediction = self.post_process(img_data, prediction)
            self.save_image(image_data, img_data, prediction, box)
            return prediction

    def preprocess(self, img_data):
        """
            TODO: 预处理阶段解决已知问题
                1. 输入图像亮度调节， 如 /home/hyq/code/comp/dataset/base/蓝牌/_蓝牌_京NFJ582_201507271606308361.jpg
                2. 多种数据增强方式，参考 https://albumentations.ai/
                3. 斜向图片的数据处理变换, 如图片
        """
        return img_data

    def detect_single(self, image_data):
        """
            Task 1 提供的数据集中大部分图像仅包含单一车辆车牌，因此结果中总返回置信度最高结果
        """

        if type(image_data) == str:
            file = image_data.encode('utf-8')
            img_data = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
        elif type(image_data) == cv2.Mat:
            img_data = image_data
        else:
            raise TypeError(f'Input type of {type(image_data)} not support')

        img_data = self.preprocess(img_data)
        prediction = self.get_prediction(image_data, img_data)

        return prediction

    def post_process(self, img_data, prediction):
        """
            后处理流程
        """
        if len(prediction) > 7:
            gray_img = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            prediction = self.catcher(gray_img)
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

    def save_image(self, image_data, img_data, prediction, box):
        out_path = 'output/'
        Path(out_path).mkdir(parents=True, exist_ok=True)
        if prediction:
            image_name = out_path + '_' + prediction + '_' + image_data.split('/')[-1]

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


if __name__ == '__main__':
    detector = CarLicenseDetector()
    result = detector.detect_single('../../dataset/base/单黄/_单行黄牌_y_鄂H0C650_2018032311021843417605_.jpg')
    print(result)
