import cv2
import regex
import hyperlpr3 as lpr3
from tqdm import tqdm
from pathlib import Path
from more_itertools import first_true


class CarLicenseDetector(object):

    def __init__(self) -> None:
        self.catcher = lpr3.LicensePlateCatcher(detect_level=1)

    def get_prediction(self, img_data):
        prediction = self.catcher(img_data)
        if not prediction:
            return None
        else:
            while type(prediction) != str:
                prediction = first_true(prediction)

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
            img_data = cv2.imread(image_data)
        elif type(img_data) == cv2.Mat:
            img_data = image_data
        else: 
            raise TypeError(f'Input type of {type(img_data)} not support')
        
        img_data = self.preprocess(img_data)
        prediction = self.get_prediction(img_data=img_data)
        prediction = self.post_process(prediction)

        return prediction
    
    def post_process(self, prediction):
        """
            后处理流程
        """
        return prediction
    
if __name__ == '__main__':

    detector = CarLicenseDetector()
    result = detector.detect_single('/home/hyq/code/comp/dataset/base/单黄/_单行黄牌_豫P95085_201803221743463736022_.jpg')
    print(result)