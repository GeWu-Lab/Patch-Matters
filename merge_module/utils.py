from PIL import Image
from shapely.geometry import box

class ImageProcessor:
    def __init__(self):
        # 加载图像，并获得图像的宽度和高度
        # self.image = Image.open(image_path).convert('RGB')
        # self.width, self.height = self.image.size
        self.image=1
    def crop_region(self, image,region_box):
        """
        裁剪图像的特定区域
        :param region_box: [x_min, y_min, x_max, y_max] 
        :return: 裁剪后的图像
        """
        return image.crop(region_box)
    def normalize_box(self, box,img_width,img_height):
        """
        对边界框进行归一化
        :param box: [x_min, y_min, x_max, y_max]
        :return: 归一化后的边界框
        """
        x_min, y_min, x_max, y_max = box
        # 归一化边界框
        x_min_normalized = x_min / img_width
        y_min_normalized = y_min / img_height
        x_max_normalized = x_max / img_width
        y_max_normalized = y_max / img_height
    
    # 返回归一化后的边界框
        return [x_min_normalized, y_min_normalized, x_max_normalized, y_max_normalized]

    def get_size(self):
        """
        获取图像的尺寸
        :return: 图像的宽度和高度
        """
        return self.width, self.height

class BoundingBoxOperations:
    @staticmethod
    def calculate_iou(rectangles):
        iou_values = []

        # 分别计算左上和右上、左下和右下的区域与相邻区域的IoU
        # 左上 (Region 1) 和右上 (Region 2)
        intersection_area = rectangles[0].intersection(rectangles[1]).area
        union_area = rectangles[0].area + rectangles[1].area - intersection_area
        iou = intersection_area / union_area
        iou_values.append(('Left Top (Region 1) and Right Top (Region 2)', iou))

        # 右上 (Region 2) 和右下 (Region 4)
        intersection_area = rectangles[1].intersection(rectangles[3]).area
        union_area = rectangles[1].area + rectangles[3].area - intersection_area
        iou = intersection_area / union_area
        iou_values.append(('Right Top (Region 2) and Right Bottom (Region 4)', iou))

        # 右下 (Region 4) 和左下 (Region 3)
        intersection_area = rectangles[3].intersection(rectangles[2]).area
        union_area = rectangles[3].area + rectangles[2].area - intersection_area
        iou = intersection_area / union_area
        iou_values.append(('Right Bottom (Region 4) and Left Bottom (Region 3)', iou))

        # 左下 (Region 3) 和左上 (Region 1)
        intersection_area = rectangles[2].intersection(rectangles[0]).area
        union_area = rectangles[2].area + rectangles[0].area - intersection_area
        iou = intersection_area / union_area
        iou_values.append(('Left Bottom (Region 3) and Left Top (Region 1)', iou))
        
        intersection_area = rectangles[4].intersection(rectangles[5]).area
        union_area = rectangles[4].area + rectangles[5].area - intersection_area
        iou = intersection_area / union_area
        iou_main=[]
        iou_main.append(('global (Region 5) and main (Region 0)', iou))
        return iou_values,iou_main

    @staticmethod
    def merge_boxes(box1, box2):
        """
        合并两个边界框，返回一个新的边界框
        :param box1: 第一个边界框
        :param box2: 第二个边界框
        :return: 合并后的边界框 [x_min, y_min, x_max, y_max]
        """
        x_min = min(box1[0], box2[0])
        y_min = min(box1[1], box2[1])
        x_max = max(box1[2], box2[2])
        y_max = max(box1[3], box2[3])
        return [x_min, y_min, x_max, y_max]
