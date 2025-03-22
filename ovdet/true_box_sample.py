from mmdet.apis import init_detector, inference_detector, DetInferencer
from mmdet.registry import VISUALIZERS
import cv2, sys, pickle, os, json
import mmcv, torch, random
from mmdet.models.detectors.two_stage import TwoStageDetector
from rich.pretty import pprint
from mmdet.registry import MODELS
import sample_tools
import ovdet
import numpy as np
from PIL import Image, ImageDraw, ImageColor
from tqdm import tqdm
def classes():
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
        'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
    ]

def sample_rpn_output(img, res, base_final_scores=0.3):
    image = Image.open(img)
    width, height = image.size
    image_box = torch.tensor([0.0, 0.0, width - 1.0, height - 1.0])

    # print('detect nums: ', len(res['predictions'][0]['scores']))
    scores_tensor = torch.tensor(res['predictions'][0]['scores'])
    boxes_tensor = torch.tensor(res['predictions'][0]['bboxes'])
    labels_tensor = torch.tensor(res['predictions'][0]['labels'])
    indices = scores_tensor > base_final_scores
    if indices.sum() == 0 and len(scores_tensor) >= 2:
        indices[:2] = True

    # print('boxes_tensor: ', boxes_tensor)
    # print('scores: ', scores_tensor)
    # print('indices: ', indices)

    res['predictions'][0]['bboxes'] = boxes_tensor[indices]
    res['predictions'][0]['scores'] = scores_tensor[indices]
    res['predictions'][0]['labels'] = labels_tensor[indices]

    if len(res['predictions'][0]['bboxes'])==0:
        res = {
            'predictions': [
                {
                    'bboxes': torch.tensor([[0.0, 0.0, width - 1.0, height - 1.0]]),
                    'scores': torch.tensor([1.0]),
                    'labels': torch.tensor([1008611])
                }
            ]
        }        
    
    # print('***********************************: ', res['predictions'][0]['labels'].tolist())
    class_name = classes()
    obj = {
        'image': img,
        'bbox': res['predictions'][0]['bboxes'].tolist(),
        'name': [class_name[i] for i in res['predictions'][0]['labels'].tolist() if i != 1008611],
        'label': res['predictions'][0]['labels'].tolist()
    }
    if len(obj['name']) == 0:
        obj['name'] = ['None']
    obj_dict = obj


    # nms_proposal, proposal_scores = sample_tools.preprocess_proposals(res_boxes, res_scores, image_box[None],
    #                                                                   shape_ratio_thr=0.25, area_ratio_thr=0.01,
    #                                                                   objectness_thr=0.85, nms_thr=0.1
    #                                                                   )
    # nms_proposal = nms_proposal.tolist()

    nms_proposal_before = res['predictions'][0]['bboxes'].tolist()
    nms_proposal = sample_tools.neighbor_rpn_merge(nms_proposal_before, iou_threshold=0.3) 
    # print('nms_proposal', nms_proposal)
    # print(f'nms_proposal len: {len(nms_proposal)}, nms_proposal_before: {len(nms_proposal_before)}')
    # print('nms_proposal_before: ', nms_proposal_before)



    checkboard_sampling = sample_tools.NeighborhoodSampling(
        max_groups=3,
        max_permutations=2,
        alpha=3.0,
        cut_off_thr=0.3,
        base_probability=0.3,
        interval=-0.1
    )

    groups_per_proposal, normed_boxes, spanned_boxes, box_ids = \
        sample_tools.multi_apply(checkboard_sampling.sample, nms_proposal,
                                 [(height, width)] * len(nms_proposal))  # can be time-consuming

    new_boxes = torch.cat([perm for single_proposal in groups_per_proposal
                           for single_group in single_proposal for perm in single_group], dim=0)

    # print('**' * 20)
    # print('new_len: ', len(new_boxes))
    # print('spanned_boxes', spanned_boxes)

    def merge_box(boxes):
        max_area = 0
        max_box = None
        x1_min = float('inf')
        y1_min = float('inf')
        x2_max = -float('inf')
        y2_max = -float('inf')
        for box in boxes:

        #     area = (box[2] - box[0]) * (box[3] - box[1])
        #     if area > max_area:
        #         max_area = area
        #         max_box = box
        # new_span_box = max_box
            x1_min = min(x1_min, box[0])
            y1_min = min(y1_min, box[1])
            x2_max = max(x2_max, box[2])
            y2_max = max(y2_max, box[3])
        new_span_box = [float(x1_min), float(y1_min), float(x2_max), float(y2_max)]
        return new_span_box

    def image_to_n_part(img, n):
        image = Image.open(img)
        W, H = image.size
        quarter_area = W *  H
        if n == 4:
            regions = [
                [0, 0, W // 2, H // 2], 
                [W // 2, 0, W, H // 2],  
                [0, H // 2, W // 2, H],  
                [W // 2, H // 2, W, H] 
            ]

        if n == 2:
            regions = [
                [0, 0, W, H // 2], 
                [0, H // 2, W, H], 
                [0, 0, W // 2, H], 
                [W // 2, 0, W, H] 
            ]
        return regions, quarter_area, W, H

    def proposal_to_n_part(img, new_span_box_list, n, rate=0.3):
        four_part, image_area, W, H= image_to_n_part(img, n)


        # for box in new_span_box_list:
        #     area = sample_tools.calculate_area(box)
        #     if area / (image_area / n) > 1.2:
        #         new_span_box_list.remove(box)

        rate = rate
        new = []
        four = [[], [], [], []] 
        for i in range(4):
            four[i].append(four_part[i])

        if n == 2:
            w_h_num = 0
            result = -1 if W / H > 1 else 1
            for box in new_span_box_list:
                if sample_tools.calculate_W_H_ratio(box) > 1:
                    w_h_num += 1
            if (w_h_num + result) > (len(new_span_box_list) // 2):
                four_part = four_part[:2]
                four = four[:2]
            else:
                four_part = four_part[2:]
                four = four[2:]


        for box in new_span_box_list:
            areas = []
            for box_part in four_part:
                area = sample_tools.intersection_area(box, box_part)
                areas.append(area)
            # max_index = areas.index(max(areas))
            # four[max_index].append(box)
            indices = [index for index, value in enumerate(areas) if value > 0.3] 
            for index in indices:
                four[index].append(box)


        for i, box in enumerate(four):

            merge_s = merge_box(box)
            new.append(merge_s)
        return new

        # for box_part in four_part:
        #     box_part_list = []
        #     for box in new_span_box_list:
        #         if sample_tools.intersection_area(box, box_part) > rate:
        #             box_part_list.append(box)
        #     if len(box_part_list) > 1:
        #         merge_s = merge_box(box_part_list)
        #         new.append(merge_s)
        #     else:
        #         new.append(box_part)
        # # new_span_box_list = new
        # return new

    def draw(img, new_span_box_list, color_list):
        i = 0
        image = cv2.imread(img)
        for new_span_box in new_span_box_list:
            if len(new_span_box) == 0:
                continue
            color = random.choice(color_list)
            bgr_color = (color[2], color[1], color[0])
            cv2.rectangle(image, (int(new_span_box[0]), int(new_span_box[1])), (int(new_span_box[2]), int(new_span_box[3])),
                          bgr_color, 4 + i)
            i += 1
        return image

    def boxes_progress(img, boxes):
        # image = cv2.imread(img)
        new_span_box_list = boxes

        # print('new_span_box_list:', new_span_box_list)
        new_span_box_list_4 = proposal_to_n_part(img, new_span_box_list, 4, rate=0.3)
        # print('box_{}_part'.format(4), new_span_box_list_4)
        new_span_box_list_2 = proposal_to_n_part(img, new_span_box_list, 2, rate=0.3)
        # print('box_{}_part'.format(2), new_span_box_list_2)

        # iou_matrix, areas = sample_tools.calculate_iou(new_span_box_list)
        # high_iou_indices = np.where(iou_matrix >= 0.9)
        # if len(high_iou_indices[0]) > 0:
        #     indices_to_remove = high_iou_indices[0][areas[high_iou_indices[0]] > areas[high_iou_indices[1]]]
        #
        #     keep_mask = np.ones(len(new_span_box_list), dtype=bool)
        #     keep_mask[indices_to_remove] = False
        #     indices_to_keep = np.where(keep_mask)[0].tolist()
        #     new_span_box_list = [new_span_box_list[i] for i in indices_to_keep]
        return new_span_box_list_2, new_span_box_list_4

    new_span_box_list = []
    for box_group in spanned_boxes:  
        new_span_box = merge_box(box_group)
        new_span_box_list.append(new_span_box)

    new_span_box_list = nms_proposal.tolist() 
    # new_span_box_list = nms_proposal_before.tolist()
    new_span_box_list_2, new_span_box_list_4 = boxes_progress(img, new_span_box_list)

    drawn_image = draw(img, new_span_box_list_2, sample_tools.color_list)
    cv2.imwrite('sample_rpn_image/{}'.format(
        img.split('/')[-1].replace('.png', '_2_box.png').replace('.jpg', '_2_box.jpg')), drawn_image)



    drawn_image = draw(img, new_span_box_list_4, sample_tools.color_list)
    cv2.imwrite('sample_rpn_image/{}'.format(
        img.split('/')[-1].replace('.png', '_4_box.png').replace('.jpg', '_4_box.jpg')), drawn_image)

    # drawn_obj = draw(img, nms_proposal, sample_tools.color_list)
    # cv2.imwrite('sample_rpn_image/{}'.format(
    #     img.split('/')[-1].replace('.png', '_nms_rpn.png').replace('.jpg', '_nms_rpn.jpg')), drawn_obj)
    img = img.split('/')[-2] + '/' + img.split('/')[-1]
    data_to_save = {
        'image': img,
        'object_detect_box': nms_proposal.tolist(),
        'four_box': new_span_box_list_4,
        'two_box': new_span_box_list_2,
    }

    return data_to_save, obj_dict


