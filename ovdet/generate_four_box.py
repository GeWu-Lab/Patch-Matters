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
import true_box_sample
import argparse


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_torch(0)




if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--model_config_file', type=str, default='ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py')
    arg.add_argument('--checkpoint_file', type=str, default='ovdet/checkpoints/this_repo_R-50-FPN_CLIP_iter_90000.pth')
    arg.add_argument('--image_folder', type=str, help='Image folder')
    arg.add_argument('--four_box_save_path', type=str, help='four box save path')
    arg.add_argument('--object_box_save_path', type=str, help='object detect save path')
    args = arg.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    inference = DetInferencer(model=args.model_config_file, weights=args.checkpoint_file, device='cuda:1')


    image_files = [os.path.join(arg.image_folder, f) for f in os.listdir(arg.image_folder) if f.lower().endswith(('.png', '.jpg'))]

    base_final_scores = 0.3
    i = 0

    final_result = []
    obj_result = []
    for img in tqdm(image_files):
        
        print('image path is doing: ', img)
        res = inference(img)
        # print(res)
        if len(res['predictions'][0]['scores']) == 0:
            image = Image.open(img)
            width, height = image.size
            res = {
                'predictions': [
                    {
                        'bboxes': torch.tensor([[0.0, 0.0, width - 1.0, height - 1.0]]),
                        'scores': torch.tensor([1.0]),
                        'labels': torch.tensor([1008611])
                    }
                ]
            }
        
        data_result, obj = true_box_sample.sample_rpn_output(img, res, base_final_scores=base_final_scores)
        final_result.append(data_result)
        obj_result.append(obj)

    with open(arg.four_box_save_path, 'w', encoding='utf-8') as file:
        json.dump(final_result, file, ensure_ascii=False, indent=4)

    # if arg.object_box_save_path is not None:
    obj_result = {item['image']: item for item in obj_result}
    with open(arg.object_box_save_path, 'w', encoding='utf-8') as file:
        json.dump(obj_result, file, ensure_ascii=False, indent=4)