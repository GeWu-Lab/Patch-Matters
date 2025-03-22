from transformers import AutoProcessor, AutoModelForPreTraining, LlavaForConditionalGeneration,Qwen2VLForConditionalGeneration
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys, re, torch
import transformers
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 
from qwen_vl_utils import process_vision_info
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
from tqdm import tqdm
sys.path.insert(0, 'submodule/CenterNet2')
sys.path.insert(0, 'submodule/detectron2')
sys.path.insert(0, 'submodule/')
# folder2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '/data/users/ruotian_peng/LLaVA-main/llava/serve'))

# # 将 folder2 添加到 sys.path
# sys.path.insert(0, folder2_path)
# from image_caption import ImageCaptionGenerator
# from detectron2.config import get_cfg
# from detectron2.data.detection_utils import read_image
# from detectron2.utils.logger import setup_logger

import json
# with open("/data/users/ruotian_peng/LLaVA-main/playground/data/sharegpt/project-1-at-2024-08-11-11-24-e46761ed.json",'r',encoding='utf-8') as load_f:
#     load_dict = json.load(load_f)
from visual_descriptor import VisualDescriptor
from llama_infer import Visualcheck


from icecream import ic
from PIL import Image
WINDOW_NAME = "LLMScore(BLIPv2+GRiT+GPT-4)"

import pyramid_global_caption as pgc

# def setup_cfg(args):
#     cfg = get_cfg()
#     if args.cpu:
#         cfg.MODEL.DEVICE="cpu"
#     # add_centernet_config(cfg)
#     # add_grit_config(cfg)
#     cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     # Set score_threshold for builtin llm_descriptor
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
#     cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
#     if args.test_task:
#         cfg.MODEL.TEST_TASK = args.test_task
#     cfg.MODEL.BEAM_SIZE = 1
#     cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
#     cfg.USE_ACT_CHECKPOINT = False
#     cfg.freeze()
#     return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="submodule/grit/configs/GRiT_B_DenseCap_ObjectDet.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument(
        "--image",
        default="/data/users/ruotian_peng/LLaVA-main/playground/data/sharegpt/figure4.png",
    )
    parser.add_argument(
        "--llm_id",
        default="llama2_7b_chat",
    )
    parser.add_argument(
        "--text_prompt",
        default="a red car and a white sheep",
        help="text prompt",
    )
    parser.add_argument(
        "--output",
        default="sample/sample_result.png",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--test-task",
        type=str,
        default='DenseCap',
        help="Choose a task to have GRiT perform",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "models/grit_b_densecap_objectdet.pth"],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    # setup_logger(name="fvcore")
    # logger = setup_logger()
    # logger.info("Arguments: " + str(args))

    # cfg = setup_cfg(args)
    # demo = VisualizationDemo(cfg)
    # openai_key = os.environ['OPENAI_KEY']
    # openai_key = 'EMPTY'
    # llama2_7b_chat_hf="/data/users/ruotian_peng/pretrain/llama-3.1-8b-instruct"
    # pipeline = transformers.pipeline(
    #         "text-generation",
    #         model=llama2_7b_chat_hf,
    #         model_kwargs={"torch_dtype": torch.float16},
    #         device_map="auto",
    #     )
    
    # processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf",
    #                                                            torch_dtype=torch.float16)  # torch.float16
    # model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",
    #                                                            torch_dtype=torch.float16,
    #                                                   low_cpu_mem_usage=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/data/users/ruotian_peng/pretrain/qwen27b", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("/data/users/ruotian_peng/pretrain/qwen27b")
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)
    # generator = ImageCaptionGenerator(
    #             model_path="/data/users/ruotian_peng/pretrain/llava-1.6-vicuna7b",
    #         )
    global_and_relation_descriptor = pgc.PyramidCaption(processor=processor,model=model)
    # local_descriptor = LocalDescriptor()
    # llm_merge = VisualDescriptor('EMPTY', pipline=pipeline,llm_id=args.llm_id)
    # llm_check = Visualcheck('EMPTY', args.llm_id)

    # llm_descriptor = VisualDescriptor(openai_key, args.llm_id)
    # llm_evaluator = EvaluationInstructor(openai_key, arg s.llm_id)
    text_prompt = args.text_prompt

    img_src = args.image
    # img = read_image(img_src, format="BGR")
    # start_time = time.time()
    # predictions, visualized_output = demo.run_on_image(img)
    #
    # # detect only objects
    # args.test_task = 'ObjectDet'
    # cfg = setup_cfg(args)
    # demo = VisualizationDemo(cfg)
    # obj_detection, obj_visualized = demo.run_on_image(img)
    #
    # logger.info(
    #     "{}: {} in {:.2f}s".format(
    #         img_src,
    #         "detected {} instances".format(len(predictions["instances"]))
    #         if "instances" in predictions
    #         else "finished",
    #         time.time() - start_time,
    #     )
    # )

    # local_object_detection = local_descriptor.dense_pred_to_caption(obj_detection)
    # local_description = local_descriptor.dense_pred_to_caption(predictions)

    # out_filename = args.output
    # visualized_output.save(out_filename)
    # obj_visualized.save(out_filename.replace('.png', '_obj.png'))
    result_selfcheck={}
    result_merge={}
    result_orign={}
    result=[]
    json_path = '/data/users/ruotian_peng/experement/densecap/sample_result_coco200.json' # 替换为你的JSON文件路径
    with open(json_path, 'r') as f:
        data_image = json.load(f)
    with open('/data/users/ruotian_peng/experement/densecap/prompt.json', 'r') as f:
        data_prompt = json.load(f)
    # for key in tqdm(range(89,len(data_image),1)):
    for i,key in tqdm(enumerate(data_image),total=len(data_image)):
        # if i>=81:   
        # if key['image']=="coco_sample_data_Image_Textualization/000000248009.jpg":
            temp={}
            # img_src='/data/users/ruotian_peng/dataset/mmvp/MMVP Images/'+data_image[i]['image'].split('/')[-1]
            img_src ='/data/users/ruotian_peng/experement/densecap/'+key['image']
            img_prompt=img_src.split('/')[-1]
            prompt=data_prompt[img_prompt]
            global_description = global_and_relation_descriptor.get_global_description(img_src,prompt).replace("\n\n", " ")

            image = Image.open(img_src)
            width, height = image.size

            ic(global_description, height, width)
            # ic(local_description)
            # ic(local_object_detection)


            # relation_description = global_and_relation_descriptor.get_local_obj_description(img_src, local_object_detection)
            # relation_description = global_and_relation_descriptor.get_dense_description(img_src, local_description)

            # four_equal_describe = global_and_relation_descriptor.get_4_description(img_src).replace("\n\n", " ")
            # ic(four_equal_describe)

            # print(four_equal_describe)
            # paragraphs = re.split(r'\n[\r\n]*', four_equal_describe)
            # for i, p in enumerate(paragraphs):
            #     print(p)
            #     print('-'*20)
            # sys.exit()

            data = r'/data/users/ruotian_peng/experement/densecap/sample_result_coco200.json'
            box_description = global_and_relation_descriptor.generate_5_self_box_description(data, img_src,prompt)
            # for j in range(len(box_description)):
            #     box_description[j].replace("\n\n", " ")
            ic(box_description)
            temp['image']=img_src
            temp['global']=global_description
            temp['size']=[height,width]
            temp['local']=box_description
            result.append(temp)            
            # check_group=llm_merge.Selfcheck_description(global_description, box_description,
            #                                                                     width, height).replace("\n\n", " ")
            # ic(check_group)
            # output={  }
            # output['result']=check_group
            # with open('result_output.json', 'w') as json_file:
            #     json.dump(output, json_file, indent=4)
            # four_description = llm_merge.generate_multi_granualrity_equal_description(global_description, four_equal_describe,
            #                                                                     width, height).replace("\n\n", " ")
            # ic(four_description)
            # scene_description = llm_merge.generate_multi_granualrity_description(global_description, box_description,
            #                                                                       width, height).replace("\n\n", " ")
            # ic(scene_description)
    #         selfcheck_description=global_and_relation_descriptor.generate_selfcheck_description(data, img_src,check_group)
    # #         # ic(selfcheck_description)
    #         result_selfcheck[img_src]=selfcheck_description
            # result_merge['/data/users/ruotian_peng/LLaVA-main/playground/data/sharegpt/figure'+str(i+1)+'.png']=scene_description
            # result_orign[img_src]=global_description
    # with open('result_selfcheck_mmvp.json', 'w') as json_file:
    #         json.dump(result_selfcheck, json_file, indent=4)
    # with open('result_merge.json', 'w') as json_file:
    #         json.dump(result_merge, json_file, indent=4)
            with open('/data/users/ruotian_peng/experement/densecap/result_3_times_5box_qwenvl2_10.7.json', 'w') as json_file:
                    json.dump(result, json_file, indent=4)
            # overall, error_counting, overall_rationale, error_counting_rationale = llm_evaluator.generate_score_with_rationale(scene_description, text_prompt)
            # ic(overall, overall_rationale)
            # ic(error_counting, error_counting_rationale)




