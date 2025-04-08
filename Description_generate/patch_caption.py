from transformers import AutoProcessor, AutoModelForPreTraining
import torch
from PIL import Image
import numpy as np
import itertools
import re, json
import icecream as ic
import sys
import os
from shapely.geometry import box
from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm
)



class PyramidCaption:
    def __init__(self, generator=None, template=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = generator
        self.template = template
    def batch_get_global_description(self, image, prompt):
        self.generator.generation_config.temperature = 0.2
        request_list=[]
        generation_info={}
        for img in image:
            request_list.append(  {'query': 'Describe this image in detail.', 'images': img})
        resp_list = inference_vllm(self.generator, self.template, request_list, generation_info=generation_info, use_tqdm=True)
        return resp_list

    def generate_5_self_box_description(self, key, image_scr, prompt,chunk_index):
        self.generator.generation_config.temperature = 0.7


        print(key['image'] + 'is progress')
        image_src = Image.open(image_scr).convert('RGB')
        width, height = image_src.size

        four_box = key['four_box']
        main_box = key['main_box']
        left_top = image_src.crop(four_box[0])
        right_top = image_src.crop(four_box[1])
        left_bottom = image_src.crop(four_box[2])
        right_bottom = image_src.crop(four_box[3])
        main = image_src.crop(main_box)

        cleaned_regions = []

        # print(region_locations)
        rectangles = [box(main_box[0], main_box[1], main_box[2], main_box[3])]
        rectangles.append(box(0,0,width,height))
        intersection_area = rectangles[0].intersection(rectangles[1]).area
        union_area = rectangles[0].area + rectangles[1].area - intersection_area
        iou = intersection_area / union_area
        generate_texts = ""
        temp_list = []
        dict_temp = {}

        if iou>0.4:
            print("iou>0.4")

            responses = []
            num_responses = 3
            generation_info = {}
            left_top.save('temp_image/'+str(chunk_index)+'left_top'+'.png')
            right_top.save('temp_image/'+str(chunk_index)+'right_top'+'.png')
            left_bottom.save('temp_image/'+str(chunk_index)+'left_bottom'+'.png')
            right_bottom.save('temp_image/'+str(chunk_index)+'right_bottom'+'.png')
            left_top_list=[{'query': 'Describe this image in detail.', 'images': ['temp_image/'+str(chunk_index)+'left_top'+'.png']}for _ in range(num_responses)]
            right_top_list=[{'query': 'Describe this image in detail.', 'images': ['temp_image/'+str(chunk_index)+'right_top'+'.png']}for _ in range(num_responses)]
            left_bottom_list=[{'query': 'Describe this image in detail.', 'images': ['temp_image/'+str(chunk_index)+'left_bottom'+'.png']}for _ in range(num_responses)]
            right_bottom_list=[{'query': 'Describe this image in detail.', 'images': ['temp_image/'+str(chunk_index)+'right_bottom'+'.png']}for _ in range(num_responses)]
            # 合并四个列表
            combined_list = left_top_list + right_top_list + left_bottom_list + right_bottom_list
        
            resp_list = inference_vllm(self.generator, self.template, combined_list, generation_info=generation_info, use_tqdm=True)
    
            print(generation_info)
            # responses=[resp_list[0]['response'].replace("\n\n", " ")+'n',resp_list[1]['response'].replace("\n\n", " ")+'n',resp_list[2]['response'].replace("\n\n", " ")+'n']
            for i, (image, name) in enumerate(
                    [(left_top, f'Region location:{four_box[0]}. Region description'),
                    (right_top, f'Region location:{four_box[1]}. Region description'),
                    (left_bottom, f'Region location:{four_box[2]}. Region description'),
                    (right_bottom, f'Region location:{four_box[3]}. Region description')]):
                responses=[resp_list[i]['response'].replace("\n\n", " ")+'n',resp_list[i+1]['response'].replace("\n\n", " ")+'n',resp_list[i+2]['response'].replace("\n\n", " ")+'n']
                # responses = []
                # num_responses = 3
                # generation_info = {}
                # image.save('/data/users/ruotian_peng/Patch_matter/description/'+str(chunk_index)+'.png')
                # request_list = [ {'query': 'Describe this image in detail.', 'images': ['/data/users/ruotian_peng/Patch_matter/description/'+str(chunk_index)+'.png']}for _ in range(num_responses)]
                # resp_list = inference_vllm(self.generator,self.template, request_list)
                # resp_list = inference_vllm(self.generator, self.template, request_list, generation_info=generation_info, use_tqdm=True)
        
                # print(generation_info)
                
                # responses=[resp_list[0]['response'].replace("\n\n", " ")+'n',resp_list[1]['response'].replace("\n\n", " ")+'n',resp_list[2]['response'].replace("\n\n", " ")+'n']
                # for _ in range(num_responses):
                #     image.save('/data/users/ruotian_peng/Patch_matter/description/'+str(chunk_index)+'.png')
                #     request_list = [ {'query': 'Describe this image in detail.', 'images': ['/data/users/ruotian_peng/Patch_matter/description/'+str(chunk_index)+'.png']}]
                #     resp_list = inference_vllm(self.generator,self.template, request_list)
                #     output=resp_list[0]['response']
                #     # output=self.generator.generate_captions(image, prompt,'llava_v0',temperature=0.7)
                #     generated_text = output.replace("\n\n", " ")
                #     generate_texts += generated_text + '\n'
                #     responses.append(generate_texts)
                #     generate_texts = ""

                if name in dict_temp:
                    if i == 4:
                        temp_loca = main_box
                        temp_loca[-1] = temp_loca[-1] - 2
                        name = f'Region location:{temp_loca}. Region description'
                        dict_temp[name] = responses
                    else:
                        temp_loca = four_box[i]
                        temp_loca[-1] = temp_loca[-1] - 1
                        name = f'Region location:{temp_loca}. Region description'
                        dict_temp[name] = responses
                else:
                    dict_temp[name] = responses
            temp_list.append(dict_temp)
        else:
            print("iou<0.4")
            responses = []
            num_responses = 3
            generation_info = {}
            left_top.save('temp_image/'+str(chunk_index)+'left_top'+'.png')
            right_top.save('temp_image/'+str(chunk_index)+'right_top'+'.png')
            left_bottom.save('temp_image/'+str(chunk_index)+'left_bottom'+'.png')
            right_bottom.save('temp_image/'+str(chunk_index)+'right_bottom'+'.png')
            main.save('temp_image/'+str(chunk_index)+'main'+'.png')
            left_top_list=[{'query': 'Describe this image in detail.', 'images': ['temp_image/'+str(chunk_index)+'left_top'+'.png']}for _ in range(num_responses)]
            right_top_list=[{'query': 'Describe this image in detail.', 'images': ['temp_image/'+str(chunk_index)+'right_top'+'.png']}for _ in range(num_responses)]
            left_bottom_list=[{'query': 'Describe this image in detail.', 'images': ['temp_image/'+str(chunk_index)+'left_bottom'+'.png']}for _ in range(num_responses)]
            right_bottom_list=[{'query': 'Describe this image in detail.', 'images': ['temp_image/'+str(chunk_index)+'right_bottom'+'.png']}for _ in range(num_responses)]
            main_list=[{'query': 'Describe this image in detail.', 'images': ['temp_image/'+str(chunk_index)+'main'+'.png']}for _ in range(num_responses)]
            # 合并四个列表
            combined_list = left_top_list + right_top_list + left_bottom_list + right_bottom_list+main_list
        
            resp_list = inference_vllm(self.generator, self.template, combined_list, generation_info=generation_info, use_tqdm=True)
    
            for i, (image, name) in enumerate(
                    [(left_top, f'Region location:{four_box[0]}. Region description'),
                    (right_top, f'Region location:{four_box[1]}. Region description'),
                    (left_bottom, f'Region location:{four_box[2]}. Region description'),
                    (right_bottom, f'Region location:{four_box[3]}. Region description'),
                    (main, f'Region location:{main_box}. Region description')]):
                responses=[resp_list[i]['response'].replace("\n\n", " ")+'n',resp_list[i+1]['response'].replace("\n\n", " ")+'n',resp_list[i+2]['response'].replace("\n\n", " ")+'n']

                if name in dict_temp:
                    if i == 4:
                        temp_loca = main_box
                        temp_loca[-1] = temp_loca[-1] - 2
                        name = f'Region location:{temp_loca}. Region description'
                        dict_temp[name] = responses
                    else:
                        temp_loca = four_box[i]
                        temp_loca[-1] = temp_loca[-1] - 1
                        name = f'Region location:{temp_loca}. Region description'
                        dict_temp[name] = responses
                else:
                    dict_temp[name] = responses
            temp_list.append(dict_temp)

        return temp_list