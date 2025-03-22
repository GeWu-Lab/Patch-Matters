from transformers import AutoProcessor, AutoModelForPreTraining
import torch
from PIL import Image
import numpy as np
import itertools
import re, json
import icecream as ic
import sys
import os
from qwen_vl_utils import process_vision_info
# 获取 folder2 的绝对路径
folder2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '/data/users/ruotian_peng/LLaVA-main/llava/serve'))

# 将 folder2 添加到 sys.path
sys.path.insert(0, folder2_path)
# from image_caption import ImageCaptionGenerator



class PyramidCaption:
    def __init__(self, model_name="llava-1.5-7b",processor=None,model=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # if model_name == "llava-1.5-7b":
        #     self.processor = AutoProcessor.from_pretrained("/data/users/ruotian_peng/pretrain/llava-1.6-vicuna7b",
        #                                                        torch_dtype=torch.float16)  # torch.float16
        #     self.model = LlavaForConditionalGeneration.from_pretrained("/data/users/ruotian_peng/pretrain/llava-1.6-vicuna7b",
        #                                                        torch_dtype=torch.float16,
        #                                                        low_cpu_mem_usage=True)
        # self.model.to(self.device)
        
        self.processor=processor
        self.model=model
        long_prompt = "Describe this image in detail."
        short_prompt = "USER: <image>\nProvide a one-sentence caption for the provided image.\nASSISTANT:"
        long_prompt='You are a powerful image captioner. Create detailed captions describing the contents of the given image. Include the object types and colors, counting the objects, object actions, texts, relative positions between objects, etc. Instead of describing the imaginary content, only describing the content one can determine confidently from the image. Please output contextually coherent descriptions directly, not itemized content in list form. Minimize aesthetic descriptions as much as possible. '
        self.prompt = long_prompt
    def get_global_description(self, image,prompt):
        images = Image.open(image).convert('RGB')

        # inputs = self.processor(
        #     text=self.prompt, images=images, return_tensors="pt"
        # ).to(self.device, torch.float16)
        # # inputs.pop("image_sizes", None)
        # image_size=inputs['image_sizes']
        # outputs = self.model.generate(
        #     **inputs,
        #     do_sample=False,
        #     image_sizes=[image_size],
        #     max_length=300
        # )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, do_sample=True,
                                                temperature=0.1,
                                                max_new_tokens=512,
                                                use_cache=True,
                                                top_p=0.95)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output=output_text[0]
        # output = self.model.generate(
        #         **inputs,
        #         do_sample=True,
        #         temperature=0.1,
        #         max_new_tokens=512,
        #         use_cache=True,
        #         top_p=0.95
        #     )
        
        # output=self.generator.generate_captions(images, prompt,'llava_v0')
        return output


    def select_teams_from_dict_function(self, obj_box):
        all_combinations = list(itertools.combinations(obj_box.keys(), 2))
        return all_combinations

    def get_local_obj_description(self, image_src, predictions):

        pattern = r'(\w+):\s*\[([0-9,\s]+)\]'
        predictions = re.findall(pattern, predictions)
        # print(predictions)

        obj_box = {}
        for obj, coords_str in predictions:
            obj = obj.strip()
            coords = [int(coord) for coord in coords_str.split(',')]
            if obj not in obj_box:
                obj_box[obj] = coords
            else:
                count = 2
                new_obj = f'{obj}_{count}'
                while new_obj in obj_box:
                    count += 1
                    new_obj = f'{obj}_{count}'
                obj_box[new_obj] = coords

        all_combinations = self.select_teams_from_dict_function(obj_box)

        # ic(obj_box)

        generated_texts = ""
        image = Image.open(image_src).convert('RGB')
        # width, height = image.size
        for team in all_combinations:
            bounding_box = [obj_box[team[0]], obj_box[team[1]]]
            boxes_array = np.array(bounding_box)
            left_upper = np.min(boxes_array[:, :2], axis=0)
            right_lower = np.max(boxes_array[:, 2:], axis=0)
            image_crop = image.crop((left_upper[0], left_upper[1], right_lower[0], right_lower[1]))
            image_crop.save('/home/haiying_he/LLMScore/llm_descriptor/image_show/image_crop-{}.jpg'.format(team))
            # print('left_upper:', left_upper)
            # print('right_lower:', right_lower)

            box_1 = np.array(obj_box[team[0]]) - np.array([left_upper[0], left_upper[1], left_upper[0], left_upper[1]])
            box_2 = np.array(obj_box[team[1]]) - np.array([left_upper[0], left_upper[1], left_upper[0], left_upper[1]])
            box_1 = [int(i) for i in box_1.tolist()]
            box_2 = [int(i) for i in box_2.tolist()]

            # prompt = self.prompt.format(team[0], box_1, team[1], box_2, [image_crop.width, image_crop.height])
            # prompt = self.prompt.format(team[0], team[1])
            # prompt = self.prompt.format(team[0], team[1])
            # prompt = self.prompt.format(team[0], team[1])

            inputs = self.processor(images=image_crop, text=self.prompt, return_tensors="pt").to(
                self.device, torch.float16)
            generated_ids = self.model.generate(**inputs,
                                                do_sample=False,
                                                max_length=300
                                                )
            generated_text = self.processor.decode(generated_ids[0][(len(inputs['input_ids'][0])):], skip_special_tokens=True)

            # print(generated_text)
            image_area = f'X:{left_upper[0]} Y:{left_upper[1]} Width:{right_lower[0]-left_upper[0]} Height:{right_lower[1]-left_upper[1]}'
            generated_text = generated_text + image_area


            generated_texts += generated_text + '\n'

        return generated_texts


    # girt object and dense bounding box
    def get_dense_description(self, image_src, local_description):

        pattern = r'([\w\s.,]+):\s*\[([\d,\s]+)\]'
        # print(local_descriptor)
        predictions = re.findall(pattern, local_description)
        # print(predictions)

        obj_box = {}
        for obj, coords_str in predictions:
            obj = obj.strip()
            coords = [int(coord) for coord in coords_str.split(',')]
            if obj not in obj_box:
                obj_box[obj] = coords
            else:
                count = 2
                new_obj = f'{obj}_{count}'
                while new_obj in obj_box:
                    count += 1
                    new_obj = f'{obj}_{count}'
                obj_box[new_obj] = coords

        image = Image.open(image_src).convert('RGB')
        # width, height = image.size
        # print('-------------', obj_box)
        generated_texts = ""
        for team in obj_box.keys():
            bounding_box = obj_box[team]
            # print('***********************bounding_box:', bounding_box, bounding_box[0])
            image_crop = image.crop((bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]))
            image_crop.save('/home/haiying_he/LLMScore/llm_descriptor/image_show/image_crop-{}.jpg'.format(team))

            inputs = self.processor(images=image_crop, text=self.prompt, return_tensors="pt").to(
                self.device, torch.float16)
            generated_ids = self.model.generate(**inputs,
                                                do_sample=False,
                                                max_length=300
                                                )
            generated_text = self.processor.decode(generated_ids[0][(len(inputs['input_ids'][0])):], skip_special_tokens=True)

            # print(generated_text)
            image_area = f'X:{bounding_box[0]} Y:{bounding_box[1]} Width:{bounding_box[2]-bounding_box[0]} Height:{bounding_box[3]-bounding_box[1]}'
            generated_text = generated_text + image_area
            generated_texts += generated_text + '\n'

        return generated_texts

    def generate_4_equal_description(self, json_file, image_scr,prompt):
        with open(json_file, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        generate_texts = ""
        for i, data in enumerate(json_data):
            if data['image'].split('/')[-1] == image_scr.split('/')[-1]:
                print(data['image'] + 'is progress')
                # image_src = Image.open('/data/users/ruotian_peng/dataset/mmvp/MMVP Images/'+data['image'].split('/')[-1]).convert('RGB')
                image_src = Image.open(image_scr).convert('RGB')
                width, height = image_src.size

                part_width = width // 2
                part_height = height // 2

                left_top_part = [0, 0, part_width, part_height]
                right_top_part = [part_width, 0, width, part_height]
                left_bottom_part = [0, part_height, part_width, height]
                right_bottom_part = [part_width, part_height, width, height]
                four_box=[left_top_part,right_top_part,left_bottom_part,right_bottom_part]
                left_top = image_src.crop(left_top_part)
                right_top = image_src.crop(right_top_part)
                left_bottom = image_src.crop(left_bottom_part)
                right_bottom = image_src.crop(right_bottom_part)

                # two_box = data['two_box']
                # left_or_top = image_src.crop(two_box[0])
                # right_or_bottom = image_src.crop(two_box[1])

                generate_texts = ""
                temp_list=[]
                dict_temp={}
                for i, (image, name) in enumerate(
                        [(left_top, f'Region location:{four_box[0]}. Region description'), (right_top, f'Region location:{four_box[1]}. Region description'),
                         (left_bottom, f'Region location:{four_box[2]}. Region description'), (right_bottom, f'Region location:{four_box[3]}. Region description')]):
                    # image=Image.open(image).convert('RGB')
                    responses=[]
                    num_responses=3
                    
                    
                    for _ in range(num_responses):
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": image,
                                    },
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ]
                        text = self.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = self.processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        )
                        inputs = inputs.to("cuda")
                        generated_ids = self.model.generate(**inputs, do_sample=True,
                                                                temperature=0.1,
                                                                max_new_tokens=512,
                                                                use_cache=True,
                                                                top_p=0.95)

                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        output_text = self.processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        output=output_text[0]
                        # output=self.generator.generate_captions(image, prompt,'llava_v0',temperature=0.7)
                        generated_text = output.replace("\n\n", " ")
                        # generated_text = f"{name}:" + generated_text+'[end]'
                        generate_texts += generated_text + '\n'
                        responses.append(generate_texts)
                        generate_texts=""

                    # dict_temp[name]=responses
                    if name in dict_temp:
                        temp_loca=four_box[i]
                        temp_loca[-1]=temp_loca[-1]-1
                        name=f'Region location:{temp_loca}. Region description'
                        # dict_temp[name].extend(responses)
                        dict_temp[name] = responses
                    else:
                        dict_temp[name] = responses
                temp_list.append(dict_temp)
                    # output=self.generator.generate_captions(image, self.prompt,'llava_v0',temperature=1)
                    # generated_text = output
                    # generated_text = f"{name}:" + generated_text+'[end]'
                    # generate_texts += generated_text + '\n'
        # del self.processor
        # del self.model

        return temp_list
    def get_4_description(self, image_src): #图像均分4份进行描述
        image_src = Image.open(image_src).convert('RGB')
        width, height = image_src.size

        part_width = width // 2
        part_height = height // 2

        left_top_part = [0, 0, part_width, part_height]
        right_top_part = [part_width, 0, width, part_height]
        left_bottom_part = [0, part_height, part_width, height]
        right_bottom_part = [part_width, part_height, width, height]

        left_top = image_src.crop(left_top_part)
        right_top = image_src.crop(right_top_part)
        left_bottom = image_src.crop(left_bottom_part)
        right_bottom = image_src.crop(right_bottom_part)
        generate_texts = ""
        # for i, (image, name) in enumerate([(left_top, 'left_top'), (right_top, 'right_top'), (left_bottom, 'left_bottom'), (right_bottom, 'right_bottom')]):
        for i, (image, name) in enumerate(
                [(left_top, f'Region location:{left_top_part}. Region description'),
                 (left_bottom, f'Region location:{left_bottom_part}. Region description'),
                 (right_top, f'Region location:{right_top_part}. Region description'),
                 (right_bottom, f'Region location:{right_bottom_part}. Region description')]):
            # inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(
            #     self.device, torch.float16)
            # generated_ids = self.model.generate(**inputs,
            #                                     do_sample=False,
            #                                     max_length=300
            #                                     )
            # image=Image.open(image).convert('RGB')
            output=self.generator.generate_captions(image, self.prompt,'llava_v0')
            generated_text = output
            generated_text = generated_text
            generate_texts += generated_text + '\n'


        # release memory
        # del self.processor
        # del self.model

        return generate_texts

    # def generate_4_self_box_description(self, json_file, image_scr):
    #     with open(json_file, 'r', encoding='utf-8') as file:
    #         json_data = json.load(file)
    #     generate_texts = ""
    #     for i, data in enumerate(json_data):
    #         if data['image'].split('/')[-1] == image_scr.split('/')[-1]:
    #             print(data['image'] + 'is progress')
    #             # image_src = Image.open('/data/users/ruotian_peng/dataset/mmvp/MMVP Images/'+data['image'].split('/')[-1]).convert('RGB')
    #             image_src=Image.open(image_scr).convert('RGB')
    #             width, height = image_src.size

    #             four_box = data['four_box']
    #             left_top = image_src.crop(four_box[0])
    #             right_top = image_src.crop(four_box[1])
    #             left_bottom = image_src.crop(four_box[2])
    #             right_bottom = image_src.crop(four_box[3])

    #             # two_box = data['two_box']
    #             # left_or_top = image_src.crop(two_box[0])
    #             # right_or_bottom = image_src.crop(two_box[1])

    #             generate_texts = ""
    #             temp_list=[]
    #             dict_temp={}
    #             for i, (image, name) in enumerate(
    #                     [(left_top, f'Region location:{four_box[0]}. Region description'), (right_top, f'Region location:{four_box[1]}. Region description'),
    #                      (left_bottom, f'Region location:{four_box[2]}. Region description'), (right_bottom, f'Region location:{four_box[3]}. Region description')]):
    #                 # image=Image.open(image).convert('RGB')
                  
    #                 output=self.generator.generate_captions(image, self.prompt,'llava_v0',temperature=0.2)
    #                 generated_text = output
    #                 generated_text = f"{name}:" + generated_text+'[end]'
    #                 generate_texts += generated_text + '\n'

    #     # del self.processor
    #     # del self.model

    #     return generate_texts
    def generate_5_self_box_description(self, json_file, image_scr,prompt):
        with open(json_file, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        path='/data/users/ruotian_peng/experement/densecap/main_box.json'
        with open(path, 'r', encoding='utf-8') as file_main:
            json_main = json.load(file_main)
        # print(json_main)
        generate_texts = ""
        for i, data in enumerate(json_data):
            if data['image'].split('/')[-1] == image_scr.split('/')[-1]:
                # for j, data_main in enumerate(json_main):
                    # print(data_main)
                img_name='/home/haiying_he/coco_sample_data_Image_Textualization/'+image_scr.split('/')[-1]
                    # if data_main['image'].split('/')[-1] == image_scr.split('/')[-1]:

                print(data['image'] + 'is progress')
                # image_src = Image.open('/data/users/ruotian_peng/dataset/mmvp/MMVP Images/'+data['image'].split('/')[-1]).convert('RGB')
                image_src=Image.open(image_scr).convert('RGB')
                width, height = image_src.size

                four_box = data['four_box']
                main_box=json_main[img_name]['main_box']
                left_top = image_src.crop(four_box[0])
                right_top = image_src.crop(four_box[1])
                left_bottom = image_src.crop(four_box[2])
                right_bottom = image_src.crop(four_box[3])
                main=image_src.crop(main_box)
                # two_box = data['two_box']
                # left_or_top = image_src.crop(two_box[0])
                # right_or_bottom = image_src.crop(two_box[1])

                generate_texts = ""
                temp_list=[]
                dict_temp={}
                for i, (image, name) in enumerate(
                        [(left_top, f'Region location:{four_box[0]}. Region description'), (right_top, f'Region location:{four_box[1]}. Region description'),
                        (left_bottom, f'Region location:{four_box[2]}. Region description'), (right_bottom, f'Region location:{four_box[3]}. Region description'),
                        (main, f'Region location:{main_box}. Region description')]):
                    # image=Image.open(image).convert('RGB')
                    responses=[]
                    num_responses=3
                    
                    
                    for _ in range(num_responses):
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": image,
                                    },
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ]
                        text = self.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = self.processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        )
                        inputs = inputs.to("cuda")
                        generated_ids = self.model.generate(**inputs, do_sample=True,
                                                                temperature=0.7,
                                                                max_new_tokens=512,
                                                                use_cache=True,
                                                                top_p=0.95)

                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        output_text = self.processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        output=output_text[0]
                        # output = self.model.generate(
                        #         **inputs,
                        #         do_sample=True,
                        #         temperature=0.7,
                        #         max_new_tokens=512,
                        #         use_cache=True,
                        #         top_p=0.95
                        #     )
                        # output=self.generator.generate_captions(image, prompt,'llava_v0',temperature=0.7)
                        generated_text = output.replace("\n\n", " ")
                        # generated_text = f"{name}:" + generated_text+'[end]'
                        generate_texts += generated_text + '\n'
                        responses.append(generate_texts)
                        generate_texts=""

                    # dict_temp[name]=responses
                 
                    if name in dict_temp:
                        if i==4:
                            temp_loca=main_box
                            temp_loca[-1]=temp_loca[-1]-1
                            name=f'Region location:{temp_loca}. Region description'
                            # dict_temp[name].extend(responses)
                            dict_temp[name] = responses
                        else:
                            temp_loca=four_box[i]
                            temp_loca[-1]=temp_loca[-1]-1
                            name=f'Region location:{temp_loca}. Region description'
                            # dict_temp[name].extend(responses)
                            dict_temp[name] = responses
                    else:
                        dict_temp[name] = responses
                temp_list.append(dict_temp)
                    # output=self.generator.generate_captions(image, self.prompt,'llava_v0',temperature=1)
                    # generated_text = output
                    # generated_text = f"{name}:" + generated_text+'[end]'
                    # generate_texts += generated_text + '\n'
        # del self.processor
        # del self.model

        return temp_list
    def generate_4_self_box_description(self, json_file, image_scr,prompt):
        with open(json_file, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        generate_texts = ""
        for i, data in enumerate(json_data):
            if data['image'].split('/')[-1] == image_scr.split('/')[-1]:
                print(data['image'] + 'is progress')
                # image_src = Image.open('/data/users/ruotian_peng/dataset/mmvp/MMVP Images/'+data['image'].split('/')[-1]).convert('RGB')
                image_src=Image.open(image_scr).convert('RGB')
                width, height = image_src.size

                four_box = data['four_box']
                left_top = image_src.crop(four_box[0])
                right_top = image_src.crop(four_box[1])
                left_bottom = image_src.crop(four_box[2])
                right_bottom = image_src.crop(four_box[3])

                # two_box = data['two_box']
                # left_or_top = image_src.crop(two_box[0])
                # right_or_bottom = image_src.crop(two_box[1])

                generate_texts = ""
                temp_list=[]
                dict_temp={}
                for i, (image, name) in enumerate(
                        [(left_top, f'Region location:{four_box[0]}. Region description'), (right_top, f'Region location:{four_box[1]}. Region description'),
                         (left_bottom, f'Region location:{four_box[2]}. Region description'), (right_bottom, f'Region location:{four_box[3]}. Region description')]):
                    # image=Image.open(image).convert('RGB')
                    responses=[]
                    num_responses=3
                    
                    
                    for _ in range(num_responses):
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": image,
                                    },
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ]
                        text = self.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = self.processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        )
                        inputs = inputs.to("cuda")
                        generated_ids = self.model.generate(**inputs, do_sample=True,
                                                                temperature=0.1,
                                                                max_new_tokens=512,
                                                                use_cache=True,
                                                                top_p=0.95)

                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        output_text = self.processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        output=output_text[0]
                        # output = self.model.generate(
                        #         **inputs,
                        #         do_sample=True,
                        #         temperature=0.7,
                        #         max_new_tokens=512,
                        #         use_cache=True,
                        #         top_p=0.95
                        #     )
                        # output=self.generator.generate_captions(image, prompt,'llava_v0',temperature=0.7)
                        generated_text = output.replace("\n\n", " ")
                        # generated_text = f"{name}:" + generated_text+'[end]'
                        generate_texts += generated_text + '\n'
                        responses.append(generate_texts)
                        generate_texts=""

                    # dict_temp[name]=responses
                    if name in dict_temp:
                        temp_loca=four_box[i]
                        temp_loca[-1]=temp_loca[-1]-1
                        name=f'Region location:{temp_loca}. Region description'
                        # dict_temp[name].extend(responses)
                        dict_temp[name] = responses
                    else:
                        dict_temp[name] = responses
                temp_list.append(dict_temp)
                    # output=self.generator.generate_captions(image, self.prompt,'llava_v0',temperature=1)
                    # generated_text = output
                    # generated_text = f"{name}:" + generated_text+'[end]'
                    # generate_texts += generated_text + '\n'
        # del self.processor
        # del self.model

        return temp_list
    def generate_selfcheck_description(self, json_file, image_scr,caption):
        with open(json_file, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        for i, data in enumerate(json_data):
            if data['image'].split('/')[-1] == image_scr.split('/')[-1]:
                print(data['image'] + 'is progress')
                # image_src = Image.open('/data/users/ruotian_peng/dataset/mmvp/MMVP Images/'+data['image'].split('/')[-1]).convert('RGB')
                image_src=Image.open(image_scr).convert('RGB')
                width, height = image_src.size
                four_box = data['four_box']
       
        generate_texts = ""
        task_instruction = f"""
        Your task is to follow the instruction to create a contextually coherent, comprehensive, and detailed overall description based on the input global description of the image and multiple regional descriptions. The descriptions are classified into three categories: sentences describing the same thing, contradictory sentences, and sentences that only appear once. Synthesize this information and generate a coherent overall description that accurately reflects the objects' attributes, locations, and relationships in the image, ensuring the narrative is logical and fluid.
        
        The whole image size is width: {width}, height: {height}
        The region of paragraph1 is [{0,0,width,height}]
        The region of paragraph2 is {four_box[0]}
        The region of paragraph3 is {four_box[1]}
        The region of paragraph4 is {four_box[2]}
        The region of paragraph5 is {four_box[3]}

        Please create a contextually coherent, comprehensive, and detailed overall description.
        """
        prompt=task_instruction+'\n\n'+caption

       

                
                    # image=Image.open(image).convert('RGB')
      
        output=self.generator.generate_captions(image_src, prompt,'llava_selfcheck')
        

        # del self.processor
        # del self.model

        return output