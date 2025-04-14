import argparse
import os
import json
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
from PIL import Image
from icecream import ic
import sys
from semantic_batch import fusion
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import transformers
from icecream import ic
from lavis.models import load_model_and_preprocess
import os
import io
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
# Setup folder2_path based on the relative path
folder2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '/data/users/ruotian_peng/LLaVA-main/llava/serve'))
import time
# Add folder2 to sys.path
sys.path.insert(0, folder2_path)
from image_caption import ImageCaptionGenerator
class BLIPScore():
    def __init__(self):
        self.name = 'blip2_image_text_matching'
        self.model_type = 'coco'
        self.device = 'cuda'

    def load_model(self):
        error = None
        try:
            self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(name='blip_image_text_matching',model_type='large',device=self.device,is_eval=True)
            return error
        except Exception as e:
            error = "Error loading model: {}".format(e)
        
            return error

    def process_image(self, image):
        image_processed = None
        error = None
        try:
            image_processed = self.vis_processors["eval"](image)[None].to(self.device)
        except Exception as e:
            error = str(e)
        return image_processed, error

    def rank_captions(self, img, caption):
        score = None
        error = None
        try:
            txt = self.text_processors["eval"](caption)
            itm_output = self.model({"image": img, "text_input": txt},
                                    match_head="itm")
            itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
            probability = itm_scores[:, 1].item()
            itc_score = self.model({"image": img, "text_input": txt},
                                match_head='itc')
            similarity = itc_score[:,0].item()

            score = (probability + similarity) / 2
        except Exception as e:
            error = str(e)
        return score,error

def get_parser():
    parser = argparse.ArgumentParser(description="Process images and generate descriptions using Llava model")
    parser.add_argument(
        "--input_spatial", type=str, default='/home/haiying_he/image-textualization/benchmark/DenseCap_Metrics/sample_result_coco200.json',
    )
    parser.add_argument(
        "--input_semantic", type=str, default='/home/haiying_he/dataset/coco2017_main_box.json',
    )
    parser.add_argument(
        "--output_folder", type=str, default='./data', 
        help="Folder to save the output description results"
    )
    parser.add_argument(
        "--chunk_index", type=int, required=True, help="Index of the current chunk"
    )
    parser.add_argument(
        "--chunk_num", type=int, required=True, help="Total number of chunks"
    )
    parser.add_argument(
        "--node_index", type=int, required=True, help="Index of the current node"
    )
    parser.add_argument(
        "--node_num", type=int, required=True, help="Total number of nodes"
    )
    return parser

def process_images(args):
    # Setup model and processor
    start_time = time.time()
    llama2_7b_chat_hf="/data/users/ruotian_peng/pretrain/llama-3.1-8b-instruct"
    llm = LLM(model=llama2_7b_chat_hf,max_model_len=40000,tensor_parallel_size=1,gpu_memory_utilization=0.87,dtype='float16')
    tokenizer = AutoTokenizer.from_pretrained(llama2_7b_chat_hf)
    blip_model = BLIPScore()
    error = blip_model.load_model()

    import json
    with open('/data/users/ruotian_peng/Patch_matter/fusion/new_global/new_global.json', 'r') as f:
        data_image = json.load(f)
    # json_path='/data/users/ruotian_peng/Patch_matter/fusion/all.json'
    # with open(json_path, 'r') as f:
    #     data_image = json.load(f)
        # image_idx=3
    # with open('/home/haiying_he/image-textualization/benchmark/DenseCap_Metrics/sample_result_coco200.json', 'r', encoding='utf-8') as file:
    #     json_data = json.load(file)
    # path='/home/haiying_he/dataset/coco2017_main_box.json'
    # with open(path, 'r', encoding='utf-8') as file_main:
    #     json_main = json.load(file_main)

    # Calculate the chunk size and slice the data accordingly
    total_images = len(data_image)
    chunk_size = total_images // args.chunk_num
    start_idx = args.chunk_index * chunk_size
    end_idx = (args.chunk_index + 1) * chunk_size if args.chunk_index < args.chunk_num - 1 else total_images
    index_list = list(range(start_idx, end_idx))
    print(index_list)
    Fusion=fusion(llm,tokenizer,blip_model,data_image)
    result = []
    num=start_idx+1
    batch_merge_five=[]
    batch_merge_main=[]
    batch_new_global=[0 for i  in range(len(data_image[start_idx:end_idx]))]
    total_main_num=[]
    main_num=[]
    for i, key in tqdm(enumerate(data_image[start_idx:end_idx]), total=end_idx - start_idx):
        temp=Fusion.batch_cal_main(key)
    
        if type(temp)==list:
            batch_merge_main.append(temp)
            total_main_num.append(i)
            main_num.append(i)
            if len(batch_merge_main) ==10:
                temp_new_global=Fusion.batch_merge_main(batch_merge_main)
                
                for num_complete in range(len(temp_new_global)):
                    # batch_new_global.append(temp_new_global[num_complete].outputs[0].text)
                    batch_new_global[main_num[num_complete]]=temp_new_global[num_complete].outputs[0].text
                main_num=[]
                batch_merge_main=[]
           
        else:
            total_main_num.append(-1)
            key['global']=temp
            batch_new_global[i]=0
        if num-start_idx==end_idx - start_idx:
                temp_new_global=Fusion.batch_merge_main(batch_merge_main)
                
                for num_complete in range(len(temp_new_global)):
                    # batch_new_global.append(temp_new_global[num_complete].outputs[0].text)
                    batch_new_global[main_num[num_complete]]=temp_new_global[num_complete].outputs[0].text
        num+=1
    temp_json=[]
  
    for i, key in tqdm(enumerate(data_image[start_idx:end_idx]), total=end_idx - start_idx):
        if i in total_main_num:
            key['global']=batch_new_global[total_main_num.index(i)]
        temp_json.append(key)

    output_file = os.path.join('new_global', f'orginal_description_chunk_{args.chunk_index}.json')
    with open(output_file, 'w') as json_file:
        json.dump(temp_json, json_file, indent=4)
    print(f"Results for chunk {args.chunk_index} saved to {output_file}")

#     folder_path = '/data/users/ruotian_peng/Patch_matter/fusion/new_global'

#     # 获取该文件夹下所有json文件
#     json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

#     # 用来存储合并后的数据
#     merged_data = []

#     # 遍历每个文件，将其内容合并
#    # 遍历每个文件，将其内容合并到同一个列表
#     for file in json_files:
#         file_path = os.path.join(folder_path, file)
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             if isinstance(data, list):  # 确保每个文件的数据是一个列表
#                 merged_data.extend(data)  # 将该列表的数据合并到merged_data


#     # 保存合并后的数据到一个新的json文件
#     output_file = '/data/users/ruotian_peng/Patch_matter/fusion/new_global/new_global.json'
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(merged_data, f, ensure_ascii=False, indent=4)

#     print(f"合并完成，结果已保存到 {output_file}")
    num=start_idx+1
    with open(output_file, 'r') as f:
        data_image = json.load(f)
    # print(data_image)
    print(len(data_image))
    # print(len(data_image[0]))
    for i, key in tqdm(enumerate(data_image), total=end_idx - start_idx):
        # print(key)
        print(num)
        # print(key)
        if num-start_idx==end_idx - start_idx:
           
            temp=Fusion.merge(key,'end')
            if type(temp)==list:
            
                complete_merge=Fusion.batch_merge_five(batch_merge_five)
                for num_complete in range(len(batch_merge_five)):
                        result_json={}
                        result_json['image']=batch_merge_five[num_complete][4]
                        result_json['description']=complete_merge[num_complete].outputs[0].text
                        result.append(result_json)
                        
            elif type(temp) == dict:
            
                result.append(temp)
        else:
            num+=1
            temp=Fusion.merge(key,'not end')
            if type(temp) == list:
                batch_merge_five.append(temp)
                print("len",len(batch_merge_five))
                if  len(batch_merge_five)==20:
                    complete_merge=Fusion.batch_merge_five(batch_merge_five)
                    for num_complete in range(len(batch_merge_five)):
                            result_json={}
                            result_json['image']=batch_merge_five[num_complete][4]
                            result_json['description']=complete_merge[num_complete].outputs[0].text
                            result.append(result_json)
                        
                    # for temp_json in temp:
                    #     result.append(temp_json)
                 
                    batch_merge_five=[]
                else:
                
                    continue
       
            elif type(temp) == dict:
                result.append(temp)
                # num+=1
                
        # num+=1
        # print(result)
        # Save the results to a JSON file
    output_file = os.path.join(args.output_folder, f'orginal_description_chunk_{args.chunk_index}.json')
    with open(output_file, 'w') as json_file:
        json.dump(result, json_file, indent=4)

    print(f"Results for chunk {args.chunk_index} saved to {output_file}")
    end_time = time.time()

    # 计算并输出程序运行时间
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time} 秒")
if __name__ == "__main__":
    # Parse arguments
    args = get_parser().parse_args()

    # Start image processing
    process_images(args)
