import argparse
import os
import json
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
from PIL import Image

import patch_caption as pgc
from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm
)
import time



def get_parser():
    parser = argparse.ArgumentParser(description="Process images and generate descriptions using Llava model")
    parser.add_argument(
        "--input_file", type=str, default='',
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

    start_time = time.time()
    model_type = 'llava1_6-vicuna-7b-instruct'

    llm_engine = get_vllm_engine(model_type, torch.float16)
    template_type = get_default_template_type(model_type)
    template = get_template(template_type, llm_engine.hf_tokenizer)
    llm_engine.generation_config.max_new_tokens = 512
        # 记录程序开始时间
   



   
    global_and_relation_descriptor = pgc.PyramidCaption(generator=llm_engine,template=template)

    # Load the JSON input file
    with open(args.input_file, 'r') as f:
        data_image = json.load(f)

    # Calculate the chunk size and slice the data accordingly
    total_images = len(data_image)
    chunk_size = total_images // args.chunk_num
    start_idx = args.chunk_index * chunk_size
    end_idx = (args.chunk_index + 1) * chunk_size if args.chunk_index < args.chunk_num - 1 else total_images

    result = []
    for i, key in tqdm(enumerate(data_image[start_idx:end_idx]), total=end_idx - start_idx):
        temp = {}
        img_src = key['image']
        img_src = '/home/pengruotian/patch_matter/coco_sample_data_Image_Textualization/'+key['image'].split('/')[-1]
        prompt = "Describe this image in detail."
    

        # Open image to get size
        image = Image.open(img_src)
        width, height = image.size

        # Generate local box descriptions (optional)
        box_description = global_and_relation_descriptor.generate_5_self_box_description(key, img_src, prompt,args.chunk_index)

        # Prepare the result for this image
        temp['image'] = img_src
        temp['size'] = [height, width]
        temp['local'] = box_description
        result.append(temp)

    # Save the results to a JSON file
    output_file = os.path.join(args.output_folder, f'orginal_description_chunk_{args.chunk_index}.json')
    with open(output_file, 'w') as json_file:
        json.dump(result, json_file, indent=4)
    print(f"Results for chunk {args.chunk_index} saved to {output_file}")
    img_list=[]
    global_list=[]
    num=0
    for i, key in tqdm(enumerate(data_image[start_idx:end_idx]), total=end_idx - start_idx):
        num+=1
        temp = {}
        img_src = key['image']
        prompt = "Describe this image in detail."
        img_list.append(img_src)
        # Get global description
        if len(img_list)==20:
            global_description = global_and_relation_descriptor.batch_get_global_description(img_list, prompt)
            for global_des in range(len(global_description)):

                global_list.append(global_description[global_des]['response'].replace("\n\n", " "))
            img_list=[]
        elif i==end_idx-start_idx-1:
            global_description = global_and_relation_descriptor.batch_get_global_description(img_list, prompt)
            for global_des in range(len(global_description)):

                global_list.append(global_description[global_des]['response'].replace("\n\n", " "))
    output_file = os.path.join(args.output_folder, f'orginal_description_chunk_{args.chunk_index}.json')
    with open(output_file, 'r') as f:
        data_image = json.load(f)
 
    for i, key in tqdm(enumerate(data_image), total=end_idx - start_idx):
     
        key['global'] = global_list[i]
    # Save the modified data back to the same JSON file
    with open(output_file, 'w') as f:
        json.dump(data_image, f, indent=4)
    print(f"Results for chunk {args.chunk_index} saved to {output_file}")
     # 记录程序结束时间
    end_time = time.time()

    # 计算并输出程序运行时间
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time} 秒")
if __name__ == "__main__":
    # Parse arguments
    args = get_parser().parse_args()

    # Start image processing
    process_images(args)
