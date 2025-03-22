import sys, ast
import random, re
import cv2
import torch, os, string, json
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import transformers
from PIL import Image
from icecream import ic
from tqdm import tqdm
import argparse

def draw(img, new_span_box_list, color_list=None):
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
                  (255, 0, 255)] if color_list is None else color_list

    image = cv2.imread(img)
    new_span_box = new_span_box_list

    color = random.choice(color_list)
    bgr_color = (color[2], color[1], color[0])
    cv2.rectangle(image, (int(new_span_box[0]), int(new_span_box[1])), (int(new_span_box[2]), int(new_span_box[3])),
                  bgr_color, 4)

    return image


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


def vlm_model(vlm_model_path, device='cuda'):
    # vlm_path = "Salesforce/blip2-opt-2.7b"
    # vlm_path = 'cache/huggingface/hub/models--Salesforce--blip2-opt-2.7b/snapshots/51572668da0eb669e01a189dc22abe6088589a24'
    vlm_path = vlm_model_path
    vis_processors = Blip2Processor.from_pretrained(vlm_path)  # "Salesforce/blip2-opt-2.7b"
    model = Blip2ForConditionalGeneration.from_pretrained(
        vlm_path, torch_dtype=torch.float16, device_map='auto'
    )
    return model, vis_processors


def generate_description(image_path, model, vis_processors, prompt=None):
    image = Image.open(image_path).convert("RGB")
    inputs = vis_processors(images=image, return_tensors="pt").to(model.device, torch.float16)
    generated_ids = model.generate(**inputs
                                   )
    generated_text = vis_processors.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

def find_common_objects(image, object, pipeline, prompt='Indirect'):
    if prompt == 'Direct':
        system = f"""
    Input:
    - You will receive a description of an image.

    Task Objective: 
    - Your task is to identify all objects are included in this description.
    - Your output should only be the object, Do not output extra information such as adjectives, colours.
    - Do NOT ignore any object in the description.
    - Do NOT give any explanation or notes on how you generate output.

    ### Example Input:

    Image Description: A red ladybug on a green leaf and a man sitting at the table.

    ### Example Output:

    ladybug, leaf, man, table.
    """
        user=f"""
                ### Image Description Words: {image}
                ### Output: 
        """
    elif prompt == 'Indirect':
        system = f"""
        Input:
        - You will receive a description of an image.

        Task Objective:
        - Your task is to find the information about the object in the description.
        - You output should be the object, Do not output extra information such as adjectives, colours.
        - Your output should be concise and simply.
        - Do NOT give any explanation or notes on how you generate output.
        """
        user=f"""
                ### Image Description Words: {image}
                ### Output: 
        """
    elif prompt == 'match':
        system = f"""
        Input:
        - You will receive a set of words from the image description.
        - Additionally you will receive a set of words from the object detection.

        Task Objective:
        - Your task is to find semantically identical words from object detection based on the words described in the image.
        - your output should fall into the category of words from object detection.
        - Your output should be concise and simply!
        - Do NOT give any explanation or notes on how you generate output!

        ### Example Input One:
        Image Description Words: ['red', 'ladybug', 'leaf', 'man', 'table', 'woman']
        Object Detection Words: ['apples', 'banana', 'orange', 'person', 'table', 'ladybug']
        ### Example Output One:
        ['ladybug', 'person', 'table']

        ### Example Input Two:
        Image Description Words: ['hydrant', 'ladybug', 'leaf', 'man', 'table', 'woman']
        Object Detection Words: ['apples', 'banana', 'orange']
        ### Example Output Two:
        ['None']
        """

        system = f"""
        Task Description:
        - You will receive two sets of words: one from the image description and another from object detection.
        - Your task is to identify semantically identical or highly related words from the object detection set that correspond to the words described in the image.
        - Your output should only include words from the object detection set that are semantically the same or closely related to the words from the image description.
        - Your output must be concise, listing only the matching words without any explanations or notes on how you generated the output.

        ### Example Input One:
        Image Description Words: ['red', 'ladybug', 'leaf', 'man', 'table', 'woman']
        Object Detection Words: ['apples', 'banana', 'orange', 'person', 'table', 'ladybug']
        ### Example Output One:
        ['ladybug', 'person', 'table']

        ### Example Input Two:
        Image Description Words: ['red', 'ladybug', 'leaf', 'man', 'table', 'woman']
        Object Detection Words: ['apples', 'banana', 'orange']
        ### Example Output Two:
        ['None']

        ### Example Input Three:
        Image Description Words: ['syringe', 'needle', 'man', 'woman']
        Object Detection Words: ['apples', 'banana', 'orange', 'person', 'car']
        ### Example Output Three:
        ['person']

        Remember, your output should only include words from the object detection!
        """
        user=f"""
                ### Image Description Words: {image}
                ### Object Detection Words: {object} 
                ### Output: 
        """
    # llama3_1_8b

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = pipeline.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
    terminators = [
                        pipeline.tokenizer.eos_token_id,
                        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]
    outputs = pipeline(
                        prompt,
                        max_new_tokens=4096,
                        pad_token_id=pipeline.tokenizer.eos_token_id,
                        eos_token_id=terminators,
                        do_sample=True,
                        temperature=0.2,
                        top_p=0.9,)
    # merged_caption = pipeline.tokenizer.decode(outputs[0]["generated_text"][len(prompt):])
    merged_caption = outputs[0]["generated_text"][len(prompt):]
    return merged_caption

def equal_four_box(image_path):
    img = Image.open(image_path)
    width, height = img.size

    half_width = width // 2
    half_height = height // 2

    boxes = [
        (0, 0, half_width, half_height),  
        (half_width, 0, width, half_height),  
        (0, half_height, half_width, height), 
        (half_width, half_height, width, height) 
    ]
    return boxes

def re_match(input_string):
    matched_content = re.findall(r"\[.*?\]", input_string)
    cleaned_items = matched_content[0][1:-1].replace("'", "").strip()


    return [cleaned_items] if "," not in cleaned_items else [item.strip() for item in cleaned_items.split(",")]



if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--image_folder', type=str, help='Image folder')
    arg.add_argument('--object_box_save_path', type=str, help='object detect path')
    arg.add_argument('--main_box_save_path', type=str, help='main box save path')
    arg.add_argument('--llm_path', type=str, help='LLM model', default='cache/huggingface/hub/mate-llama-3.1-8b-instruct')
    arg.add_argument('--vlm_model_path', type=str, help='vlm model path', default='Salesforce/blip2-opt-2.7b')
    arg.add_argument('--visual_save_folder', type=str, help='Save Visual Image folder', default='ovdet/main_box_visual')
    args = arg.parse_args()


    all_path = arg.image_folder
    save = arg.visual_save_folder
    object_detection_path = arg.object_box_save_path
    save_path = arg.main_box_save_path


    progress_path = f'ovdet/main_box_progress.json'

    with open(object_detection_path, 'r', encoding='utf-8') as file:
        object_detection = json.load(file)


    vlm_model_path = arg.vlm_model_path
    lvlm_model, vis_processors = vlm_model(vlm_model_path)
    llama_chat_hf = arg.llm_path
    pipeline = transformers.pipeline(
                "text-generation",
                model=llama_chat_hf,
                model_kwargs={"torch_dtype": torch.float16},
                device_map="auto",
            )

    save_dict = {}

    try: 
        with open(progress_path, 'r') as f:
            save_dict = json.load(f)
    except FileNotFoundError:
        save_dict = {}

    for img in tqdm(os.listdir(all_path)):
        
        all_bbox = []
        image_path = os.path.join(all_path, img)

        if image_path in save_dict:
            continue


        image = Image.open(image_path)
        width, height = image.size

        object_before = object_detection[image_path]['name']
        if image_path in object_detection:
            object = list(set(object_before))
        else:
            object = ['None']

        text = generate_description(image_path, lvlm_model, vis_processors)
        # ic(text)
        print('Image path: ', img)
        print('--------------\n', text, '\n----------------\n')

        common_objects = find_common_objects(text, object, pipeline, prompt='Direct').lower()

        common_objects = ''.join([c for c in common_objects if c not in string.punctuation])
        words_remove = ['sure', 'I', 's', 'i', 'im', 'understand', 'here', 'is', 'the', 'output', 'for', 'the', 'given',
                        'image', 'description', 'best', 'free', 'image', 'output', 'recipe', 'graduates', 'on', 'at', 'a', 'the',
                        'an', 'those', 'second', 'ready', 'to', 'help', 'please', 'provide', 'and', 'will', 'included',
                        'in', 'with', 'living']
        common_objects = [word for word in common_objects.split() if word not in words_remove]
        if len(common_objects) == 0:
            common_objects = ['None']
        print('🍅🍅🍅🍅🍅 common_objects\n', common_objects)
        print('🍑🍑🍑🍑🍑 object detect\n', object)
        # common_objects = find_common_objects(text, object, llm_m, llm_tokenizer, device, prompt='Indirect')
        # print('--------------\n', common_objects, '\n----------------\n')
        common_objects = find_common_objects(common_objects, object, pipeline, prompt='match').strip()
        # print(common_objects)
        # common_objects = re.findall(r"\['(.*?)'\]", common_objects)
        # print(common_objects)
        # common_objects = ast.literal_eval(common_objects)
        common_objects = re_match(common_objects)
        print('🍌🍌🍌🍌🍌 match result\n', common_objects)

        co_existence = [name for name in common_objects if name in object]
        print('🥝🥝🥝🥝🥝 co_existence\n', co_existence)
        if len(co_existence) == 0:
            co_existence = ['None']
            all_bbox.append([0.0, 0.0, width - 1.0, height - 1.0])
        else:
            index = [ind for ind, value in enumerate(object_before) if value in co_existence]
            all_bbox.extend([object_detection[image_path]['bbox'][i] for i in index])

        print("🍎🍎🍎🍎🍎🍎🍎 all_bbox")
        print(all_bbox)
        all_bbox = merge_box(all_bbox)
        print('merge: ', all_bbox)

        equal_box = equal_four_box(image_path)
        save_dict[image_path] = {"main_box": all_bbox,
                                "equal_four_box": equal_box}


        drawn_image = draw(image_path, all_bbox)
        cv2.imwrite('{}/{}'.format(save, img
                                ), drawn_image)

        with open(progress_path, 'w') as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=4)


    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(save_dict, file, ensure_ascii=False, indent=4)
    print('save to: ', save_path)
    print("all numbers:", len(save_dict))   
