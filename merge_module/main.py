from utils import ImageProcessor
from utils import BoundingBoxOperations
from description_merging import DescriptionMerger
from blip_model import BLIPScore
import transformers
import json
from tqdm import tqdm
from PIL import Image
import re
import torch
import argparse
import argparse
def cal_similarity_iou(des,merge,blip_model):
    similarity_list=[]
    if des==[]:
        print(1)
        similarity_list.append(0)  
        return similarity_list
    for triple in des:
    
        match = re.search(r'Region \d+', triple)
        if match:
            number=match.group(0).split(' ')[-1]
            # 删除 "Region" 部分，仅保留数字部分
            modified_text = re.sub(r'Region \d', '', triple)
            modified_text = re.sub(r'\(', '', modified_text)
            modified_text = re.sub(r'\)', '', modified_text)
            modified_text = re.sub(r'<', '', modified_text)
            modified_text = re.sub(r'>', '', modified_text)
            modified_text = re.sub(r',', '', modified_text)
            img, error = blip_model.process_image(merge)
            
            score,error = blip_model.rank_captions(img, modified_text)
            # print(modified_text,':',score)

            similarity_list.append(score)
        
        else:
            similarity_list.append(0)  
    return similarity_list
def cal_similarity_same(des,merge,blip_model):
    # dict_image={'1':left_top,'2':right_top,'3':left_bottom,'4':right_bottom}
    similarity_list=[]
    if des==[]:
        print(1)
        similarity_list.append(0)  
        return similarity_list
    for triple in des:
    
        match = re.search(r'Description \d+', triple)
        if match:
            number=match.group(0).split(' ')[-1]
            # 删除 "Region" 部分，仅保留数字部分
            modified_text = re.sub(r'Description \d', '', triple)
            modified_text = re.sub(r'\(', '', modified_text)
            modified_text = re.sub(r'\)', '', modified_text)
            modified_text = re.sub(r'<', '', modified_text)
            modified_text = re.sub(r'>', '', modified_text)
            modified_text = re.sub(r',', '', modified_text)
            img, error = blip_model.process_image(merge)
            
            score,error = blip_model.rank_captions(img, modified_text)
            # print(modified_text,':',score)

            similarity_list.append(score)
        
        else:
            similarity_list.append(0)  
    return similarity_list
def cal_similarity(dict_image,des,region_dict,blip_model):
    # dict_image={'1':left_top,'2':right_top,'3':left_bottom,'4':right_bottom}
    similarity_list=[]
    if des==[]:
        print(1)
        similarity_list.append(0)  
        return similarity_list
    for triple in des:
    
        match = re.search(r'Region \d+', triple)
        if match:
            number=match.group(0).split(' ')[-1]
            # 删除 "Region" 部分，仅保留数字部分
            modified_text = re.sub(r'Region \d', '', triple)
            modified_text = re.sub(r'\(', '', modified_text)
            modified_text = re.sub(r'\)', '', modified_text)
            modified_text = re.sub(r'<', '', modified_text)
            modified_text = re.sub(r'>', '', modified_text)
            modified_text = re.sub(r',', '', modified_text)
            
            img, error = blip_model.process_image(dict_image[str(region_dict[number]+1)])
            # fig, ax = plt.subplots()
            # ax.imshow(dict_image[str(region_dict[number]+1)])
            # plt.show()
            score,error = blip_model.rank_captions(img, modified_text)
            # print(modified_text,':',score)

            similarity_list.append(score)
        
        else:
            similarity_list.append(0)  
        return similarity_list
    
    
def main(args):
    # 初始化处理模块
    json_path = args.json_path
    box_path=args.box_path
    main_box_path=args.main_box_path
    llama3_7b_chat_hf = args.llama2_model_path
    # json_path='/data/users/ruotian_peng/experement/densecap/result_3_times_5box_9.26.json'
    # llama3_7b_chat_hf="/data/users/ruotian_peng/pretrain/llama-3.1-8b-instruct"
    # path='/home/haiying_he/dataset/coco2017_main_box.json'
    with open(json_path, 'r') as f:
        data_image = json.load(f)
    with open(box_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    with open(main_box_path, 'r', encoding='utf-8') as file_main:
        json_main = json.load(file_main)
    blip_model = BLIPScore()

    pipeline = transformers.pipeline(
                "text-generation",
                model=llama3_7b_chat_hf,
                model_kwargs={"torch_dtype": torch.float16},
                device_map="auto",
            )
    description_merger = DescriptionMerger(pipeline)
    error = blip_model.load_model()
    result_list=[]
    # for image_idx,key_name in tqdm(enumerate(data_image),total=len(data_image)):
    for image_idx in tqdm(range(0,202,1)):
        for i, data in enumerate(json_data):
            if data['image'].split('/')[-1] == data_image[image_idx]['image'].split('/')[-1]:
                image_src=Image.open(data_image[image_idx]['image']).convert('RGB')
                width, height = image_src.size
                main_key='/home/haiying_he/coco_sample_data_Image_Textualization/'+data_image[image_idx]['image'].split('/')[-1]
                four_box = data['four_box']
                left_top = image_src.crop(four_box[0])
                right_top = image_src.crop(four_box[1])
                left_bottom = image_src.crop(four_box[2])
                right_bottom = image_src.crop(four_box[3])
                main_box=image_src.crop(json_main[main_key]['main_box'])
                dict_image={'1':left_top,'2':right_top,'3':left_bottom,'4':right_bottom,'5':main_box}
                
                region_locations=[]
                region_descriptions=[]
                # 遍历字典中的每个键值对
                # print(data_image[image_idx]['local'][0])
                for key, descriptions in data_image[image_idx]['local'][0].items():
                    # 提取 location
                    location_match = re.search(r'location:\[([^\]]+)\]', key)
                    if location_match:
                        location = location_match.group(1).split(', ')
                        location = list(map(float, location))  # 将字符串转为浮点数列表
                    region_locations.append(location)
                    
                    # # 提取 description 列表
                    for i in range(len(descriptions)):
                        # 将 location 和 description 配对并添加到结果中
                        descriptions[i]=descriptions[i].replace('\n','')
                        descriptions[i]=descriptions[i].replace('</s>','')
                    region_descriptions.append(descriptions)
         
                from shapely.geometry import box
                cleaned_regions = []

                # print(region_locations)
                rectangles = [box(region[0], region[1], region[2], region[3]) for region in region_locations]
                rectangles.append(box(0,0,width,height))

                bbox_ops = BoundingBoxOperations()
                iou_values,iou_main = bbox_ops.calculate_iou(rectangles)
                global_modified=data_image[image_idx]['global']
               
                if iou_main[0][1]<0.4:
                    temp_merge_list=[]
                    index_iou=[]
                
                    
                    temp_group=description_merger.group_sameregion_sentence(region_descriptions[4][0],region_descriptions[4][1],region_descriptions[4][2])
                    # region_dict_temp={'1': region_dict[str(number_iou)][0],'2': region_dict[str(number_iou)][1]}
                    categories = {
                        'For Triples Describing the Same Thing': [],
                        'For Contradictory Triples': [],
                        'For Unique Triples': []
                    }
                
                    same_thing_pattern = re.findall(r'- Group \d+ Combined Description: "(.*?)"', temp_group)
                    categories['For Triples Describing the Same Thing'].extend(same_thing_pattern)

                    # 提取 "矛盾的" 内容
                    contradictory_pattern = re.findall(r'- \[(.*?)\]', temp_group, re.DOTALL)
                    contradictory_list = []
                    for item in contradictory_pattern:
                        contradictory_descriptions = re.findall(r'"(.*?)" \(Description \d+\)', item)
                        regions = re.findall(r'"(.*?)" (\(Description \d+\))', item)
                        contradictory_combined = [f'{desc} {region}' for desc, region in regions]
                        contradictory_list.append(contradictory_combined)
                    categories['For Contradictory Triples'].extend(contradictory_list)
                    # 提取 "独特的" 内容并保留 Region 信息
                    unique_pattern = re.findall(r'- "(.*?)" \(Description \d+\)', temp_group)
                    unique_list = re.findall(r'- "(.*?)" (\(Description \d+\))', temp_group)
                    unique_combined = [f'{desc} {region}' for desc, region in unique_list]
                    categories['For Unique Triples'].extend(unique_combined)
                    # # print(categories)

                
                    if categories['For Contradictory Triples'] is not None:
                        similarity_contra_list=[]
                    
                        for triple in categories['For Contradictory Triples']:
                            # region_1=re.findall('Region \d', triple[0])[0]
                            # region_2=re.findall('Region \d', triple[1])[0]
                            similarity_contra=cal_similarity_same(triple,dict_image[str(4+1)],blip_model )
                            similarity_contra_list.append(similarity_contra)
                    similarity_unique=cal_similarity_same(categories['For Unique Triples'],dict_image[str(4+1)],blip_model )
                    # similarity_same=cal_similarity_same(categories['For Triples Describing the Same Thing'],merge_patch )
                    supplement_all=[]
                    supplement_contra=[]
                    supplement_unique=[]
                    supplement_same=categories['For Triples Describing the Same Thing'] 
                    supplement_all=categories['For Triples Describing the Same Thing'] 
                    for i,contra in enumerate(similarity_contra_list):
                        # contra_list=[]
                        # for sim in contra:
                        #     contra_list.append(contra[sim])
                        max_contra=max(contra)

                        if (max_contra>0.3):
                            list_label=contra.index(max_contra)
                            supplement_contra.append(categories['For Contradictory Triples'][i][list_label])
                            supplement_all.append(re.sub(r'\s*\(Description \d+\)', '', categories['For Contradictory Triples'][i][list_label]))

                    for i,sim in enumerate(similarity_unique):
                        if sim>0.3:
                            supplement_unique.append(categories['For Unique Triples'][i])
                            supplement_all.append(re.sub(r'\s*\(Description \d+\)', '', categories['For Unique Triples'][i]))
                    real_mainbox=description_merger.merge_sameregion(region_descriptions[4][0],region_descriptions[4][1],region_descriptions[4][2],supplement_all)
                    global_modified=description_merger.merge_mainbox(data_image[image_idx]['global'],real_mainbox)
                label_iou=[]
                region_dict={'0':[0,1],'1':[1,3],'2':[2,3],'3':[0,2]}
                for i in iou_values:
                    if i[1]>0.4:
                        label_iou.append(1)
                    else:
                        label_iou.append(0)
                if label_iou==[0,0,0,0]:
                    temp_merge_list=[]
                    for region_number in range(len(region_descriptions)):
                        temp_group=description_merger.group_sameregion_sentence(region_descriptions[region_number][0],region_descriptions[region_number][1],region_descriptions[region_number][2])
                        # region_dict_temp={'1': region_dict[str(number_iou)][0],'2': region_dict[str(number_iou)][1]}
                        # print(temp_group)
                        categories = {
                            'For Triples Describing the Same Thing': [],
                            'For Contradictory Triples': [],
                            'For Unique Triples': []
                        }
                    
                        same_thing_pattern = re.findall(r'- Group \d+ Combined Description: "(.*?)"', temp_group)
                        categories['For Triples Describing the Same Thing'].extend(same_thing_pattern)

                        # 提取 "矛盾的" 内容
                        contradictory_pattern = re.findall(r'- \[(.*?)\]', temp_group, re.DOTALL)
                        contradictory_list = []
                        for item in contradictory_pattern:
                            contradictory_descriptions = re.findall(r'"(.*?)" \(Description \d+\)', item)
                            regions = re.findall(r'"(.*?)" (\(Description \d+\))', item)
                            contradictory_combined = [f'{desc} {region}' for desc, region in regions]
                            contradictory_list.append(contradictory_combined)
                        categories['For Contradictory Triples'].extend(contradictory_list)
                        # 提取 "独特的" 内容并保留 Region 信息
                        unique_pattern = re.findall(r'- "(.*?)" \(Description \d+\)', temp_group)
                        unique_list = re.findall(r'- "(.*?)" (\(Description \d+\))', temp_group)
                        unique_combined = [f'{desc} {region}' for desc, region in unique_list]
                        categories['For Unique Triples'].extend(unique_combined)
                        # print(categories)

                       
                        if categories['For Contradictory Triples'] is not None:
                            similarity_contra_list=[]
                    
                            for triple in categories['For Contradictory Triples']:
                                # region_1=re.findall('Region \d', triple[0])[0]
                                # region_2=re.findall('Region \d', triple[1])[0]
                                # print(triple)
                                similarity_contra=cal_similarity_same(triple,dict_image[str(region_number+1)],blip_model )
                                
                                similarity_contra_list.append(similarity_contra)
                                # print(similarity_contra_list)
                        similarity_unique=cal_similarity_same(categories['For Unique Triples'],dict_image[str(region_number+1)],blip_model )
                        # similarity_same=cal_similarity_same(categories['For Triples Describing the Same Thing'],merge_patch )
                        supplement_all=[]
                        supplement_contra=[]
                        supplement_unique=[]
                        supplement_same=categories['For Triples Describing the Same Thing'] 
                        supplement_all=categories['For Triples Describing the Same Thing'] 
                        for i,contra in enumerate(similarity_contra_list):
                            # contra_list=[]
                            # for sim in contra:
                            #     contra_list.append(contra[sim])
                            max_contra=max(contra)

                            if (max_contra>0.3):
                                list_label=contra.index(max_contra)
                                supplement_contra.append(categories['For Contradictory Triples'][i][list_label])
                                supplement_all.append(re.sub(r'\s*\(Description \d+\)', '', categories['For Contradictory Triples'][i][list_label]))

                        for i,sim in enumerate(similarity_unique):
                            if sim>0.3:
                                supplement_unique.append(categories['For Unique Triples'][i])
                                supplement_all.append(re.sub(r'\s*\(Description \d+\)', '', categories['For Unique Triples'][i]))
                        temp_merge=description_merger.merge_sameregion(region_descriptions[region_number][0],region_descriptions[region_number][1],region_descriptions[region_number][2],supplement_all)
                        temp_merge_list.append(temp_merge)
                    # print(temp_merge_list)
                    # if len(temp_merge_list)==4:
                    complete_merge=description_merger.merge_five([width,height],global_modified, region_locations[0], temp_merge_list[0],  region_locations[1], temp_merge_list[1], region_locations[2], temp_merge_list[2],  region_locations[3], temp_merge_list[3])
                    
                else:
                    temp_merge_list=[]
                    index_iou=[]
                    real_region_des=[]
                    for region_number in range(len(region_descriptions)):
                        temp_group=description_merger.group_sameregion_sentence(region_descriptions[region_number][0],region_descriptions[region_number][1],region_descriptions[region_number][2])
                        # region_dict_temp={'1': region_dict[str(number_iou)][0],'2': region_dict[str(number_iou)][1]}
                        categories = {
                            'For Triples Describing the Same Thing': [],
                            'For Contradictory Triples': [],
                            'For Unique Triples': []
                        }
                    
                        same_thing_pattern = re.findall(r'- Group \d+ Combined Description: "(.*?)"', temp_group)
                        categories['For Triples Describing the Same Thing'].extend(same_thing_pattern)

                        # 提取 "矛盾的" 内容
                        contradictory_pattern = re.findall(r'- \[(.*?)\]', temp_group, re.DOTALL)
                        contradictory_list = []
                        for item in contradictory_pattern:
                            contradictory_descriptions = re.findall(r'"(.*?)" \(Description \d+\)', item)
                            regions = re.findall(r'"(.*?)" (\(Description \d+\))', item)
                            contradictory_combined = [f'{desc} {region}' for desc, region in regions]
                            contradictory_list.append(contradictory_combined)
                        categories['For Contradictory Triples'].extend(contradictory_list)
                        # 提取 "独特的" 内容并保留 Region 信息
                        unique_pattern = re.findall(r'- "(.*?)" \(Description \d+\)', temp_group)
                        unique_list = re.findall(r'- "(.*?)" (\(Description \d+\))', temp_group)
                        unique_combined = [f'{desc} {region}' for desc, region in unique_list]
                        categories['For Unique Triples'].extend(unique_combined)
                        # # print(categories)

                    
                        if categories['For Contradictory Triples'] is not None:
                            similarity_contra_list=[]
                        
                            for triple in categories['For Contradictory Triples']:
                                # region_1=re.findall('Region \d', triple[0])[0]
                                # region_2=re.findall('Region \d', triple[1])[0]
                                similarity_contra=cal_similarity_same(triple,dict_image[str(region_number+1)],blip_model )
                                similarity_contra_list.append(similarity_contra)
                        similarity_unique=cal_similarity_same(categories['For Unique Triples'],dict_image[str(region_number+1)],blip_model )
                        # similarity_same=cal_similarity_same(categories['For Triples Describing the Same Thing'],merge_patch )
                        supplement_all=[]
                        supplement_contra=[]
                        supplement_unique=[]
                        supplement_same=categories['For Triples Describing the Same Thing'] 
                        supplement_all=categories['For Triples Describing the Same Thing'] 
                        for i,contra in enumerate(similarity_contra_list):
                            # contra_list=[]
                            # for sim in contra:
                            #     contra_list.append(contra[sim])
                            max_contra=max(contra)

                            if (max_contra>0.3):
                                list_label=contra.index(max_contra)
                                supplement_contra.append(categories['For Contradictory Triples'][i][list_label])
                                supplement_all.append(re.sub(r'\s*\(Description \d+\)', '', categories['For Contradictory Triples'][i][list_label]))

                        for i,sim in enumerate(similarity_unique):
                            if sim>0.3:
                                supplement_unique.append(categories['For Unique Triples'][i])
                                supplement_all.append(re.sub(r'\s*\(Description \d+\)', '', categories['For Unique Triples'][i]))
                        real_region_des.append(description_merger.merge_sameregion(region_descriptions[region_number][0],region_descriptions[region_number][1],region_descriptions[region_number][2],supplement_all))
                    for number_iou in range(len(label_iou)):
                        
                        if label_iou[number_iou]==1:
                            index_iou.append(number_iou)
                            temp_group=description_merger.group_two_sentence(real_region_des[region_dict[str(number_iou)][0]],real_region_des[region_dict[str(number_iou)][1]])
                            region_dict_temp={'1': region_dict[str(number_iou)][0],'2': region_dict[str(number_iou)][1]}
                            categories = {
                                'For Triples Describing the Same Thing': [],
                                'For Contradictory Triples': [],
                                'For Unique Triples': []
                            }
                        
                            same_thing_pattern = re.findall(r'- Group \d+ Combined Description: "(.*?)"', temp_group)
                            categories['For Triples Describing the Same Thing'].extend(same_thing_pattern)

                            # 提取 "矛盾的" 内容
                            contradictory_pattern = re.findall(r'- \[(.*?)\]', temp_group, re.DOTALL)
                            contradictory_list = []
                            for item in contradictory_pattern:
                                contradictory_descriptions = re.findall(r'"(.*?)" \(Region \d+\)', item)
                                regions = re.findall(r'"(.*?)" (\(Region \d+\))', item)
                                contradictory_combined = [f'{desc} {region}' for desc, region in regions]
                                contradictory_list.append(contradictory_combined)
                            categories['For Contradictory Triples'].extend(contradictory_list)
                            # 提取 "独特的" 内容并保留 Region 信息
                            unique_pattern = re.findall(r'- "(.*?)" \(Region \d+\)', temp_group)
                            unique_list = re.findall(r'- "(.*?)" (\(Region \d+\))', temp_group)
                            unique_combined = [f'{desc} {region}' for desc, region in unique_list]
                            categories['For Unique Triples'].extend(unique_combined)
                            # print(categories)

                        
                            if categories['For Contradictory Triples'] is not None:
                                similarity_contra_list=[]
                                region_merge1=region_dict[str(number_iou)]
                                bbox=bbox_ops.merge_boxes(region_locations[region_merge1[0]],region_locations[region_merge1[1]])
                                merge_patch = image_src.crop(bbox)
                                for triple in categories['For Contradictory Triples']:
                                    # region_1=re.findall('Region \d', triple[0])[0]
                                    # region_2=re.findall('Region \d', triple[1])[0]
                                    similarity_contra=cal_similarity_iou(triple,merge_patch,blip_model )
                                    similarity_contra_list.append(similarity_contra)
                            similarity_unique=cal_similarity(dict_image,categories['For Unique Triples'],region_dict_temp,blip_model)
                            # similarity_same=cal_similarity_same(categories['For Triples Describing the Same Thing'],merge_patch )
                            supplement_all=[]
                            supplement_contra=[]
                            supplement_unique=[]
                            supplement_same=categories['For Triples Describing the Same Thing'] 
                            supplement_all=categories['For Triples Describing the Same Thing'] 
                            for i,contra in enumerate(similarity_contra_list):
                                max_contra=max(contra)

                                if (max_contra>0.3):
                                    list_label=contra.index(max_contra)
                                    supplement_contra.append(categories['For Contradictory Triples'][i][list_label])
                                    supplement_all.append(re.sub(r'\s*\(Region \d+\)', '', categories['For Contradictory Triples'][i][list_label]))

                            for i,sim in enumerate(similarity_unique):
                                if sim>0.3:
                                    supplement_unique.append(categories['For Unique Triples'][i])
                                    supplement_all.append(re.sub(r'\s*\(Region \d+\)', '', categories['For Unique Triples'][i]))
                        
                            temp_merge=description_merger.merge_iou([width,height], region_locations[region_dict[str(number_iou)][0]], real_region_des[region_dict[str(number_iou)][0]], region_locations[region_dict[str(number_iou)][1]] ,real_region_des[region_dict[str(number_iou)][1]], supplement_all)
                            temp_merge_list.append(temp_merge)
                    if len(temp_merge_list)==1:
                        region_merge1=region_dict[str(index_iou[0])]
                        temp_loca=[item for item in [0,1,2,3] if item not in region_merge1 ]
                        # print(temp_loca,region_merge1)
                        new_region1=bbox_ops.merge_boxes(region_locations[region_merge1[0]],region_locations[region_merge1[1]])
                        complete_merge=description_merger.merge_four([width,height],global_modified, new_region1, temp_merge_list[0], region_locations[temp_loca[0]], region_descriptions[temp_loca[0]],region_locations[temp_loca[1]], region_descriptions[temp_loca[1]])
                    elif len(temp_merge_list)==2:
                        region_merge1=region_dict[str(index_iou[0])]
                        region_merge2=region_dict[str(index_iou[1])]
                        new_region1=bbox_ops.merge_boxes(region_locations[region_merge1[0]],region_locations[region_merge1[1]])
                        new_region2=bbox_ops.merge_boxes(region_locations[region_merge2[0]],region_locations[region_merge2[1]])
                        complete_merge=description_merger.merge_three([width,height],global_modified, new_region1, temp_merge_list[0], new_region2, temp_merge_list[1])
                    elif len(temp_merge_list)==3:
                        region_merge1=region_dict[str(index_iou[0])]
                        region_merge2=region_dict[str(index_iou[1])]
                        region_merge3=region_dict[str(index_iou[2])]
                        new_region1=bbox_ops.merge_boxes(region_locations[region_merge1[0]],region_locations[region_merge1[1]])
                        new_region2=bbox_ops.merge_boxes(region_locations[region_merge2[0]],region_locations[region_merge2[1]])
                        new_region3=bbox_ops.merge_boxes(region_locations[region_merge3[0]],region_locations[region_merge3[1]])
                        complete_merge=description_merger.merge_four([width,height],global_modified, new_region1, temp_merge_list[0], new_region2, temp_merge_list[1],new_region3, temp_merge_list[2])
                result_json={}
                result_json['image']=data_image[image_idx]['image']
                result_json['description']=complete_merge
                result_list.append(result_json)
                with open(args.output_path, 'w') as json_file:
                        json.dump(result_list, json_file, indent=4)
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and descriptions.")
 
    
    # 添加命令行参数
    parser.add_argument('--json_path', type=str, default='/data/users/ruotian_peng/experement/densecap/result_3_times_5box_llava1.5_10.6.json', help='Path to the JSON file with image data.')
    parser.add_argument('--box_path', type=str, default='/data/users/ruotian_peng/experement/densecap/sample_result_coco200.json', help='Path to the image textualization result file.')
    parser.add_argument('--main_box_path', type=str, default='/data/users/ruotian_peng/experement/densecap/main_box.json', help='Path to the image textualization result file.')
    parser.add_argument('--llama2_model_path', type=str, default='/data/users/ruotian_peng/pretrain/llama-3.1-8b-instruct', help='Path to the LLaMA model.')
    parser.add_argument('--output_path', type=str,  default='/data/users/ruotian_peng/experement/densecap/text_llava1.5.json',help='Path to save the output JSON file.')

    # 解析参数
    args = parser.parse_args()

    # 调用 main 函数并传递参数
    main(args)

