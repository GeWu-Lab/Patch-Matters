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
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
import multiprocessing
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = str(os.environ["LOCAL_RANK"])
import json
class fusion:
    def __init__(self, llm,tokenizer,blip_model,data_image):
        self.llm=llm
        self.tokenizer=tokenizer
        self.blip_model=blip_model
        self.data_image=data_image

    def merge_iou(self,image_size,region1_location,region1_description,region2_location,region2_description,supplement):
        system = """
            You are a language model tasked with generating a coherent and hallucination-free caption based on the visual content of two image regions. You are provided with detailed descriptions of these regions along with a list of reliable details extracted from the visual content. Your goal is to combine the information from both descriptions and the reliable content list to generate a unified caption that accurately represents the merged visual content of both regions.

            ### Information Provided:

            1. **Image Details**:
            - Image Size: {width, height}

            2. **Region 1**:
            - Location: {region1_location}
            - Description: {region1_description}

            3. **Region 2**:
            - Location: {region2_location}
            - Description: {region2_description}

            4. **Reliable Content List**:
            - A list of highly reliable and consistent details extracted from both Region 1 and Region 2:
                - {reliable_content_list}

            ### Instructions:

            - **Step 1**: Use the reliable content list to establish the foundation of the final caption, ensuring that only trustworthy information is included.
            - **Step 2**: Cross-reference the descriptions of Region 1 and Region 2 to enhance the caption, ensuring that the final description is coherent and accurately merges the visual content of both regions.
            - **Step 3**: Generate a final, hallucination-free caption that avoids any contradictions or conflicting information, while ensuring the description remains clear and cohesive.

            ### Example Scenario:

            #### Image Details:
            - Image Size: [1024, 768]

            #### Region 1:
            - Location: [150, 250, 550, 650]
            - Description: "A man in a green jacket is standing near a large tree, with a park bench nearby. He is holding a small book, and there are flowers around the base of the tree. The scene suggests a calm, outdoor setting."

            #### Region 2:
            - Location: [600, 250, 1000, 650]
            - Description: "A man in a green jacket is sitting on a park bench next to a tree, holding a book. The bench is surrounded by flowers, and there is a small bird perched on the back of the bench. The atmosphere feels peaceful, and the weather appears clear."

            #### Reliable Content List:
            - 'A man in a green jacket is near a tree.'
            - 'The man is holding a book.'
            - 'There are flowers around the tree.'
            - 'The man is near or sitting on a park bench.'
            - 'A bird is perched on the back of the bench.'
            - 'The atmosphere is peaceful and calm.'

            #### Example Output:
            "A man in a green jacket is holding a small book. Flowers surround the base of the tree, and a bird is perched on the back of the bench. The scene suggests a peaceful, calm outdoor environment, with the man seemingly enjoying a moment of quiet reflection."
            """


        user=f"""
            ### Your Task:
            Generate a caption that accurately reflects the most reliable information from the provided triples and image regions, ensuring that no contradictory information is included. Do not include any explanations or thought processes, directly output the final caption without any prefixes.

            #### Input:
            Image Size: {image_size}

            Region 1:
            - Location: {region1_location}
            - Description: {region1_description}

            Region 2:
            - Location: {region2_location}
            - Description: {region2_description}

            Reliable Content List:
            - {supplement}
            """
        messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]
        sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=4096,stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")])
        text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
        output = self.llm.generate([text], sampling_params)
        generated_text= output[0].outputs[0].text
        
        return generated_text
    def batch_merge_iou(self,image_size,region_locations,real_region_des,supplement,label_iou):
        system = """
            You are a language model tasked with generating a coherent and hallucination-free caption based on the visual content of two image regions. You are provided with detailed descriptions of these regions along with a list of reliable details extracted from the visual content. Your goal is to combine the information from both descriptions and the reliable content list to generate a unified caption that accurately represents the merged visual content of both regions.

            ### Information Provided:

            1. **Image Details**:
            - Image Size: {width, height}

            2. **Region 1**:
            - Location: {region1_location}
            - Description: {region1_description}

            3. **Region 2**:
            - Location: {region2_location}
            - Description: {region2_description}

            4. **Reliable Content List**:
            - A list of highly reliable and consistent details extracted from both Region 1 and Region 2:
                - {reliable_content_list}

            ### Instructions:

            - **Step 1**: Use the reliable content list to establish the foundation of the final caption, ensuring that only trustworthy information is included.
            - **Step 2**: Cross-reference the descriptions of Region 1 and Region 2 to enhance the caption, ensuring that the final description is coherent and accurately merges the visual content of both regions.
            - **Step 3**: Generate a final, hallucination-free caption that avoids any contradictions or conflicting information, while ensuring the description remains clear and cohesive.

            ### Example Scenario:

            #### Image Details:
            - Image Size: [1024, 768]

            #### Region 1:
            - Location: [150, 250, 550, 650]
            - Description: "A man in a green jacket is standing near a large tree, with a park bench nearby. He is holding a small book, and there are flowers around the base of the tree. The scene suggests a calm, outdoor setting."

            #### Region 2:
            - Location: [600, 250, 1000, 650]
            - Description: "A man in a green jacket is sitting on a park bench next to a tree, holding a book. The bench is surrounded by flowers, and there is a small bird perched on the back of the bench. The atmosphere feels peaceful, and the weather appears clear."

            #### Reliable Content List:
            - 'A man in a green jacket is near a tree.'
            - 'The man is holding a book.'
            - 'There are flowers around the tree.'
            - 'The man is near or sitting on a park bench.'
            - 'A bird is perched on the back of the bench.'
            - 'The atmosphere is peaceful and calm.'

            #### Example Output:
            "A man in a green jacket is holding a small book. Flowers surround the base of the tree, and a bird is perched on the back of the bench. The scene suggests a peaceful, calm outdoor environment, with the man seemingly enjoying a moment of quiet reflection."
            """

        prompt=[]
        region_dict={'0':[0,1],'1':[1,3],'2':[2,3],'3':[0,2]}
        id_supply=0
        for number_iou in range(len(label_iou)):
                        
            if label_iou[number_iou]==1:
          
                user=f"""
                ### Your Task:
                Generate a caption that accurately reflects the most reliable information from the provided triples and image regions, ensuring that no contradictory information is included. Do not include any explanations or thought processes, directly output the final caption without any prefixes.

                #### Input:
                Image Size: {image_size}

                Region 1:
                - Location: {region_locations[region_dict[str(number_iou)][0]]}
                - Description: {real_region_des[region_dict[str(number_iou)][0]]}

                Region 2:
                - Location: {region_locations[region_dict[str(number_iou)][1]]}
                - Description: {real_region_des[region_dict[str(number_iou)][1]]}

                Reliable Content List:
                - {supplement[id_supply]}
                """
                messages = [
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                            ]
                text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
                )
                prompt.append(text)
                id_supply+=1
        
        sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=4096,stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")])
      
        output = self.llm.generate(prompt, sampling_params)
        # generated_text= output[0].outputs[0].text
        
        return output
    def normalize_box(self,box, img_width, img_height):
        x_min, y_min, x_max, y_max = box
        # 归一化边界框
        x_min_normalized = x_min / img_width
        y_min_normalized = y_min / img_height
        x_max_normalized = x_max / img_width
        y_max_normalized = y_max / img_height
        
        # 返回归一化后的边界框
        return [x_min_normalized, y_min_normalized, x_max_normalized, y_max_normalized]
    def merge_box(self,box1,box2):
        x_min = min(box1[0], box2[0])
        y_min = min(box1[1], box2[1])
        x_max = max(box1[2], box2[2])
        y_max = max(box1[3], box2[3])

    # 合并后的边框表示
        merged_box = [x_min, y_min, x_max, y_max]
        return merged_box
    def merge_four(self,image_size, global_description, region1_location, region1_description, region2_location, region2_description,region3_location,region3_description):
        
        system ="""
    ###Input### 
    • You will receive a global caption describing an image. 
    • Additionally, you will have access to **three region captions** generated for specific regions within the image, along with their specific location information. 
    • Both global and region captions may contain noise or errors.

    ###Task Objective### 
    • Your goal is to modify and enhance the global caption by integrating accurate details from the **three region captions** and their respective locations. 
    • The global caption should be enriched with specific, accurate details from the regions and corrected where necessary. 
    • Focus on using the region captions to correct any inaccuracies or hallucinations in the global caption. 
    • The updated global caption must contain more detail than the original global caption by including relevant information from the region captions. 
    • You only give the updated global caption as output, without any additional information. 
    • Do NOT give any explanation or notes on how you generate this caption.

    ###Input INFORMATION EXPLANATION### 
    1.Global Description: It provides the initial global image description, which captures the primary semantic information of the image. However, some of the described objects may be hallucinated, and certain details are either missing or insufficiently described, requiring additional information for correction and enhancement.
    2.**Three Region Descriptions**: They provide descriptions of different regions, focusing on specific parts of the image. These regions include more detailed object features and finer details. These descriptions have undergone hallucination filtering, making them more reliable than the global description.
    3.Region Locations: It uses a normalized coordinate system where both x (horizontal) and y (vertical) axes range from 0 to 1. The x-coordinate starts at 0 on the image’s left edge and increases to 1 towards the right edge. Similarly, the y-coordinate starts at 0 at the top edge and increases to 1 towards the bottom. This system uses four coordinates to define the corners of a rectangle within the image: [x1, y1, x2, y2], representing the top-left and bottom-right corners of the rectangle, respectively.

    ###Guidelines### 
    • Through the extra information of the three regions, some objects may represent the same thing. When adding objects to the original description, it is important to avoid duplication. 
    • Combine Information: Extract and integrate key details from both the global and region captions, giving priority to the region captions for more specific or accurate details. 
    • Modify and Enhance: Add relevant details from the region captions to enrich the global description. Correct any hallucinations or inaccuracies in the global caption using the region captions. 
    • Consider Location: Ensure that spatial information from the region captions is incorporated to provide a more coherent and accurate description of the image. 
    • Filter Noise: Remove any inaccurate, irrelevant, or conflicting information from the global caption, especially when the region captions offer more precise details. 
    • Enhance Detail: Ensure that the final global caption contains more detailed and refined visual information than the original, using the region captions to add specificity.

    ###In-Context Examples### 
    [Chain of thought is placed within a pair of "@@@" (only in the examples will you be provided with a chain of thoughts to help you understand; in the actual task, these will not be given to you.)] 

    ###Example1:### 
    Global Description: 
    A busy street with many cars and pedestrians. There are buildings on both sides, and the sun is shining brightly.

    **Region1**: 
    - Location:[0.10, 0.20, 0.50, 0.60] 
    - Description: A red car is parked by the sidewalk near a large tree. A man in a yellow jacket is walking by, holding a newspaper in his hand. 

    **Region2**: 
    - Location:[0.60, 0.10, 0.90, 0.50] 
    - Description: A group of four people are waiting at a crosswalk. One of them is pushing a stroller, while the others are looking at their phones. 

    **Region3**: 
    - Location:[0.30, 0.70, 0.70, 0.90] 
    - Description: A cyclist in a blue shirt is riding past a store with a large display window. The store is selling colorful balloons, and there are several customers browsing inside.

    @@@ Chain of Thought: 
    1.The global description mentions a busy street with cars and pedestrians, which is generally accurate. However, the region descriptions provide more specific details, such as the red car, the man in the yellow jacket, and the group of people waiting at the crosswalk. These details should be incorporated into the global description. 
    2.The original global caption does not mention the cyclist or the store, both of which are important features described in Region 3. The description of the cyclist and the store with colorful balloons should be added for accuracy and specificity.
    3.The locations of the objects in each region (e.g., red car on the left, group of people on the right) should be reflected to ensure spatial coherence. 
    @@@

    Your Modified Description: 
    A busy street with cars and pedestrians under a bright sun. On the left side of the street, a red car is parked by the sidewalk next to a large tree, with a man in a yellow jacket walking by, holding a newspaper. On the right side, a group of four people are waiting at a crosswalk, one of them pushing a stroller while the others are looking at their phones. In the middle of the street, a cyclist in a blue shirt rides past a store with a large display window, where colorful balloons are on sale and several customers are browsing inside.

    ###Example2:### 
    Global Description: 
    A beach scene with people enjoying the sunny weather. Some are swimming, and others are relaxing on the sand. 

    **Region1**: 
    - Location:[0.15, 0.30, 0.50, 0.65] 
    - Description: A family is sitting on a large beach towel. The mother is wearing a pink hat, and the father is holding a beach ball, while their two children are playing with a bucket and spade.

    **Region2**: 
    - Location:[0.60, 0.20, 0.80, 0.50] 
    - Description: Two surfers are standing near the water, holding their surfboards. The waves are crashing against the shore behind them.

    **Region3**: 
    - Location:[0.70, 0.70, 0.90, 0.90] 
    - Description: A group of teenagers are playing volleyball near the edge of the beach. One girl is about to serve the ball, while the others are standing ready to play.

    @@@ Chain of Thought: 
    1.The global description provides a general overview of the beach scene, but lacks the specific details from the region descriptions. 
    2.Region 1 mentions a family on the beach, which should be added to enrich the description. Region 2 describes surfers near the water, and Region 3 details teenagers playing volleyball. These should all be included to provide a more detailed picture of the beach activity. 
    3.The spatial locations of the different groups should also be included for coherence. 
    @@@

    Your Modified Description: 
    A sunny beach where people are enjoying the weather. In the center of the beach, a family sits on a large towel. The mother, wearing a pink hat, is sitting next to the father, who holds a beach ball, while their two children play with a bucket and spade. Near the water on the right side, two surfers stand with their surfboards as waves crash against the shore. In the background, a group of teenagers are playing volleyball, with one girl preparing to serve the ball while the others stand ready.
    """


        print(image_size)
        image_width=image_size[0]
        image_height=image_size[1]
        region1_location=self.normalize_box(region1_location,image_width,image_height)
        region2_location=self.normalize_box(region2_location,image_width,image_height)
        user=f"""
    ###TASK### 
    Please provide the modified description directly. 
    Global Description: 
    {global_description}
    Region1
    - Location: {region1_location}
    - Description: {region1_description}
    Region2
    - Location: {region2_location}
    - Description: {region2_description}
            """
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        
        sampling_params = SamplingParams(temperature=0.2, top_p=0.9,max_tokens=4096,stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")])
        text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
        output= self.llm.generate([text], sampling_params)
        generated_text= output[0].outputs[0].text
        
        return generated_text
        

    def merge_three(self,image_size, global_description, region1_location, region1_description, region2_location, region2_description):
        
        system ="""
    ###Input###
    • You will receive a global caption describing an image.
    • Additionally, you will have access to region captions generated for specific regions within the image, along with their specific location information.
    • Both global and local captions may contain noise or errors.

    ###Task Objective###
    • Your goal is to modify and enhance the global caption by integrating accurate details from the region captions and its location. 
    • The global caption should be enriched with specific, accurate details from the regions and corrected where necessary.
    • Focus on using the region captions to correct any inaccuracies or hallucinations in the global caption.
    • The updated global caption must contain more detail than the original global caption by including relevant information from the region captions.
    • You only give the updated global caption as output, without any additional information.
    • Do NOT give any explanation or notes on how you generate this caption.
        
    ###Input INFORMATION EXPLANATION###
    1.Global Description: It provides the initial global image description, which captures the primary semantic information of the image. However, some of the described objects are hallucinated, and certain details are either missing or insufficiently described, requiring additional information for correction and enhancement
    2.Region Description: It provides descriptions of different regions, focusing on only specific parts of the image. As a result, it includes more detailed object features and finer details. Additionally, this section has undergone hallucination filtering, making the descriptions more reliable compared to the global description.
    3.Region Location: It uses a normalized coordinate system where both x (horizontal) and y (vertical) axes range from 0 to 1. The x-coordinate starts at 0 on the image’s left edge and increases to 1 towards the right edge. Similarly, the y-coordinate starts at 0 at the top edge and increases to 1 towards the bottom. This system uses four coordinates to define the corners of a rectangle within the image: [x1, y1, x2, y2], representing the top-left and bottom-right corners of the rectangle, respectively. For instance, a positioning of [0.00, 0.00, 0.50, 0.50] means the object’s top-left corner is at (0.00, 0.00) and its bottom-right corner is at (0.50, 0.50), placing the object in the upper left quarter of the image. Similarly, [0.50, 0.00, 1.00, 0.50] positions the object in the upper right quarter, with corners at (0.50, 0.00) and (1.00, 0.50). A positioning of [0.00, 0.50, 0.50, 1.00] places the object in the bottom left quarter, with corners at (0.00, 0.50) and (0.50, 1.00), while [0.50, 0.50, 1.00, 1.00] positions it in the bottom right quarter, with corners at (0.50, 0.50) and (1.00, 1.00). Moreover, by comparing these coordinates, you can determine the relative positions of objects. For example, an object with positioning [0.20, 0.20, 0.40, 0.40] is to the left of another with [0.30, 0.30, 0.50, 0.50].

    ###Guidelines###
    • Through the extra information of different regions, some Objects may represent the same thing. When adding Objects to the Original Description, it is important to avoid duplication.
    • Combine Information: Extract and integrate key details from both the global and local (region) captions, giving priority to the region captions for more specific or accurate details.
    • Modify and Enhance: Add relevant details from the region captions to enrich the global description. Correct any hallucinations or inaccuracies in the global caption using the region captions.
    • Consider Location: Ensure that spatial information from the region captions is incorporated to provide a more coherent and accurate description of the image.
    • Filter Noise: Remove any inaccurate, irrelevant, or conflicting information from the global caption, especially when region captions offer more precise details.
    • Enhance Detail: Ensure that the final global caption contains more detailed and refined visual information than the original, using the region captions to add specificity.

    ###In-Context Examples###
    [Chain of thought is placed within a pair of "@@@" (remember only in the Examples will you be provided with a chain of thoughts to help you understand; in the actual task, these will not be given to you.)]
    ###Example1:###
    Global Description:
    A large open-air market is bustling with activity. People are walking around, browsing stalls under colorful tents, and a mix of goods, such as fruits, vegetables, and clothing, is being sold. The sky is clear, and the sun is shining brightly. A food truck is parked in the middle of the market, serving hot food to people waiting in line.

    Region1:
    - Location:[0.05, 0.20, 0.40, 0.60]
    - Description:A fruit stand with neatly arranged piles of oranges, bananas, and apples. A vendor wearing a green apron is standing behind the stall, handing a bag of oranges to a woman in a blue dress. The woman is holding a wicker basket, and there are several other customers browsing nearby.

    Region2:
    - Location:[0.50, 0.10, 0.80, 0.50]
    - Description:A clothing stall displaying a variety of colorful scarves and hats hanging on racks. A group of three teenagers, two girls and one boy, are looking at the items. One girl is holding a red scarf, while the boy is trying on a wide-brimmed hat. The vendor, a middle-aged man with a beard, is standing by, smiling.

    @@@ Chain of Thought: 
    1.The global description accurately captures the bustling nature of the market and the sunny weather, both of which are supported by the region descriptions. These details should be retained. However, the mention of a food truck is not supported by the region descriptions and should be removed. 
    2.Additionally, the reference to colorful tents is not confirmed by the regions, so it should either be removed or reworded to reflect the actual scene. The region descriptions provide valuable specific details. Region 1 describes a fruit stand located on the left side of the market ([0.05, 0.20, 0.40, 0.60]), with key details like the vendor’s green apron, the types of fruits being sold, and the interaction with a woman carrying a wicker basket. Region 2 focuses on a clothing stall on the right side of the market ([0.50, 0.10, 0.80, 0.50]), detailing the colorful scarves and hats on display and the interactions between the vendor and three teenagers. These details should be seamlessly integrated into the updated global caption to provide more accuracy and spatial coherence.
    @@@

    Your Modified Description:
    A large open-air market is bustling with activity under a clear, sunny sky. People are walking around, browsing various stalls where goods such as fruits, vegetables, and clothing are being sold. On the left side of the market, a fruit stand displays neatly arranged piles of oranges, bananas, and apples, with a vendor in a green apron handing a bag of oranges to a woman in a blue dress holding a wicker basket. Several other customers are browsing the fresh produce nearby. Meanwhile, on the right side of the market, a clothing stall showcases colorful scarves and hats. Two girls and a boy are examining the items, with one girl holding a red scarf while the boy tries on a wide-brimmed hat. The vendor, smiling, stands nearby.

    ###Example2:###
    Global Description:
    Three friends are sitting on a bench in the park, chatting and laughing. The sun is shining brightly, and people are scattered around the park, enjoying the weather. A man is jogging along the path, and there’s a pond with ducks swimming nearby.

    Region1:
    - Location:[0.10, 0.25, 0.40, 0.60]
    - Description:Two women are sitting on a bench under a tree. One is wearing a blue T-shirt and shorts, while the other is dressed in a white sundress. They are chatting and laughing, with one of the women holding a cup of coffee. There’s a picnic blanket on the ground near the bench with some snacks on it.
    Region2:
    - Location:[0.50, 0.10, 0.80, 0.50]
    - Description: A man in a green T-shirt and jeans is standing next to the bench, holding a water bottle in one hand while looking at the two women. He seems to be engaged in their conversation, smiling and occasionally glancing at his phone.

    @@@ Chain of Thought: 
    1.The global description mentions “Three friends are sitting on a bench in the park, chatting and laughing.”, while Region 1 describes two women sitting there and Region 2 describes a man standing nearby. There is a man standing next to the bench. This suggests that the global description is not of three friends sitting and talking together; it should be two women sitting on a bench with a man standing next to them as they talk. Therefore the three friends sitting on a bench in the original text should be changed to, two women sitting on a bench with a man standing next to them.
    2.The global description also mentions a man jogging and a pond with ducks, neither of which are supported by the region description and should therefore be removed. The area descriptions provide more specific details, such as the coffee cup, the woman's dress, and the picnic blanket, all of which add a richer connotation to the final title. The updated title should focus on accurately portraying these characters and their interactions.
    @@@

    Your Modified Description:
    In a sunny park, two women sit on a bench under the shade of a large tree, chatting and laughing. One woman, dressed in a blue T-shirt and shorts, holds a cup of coffee, smiling as she shares a story. The other woman, in a white sundress, listens attentively with a warm smile. Nearby, a man in a green T-shirt and jeans stands holding a water bottle and occasionally checks his phone but remains engaged in their conversation, smiling and chiming in now and then. A picnic blanket is spread out on the grass beside the bench, with some snacks on it, adding to the relaxed atmosphere.
    """

        print(image_size)
        image_width=image_size[0]
        image_height=image_size[1]
        region1_location=self.normalize_box(region1_location,image_width,image_height)
        region2_location=self.normalize_box(region2_location,image_width,image_height)
        user=f"""
    ###TASK### 
    Please provide the modified description directly. 
    Global Description: 
    {global_description}
    Region1
    - Location: {region1_location}
    - Description: {region1_description}
    Region2
    - Location: {region2_location}
    - Description: {region2_description}
            """
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        
        sampling_params = SamplingParams(temperature=0.2, top_p=0.9,max_tokens=4096,stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")])
        text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
        output = self.llm.generate([text], sampling_params)
        generated_text= output[0].outputs[0].text
        
        return generated_text
    def batch_group_two_sentence(self,batch_group_two):
        system = """
        You are a language modeler tasked with analyzing two passages describing different areas of the same picture. Your goal is to process these descriptions step by step, reasoning through each task logically and systematically. Please follow the steps outlined below carefully and directly output the final results of Step 4 without displaying other words.

        Guidelines:

        
        - Step 1: Identifying Similar Descriptions
        1. Deep Comparison of Phrases: For each sentence in the descriptions, identify whether it refers to the same object, relationship, or action across different regions. Pay close attention to both direct mentions of the object and indirect references (e.g., "A man in a black t-shirt sits by the table" and "Someone wearing a black t-shirt is sitting by the table" should be considered as referring to the same object). Use semantic similarity and context clues to ensure sentences describing the same object are grouped together.
        2. Multiple Expression Handling: Group all sentences that describe the same object, relationship, or action, even if they involve different perspectives (e.g., active vs. passive voice, different levels of detail, or indirect references). Make sure that subtle differences in wording do not prevent grouping of similar descriptions, and ensure that any sentence identified as semantically similar will not appear in other groupings.
        3. Extra Information Should Not Be Merged: If one sentence includes additional information or attributes that are not present in the other sentence, do not merge them as the same description. Only group sentences that describe the same object, relationship, or action with similar levels of detail. Sentences with extra information should be considered separately for the unique category if applicable.

        - Step 2: Identifying Contradictory Descriptions
        1. Focus on Contradictions Around the Same Object and Attribute: Identify sentences that describe the same object and the same attribute (e.g., color, size, action) but provide conflicting information. For example, two sentences describing the color of a person's shirt, but with different colors, would be considered contradictory. Ensure that only descriptions of the same attribute for the same object are considered contradictory.
        2. Present Contradictory Pairs Without Modification: Output both contradictory sentences exactly as they appear without modifying or simplifying them. 
        3. Ensure Exclusivity from Similar Descriptions: If a sentence has already been identified as part of a similar description group in Step 1, it should not be considered again for this step.
        4. Attribute Absence Does Not Indicate Contradiction: If one sentence describes an attribute of an object (e.g., the color of a shirt) and the other sentence does not describe that attribute, this does not count as a contradiction. Instead, the sentence with the additional attribute description should be considered for the unique category if applicable.
    - Step 3: Identifying Unique Descriptions
        1. Identify Unique Descriptions: Look for details that appear only once across both regions and do not belong to any previously identified groups (similar or contradictory).
        2. Assess Importance: Focus on descriptions that provide specific or essential information about objects, relationships, or attributes in the scene, and ensure that they have not been classified in previous steps.

        - Step 4: Synthesizing and Refining the Output
        1. For Similar Descriptions: Synthesize a new sentence that uses only the shared semantics from both descriptions. Ensure that this new sentence is coherent and accurately represents the common information.
        2. For Contradictory Descriptions: Present contradictory pairs clearly, indicating the exact sentences that contradict each other, without modification. Ensure that each contradictory sentence pair is followed by its corresponding region (e.g., ["The sky is clear with a few scattered clouds." (Region 1), "The sky is filled with vibrant orange and pink hues." (Region 2)]).
        3. For Unique Descriptions: List the unique descriptions that are most critical to the scene, specifying the region they belong to.
        4. Direct Output of Final Results: Once you have completed each step, output only the final results for each group (similar, contradictory, unique) without providing explanations, thoughts, or reflections on your reasoning process. Ensure that only the results are shown, without additional commentary.

        Remember to directly output the final results of Step 4 without displaying other words.

        ### Example1:

        ###Input Descriptions:

        
        Region 1: "A tall man in a black suit is standing under a large oak tree. The sun is setting, casting an orange glow over the landscape. In the background, a woman in a red dress is walking along a dirt path, her hair flowing in the wind. The sky is clear with a few scattered clouds, and a dog runs playfully in the grass nearby."

        Region 2: "A man wearing a dark suit stands beneath an old oak tree as the sun sets. The sky is filled with vibrant orange and pink hues. In the distance, a woman in a red dress strolls down a dirt path, her hair blowing in the breeze. A brown dog plays in the grassy field next to her."

        ###Output:

        For Similar Descriptions:
        - Group 1 Combined Description: "A man in a dark suit is standing under an oak tree as the sun sets."
        - Group 2 Combined Description: "A woman in a red dress walks along a dirt path, her hair blowing in the wind."
        - Group 3 Combined Description: "A dog is running/playing in the grassy field nearby."

        For Contradictory Descriptions:
        - ["The sky is clear with a few scattered clouds." (Region 1), "The sky is filled with vibrant orange and pink hues." (Region 2)]

        For Unique Descriptions:
        - "The sun is casting an orange glow over the landscape." (Region 1)
        - "A brown dog plays in the grassy field next to her." (Region 2)

        ###Example2:

        ### Input Descriptions:

            Region 1: "A group of children are playing soccer on a green field, with a tall tree casting a long shadow nearby. The sky is bright and clear, and in the background, a large red building stands next to a busy road. A dog is barking at the children from the side of the field. A bicycle is lying on the ground near the tree."

            Region 2: "Several children are playing a soccer game on a grassy field. A large tree is standing near the field, and its shadow falls over part of the grass. In the background, a big red building is situated beside a quiet street. The sky is filled with clouds. A dog is running around the field, and a bicycle is parked near the tree."

        ### Output:

        For Similar Descriptions:
        - Group 1 Combined Description: "Several children are playing soccer on a green field."
        - Group 2 Combined Description: "A tall tree casts a shadow nearby."
        - Group 3 Combined Description: "A large red building stands beside a road in the background."
        - Group 4 Combined Description: "A dog is near the field."
        - Group 5 Combined Description: "A bicycle is near the tree."

        For Contradictory Descriptions:
        - ["The sky is bright and clear." (Region 1), "The sky is filled with clouds." (Region 2)]
        - ["The road is busy." (Region 1), "The street is quiet." (Region 2)]
        - ["The dog is barking." (Region 1), "The dog is running around." (Region 2)]

        For Unique Descriptions:
        - "A bicycle is lying on the ground near the tree." (Region 1)
        - "A bicycle is parked near the tree." (Region 2)
        """
        prompt=[]
     
        for i in range(len(batch_group_two)):
          
            user=f"""
                    ###Region 1: {batch_group_two[i][0]}

                    ###Region 2: {batch_group_two[i][1]}

                    Please directly output the final results of Step 4 without displaying the intermediate steps, strictly follow the example output and do not include any additional comments..
                    """
            messages = [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ]
            text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
            prompt.append(text)
        
        sampling_params = SamplingParams(temperature=0.2, top_p=0.9,max_tokens=4096,stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")])
        output = self.llm.generate(prompt, sampling_params)
        # generated_text= output[0].outputs[0].text
        
        return output 
    def group_two_sentence(self,description1,description2):
        system = """
        You are a language modeler tasked with analyzing two passages describing different areas of the same picture. Your goal is to process these descriptions step by step, reasoning through each task logically and systematically. Please follow the steps outlined below carefully and directly output the final results of Step 4 without displaying other words.

        Guidelines:

        
        - Step 1: Identifying Similar Descriptions
        1. Deep Comparison of Phrases: For each sentence in the descriptions, identify whether it refers to the same object, relationship, or action across different regions. Pay close attention to both direct mentions of the object and indirect references (e.g., "A man in a black t-shirt sits by the table" and "Someone wearing a black t-shirt is sitting by the table" should be considered as referring to the same object). Use semantic similarity and context clues to ensure sentences describing the same object are grouped together.
        2. Multiple Expression Handling: Group all sentences that describe the same object, relationship, or action, even if they involve different perspectives (e.g., active vs. passive voice, different levels of detail, or indirect references). Make sure that subtle differences in wording do not prevent grouping of similar descriptions, and ensure that any sentence identified as semantically similar will not appear in other groupings.
        3. Extra Information Should Not Be Merged: If one sentence includes additional information or attributes that are not present in the other sentence, do not merge them as the same description. Only group sentences that describe the same object, relationship, or action with similar levels of detail. Sentences with extra information should be considered separately for the unique category if applicable.

        - Step 2: Identifying Contradictory Descriptions
        1. Focus on Contradictions Around the Same Object and Attribute: Identify sentences that describe the same object and the same attribute (e.g., color, size, action) but provide conflicting information. For example, two sentences describing the color of a person's shirt, but with different colors, would be considered contradictory. Ensure that only descriptions of the same attribute for the same object are considered contradictory.
        2. Present Contradictory Pairs Without Modification: Output both contradictory sentences exactly as they appear without modifying or simplifying them. 
        3. Ensure Exclusivity from Similar Descriptions: If a sentence has already been identified as part of a similar description group in Step 1, it should not be considered again for this step.
        4. Attribute Absence Does Not Indicate Contradiction: If one sentence describes an attribute of an object (e.g., the color of a shirt) and the other sentence does not describe that attribute, this does not count as a contradiction. Instead, the sentence with the additional attribute description should be considered for the unique category if applicable.
    - Step 3: Identifying Unique Descriptions
        1. Identify Unique Descriptions: Look for details that appear only once across both regions and do not belong to any previously identified groups (similar or contradictory).
        2. Assess Importance: Focus on descriptions that provide specific or essential information about objects, relationships, or attributes in the scene, and ensure that they have not been classified in previous steps.

        - Step 4: Synthesizing and Refining the Output
        1. For Similar Descriptions: Synthesize a new sentence that uses only the shared semantics from both descriptions. Ensure that this new sentence is coherent and accurately represents the common information.
        2. For Contradictory Descriptions: Present contradictory pairs clearly, indicating the exact sentences that contradict each other, without modification. Ensure that each contradictory sentence pair is followed by its corresponding region (e.g., ["The sky is clear with a few scattered clouds." (Region 1), "The sky is filled with vibrant orange and pink hues." (Region 2)]).
        3. For Unique Descriptions: List the unique descriptions that are most critical to the scene, specifying the region they belong to.
        4. Direct Output of Final Results: Once you have completed each step, output only the final results for each group (similar, contradictory, unique) without providing explanations, thoughts, or reflections on your reasoning process. Ensure that only the results are shown, without additional commentary.

        Remember to directly output the final results of Step 4 without displaying other words.

        ### Example1:

        ###Input Descriptions:

        
        Region 1: "A tall man in a black suit is standing under a large oak tree. The sun is setting, casting an orange glow over the landscape. In the background, a woman in a red dress is walking along a dirt path, her hair flowing in the wind. The sky is clear with a few scattered clouds, and a dog runs playfully in the grass nearby."

        Region 2: "A man wearing a dark suit stands beneath an old oak tree as the sun sets. The sky is filled with vibrant orange and pink hues. In the distance, a woman in a red dress strolls down a dirt path, her hair blowing in the breeze. A brown dog plays in the grassy field next to her."

        ###Output:

        For Similar Descriptions:
        - Group 1 Combined Description: "A man in a dark suit is standing under an oak tree as the sun sets."
        - Group 2 Combined Description: "A woman in a red dress walks along a dirt path, her hair blowing in the wind."
        - Group 3 Combined Description: "A dog is running/playing in the grassy field nearby."

        For Contradictory Descriptions:
        - ["The sky is clear with a few scattered clouds." (Region 1), "The sky is filled with vibrant orange and pink hues." (Region 2)]

        For Unique Descriptions:
        - "The sun is casting an orange glow over the landscape." (Region 1)
        - "A brown dog plays in the grassy field next to her." (Region 2)

        ###Example2:

        ### Input Descriptions:

            Region 1: "A group of children are playing soccer on a green field, with a tall tree casting a long shadow nearby. The sky is bright and clear, and in the background, a large red building stands next to a busy road. A dog is barking at the children from the side of the field. A bicycle is lying on the ground near the tree."

            Region 2: "Several children are playing a soccer game on a grassy field. A large tree is standing near the field, and its shadow falls over part of the grass. In the background, a big red building is situated beside a quiet street. The sky is filled with clouds. A dog is running around the field, and a bicycle is parked near the tree."

        ### Output:

        For Similar Descriptions:
        - Group 1 Combined Description: "Several children are playing soccer on a green field."
        - Group 2 Combined Description: "A tall tree casts a shadow nearby."
        - Group 3 Combined Description: "A large red building stands beside a road in the background."
        - Group 4 Combined Description: "A dog is near the field."
        - Group 5 Combined Description: "A bicycle is near the tree."

        For Contradictory Descriptions:
        - ["The sky is bright and clear." (Region 1), "The sky is filled with clouds." (Region 2)]
        - ["The road is busy." (Region 1), "The street is quiet." (Region 2)]
        - ["The dog is barking." (Region 1), "The dog is running around." (Region 2)]

        For Unique Descriptions:
        - "A bicycle is lying on the ground near the tree." (Region 1)
        - "A bicycle is parked near the tree." (Region 2)
        """
        user=f"""
                    ###Region 1: {description1}

                    ###Region 2: {description2}

                    Please directly output the final results of Step 4 without displaying the intermediate steps, strictly follow the example output and do not include any additional comments..
                    """
        messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]
        sampling_params = SamplingParams(temperature=0.2, top_p=0.9,max_tokens=4096,stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")])
        text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
        output = self.llm.generate([text], sampling_params)
        generated_text= output[0].outputs[0].text
        
        return generated_text
    def batch_merge_sameregion_main(self,region_descriptions,supplement):
        system = """
            You are a language model tasked with generating a coherent, detailed, and hallucination-free description based on the visual content of three areas. You are provided with detailed descriptions of these areas along with a list of reliable details. Your goal is to combine the information from all descriptions and the reliable content list to generate a unified, precise description that accurately represents the merged content of the areas.

            ### Information Provided:

            1. **Area Descriptions**:
            - Description 1: {description1}
            - Description 2: {description2}
            - Description 3: {description3}

            2. **Reliable Content List**:
            - A list of highly reliable and consistent details extracted from all three descriptions:
                - {reliable_content_list}

            ### Instructions:

            - **Step 1**: Start by using the reliable content list as the foundation for the final description. Only use the details from this list as a trustworthy base, ensuring consistency throughout.

            - **Step 2**: Cross-reference the three area descriptions and selectively incorporate relevant details to enhance the description. Ensure that only well-supported and confirmed information is added, and avoid introducing any uncertain or speculative content.

            - **Step 3**: Generate the final, highly detailed description. Make sure that the description includes as much information as possible, but it must be entirely based on the provided descriptions and reliable content list. Do not add any details that are uncertain or cannot be verified by the provided information.

            - **Step 4**: Ensure the final description is clear, coherent, free of contradictions or hallucinations, and avoids any speculative or unconfirmed information.

            Directly output the final, polished description without any additional commentary.

            ### Example Scenario:


            #### **Area Descriptions**:
            - **Description 1**: "The park is filled with lush greenery. There are children playing near the fountain at the center. A few adults are sitting on the benches nearby, chatting with each other. To the right, a group of people is gathered around a food cart, where a man is serving ice cream. The weather is sunny, and the sky is clear."
            - **Description 2**: "In the park, several children are running around near the large fountain, which is surrounded by a stone pathway. On the left side, there are a few benches, and adults are sitting and talking. Near the pathway, there is an ice cream cart with a small line of people waiting to buy snacks."
            - **Description 3**: "The park is full of life, with children playing near the fountain in the middle. The fountain is large and has water spraying from its top. Some adults are sitting on benches near the trees, while others are walking around the fountain. A food cart stands near the path, and the sky is bright blue."

            #### **Reliable Content List**:
            ['The park has a fountain in the center, surrounded by children playing.','There are benches with adults sitting and chatting.','There is a food cart near the fountain, serving snacks.','The weather is sunny, with clear skies.']

            ### Example output:
            "The park is alive with activity, featuring a large fountain at its center, where children are seen running and playing joyfully. Surrounding the fountain, a stone pathway leads to several benches, where adults sit and chat in the shade. To the right of the fountain, a food cart serves snacks to a small line of people. The bright blue sky and sunny weather complete the vibrant and cheerful atmosphere of the park."
            """

        prompt=[]
        for i in range(len(region_descriptions)):
            user=f"""
            ### Your Task:
            Generate a caption that accurately reflects the most reliable information from the provided infromation, ensuring that no contradictory information is included. Do not include any explanations or thought processes, directly output the final caption without any prefixes.

            #### Input:

            **Area Descriptions**:
            - Description 1: {region_descriptions[i][0][0][0]}
            - Description 2: {region_descriptions[i][0][0][1]}
            - Description 3: {region_descriptions[i][0][0][2]}
            **Reliable Content List**:
            - {supplement[i]}
            """
            messages = [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ]
            text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
            prompt.append(text)
        sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=4096,stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")])
      
        output = self.llm.generate(prompt, sampling_params)
        # generated_text= output[0].outputs[0].text
        
        return output
    def batch_merge_sameregion(self,region_descriptions,supplement):
        system = """
            You are a language model tasked with generating a coherent, detailed, and hallucination-free description based on the visual content of three areas. You are provided with detailed descriptions of these areas along with a list of reliable details. Your goal is to combine the information from all descriptions and the reliable content list to generate a unified, precise description that accurately represents the merged content of the areas.

            ### Information Provided:

            1. **Area Descriptions**:
            - Description 1: {description1}
            - Description 2: {description2}
            - Description 3: {description3}

            2. **Reliable Content List**:
            - A list of highly reliable and consistent details extracted from all three descriptions:
                - {reliable_content_list}

            ### Instructions:

            - **Step 1**: Start by using the reliable content list as the foundation for the final description. Only use the details from this list as a trustworthy base, ensuring consistency throughout.

            - **Step 2**: Cross-reference the three area descriptions and selectively incorporate relevant details to enhance the description. Ensure that only well-supported and confirmed information is added, and avoid introducing any uncertain or speculative content.

            - **Step 3**: Generate the final, highly detailed description. Make sure that the description includes as much information as possible, but it must be entirely based on the provided descriptions and reliable content list. Do not add any details that are uncertain or cannot be verified by the provided information.

            - **Step 4**: Ensure the final description is clear, coherent, free of contradictions or hallucinations, and avoids any speculative or unconfirmed information.

            Directly output the final, polished description without any additional commentary.

            ### Example Scenario:


            #### **Area Descriptions**:
            - **Description 1**: "The park is filled with lush greenery. There are children playing near the fountain at the center. A few adults are sitting on the benches nearby, chatting with each other. To the right, a group of people is gathered around a food cart, where a man is serving ice cream. The weather is sunny, and the sky is clear."
            - **Description 2**: "In the park, several children are running around near the large fountain, which is surrounded by a stone pathway. On the left side, there are a few benches, and adults are sitting and talking. Near the pathway, there is an ice cream cart with a small line of people waiting to buy snacks."
            - **Description 3**: "The park is full of life, with children playing near the fountain in the middle. The fountain is large and has water spraying from its top. Some adults are sitting on benches near the trees, while others are walking around the fountain. A food cart stands near the path, and the sky is bright blue."

            #### **Reliable Content List**:
            ['The park has a fountain in the center, surrounded by children playing.','There are benches with adults sitting and chatting.','There is a food cart near the fountain, serving snacks.','The weather is sunny, with clear skies.']

            ### Example output:
            "The park is alive with activity, featuring a large fountain at its center, where children are seen running and playing joyfully. Surrounding the fountain, a stone pathway leads to several benches, where adults sit and chat in the shade. To the right of the fountain, a food cart serves snacks to a small line of people. The bright blue sky and sunny weather complete the vibrant and cheerful atmosphere of the park."
            """

        prompt=[]
        for i in range(len(region_descriptions)-1):
            user=f"""
            ### Your Task:
            Generate a caption that accurately reflects the most reliable information from the provided infromation, ensuring that no contradictory information is included. Do not include any explanations or thought processes, directly output the final caption without any prefixes.

            #### Input:

            **Area Descriptions**:
            - Description 1: {region_descriptions[i][0]}
            - Description 2: {region_descriptions[i][1]}
            - Description 3: {region_descriptions[i][2]}
            **Reliable Content List**:
            - {supplement[i]}
            """
            messages = [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ]
            text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
            prompt.append(text)
        sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=4096,stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")])
      
        output = self.llm.generate(prompt, sampling_params)
        # generated_text= output[0].outputs[0].text
        
        return output
    def merge_sameregion(self,description1,description2,description3,supplement):
        system = """
            You are a language model tasked with generating a coherent, detailed, and hallucination-free description based on the visual content of three areas. You are provided with detailed descriptions of these areas along with a list of reliable details. Your goal is to combine the information from all descriptions and the reliable content list to generate a unified, precise description that accurately represents the merged content of the areas.

            ### Information Provided:

            1. **Area Descriptions**:
            - Description 1: {description1}
            - Description 2: {description2}
            - Description 3: {description3}

            2. **Reliable Content List**:
            - A list of highly reliable and consistent details extracted from all three descriptions:
                - {reliable_content_list}

            ### Instructions:

            - **Step 1**: Start by using the reliable content list as the foundation for the final description. Only use the details from this list as a trustworthy base, ensuring consistency throughout.

            - **Step 2**: Cross-reference the three area descriptions and selectively incorporate relevant details to enhance the description. Ensure that only well-supported and confirmed information is added, and avoid introducing any uncertain or speculative content.

            - **Step 3**: Generate the final, highly detailed description. Make sure that the description includes as much information as possible, but it must be entirely based on the provided descriptions and reliable content list. Do not add any details that are uncertain or cannot be verified by the provided information.

            - **Step 4**: Ensure the final description is clear, coherent, free of contradictions or hallucinations, and avoids any speculative or unconfirmed information.

            Directly output the final, polished description without any additional commentary.

            ### Example Scenario:


            #### **Area Descriptions**:
            - **Description 1**: "The park is filled with lush greenery. There are children playing near the fountain at the center. A few adults are sitting on the benches nearby, chatting with each other. To the right, a group of people is gathered around a food cart, where a man is serving ice cream. The weather is sunny, and the sky is clear."
            - **Description 2**: "In the park, several children are running around near the large fountain, which is surrounded by a stone pathway. On the left side, there are a few benches, and adults are sitting and talking. Near the pathway, there is an ice cream cart with a small line of people waiting to buy snacks."
            - **Description 3**: "The park is full of life, with children playing near the fountain in the middle. The fountain is large and has water spraying from its top. Some adults are sitting on benches near the trees, while others are walking around the fountain. A food cart stands near the path, and the sky is bright blue."

            #### **Reliable Content List**:
            ['The park has a fountain in the center, surrounded by children playing.','There are benches with adults sitting and chatting.','There is a food cart near the fountain, serving snacks.','The weather is sunny, with clear skies.']

            ### Example output:
            "The park is alive with activity, featuring a large fountain at its center, where children are seen running and playing joyfully. Surrounding the fountain, a stone pathway leads to several benches, where adults sit and chat in the shade. To the right of the fountain, a food cart serves snacks to a small line of people. The bright blue sky and sunny weather complete the vibrant and cheerful atmosphere of the park."
            """

        user=f"""
            ### Your Task:
            Generate a caption that accurately reflects the most reliable information from the provided infromation, ensuring that no contradictory information is included. Do not include any explanations or thought processes, directly output the final caption without any prefixes.

            #### Input:

            **Area Descriptions**:
            - Description 1: {description1}
            - Description 2: {description2}
            - Description 3: {description3}


            **Reliable Content List**:
            - {supplement}
            """
        messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]
        sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=4096,stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")])
        text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
        output = self.llm.generate([text], sampling_params)
        generated_text= output[0].outputs[0].text
        
        return generated_text
    def batch_merge_mainbox(self,region_descriptions,main_des):
        system = """
    ### Input ###
    • You will receive a global description that provides an overall view of the image.
    • Additionally, you will be provided with a detailed region description, which focuses on a specific area within the image.

    ### Task Objective ###
    • Modify and enhance the global description by integrating accurate details from the region description.
    • Do not introduce any elements or details that are not present in the given region description or the original global description.
    • The output should be an enriched global description with relevant details seamlessly integrated from the region description.

    ### Input INFORMATION EXPLANATION ###
    1. Global Description: This is the initial, broader description of the image, covering the main elements and objects but potentially lacking specific details.
    2. Region Description: It offers detailed information about a specific section of the image, often containing more precise or additional details about objects and actions.

    ### Guidelines ###
    • Integrate specific details from the region description into the global description where relevant.
    • Ensure the updated global description remains coherent, natural, and more detailed than the original.
    • Do not add any new elements or make assumptions beyond the given descriptions.

    ### Example ###
    Global Description:
    A busy marketplace with various stalls can be seen, with people walking around and shopping. There are different goods on display, such as fruits, vegetables, and clothes. In the background, the sky is partly cloudy, and a few birds are flying.

    Region Description:
    A fruit stand in the center of the market is displaying piles of bright oranges, green apples, and ripe bananas. A vendor wearing a green apron is helping a customer select some oranges. The customer is holding a wicker basket.

    Your Modified Global Description:
    A busy marketplace with various stalls can be seen, with people walking around and shopping. There are different goods on display, such as fruits, vegetables, and clothes. In the center of the market, a fruit stand showcases bright oranges, green apples, and ripe bananas, while a vendor wearing a green apron assists a customer in selecting some oranges. The customer holds a wicker basket. In the background, the sky is partly cloudy, and a few birds are flying.

            """
        prompt=[]
        for i in range(len(region_descriptions)):
            user=f"""
        ###TASK### 
        Please provide the modified description directly. 
        Global Description: 
        {region_descriptions[i][0][2]}
        Region Description:
        {main_des[i].outputs[0].text}
                """
            messages = [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ]
            text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
            prompt.append(text)
        sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=4096,stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")])
       
        output = self.llm.generate(prompt, sampling_params)
        # generated_text= output[0].outputs[0].text
        
        return output 
    def merge_mainbox(self,description1,description2):
        system = """
    ### Input ###
    • You will receive a global description that provides an overall view of the image.
    • Additionally, you will be provided with a detailed region description, which focuses on a specific area within the image.

    ### Task Objective ###
    • Modify and enhance the global description by integrating accurate details from the region description.
    • Do not introduce any elements or details that are not present in the given region description or the original global description.
    • The output should be an enriched global description with relevant details seamlessly integrated from the region description.

    ### Input INFORMATION EXPLANATION ###
    1. Global Description: This is the initial, broader description of the image, covering the main elements and objects but potentially lacking specific details.
    2. Region Description: It offers detailed information about a specific section of the image, often containing more precise or additional details about objects and actions.

    ### Guidelines ###
    • Integrate specific details from the region description into the global description where relevant.
    • Ensure the updated global description remains coherent, natural, and more detailed than the original.
    • Do not add any new elements or make assumptions beyond the given descriptions.

    ### Example ###
    Global Description:
    A busy marketplace with various stalls can be seen, with people walking around and shopping. There are different goods on display, such as fruits, vegetables, and clothes. In the background, the sky is partly cloudy, and a few birds are flying.

    Region Description:
    A fruit stand in the center of the market is displaying piles of bright oranges, green apples, and ripe bananas. A vendor wearing a green apron is helping a customer select some oranges. The customer is holding a wicker basket.

    Your Modified Global Description:
    A busy marketplace with various stalls can be seen, with people walking around and shopping. There are different goods on display, such as fruits, vegetables, and clothes. In the center of the market, a fruit stand showcases bright oranges, green apples, and ripe bananas, while a vendor wearing a green apron assists a customer in selecting some oranges. The customer holds a wicker basket. In the background, the sky is partly cloudy, and a few birds are flying.

            """

        user=f"""
    ###TASK### 
    Please provide the modified description directly. 
    Global Description: 
    {description1}
    Region Description:
    {description2}
            """
        messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]
        sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=4096,stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")])
        text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
        output = self.llm.generate([text], sampling_params)
        generated_text= output[0].outputs[0].text
        
        return generated_text
    def batch_merge_five(self,temp_merge_five):
        
        system ="""
    ###Input###
    • You will receive a global caption describing an image.
    • Additionally, you will have access to region captions generated for specific regions within the image, along with their specific location information.
    • Both global and local captions may contain noise or errors.

    ###Task Objective###
    • Your goal is to modify and enhance the global caption by integrating accurate details from the region captions and their location.
    • The global caption should be enriched with specific, accurate details from the regions and corrected where necessary.
    • Focus on using the region captions to correct any inaccuracies or hallucinations in the global caption.
    • The updated global caption must contain more detail than the original global caption by including relevant information from the region captions.
    • You only give the updated global caption as output, without any additional information.
    • Do NOT give any explanation or notes on how you generate this caption.

    ###Input INFORMATION EXPLANATION###
    1. Global Description: It provides the initial global image description, which captures the primary semantic information of the image. However, some of the described objects are hallucinated, and certain details are either missing or insufficiently described, requiring additional information for correction and enhancement.
    2. Region Description: It provides descriptions of different regions, focusing on specific parts of the image. These include more detailed object features and finer details. Additionally, this section has undergone hallucination filtering, making the descriptions more reliable compared to the global description.
    3. Region Location: It uses a normalized coordinate system where both x (horizontal) and y (vertical) axes range from 0 to 1. The x-coordinate starts at 0 on the image’s left edge and increases to 1 towards the right edge. Similarly, the y-coordinate starts at 0 at the top edge and increases to 1 towards the bottom. This system uses four coordinates to define the corners of a rectangle within the image: [x1, y1, x2, y2], representing the top-left and bottom-right corners of the rectangle, respectively.

    ###Guidelines###
    • Through the extra information of different regions, some objects may represent the same thing. When adding objects to the original description, it is important to avoid duplication.
    • Combine Information: Extract and integrate key details from both the global and local (region) captions, giving priority to the region captions for more specific or accurate details.
    • Modify and Enhance: Add relevant details from the region captions to enrich the global description. Correct any hallucinations or inaccuracies in the global caption using the region captions.
    • Consider Location: Ensure that spatial information from the region captions is incorporated to provide a more coherent and accurate description of the image.
    • Filter Noise: Remove any inaccurate, irrelevant, or conflicting information from the global caption, especially when region captions offer more precise details.
    • Enhance Detail: Ensure that the final global caption contains more detailed and refined visual information than the original, using the region captions to add specificity.

    ###In-Context Examples###
    [Chain of thought is placed within a pair of "@@@" (remember only in the Examples will you be provided with a chain of thoughts to help you understand; in the actual task, these will not be given to you.)]

    ###Example 1:###
    Global Description:
    A large open-air market is bustling with activity. People are walking around, browsing stalls under colorful tents, and a mix of goods, such as fruits, vegetables, and clothing, is being sold. The sky is clear, and the sun is shining brightly. A food truck is parked in the middle of the market, serving hot food to people waiting in line.

    Region 1:
    - Location: [0.05, 0.20, 0.40, 0.60]
    - Description: A fruit stand with neatly arranged piles of oranges, bananas, and apples. A vendor wearing a green apron is standing behind the stall, handing a bag of oranges to a woman in a blue dress. The woman is holding a wicker basket, and there are several other customers browsing nearby.

    Region 2:
    - Location: [0.50, 0.10, 0.80, 0.50]
    - Description: A clothing stall displaying a variety of colorful scarves and hats hanging on racks. A group of three teenagers, two girls and one boy, are looking at the items. One girl is holding a red scarf, while the boy is trying on a wide-brimmed hat. The vendor, a middle-aged man with a beard, is standing by, smiling.

    Region 3:
    - Location: [0.10, 0.70, 0.40, 0.90]
    - Description: A vegetable stall with baskets of tomatoes, cucumbers, and carrots neatly displayed. An older woman is picking up a tomato while talking to the vendor, who is organizing the produce.

    Region 4:
    - Location: [0.75, 0.10, 0.95, 0.30]
    - Description: A food cart selling ice cream, with a line of people waiting. The vendor, wearing a blue hat, is handing an ice cream cone to a little girl. A man in the line is holding a small dog on a leash.

    @@@ Chain of Thought:
    1. The global description accurately captures the bustling market and sunny weather. However, the mention of colorful tents and a food truck is not supported by any of the region descriptions and should be removed.
    2. The region descriptions provide more specific details. Region 1 describes a fruit stand on the left ([0.05, 0.20, 0.40, 0.60]) with a vendor and a woman in a blue dress holding a wicker basket, which should be incorporated. Region 2 adds the detail of a clothing stall on the right ([0.50, 0.10, 0.80, 0.50]), with teenagers looking at scarves and hats, which can replace the generic mention of clothing in the global caption.
    3. Region 3 describes a vegetable stall with a vendor and an older woman ([0.10, 0.70, 0.40, 0.90]), which provides further specific detail and should be integrated into the global description. Region 4, which describes a food cart serving ice cream, replaces the unsupported food truck mention and adds specific details about the line of people.

    @@@

    Your Modified Description:
    A large open-air market is bustling with activity under a clear, sunny sky. People are walking around, browsing various stalls selling fruits, vegetables, and clothing. On the left side of the market, a fruit stand displays neatly arranged piles of oranges, bananas, and apples, with a vendor in a green apron handing a bag of oranges to a woman in a blue dress holding a wicker basket. Several other customers are browsing the fresh produce nearby. To the right, a clothing stall showcases colorful scarves and hats. Two girls and a boy are examining the items, with one girl holding a red scarf while the boy tries on a wide-brimmed hat. Further back, a vegetable stall is lined with baskets of tomatoes, cucumbers, and carrots, where an older woman picks up a tomato while chatting with the vendor. At the far right of the market, a food cart serves ice cream, with a line of people waiting as the vendor hands a cone to a little girl, while a man holds a small dog on a leash.

    ###Example 2:###
    Global Description:
    Three friends are sitting on a bench in the park, chatting and laughing. The sun is shining brightly, and people are scattered around the park, enjoying the weather. A man is jogging along the path, and there’s a pond with ducks swimming nearby.

    Region 1:
    - Location: [0.10, 0.25, 0.40, 0.60]
    - Description: Two women are sitting on a bench under a tree. One is wearing a blue T-shirt and shorts, while the other is dressed in a white sundress. They are chatting and laughing, with one of the women holding a cup of coffee. There’s a picnic blanket on the ground near the bench with some snacks on it.

    Region 2:
    - Location: [0.50, 0.10, 0.80, 0.50]
    - Description: A man in a green T-shirt and jeans is standing next to the bench, holding a water bottle in one hand while looking at the two women. He seems to be engaged in their conversation, smiling and occasionally glancing at his phone.

    Region 3:
    - Location: [0.60, 0.70, 0.80, 0.90]
    - Description: A man is jogging along a path, wearing headphones and a blue tank top. He is passing by a group of trees and a flower bed filled with brightly colored flowers.

    Region 4:
    - Location: [0.10, 0.60, 0.30, 0.90]
    - Description: A small pond with ducks swimming near the shore. A child is throwing breadcrumbs into the water, and the ducks are gathering around. A couple is sitting on a bench nearby, watching the scene.

    @@@ Chain of Thought:
    1. The global description mentions three friends sitting on a bench, but Region 1 and Region 2 confirm that it is actually two women sitting on the bench, with a man standing nearby. This needs to be corrected.
    2. The global description of a man jogging and a pond with ducks are both supported by Region 3 and Region 4, so these details should be retained and enriched with the specific details provided by the regions.
    3. Additional details from Region 1 about the picnic blanket and the coffee cup add richness
    """
        prompt=[]
        for i in range(len(temp_merge_five)):
            image_width=temp_merge_five[i][0][0]
            image_height=temp_merge_five[i][0][1]
            region1_location=self.normalize_box(temp_merge_five[i][2][0],image_width,image_height)
            region2_location=self.normalize_box(temp_merge_five[i][2][1],image_width,image_height)
            region3_location=self.normalize_box(temp_merge_five[i][2][2],image_width,image_height)
            region4_location=self.normalize_box(temp_merge_five[i][2][3],image_width,image_height)
            user=f"""
            ###TASK### 
            Please provide the modified description directly. 
            Global Description: 
            {temp_merge_five[i][1]}
            Region1
            - Location: {region1_location}
            - Description: {temp_merge_five[i][3][0]}
            Region2
            - Location: {region2_location}
            - Description: {temp_merge_five[i][3][1]}
            Region3
            - Location: {region3_location}
            - Description: {temp_merge_five[i][3][2]}
            Region4
            - Location: {region4_location}
            - Description: {temp_merge_five[i][3][3]}
                    """
                
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
            prompt.append(text)
     
        
        sampling_params = SamplingParams(temperature=0.2, top_p=0.9,max_tokens=4096,stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")])
        output = self.llm.generate(prompt, sampling_params)

        
        return output
    def merge_five(self,image_size, global_description, region1_location, region1_description, region2_location, region2_description, region3_location, region3_description, region4_location, region4_description):
        
        system ="""
    ###Input###
    • You will receive a global caption describing an image.
    • Additionally, you will have access to region captions generated for specific regions within the image, along with their specific location information.
    • Both global and local captions may contain noise or errors.

    ###Task Objective###
    • Your goal is to modify and enhance the global caption by integrating accurate details from the region captions and their location.
    • The global caption should be enriched with specific, accurate details from the regions and corrected where necessary.
    • Focus on using the region captions to correct any inaccuracies or hallucinations in the global caption.
    • The updated global caption must contain more detail than the original global caption by including relevant information from the region captions.
    • You only give the updated global caption as output, without any additional information.
    • Do NOT give any explanation or notes on how you generate this caption.

    ###Input INFORMATION EXPLANATION###
    1. Global Description: It provides the initial global image description, which captures the primary semantic information of the image. However, some of the described objects are hallucinated, and certain details are either missing or insufficiently described, requiring additional information for correction and enhancement.
    2. Region Description: It provides descriptions of different regions, focusing on specific parts of the image. These include more detailed object features and finer details. Additionally, this section has undergone hallucination filtering, making the descriptions more reliable compared to the global description.
    3. Region Location: It uses a normalized coordinate system where both x (horizontal) and y (vertical) axes range from 0 to 1. The x-coordinate starts at 0 on the image’s left edge and increases to 1 towards the right edge. Similarly, the y-coordinate starts at 0 at the top edge and increases to 1 towards the bottom. This system uses four coordinates to define the corners of a rectangle within the image: [x1, y1, x2, y2], representing the top-left and bottom-right corners of the rectangle, respectively.

    ###Guidelines###
    • Through the extra information of different regions, some objects may represent the same thing. When adding objects to the original description, it is important to avoid duplication.
    • Combine Information: Extract and integrate key details from both the global and local (region) captions, giving priority to the region captions for more specific or accurate details.
    • Modify and Enhance: Add relevant details from the region captions to enrich the global description. Correct any hallucinations or inaccuracies in the global caption using the region captions.
    • Consider Location: Ensure that spatial information from the region captions is incorporated to provide a more coherent and accurate description of the image.
    • Filter Noise: Remove any inaccurate, irrelevant, or conflicting information from the global caption, especially when region captions offer more precise details.
    • Enhance Detail: Ensure that the final global caption contains more detailed and refined visual information than the original, using the region captions to add specificity.

    ###In-Context Examples###
    [Chain of thought is placed within a pair of "@@@" (remember only in the Examples will you be provided with a chain of thoughts to help you understand; in the actual task, these will not be given to you.)]

    ###Example 1:###
    Global Description:
    A large open-air market is bustling with activity. People are walking around, browsing stalls under colorful tents, and a mix of goods, such as fruits, vegetables, and clothing, is being sold. The sky is clear, and the sun is shining brightly. A food truck is parked in the middle of the market, serving hot food to people waiting in line.

    Region 1:
    - Location: [0.05, 0.20, 0.40, 0.60]
    - Description: A fruit stand with neatly arranged piles of oranges, bananas, and apples. A vendor wearing a green apron is standing behind the stall, handing a bag of oranges to a woman in a blue dress. The woman is holding a wicker basket, and there are several other customers browsing nearby.

    Region 2:
    - Location: [0.50, 0.10, 0.80, 0.50]
    - Description: A clothing stall displaying a variety of colorful scarves and hats hanging on racks. A group of three teenagers, two girls and one boy, are looking at the items. One girl is holding a red scarf, while the boy is trying on a wide-brimmed hat. The vendor, a middle-aged man with a beard, is standing by, smiling.

    Region 3:
    - Location: [0.10, 0.70, 0.40, 0.90]
    - Description: A vegetable stall with baskets of tomatoes, cucumbers, and carrots neatly displayed. An older woman is picking up a tomato while talking to the vendor, who is organizing the produce.

    Region 4:
    - Location: [0.75, 0.10, 0.95, 0.30]
    - Description: A food cart selling ice cream, with a line of people waiting. The vendor, wearing a blue hat, is handing an ice cream cone to a little girl. A man in the line is holding a small dog on a leash.

    @@@ Chain of Thought:
    1. The global description accurately captures the bustling market and sunny weather. However, the mention of colorful tents and a food truck is not supported by any of the region descriptions and should be removed.
    2. The region descriptions provide more specific details. Region 1 describes a fruit stand on the left ([0.05, 0.20, 0.40, 0.60]) with a vendor and a woman in a blue dress holding a wicker basket, which should be incorporated. Region 2 adds the detail of a clothing stall on the right ([0.50, 0.10, 0.80, 0.50]), with teenagers looking at scarves and hats, which can replace the generic mention of clothing in the global caption.
    3. Region 3 describes a vegetable stall with a vendor and an older woman ([0.10, 0.70, 0.40, 0.90]), which provides further specific detail and should be integrated into the global description. Region 4, which describes a food cart serving ice cream, replaces the unsupported food truck mention and adds specific details about the line of people.

    @@@

    Your Modified Description:
    A large open-air market is bustling with activity under a clear, sunny sky. People are walking around, browsing various stalls selling fruits, vegetables, and clothing. On the left side of the market, a fruit stand displays neatly arranged piles of oranges, bananas, and apples, with a vendor in a green apron handing a bag of oranges to a woman in a blue dress holding a wicker basket. Several other customers are browsing the fresh produce nearby. To the right, a clothing stall showcases colorful scarves and hats. Two girls and a boy are examining the items, with one girl holding a red scarf while the boy tries on a wide-brimmed hat. Further back, a vegetable stall is lined with baskets of tomatoes, cucumbers, and carrots, where an older woman picks up a tomato while chatting with the vendor. At the far right of the market, a food cart serves ice cream, with a line of people waiting as the vendor hands a cone to a little girl, while a man holds a small dog on a leash.

    ###Example 2:###
    Global Description:
    Three friends are sitting on a bench in the park, chatting and laughing. The sun is shining brightly, and people are scattered around the park, enjoying the weather. A man is jogging along the path, and there’s a pond with ducks swimming nearby.

    Region 1:
    - Location: [0.10, 0.25, 0.40, 0.60]
    - Description: Two women are sitting on a bench under a tree. One is wearing a blue T-shirt and shorts, while the other is dressed in a white sundress. They are chatting and laughing, with one of the women holding a cup of coffee. There’s a picnic blanket on the ground near the bench with some snacks on it.

    Region 2:
    - Location: [0.50, 0.10, 0.80, 0.50]
    - Description: A man in a green T-shirt and jeans is standing next to the bench, holding a water bottle in one hand while looking at the two women. He seems to be engaged in their conversation, smiling and occasionally glancing at his phone.

    Region 3:
    - Location: [0.60, 0.70, 0.80, 0.90]
    - Description: A man is jogging along a path, wearing headphones and a blue tank top. He is passing by a group of trees and a flower bed filled with brightly colored flowers.

    Region 4:
    - Location: [0.10, 0.60, 0.30, 0.90]
    - Description: A small pond with ducks swimming near the shore. A child is throwing breadcrumbs into the water, and the ducks are gathering around. A couple is sitting on a bench nearby, watching the scene.

    @@@ Chain of Thought:
    1. The global description mentions three friends sitting on a bench, but Region 1 and Region 2 confirm that it is actually two women sitting on the bench, with a man standing nearby. This needs to be corrected.
    2. The global description of a man jogging and a pond with ducks are both supported by Region 3 and Region 4, so these details should be retained and enriched with the specific details provided by the regions.
    3. Additional details from Region 1 about the picnic blanket and the coffee cup add richness
    """
        image_width=image_size[0]
        image_height=image_size[1]
        region1_location=self.normalize_box(region1_location,image_width,image_height)
        region2_location=self.normalize_box(region2_location,image_width,image_height)
        region3_location=self.normalize_box(region3_location,image_width,image_height)
        region4_location=self.normalize_box(region4_location,image_width,image_height)
        user=f"""
    ###TASK### 
    Please provide the modified description directly. 
    Global Description: 
    {global_description}
    Region1
    - Location: {region1_location}
    - Description: {region1_description}
    Region2
    - Location: {region2_location}
    - Description: {region2_description}
    Region3
    - Location: {region3_location}
    - Description: {region3_description}
    Region4
    - Location: {region4_location}
    - Description: {region4_description}
            """
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        
        sampling_params = SamplingParams(temperature=0.2, top_p=0.9,max_tokens=4096,stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")])
        text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
        output = self.llm.generate([text], sampling_params)
        generated_text= output[0].outputs[0].text
        
        return generated_text
    def batch_group_sameregion_sentence_main(self,region_descriptions):
        system = """
        You are a language modeler tasked with analyzing three passages describing the same area of a picture. Your goal is to process these descriptions step by step, reasoning through each task logically and systematically. Please follow the steps outlined below and directly output the final results of Step 4 without providing explanations or additional words.

        Guidelines:

        - Step 1: Identifying Similar Descriptions
        1. Compare the descriptions to identify sentences that describe the same object, relationship, or action using semantic similarity and context.
        2. Only group sentences as "similar descriptions" if they appear in at least **two or more** passages. If a description only appears in one passage, it should not be included in this group.
        3. Combine these similar descriptions into a coherent single sentence.
        4. **Important**: Once a description is grouped into the "similar descriptions" category, it should *not* appear again in the "contradictory descriptions" category.

        - Step 2: Identifying Contradictory Descriptions
        1. Find sentences that describe the same object but provide conflicting attributes (e.g., color, size, or action). 
        2. Ensure that if a sentence describes an object or attribute that is not mentioned in the other descriptions, it should *not* be considered contradictory. Instead, this sentence should be moved to the unique descriptions category.
        3. Sentences already grouped under "similar descriptions" should not be included in contradictory descriptions.
        4. Present only sentences that describe the same object and attribute but provide conflicting information in pairs, as they are, without modification.

        - Step 3: Identifying Unique Descriptions
        1. Identify any sentence that describes an object or detail not mentioned in the other passages or that only appears in one passage.
        2. List these unique descriptions along with their respective passage.

        - Step 4: Synthesizing and Refining the Output
        1. For similar descriptions: Merge into a single sentence that captures the shared semantics.
        2. For contradictory descriptions: Present them as pairs, showing the conflicting information from different passages.
        3. For unique descriptions: List the unique details from each passage.

        Remember to directly output the final results of Step 4 without displaying other words.

        ### Example:

        ###Input Descriptions:

        Description 1: "A tall man in a black suit is standing under a large oak tree. The sun is setting, casting an orange glow over the landscape. The sky is clear with a few scattered clouds. A woman in a red dress is walking along a dirt path, and a dog runs playfully in the grass nearby."

        Description 2: "A man wearing a dark suit stands beneath an old oak tree as the sun sets. The sky is filled with vibrant orange and pink hues. In the distance, a woman in a red dress strolls down a dirt path, and a brown dog plays in the grassy field next to her."

        Description 3: "A man in a dark suit stands under a large tree at sunset. The sky is mostly cloudy with patches of color. A woman in a red dress walks along a path, and a dog is seen playing nearby."

        ###Output:

        For Similar Descriptions:
        - Group 1 Combined Description: "A man in a dark suit is standing under a large tree as the sun sets."
        - Group 2 Combined Description: "A woman in a red dress is walking along a dirt path."
        - Group 3 Combined Description: "A dog is playing in the grass."


        For Contradictory Descriptions:
        - ["The sky is clear with a few scattered clouds." (Description 1), "The sky is filled with vibrant orange and pink hues." (Description 2), "The sky is mostly cloudy with patches of color." (Description 3)]

        For Unique Descriptions:
        - "The sun is casting an orange glow over the landscape." (Description 1)

        """
        prompt=[]
        # print(region_descriptions)
        for i in range(len(region_descriptions)):
            user=f"""
                        ###Description 1: {region_descriptions[i][0][0][0]}

                        ###Description 2: {region_descriptions[i][0][0][1]}

                        ###Description 2: {region_descriptions[i][0][0][2]}

                        Please directly output the final results of Step 4 without displaying the intermediate steps, strictly follow the example output and do not include any additional comments..
                        """
            messages = [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ]
            text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
            prompt.append(text)
        sampling_params = SamplingParams(temperature=0.2, top_p=0.9,max_tokens=4096,stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")])
        
        output = self.llm.generate(prompt, sampling_params)
        # generated_text= output[0].outputs[0].text
        
        return output
    def batch_group_sameregion_sentence(self,region_descriptions):
        system = """
        You are a language modeler tasked with analyzing three passages describing the same area of a picture. Your goal is to process these descriptions step by step, reasoning through each task logically and systematically. Please follow the steps outlined below and directly output the final results of Step 4 without providing explanations or additional words.

        Guidelines:

        - Step 1: Identifying Similar Descriptions
        1. Compare the descriptions to identify sentences that describe the same object, relationship, or action using semantic similarity and context.
        2. Only group sentences as "similar descriptions" if they appear in at least **two or more** passages. If a description only appears in one passage, it should not be included in this group.
        3. Combine these similar descriptions into a coherent single sentence.
        4. **Important**: Once a description is grouped into the "similar descriptions" category, it should *not* appear again in the "contradictory descriptions" category.

        - Step 2: Identifying Contradictory Descriptions
        1. Find sentences that describe the same object but provide conflicting attributes (e.g., color, size, or action). 
        2. Ensure that if a sentence describes an object or attribute that is not mentioned in the other descriptions, it should *not* be considered contradictory. Instead, this sentence should be moved to the unique descriptions category.
        3. Sentences already grouped under "similar descriptions" should not be included in contradictory descriptions.
        4. Present only sentences that describe the same object and attribute but provide conflicting information in pairs, as they are, without modification.

        - Step 3: Identifying Unique Descriptions
        1. Identify any sentence that describes an object or detail not mentioned in the other passages or that only appears in one passage.
        2. List these unique descriptions along with their respective passage.

        - Step 4: Synthesizing and Refining the Output
        1. For similar descriptions: Merge into a single sentence that captures the shared semantics.
        2. For contradictory descriptions: Present them as pairs, showing the conflicting information from different passages.
        3. For unique descriptions: List the unique details from each passage.

        Remember to directly output the final results of Step 4 without displaying other words.

        ### Example:

        ###Input Descriptions:

        Description 1: "A tall man in a black suit is standing under a large oak tree. The sun is setting, casting an orange glow over the landscape. The sky is clear with a few scattered clouds. A woman in a red dress is walking along a dirt path, and a dog runs playfully in the grass nearby."

        Description 2: "A man wearing a dark suit stands beneath an old oak tree as the sun sets. The sky is filled with vibrant orange and pink hues. In the distance, a woman in a red dress strolls down a dirt path, and a brown dog plays in the grassy field next to her."

        Description 3: "A man in a dark suit stands under a large tree at sunset. The sky is mostly cloudy with patches of color. A woman in a red dress walks along a path, and a dog is seen playing nearby."

        ###Output:

        For Similar Descriptions:
        - Group 1 Combined Description: "A man in a dark suit is standing under a large tree as the sun sets."
        - Group 2 Combined Description: "A woman in a red dress is walking along a dirt path."
        - Group 3 Combined Description: "A dog is playing in the grass."


        For Contradictory Descriptions:
        - ["The sky is clear with a few scattered clouds." (Description 1), "The sky is filled with vibrant orange and pink hues." (Description 2), "The sky is mostly cloudy with patches of color." (Description 3)]

        For Unique Descriptions:
        - "The sun is casting an orange glow over the landscape." (Description 1)

        """
        prompt=[]
        for i in range(len(region_descriptions)-1):
            user=f"""
                        ###Description 1: {region_descriptions[i][0]}

                        ###Description 2: {region_descriptions[i][1]}

                        ###Description 2: {region_descriptions[i][2]}

                        Please directly output the final results of Step 4 without displaying the intermediate steps, strictly follow the example output and do not include any additional comments..
                        """
            messages = [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ]
            text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
            prompt.append(text)
        sampling_params = SamplingParams(temperature=0.2, top_p=0.9,max_tokens=4096,stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")])
        
        output = self.llm.generate(prompt, sampling_params)
        # generated_text= output[0].outputs[0].text
        
        return output
    def group_sameregion_sentence(self,description1,description2,description3):
            system = """
            You are a language modeler tasked with analyzing three passages describing the same area of a picture. Your goal is to process these descriptions step by step, reasoning through each task logically and systematically. Please follow the steps outlined below and directly output the final results of Step 4 without providing explanations or additional words.

            Guidelines:

            - Step 1: Identifying Similar Descriptions
            1. Compare the descriptions to identify sentences that describe the same object, relationship, or action using semantic similarity and context.
            2. Only group sentences as "similar descriptions" if they appear in at least **two or more** passages. If a description only appears in one passage, it should not be included in this group.
            3. Combine these similar descriptions into a coherent single sentence.
            4. **Important**: Once a description is grouped into the "similar descriptions" category, it should *not* appear again in the "contradictory descriptions" category.

            - Step 2: Identifying Contradictory Descriptions
            1. Find sentences that describe the same object but provide conflicting attributes (e.g., color, size, or action). 
            2. Ensure that if a sentence describes an object or attribute that is not mentioned in the other descriptions, it should *not* be considered contradictory. Instead, this sentence should be moved to the unique descriptions category.
            3. Sentences already grouped under "similar descriptions" should not be included in contradictory descriptions.
            4. Present only sentences that describe the same object and attribute but provide conflicting information in pairs, as they are, without modification.

            - Step 3: Identifying Unique Descriptions
            1. Identify any sentence that describes an object or detail not mentioned in the other passages or that only appears in one passage.
            2. List these unique descriptions along with their respective passage.

            - Step 4: Synthesizing and Refining the Output
            1. For similar descriptions: Merge into a single sentence that captures the shared semantics.
            2. For contradictory descriptions: Present them as pairs, showing the conflicting information from different passages.
            3. For unique descriptions: List the unique details from each passage.

            Remember to directly output the final results of Step 4 without displaying other words.

            ### Example:

            ###Input Descriptions:

            Description 1: "A tall man in a black suit is standing under a large oak tree. The sun is setting, casting an orange glow over the landscape. The sky is clear with a few scattered clouds. A woman in a red dress is walking along a dirt path, and a dog runs playfully in the grass nearby."

            Description 2: "A man wearing a dark suit stands beneath an old oak tree as the sun sets. The sky is filled with vibrant orange and pink hues. In the distance, a woman in a red dress strolls down a dirt path, and a brown dog plays in the grassy field next to her."

            Description 3: "A man in a dark suit stands under a large tree at sunset. The sky is mostly cloudy with patches of color. A woman in a red dress walks along a path, and a dog is seen playing nearby."

            ###Output:

            For Similar Descriptions:
            - Group 1 Combined Description: "A man in a dark suit is standing under a large tree as the sun sets."
            - Group 2 Combined Description: "A woman in a red dress is walking along a dirt path."
            - Group 3 Combined Description: "A dog is playing in the grass."


            For Contradictory Descriptions:
            - ["The sky is clear with a few scattered clouds." (Description 1), "The sky is filled with vibrant orange and pink hues." (Description 2), "The sky is mostly cloudy with patches of color." (Description 3)]

            For Unique Descriptions:
            - "The sun is casting an orange glow over the landscape." (Description 1)

            """
            user=f"""
                        ###Description 1: {description1}

                        ###Description 2: {description2}

                        ###Description 2: {description3}

                        Please directly output the final results of Step 4 without displaying the intermediate steps, strictly follow the example output and do not include any additional comments..
                        """
            messages = [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ]
            sampling_params = SamplingParams(temperature=0.2, top_p=0.9,max_tokens=4096,stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")])
            text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
            output = self.llm.generate([text], sampling_params)
            generated_text= output[0].outputs[0].text
            
            return generated_text
    def batch_merge_main(self,image_idx):
    
            
                # image_src = Image.open('/data/users/ruotian_peng/dataset/mmvp/MMVP Images/'+data['image'].split('/')[-1]).convert('RGB')
    
 
        def cal_similarity_same(des,merge):
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
                    img, error = self.blip_model.process_image(merge)
                    
                    score,error = self.blip_model.rank_captions(img, modified_text)
                    # print(modified_text,':',score)

                    similarity_list.append(score)
                
                else:
                    similarity_list.append(0)  
            return similarity_list
        batch_temp_group=self.batch_group_sameregion_sentence_main(image_idx)
        batch_supple=[]
        for num_merge in range(len(image_idx)):
        
            
            # region_dict_temp={'1': region_dict[str(number_iou)][0],'2': region_dict[str(number_iou)][1]}
            categories = {
                'For Triples Describing the Same Thing': [],
                'For Contradictory Triples': [],
                'For Unique Triples': []
            }
        
            same_thing_pattern = re.findall(r'- Group \d+ Combined Description: "(.*?)"', batch_temp_group[num_merge].outputs[0].text)
            categories['For Triples Describing the Same Thing'].extend(same_thing_pattern)

            # 提取 "矛盾的" 内容
            contradictory_pattern = re.findall(r'- \[(.*?)\]', batch_temp_group[num_merge].outputs[0].text, re.DOTALL)
            contradictory_list = []
            for item in contradictory_pattern:
                contradictory_descriptions = re.findall(r'"(.*?)" \(Description \d+\)', item)
                regions = re.findall(r'"(.*?)" (\(Description \d+\))', item)
                contradictory_combined = [f'{desc} {region}' for desc, region in regions]
                contradictory_list.append(contradictory_combined)
            categories['For Contradictory Triples'].extend(contradictory_list)
            # 提取 "独特的" 内容并保留 Region 信息
            unique_pattern = re.findall(r'- "(.*?)" \(Description \d+\)', batch_temp_group[num_merge].outputs[0].text)
            unique_list = re.findall(r'- "(.*?)" (\(Description \d+\))', batch_temp_group[num_merge].outputs[0].text)
            unique_combined = [f'{desc} {region}' for desc, region in unique_list]
            categories['For Unique Triples'].extend(unique_combined)
            # # print(categories)

        
            if categories['For Contradictory Triples'] is not None:
                similarity_contra_list=[]
            
                for triple in categories['For Contradictory Triples']:
                    # region_1=re.findall('Region \d', triple[0])[0]
                    # region_2=re.findall('Region \d', triple[1])[0]
                    similarity_contra=cal_similarity_same(triple,image_idx[num_merge][0][1][str(4+1)])
                    similarity_contra_list.append(similarity_contra)
            similarity_unique=cal_similarity_same(categories['For Unique Triples'],image_idx[num_merge][0][1][str(4+1)] )
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
            batch_supple.append(supplement_all)
        batch_real_mainbox=self.batch_merge_sameregion_main(image_idx,batch_supple)

        global_modified=self.batch_merge_mainbox(image_idx,batch_real_mainbox)
        return global_modified
    def batch_cal_main(self,image_idx):
       
            
                # image_src = Image.open('/data/users/ruotian_peng/dataset/mmvp/MMVP Images/'+data['image'].split('/')[-1]).convert('RGB')
    
        image_src=Image.open('../coco_image/coco_sample_data_Image_Textualization/'+image_idx['image'].split('/')[-1]).convert('RGB')
        width, height = image_src.size
        main_key='../coco_image/coco_sample_data_Image_Textualization/'+image_idx['image'].split('/')[-1]
        main_box=image_src.crop(image_idx['main_box'])
        dict_image={'5':main_box}
        
       
        region_locations=[]
        region_descriptions=[]
        # 遍历字典中的每个键值对
        # print(data_image[image_idx]['local'][0])
        for key, descriptions in image_idx['local'][0].items():
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
        # 初始化一个空列表来存储重叠面积和对应的区域对
        # 初始化一个空列表来存储每对区域的IoU
        iou_values = []
        # print(region_locations)
        # print(rectangles)
        # 分别计算左上和右上、左下和右下的区域与相邻区域的IoU
        # 左上 (Region 1) 和右上 (Region 2)
        # intersection_area = rectangles[0].intersection(rectangles[1]).area
        # union_area = rectangles[0].area + rectangles[1].area - intersection_area
        # iou = intersection_area / union_area
        # iou_values.append(('Left Top (Region 1) and Right Top (Region 2)', iou))

        # # 右上 (Region 2) 和右下 (Region 4)
        # intersection_area = rectangles[1].intersection(rectangles[3]).area
        # union_area = rectangles[1].area + rectangles[3].area - intersection_area
        # iou = intersection_area / union_area
        # iou_values.append(('Right Top (Region 2) and Right Bottom (Region 4)', iou))

        # # 右下 (Region 4) 和左下 (Region 3)
        # intersection_area = rectangles[3].intersection(rectangles[2]).area
        # union_area = rectangles[3].area + rectangles[2].area - intersection_area
        # iou = intersection_area / union_area
        # iou_values.append(('Right Bottom (Region 4) and Left Bottom (Region 3)', iou))

        # # 左下 (Region 3) 和左上 (Region 1)
        # intersection_area = rectangles[2].intersection(rectangles[0]).area
        # union_area = rectangles[2].area + rectangles[0].area - intersection_area
        # iou = intersection_area / union_area
        # iou_values.append(('Left Bottom (Region 3) and Left Top (Region 1)', iou))
        
        intersection_area = rectangles[4].intersection(rectangles[5]).area
        union_area = rectangles[4].area + rectangles[5].area - intersection_area
        iou = intersection_area / union_area
        iou_main=[]
        iou_main.append(('global (Region 5) and main (Region 0)', iou))

        batch_main_box=[]
        if iou_main[0][1]<0.4:
            print('iou<0.4')
            batch_main_box.append([region_descriptions[4],dict_image,image_idx['global'],image_idx['image']])
            return batch_main_box
        else:
            return image_idx['global']

    def merge(self,image_idx,num):
        # llama2_7b_chat_hf="/data/users/ruotian_peng/pretrain/llama-3.1-8b-instruct"
        # llm = LLM(model=llama2_7b_chat_hf,max_model_len=40000,tensor_parallel_size=1,gpu_memory_utilization=0.85)
        # tokenizer = AutoTokenizer.from_pretrained(llama2_7b_chat_hf)


        # pipeline = transformers.pipeline(
        #             "text-generation",
        #             model=llama2_7b_chat_hf,
        #             model_kwargs={"torch_dtype": torch.float16},
        #             device_map="auto",
        #         )
        # model = AutoModel.from_pretrained("/data/users/ruotian_peng/pretrain/siglip-google")
        # processor = AutoProcessor.from_pretrained("/data/users/ruotian_peng/pretrain/siglip-google")

    
        


        # blip_model = BLIPScore()
        # error = blip_model.load_model()

        
        result_list=[]
        # for image_idx,key_name in tqdm(enumerate(data_image),total=len(data_image)):
        temp_merge_five=[]
        # for i, data in enumerate(self.json_data):
            # print(data['image'].split('/')[-1])
            # print(self.data_image[image_idx])
            # if data['image'].split('/')[-1] == image_idx['image'].split('/')[-1]:
            
                # image_src = Image.open('/data/users/ruotian_peng/dataset/mmvp/MMVP Images/'+data['image'].split('/')[-1]).convert('RGB')
            
        image_src=Image.open('../coco_image/coco_sample_data_Image_Textualization/'+image_idx['image'].split('/')[-1]).convert('RGB')
        width, height = image_src.size
        main_key='../coco_image/coco_sample_data_Image_Textualization/'+image_idx['image'].split('/')[-1]
        four_box = image_idx['four_box']
        left_top = image_src.crop(four_box[0])
        right_top = image_src.crop(four_box[1])
        left_bottom = image_src.crop(four_box[2])
        right_bottom = image_src.crop(four_box[3])
        dict_image={'1':left_top,'2':right_top,'3':left_bottom,'4':right_bottom}
        def cal_similarity_iou(des,merge):
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
                    img, error = self.blip_model.process_image(merge)
                    
                    score,error = self.blip_model.rank_captions(img, modified_text)
                    # print(modified_text,':',score)

                    similarity_list.append(score)
                
                else:
                    similarity_list.append(0)  
            return similarity_list
        def cal_similarity_same(des,merge):
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
                    img, error = self.blip_model.process_image(merge)
                    
                    score,error = self.blip_model.rank_captions(img, modified_text)
                    # print(modified_text,':',score)

                    similarity_list.append(score)
                
                else:
                    similarity_list.append(0)  
            return similarity_list
        def cal_similarity(des,region_dict):
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
                    
                    img, error = self.blip_model.process_image(dict_image[str(region_dict[number]+1)])
                    # fig, ax = plt.subplots()
                    # ax.imshow(dict_image[str(region_dict[number]+1)])
                    # plt.show()
                    score,error = self.blip_model.rank_captions(img, modified_text)
                    # print(modified_text,':',score)

                    similarity_list.append(score)
                
                else:
                    similarity_list.append(0)  
            return similarity_list
        region_locations=[]
        region_descriptions=[]
        # 遍历字典中的每个键值对
        # print(data_image[image_idx]['local'][0])
        for key, descriptions in image_idx['local'][0].items():
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
        # 初始化一个空列表来存储重叠面积和对应的区域对
        # 初始化一个空列表来存储每对区域的IoU
        iou_values = []
        # print(region_locations)
        # print(rectangles)
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
        
    
        global_modified=image_idx['global']
        
        
        label_iou=[]
        region_dict={'0':[0,1],'1':[1,3],'2':[2,3],'3':[0,2]}
        for i in iou_values:
            if i[1]>0.4:
                label_iou.append(1)
            else:
                label_iou.append(0)
        if label_iou==[0,0,0,0]:
            temp_merge_list=[]
            output=self.batch_group_sameregion_sentence(region_descriptions)
            batch_supplement_all=[]
            for region_number in range(len(region_descriptions)-1):
                temp_group=output[region_number].outputs[0].text
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
                # # print(categories)

        

                if categories['For Contradictory Triples'] is not None:
                    similarity_contra_list=[]
            
                    for triple in categories['For Contradictory Triples']:
                        # region_1=re.findall('Region \d', triple[0])[0]
                        # region_2=re.findall('Region \d', triple[1])[0]
                        # print(triple)
                        similarity_contra=cal_similarity_same(triple,dict_image[str(region_number+1)] )
                        similarity_contra_list.append(similarity_contra)
                        # print(similarity_contra_list)
                similarity_unique=cal_similarity_same(categories['For Unique Triples'],dict_image[str(region_number+1)] )
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
                batch_supplement_all.append(supplement_all)
            temp_merge=self.batch_merge_sameregion(region_descriptions,batch_supplement_all)
            for combined_merge in range(len(region_descriptions)-1):
                temp_merge_list.append(temp_merge[combined_merge].outputs[0].text)

            temp_merge_five=[[width,height],global_modified,region_locations,temp_merge_list,image_idx['image']]

            return temp_merge_five

        else:
            
            temp_merge_list=[]
            index_iou=[]
            real_region_des=[]
            output=self.batch_group_sameregion_sentence(region_descriptions)
            batch_supplement_all=[]
            for region_number in range(len(region_descriptions)-1):
                temp_group=output[region_number].outputs[0].text
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
                        similarity_contra=cal_similarity_same(triple,dict_image[str(region_number+1)] )
                        similarity_contra_list.append(similarity_contra)
                similarity_unique=cal_similarity_same(categories['For Unique Triples'],dict_image[str(region_number+1)] )
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
                batch_supplement_all.append(supplement_all)
            real_merge=self.batch_merge_sameregion(region_descriptions,batch_supplement_all)
            for combined_merge in range(len(region_descriptions)-1):
                real_region_des.append(real_merge[combined_merge].outputs[0].text)
            # real_region_des.append(self.merge_sameregion(region_descriptions[region_number][0],region_descriptions[region_number][1],region_descriptions[region_number][2],supplement_all))
            batch_group_two=[]
            batch_mergeiou=[]
            for number_iou in range(len(label_iou)):
                if label_iou[number_iou]==1:
                    batch_group_two.append([real_region_des[region_dict[str(number_iou)][0]],real_region_des[region_dict[str(number_iou)][1]]])
                
            batch_group=self.batch_group_two_sentence(batch_group_two)
            id_group_two=0
        
            for number_iou in range(len(label_iou)):
                
                if label_iou[number_iou]==1:
                    index_iou.append(number_iou)
                    temp_group=batch_group[id_group_two].outputs[0].text
                    id_group_two+=1
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
                        bbox=self.merge_box(region_locations[region_merge1[0]],region_locations[region_merge1[1]])
                        merge_patch = image_src.crop(bbox)
                        for triple in categories['For Contradictory Triples']:
                            # region_1=re.findall('Region \d', triple[0])[0]
                            # region_2=re.findall('Region \d', triple[1])[0]
                            similarity_contra=cal_similarity_iou(triple,merge_patch )
                            similarity_contra_list.append(similarity_contra)
                    similarity_unique=cal_similarity(categories['For Unique Triples'],region_dict_temp)
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
                    batch_mergeiou.append(supplement_all)
            temp_merge=self.batch_merge_iou([width,height], region_locations, real_region_des,batch_mergeiou,label_iou)
            for iou_merge in temp_merge:
                temp_merge_list.append(iou_merge.outputs[0].text )
            # print(len(temp_merge_list))
            if len(temp_merge_list)==1:
                region_merge1=region_dict[str(index_iou[0])]
                temp_loca=[item for item in [0,1,2,3] if item not in region_merge1 ]
                # print(temp_loca,region_merge1)
                new_region1=self.merge_box(region_locations[region_merge1[0]],region_locations[region_merge1[1]])
                complete_merge=self.merge_four([width,height],global_modified, new_region1, temp_merge_list[0], region_locations[temp_loca[0]], region_descriptions[temp_loca[0]],region_locations[temp_loca[1]], region_descriptions[temp_loca[1]])
            elif len(temp_merge_list)==2:
                region_merge1=region_dict[str(index_iou[0])]
                region_merge2=region_dict[str(index_iou[1])]
                new_region1=self.merge_box(region_locations[region_merge1[0]],region_locations[region_merge1[1]])
                new_region2=self.merge_box(region_locations[region_merge2[0]],region_locations[region_merge2[1]])
                complete_merge=self.merge_three([width,height],global_modified, new_region1, temp_merge_list[0], new_region2, temp_merge_list[1])
            elif len(temp_merge_list)==3:
                region_merge1=region_dict[str(index_iou[0])]
                region_merge2=region_dict[str(index_iou[1])]
                region_merge3=region_dict[str(index_iou[2])]
                new_region1=self.merge_box(region_locations[region_merge1[0]],region_locations[region_merge1[1]])
                new_region2=self.merge_box(region_locations[region_merge2[0]],region_locations[region_merge2[1]])
                new_region3=self.merge_box(region_locations[region_merge3[0]],region_locations[region_merge3[1]])
                complete_merge=self.merge_four([width,height],global_modified, new_region1, temp_merge_list[0], new_region2, temp_merge_list[1],new_region3, temp_merge_list[2])
            elif len(temp_merge_list)==4:
                region_merge1=region_dict[str(index_iou[0])]
                region_merge2=region_dict[str(index_iou[1])]
                region_merge3=region_dict[str(index_iou[2])]
                region_merge4=region_dict[str(index_iou[3])]
                new_region1=self.merge_box(region_locations[region_merge1[0]],region_locations[region_merge1[1]])
                new_region2=self.merge_box(region_locations[region_merge2[0]],region_locations[region_merge2[1]])
                new_region3=self.merge_box(region_locations[region_merge3[0]],region_locations[region_merge3[1]])
                new_region4=self.merge_box(region_locations[region_merge4[0]],region_locations[region_merge4[1]])
                complete_merge=self.merge_five([width,height],global_modified, new_region1, temp_merge_list[0], new_region2, temp_merge_list[1],new_region3, temp_merge_list[2],new_region4, temp_merge_list[3])
        result_json={}
        result_json['image']=image_idx['image']
        result_json['description']=complete_merge
        return result_json
        # result_list.append(result_json)
        # with open('/data/users/ruotian_peng/experement/ablation/iou0.5_blip0.3_1.24.json', 'w') as json_file:
        #         json.dump(result_list, json_file, indent=4)