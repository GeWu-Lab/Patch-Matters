import transformers
import torch




import  torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch



class Visualcheck:
    def __init__(self, api_key, llm_id="llama2_7b_chat"):
        # openai.api_key = api_key
        self.llm_id = llm_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        llama2_7b_chat_hf="/data/users/ruotian_peng/pretrain/llama-3.1-8b-instruct"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=llama2_7b_chat_hf,
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto",
        )
    def Selfcheck_description(self, global_description, local_description,
                                                width, height):


        # prompt = f"\nGlobal Description: {local_description}\nLocal Description: {global_description}\nThe image resolution is:{width}X{height}\nBased on the global description, local description of the generated image, please generate a detailed image description (only one paragraph with no more than 10 sentences) that describe the color, spatial position, shape, size, material of each object, and relationship among objects. The location of the object should be in natural language format instead of numerical coordinates.\n"
        prompt  = ''
        if self.llm_id in ["gpt-3.5-turbo", "gpt-4"]:
            # completion = openai.ChatCompletion.create(
            #     model=self.llm_id, 
            #     messages = [
            #     {"role": "user", "content" : prompt}]
            # )
            return

        elif self.llm_id in ["llama2_7b_chat"]:
            image_size = f"Image size:[0, 0, {width}, {height}]. Image description:"
            # global_description = global_description + image_size
            
            local_description = re.split(r'\n[\r\n]*', local_description)
            local_description_list = [paragragh for i, paragragh in enumerate(local_description) if paragragh]
            # print('local_description_list:',local_description_list)

            region_descriptions = []
            for desc in local_description_list:
                # 检查字符串是否包含 'Region description:'
                if 'Region description:' in desc:
                    # 提取 'Region description:' 后面的内容
                    region_description = desc.split('Region description:')[1].strip()
                    region_descriptions.append(region_description)
            # from transformers import AutoTokenizer, AutoModelForCausalLM
            # llama2_7b_chat_hf = "/data/users/haiying_he/cache/huggingface/hub/meta-llama/Llama-2-7b-chat-hf"
            # tokenizer = AutoTokenizer.from_pretrained(llama2_7b_chat_hf)
            # model = AutoModelForCausalLM.from_pretrained(llama2_7b_chat_hf, torch_dtype=torch.float16).to(self.device)
            system="""
            You are a language model tasked with analyzing five paragraphs. Your tasks are:

            1. **Identify sentences describing the same thing**: Across all paragraphs, identify sentences that describe the same thing and group them together. Ensure that each sentence belongs to only one group, and identical sentences should be grouped together.
            2. **Identify contradictory sentences**: Across all paragraphs, identify sentences that contradict each other and list them as contradictory pairs. Contradictory sentences should only appear in the "contradictory pairs" list and should not be categorized as "describing the same thing."
            3. **Identify unique sentences**: Across all paragraphs, identify sentences that appear only once and list them. If a sentence has already been categorized (e.g., as describing the same thing or contradictory), it should not be listed as unique.

            Please ensure the following:
            - Consider each sentence in its entirety and do not split sentences, even if they contain multiple clauses.
            - Each sentence should be strictly categorized into only one group, avoiding multiple categorizations.

            You will receive five paragraphs of text. Each paragraph may contain multiple sentences. You need to compare the sentences across these paragraphs to find those that describe the same thing, those that contradict each other, and those that appear only once.

            Example Input Paragraphs:

            Paragraph1: The dining room is connected to the living room by an open archway, and a wooden table with a red and white checkered tablecloth can be seen through it. The dining room is brightly lit by a chandelier, making the space feel warm and inviting. On the table, there is a vase with fresh flowers.

            Paragraph2: A man and a woman are seated at the table. The man is wearing a blue shirt, and the woman is wearing a red dress. The dining room is connected to the living room, which can be seen through the open archway. A large painting of a landscape hangs on the wall above the sofa.

            Paragraph3: The dining room features a wooden table covered with a red and white checkered tablecloth. The table is set for a meal, with plates, forks, and glasses arranged neatly. A window allows natural light to flood the room, and a vase of flowers is placed in the center of the table.

            Paragraph4: The room is dimly lit by a single lamp in the corner, casting long shadows across the floor. The wooden table, draped with a red and white checkered tablecloth, is surrounded by four chairs. The windows are adorned with heavy velvet curtains.

            Paragraph5: The couple is sitting at the table, engaged in conversation. The man is wearing a blue shirt, and the woman is wearing a red dress. The table is covered with a red and white checkered tablecloth, and the room is softly lit, creating a cozy atmosphere.

            Example Output:

            Sentences describing the same thing:

            Group 1:
            - "The dining room is connected to the living room by an open archway, and a wooden table with a red and white checkered tablecloth can be seen through it. (Paragraph 1)"
            - "The dining room is connected to the living room, which can be seen through the open archway. (Paragraph 2)"

            Group 2:
            - "The table is covered with a red and white checkered tablecloth. (Paragraph 3)"
            - "The wooden table, draped with a red and white checkered tablecloth, is surrounded by four chairs. (Paragraph 4)"
            - "The table is covered with a red and white checkered tablecloth, and the room is softly lit, creating a cozy atmosphere. (Paragraph 5)"

            Group 3:
            - "A vase of flowers is placed in the center of the table. (Paragraph 3)"
            - "On the table, there is a vase with fresh flowers. (Paragraph 1)"

            Group 4:
            - "A man and a woman are seated at the table. (Paragraph 2)"
            - "The couple is sitting at the table, engaged in conversation. (Paragraph 5)"

            Group 5:
            - "The man is wearing a blue shirt, and the woman is wearing a red dress. (Paragraph 2)"
            - "The man is wearing a blue shirt, and the woman is wearing a red dress. (Paragraph 5)"

            Contradictory sentence pairs:

            ["The dining room is brightly lit by a chandelier, making the space feel warm and inviting. (Paragraph 1)", "The room is dimly lit by a single lamp in the corner, casting long shadows across the floor. (Paragraph 4)"]

            Sentences that only appear once:

            - "A large painting of a landscape hangs on the wall above the sofa. (Paragraph 2)"
            - "The windows are adorned with heavy velvet curtains. (Paragraph 4)"
            - "The table is set for a meal, with plates, forks, and glasses arranged neatly. (Paragraph 3)"
            """
            user=f"""
            ###Paragraph 1: {global_description}

            ###Paragraph 2: {region_descriptions[0]}

            ###Paragraph 3: {region_descriptions[1]}

            ###Paragraph 4: {region_descriptions[2]}
            
            ###Paragraph 5: {region_descriptions[3]}
            """
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            prompt = self.pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            terminators = [
                    self.pipeline.tokenizer.eos_token_id,
                    self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
            print(prompt)
            outputs = self.pipeline(
                    prompt,
                    max_new_tokens=4096,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,)
            # inputs = self.tokenizer(llama3_8b_prompt, return_tensors="pt").to(self.device)
            # outputs = self.model.generate(**inputs, do_sample=True,max_length=4096)
            # merged_caption = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

            # print('---------\n', tokenizer.decode(
            #     outputs[0][:], skip_special_tokens=True), '-----------\n'
            #     )
            merged_caption=outputs[0]["generated_text"][len(prompt):]
            return merged_caption
        else:
                # completion = openai.ChatCompletion.create(
                #     model=self.llm_id, 
                #     messages = [
                #     {"role": "user", "content" : prompt}]
                # )
                return
        # return completion['choices'][0]['message']['content'].strip().replace("\n", " ")
   