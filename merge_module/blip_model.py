from lavis.models import load_model_and_preprocess
import torch

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
