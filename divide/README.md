# Patch Matters: Training-free Fine-grained Image Caption Enhancement via Local Perception


## Installation

```bash
conda create -n patch_matters python==3.8.19

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install -r requirements.txt
```

## Usage

### Checkpoints

Following [Ovdet](https://github.com/wusize/ovdet) to download weights and files and place them in the following format.

`ovdet/checkpoints`
- clip_vitb32.pth
- res50_fpn_soco_star_400.pth
- this_repo_R-50-FPN_CLIP_iter_90000.pth

`ovdet/data/coco/annotations`
- instances_train2017.json
- instances_val2017.json

`ovdet/data/coco/wusize`
- captions_train2017_tags_allcaps.json
- instances_train2017_base.json
- instances_train2017_novel.json
- instances_val2017_base.json
- instances_val2017_novel.json


### Run

```python
python divide/generate_four_box.py --image_folder 'your image folder' --four_box_save_path 'four_box.json' --object_box_save_path 'object_box.json'

python ovdet/get_main_box.py --image_folder 'your image folder' --object_box_save_path 'object_box.json' --main_box_save_path 'main_box.json'

python ovdet/get_main_box.py --image_folder 'your image folder' --object_box_save_path 'object_box.json' --main_box_save_path 'main_box.json'
```