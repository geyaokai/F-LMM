import sys
from argparse import Namespace
from pathlib import Path
from PIL import Image
from mmengine.config import Config
sys.path.insert(0, str(Path('.').resolve()))
from scripts.demo.interact import load_model  # imports your existing helpers

cfg_path = Path('configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py')
roi_path = Path('scripts/demo/results/qwen/20251218_052356/round_03/roi_00.png')

args = Namespace(
    config=str(cfg_path),
    checkpoint=None,
    device='cuda',          # adjust to your device, e.g., cuda:0
    device_map='auto',      # or 'none' if you want to keep everything on one GPU
    device_max_memory=None,
    image=None,
    max_new_tokens=256,
    phrase_max_tokens=64,
    max_phrases=6,
    results_dir='scripts/demo/results/qwen',
    inspect_prompt='Describe this region in detail.',
    no_sam=False,
    extra_prompt='',
    max_history_turns=0,
)

cfg = Config.fromfile(str(cfg_path))
model = load_model(cfg, args)

img = Image.open(roi_path).convert('RGB')
min_side = 28
width, height = img.size
scale = max(min_side / width, min_side / height)
new_width = max(min_side, int(round(width * scale)))
new_height = max(min_side, int(round(height * scale)))
img = img.resize((new_width, new_height), Image.BICUBIC)
out = model.answer(image=img, question='what is the color of it', history=None, max_new_tokens=args.max_new_tokens)
print("OUTPUT_HAS_ADDCRITERION:", "addCriterion" in out["output_text"])
print("OUTPUT_TEXT_REPR:", repr(out["output_text"]))