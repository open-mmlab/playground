import torch
from tqdm import tqdm

from mmrotate.utils import register_all_modules
from data import build_data_loader, build_evaluator, build_visualizer

from segment_anything import sam_model_registry, SamPredictor
from mmrotate.registry import MODELS

from mmengine import Config
from mmengine.runner.checkpoint import _load_checkpoint

from engine import single_sample_step


register_all_modules(init_default_scope=True)

SHOW = True
FORMAT_ONLY = True
MERGE_PATCHES = True
SET_MIN_BOX = False


if __name__ == '__main__':

    sam_checkpoint = r"../segment-anything/checkpoints/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda"

    ckpt_path = './rotated_fcos_sep_angle_r50_fpn_1x_dota_le90-0be71a0c.pth'
    model_cfg_path = 'configs/rotated_fcos/rotated-fcos-hbox-le90_r50_fpn_1x_dota.py'
    # ckpt_path = './rotated_fcos_kld_r50_fpn_1x_dota_le90-ecafdb2b.pth'
    # model_cfg_path = 'configs/rotated_fcos/rotated-fcos-le90_r50_fpn_kld_1x_dota.py'

    model_cfg = Config.fromfile(model_cfg_path).model
    if SET_MIN_BOX:
        model_cfg.test_cfg['min_bbox_size'] = 10

    model = MODELS.build(model_cfg)
    model.init_weights()
    checkpoint = _load_checkpoint(ckpt_path, map_location='cpu')
    sd = checkpoint.get('state_dict', checkpoint)
    print(model.load_state_dict(sd))

    dataloader = build_data_loader('test_without_hbox')
    # dataloader = build_data_loader('trainval_with_hbox')
    evaluator = build_evaluator(MERGE_PATCHES, FORMAT_ONLY)
    evaluator.dataset_meta = dataloader.dataset.metainfo

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    model = model.to(device=device)
    sam = sam.to(device=device)

    predictor = SamPredictor(sam)

    model.eval()
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

        evaluator = single_sample_step(i, data, model, predictor, evaluator, dataloader, device, SHOW)

    torch.save(evaluator, './evaluator.pth')

    metrics = evaluator.evaluate(len(dataloader.dataset))
