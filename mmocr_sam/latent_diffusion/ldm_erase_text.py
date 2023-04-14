import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def make_batch(image, mask_pil_image, img_size, device):
    if isinstance(image, str):
        if img_size is not None:
            image = np.array(Image.open(image).convert("RGB").resize(img_size))
        else:
            image = np.array(Image.open(image).convert("RGB")) # need to resize to a image_size
    else:
        if img_size is not None:
            image = np.array(image.convert("RGB").resize(img_size))
        else:
            image = np.array(image.convert("RGB")) # need to resize to a image_size
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    if img_size is not None:
        mask = np.array(mask_pil_image.convert("L").resize(img_size))
    else:
        mask = np.array(mask_pil_image.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1 - mask) * image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k] * 2.0 - 1.0
    return batch


def erase_text_from_image(img_path,
                          mask_pil_img,
                          model,
                          device,
                          opt,
                          img_size=None,
                          steps=None):
    sampler = DDIMSampler(model)
    with torch.no_grad():
        with model.ema_scope():

            if img_size is None:
                batch = make_batch(
                    img_path,
                    mask_pil_img,
                    img_size=opt.img_size,
                    device=device)
            else:
                batch = make_batch(
                    img_path, mask_pil_img, img_size=img_size, device=device)
            # encode masked image and concat downsampled mask
            c = model.cond_stage_model.encode(batch["masked_image"])
            cc = torch.nn.functional.interpolate(
                batch["mask"], size=c.shape[-2:])
            c = torch.cat((c, cc), dim=1)

            shape = (c.shape[1] - 1, ) + c.shape[2:]
            if steps is None:
                samples_ddim, _ = sampler.sample(
                    S=opt.steps,
                    conditioning=c,
                    batch_size=c.shape[0],
                    shape=shape,
                    verbose=False)
            else:
                samples_ddim, _ = sampler.sample(
                    S=steps,
                    conditioning=c,
                    batch_size=c.shape[0],
                    shape=shape,
                    verbose=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)

            image = torch.clamp((batch["image"] + 1.0) / 2.0, min=0.0, max=1.0)
            mask = torch.clamp((batch["mask"] + 1.0) / 2.0, min=0.0, max=1.0)
            predicted_image = torch.clamp(
                (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            inpainted = (1 - mask) * image + mask * predicted_image
            inpainted = inpainted.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255

            return Image.fromarray(inpainted.astype(np.uint8))
