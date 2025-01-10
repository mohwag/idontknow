import torch

import os
import sys
import json
import hashlib
import traceback
import math
import time
import random

from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import numpy as np
import safetensors.torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet

import comfy.clip_vision

import comfy.model_management
from comfy.cli_args import args

import importlib

import folder_paths
import latent_preview

class mohwagLoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name": (folder_paths.get_filename_list("loras"), ),
                              "strength": ("FLOAT", {"default": 0, "min": -10.0, "max": 10.0, "step": 0.04}),
                              }}
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "mohwag"

    def load_lora(self, model, clip, lora_name, strength):
        if strength == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength, strength)
        return (model_lora, clip_lora)
    

class mohwagModelMerge:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model1": ("MODEL",),
                              "clip1": ("CLIP",),
                              "model2": ("MODEL",),
                              "clip2": ("CLIP",),
                              "ratio": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.04}),
                              }}
    RETURN_TYPES = ("MODEL","CLIP")
    FUNCTION = "merge"
    CATEGORY = "mohwag"

    def merge(self, model1, model2, clip1, clip2, ratio):
        m = model1.clone()
        kp = model2.get_key_patches("diffusion_model.")
        for k in kp:
            m.add_patches({k: kp[k]}, 1.0 - ratio, ratio)

        n = clip1.clone()
        lp = clip2.get_key_patches()
        for l in lp:
            if l.endswith(".position_ids") or l.endswith(".logit_scale"):
                continue
            n.add_patches({l: lp[l]}, 1.0 - ratio, ratio)
        return (m,n)


class mohwagToPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "model": ("MODEL",),
                     "clip": ("CLIP",),
                     "vae": ("VAE",),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "defaultBehavior": "input"}),
                     "sampler_name": (comfy.samplers.SAMPLER_NAMES, ),
                     "scheduler": (comfy.samplers.SCHEDULER_NAMES, ),
                     }}

    RETURN_TYPES = ("MOHWAG_PIPE", )
    RETURN_NAMES = ("mohwagPipe", )
    FUNCTION = "doittp"
    CATEGORY = "mohwag"

    def doittp(self, *args, **kwargs):
        pipe = (kwargs['model'], kwargs['clip'], kwargs['vae'], kwargs['noise_seed'], kwargs['sampler_name'], kwargs['scheduler'],)
        return (pipe, )


class mohwagFromPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mohwagPipe": ("MOHWAG_PIPE",), }, }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "INT", comfy.samplers.SAMPLER_NAMES, comfy.samplers.SCHEDULER_NAMES)
    RETURN_NAMES = ("model", "clip", "vae", "noise_seed", "sampler_name", "scheduler")
    FUNCTION = "doitfp"
    CATEGORY = "mohwag"

    def doitfp(self, mohwagPipe):
        model, clip, vae, noise_seed, sampler_name, scheduler = mohwagPipe
        return model, clip, vae, noise_seed, sampler_name, scheduler


class mohwagclipBranchPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mohwagPipe": ("MOHWAG_PIPE",), }, }

    RETURN_TYPES = ("MOHWAG_PIPE", "CLIP")
    RETURN_NAMES = ("mohwagPipe", "clip")
    FUNCTION = "doitfp"
    CATEGORY = "mohwag"

    def doitfp(self, mohwagPipe):
        model, clip, vae, noise_seed, sampler_name, scheduler = mohwagPipe
        return mohwagPipe, clip
    

def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return out


class mohwagPipeKSA:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"mohwagPipe": ("MOHWAG_PIPE",),
                    "latent_image": ("LATENT", ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 20, "min": 0, "max": 10000}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "add_noise": (["enable", "disable"], ),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                     }
                }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "image")
    FUNCTION = "sample"
    CATEGORY = "mohwag"

    def sample(self, mohwagPipe, latent_image, positive, negative, cfg, start_at_step, end_at_step, steps, add_noise, return_with_leftover_noise, denoise=1.0):
        model, clip, vae, noise_seed, sampler_name, scheduler = mohwagPipe
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        rslt = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
        rslt1 = (rslt, )
        rslt2 = vae.decode(rslt["samples"])
        rslt3 = (rslt2, )
        return (rslt1[0], rslt3[0])


class mohwagPipeKSA2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"mohwagPipe": ("MOHWAG_PIPE",),
                    "latent_image": ("LATENT", ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 20, "min": 0, "max": 10000}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "add_noise": (["enable", "disable"], ),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                     }
                }

    RETURN_TYPES = ("LATENT", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("latent", "image", "end_at_step", "steps")
    FUNCTION = "sample"
    CATEGORY = "mohwag"    

    def sample(self, mohwagPipe, latent_image, positive, negative, cfg, start_at_step, end_at_step, steps, add_noise, return_with_leftover_noise, denoise=1.0):
        model, clip, vae, noise_seed, sampler_name, scheduler = mohwagPipe
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        rslt = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
        rslt1 = (rslt, )
        rslt2 = vae.decode(rslt["samples"])
        rslt3 = (rslt2, )
        return (rslt1[0], rslt3[0], end_at_step, steps)



class mohwagEasyWH:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("STRING", {"multiline": True, "default": "512"}),
            "height": ("STRING", {"multiline": True, "default": "512"}),
        }}

    RETURN_TYPES = ("FLOAT", "INT", "FLOAT", "INT")
    RETURN_Names = ("floatW", "intW", "floatH", "intH")
    FUNCTION = "fun"
    CATEGORY = "mohwag"

    def fun(self, width, height):

        if (width != ""):
            answerW = eval(width)

        if (height != ""):
            answerH = eval(height)

        return (answerW, int(answerW), answerH, int(answerH))


class mohwagEasyScaleWH:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("INT", {"default": 512, "min": 0, "max": 10000, "step":8}),
            "height": ("INT", {"default": 512, "min": 0, "max": 10000, "step":8}),
            "multOver8": ("INT", {"default": 8, "min": 0, "max": 64, "step":1}),
        }}

    RETURN_TYPES = ("FLOAT", "INT", "FLOAT", "INT")
    RETURN_Names = ("floatW", "intW", "floatH", "intH")
    FUNCTION = "fun"
    CATEGORY = "mohwag"

    def fun(self, width, height, multOver8):

        if (width != ""):
            answerW = multOver8 * width / 8 

        if (height != ""):
            answerH = multOver8 * height / 8

        return (answerW, int(answerW), answerH, int(answerH))


class mohwagEasyWHL:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("STRING", {"multiline": True, "default": "512"}),
            "height": ("STRING", {"multiline": True, "default": "512"}),
        }}

    RETURN_TYPES = ("FLOAT", "INT", "FLOAT", "INT", "LATENT")
    RETURN_Names = ("floatW", "intW", "floatH", "intH", "latent")
    FUNCTION = "fun"
    CATEGORY = "mohwag"

    def fun(self, width, height):

        if (width != ""):
            answerW = eval(width)
            ansIntW = int(answerW)

        if (height != ""):
            answerH = eval(height)
            ansIntH = int(answerH)

        latent = torch.zeros([1, 4, ansIntH // 8, ansIntW // 8], device=self.device)
        latentDictTuple = ({"samples":latent}, )


        return (answerW, ansIntW, answerH, ansIntH, latentDictTuple[0])


class mohwagEasyScaleWHL:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
             "samples": ("LATENT",), "upscale_method": (cls.upscale_methods,),
             "width": ("INT", {"default": 512, "min": 0, "max": 10000, "step":8}),
             "height": ("INT", {"default": 512, "min": 0, "max": 10000, "step":8}),
             "multOver8": ("INT", {"default": 8, "min": 0, "max": 64, "step":1}),
             "crop": (cls.crop_methods,)}}

    RETURN_TYPES = ("FLOAT", "INT", "FLOAT", "INT", "LATENT")
    RETURN_Names = ("floatW", "intW", "floatH", "intH", "latent")
    FUNCTION = "fun"
    CATEGORY = "mohwag"

    def fun(self, samples, upscale_method, crop, width, height, multOver8):

        if (width != ""):
            answerW = multOver8 * width / 8 
            ansIntW = int(answerW)

        if (height != ""):
            answerH = multOver8 * height / 8
            ansIntH = int(answerH)

        s = samples.copy()
        s["samples"] = comfy.utils.common_upscale(samples["samples"], ansIntW // 8, ansIntH // 8, upscale_method, crop)
        s_tup = (s, )


        return (answerW, ansIntW, answerH, ansIntH, s_tup[0])
    

'''
class CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"text": ("STRING", {"multiline": True}), 
                 "clip": ("CLIP", )}}
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], )
'''

class mohwagEasyScale2WHL:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
             "samples": ("LATENT",),
             "width": ("INT", {"defaultBehavior": "input"}),
             "height": ("INT", {"defaultBehavior": "input"}),
             "multOver8": ("INT", {"default": 8, "min": 0, "max": 64, "step":1}),
             "upscale_method": (cls.upscale_methods,),
             "crop": (cls.crop_methods,)}}

    RETURN_TYPES = ("FLOAT", "INT", "FLOAT", "INT", "LATENT")
    RETURN_Names = ("floatW", "intW", "floatH", "intH", "latent")
    FUNCTION = "fun"
    CATEGORY = "mohwag"

    def fun(self, samples, upscale_method, crop, width, height, multOver8):

        if (width != ""):
            answerW = multOver8 * width / 8 
            ansIntW = int(answerW)

        if (height != ""):
            answerH = multOver8 * height / 8
            ansIntH = int(answerH)

        s = samples.copy()
        s["samples"] = comfy.utils.common_upscale(samples["samples"], ansIntW // 8, ansIntH // 8, upscale_method, crop)
        s_tup = (s, )


        return (answerW, ansIntW, answerH, ansIntH, s_tup[0])




NODE_CLASS_MAPPINGS = {
"mohwagLoraLoader": mohwagLoraLoader,
"mohwagModelMerge": mohwagModelMerge,
"mohwagToPipe": mohwagToPipe,
"mohwagFromPipe": mohwagFromPipe,
"mohwagclipBranchPipe": mohwagclipBranchPipe,
"mohwagPipeKSA": mohwagPipeKSA,
"mohwagPipeKSA2": mohwagPipeKSA2,
"mohwagEasyWH": mohwagEasyWH,
"mohwagEasyScaleWH": mohwagEasyScaleWH,
"mohwagEasyWHL": mohwagEasyWHL,
"mohwagEasyScaleWHL": mohwagEasyScaleWHL,
"mohwagEasyScale2WHL": mohwagEasyScale2WHL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mohwagLoraLoader": "mohwagLoraLoader",
    "mohwagModelMerge": "mohwagModelMerge",
    "mohwagToPipe": "mohwagToPipe",
    "mohwagFromPipe": "mohwagFromPipe",
    "mohwagclipBranchPipe": "mohwagclipBranchPipe",
    "mohwagPipeKSA": "mohwagPipeKSA",
    "mohwagPipeKSA2": "mohwagPipeKSA2",
    "mohwagEasyWH": "mohwagEasyWH",
    "mohwagEasyScaleWH": "mohwagEasyScaleWH",
    "mohwagEasyWHL": "mohwagEasyWHL",
    "mohwagEasyScaleWHL": "mohwagEasyScaleWHL",
    "mohwagEasyScale2WHL": "mohwagEasyScale2WHL",
}