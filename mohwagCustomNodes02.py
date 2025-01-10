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



class mwFullPipe_Load:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "defaultBehavior": "input"}),
                     "sampler_name": (comfy.samplers.SAMPLER_NAMES, ),
                     "scheduler": (comfy.samplers.SCHEDULER_NAMES, ),
                     }}
    RETURN_TYPES = ("MWFULLPIPE", )
    RETURN_NAMES = ("mwFullPipe", )
    FUNCTION = "doittp"
    CATEGORY = "mohwag"

    def doittp(self, ckpt_name, noise_seed, sampler_name, scheduler):

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        dostuff = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        ckptPipe1b = dostuff[:3]
        mwCkptPipe = (ckptPipe1b, ckptPipe1b)

        mwFullPipe = mwCkptPipe, noise_seed, sampler_name, scheduler
        return (mwFullPipe,)


class mwFullPipe_ckptMerge:
    ckptSelect = ["base", "current"]
    #ckptSelect2 = ["fullPipeCkpt", "base2", "current"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "mwFullPipe": ("MWFULLPIPE",),
                              #"mwCkptPipe2": ("MWCKPTPIPE",),
                              "fullPipeCkpt_InputSelect": (s.ckptSelect, {"default": "current"}),
                              "wgtFullPipeCkpt": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.04}),
                              "ckpt2_name": (folder_paths.get_filename_list("checkpoints"), )
                              #"ckpt2_InputSelect": (s.ckptSelect, {"default": "current"}),
                              #"ckptBase_OutputSelect": (s.ckptSelect2, {"default": "base1"}),
                              }}
    RETURN_TYPES = ("MWFULLPIPE",)
    RETURN_NAMES = ("mwFullPipe",)
    FUNCTION = "merge"
    CATEGORY = "mohwag"

    def merge(self, mwFullPipe, fullPipeCkpt_InputSelect, wgtFullPipeCkpt, ckpt2_name):

        mwCkptPipe1, noise_seed, sampler_name, scheduler = mwFullPipe
        ckptPipe1b, ckptPipe1c = mwCkptPipe1    
        vaeAlways = ckptPipe1b[2]

        model1, clip1, _ = ckptPipe1c #if fullPipeCkpt_InputSelect == "current"
        if fullPipeCkpt_InputSelect == "base":
            model1, clip1, _ = ckptPipe1b
        
        ckpt2_path = folder_paths.get_full_path("checkpoints", ckpt2_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt2_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        model2, clip2, _ = out[:3]


        m = model1.clone()
        kp = model2.get_key_patches("diffusion_model.")
        for k in kp:
            m.add_patches({k: kp[k]}, 1.0 - wgtFullPipeCkpt, wgtFullPipeCkpt)
        n = clip1.clone()
        lp = clip2.get_key_patches()
        for l in lp:
            if l.endswith(".position_ids") or l.endswith(".logit_scale"):
                continue
            n.add_patches({l: lp[l]}, 1.0 - wgtFullPipeCkpt, wgtFullPipeCkpt)
        ckptPipeNew = (m, n, vaeAlways)

        mwCkptPipeNew = (ckptPipe1b, ckptPipeNew)
        mwFullPipeNew = mwCkptPipeNew, noise_seed, sampler_name, scheduler
        return (mwFullPipeNew,)


class mwFullPipe_AddLora:
    ckptSelect = ["base", "current"]
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "mwFullPipe": ("MWFULLPIPE",),
                              "fullPipeCkpt_InputSelect": (s.ckptSelect, {"default": "current"}),
                              "lora_name": (folder_paths.get_filename_list("loras"), ),
                              "strength": ("FLOAT", {"default": 0, "min": -10.0, "max": 10.0, "step": 0.04}),
                              }}
    
    RETURN_TYPES = ("MWFULLPIPE",)
    RETURN_NAMES = ("mwFullPipe",)
    FUNCTION = "load_lora"
    CATEGORY = "mohwag"

    #def load_lora(self, model, clip, lora_name, strength):
    def load_lora(self, mwFullPipe, lora_name, strength, fullPipeCkpt_InputSelect):

        if strength == 0:
            return (mwFullPipe,)

        mwCkptPipe1, noise_seed, sampler_name, scheduler = mwFullPipe
        ckptPipe1b, ckptPipe1c = mwCkptPipe1
        vaeAlways = ckptPipe1b[2]

        model1, clip1, _ = ckptPipe1c #if fullPipeCkpt_InputSelect == "current"
        if fullPipeCkpt_InputSelect == "base":
            model1, clip1, _ = ckptPipe1b


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

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model1, clip1, lora, strength, strength)
        ckptPipeNew = model_lora, clip_lora, vaeAlways

        mwCkptPipeNew = (ckptPipe1b, ckptPipeNew)
        mwFullPipeNew = mwCkptPipeNew, noise_seed, sampler_name, scheduler
        return (mwFullPipeNew,)
    

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


class mwFullPipe_KSA2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"mwFullPipe": ("MWFULLPIPE",),
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

    def sample(self, mwFullPipe, latent_image, positive, negative, cfg, start_at_step, end_at_step, steps, add_noise, return_with_leftover_noise, denoise=1.0):

        mwCkptPipe, noise_seed, sampler_name, scheduler = mwFullPipe

        ckptPipe1b, ckptPipe1c = mwCkptPipe

        model1c, _, _ = ckptPipe1c

        vaeAlways = ckptPipe1b[2]


        #model, clip, vae, noise_seed, sampler_name, scheduler = mohwagPipe
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        rslt = common_ksampler(model1c, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
        rslt1 = (rslt, )
        rslt2 = vaeAlways.decode(rslt["samples"])
        rslt3 = (rslt2, )
        return (rslt1[0], rslt3[0], end_at_step, steps)


class mwFullPipe_ClipBranch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }

    RETURN_TYPES = ("MWFULLPIPE", "CLIP",)
    RETURN_NAMES = ("mwFullPipe", "clip",)
    FUNCTION = "doitfp"
    CATEGORY = "mohwag"

    def doitfp(self, mwFullPipe):
        mwCkptPipe1, _, _, _ = mwFullPipe
        _, ckptPipe1c = mwCkptPipe1
        _, clip2c, _ = ckptPipe1c
        return (mwFullPipe, clip2c)


class mohwagEasyWHL:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "wd": ("STRING", {"multiline": True, "default": "512"}),
            "ht": ("STRING", {"multiline": True, "default": "512"}),
        }}

    RETURN_TYPES = ("LATENT", "DMNSN", "DMNSN")
    RETURN_NAMES = ("ltnt", "wd", "ht")
    FUNCTION = "fun"
    CATEGORY = "mohwag"

    def fun(self, wd, ht):

        if (wd != ""):
            valW = eval(wd)
            valIntW = int(valW)
            strW = str(valIntW)


        if (ht != ""):
            valH = eval(ht)
            valIntH = int(valH)
            strH = str(valIntH)

        latent = torch.zeros([1, 4, valIntH // 8, valIntW // 8], device=self.device)
        latentDictTuple = ({"samples":latent}, )


        return (latentDictTuple[0], strW, strH)


class mohwagEasyScaleWHL:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
             "ltnt": ("LATENT",),
             "wd": ("DMNSN",),
             "ht": ("DMNSN",),
             "multOver8": ("INT", {"default": 8, "min": 0, "max": 64, "step":1}),
             "upscale_method": (cls.upscale_methods,),
             "crop": (cls.crop_methods,)}}

    RETURN_TYPES = ("LATENT", "DMNSN", "DMNSN")
    RETURN_NAMES = ("ltnt", "wd", "ht")
    FUNCTION = "fun"
    CATEGORY = "mohwag"

    def fun(self, ltnt, upscale_method, crop, wd, ht, multOver8):

        if (wd != ""):
            valW = eval(wd)
            rsltW = multOver8 * valW / 8 
            rsltIntW = int(rsltW)
            rsltStrW = str(rsltIntW)

        if (ht != ""):
            valH = eval(ht)
            rsltH = multOver8 * valH / 8 
            rsltIntH = int(rsltH)
            rsltStrH = str(rsltIntH)

        s = ltnt.copy()
        s["samples"] = comfy.utils.common_upscale(ltnt["samples"], rsltIntW // 8, rsltIntH // 8, upscale_method, crop)
        s_tup = (s, )


        return (s_tup[0], rsltStrW, rsltStrH)


class mwCondStart:
    compositionTypes = ["textConcat", "condConcat", "combBoth"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "compType": (s.compositionTypes, {"default": "combBoth"}),
            "text": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ("MWCONDPIPE",)
    RETURN_NAMES = ("mwCondPipe",)
    FUNCTION = "encode"
    CATEGORY = "mohwag"

    def encode(self, clip, text, compType):

        if compType == "textConcat":
            return (([], clip, text, compType),)

        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        condOut = ([[cond, {"pooled_output": pooled}]], )

        if compType == "condConcat":
            return ((condOut[0], clip, "", compType),)
        
        return ((condOut[0], clip, text, compType),)



class mwCondAfterStart:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mwCondPipe": ("MWCONDPIPE",),
            "text": ("STRING", {"multiline": True})}}
    RETURN_TYPES = ("MWCONDPIPE",)
    RETURN_NAMES = ("mwCondPipe",)
    FUNCTION = "concat"
    CATEGORY = "mohwag"

    def concat(self, mwCondPipe, text):

        _, _, _, compType = mwCondPipe

        condOut = ([],)
        textOut = ""

        if compType != "textConcat":
            condInc, clip, _, _ = mwCondPipe

            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            condAddi = ([[cond, {"pooled_output": pooled}]], )
            condAdd = condAddi[0]

            out = []
            cond_Add = condAdd[0][0]
            for i in range(len(condInc)):
                t1 = condInc[i][0]
                tw = torch.cat((t1, cond_Add),1)
                n = [tw, condInc[i][1].copy()]
                out.append(n)
            condOut = (out, )

        if compType != "condConcat":
            _, _, textInc, _ = mwCondPipe
            textOut = textInc + ", " + text

        return ((condOut[0], clip, textOut, compType),)


class mwCondSend:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mwCondPipe": ("MWCONDPIPE",)}}
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("condOut",)
    FUNCTION = "combine"
    CATEGORY = "mohwag"

    def combine(self, mwCondPipe):
        #condInc, clip, textInc, compType = mwCondPipe

        _, _, _, compType = mwCondPipe

        if compType != "condConcat":
            _, clip, textInc, _ = mwCondPipe
            tokens = clip.tokenize(textInc)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            textCondOut = ([[cond, {"pooled_output": pooled}]], )

        if compType == "textConcat":
            return (textCondOut[0],)


        if compType != "textConcat":
            condInc, _, _, _ = mwCondPipe
            concCondOut = (condInc,)

        if compType == "condConcat":
            return  (concCondOut[0],)


        bothCondOut = (textCondOut[0] + concCondOut[0], )
        return (bothCondOut[0],)







NODE_CLASS_MAPPINGS = {
    "mwFullPipe_Load": mwFullPipe_Load,
    "mwFullPipe_ckptMerge": mwFullPipe_ckptMerge,
    "mwFullPipe_AddLora": mwFullPipe_AddLora,
    "mwFullPipe_KSA2": mwFullPipe_KSA2,
    "mwFullPipe_ClipBranch": mwFullPipe_ClipBranch,
    "mohwagEasyWHL": mohwagEasyWHL,
    "mohwagEasyScaleWHL": mohwagEasyScaleWHL,
    "mwCondStart": mwCondStart,
    "mwCondAfterStart": mwCondAfterStart,
    "mwCondSend": mwCondSend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mwFullPipe_Load": "mwFullPipe_Load",
    "mwFullPipe_ckptMerge": "mwFullPipe_ckptMerge",
    "mwFullPipe_AddLora": "mwFullPipe_AddLora",
    "mwFullPipe_KSA2": "mwFullPipe_KSA2",
    "mwFullPipe_ClipBranch": "mwFullPipe_ClipBranch",
    "mohwagEasyWHL": "mohwagEasyWHL",
    "mohwagEasyScaleWHL": "mohwagEasyScaleWHL",
    "mwCondStart": "mwCondStart",
    "mwCondAfterStart": "mwCondAfterStart",
    "mwCondSend": "mwCondSend",
}