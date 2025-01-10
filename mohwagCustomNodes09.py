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
    def INPUT_TYPES(moh):
        return {"required": {
                     "mwCkpt": (("MWCKPT",)),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "sampler_name": (comfy.samplers.SAMPLER_NAMES, ),
                     "scheduler": (comfy.samplers.SCHEDULER_NAMES, ),
                     }}
    RETURN_TYPES = ("MWFULLPIPE", )
    RETURN_NAMES = ("mwFullPipe", )
    FUNCTION = "mwFPL"
    CATEGORY = "mohwag"

    def mwFPL(self, mwCkpt, seed, sampler_name, scheduler):

        mwFullPipe = mwCkpt, seed, sampler_name, scheduler
        return (mwFullPipe,)



class mwCkpt_Load: #comfy nodes.py LoraLoader
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": { 
                              "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                              }}
    RETURN_TYPES = ("MWCKPT",)
    RETURN_NAMES = ("mwCkpt",)
    FUNCTION = "mwCPL"

    CATEGORY = "mohwag"

    def mwCPL(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return (out[:3],)
    


class mwFullPipe_ckptMerge:
    ckptSelect = ["base", "current"]

    @classmethod
    def INPUT_TYPES(moh):
        return {"required": { "mwFullPipe": ("MWFULLPIPE",),
                              "mwCkpt": ("MWCKPT",),
                              "wgtFullPipeCkpt": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.04}),

                              }}
    RETURN_TYPES = ("MWFULLPIPE",)
    RETURN_NAMES = ("mwFullPipe",)
    FUNCTION = "mwFPCPM"
    CATEGORY = "mohwag"

    def mwFPCPM(self, mwFullPipe, mwCkpt, wgtFullPipeCkpt):

        if wgtFullPipeCkpt == 1:
            return (mwFullPipe,)

        mwCkpt1, seed, sampler_name, scheduler = mwFullPipe
        vaeAlways = mwCkpt1[2]

        model1, clip1, _ = mwCkpt1
        
        model2, clip2, _ = mwCkpt[:3]

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
        mwCkptNew = (m, n, vaeAlways)

        mwFullPipeNew = mwCkptNew, seed, sampler_name, scheduler
        return (mwFullPipeNew,)
    


class mwLora_Load: #comfy nodes.py LoraLoader
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": { 
                              "lora_name": (folder_paths.get_filename_list("loras"), ),
                              }}
    RETURN_TYPES = ("MWLORA",)
    RETURN_NAMES = ("mwLora",)
    FUNCTION = "mwLL"

    CATEGORY = "mohwag"

    def mwLL(self, lora_name):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        return (lora,)



class mwFullPipe_loraMerge:
    ckptSelect = ["base", "current"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": { "mwFullPipe": ("MWFULLPIPE",),
                              "mwLora": ("MWLORA",),
                              "strength": ("FLOAT", {"default": 0, "min": -10.0, "max": 10.0, "step": 0.04}),
                              }}
    RETURN_TYPES = ("MWFULLPIPE",)
    RETURN_NAMES = ("mwFullPipe",)
    FUNCTION = "mwFPLM"
    CATEGORY = "mohwag"

    def mwFPLM(self, mwFullPipe, mwLora, strength):

        if strength == 0:
            return (mwFullPipe,)

        mwCkpt1, seed, sampler_name, scheduler = mwFullPipe
        vaeAlways = mwCkpt1[2]

        model1, clip1, _ = mwCkpt1

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model1, clip1, mwLora, strength, strength)
        mwCkptNew = model_lora, clip_lora, vaeAlways

        mwFullPipeNew = mwCkptNew, seed, sampler_name, scheduler
        return (mwFullPipeNew,)



class mwLtntPipe_Create: #WAS_Latent_Size_To_Number
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "ltnt": ("LATENT", ),
            "initialW": ("INT", {"default": 512, "min": 64, "max": 3072, "step": 64 }),
            "initialH": ("INT", {"default": 512, "min": 64, "max": 3072, "step": 64 }),
            }}

    RETURN_TYPES = ("MWLTNTPIPE",)
    RETURN_NAMES = ("ltntPipe",)
    FUNCTION = "mwLPC"
    CATEGORY = "mohwag"

    def mwLPC(self, ltnt, initialW, initialH):
        size_dict = {}
        i = 0
        for tensor in ltnt['samples'][0]:
            shape = tensor.shape
            tensor_height = shape[-2]
            tensor_width = shape[-1]
            size_dict.update({i:[tensor_width, tensor_height]})
        outputW = 8 * float(size_dict[0][0])
        outputH = 8 * float(size_dict[0][1])
        return ((ltnt, initialW, initialH, outputW, outputH),)



class mwLtntPipe_CropScale: #WAS_Latent_Size_To_Number
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "ltntPipe": ("MWLTNTPIPE", ),
            "cropLft": ("INT", {"default": 0, "min": 0, "max": 3072, "step": 8}),
            "cropRgt": ("INT", {"default": 0, "min": 0, "max": 3072, "step": 8}),
            "outDeltaW": ("INT", {"default": 0, "min": 0, "max": 3072, "step": 32}),
            "cropTop": ("INT", {"default": 0, "min": 0, "max": 3072, "step": 8}),
            "cropBtm": ("INT", {"default": 0, "min": 0, "max": 3072, "step": 8}),
            "outDeltaH": ("INT", {"default": 0, "min": 0, "max": 3072, "step": 32}),
            "cropToOutput_upscaleMethod": (moh.upscale_methods,),
            }}

    RETURN_TYPES = ("MWLTNTPIPE", "INT", "INT",)
    RETURN_NAMES = ("ltntPipe", "wd", "ht",)
    FUNCTION = "mwLPCS"
    CATEGORY = "mohwag"

    def mwLPCS(self, ltntPipe, cropLft, cropRgt, outDeltaW, cropTop, cropBtm, outDeltaH, cropToOutput_upscaleMethod):

        ltnt, _, _, inputW, inputH = ltntPipe

        outputW = inputW - outDeltaW
        outputH = inputH - outDeltaH

        if (cropLft == 0 and cropRgt == 0) and (cropTop == 0 and cropBtm == 0):
            if outDeltaW == 0 and outDeltaH == 0:
                return (ltntPipe,)
            else:
                cropdLtnt = ltnt.copy()
        else:
            offsetW = cropLft
            endW = inputW - cropRgt
            offsetH = cropTop
            endH = inputH - cropBtm

            offsetW8 = offsetW // 8
            endW8 = endW // 8
            offsetH8 = offsetH // 8
            endH8 = endH // 8

            cropdLtnt = ltnt.copy()
            ltntSamples = ltnt['samples']
            cropdLtnt['samples'] = ltntSamples[:,:,offsetH8:endH8, offsetW8:endW8]

        if cropLft + cropRgt == outDeltaW and cropTop + cropBtm == outDeltaH:
            finalLtnt = cropdLtnt
        else:
            crop = "disabled"
            upscaleLtnt = cropdLtnt.copy()
            upscaleLtnt["samples"] = comfy.utils.common_upscale(cropdLtnt["samples"], outputW // 8, outputH  // 8, cropToOutput_upscaleMethod, crop)
            finalLtnt = upscaleLtnt

        ltntPipeNew = finalLtnt, outputW, outputH, outputW, outputH

        return (ltntPipeNew, outputW, outputH)


class mwLtntPipeBranch1: #WAS_Latent_Size_To_Number
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "ltntPipe": ("MWLTNTPIPE", ),
            }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "mwLPBO"
    CATEGORY = "mohwag"

    def mwLPBO(self, ltntPipe):

        ltnt, _, _, _, _ = ltntPipe

        return (ltnt,)
    

class mwLtntPipe_View: #WAS_Latent_Size_To_Number
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "mwFullPipe": ("MWFULLPIPE",),
            "ltntPipe": ("MWLTNTPIPE", ),
            }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("img",)
    FUNCTION = "mwLPV"
    CATEGORY = "mohwag"

    def mwLPV(self, mwFullPipe, ltntPipe):

        ltnt, _, _, _, _ = ltntPipe

        mwCkpt1, _, _, _ = mwFullPipe
        vaeAlways = mwCkpt1[2]
    
        img = vaeAlways.decode(ltnt["samples"])
        imgOut = (img, )

        return imgOut



def cuString(aString:str) -> str:
    return aString.strip().replace("  ", " ").replace("  ", " ")

def prepString(aBlock:str) -> list[str]:
    aList = aBlock.splitlines()
    while("" in aList):
        aList.remove("")

    aList = map(cuString, aList)
    aList = list(aList)
    return aList   



class mwCond:
    compositionTypes = ["textConcat", "condConcat", "combBoth"]

    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            #"clip": ("CLIP", ),
            "compType": (moh.compositionTypes, {"default": "combBoth"}),
            "condText": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ("MWCOND",)
    RETURN_NAMES = ("mwCond",)
    FUNCTION = "mwC"
    CATEGORY = "mohwag"

    def mwC(self, compType, condText):
        condListF = prepString(condText)
        return ((compType, condListF),)



def func_mwCondPrep(aclip, amwCond):

    compType, condListF = amwCond
    textConcF = ", ".join(map(str,condListF))

    if compType != "textConcat":
        condInc = []
        for aString in condListF:
            tokens = aclip.tokenize(aString)
            cond, pooled = aclip.encode_from_tokens(tokens, return_pooled=True)
            condAdd =  [[cond, {"pooled_output": pooled}]]
            if len(condInc) == 0:
                condCurr = condAdd
            else:
                concCondOut = []
                condAddPart = condAdd[0][0]
                for i in range(len(condInc)):
                    t1 = condInc[i][0]
                    tw = torch.cat((t1, condAddPart),1)
                    n = [tw, condInc[i][1].copy()]
                    concCondOut.append(n)
                    condCurr = concCondOut
            condInc = condCurr
        concCondOut = condInc
    if compType == "condConcat":
        return  (concCondOut,)

    if compType != "condConcat":
        tokens = aclip.tokenize(textConcF)
        cond, pooled = aclip.encode_from_tokens(tokens, return_pooled=True)
        textCondOut = [[cond, {"pooled_output": pooled}]]
    if compType == "textConcat":
        return (textCondOut,)

    bothCondOut = (textCondOut + concCondOut,)
    return bothCondOut



class mwCondPrep:
    compositionTypes = ["textConcat", "condConcat", "combBoth"]

    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "clip": ("CLIP", ),
            "mwCond": ("MWCOND", ),
            #"condText": ("STRING", {"multiline": True})
            }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("cond",)
    FUNCTION = "mwCP"
    CATEGORY = "mohwag"

    def mwCP(self, clip, mwCond):
        condReturn = func_mwCondPrep(clip, mwCond)
        return (condReturn[0],)



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



class mwFullPipe_KSAStart:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
    @classmethod
    def INPUT_TYPES(moh):
        return {"required":
                    {"mwFullPipe": ("MWFULLPIPE",),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "stepEnd": ("INT", {"default": 20, "min": 0, "max": 10000}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "seed_deltaVsOrig": ("INT", {"default": 0, "min": 0, "max": 10000, "defaultBehavior": "input"}),
                    "add_noise": (["enable", "disable"], ),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "startW": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 32 }),
                    "startH": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 32 }),
                    },
                }

    RETURN_TYPES = ("MWFULLPIPE", "MWLTNTPIPE", "CONDITIONING", "CONDITIONING", "INT", "INT", "LATENT", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("fullPipe", "ltntPipe", "positive", "negative", "stepEnd", "steps", "latent", "image", "wd", "ht")
    FUNCTION = "mwFPKSAS"
    CATEGORY = "mohwag"

    def mwFPKSAS(self, mwFullPipe, positive, negative, cfg, stepEnd, steps, seed_deltaVsOrig, add_noise, return_with_leftover_noise, startW, startH, denoise=1.0):
        stepStart = 0
        
        mwCkpt1, seedOrig, sampler_name, scheduler = mwFullPipe
        model1, _, _ = mwCkpt1
        seed = seedOrig + seed_deltaVsOrig
        vaeAlways = mwCkpt1[2]

        #create latent
        latent = torch.zeros([1, 4, startH // 8, startW // 8], device=self.device)
        latentDictTuple = ({"samples":latent}, )


        #positiveUse = positive
        #negativeUse = negative


        #diffusion run
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        ltnt = common_ksampler(model1, seed, steps, cfg, sampler_name, scheduler, positive, negative, latentDictTuple[0], denoise=denoise, disable_noise=disable_noise, start_step=stepStart, last_step=stepEnd, force_full_denoise=force_full_denoise)
        ltntOut = (ltnt, )
        
        img = vaeAlways.decode(ltnt["samples"])
        imgOut = (img, )

        ltntPipe = ltntOut[0], startW, startH, startW, startH

        return (mwFullPipe, ltntPipe, positive, negative, stepEnd, steps, ltntOut[0], imgOut[0], startW, startH)
  


class mwFullPipe_KSA:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required":
                    {"mwFullPipe": ("MWFULLPIPE",),
                    "ltntPipe": ("MWLTNTPIPE", ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "stepStart": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "stepEnd": ("INT", {"default": 20, "min": 0, "max": 10000}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "seed_deltaVsOrig": ("INT", {"default": 0, "min": 0, "max": 10000, "defaultBehavior": "input"}),
                    "add_noise": (["enable", "disable"], ),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "multOver8": ("INT", {"default": 8, "min": 0, "max": 64, "step":1}),
                    "upscale_method": (moh.upscale_methods,),
                     },

                }
    RETURN_TYPES = ("MWFULLPIPE", "MWLTNTPIPE", "CONDITIONING", "CONDITIONING", "INT", "INT", "LATENT", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("fullPipe", "ltntPipe", "positive", "negative", "stepEnd", "steps", "latent", "image", "wd", "ht")
    FUNCTION = "mwFPKSA"
    CATEGORY = "mohwag"

    def mwFPKSA(self, mwFullPipe, ltntPipe, positive, negative, cfg, stepStart, stepEnd, steps, seed_deltaVsOrig, add_noise, return_with_leftover_noise, multOver8, upscale_method, denoise=1.0):
        
        mwCkpt1, seedOrig, sampler_name, scheduler = mwFullPipe
        model1, _, _ = mwCkpt1
        seed = seedOrig + seed_deltaVsOrig
        vaeAlways = mwCkpt1[2]

        inputLtnt, startWInit, startHInit, inputWInit, inputHInit = ltntPipe

        startW = int(startWInit)
        startH = int(startHInit)
        inputW = int(inputWInit)
        inputH = int(inputHInit)

        outputW = int(startW * multOver8 / 8)
        outputH = int(startH * multOver8 / 8)

        #run input latent upscale
        if outputW == inputW and outputH == inputH:
            scaleLtnt = inputLtnt
            scaleLtntOut = (scaleLtnt,)
        else:
            crop = "disabled"
            scaleLtnt = inputLtnt.copy()

            scaleLtnt["samples"] = comfy.utils.common_upscale(inputLtnt["samples"], outputW // 8, outputH // 8, upscale_method, crop)
            scaleLtntOut = (scaleLtnt,)


        #diffusion run
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        runLtnt = common_ksampler(model1, seed, steps, cfg, sampler_name, scheduler, positive, negative, scaleLtntOut[0], denoise=denoise, disable_noise=disable_noise, start_step=stepStart, last_step=stepEnd, force_full_denoise=force_full_denoise)
        runLtntOut = (runLtnt, )

        img = vaeAlways.decode(runLtnt["samples"])
        imgOut = (img, )

        ltntPipe = runLtntOut[0], startW, startH, outputW, outputH

        return (mwFullPipe, ltntPipe, positive, negative, stepEnd, steps, runLtntOut[0], imgOut[0], outputW, outputH)
    


class mwModelBranch1:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "mwMBO"
    CATEGORY = "mohwag"

    def mwMBO(self, mwFullPipe):
        mwCkpt1, _, _, _ = mwFullPipe
        model1, _, _ = mwCkpt1
        return (model1,)

class mwModelBranch2:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }

    RETURN_TYPES = ("MWFULLPIPE", "MODEL",)
    RETURN_NAMES = ("mwFullPipe", "model",)
    FUNCTION = "mwMBT"
    CATEGORY = "mohwag"

    def mwMBT(self, mwFullPipe):
        mwCkpt1, _, _, _ = mwFullPipe
        model1, _, _ = mwCkpt1
        return (mwFullPipe, model1)



class mwClipBranch1:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "mwCBO"
    CATEGORY = "mohwag"

    def mwCBO(self, mwFullPipe):
        mwCkpt1, _, _, _ = mwFullPipe
        _, clip1, _ = mwCkpt1
        return (clip1,)

class mwClipBranch2:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }

    RETURN_TYPES = ("MWFULLPIPE", "CLIP",)
    RETURN_NAMES = ("mwFullPipe", "clip",)
    FUNCTION = "mwCBT"
    CATEGORY = "mohwag"

    def mwCBT(self, mwFullPipe):
        mwCkpt1, _, _, _ = mwFullPipe
        _, clip1, _ = mwCkpt1
        return (mwFullPipe, clip1)



class mwVaeBranch1:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "mwVBO"
    CATEGORY = "mohwag"

    def mwVBO(self, mwFullPipe):
        mwCkpt1, _, _, _ = mwFullPipe
        _, _, vae1 = mwCkpt1
        return (vae1,)

class mwVaeBranch2:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }

    RETURN_TYPES = ("MWFULLPIPE", "VAE",)
    RETURN_NAMES = ("mwFullPipe", "vae",)
    FUNCTION = "mwVBT"
    CATEGORY = "mohwag"

    def mwVBT(self, mwFullPipe):
        mwCkpt1, _, _, _ = mwFullPipe
        _, _, vae1 = mwCkpt1
        return (mwFullPipe, vae1)



class mwCkptBranch1:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "mwCPBO"
    CATEGORY = "mohwag"

    def mwCPBO(self, mwFullPipe):
        mwCkpt1, _, _, _ = mwFullPipe
        model1, clip1, vaeAlways = mwCkpt1
        return (model1, clip1, vaeAlways)

class mwCkptBranch2:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }

    RETURN_TYPES = ("MWFULLPIPE", "MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("mwFullPipe", "model", "clip", "vae")
    FUNCTION = "mwCPBT"
    CATEGORY = "mohwag"

    def mwCPBT(self, mwFullPipe):
        mwCkpt1, _, _, _ = mwFullPipe
        model1, clip1, vaeAlways = mwCkpt1
        return (mwFullPipe, model1, clip1, vaeAlways)
    


class mwSchedEdit:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",),
                            "sampler_name": (comfy.samplers.SAMPLER_NAMES, ),
                            "scheduler": (comfy.samplers.SCHEDULER_NAMES, ),
                            }}
    RETURN_TYPES = ("MWFULLPIPE",)
    RETURN_NAMES = ("mwFullPipe",)
    FUNCTION = "mwSE"
    CATEGORY = "mohwag"

    def mwSE(self, mwFullPipe, sampler_name, scheduler):
        mwCkpt1, seed1, _, _ = mwFullPipe
        out = mwCkpt1, seed1, sampler_name, scheduler
        return (out,)


class mwFullPipeBranch1:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", comfy.samplers.SAMPLER_NAMES, comfy.samplers.SCHEDULER_NAMES)
    RETURN_NAMES = ("model", "clip", "vae", "sampler_name", "scheduler")
    FUNCTION = "mwFPBO"
    CATEGORY = "mohwag"

    def mwFPBO(self, mwFullPipe):
        mwCkpt1, seed1, sampler_name1, scheduler1 = mwFullPipe
        model1, clip1, vaeAlways = mwCkpt1
        return (model1, clip1, vaeAlways, sampler_name1, scheduler1)

class mwText:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "txtInOut"
    CATEGORY = "mohwag"

    def txtInOut(self, text):
        return (text,)




NODE_CLASS_MAPPINGS = {
"mwFullPipe_Load": mwFullPipe_Load,
"mwCkpt_Load": mwCkpt_Load,
"mwFullPipe_ckptMerge": mwFullPipe_ckptMerge,
"mwLora_Load": mwLora_Load,
"mwFullPipe_loraMerge": mwFullPipe_loraMerge,
"mwLtntPipe_Create": mwLtntPipe_Create,
"mwLtntPipe_CropScale": mwLtntPipe_CropScale,
"mwLtntPipeBranch1": mwLtntPipeBranch1,
"mwLtntPipe_View": mwLtntPipe_View,
"mwCond": mwCond,
"mwCondPrep": mwCondPrep,
"mwFullPipe_KSAStart": mwFullPipe_KSAStart,
"mwFullPipe_KSA": mwFullPipe_KSA,
"mwModelBranch1": mwModelBranch1,
"mwModelBranch2": mwModelBranch2,
"mwClipBranch1": mwClipBranch1,
"mwClipBranch2": mwClipBranch2,
"mwVaeBranch1": mwVaeBranch1,
"mwVaeBranch2": mwVaeBranch2,
"mwCkptBranch1": mwCkptBranch1,
"mwCkptBranch2": mwCkptBranch2,
"mwSchedEdit": mwSchedEdit,
"mwFullPipeBranch1": mwFullPipeBranch1,
"mwText": mwText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
"mwFullPipe_Load": "mwFullPipe_Load",
"mwCkpt_Load": "mwCkpt_Load",
"mwFullPipe_ckptMerge": "mwFullPipe_ckptMerge",
"mwLora_Load": "mwLora_Load",
"mwFullPipe_loraMerge": "mwFullPipe_loraMerge",
"mwLtntPipe_Create": "mwLtntPipe_Create",
"mwLtntPipe_CropScale": "mwLtntPipe_CropScale",
"mwLtntPipeBranch1": "mwLtntPipeBranch1",
"mwLtntPipe_View": "mwLtntPipe_View",
"mwCond": "mwCond",
"mwCondPrep": "mwCondPrep",
"mwFullPipe_KSAStart": "mwFullPipe_KSAStart",
"mwFullPipe_KSA": "mwFullPipe_KSA",
"mwModelBranch1": "mwModelBranch1",
"mwModelBranch2": "mwModelBranch2",
"mwClipBranch1": "mwClipBranch1",
"mwClipBranch2": "mwClipBranch2",
"mwVaeBranch1": "mwVaeBranch1",
"mwVaeBranch2": "mwVaeBranch2",
"mwCkptBranch1": "mwCkptBranch1",
"mwCkptBranch2": "mwCkptBranch2",
"mwSchedEdit": "mwSchedEdit",
"mwFullPipeBranch1": "mwFullPipeBranch1",
"mwText": "mwText",
}