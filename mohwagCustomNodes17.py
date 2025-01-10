import torch

import os
import sys
import json
import hashlib
import traceback
import math
import time
import random

from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageSequence
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

from torchvision.transforms import functional as TF
from pathlib import Path

#pulling in funcs
from nodes import MAX_RESOLUTION, ConditioningSetArea, CheckpointLoaderSimple, common_ksampler
condSetAreaMethod = ConditioningSetArea().append

from comfy_extras.nodes_model_merging import ModelMergeSimple, CLIPMergeSimple
modelMergeMethod = ModelMergeSimple().merge
clipMergeMethod = CLIPMergeSimple().merge
ckptLoaderSimpleMethod = CheckpointLoaderSimple().load_checkpoint



class mwBatch:# Zuellni
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "input_dir": ("STRING",{"default": folder_paths.get_input_directory()}),
                #"squareSize": ("INT", {"default": 224, "min": 0, "max": MAX_RESOLUTION, "step": 8 }),
                "squareSize": ("INT", {"default": 224, "min": 0, "max": 10000, "step": 8}),
            },
        }

    CATEGORY = "mohwag/mod3P"
    FUNCTION = "MWB"
    RETURN_NAMES = ("IMAGES", "MASKS")
    RETURN_TYPES = ("IMAGE", "MASK")

    def MWB(self, input_dir, squareSize):
   
        input_dir = Path(input_dir)
        files = []

        for file in ["bmp", "gif", "jpeg", "jpg", "png", "webp"]:
            files.extend(input_dir.glob(f"*.{file}"))

        if not files:
            raise comfy.model_management.InterruptProcessingException()

        pil_images = []

        for file in files:
            image = Image.open(file)

            if getattr(image, "is_animated", True):
                for frame in ImageSequence.Iterator(image):
                    pil_images.append(frame.copy().convert("RGBA"))
            else:
                pil_images.append(image.convert("RGBA"))

        images = []

        for image in pil_images:
            image = TF.to_tensor(image)
            image[:3, image[3, :, :] == 0] = 0
            images.append(image)

        #min_height = min([i.shape[1] for i in images])
        #min_width = min([i.shape[2] for i in images])
        #min_dim = min(min_height, min_width) // 8 * 8
        images = [TF.resize(i, squareSize) for i in images]

        #min_height = min([i.shape[1] for i in images]) // 8 * 8
        #min_width = min([i.shape[2] for i in images]) // 8 * 8
        images = [TF.center_crop(i, (squareSize, squareSize)) for i in images]

        images = torch.stack(images)
        images = images.permute(0, 2, 3, 1)
        masks = images[:, :, :, 3]
        images = images[:, :, :, :3]
    
        return (images, masks)

class mwpsConditioningSetArea(ConditioningSetArea):
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
                    "cond": ("CONDITIONING",),
                     "psPipe": ("MWPOSSIZEPIPE",),
                     "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                     }}
    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("cond", )
    FUNCTION = "mwCSA"
    CATEGORY = "mohwag/modCUI"

    def mwCSA(self, cond, psPipe, strength):
        x, y, w, h = psPipe
        outCond = condSetAreaMethod(cond, w, h, x, y, strength)
        return (outCond[0],)


class mwFullPipe_Load:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
                     "mwCkpt": ("MWCKPT",{"forceInput": True}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "sampler_name": (comfy.samplers.SAMPLER_NAMES, ),
                     "scheduler": (comfy.samplers.SCHEDULER_NAMES, ),
                     }}
    RETURN_TYPES = ("MWFULLPIPE", )
    RETURN_NAMES = ("mwFullPipe", )
    FUNCTION = "mwFPL"
    CATEGORY = "mohwag/FullPipe_BuildLoadMerge"

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

    CATEGORY = "mohwag/FullPipe_BuildLoadMerge"

    def mwCPL(self, ckpt_name):
        rslt = ckptLoaderSimpleMethod(ckpt_name)
        return (rslt,)
    
class mwCkpt_modelEdit: #comfy nodes.py LoraLoader
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": { 
                              "mwCkpt": ("MWCKPT", ),
                              "model": ("MODEL",)
                              }}
    RETURN_TYPES = ("MWCKPT",)
    RETURN_NAMES = ("mwCkpt",)
    FUNCTION = "mwCME"

    CATEGORY = "mohwag/FullPipe_BuildLoadMerge"

    def mwCME(self, mwCkpt, model):
        _, clip1, vaeAlways = mwCkpt
        mwCkptNew = model, clip1, vaeAlways
        return (mwCkptNew,)


class mwCkpt_ckptMerge:
    ckptSelect = ["base", "current"]

    @classmethod
    def INPUT_TYPES(moh):
        return {"required": { "mwCkpt1": ("MWCKPT",),
                              "mwCkpt2": ("MWCKPT",),
                              "wgt_Ckpt1": ("FLOAT", {"default": 0, "min": -1.0, "max": 2.0, "step": 0.04}),

                              }}
    RETURN_TYPES = ("MWCKPT",)
    RETURN_NAMES = ("mwCkpt",)
    FUNCTION = "mwCCM"
    CATEGORY = "mohwag/FullPipe_BuildLoadMerge"

    def mwCCM(self, mwCkpt1, mwCkpt2, wgt_Ckpt1):

        if wgt_Ckpt1 == 1:
            return (mwCkpt1,)
        
        vaeAlways = mwCkpt1[2]
        amodel2, aclip2, _ = mwCkpt2[:3]

        if wgt_Ckpt1 == 0:
            mwCkptNew = (amodel2, aclip2, vaeAlways)
            return (mwCkptNew,)

        amodel1, aclip1, _ = mwCkpt1[:3]

        modelResult = modelMergeMethod(amodel1, amodel2, wgt_Ckpt1)[0]
        clipResult = clipMergeMethod(aclip1, aclip2, wgt_Ckpt1)[0]

        mwCkptNew = (modelResult, clipResult, vaeAlways)

        return (mwCkptNew,)




class mwFullPipe_ckptMerge:
    ckptSelect = ["base", "current"]

    @classmethod
    def INPUT_TYPES(moh):
        return {"required": { "mwFullPipe": ("MWFULLPIPE",),
                              "mwCkpt": ("MWCKPT",),
                              "wgtFullPipeCkpt": ("FLOAT", {"default": 0, "min": -1.0, "max": 2.0, "step": 0.04}),

                              }}
    RETURN_TYPES = ("MWFULLPIPE",)
    RETURN_NAMES = ("mwFullPipe",)
    FUNCTION = "mwFPCPM"
    CATEGORY = "mohwag/FullPipe_BuildLoadMerge"

    def mwFPCPM(self, mwFullPipe, mwCkpt, wgtFullPipeCkpt):

        if wgtFullPipeCkpt == 1:
            return (mwFullPipe,)

        mwCkpt1, seed, sampler_name, scheduler = mwFullPipe
        vaeAlways = mwCkpt1[2]

        amodel2, aclip2, _ = mwCkpt[:3]
        
        if wgtFullPipeCkpt == 0:
            mwCkptNew = (amodel2, aclip2, vaeAlways)
            mwFullPipeNew = mwCkptNew, seed, sampler_name, scheduler
            return (mwFullPipeNew,)
        
        amodel1, aclip1, _ = mwCkpt1

        modelResult = modelMergeMethod(amodel1, amodel2, wgtFullPipeCkpt)[0]
        clipResult = clipMergeMethod(aclip1, aclip2, wgtFullPipeCkpt)[0]

        mwCkptNew = (modelResult, clipResult, vaeAlways)
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

    CATEGORY = "mohwag/FullPipe_BuildLoadMerge"

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
    CATEGORY = "mohwag/FullPipe_BuildLoadMerge"

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
           # "initialW": ("INT", {"default": 512, "min": 64, "max": 3072, "step": 64 }),
            #"initialH": ("INT", {"default": 512, "min": 64, "max": 3072, "step": 64 }),
            "initialS": ("MWSIZEPIPE",)
            }}

    RETURN_TYPES = ("MWLTNTPIPE",)
    RETURN_NAMES = ("ltntPipe",)
    FUNCTION = "mwLPC"
    CATEGORY = "mohwag/LatentPipe"

    def mwLPC(self, ltnt, initialS):

        initialW, initialH = initialS

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
            "cropLft_units": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
            "cropRgt_units": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
            "TotDeltaW_units": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 4}),
            "cropTop_units": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
            "cropBtm_units": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
            "TotDeltaH_units": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 4}),
            "cropToOutput_upscaleMethod": (moh.upscale_methods,),
            }}

    #RETURN_TYPES = ("MWLTNTPIPE", "INT", "INT", "INT", "INT")
    #RETURN_NAMES = ("ltntPipe", "wd", "ht", "x0", "y0")
    RETURN_TYPES = ("MWLTNTPIPE", "MWSIZEPIPE", "MWPOSSIZEPIPE", "MWSIZEPIPE")
    RETURN_NAMES = ("ltntPipe", "sPipe", "psPipe", "new_sPipe_start")
    FUNCTION = "mwLPCS"
    CATEGORY = "mohwag/LatentPipe"

    def mwLPCS(self, ltntPipe, cropLft_units, cropRgt_units, TotDeltaW_units, cropTop_units, cropBtm_units, TotDeltaH_units, cropToOutput_upscaleMethod):

        ltnt, startW, startH, inputW, inputH = ltntPipe

        unitVal = 8 * inputW / startW

        cropLft = int(unitVal * cropLft_units)
        cropRgt = int(unitVal * cropRgt_units)
        TotDeltaW = int(unitVal * TotDeltaW_units)
        cropTop = int(unitVal * cropTop_units)
        cropBtm = int(unitVal * cropBtm_units)
        TotDeltaH = int(unitVal * TotDeltaH_units)

        outputW = int(inputW - TotDeltaW)
        outputH = int(inputH - TotDeltaH)

        outputS = outputW, outputH
        outputPS = cropLft, cropTop, outputW, outputH


        if (cropLft == 0 and cropRgt == 0) and (cropTop == 0 and cropBtm == 0):
            if TotDeltaW == 0 and TotDeltaH == 0:
                return (ltntPipe, outputS, outputPS)
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

        if cropLft + cropRgt == TotDeltaW and cropTop + cropBtm == TotDeltaH:
            finalLtnt = cropdLtnt
        else:
            crop = "disabled"
            upscaleLtnt = cropdLtnt.copy()
            upscaleLtnt["samples"] = comfy.utils.common_upscale(cropdLtnt["samples"], outputW // 8, outputH  // 8, cropToOutput_upscaleMethod, crop)
            finalLtnt = upscaleLtnt

        newStartW = int(startW - (8 * TotDeltaW_units))
        newStartH = int(startH - (8 * TotDeltaH_units))

        ltntPipeNew = finalLtnt, newStartW, newStartH, outputW, outputH
        newSPipeStart = newStartW, newStartH

        return (ltntPipeNew, outputS, outputPS, newSPipeStart)



class mwLtntPipeBranch1: #WAS_Latent_Size_To_Number
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "ltntPipe": ("MWLTNTPIPE", ),
            }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "mwLPBO"
    CATEGORY = "mohwag/LatentPipe"

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

    RETURN_TYPES = ("IMAGE", "MWLTNTPIPE")
    RETURN_NAMES = ("img", "ltntpipe")
    FUNCTION = "mwLPV"
    CATEGORY = "mohwag/LatentPipe"

    def mwLPV(self, mwFullPipe, ltntPipe):

        ltnt, _, _, _, _ = ltntPipe

        mwCkpt1, _, _, _ = mwFullPipe
        vaeAlways = mwCkpt1[2]
    
        img = vaeAlways.decode(ltnt["samples"])
        imgOut = img

        return (imgOut, ltntPipe)



def cuString(aString:str) -> str:
    return aString.strip().replace("    ", " ",-1).replace("   ", " ",-1).replace("  ", " ",-1).replace("  ", " ",-1)

def prepString(aBlock:str) -> list[str]:
    aList = aBlock.splitlines()
    while("" in aList):
        aList.remove("")

    condListL = []
    for x in aList:
        condListL.append(x.split("||")[0])
    condListL = map(cuString, condListL)
    condListL = list(condListL)

    condListR = []
    for x in aList:
        condListR.append(x.split("||")[-1])
    condListR = map(cuString, condListR)
    condListR = list(condListR)

    return (condListL, condListR)


def func_mwCondPrep(aclip, amwCond):

    acompType, acondListF = amwCond
    acondListL, acondListR = acondListF
    textConcF = ", ".join(map(str,acondListL))

    if acompType != "textConcat":
        condInc = []
        for aString in acondListR:
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
    if acompType == "condConcat":
        return  (concCondOut,)

    if acompType != "condConcat":
        tokens = aclip.tokenize(textConcF)
        cond, pooled = aclip.encode_from_tokens(tokens, return_pooled=True)
        textCondOut = [[cond, {"pooled_output": pooled}]]
    if acompType == "textConcat":
        return (textCondOut,)

    bothCondOut = (textCondOut + concCondOut,)
    return bothCondOut


def func_mwCondPrepXL(aclip, amwCond, awidth, aheight, acrop_w, acrop_h, atarget_width, atarget_height):

    acompType, acondListF = amwCond
    acondListL, acondListR = acondListF
    textConcF = ", ".join(map(str,acondListL))

    if acompType != "textConcat":
        condInc = []
        for aString in acondListR:
            tokens = aclip.tokenize(aString)
            tokens["l"] = aclip.tokenize(aString)["l"]
            if len(tokens["l"]) != len(tokens["g"]):
                empty = aclip.tokenize("")
                while len(tokens["l"]) < len(tokens["g"]):
                    tokens["l"] += empty["l"]
                while len(tokens["l"]) > len(tokens["g"]):
                    tokens["g"] += empty["g"]
            cond, pooled = aclip.encode_from_tokens(tokens, return_pooled=True)
            condAdd = [[cond, {"pooled_output": pooled, "width": awidth, "height": aheight, "crop_w": acrop_w, "crop_h": acrop_h, "target_width": atarget_width, "target_height": atarget_height}]]               
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

    if acompType == "condConcat":
        return  (concCondOut,)
    
    if acompType != "condConcat":
        tokens = aclip.tokenize(textConcF)
        tokens["l"] = aclip.tokenize(textConcF)["l"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = aclip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        cond, pooled = aclip.encode_from_tokens(tokens, return_pooled=True)
        textCondOut = [[cond, {"pooled_output": pooled, "width": awidth, "height": aheight, "crop_w": acrop_w, "crop_h": acrop_h, "target_width": atarget_width, "target_height": atarget_height}]]   

    if acompType == "textConcat":
        return (textCondOut,)

    bothCondOut = (textCondOut + concCondOut,)
    return bothCondOut


class mwCond:
    compositionTypes = ["textConcat", "condConcat", "combBoth"]

    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "clip": ("CLIP", ),
            "compType": (moh.compositionTypes, {"default": "textConcat"}),
            "condText": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "mwC"
    CATEGORY = "mohwag/Conditioning"

    def mwC(self, clip, compType, condText):
        condListF = prepString(condText)
        condReturn = func_mwCondPrep(clip, (compType, condListF))

        return condReturn


class mwCondXL:
    compositionTypes = ["textConcat", "condConcat", "combBoth"]

    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "clip": ("CLIP", ),
            "compType": (moh.compositionTypes, {"default": "textConcat"}),
            "condText": ("STRING", {"multiline": True}),
            "width": ("INT", {"default": 3072, "min": 0, "max": MAX_RESOLUTION, }),
            "height": ("INT", {"default": 3072, "min": 0, "max": MAX_RESOLUTION, }),
            "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, }),
            "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, }),
            "target_width": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION, "step":64}),
            "target_height": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION, "step":64}),            
            }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "mwCXL"
    CATEGORY = "mohwag/Conditioning"

    def mwCXL(self, clip, compType, condText, width, height, crop_w, crop_h, target_width, target_height):
        condListF = prepString(condText)
        condReturn = func_mwCondPrepXL(clip, (compType, condListF), width, height, crop_w, crop_h, target_width, target_height)
        return condReturn


class mwCondXLa:
    compositionTypes = ["textConcat", "condConcat", "combBoth"]

    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "clip": ("CLIP", ),
            "compType": (moh.compositionTypes, {"default": "textConcat"}),
            "refImgsPixelMult": ("FLOAT", {"default": 0.75, "min": 0.25, "max": 7.75, "step": 0.25}),  
            "condText": ("STRING", {"multiline": True}),
            #"actualW": ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, }),
            #"actualH": ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, }),  
            "sPipe_actual": ("MWSIZEPIPE",)
            }}

    RETURN_TYPES = ("CONDITIONING", "CLIP", "MWSIZEPIPE", "STRING")
    RETURN_NAMES = ("conditioning", "clip", "sPipe_actual", "condText")
    FUNCTION = "mwCXLa"
    CATEGORY = "mohwag/Conditioning"

    def mwCXLa(self, clip, compType, refImgsPixelMult, condText, sPipe_actual):
        condListF = prepString(condText)

        aspectRatList = [0.25, 0.26, 0.27, 0.28, 0.32, 0.33, 0.35, 0.4, 0.42, 0.48, 0.5, 0.52, 0.57, 0.6, 0.68, 0.72, 0.78, 0.82, 0.88, 0.94, 1, 1.07, 1.13, 1.21, 1.29, 1.38, 1.46, 1.67, 1.75, 2, 2.09, 2.4, 2.5, 2.89, 3, 3.11, 3.63, 3.75, 3.88, 4]
        targetW_list = [512, 512, 512, 512, 576, 576, 576, 640, 640, 704, 704, 704, 768, 768, 832, 832, 896, 896, 960, 960, 1024, 1024, 1088, 1088, 1152, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048]
        targetH_list = [2048, 1984, 1920, 1856, 1792, 1728, 1664, 1600, 1536, 1472, 1408, 1344, 1344, 1280, 1216, 1152, 1152, 1088, 1088, 1024, 1024, 960, 960, 896, 896, 832, 832, 768, 768, 704, 704, 640, 640, 576, 576, 576, 512, 512, 512, 512]
        #targetWoverH_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 128, 192, 256, 320, 384, 512, 576, 704, 768, 896, 960, 1088, 1152, 1216, 1344, 1408, 1472, 1536]
        #targetHoverW_list = [1536, 1472, 1408, 1344, 1216, 1152, 1088, 960, 896, 768, 704, 640, 576, 512, 384, 320, 256, 192, 128, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        actualW, actualH = sPipe_actual

        #GET START IMAGE SIZE
        ##start ratio and size before upscaling
        rats = round((np.exp(1))**(np.log(actualW / actualH) * (2/3)),2)
        testListrs = [abs(round(x - rats,2)) for x in aspectRatList]
        testMinrs = min(testListrs)
        testMinLocrs = testListrs.index(testMinrs)
        startW = targetW_list[testMinLocrs]
        startH = targetH_list[testMinLocrs]

        ##start ratio and size final (i.e. w/ upscaling)
        actualPx = actualW * actualH
        refImgsPx = actualPx * refImgsPixelMult
        refImgRat = startW / startH
        refImgsH = (refImgsPx / refImgRat)**(1/2)
        refImgsW = refImgsPx / refImgsH
        start_width = int(round(refImgsW,0))
        start_height = int(round(refImgsH,0))

        #GET TARGET SIZE AND CROP
        ##target size with ratio that matches actual size ratio
        targetW = 512
        targetH = 512
        
        if min(actualW, actualH) > 255:
            rat = round(actualW / actualH, 2)
            testListr = [abs(round(x - rat,2)) for x in aspectRatList]
            testMinr = min(testListr)
            testMinLocr = testListr.index(testMinr)
            targetW = targetW_list[testMinLocr]
            targetH = targetH_list[testMinLocr]


        ## expected crop, f(startS ratio, targetS ratio, targetS)
        if targetW < targetH:
            cropW = int(max(64, (((targetH / startH) * startW) - targetW) /2))
            cropH = int(64)
        else:
            cropH = int(max(64, (((targetW / startW) * startH) - targetH) /2))
            cropW = int(64)           


        condReturn = func_mwCondPrepXL(clip, (compType, condListF), start_width, start_height, cropW, cropH, int(targetW), int(targetH))

        return (condReturn[0], clip, sPipe_actual, condText)


'''
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
'''


class mwFullPipe_KSAStart:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
    @classmethod
    def INPUT_TYPES(moh):
        return {"required":
                    {"fullPipe": ("MWFULLPIPE",),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg": ("FLOAT", {"default": 8.0, "min": -100.0, "max": 100.0, "step":0.5, "round": 0.01}),
                    "stepEnd": ("INT", {"default": 48, "min": 0, "max": 10000}),
                    "steps": ("INT", {"default": 48, "min": 1, "max": 10000}),
                    "seed_deltaVsOrig": ("INT", {"default": 0, "min": 0, "max": 10000, "defaultBehavior": "input"}),
                    "add_noise": (["enable", "disable"], ),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    #"startW": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32 }),
                    #"startH": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32 }),
                    "startS": ("MWSIZEPIPE",),
                    #"nextMO8": ("INT", {"default": 8, "min": 0, "max": 64, "step":1}),
                    },
                }

    RETURN_TYPES = ("MWFULLPIPE", "MWLTNTPIPE", "CONDITIONING", "CONDITIONING", "INT", "INT", "LATENT", "IMAGE", "MWSIZEPIPE")
    RETURN_NAMES = ("fullPipe", "ltntPipe", "positive", "negative", "stepEnd", "steps", "latent", "image", "sPipe")
    FUNCTION = "mwFPKSAS"
    CATEGORY = "mohwag/Sampling"

    def mwFPKSAS(self, fullPipe, positive, negative, cfg, stepEnd, steps, seed_deltaVsOrig, add_noise, return_with_leftover_noise, startS, denoise=1.0):
        stepStart = 0
        
        startW, startH = startS

        mwCkpt1, seedOrig, sampler_name, scheduler = fullPipe
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
        #ltnt = common_ksampler(model1, seed, steps, cfg, sampler_name, scheduler, positive, negative, latentDictTuple[0], denoise=denoise, disable_noise=disable_noise, start_step=stepStart, last_step=stepEnd, force_full_denoise=force_full_denoise)
        ltnt = common_ksampler(model1, seed, steps, cfg, sampler_name, scheduler, positive, negative, latentDictTuple[0], denoise=denoise, disable_noise=disable_noise, start_step=stepStart, last_step=stepEnd, force_full_denoise=force_full_denoise)[0]
        ltntOut = (ltnt, )
        
        img = vaeAlways.decode(ltnt["samples"])
        imgOut = (img, )

        ltntPipe = ltntOut[0], startW, startH, startW, startH

        #nextW = int(startW * nextMO8 / 8)
        #nextH = int(startH * nextMO8 / 8)

        startS = startW, startH

        return (fullPipe, ltntPipe, positive, negative, stepEnd, steps, ltntOut[0], imgOut[0], startS)
  


class mwFullPipe_KSA:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required":
                    {"fullPipe": ("MWFULLPIPE",),
                    "ltntPipe": ("MWLTNTPIPE", ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg": ("FLOAT", {"default": 8.0, "min": -100.0, "max": 100.0, "step":0.5, "round": 0.01}),
                    "stepStart": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "stepEnd": ("INT", {"default": 20, "min": 0, "max": 10000}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "seed_deltaVsOrig": ("INT", {"default": 0, "min": 0, "max": 10000, "defaultBehavior": "input"}),
                    "add_noise": (["enable", "disable"], ),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "multOver8": ("INT", {"default": 8, "min": 0, "max": 64, "step":1}),
                    #"nextMO8": ("INT", {"default": 8, "min": 0, "max": 64, "step":1}),
                    "upscale_method": (moh.upscale_methods,),
                     },

                }
    RETURN_TYPES = ("MWFULLPIPE", "MWLTNTPIPE", "CONDITIONING", "CONDITIONING", "INT", "INT", "LATENT", "IMAGE", "MWSIZEPIPE")
    RETURN_NAMES = ("fullPipe", "ltntPipe", "positive", "negative", "stepEnd", "steps", "latent", "image", "sPipe")
    FUNCTION = "mwFPKSA"
    CATEGORY = "mohwag/Sampling"

    def mwFPKSA(self, fullPipe, ltntPipe, positive, negative, cfg, stepStart, stepEnd, steps, seed_deltaVsOrig, add_noise, return_with_leftover_noise, multOver8, upscale_method, denoise=1.0):
        
        mwCkpt1, seedOrig, sampler_name, scheduler = fullPipe
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
        #runLtnt = common_ksampler(model1, seed, steps, cfg, sampler_name, scheduler, positive, negative, scaleLtntOut[0], denoise=denoise, disable_noise=disable_noise, start_step=stepStart, last_step=stepEnd, force_full_denoise=force_full_denoise)
        runLtnt = common_ksampler(model1, seed, steps, cfg, sampler_name, scheduler, positive, negative, scaleLtntOut[0], denoise=denoise, disable_noise=disable_noise, start_step=stepStart, last_step=stepEnd, force_full_denoise=force_full_denoise)[0]
        runLtntOut = (runLtnt, )

        img = vaeAlways.decode(runLtnt["samples"])
        imgOut = (img, )

        ltntPipe = runLtntOut[0], startW, startH, outputW, outputH

        #nextW = int(startW * nextMO8 / 8)
        #nextH = int(startH * nextMO8 / 8)

        outputS = outputW, outputH
        return (fullPipe, ltntPipe, positive, negative, stepEnd, steps, runLtntOut[0], imgOut[0], outputS)
    


class mwModelBranch1:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "mwMBO"
    CATEGORY = "mohwag/FullPipeBranch1"

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
    CATEGORY = "mohwag/FullPipeBranch2"

    def mwMBT(self, mwFullPipe):
        mwCkpt1, _, _, _ = mwFullPipe
        model1, _, _ = mwCkpt1
        return (mwFullPipe, model1)


class mwModelBranch1_ckpt:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwCkpt": ("MWCKPT",), }, }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "mwMBOC"
    CATEGORY = "mohwag/FullPipeBranch1"

    def mwMBOC(self, mwCkpt):
        model1, _, _ = mwCkpt
        return (model1,)
'''
class mwModelBranch2_ckpt:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwCkpt": ("MWCKPT",), }, }

    RETURN_TYPES = ("MWFULLPIPE", "MODEL",)
    RETURN_NAMES = ("mwFullPipe", "model",)
    FUNCTION = "mwMBTC"
    CATEGORY = "mohwag/FullPipeBranch2"

    def mwMBTC(self, mwFullPipe):
        mwCkpt1, _, _, _ = mwFullPipe
        model1, _, _ = mwCkpt1
        return (mwFullPipe, model1)
'''

class mwClipBranch1:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "mwCBO"
    CATEGORY = "mohwag/FullPipeBranch1"

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
    CATEGORY = "mohwag/FullPipeBranch2"

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
    CATEGORY = "mohwag/FullPipeBranch1"

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
    CATEGORY = "mohwag/FullPipeBranch2"

    def mwVBT(self, mwFullPipe):
        mwCkpt1, _, _, _ = mwFullPipe
        _, _, vae1 = mwCkpt1
        return (mwFullPipe, vae1)



class mwCkptBranch1:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }

    RETURN_TYPES = ("MWCKPT", "MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("mwCkpt", "model", "clip", "vae")
    FUNCTION = "mwCPBO"
    CATEGORY = "mohwag/FullPipeBranch1"

    def mwCPBO(self, mwFullPipe):
        mwCkpt1, _, _, _ = mwFullPipe
        model1, clip1, vaeAlways = mwCkpt1
        return (mwCkpt1, model1, clip1, vaeAlways)

class mwCkptBranch2:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }

    RETURN_TYPES = ("MWFULLPIPE", "MWCKPT", "MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("mwFullPipe", "mwCkpt", "model", "clip", "vae")
    FUNCTION = "mwCPBT"
    CATEGORY = "mohwag/FullPipeBranch2"

    def mwCPBT(self, mwFullPipe):
        mwCkpt1, _, _, _ = mwFullPipe
        model1, clip1, vaeAlways = mwCkpt1
        return (mwFullPipe, mwCkpt1, model1, clip1, vaeAlways)
    

class mwSeedBranch1:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "mwSBO"
    CATEGORY = "mohwag/FullPipeBranch1"

    def mwSBO(self, mwFullPipe):
        _, seed1, _, _ = mwFullPipe
        return (seed1,)

class mwSeedBranch2:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }

    RETURN_TYPES = ("MWFULLPIPE", "INT")
    RETURN_NAMES = ("mwFullPipe", "seed")
    FUNCTION = "mwSBT"
    CATEGORY = "mohwag/FullPipeBranch2"

    def mwSBT(self, mwFullPipe):
        _, seed1, _, _ = mwFullPipe
        return (mwFullPipe, seed1)


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
    CATEGORY = "mohwag/FullPipe_BuildLoadMerge"

    def mwSE(self, mwFullPipe, sampler_name, scheduler):
        mwCkpt1, seed1, _, _ = mwFullPipe
        out = mwCkpt1, seed1, sampler_name, scheduler
        return (out,)


class mwModelEdit:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",),
                            "model": ("MODEL",),
                            }}
    RETURN_TYPES = ("MWFULLPIPE",)
    RETURN_NAMES = ("mwFullPipe",)
    FUNCTION = "mwME"
    CATEGORY = "mohwag/FullPipe_BuildLoadMerge"

    def mwME(self, mwFullPipe, model):
        mwCkpt1, seed, sampler_name, scheduler = mwFullPipe
        _, clip1, vaeAlways = mwCkpt1
        mwCkptNew = model, clip1, vaeAlways
        mwFullPipeNew = mwCkptNew, seed, sampler_name, scheduler
        return (mwFullPipeNew,)


class mwFullPipeBranch1:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", comfy.samplers.SAMPLER_NAMES, comfy.samplers.SCHEDULER_NAMES)
    RETURN_NAMES = ("model", "seed", "clip", "vae", "sampler_name", "scheduler")
    FUNCTION = "mwFPBO"
    CATEGORY = "mohwag/FullPipeBranch1"

    def mwFPBO(self, mwFullPipe):
        mwCkpt1, seed1, sampler_name1, scheduler1 = mwFullPipe
        model1, clip1, vaeAlways = mwCkpt1
        return (model1, seed1, clip1, vaeAlways, sampler_name1, scheduler1)



class mwText:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "mwTxt"
    CATEGORY = "mohwag/Utils"

    def mwTxt(self, text):
        return (text,)


class mwNumInt:
    numMult = 1

    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "numInt": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": moh.numMult}),
            }
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "mwNI"
    CATEGORY = "mohwag/Int"

    def mwNI(self, numInt):
        numOut = numInt
        return (int(numOut),)


class mwNumIntx8:
    numMult = 8

    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "numInt": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": moh.numMult}),
            }
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "mwNI"
    CATEGORY = "mohwag/Int"

    def mwNI(self, numInt):
        numOut = numInt
        return (int(numOut),)


class mwNumIntx16:
    numMult = 16

    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "numInt": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": moh.numMult}),
            }
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "mwNI"
    CATEGORY = "mohwag/Int"

    def mwNI(self, numInt):
        numOut = numInt
        return (int(numOut),)


class mwNumIntx32:
    numMult = 32

    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "numInt": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": moh.numMult}),
            }
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "mwNI"
    CATEGORY = "mohwag/Int"

    def mwNI(self, numInt):
        numOut = numInt
        return (int(numOut),)


class mwNumIntx64:
    numMult = 64

    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "numInt": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": moh.numMult}),
            }
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "mwNI"
    CATEGORY = "mohwag/Int"

    def mwNI(self, numInt):
        numOut = numInt
        return (int(numOut),)


class mwNumIntx64s:
    numMult = 64

    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": -10000, "max": 10000, "step": moh.numMult}),
                "height": ("INT", {"default": 1024, "min": -10000, "max": 10000, "step": moh.numMult}),
            }
        }
    RETURN_TYPES = ("MWSIZEPIPE",)
    RETURN_NAMES = ("sPipe",)
    FUNCTION = "mwNI"
    CATEGORY = "mohwag/psPipe"

    def mwNI(self, width, height):
        sPipeOut = width, height
        return (sPipeOut,)
  

class mwsPipeCreate:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": -10000, "max": 10000}),
                "height": ("INT", {"default": 1024, "min": -10000, "max": 10000}),
            }
        }
    RETURN_TYPES = ("MWSIZEPIPE",)
    RETURN_NAMES = ("sPipe",)
    FUNCTION = "mwSPC"
    CATEGORY = "mohwag/psPipe"

    def mwSPC(self, width, height):
        sPipeOut = width, height
        return (sPipeOut,) 

class mwsPipeBranch:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"sPipe": ("MWSIZEPIPE",), }, }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("w", "h")
    FUNCTION = "mwPBO"
    CATEGORY = "mohwag/psPipe"

    def mwPBO(self, sPipe):
        w, h = sPipe
        return (w, h)
    

class mwpsPipeBranch:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"psPipe": ("MWPOSSIZEPIPE",), }, }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("x", "y", "w", "h")
    FUNCTION = "mwPSPBO"
    CATEGORY = "mohwag/psPipe"

    def mwPSPBO(self, psPipe):
        x, y, w, h = psPipe
        return (x, y, w, h)


class mwMultXY_divZ:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "x": ("INT", {"default": 1, "min": -10000, "max": 10000, "step": 1}),
                "y": ("INT", {"default": 1, "min": -10000, "max": 10000, "step": 1}),
                "z": ("INT", {"default": 1, "min": -10000, "max": 10000, "step": 1}),
            }
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "mwMTH"
    CATEGORY = "mohwag/Int"

    def mwMTH(self, x, y, z):
        numOut = x * y / z
        return (int(numOut), )


class mwsPipe_MultOver8:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "sPipe_start": ("MWSIZEPIPE",),
                "multOver8": ("INT", {"default": 8, "min": 0, "max": 64, "step":1}),
            }
        }
    RETURN_TYPES = ("MWSIZEPIPE", "INT", "MWSIZEPIPE",)
    RETURN_NAMES = ("sPipe_start", "multOver8", "sPipe_curr",)
    FUNCTION = "mwNSP"
    CATEGORY = "mohwag/other"

    def mwNSP(self, sPipe_start, multOver8):
        initw, inith = sPipe_start

        outw = int(multOver8 * initw // 8)
        outh = int(multOver8 * inith // 8)

        sPipeCurr = outw, outh

        return (sPipe_start, multOver8, sPipeCurr, )
    

class mwsPipe_NextMO8:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "sPipe_start": ("MWSIZEPIPE",),
                "nextMO8": ("INT", {"default": 8, "min": 0, "max": 64, "step":1}),
            }
        }
    RETURN_TYPES = ("MWSIZEPIPE", "INT", "MWSIZEPIPE",)
    RETURN_NAMES = ("sPipe_start", "nextMO8", "sPipe_next",)
    FUNCTION = "mwNSP"
    CATEGORY = "mohwag/other"

    def mwNSP(self, sPipe_start, nextMO8):
        initw, inith = sPipe_start

        outw = int(nextMO8 * initw // 8)
        outh = int(nextMO8 * inith // 8)

        sPipeOut = outw, outh

        return (sPipe_start, nextMO8, sPipeOut, )


class mwCompScale:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                #"x": ("INT", {"default": 0, "min": 0, "step": 1}),
                #"y": ("INT", {"default": 0, "min": 0, "step": 1}),
                #"w": ("INT", {"default": 64, "min": 64, "step": 1}),
                #"h": ("INT", {"default": 64, "min": 64, "step": 1}),
                "psPipe": ("MWPOSSIZEPIPE",),
                "multOver8": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "nextMO8": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
            }
        }
    RETURN_TYPES = ("MWSIZEPIPE", "MWPOSSIZEPIPE")
    RETURN_NAMES = ("sPipe", "psPipe")
    FUNCTION = "mwCS"
    CATEGORY = "mohwag/other"

    def mwCS(self, psPipe, multOver8, nextMO8):

        x, y, w, h = psPipe

        upscaleX = x * nextMO8 / multOver8
        upscaleY = y * nextMO8 / multOver8

        upscaleW = w * nextMO8 / multOver8
        upscaleH = h * nextMO8 / multOver8

        upscaleX2 = upscaleX + upscaleW
        upscaleY2 = upscaleY + upscaleH


        #modX = 8 * (upscaleX // 8)
        #modY = 8 * (upscaleY // 8)

        modX = upscaleX
        modY = upscaleY

        modW = 8 * (upscaleW // 8)
        modH = 8 * (upscaleH // 8)

        modX2 = modX + modW
        modY2 = modY + modH


        if modX2 < upscaleX2:
            modX2 += 8
        if modY2 < upscaleY2:
            modY2 += 8

        if modX2 < upscaleX2:
            modX2 += 8
        if modY2 < upscaleY2:
            modY2 += 8


        xout = int(modX)
        yout = int(modY)
        wout = int(modX2 - modX)
        hout = int(modY2 - modY)

        outputS = wout, hout
        outputPS = xout, yout, wout, hout

        return (outputS, outputPS)


class mwMaskBoundingBoxRF:
    def __init__(self, device="cpu"):
        self.device = device
        
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "mask_bounding_box": ("MASK",),
                "image_mapped": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "deltaLft_units": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "deltaRgt_units": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "deltaTop_units": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "deltaBtm_units": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MWSIZEPIPE", "MWPOSSIZEPIPE", "MASK", "IMAGE")
    RETURN_NAMES = ("sPipe", "psPipe", "bounded mask", "bounded image")
    FUNCTION = "mwMBB"
    CATEGORY = "mohwag/other"
    
    #def mwMBB(self, mask_bounding_box, image_mapped, threshold, unitSize, deltaLft_units, deltaRgt_units, deltaTop_units, deltaBtm_units):
    def mwMBB(self, mask_bounding_box, image_mapped, threshold, deltaLft_units, deltaRgt_units, deltaTop_units, deltaBtm_units):

        #unitS = int(unitSize)
        unitS = 8

        # Get the mask where pixel values are above the threshold
        mask_above_threshold = mask_bounding_box > threshold

        # Compute the bounding box
        non_zero_positions = torch.nonzero(mask_above_threshold)
        if len(non_zero_positions) == 0:
            return (0, 0, 0, 0, 0, 0, torch.zeros_like(mask_bounding_box), torch.zeros_like(image_mapped))

        min_x = int(torch.min(non_zero_positions[:, 1]))
        max_x = int(torch.max(non_zero_positions[:, 1]))
        min_y = int(torch.min(non_zero_positions[:, 0]))
        max_y = int(torch.max(non_zero_positions[:, 0]))

        min_x_mw = unitS * (min_x // unitS)
        min_y_mw = unitS * (min_y // unitS)

        len_x = max_x - min_x
        len_y = max_y - min_y

        len_x_mw = unitS * (len_x // unitS)
        len_y_mw = unitS * (len_y // unitS)

        max_x_mw = min_x_mw + len_x_mw
        max_y_mw = min_y_mw + len_y_mw

        if max_x_mw < max_x:
            max_x_mw += unitS
            len_x_mw += unitS

        if max_y_mw < max_y:
            max_y_mw += unitS
            len_y_mw += unitS

        if max_x_mw < max_x:
            max_x_mw += unitS
            len_x_mw += unitS

        if max_y_mw < max_y:
            max_y_mw += unitS
            len_y_mw += unitS


        imgPrep = Image.fromarray(np.clip(255. * image_mapped.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        imgW = imgPrep.size[0]
        imgH = imgPrep.size[1]

        #lazy adjust for manual refinement ("RF")
        min_x_mw = int(max(0, min_x_mw + (deltaLft_units * unitS)))
        max_x_mw = int(min(imgW, max_x_mw + (deltaRgt_units * unitS)))
        min_y_mw = int(max(0, min_y_mw + (deltaTop_units * unitS)))
        max_y_mw = int(min(imgH, max_y_mw + (deltaBtm_units * unitS)))

        len_x_mw = int(max_x_mw - min_x_mw)
        len_y_mw = int(max_y_mw - min_y_mw)
    
        # Extract raw bounded mask
        raw_bb = mask_bounding_box[min_y_mw:max_y_mw,min_x_mw:max_x_mw]
        raw_img = image_mapped[:,min_y_mw:max_y_mw,min_x_mw:max_x_mw,:]

        outputS = len_x_mw, len_y_mw
        outputPS = min_x_mw, min_y_mw, len_x_mw, len_y_mw

        return (outputS, outputPS, raw_bb, raw_img)

class mwMaskBoundingBoxRF64:
    def __init__(self, device="cpu"):
        self.device = device
        
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "mask_bounding_box": ("MASK",),
                "image_mapped": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "deltaLft_units": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "deltaRgt_units": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "deltaTop_units": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "deltaBtm_units": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MWSIZEPIPE", "MWPOSSIZEPIPE", "MASK", "IMAGE")
    RETURN_NAMES = ("sPipe", "psPipe", "bounded mask", "bounded image")
    FUNCTION = "mwMBBS"
    CATEGORY = "mohwag/other"
    
    #def mwMBB(self, mask_bounding_box, image_mapped, threshold, unitSize, deltaLft_units, deltaRgt_units, deltaTop_units, deltaBtm_units):
    def mwMBBS(self, mask_bounding_box, image_mapped, threshold, deltaLft_units, deltaRgt_units, deltaTop_units, deltaBtm_units):

        #unitS = int(unitSize)
        unitS = 8
        multReq = 64

        # Get the mask where pixel values are above the threshold
        mask_above_threshold = mask_bounding_box > threshold

        # Compute the bounding box
        non_zero_positions = torch.nonzero(mask_above_threshold)
        if len(non_zero_positions) == 0:
            return (0, 0, 0, 0, 0, 0, torch.zeros_like(mask_bounding_box), torch.zeros_like(image_mapped))

        imgPrep = Image.fromarray(np.clip(255. * image_mapped.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        imgW = imgPrep.size[0]
        imgH = imgPrep.size[1]


        x0i = int(torch.min(non_zero_positions[:, 1]))
        x1i = int(torch.max(non_zero_positions[:, 1]))
        y0i = int(torch.min(non_zero_positions[:, 0]))
        y1i = int(torch.max(non_zero_positions[:, 0]))


        x0_1 = max(0, x0i + (deltaLft_units * unitS))
        x1_1 = min(imgW, x1i + (deltaRgt_units * unitS))
        y0_1 = max(0, y0i + (deltaTop_units * unitS))
        y1_1 = min(imgH, y1i + (deltaBtm_units * unitS))

        avg_xi = (x0_1 + x1_1) //2
        avg_yi = (y0_1 + y1_1) //2

        max_w = multReq * (imgW //multReq)
        max_h = multReq * (imgH //multReq)

        fin_w = min(max_w, multReq * round((x1_1 - x0_1)/multReq, 0))
        fin_h = min(max_h, multReq * round((y1_1 - y0_1)/multReq, 0))

        x0_2 = avg_xi - (fin_w /2)
        x1_2 = avg_xi + (fin_w /2)
        y0_2 = avg_yi - (fin_h /2)
        y1_2 = avg_yi + (fin_h /2)


        if x0_2 < 0:
            x0_2 = 0
            x1_2 = fin_w

        if  x1_2 > imgW:
            x0_2 = imgW - fin_w
            x1_2 = imgW

        if y0_2 < 0:
            y0_2 = 0
            y1_2 = fin_h

        if  y1_2 > imgH:
            y0_2 = imgH - fin_h
            y1_2 = imgH

        min_xf = int(x0_2)
        max_xf = int(x1_2)
        min_yf = int(y0_2)
        max_yf = int(y1_2)      

        fin_w = int(max_xf - min_xf)
        fin_h = int(max_yf - min_yf)        

        # Extract raw bounded mask
        raw_bb = mask_bounding_box[min_yf:max_yf,min_xf:max_xf]
        raw_img = image_mapped[:,min_yf:max_yf,min_xf:max_xf,:]

        outputS = fin_w, fin_h
        outputPS = min_xf, min_yf, fin_w, fin_h

        return (outputS, outputPS, raw_bb, raw_img)

NODE_CLASS_MAPPINGS = {
"mwBatch": mwBatch,
"mwpsConditioningSetArea": mwpsConditioningSetArea,
"mwFullPipe_Load": mwFullPipe_Load,
"mwCkpt_Load": mwCkpt_Load,
"mwCkpt_modelEdit": mwCkpt_modelEdit,
"mwFullPipe_ckptMerge": mwFullPipe_ckptMerge,
"mwCkpt_ckptMerge": mwCkpt_ckptMerge,
"mwLora_Load": mwLora_Load,
"mwFullPipe_loraMerge": mwFullPipe_loraMerge,
"mwLtntPipe_Create": mwLtntPipe_Create,
"mwLtntPipe_CropScale": mwLtntPipe_CropScale,
"mwLtntPipeBranch1": mwLtntPipeBranch1,
"mwLtntPipe_View": mwLtntPipe_View,
"mwCond": mwCond,
"mwCondXL": mwCondXL,
"mwCondXLa": mwCondXLa,
"mwFullPipe_KSAStart": mwFullPipe_KSAStart,
"mwFullPipe_KSA": mwFullPipe_KSA,
"mwModelBranch1": mwModelBranch1,
"mwModelBranch2": mwModelBranch2,
"mwModelBranch1_ckpt": mwModelBranch1_ckpt,
"mwClipBranch1": mwClipBranch1,
"mwClipBranch2": mwClipBranch2,
"mwVaeBranch1": mwVaeBranch1,
"mwVaeBranch2": mwVaeBranch2,
"mwCkptBranch1": mwCkptBranch1,
"mwCkptBranch2": mwCkptBranch2,
"mwSchedEdit": mwSchedEdit,
"mwModelEdit": mwModelEdit,
"mwFullPipeBranch1": mwFullPipeBranch1,
"mwText": mwText,
"mwNumInt": mwNumInt,
"mwNumIntx8": mwNumIntx8,
"mwNumIntx16": mwNumIntx16,
"mwNumIntx32": mwNumIntx32,
"mwNumIntx64": mwNumIntx64,
"mwNumIntx64s": mwNumIntx64s,
"mwsPipeCreate": mwsPipeCreate,
"mwsPipeBranch": mwsPipeBranch,
"mwpsPipeBranch": mwpsPipeBranch,
"mwMultXY_divZ": mwMultXY_divZ,
"mwsPipe_MultOver8": mwsPipe_MultOver8,
"mwsPipe_NextMO8": mwsPipe_NextMO8,
"mwCompScale": mwCompScale,
"mwMaskBoundingBoxRF": mwMaskBoundingBoxRF,
"mwMaskBoundingBoxRF64": mwMaskBoundingBoxRF64,
}

NODE_DISPLAY_NAME_MAPPINGS = {
"mwBatch": "Batch",
"mwpsConditioningSetArea": "psConditioningSetArea",
"mwFullPipe_Load": "FullPipe_Load",
"mwCkpt_Load": "Ckpt_Load",
"mwCkpt_modelEdit": "Ckpt_modelEdit",
"mwFullPipe_ckptMerge": "FullPipe_ckptMerge",
"mwCkpt_ckptMerge": "Ckpt_ckptMerge",
"mwLora_Load": "Lora_Load",
"mwFullPipe_loraMerge": "FullPipe_loraMerge",
"mwLtntPipe_Create": "LtntPipe_Create",
"mwLtntPipe_CropScale": "LtntPipe_CropScale",
"mwLtntPipeBranch1": "LtntPipeBranch1",
"mwLtntPipe_View": "LtntPipe_View",
"mwCond": "Cond",
"mwCondXL": "CondXL",
"mwCondXLa": "CondXLa",
"mwFullPipe_KSAStart": "FullPipe_KSAStart",
"mwFullPipe_KSA": "FullPipe_KSA",
"mwModelBranch1": "ModelBranch1",
"mwModelBranch2": "ModelBranch2",
"mwModelBranch1_ckpt": "ModelBranch1_ckpt",
"mwClipBranch1": "ClipBranch1",
"mwClipBranch2": "ClipBranch2",
"mwVaeBranch1": "VaeBranch1",
"mwVaeBranch2": "VaeBranch2",
"mwCkptBranch1": "CkptBranch1",
"mwCkptBranch2": "CkptBranch2",
"mwSchedEdit": "SchedEdit",
"mwModelEdit": "ModelEdit",
"mwFullPipeBranch1": "FullPipeBranch1",
"mwText": "Text",
"mwNumInt": "NumInt",
"mwNumIntx8": "NumIntx8",
"mwNumIntx16": "NumIntx16",
"mwNumIntx32": "NumIntx32",
"mwNumIntx64": "NumIntx64",
"mwNumIntx64s": "NumIntx64s",
"mwsPipeCreate": "sPipeCreate",
"mwsPipeBranch": "sPipeBranch",
"mwpsPipeBranch": "psPipeBranch",
"mwMultXY_divZ": "MultXY_divZ",
"mwsPipe_MultOver8": "sPipe_MultOver8",
"mwsPipe_NextMO8": "sPipe_NextMO8",
"mwCompScale": "CompScale",
"mwMaskBoundingBoxRF": "MaskBoundingBoxRF",
"mwMaskBoundingBoxRF64": "MaskBoundingBoxRF64",
}