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
import scipy.ndimage
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
from nodes import MAX_RESOLUTION, ConditioningSetArea, CheckpointLoaderSimple, ImageScale, EmptyLatentImage, common_ksampler
condSetAreaMethod = ConditioningSetArea().append
imageScaleMethod = ImageScale().upscale  #(self, image, upscale_method, width, height, crop)
emptyLatentImageMethod = EmptyLatentImage().generate  #(self, width, height, batch_size=1)

from comfy_extras.nodes_model_merging import ModelMergeSimple, CLIPMergeSimple
modelMergeMethod = ModelMergeSimple().merge
clipMergeMethod = CLIPMergeSimple().merge
ckptLoaderSimpleMethod = CheckpointLoaderSimple().load_checkpoint

from comfy_extras.nodes_mask import SolidMask, MaskComposite, FeatherMask, InvertMask, ImageCompositeMasked, MaskToImage, LatentCompositeMasked  #composite, 
solidMaskMethod = SolidMask().solid  #(self, value, width, height)
maskCompositeMethod = MaskComposite().combine  #(self, destination, source, x, y, operation)
featherMaskMethod = FeatherMask().feather  #(self, mask, left, top, right, bottom)
invertMaskMethod = InvertMask().invert  #(self, mask)
imageCompositeMaskedMethod = ImageCompositeMasked().composite  #(self, destination, source, x, y, resize_source, mask = None)
maskToImageMethod = MaskToImage().mask_to_image  #(self, mask)
latentCompositeMaskedMethod = LatentCompositeMasked().composite  #(self, destination, source, x, y, resize_source, mask = None):

from comfy_extras.nodes_images import ImageCrop
imageCropMethod = ImageCrop().crop  #(self, image, width, height, x, y)





class mwGridOverlay:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "image": ("IMAGE",),
                "sPipe": ("MWSIZEPIPE",),
                "scndSpacing": ([8, 16, 32, 64, 128, 256, 512], {"default": 64}),
                "primSpacMult": ([2, 4, 8, 16], {"default": 4}),
                "scndThickness": ("INT", {"default": 2, "min": 2, "max": 16, "step": 2}),
                "primThickMult": ([2, 4, 8, 16], {"default": 4}),
                "lineColor": (["white", "black"],),
                "transparency": ("INT", {"default": 80, "min": 20, "max": 100, "step": 5})
            }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "mwGO"
    CATEGORY = "mohwag/manip"

    def mwGO(self, image, sPipe, primSpacMult, scndSpacing, scndThickness, primThickMult, lineColor, transparency):
        
        #primSpacing = max(scndSpacing * 2, primSpacing)
        transparency = transparency /100

        #primSpacMult = primSpacing //scndSpacing
        primThickness = scndThickness * primThickMult

        scndSpecSpc = scndSpacing - (scndThickness //2)
        primSpecSpc = scndSpacing - ((primThickness) //2)

        w, h = sPipe

        transMask = solidMaskMethod(transparency, w, h)[0]
        backMask = solidMaskMethod(0.0, w, h)[0]

        wCount = (w //scndSpacing) + (w % scndSpacing > 0)
        hCount = (h //scndSpacing) + (h % scndSpacing > 0)

        vtFrontMaskS = solidMaskMethod(1.0, scndThickness, h)[0]
        hzFrontMaskS = solidMaskMethod(1.0, w, scndThickness)[0]

        vtFrontMaskP = solidMaskMethod(1.0, primThickness, h)[0]
        hzFrontMaskP = solidMaskMethod(1.0, w, primThickness)[0]

        vtFrontMaskStart = solidMaskMethod(1.0, primThickness //2, h)[0]
        hzFrontMaskStart = solidMaskMethod(1.0, w, primThickness //2)[0]

        finalMask = maskCompositeMethod(backMask, vtFrontMaskStart, 0, 0, "add")[0]
        finalMask = maskCompositeMethod(finalMask, hzFrontMaskStart, 0, 0, "add")[0]

        for xi in range(wCount):
            if (xi + 1) % primSpacMult == 0:
                finalMask = maskCompositeMethod(finalMask, vtFrontMaskP, (xi * scndSpacing) + primSpecSpc, 0, "add")[0]
            else:
                finalMask = maskCompositeMethod(finalMask, vtFrontMaskS, (xi * scndSpacing) + scndSpecSpc, 0, "add")[0]

        for yi in range(hCount):
            if (yi + 1) % primSpacMult == 0:
                finalMask = maskCompositeMethod(finalMask, hzFrontMaskP, 0, (yi * scndSpacing) + primSpecSpc, "add")[0]
            else:
                finalMask = maskCompositeMethod(finalMask,  hzFrontMaskS, 0, (yi * scndSpacing) + scndSpecSpc, "add")[0]

        if lineColor == "white":
            gridImage = maskToImageMethod(finalMask)[0]
        else:
            gridImage = maskToImageMethod(invertMaskMethod(finalMask)[0])[0]
        
        finalMask = maskCompositeMethod(finalMask, transMask, 0, 0, "multiply")[0]

        return imageCompositeMaskedMethod(image, gridImage, 0, 0, False, finalMask)


class mwMaskSegment:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "sPipe": ("MWSIZEPIPE",),
                "invert": ("BOOLEAN", {"default": False}),
                "unitVal": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),               
                "deltaLft_units": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "deltaRgt_units": ("INT", {"default": 0, "min": -1024, "max": 0, "step": 1}),
                "deltaTop_units": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "deltaBtm_units": ("INT", {"default": 0, "min": -1024, "max": 0, "step": 1}),
                "unitVal_feath": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),               
                "feathLft_units": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "feathRgt_units": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "feathTop_units": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "feathBtm_units": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
            }}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "mwMS"
    CATEGORY = "mohwag/mask"

    def mwMS(self, sPipe, invert, unitVal, deltaLft_units, deltaRgt_units, deltaTop_units, deltaBtm_units, unitVal_feath, feathLft_units, feathRgt_units, feathTop_units, feathBtm_units):
        
        w, h = sPipe
        backMask = solidMaskMethod(0.0, w, h)[0]

        x0 = 0 + unitVal * deltaLft_units
        x1 = w + (unitVal * deltaRgt_units)
        y0 = 0 + unitVal * deltaTop_units
        y1 = h + (unitVal * deltaBtm_units)


        x0 = min(w - 1, x0)
        x1 = max(0 + 1, x1)
        y0 = min(h - 1, y0)
        y1 = max(0 + 1, y1)

        w0 = max(1, x1 - x0)
        h0 = max(1, y1 - y0)


        fLft = unitVal_feath * feathLft_units
        fRgt = unitVal_feath * feathRgt_units
        fTop = unitVal_feath * feathTop_units
        fBtm = unitVal_feath * feathBtm_units

        frontMask_preFeath = solidMaskMethod(1.0, w0, h0)[0]
        frontMask = featherMaskMethod(frontMask_preFeath, fLft, fTop, fRgt, fBtm)[0]  #(self, mask, left, top, right, bottom)
        
        combMask = maskCompositeMethod(backMask, frontMask, x0, y0, "add")[0]

        if invert:
            finMask = invertMaskMethod(combMask)[0]
        else:
            finMask = combMask
        
        return (finMask,)

class mwMaskCompSameSize:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "destination": ("MASK",),
                "source": ("MASK",),
                "operation": (["multiply", "add", "subtract", "and", "or", "xor"],),
            }}
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "mwCSS"
    CATEGORY = "mohwag/mask"
    def mwCSS(self, destination, source, operation):
        outMask = maskCompositeMethod(destination, source, 0, 0, operation)[0]
        return (outMask,)

class mwMaskTweak:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "mask": ("MASK",),
                "operation0": (["multiply", "add"], {"default": "add"}),
                "operation1": (["none", "multiply", "add"], {"default": "none"}),
                "operation2": (["none", "multiply", "add"], {"default": "none"}),
                "operation3": (["none", "multiply", "add"], {"default": "none"}),
            }}
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "mwMT"
    CATEGORY = "mohwag/mask"
    def mwMT(self, mask, operation0, operation1, operation2, operation3):
        outMask = maskCompositeMethod(mask, mask, 0, 0, operation0)[0]
        if operation1 != "none":
            outMask = maskCompositeMethod(outMask, outMask, 0, 0, operation1)[0]
        if operation2 != "none":
            outMask = maskCompositeMethod(outMask, outMask, 0, 0, operation2)[0]
        if operation3 != "none":
            outMask = maskCompositeMethod(outMask, outMask, 0, 0, operation3)[0]
        return (outMask,)


class mwMaskStack:
    maskOps = ["multiply", "add", "subtract", "and", "or", "xor"]
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "operation1": (moh.maskOps,),
                "operation2": (moh.maskOps,),
                "operation3": (moh.maskOps,),
                "operation4": (moh.maskOps,),
            },
            "optional": {
                "mask0": ("MASK",),
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "mask3": ("MASK",),
                "mask4": ("MASK",),
            }}
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "mwMS"
    CATEGORY = "mohwag/mask"
    def mwMS(self, operation1, operation2, operation3, operation4, mask0 = None, mask1 = None, mask2 = None, mask3 = None, mask4 = None):
        #outMask = maskCompositeMethod(mask0, mask1, 0, 0, operation1)[0]
        notNoneCnt = 0
        if mask0 != None:
            outMask = mask0
            notNoneCnt = notNoneCnt + 1
        if mask1 != None:
            outMask = maskCompositeMethod(outMask, mask1, 0, 0, operation1)[0]
            notNoneCnt = notNoneCnt + 1     
        if mask2 != None:
            outMask = maskCompositeMethod(outMask, mask2, 0, 0, operation2)[0]
            notNoneCnt = notNoneCnt + 1
        if mask3 != None:
            outMask = maskCompositeMethod(outMask, mask3, 0, 0, operation3)[0]
            notNoneCnt = notNoneCnt + 1
        if mask4 != None:
            outMask = maskCompositeMethod(outMask, mask4, 0, 0, operation4)[0]
            notNoneCnt = notNoneCnt + 1
        if notNoneCnt == 0:
            outMask = solidMaskMethod(1.0, 8, 8)[0]  #(self, value, width, height)
        return (outMask,)


class mwImageCompositeMasked:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "psPipe": ("MWPOSSIZEPIPE",),
            },
            "optional": {
                "mask": ("MASK",),
            }}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "mwICM"
    CATEGORY = "mohwag/image"
    def mwICM(self, destination, source, psPipe, mask = None):
        x, y, w, h = psPipe
        output = imageCompositeMaskedMethod(destination, source, x, y, False, mask)[0]
        return (output,)


class mwImageScale:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "image": ("IMAGE",),
            "upscale_method": (moh.upscale_methods,),
            "sPipe": ("MWSIZEPIPE",),
            }}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "mwIS"
    CATEGORY = "mohwag/image"
    def mwIS(self, image, upscale_method, sPipe):
        w, h = sPipe
        output = imageScaleMethod(image, upscale_method, w, h, "disabled")[0]   #(self, image, upscale_method, width, height, crop)
        return (output,)


class mwImageCrop:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "image": ("IMAGE",),
            "psPipe": ("MWPOSSIZEPIPE",),
            }}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "mwIC"
    CATEGORY = "mohwag/image"
    def mwIC(self, image, psPipe):
        x, y, w, h = psPipe
        output = imageCropMethod(image, w, h, x, y)[0]  ##(self, image, width, height, x, y)
        return (output,)
    

class mwImageCropwParams:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "image": ("IMAGE",),
            "sPipe": ("MWSIZEPIPE",),
            "unitVal": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),               
            "deltaLft_units": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
            "deltaRgt_units": ("INT", {"default": 0, "min": -1024, "max": 0, "step": 1}),
            "deltaTop_units": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
            "deltaBtm_units": ("INT", {"default": 0, "min": -1024, "max": 0, "step": 1}),
            }}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "mwICWP"
    CATEGORY = "mohwag/image"
    def mwICWP(self, image, sPipe, unitVal, deltaLft_units, deltaRgt_units, deltaTop_units, deltaBtm_units):
        w, h = sPipe

        x = int(deltaLft_units * unitVal)
        y = int(deltaTop_units * unitVal)
        w = int(w - x + (deltaRgt_units * unitVal))
        h = int(h - y + (deltaBtm_units * unitVal))

        output = imageCropMethod(image, w, h, x, y)[0]  ##(self, image, width, height, x, y)
        return (output,)


class mwImageConform_StartSizeXL:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "image": ("IMAGE",),
            "upscale_method": (moh.upscale_methods,),
            }}

    RETURN_TYPES = ("IMAGE", "MWSIZEPIPE")
    RETURN_NAMES = ("image", "sPipe")
    FUNCTION = "mwICSSXL"
    CATEGORY = "mohwag/image"

    def mwICSSXL(self, image, upscale_method):
        aspectRatList = [0.25, 0.26, 0.27, 0.28, 0.32, 0.33, 0.35, 0.4, 0.42, 0.48, 0.5, 0.52, 0.57, 0.6, 0.68, 0.72, 0.78, 0.82, 0.88, 0.94, 1, 1.07, 1.13, 1.21, 1.29, 1.38, 1.46, 1.67, 1.75, 2, 2.09, 2.4, 2.5, 2.89, 3, 3.11, 3.63, 3.75, 3.88, 4]
        targetW_list = [512, 512, 512, 512, 576, 576, 576, 640, 640, 704, 704, 704, 768, 768, 832, 832, 896, 896, 960, 960, 1024, 1024, 1088, 1088, 1152, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048]
        targetH_list = [2048, 1984, 1920, 1856, 1792, 1728, 1664, 1600, 1536, 1472, 1408, 1344, 1344, 1280, 1216, 1152, 1152, 1088, 1088, 1024, 1024, 960, 960, 896, 896, 832, 832, 768, 768, 704, 704, 640, 640, 576, 576, 576, 512, 512, 512, 512]

        modImage = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        actualW = modImage.size[0]
        actualH = modImage.size[1]

        rats = actualW / actualH
        testListrs = [abs(round(x - rats,3)) for x in aspectRatList]
        testMinrs = min(testListrs)
        testMinLocrs = testListrs.index(testMinrs)
        startW = targetW_list[testMinLocrs]
        startH = targetH_list[testMinLocrs]

        outImage = imageScaleMethod(image, upscale_method, startW, startH, "disabled")[0]   #(self, image, upscale_method, width, height, crop)
        out_sPipe = startW, startH
        return (outImage, out_sPipe)

    
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

    CATEGORY = "mohwag/image"
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
        images = [TF.resize(i, squareSize) for i in images]
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
    CATEGORY = "mohwag/Conditioning"
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
    
class mwCkpt_modelEdit:
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


class mwLtntPipe_Create:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "ltnt": ("LATENT", ),
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


class mwLtntPipe_Create2:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "image": ("IMAGE", ),
            "mwFullPipe": ("MWFULLPIPE",),
            "initialS": ("MWSIZEPIPE",)
            }}
    RETURN_TYPES = ("MWLTNTPIPE",)
    RETURN_NAMES = ("ltntPipe",)
    FUNCTION = "mwLPC"
    CATEGORY = "mohwag/LatentPipe"
    def mwLPC(self, image, mwFullPipe, initialS):
        initialW, initialH = initialS
        mwCkpt1, _, _, _ = mwFullPipe
        _, _, vaeAlways = mwCkpt1
        ltnt = {"samples":vaeAlways.encode(image[:,:,:,:3])}
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


class mwLtntPipe_CropScale:
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
    CATEGORY = "mohwag/manip"
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
        output_sPipeStart = startW, startH
        if (cropLft == 0 and cropRgt == 0) and (cropTop == 0 and cropBtm == 0):
            if TotDeltaW == 0 and TotDeltaH == 0:
                return (ltntPipe, outputS, outputPS, output_sPipeStart)
            else:
                cropdLtnt = ltnt.copy()
        else:
            offsetW = cropLft
            endW = inputW - cropRgt
            offsetH = cropTop
            endH = inputH - cropBtm

            offsetW8 = int(offsetW // 8)
            endW8 = int(endW // 8)
            offsetH8 = int(offsetH // 8)
            endH8 = int(endH // 8)

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

class mwLtntPipe_VertSqueeze:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "ltntPipe": ("MWLTNTPIPE", ),
            "y": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "sqzVertRng_units": ("INT", {"default": 1, "min": 1, "max": 1024, "step": 1}),
            "sqzDelta_units": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
            "cropToOutput_upscaleMethod": (moh.upscale_methods,),
            }}

    RETURN_TYPES = ("MWLTNTPIPE", "MWSIZEPIPE", "MWSIZEPIPE")
    RETURN_NAMES = ("ltntPipe", "sPipe", "new_sPipe_start")
    FUNCTION = "mwLPVS"
    CATEGORY = "mohwag/manip"

    def mwLPVS(self, ltntPipe, y, sqzVertRng_units, sqzDelta_units, cropToOutput_upscaleMethod):
        sqzEndHt_units = sqzVertRng_units - sqzDelta_units
        ltnt, startW, startH, inputW, inputH = ltntPipe
        unitVal = 8 * inputW / startW
        y1 = int(y)
        y2 = int(y + (sqzVertRng_units * unitVal))
        y3 = int(inputH)
        x1 = int(inputW)
        sqzH = int(sqzEndHt_units * unitVal)
        ltntSamples = ltnt['samples']

        if y2 < y3:
            ltntBtm = ltnt.copy()
            ltntBtm['samples'] = ltntSamples[:,:,y2//8:y3//8, 0:x1//8]
            
        ltntMdl = ltnt.copy()
        ltntMdl['samples'] = ltntSamples[:,:,y1//8:y2//8, 0:x1//8]

        ltntMdlSqz = ltntMdl.copy()
        ltntMdlSqz['samples'] = comfy.utils.common_upscale(ltntMdl["samples"], x1//8, sqzH//8, cropToOutput_upscaleMethod, "disabled")

        TotDeltaH_units = sqzDelta_units
        TotDeltaW_units = 0

        TotDeltaW = int(unitVal * TotDeltaW_units)
        TotDeltaH = int(unitVal * TotDeltaH_units)

        outputW = int(inputW - TotDeltaW)
        outputH = int(inputH - TotDeltaH)

        outputS = outputW, outputH

        newStartW = int(startW - (8 * TotDeltaW_units))
        newStartH = int(startH - (8 * TotDeltaH_units))

        newSPipeStart = newStartW, newStartH

        #latentCompositeMaskedMethod = LatentCompositeMasked().composite  #(self, destination, source, x, y, resize_source, mask = None):

        backLtnt = ltnt.copy()
        backLtnt['samples'] = ltntSamples[:,:,0:outputH//8, 0:outputW//8]

        addMdlLtnt = latentCompositeMaskedMethod(backLtnt, ltntMdlSqz, 0, y1, False, None)[0]

        sqzY2 = int(y1 + sqzH)

        if sqzY2 < outputH:
            addBtmLtnt = latentCompositeMaskedMethod(addMdlLtnt, ltntBtm, 0, sqzY2, False, None)[0]
        else:
            addBtmLtnt = addMdlLtnt
    
        ltntPipeNew = addBtmLtnt, newStartW, newStartH, outputW, outputH
        return (ltntPipeNew, outputS, newSPipeStart)


class mwLtntPipe_VertSqzExpnd:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "ltntPipe": ("MWLTNTPIPE", ),
            "y": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "sqzVertRng_units": ("INT", {"default": 1, "min": 1, "max": 1024, "step": 1}),
            "sqzDelta_units": ("INT", {"default": 0, "min": -1024, "max": 1024, "step": 4}),
            "cropToOutput_upscaleMethod": (moh.upscale_methods,),
            }}

    RETURN_TYPES = ("MWLTNTPIPE", "MWSIZEPIPE", "MWSIZEPIPE")
    RETURN_NAMES = ("ltntPipe", "sPipe", "new_sPipe_start")
    FUNCTION = "mwLPVS"
    CATEGORY = "mohwag/manip"

    def mwLPVS(self, ltntPipe, y, sqzVertRng_units, sqzDelta_units, cropToOutput_upscaleMethod):

        sqzEndHt_units = sqzVertRng_units + sqzDelta_units
        ltnt, startW, startH, inputW, inputH = ltntPipe
        unitVal = 8 * inputW / startW
        y1 = int(y)
        y2 = int(y + (sqzVertRng_units * unitVal))
        y3 = int(inputH)
        x1 = int(inputW)

        sqzH = int(sqzEndHt_units * unitVal)

        ltntSamples = ltnt['samples']
   
        if y1 > 0:
            ltntTop = ltnt.copy()
            ltntTop['samples'] = ltntSamples[:,:,0:y1//8, 0:x1//8]

        if y2 < y3:
            ltntBtm = ltnt.copy()
            ltntBtm['samples'] = ltntSamples[:,:,y2//8:y3//8, 0:x1//8]
            
        ltntMdl = ltnt.copy()
        ltntMdl['samples'] = ltntSamples[:,:,y1//8:y2//8, 0:x1//8]

        ltntMdlSqz = ltntMdl.copy()
        ltntMdlSqz['samples'] = comfy.utils.common_upscale(ltntMdl["samples"], x1//8, sqzH//8, cropToOutput_upscaleMethod, "disabled")

        TotDeltaH_units = sqzDelta_units
        TotDeltaW_units = 0

        TotDeltaW = int(unitVal * TotDeltaW_units)
        TotDeltaH = int(unitVal * TotDeltaH_units)

        outputW = int(inputW + TotDeltaW)
        outputH = int(inputH + TotDeltaH)

        outputS = outputW, outputH

        newStartW = int(startW + (8 * TotDeltaW_units))
        newStartH = int(startH + (8 * TotDeltaH_units))

        newSPipeStart = newStartW, newStartH
        
        #emptyLatentImageMethod = EmptyLatentImage().generate  #(self, width, height, batch_size=1)
        #latentCompositeMaskedMethod = LatentCompositeMasked().composite  #(self, destination, source, x, y, resize_source, mask = None):

        #backLtnt = ltnt.copy()
        endSizeLtntEmpty = emptyLatentImageMethod(outputW, outputH, 1)[0]
        endSizeLtntEmptySamples = endSizeLtntEmpty['samples']
        backLtnt = endSizeLtntEmpty.copy()
        backLtnt['samples'] = endSizeLtntEmptySamples[:,:,0:outputH//8, 0:outputW//8]

        if y1 > 0:
            addTopLtnt = latentCompositeMaskedMethod(backLtnt, ltntTop, 0, 0, False, None)[0]
        else:
            addTopLtnt = backLtnt

        addMdlLtnt = latentCompositeMaskedMethod(addTopLtnt, ltntMdlSqz, 0, y1, False, None)[0]

        sqzY2 = int(y1 + sqzH)
        if sqzY2 < outputH:
            addBtmLtnt = latentCompositeMaskedMethod(addMdlLtnt, ltntBtm, 0, sqzY2, False, None)[0]
        else:
            addBtmLtnt = addMdlLtnt
    
        ltntPipeNew = addBtmLtnt, newStartW, newStartH, outputW, outputH
        return (ltntPipeNew, outputS, newSPipeStart)

class mwImage_VertSqzExpnd:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "image": ("IMAGE", ),
            "sPipeStart": ("MWSIZEPIPE",),
            "y": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "sqzVertRng_units": ("INT", {"default": 1, "min": 1, "max": 1024, "step": 1}),
            "sqzDelta_units": ("INT", {"default": 0, "min": -1024, "max": 1024, "step": 1}),
            "scaleMethod": (moh.upscale_methods,),
            }}
    RETURN_TYPES = ("IMAGE", "MWSIZEPIPE", "MWSIZEPIPE")
    RETURN_NAMES = ("image", "sPipe", "new_sPipeStart")
    FUNCTION = "mwLPVS"
    CATEGORY = "mohwag/manip"
    def mwLPVS(self, image, sPipeStart, y, sqzVertRng_units, sqzDelta_units, scaleMethod):
        modImage = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        inputW = modImage.size[0]
        inputH = modImage.size[1]
        
        if sqzDelta_units == 0:
            return (image, (inputW, inputH), sPipeStart)

        startW, startH = sPipeStart
        unitVal = 8 * inputW / startW
        sqzRngH = int(sqzVertRng_units * unitVal)
        sqzdH_units = sqzVertRng_units + sqzDelta_units
        sqzdH = int(sqzdH_units * unitVal)
        sqzDeltaH = sqzdH - sqzRngH
        y1 = int(y)
        y2_0 = int(y + sqzRngH)
        y3_0 = int(inputH)
        x1 = int(inputW)
        y2_1 = int(y + sqzdH)
        y3_1 = y3_0 + sqzDeltaH

        if y1 > 0:
            imgTop = imageCropMethod(image, x1, y1, 0, 0)[0]
        imgMdlSqzd = imageScaleMethod(imageCropMethod(image, x1, sqzRngH, 0, y1)[0], scaleMethod, x1, sqzdH, "disabled")[0]
        if y2_0 < y3_0:
            imgBtm = imageCropMethod(image, x1, y3_0 - y2_0, 0, y2_0)[0]

        if sqzDelta_units < 0:
            backImage = imageCropMethod(image, x1, y3_1, 0, 0)[0]
        else:
            backImage = imageScaleMethod(image, "bicubic", x1, y3_1, "disabled")[0]

        if y1 > 0:
            addTopImg = imageCompositeMaskedMethod(backImage, imgTop, 0, 0, False, None)[0]
        else:
            addTopImg = backImage
        addMdlImg = imageCompositeMaskedMethod(addTopImg, imgMdlSqzd, 0, y1, False, None)[0]
        if y2_0 < y3_0:
            addBtmImg = imageCompositeMaskedMethod(addMdlImg, imgBtm, 0, y2_1, False, None)[0]
        else:
            addBtmImg = addMdlImg

        TotDeltaH_units = sqzDelta_units
        TotDeltaW_units = 0
        TotDeltaW = int(unitVal * TotDeltaW_units)
        TotDeltaH = int(unitVal * TotDeltaH_units)
        outputW = int(inputW + TotDeltaW)
        outputH = int(inputH + TotDeltaH)
        outputS = outputW, outputH
        newStartW = int(startW + (8 * TotDeltaW_units))
        newStartH = int(startH + (8 * TotDeltaH_units))
        newSPipeStart = newStartW, newStartH
        return (addBtmImg, outputS, newSPipeStart)

class mwImageVertTaper:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "image": ("IMAGE", ),
            "y0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "y1": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "x0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),            
            "xtl": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbl": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),   
            "x1": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),            
            "xtr": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbr": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "upscaleMethod": (moh.upscale_methods,),
            }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "mwIVT"
    CATEGORY = "mohwag/manip"

    def mwIVT(self, image, y0, y1, x0, xtl, xbl, x1, xtr, xbr, upscaleMethod):

        modImage = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        imgW = int(modImage.size[0])
        #imgH = int(modImage.size[1])

        midW = x1 - x0
        midH = y1 - y0

        slopeL = (xbl - xtl) / (midH)
        slopeR = (xbr - xtr) / (midH)

        interL = xtl
        interR = xtr

        imgFinal = image

        for yi in range(midH):

            pointL = int(interL + (yi * slopeL))
            pointR = int(interR + (yi * slopeR))
            sqzW = int((pointR - pointL))

            imgMdli = imageCropMethod(image, midW, 1, x0, y0 + yi)[0]  #(self, image, width, height, x, y)
            imgMdliSqz = imageScaleMethod(imgMdli, upscaleMethod, sqzW, 1, "disabled")[0]  #(self, image, upscale_method, width, height, crop)

            if x0 > 0:
                imgLfti = imageCropMethod(image, min(x0, pointL), 1, x0 - min(x0, pointL), y0 + yi)[0]
                addLft = imageCompositeMaskedMethod(imgFinal, imgLfti, max(0, pointL - x0) , y0 + yi, False, None)[0]  #(self, destination, source, x, y, resize_source, mask = None)
            else:
                addLft = imgFinal

            addMdlSqz = imageCompositeMaskedMethod(addLft, imgMdliSqz, pointL , y0 + yi, False, None)[0]

            if x1 < imgW:
                imgRgti = imageCropMethod(image, min(imgW - x1, imgW - pointR), 1, x1, y0 + yi)[0]
                addRgt = imageCompositeMaskedMethod(addMdlSqz, imgRgti, pointR , y0 + yi, False, None)[0]
            else:
                addRgt = addMdlSqz

            imgFinal = addRgt

        return (imgFinal,)


class mwImageVertTaper2:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "image": ("IMAGE", ),
            #"sCurveMult": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            "y0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "y1": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xtl0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),            
            "xtr0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbl0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),   
            "xbr0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),            
            "xtld": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),            
            "xtrd": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
            "xbld": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),   
            "xbrd": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),     
            "upscaleMethod": (moh.upscale_methods,),
            }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "mwIVTT"
    CATEGORY = "mohwag/manip"

    def mwIVTT(self, image, y0, y1, xtl0, xtr0, xbl0, xbr0, xtld, xtrd, xbld, xbrd, upscaleMethod):
        xtl1 = xtl0 + xtld
        xtr1 = xtr0 + xtrd
        xbl1 = xbl0 + xbld
        xbr1 = xbr0 + xbrd


        modImage = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        imgW = int(modImage.size[0])
        #imgH = int(modImage.size[1])

        #midW = x1 - x0
        midH = int(y1 - y0)

        slopeL0 = (xbl0 - xtl0) / (midH)
        slopeR0 = (xbr0 - xtr0) / (midH)
        slopeL1 = (xbl1 - xtl1) / (midH)
        slopeR1 = (xbr1 - xtr1) / (midH)

        interL0 = xtl0
        interR0 = xtr0
        interL1 = xtl1
        interR1 = xtr1


        avg_yi = ((y1 - y0) //2) - 1

        #crossX0 = ((interL0*slopeR0)-(interR0*slopeL0))/((slopeR0 - slopeL0))
        #crossX1 = ((interL1*slopeR1)-(interR1*slopeL1))/((slopeR1 - slopeL1))

        imgFinal = image

        for yi in range(midH):

            pointL0 = int(interL0 + (yi * slopeL0))
            pointR0 = int(interR0 + (yi * slopeR0))
            pointL1 = int(interL1 + (yi * slopeL1))
            pointR1 = int(interR1 + (yi * slopeR1))

            strtW = int(pointR0 - pointL0)
            sqzW = int(pointR1 - pointL1)
            '''
            strtPct_bfCross = (crossX0 - pointL0) / (pointR0 - pointL0)
            sqzL = crossX1 - (sqzW * strtPct_bfCross)
            sqzR = sqzL + sqzW
            '''
            imgMdli = imageCropMethod(image, strtW, 1, pointL0, y0 + yi)[0]  #(self, image, width, height, x, y)
            imgMdliSqz = imageScaleMethod(imgMdli, upscaleMethod, sqzW, 1, "disabled")[0]  #(self, image, upscale_method, width, height, crop)

            if pointL0 > 0:
                imgLfti = imageCropMethod(image, min(pointL0, pointL1), 1, pointL0 - min(pointL0, pointL1), y0 + yi)[0]
                addLft = imageCompositeMaskedMethod(imgFinal, imgLfti, max(0, pointL1 - pointL0) , y0 + yi, False, None)[0]  #(self, destination, source, x, y, resize_source, mask = None)
            else:
                addLft = imgFinal

            addMdlSqz = imageCompositeMaskedMethod(addLft, imgMdliSqz, pointL1 , y0 + yi, False, None)[0]

            if pointR0 < imgW:
                imgRgti = imageCropMethod(image, min(imgW - pointR0, imgW - pointR1), 1, pointR0, y0 + yi)[0]
                addRgt = imageCompositeMaskedMethod(addMdlSqz, imgRgti, pointR1 , y0 + yi, False, None)[0]
            else:
                addRgt = addMdlSqz

            imgFinal = addRgt

        return (imgFinal,)


mwImageVertTaper2Method = mwImageVertTaper2().mwIVTT

class mwImageVertTaper3:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "image": ("IMAGE", ),
            #"sCurveMult": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            "y0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "y1": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "y2": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xtl0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),            
            "xtr0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbl0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),   
            "xbr0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),            
            #"xtld": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),            
            #"xtrd": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
            "xbld": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),   
            "xbrd": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),     
            "upscaleMethod": (moh.upscale_methods,),
            }}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "mwIVTTH"
    CATEGORY = "mohwag/manip"
    #def mwIVTTH(self, image, y0, y1, y2, xtl0, xtr0, xbl0, xbr0, xtld, xtrd, xbld, xbrd, upscaleMethod):
    def mwIVTTH(self, image, y0, y1, y2, xtl0, xtr0, xbl0, xbr0, xbld, xbrd, upscaleMethod):
        xtld = int(0)
        xtrd = int(0)
        round0 = mwImageVertTaper2Method(image, y0, y1, xtl0, xtr0, xbl0, xbr0, xtld, xtrd, xbld, xbrd, upscaleMethod)[0]
        round1 = mwImageVertTaper2Method(round0, y1, y2, xbl0, xbr0, xtl0, xtr0, xbld, xbrd, xtld, xtrd, upscaleMethod)[0]
        return(round1,)

class mwImageVertTaper4:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "image": ("IMAGE", ),
            #"sCurveMult": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            "y0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xtl": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xtm0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xtr": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "y1": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),  
            "xbl": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbm0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbr": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),            
            "xtmd": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),            
            "xbmd": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),   
            "upscaleMethod": (moh.upscale_methods,),
            }}

    RETURN_TYPES = ("IMAGE", "MWPIPEIVTFOP")
    RETURN_NAMES = ("image", "ivt4Pipe")
    FUNCTION = "mwIVTFO"
    CATEGORY = "mohwag/manip"

    def mwIVTFO(self, image, y0, xtl, xtm0, xtr, y1, xbl, xbm0, xbr, xtmd, xbmd, upscaleMethod):
        xtm1 = xtm0 + xtmd
        xbm1 = xbm0 + xbmd

        modImage = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        imgW = int(modImage.size[0])

        midH = int(y1 - y0)

        slopeL = (xbl - xtl) / (midH)
        slopeM0 = (xbm0 - xtm0) / (midH)
        slopeR = (xbr - xtr) / (midH)
        slopeM1 = (xbm1 - xtm1) / (midH)

        interL = xtl
        interM0 = xtm0
        interR = xtr
        interM1 = xtm1

        imgFinal = image

        for yi in range(midH):
            pointL = int(interL + (yi * slopeL))
            pointM0 = int(interM0 + (yi * slopeM0))
            pointR = int(interR + (yi * slopeR))
            pointM1 = int(interM1 + (yi * slopeM1))

            strtWL = int(pointM0 - pointL)
            strtWR = int(pointR - pointM0)
            sqzWL = int(pointM1 - pointL)
            sqzWR = int(pointR - pointM1)

            imgMdlLi = imageCropMethod(image, strtWL, 1, pointL, y0 + yi)[0]  #(self, image, width, height, x, y)
            imgMdlRi = imageCropMethod(image, strtWR, 1, pointM0, y0 + yi)[0]
            imgMdlLiSqz = imageScaleMethod(imgMdlLi, upscaleMethod, sqzWL, 1, "disabled")[0]  #(self, image, upscale_method, width, height, crop)
            imgMdlRiSqz = imageScaleMethod(imgMdlRi, upscaleMethod, sqzWR, 1, "disabled")[0]

            addMdlLSqz = imageCompositeMaskedMethod(imgFinal, imgMdlLiSqz, pointL , y0 + yi, False, None)[0]
            addMdlRSqz = imageCompositeMaskedMethod(addMdlLSqz, imgMdlRiSqz, pointM1 , y0 + yi, False, None)[0]
            imgFinal = addMdlRSqz

        out_ivt4Pipe = y1, xbl, xbm0, xbr, xbmd
        return (imgFinal, out_ivt4Pipe)

mwImageVertTaper4Method = mwImageVertTaper4().mwIVTFO

class mwImageVertTaper4p:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "image": ("IMAGE", ),
            "ivt4Pipe": ("MWPIPEIVTFOP",),
            "y1": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),  
            "xbl": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbm0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbr": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbmd": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
            "upscaleMethod": (moh.upscale_methods,),
            }}
    RETURN_TYPES = ("IMAGE", "MWPIPEIVTFOP")
    RETURN_NAMES = ("image", "ivt4Pipe")
    FUNCTION = "mwIVTFOP"
    CATEGORY = "mohwag/manip"
    #def mwIVTTH(self, image, y0, y1, y2, xtl0, xtr0, xbl0, xbr0, xtld, xtrd, xbld, xbrd, upscaleMethod):
    def mwIVTFOP(self, image, ivt4Pipe, y1, xbl, xbm0, xbr, xbmd, upscaleMethod):
        y0, xtl, xtm0, xtr, xtmd = ivt4Pipe
        out_taper = mwImageVertTaper4Method(image, y0, xtl, xtm0, xtr, y1, xbl, xbm0, xbr, xtmd, xbmd, upscaleMethod)[0]
        out_ivt4Pipe = y1, xbl, xbm0, xbr, xbmd
        return(out_taper, out_ivt4Pipe)


class mwImageVertTaper5:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "image": ("IMAGE", ),
            #"sCurveMult": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            "y0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xtl": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xtm0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xtr": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "y1": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),  
            "xbl": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbm0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbr": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "y2": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            #"xtmd": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
            "xbmd": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
            "upscaleMethod": (moh.upscale_methods,),
            }}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "mwIVTFI"
    CATEGORY = "mohwag/manip"
    #def mwIVTTH(self, image, y0, y1, y2, xtl0, xtr0, xbl0, xbr0, xtld, xtrd, xbld, xbrd, upscaleMethod):
    def mwIVTFI(self, image, y0, xtl, xtm0, xtr, y1, xbl, xbm0, xbr, y2, xbmd, upscaleMethod):
        xtmd = int(0)
        round0 = mwImageVertTaper4Method(image,  y0, xtl, xtm0, xtr, y1, xbl, xbm0, xbr, xtmd, xbmd, upscaleMethod)[0]
        round1 = mwImageVertTaper4Method(round0, y1, xbl, xbm0, xbr, y2, xtl, xtm0, xtr, xbmd, xtmd, upscaleMethod)[0]
        return(round1,)


class mwImageVertTaper6:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "image": ("IMAGE", ),
            #"sCurveMult": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            "y0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xtl": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xtml0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xtmr0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xtr": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "y1": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),  
            "xbl": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbml0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbmr0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbr": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),            
            "xtmld": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
            "xtmrd": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),            
            "xbmld": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
            "xbmrd": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
            "upscaleMethod": (moh.upscale_methods,),
            }}

    RETURN_TYPES = ("IMAGE", "MWPIPEIVTSIP")
    RETURN_NAMES = ("image", "ivt6Pipe")
    FUNCTION = "mwIVTSI"
    CATEGORY = "mohwag/manip"

    def mwIVTSI(self, image, y0, xtl, xtml0, xtmr0, xtr, y1, xbl, xbml0, xbmr0, xbr, xtmld, xtmrd, xbmld, xbmrd, upscaleMethod):
        xtml1 = xtml0 + xtmld
        xtmr1 = xtmr0 + xtmrd
        xbml1 = xbml0 + xbmld
        xbmr1 = xbmr0 + xbmrd

        modImage = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        imgW = int(modImage.size[0])

        midH = int(y1 - y0)

        slopeL = (xbl - xtl) / (midH)
        slopeML0 = (xbml0 - xtml0) / (midH)
        slopeMR0 = (xbmr0 - xtmr0) / (midH)
        slopeR = (xbr - xtr) / (midH)
        slopeML1 = (xbml1 - xtml1) / (midH)
        slopeMR1 = (xbmr1 - xtmr1) / (midH)

        interL = xtl
        interML0 = xtml0
        interMR0 = xtmr0
        interR = xtr
        interML1 = xtml1
        interMR1 = xtmr1

        imgFinal = image

        for yi in range(midH):
            pointL = int(interL + (yi * slopeL))
            pointML0 = int(interML0 + (yi * slopeML0))
            pointMR0 = int(interMR0 + (yi * slopeMR0))
            pointR = int(interR + (yi * slopeR))
            pointML1 = int(interML1 + (yi * slopeML1))
            pointMR1 = int(interMR1 + (yi * slopeMR1))

            strtWL = int(pointML0 - pointL)
            strtWM = int(pointMR0 - pointML0)
            strtWR = int(pointR - pointMR0)
            sqzWL = int(pointML1 - pointL)
            sqzWM = int(pointMR1 - pointML1)
            sqzWR = int(pointR - pointMR1)

            imgMdlLi = imageCropMethod(image, strtWL, 1, pointL, y0 + yi)[0]  #(self, image, width, height, x, y)
            imgMdlMi = imageCropMethod(image, strtWM, 1, pointML0, y0 + yi)[0]
            imgMdlRi = imageCropMethod(image, strtWR, 1, pointMR0, y0 + yi)[0]
            imgMdlLiSqz = imageScaleMethod(imgMdlLi, upscaleMethod, sqzWL, 1, "disabled")[0]  #(self, image, upscale_method, width, height, crop)
            imgMdlMiSqz = imageScaleMethod(imgMdlMi, upscaleMethod, sqzWM, 1, "disabled")[0]
            imgMdlRiSqz = imageScaleMethod(imgMdlRi, upscaleMethod, sqzWR, 1, "disabled")[0]

            addMdlLSqz = imageCompositeMaskedMethod(imgFinal, imgMdlLiSqz, pointL , y0 + yi, False, None)[0]
            addMdlMSqz = imageCompositeMaskedMethod(addMdlLSqz, imgMdlMiSqz, pointML1 , y0 + yi, False, None)[0]
            addMdlRSqz = imageCompositeMaskedMethod(addMdlMSqz, imgMdlRiSqz, pointMR1 , y0 + yi, False, None)[0]
            imgFinal = addMdlRSqz

        out_ivt6Pipe = y1, xbl, xbml0, xbmr0, xbr, xbmld, xbmrd
        return (imgFinal, out_ivt6Pipe)

mwImageVertTaper6Method = mwImageVertTaper6().mwIVTSI

class mwImageVertTaper6p:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "image": ("IMAGE",),
            "ivt6Pipe": ("MWPIPEIVTSIP",),
            "y1": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),  
            "xbl": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbml0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbmr0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbr": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),            
            "xbmld": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
            "xbmrd": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
            "upscaleMethod": (moh.upscale_methods,),
            }}
    RETURN_TYPES = ("IMAGE", "MWPIPEIVTSIP")
    RETURN_NAMES = ("image", "ivt6Pipe")
    FUNCTION = "mwIVTSIP"
    CATEGORY = "mohwag/manip"
    def mwIVTSIP(self, image, ivt6Pipe, y1, xbl, xbml0, xbmr0, xbr, xbmld, xbmrd, upscaleMethod):
        y0, xtl, xtml0, xtmr0, xtr, xtmld, xtmrd = ivt6Pipe
        out_taper = mwImageVertTaper6Method(image, y0, xtl, xtml0, xtmr0, xtr, y1, xbl, xbml0, xbmr0, xbr, xtmld, xtmrd, xbmld, xbmrd, upscaleMethod)[0]
        out_ivt6Pipe = y1, xbl, xbml0, xbmr0, xbr, xbmld, xbmrd
        return(out_taper, out_ivt6Pipe)


class mwImageVertTaper7:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "image": ("IMAGE", ),
            #"sCurveMult": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            "y0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xtl": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xtml0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xtmr0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xtr": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "y1": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),  
            "xbl": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbml0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbmr0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbr": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),            
            #"xtmld": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
            #"xtmrd": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
            "y2": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbmld": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
            "xbmrd": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
            "upscaleMethod": (moh.upscale_methods,),
            }}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "mwIVTSE"
    CATEGORY = "mohwag/manip"
    #def mwIVTTH(self, image, y0, y1, y2, xtl0, xtr0, xbl0, xbr0, xtld, xtrd, xbld, xbrd, upscaleMethod):
    def mwIVTSE(self, image, y0, xtl, xtml0, xtmr0, xtr, y1, xbl, xbml0, xbmr0, xbr, y2, xbmld, xbmrd, upscaleMethod):
        xtmld = int(0)
        xtmrd = int(0)
        round0 = mwImageVertTaper6Method(image,  y0, xtl, xtml0, xtmr0, xtr, y1, xbl, xbml0, xbmr0, xbr, xtmld, xtmrd, xbmld, xbmrd, upscaleMethod)[0]
        round1 = mwImageVertTaper6Method(round0, y1, xbl, xbml0, xbmr0, xbr, y2, xtl, xtml0, xtmr0, xtr, xbmld, xbmrd, xtmld, xtmrd, upscaleMethod)[0]
        return(round1,)



class mwLtntPipe_VertTaper:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "ltntPipe": ("MWLTNTPIPE", ),
            "x0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),            
            "y0": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "x1": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),            
            "y1": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xtl": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),            
            "xtr": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "xbl": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),            
            "xbr": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "cropToOutput_upscaleMethod": (moh.upscale_methods,),
            }}

    RETURN_TYPES = ("MWLTNTPIPE",)
    RETURN_NAMES = ("ltntPipe",)
    FUNCTION = "mwLPVT"
    CATEGORY = "mohwag/manip"

    def mwLPVT(self, ltntPipe, x0, y0, x1, y1, xtl, xtr, xbl, xbr, cropToOutput_upscaleMethod):

        midW = x1 - x0
        midW_E = midW //8
        midH = y1 - y0
        midH_E = midH //8
        x0_E = x0 //8
        y0_E = y0 //8
        x1_E = x1 //8
        y1_E = y1 //8

        ltnt, startW, startH, inputW, inputH = ltntPipe
        ltntSamples = ltnt['samples']
        
        ltntMid0 = ltnt.copy()
        ltntMid0['samples'] = ltntSamples[:,:,y0_E:y1_E, x0_E:x1_E]
        ltntMid0Samples = ltntMid0["samples"]

        if x0 > 0:
            ltntLft0 = ltnt.copy()
            ltntLft0["samples"] = ltntSamples[:,:,y0_E:y1_E, 0:x0_E]
            ltntLft0Samples = ltntLft0["samples"]

        if x1 < inputW:
            ltntRgt0 = ltnt.copy()
            ltntRgt0["samples"] = ltntSamples[:,:,y0_E:y1_E, x1_E:inputW//8]
            ltntRgt0Samples = ltntRgt0["samples"]



        slopeL = (xbl - xtl) / (midH)
        slopeR = (xbr - xtr) / (midH)

        interL = xtl - x0
        interR = xtr - x0

        ltntFinal = ltnt.copy()

        for yi in range(midH_E):
            ltntMidi = ltntMid0.copy()
            ltntMidi['samples'] = ltntMid0Samples[:,:,int(yi):int(yi+1), int(0):int(midW_E)]
            #ltntMidiSamples = ltntMidi["samples"]

            pointL = interL + (yi * 8 * slopeL)

            pointR = interR + (yi * 8 * slopeR)

            sqzW = int(pointR - pointL)

            ltntMidiSqz = ltntMidi.copy()
            ltntMidiSqz['samples'] = comfy.utils.common_upscale(ltntMidi["samples"], int(max(1,sqzW//8)), int(1), cropToOutput_upscaleMethod, "disabled") ####

            if x0 > 0:
                if pointL < 0:
                    ltntLfti = ltntLft0.copy()
                    ltntLfti["samples"] = ltntLft0Samples[:,:,int(yi):int(yi+1), int(-1*pointL//8):int(x0_E)]
                else:
                    ltntLfti = ltntLft0.copy()

            if x1 < inputW:
                if pointR > midW:
                    ltntRgti = ltntRgt0.copy()
                    ltntRgti["samples"] = ltntRgt0Samples[:,:,int(yi):int(yi+1), int(x1_E):int((inputW - pointR)//8)]
                else:
                    ltntRgti = ltntRgt0.copy()

            if x0 > 0:
                if pointL <= 0:  #latentCompositeMaskedMethod = LatentCompositeMasked().composite  #(self, destination, source, x, y, resize_source, mask = None):
                    addLeft = latentCompositeMaskedMethod(ltntFinal, ltntLfti, int(0), int(y0 + (yi * 8)), False, None)[0]
                else:
                    addLeft = latentCompositeMaskedMethod(ltntFinal, ltntLfti, int(pointL), int(y0 + (yi * 8)), False, None)[0]
            else:
                addLeft = ltntFinal.copy()

            addMidSqz = latentCompositeMaskedMethod(addLeft, ltntMidiSqz, int(x0 + pointL), int(y0 + (yi * 8)), False, None)[0]

            if x1 < inputW:
                addRgt = latentCompositeMaskedMethod(addMidSqz, ltntRgti, int(x0 + pointR), int(y0 + (yi * 8)), False, None)[0]
                ltntFinal = addRgt.copy()
            else:
                ltntFinal = addMidSqz.copy()

        ltntPipeNew = ltntFinal, startW, startH, inputW, inputH

        return (ltntPipeNew,)

class mwLtntPipeBranch1: #WAS_Latent_Size_To_Number
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "ltntPipe": ("MWLTNTPIPE", ),
            }}
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent", "sPipe_start", "sPipe_curr")
    FUNCTION = "mwLPBO"
    CATEGORY = "mohwag/LatentPipe"
    def mwLPBO(self, ltntPipe):
        ltnt, initialW, initialH, outputW, outputH = ltntPipe
        return (ltnt, (initialW, initialH), (outputW, outputH))
    

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
            "refImgsPixelMult": ("FLOAT", {"default": 1.00, "min": 0, "max": 8, "step": 0.05}),  
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
        
        actualW, actualH = sPipe_actual

        #GET START IMAGE SIZE
        ##start ratio and size before upscaling
        rats = round((np.exp(1))**(np.log(actualW / actualH) * (2/3)),2)
        testListrs = [abs(round(x - rats,3)) for x in aspectRatList]
        testMinrs = min(testListrs)
        testMinLocrs = testListrs.index(testMinrs)
        startW = targetW_list[testMinLocrs]
        startH = targetH_list[testMinLocrs]

        ##start ratio and size final (i.e. w/ upscaling)
        #actualPx = actualW * actualH
        actualPx = startW * startH
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
            testListr = [abs(round(x - rat,3)) for x in aspectRatList]
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
                    "startS": ("MWSIZEPIPE",),
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
        #diffusion run
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        ltnt = common_ksampler(model1, seed, steps, cfg, sampler_name, scheduler, positive, negative, latentDictTuple[0], denoise=denoise, disable_noise=disable_noise, start_step=stepStart, last_step=stepEnd, force_full_denoise=force_full_denoise)[0]
        ltntOut = (ltnt, )
        img = vaeAlways.decode(ltnt["samples"])
        imgOut = (img, )
        ltntPipe = ltntOut[0], startW, startH, startW, startH
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
                    "upscale_method": (moh.upscale_methods,),
                     },}
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
            }}
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
            }}
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
            }}
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
            }}
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
            }}
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
            }}
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
            }}
    RETURN_TYPES = ("MWSIZEPIPE",)
    RETURN_NAMES = ("sPipe",)
    FUNCTION = "mwNI"
    CATEGORY = "mohwag/psPipe"
    def mwNI(self, width, height):
        sPipeOut = width, height
        return (sPipeOut,)


class mwStartSizeXL:
    xlSelect = ["0.25 | 512x2048", "0.26 | 512x1984", "0.27 | 512x1920", "0.28 | 512x1856", "0.32 | 576x1792", "0.33 | 576x1728", "0.35 | 576x1664", "0.40 | 640x1600", "0.42 | 640x1536", "0.48 | 704x1472", "0.50 | 704x1408", "0.52 | 704x1344", "0.57 | 768x1344", "0.60 | 768x1280", "0.68 | 832x1216", "0.72 | 832x1152", "0.78 | 896x1152", "0.82 | 896x1088", "0.88 | 960x1088", "0.94 | 960x1024", "1.00 | 1024x1024", "1.07 | 1024x960", "1.13 | 1088x960", "1.21 | 1088x896", "1.29 | 1152x896", "1.38 | 1152x832", "1.46 | 1216x832", "1.67 | 1280x768", "1.75 | 1344x768", "2.00 | 1408x704", "2.09 | 1472x704", "2.40 | 1536x640", "2.50 | 1600x640", "2.89 | 1664x576", "3.00 | 1728x576", "3.11 | 1792x576", "3.62 | 1856x512", "3.75 | 1920x512", "3.88 | 1984x512", "4.00 | 2048x512"]
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "dimSelect": (moh.xlSelect,),
            }}
    RETURN_TYPES = ("MWSIZEPIPE",)
    RETURN_NAMES = ("sPipe",)
    FUNCTION = "mwSSXL"
    CATEGORY = "mohwag/psPipe"
    def mwSSXL(self, dimSelect):
        listSelect = ["0.25 | 512x2048", "0.26 | 512x1984", "0.27 | 512x1920", "0.28 | 512x1856", "0.32 | 576x1792", "0.33 | 576x1728", "0.35 | 576x1664", "0.40 | 640x1600", "0.42 | 640x1536", "0.48 | 704x1472", "0.50 | 704x1408", "0.52 | 704x1344", "0.57 | 768x1344", "0.60 | 768x1280", "0.68 | 832x1216", "0.72 | 832x1152", "0.78 | 896x1152", "0.82 | 896x1088", "0.88 | 960x1088", "0.94 | 960x1024", "1.00 | 1024x1024", "1.07 | 1024x960", "1.13 | 1088x960", "1.21 | 1088x896", "1.29 | 1152x896", "1.38 | 1152x832", "1.46 | 1216x832", "1.67 | 1280x768", "1.75 | 1344x768", "2.00 | 1408x704", "2.09 | 1472x704", "2.40 | 1536x640", "2.50 | 1600x640", "2.89 | 1664x576", "3.00 | 1728x576", "3.11 | 1792x576", "3.62 | 1856x512", "3.75 | 1920x512", "3.88 | 1984x512", "4.00 | 2048x512"]
        targetW_list = [512, 512, 512, 512, 576, 576, 576, 640, 640, 704, 704, 704, 768, 768, 832, 832, 896, 896, 960, 960, 1024, 1024, 1088, 1088, 1152, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048]
        targetH_list = [2048, 1984, 1920, 1856, 1792, 1728, 1664, 1600, 1536, 1472, 1408, 1344, 1344, 1280, 1216, 1152, 1152, 1088, 1088, 1024, 1024, 960, 960, 896, 896, 832, 832, 768, 768, 704, 704, 640, 640, 576, 576, 576, 512, 512, 512, 512]
        listLoc = listSelect.index(dimSelect)
        width = targetW_list[listLoc]
        height = targetH_list[listLoc]
        sPipeOut = width, height
        return (sPipeOut,)


class mwNumIntBinaryIsh:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "x1024": ("INT", {"default": 0, "min": -16, "max": 16, "step": 1}),
                "x256": ("INT", {"default": 0, "min": -16, "max": 16, "step": 1}),
                "x64": ("INT", {"default": 0, "min": -16, "max": 16, "step": 1}),
                "x16": ("INT", {"default": 0, "min": -16, "max": 16, "step": 1}),
                "x8": ("INT", {"default": 0, "min": -16, "max": 16, "step": 1}),
                "x1": ("INT", {"default": 0, "min": -16, "max": 16, "step": 1}),
            }
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "mwNIBI"
    CATEGORY = "mohwag/Int"
    def mwNIBI(self, x1024, x256, x64, x16, x8, x1):
        rslt = int((1024 * x1024) + (256 * x256) + (64 * x64) + (16 * x16) + (8 * x8) + (x1))
        return (rslt,)

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

class mwpsPipeCreate:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "x": ("INT", {"default": 0, "min": -10000, "max": 10000}),
                "y": ("INT", {"default": 0, "min": -10000, "max": 10000}),
                "width": ("INT", {"default": 1024, "min": -10000, "max": 10000}),
                "height": ("INT", {"default": 1024, "min": -10000, "max": 10000}),
            }
        }
    RETURN_TYPES = ("MWPOSSIZEPIPE",)
    RETURN_NAMES = ("psPipe",)
    FUNCTION = "mwPSPC"
    CATEGORY = "mohwag/psPipe"

    def mwPSPC(self, x, y, width, height):
        psPipeOut = x, y, width, height
        return (psPipeOut,) 

class mwpsPipeCreate2:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "x": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "y": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "sPipe": ("MWSIZEPIPE",),
            }
        }
    RETURN_TYPES = ("MWPOSSIZEPIPE",)
    RETURN_NAMES = ("psPipe",)
    FUNCTION = "mwPSPCT"
    CATEGORY = "mohwag/psPipe"
    def mwPSPCT(self, x, y, sPipe):
        w, h = sPipe
        psPipeOut = x, y, w, h
        return (psPipeOut,) 

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

class mwpsPipeBranch2:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "psPipe": ("MWPOSSIZEPIPE",),
            }}
    RETURN_TYPES = ("MWSIZEPIPE", "INT", "INT")
    RETURN_NAMES = ("sPipe", "x", "y")
    FUNCTION = "mwSPBT"
    CATEGORY = "mohwag/psPipe"
    def mwSPBT(self, psPipe):
        x, y, w, h = psPipe
        sPipeOut = w, h
        return (sPipeOut, x , y)

class mwImageTo_sPipe:   #WAS
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "image": ("IMAGE",),
            }}
    RETURN_TYPES = ("MWSIZEPIPE",)
    RETURN_NAMES = ("sPipe",)
    FUNCTION = "mwITSP"
    CATEGORY = "mohwag/psPipe"
    def mwITSP(self, image):
        modImage = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        return((modImage.size[0], modImage.size[1]),)


class mwMultXY_divZ:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "x": ("INT", {"default": 1, "min": -10000, "max": 10000, "step": 1}),
                "y": ("INT", {"default": 1, "min": -10000, "max": 10000, "step": 1}),
                "z": ("INT", {"default": 1, "min": -10000, "max": 10000, "step": 1}),
            }}
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "mwMTH"
    CATEGORY = "mohwag/scaling"
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
    CATEGORY = "mohwag/scaling"

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
            }}
    RETURN_TYPES = ("MWSIZEPIPE", "INT", "MWSIZEPIPE",)
    RETURN_NAMES = ("sPipe_start", "nextMO8", "sPipe_next",)
    FUNCTION = "mwNSP"
    CATEGORY = "mohwag/scaling"
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
                "psPipe": ("MWPOSSIZEPIPE",),
                "multOver8": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "nextMO8": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
            }}
    RETURN_TYPES = ("MWSIZEPIPE", "MWPOSSIZEPIPE")
    RETURN_NAMES = ("sPipe", "psPipe")
    FUNCTION = "mwCS"
    CATEGORY = "mohwag/scaling"
    def mwCS(self, psPipe, multOver8, nextMO8):
        x, y, w, h = psPipe
        upscaleX = x * nextMO8 / multOver8
        upscaleY = y * nextMO8 / multOver8
        upscaleW = w * nextMO8 / multOver8
        upscaleH = h * nextMO8 / multOver8
        upscaleX2 = upscaleX + upscaleW
        upscaleY2 = upscaleY + upscaleH

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
            }}
    RETURN_TYPES = ("MWSIZEPIPE", "MWPOSSIZEPIPE", "MASK", "IMAGE")
    RETURN_NAMES = ("sPipe", "psPipe", "bounded mask", "bounded image")
    FUNCTION = "mwMBB"
    CATEGORY = "mohwag/mask"
    def mwMBB(self, mask_bounding_box, image_mapped, threshold, deltaLft_units, deltaRgt_units, deltaTop_units, deltaBtm_units):
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


def conformSizeParams(aimgD, amultReq, ad0, ad1):
    if (ad0 <= 0 and ad1 >= aimgD):
        absOvr0 = 0 - ad0
        absOvr1 = ad1 - aimgD
        amaxD = amultReq * (aimgD //amultReq)
        if absOvr0 > absOvr1:
            ad0_1 = 0
            ad1_1 = amaxD
        else:
            ad0_1 = aimgD - amaxD
            ad1_1 = aimgD
    elif (ad0 <= 0):
        ad0_1 = 0
        ad1_1 = amultReq * (ad1 //amultReq)
    elif (ad1 >= aimgD):
        ad0_1 = aimgD - (amultReq * ((aimgD - ad0) //amultReq))
        ad1_1 = aimgD
    else:
        amodD = amultReq * ((ad1 - ad0) //amultReq)
        amodD_half = amodD //2
        amidD = (ad0 + ad1) //2
        ad0_1 = amidD - amodD_half
        ad1_1 = amidD + amodD_half
    ad0_f = int(ad0_1)
    ad1_f = int(ad1_1)
    return (ad0_f, ad1_f)

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
            }}
    RETURN_TYPES = ("MWSIZEPIPE", "MWPOSSIZEPIPE", "MASK", "IMAGE")
    RETURN_NAMES = ("sPipe", "psPipe", "bounded mask", "bounded image")
    FUNCTION = "mwMBBS"
    CATEGORY = "mohwag/mask"
    def mwMBBS(self, mask_bounding_box, image_mapped, threshold, deltaLft_units, deltaRgt_units, deltaTop_units, deltaBtm_units):
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

        x0_1 = x0i + (deltaLft_units * unitS)
        x1_1 = x1i + (deltaRgt_units * unitS)
        y0_1 = y0i + (deltaTop_units * unitS) 
        y1_1 = y1i + (deltaBtm_units * unitS)

        x0_2, x1_2 = conformSizeParams(imgW, multReq, x0_1, x1_1)
        y0_2, y1_2 = conformSizeParams(imgH, multReq, y0_1, y1_1)

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
    

class mwTestMaskConvert:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "stndMask": ("MASK",),
            }}
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("cnvtdMask",)
    FUNCTION = "mwTMC"
    CATEGORY = "mohwag/tester"
    def mwTMC(self, stndMask):
        outMask = stndMask[0]
        return (outMask,)

NODE_CLASS_MAPPINGS = {
"mwGridOverlay": mwGridOverlay,
"mwMaskSegment": mwMaskSegment,
"mwMaskCompSameSize": mwMaskCompSameSize,
"mwMaskTweak": mwMaskTweak,
"mwMaskStack": mwMaskStack,
"mwImageCompositeMasked": mwImageCompositeMasked,
"mwImageScale": mwImageScale,
"mwImageCrop": mwImageCrop,
"mwImageCropwParams": mwImageCropwParams,
"mwImageConform_StartSizeXL": mwImageConform_StartSizeXL,
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
"mwLtntPipe_Create2": mwLtntPipe_Create2,
"mwLtntPipe_CropScale": mwLtntPipe_CropScale,
"mwLtntPipe_VertSqueeze": mwLtntPipe_VertSqueeze,
"mwLtntPipe_VertSqzExpnd": mwLtntPipe_VertSqzExpnd,
"mwImage_VertSqzExpnd": mwImage_VertSqzExpnd,
"mwImageVertTaper": mwImageVertTaper,
"mwImageVertTaper2": mwImageVertTaper2,
"mwImageVertTaper3": mwImageVertTaper3,
"mwImageVertTaper4": mwImageVertTaper4,
"mwImageVertTaper4p": mwImageVertTaper4p,
"mwImageVertTaper5": mwImageVertTaper5,
"mwImageVertTaper6": mwImageVertTaper6,
"mwImageVertTaper6p": mwImageVertTaper6p,
"mwImageVertTaper7": mwImageVertTaper7,
"mwLtntPipe_VertTaper": mwLtntPipe_VertTaper,
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
"mwStartSizeXL": mwStartSizeXL,
"mwNumIntBinaryIsh": mwNumIntBinaryIsh,
"mwsPipeCreate": mwsPipeCreate,
"mwpsPipeCreate": mwpsPipeCreate,
"mwpsPipeCreate2": mwpsPipeCreate2,
"mwsPipeBranch": mwsPipeBranch,
"mwpsPipeBranch": mwpsPipeBranch,
"mwpsPipeBranch2": mwpsPipeBranch2,
"mwImageTo_sPipe": mwImageTo_sPipe,
"mwMultXY_divZ": mwMultXY_divZ,
"mwsPipe_MultOver8": mwsPipe_MultOver8,
"mwsPipe_NextMO8": mwsPipe_NextMO8,
"mwCompScale": mwCompScale,
"mwMaskBoundingBoxRF": mwMaskBoundingBoxRF,
"mwMaskBoundingBoxRF64": mwMaskBoundingBoxRF64,
"mwTestMaskConvert": mwTestMaskConvert,
}

NODE_DISPLAY_NAME_MAPPINGS = {
"mwGridOverlay": "GridOverlay",
"mwMaskSegment": "MaskSegment",
"mwMaskCompSameSize": "MaskCompSameSize",
"mwMaskTweak": "MaskTweak",
"mwMaskStack": "MaskStack",
"mwImageCompositeMasked": "mwImageCompositeMasked",
"mwImageScale": "ImageScale",
"mwImageCrop": "ImageCrop",
"mwImageCropwParams": "ImageCropwParams",
"mwImageConform_StartSizeXL": "ImageConform_StartSizeXL",
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
"mwLtntPipe_Create2": "LtntPipe_Create2",
"mwLtntPipe_CropScale": "LtntPipe_CropScale",
"mwLtntPipe_VertSqueeze": "LtntPipe_VertSqueeze",
"mwLtntPipe_VertSqzExpnd": "LtntPipe_VertSqzExpnd",
"mwImage_VertSqzExpnd": "Image_VertSqzExpnd",
"mwImageVertTaper": "ImageVertTaper",
"mwImageVertTaper2": "ImageVertTaper2",
"mwImageVertTaper3": "ImageVertTaper3",
"mwImageVertTaper4": "ImageVertTaper4",
"mwImageVertTaper4p": "ImageVertTaper4p",
"mwImageVertTaper5": "ImageVertTaper5",
"mwImageVertTaper6": "ImageVertTaper6",
"mwImageVertTaper6p": "ImageVertTaper6p",
"mwImageVertTaper7": "ImageVertTaper7",
"mwLtntPipe_VertTaper": "LtntPipe_VertTaper",
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
"mwStartSizeXL": "StartSizeXL",
"mwNumIntBinaryIsh": "NumIntBinaryIsh",
"mwsPipeCreate": "sPipeCreate",
"mwpsPipeCreate": "psPipeCreate",
"mwpsPipeCreate2": "psPipeCreate2",
"mwsPipeBranch": "sPipeBranch",
"mwpsPipeBranch": "psPipeBranch",
"mwpsPipeBranch2": "psPipeBranch2",
"mwImageTo_sPipe": "ImageTo_sPipe",
"mwMultXY_divZ": "MultXY_divZ",
"mwsPipe_MultOver8": "sPipe_MultOver8",
"mwsPipe_NextMO8": "sPipe_NextMO8",
"mwCompScale": "CompScale",
"mwMaskBoundingBoxRF": "MaskBoundingBoxRF",
"mwMaskBoundingBoxRF64": "MaskBoundingBoxRF64",
"mwTestMaskConvert": "TestMaskConvert",
}