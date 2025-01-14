import torch

import os
import sys
import json
import hashlib
import traceback
import math
import time
import random
import copy

from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageSequence
from PIL.PngImagePlugin import PngInfo
import numpy as np
import scipy.ndimage
import safetensors.torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import node_helpers
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
from nodes import MAX_RESOLUTION, ConditioningSetArea, CheckpointLoaderSimple, ImageScale, EmptyLatentImage, ConditioningCombine, ConditioningAverage, ConditioningConcat, ConditioningSetTimestepRange, ConditioningSetAreaStrength, CLIPSetLastLayer, ImageBatch, ControlNetLoader, ControlNetApplyAdvanced, LatentUpscale, VAEEncode, common_ksampler
condSetAreaMethod = ConditioningSetArea().append
imageScaleMethod = ImageScale().upscale  #(self, image, upscale_method, width, height, crop)
emptyLatentImageMethod = EmptyLatentImage().generate  #(self, width, height, batch_size=1)
conditioningCombineMethod = ConditioningCombine().combine  #(self, conditioning_1, conditioning_2)
conditioningAverageMethod = ConditioningAverage().addWeighted  #(self, conditioning_to, conditioning_from, conditioning_to_strength)
conditioningConcatMethod = ConditioningConcat().concat  #(self, conditioning_to, conditioning_from)
conditioningSetTimestepRangeMethod = ConditioningSetTimestepRange().set_range  
conditioningSetAreaStrengthMethod = ConditioningSetAreaStrength().append  #(self, conditioning, strength)
clipSetLastLayerMethod = CLIPSetLastLayer().set_last_layer  #(self, clip, stop_at_clip_layer)
imageBatchMethod = ImageBatch().batch  #(self, image1, image2)
controlNetLoaderMethod = ControlNetLoader().load_controlnet  #(self, control_net_name)
controlNetApplyAdvancedMethod = ControlNetApplyAdvanced().apply_controlnet  #(self, positive, negative, control_net, image, strength, start_percent, end_percent, vae=None)
latentUpscaleMethod = LatentUpscale().upscale  #(self, samples, upscale_method, width, height, crop)
vaeEncodeMethod = VAEEncode().encode  #(self, vae, pixels)

from comfy_extras.nodes_model_merging import ModelMergeSimple, CLIPMergeSimple
modelMergeMethod = ModelMergeSimple().merge
clipMergeMethod = CLIPMergeSimple().merge
ckptLoaderSimpleMethod = CheckpointLoaderSimple().load_checkpoint

from comfy_extras.nodes_mask import SolidMask, MaskComposite, FeatherMask, InvertMask, ImageCompositeMasked, MaskToImage, LatentCompositeMasked, ImageToMask  #composite, 
solidMaskMethod = SolidMask().solid  #(self, value, width, height)
maskCompositeMethod = MaskComposite().combine  #(self, destination, source, x, y, operation)
featherMaskMethod = FeatherMask().feather  #(self, mask, left, top, right, bottom)
invertMaskMethod = InvertMask().invert  #(self, mask)
imageCompositeMaskedMethod = ImageCompositeMasked().composite  #(self, destination, source, x, y, resize_source, mask = None)
maskToImageMethod = MaskToImage().mask_to_image  #(self, mask)
latentCompositeMaskedMethod = LatentCompositeMasked().composite  #(self, destination, source, x, y, resize_source, mask = None):
imageToMaskMethod = ImageToMask().image_to_mask  #(self, image, channel = "green")

from comfy_extras.nodes_images import ImageCrop, ImageFromBatch
imageCropMethod = ImageCrop().crop  #(self, image, width, height, x, y)
imageFromBatchMethod = ImageFromBatch().frombatch  #self, image, batch_index, length

from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXL
clipTextEncodeSDXL_Method = CLIPTextEncodeSDXL().encode  #(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l)

clipsegPathFix = importlib.import_module('custom_nodes.ComfyUI-CLIPSeg.custom_nodes.clipseg') #self, image: torch.Tensor, text: str, blur: float, threshold: float, dilation_factor: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
clipSegMethod = clipsegPathFix.CLIPSeg().segment_image    #clipsegPathFix().segment_image


def sdxl_cond_sizes(asPipe_actual, arefImgsPixelMult):
    aspectRatList = [0.25, 0.26, 0.27, 0.28, 0.32, 0.33, 0.35, 0.4, 0.42, 0.48, 0.5, 0.52, 0.57, 0.6, 0.68, 0.72, 0.78, 0.82, 0.88, 0.94, 1, 1.07, 1.13, 1.21, 1.29, 1.38, 1.46, 1.67, 1.75, 2, 2.09, 2.4, 2.5, 2.89, 3, 3.11, 3.63, 3.75, 3.88, 4]
    targetW_list = [512, 512, 512, 512, 576, 576, 576, 640, 640, 704, 704, 704, 768, 768, 832, 832, 896, 896, 960, 960, 1024, 1024, 1088, 1088, 1152, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048]
    targetH_list = [2048, 1984, 1920, 1856, 1792, 1728, 1664, 1600, 1536, 1472, 1408, 1344, 1344, 1280, 1216, 1152, 1152, 1088, 1088, 1024, 1024, 960, 960, 896, 896, 832, 832, 768, 768, 704, 704, 640, 640, 576, 576, 576, 512, 512, 512, 512]

    actualW, actualH = asPipe_actual
    #GET START IMAGE SIZE
    ##start ratio and size before upscaling
    rats = round((np.exp(1))**(np.log(actualW / actualH) * (2/3)),2)
    testListrs = [abs(round(x - rats,3)) for x in aspectRatList]
    testMinrs = min(testListrs)
    testMinLocrs = testListrs.index(testMinrs)
    startW = targetW_list[testMinLocrs]
    startH = targetH_list[testMinLocrs]
    ##start ratio and size final (i.e. w/ upscaling)
    actualPx = startW * startH
    refImgsPx = actualPx * arefImgsPixelMult
    refImgRat = startW / startH
    refImgsH = (refImgsPx / refImgRat)**(1/2)
    refImgsW = refImgsPx / refImgsH
    start_width = int(round(refImgsW,0))
    start_height = int(round(refImgsH,0))
    #GET TARGET SIZE AND CROP
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
    return(start_width, start_height, cropW, cropH, int(targetW), int(targetH))


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
mwImageTo_sPipeMethod = mwImageTo_sPipe().mwITSP


class mwMaskTo_sPipe:   #WAS
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "mask": ("MASK",),
            }}
    RETURN_TYPES = ("MWSIZEPIPE",)
    RETURN_NAMES = ("sPipe",)
    FUNCTION = "mwMTSP"
    CATEGORY = "mohwag/psPipe"
    def mwMTSP(self, mask):
        return mwImageTo_sPipeMethod(maskToImageMethod(mask)[0])
mwMaskTo_sPipeMethod = mwMaskTo_sPipe().mwMTSP

class mwGridOverlay:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "images": ("IMAGE",),
                "scndSpacing_u": ([1, 2, 4, 8, 16, 32, 64], {"default": 8}),  #([8, 16, 32, 64, 128, 256, 512], {"default": 64}),
                "primSpacMult": ([2, 4, 8, 16], {"default": 4}),
                "scndThickness": ("INT", {"default": 1, "min": 1, "max": 16, "step": 2}),
                "primThickMult": ([2, 4, 8, 16], {"default": 4}),
                "lineColor": (["white", "black"],),
                "transparency": ("INT", {"default": 30, "min": 10, "max": 100, "step": 5})
            },
            "optional": {
                "sPipeS_opt": ("MWSIZEPIPE",),
            }}
    RETURN_TYPES = ("IMAGE", "MWSIZEPIPE", "IMAGE")
    RETURN_NAMES = ("images_pt", "sPipeS_pt", "image_wGrid")
    FUNCTION = "mwGO"
    CATEGORY = "mohwag/manip"
    def mwGO(self, images, scndSpacing_u, primSpacMult, scndThickness, primThickMult, lineColor, transparency, sPipeS_opt = None):
        if images.shape[0] > 1:
            image = imageFromBatchMethod(images, 0, 1)[0]  #self, image, batch_index, length
        else:
            image = images
        transparency = transparency /100
        primThickness = scndThickness * primThickMult
        w, h = mwImageTo_sPipeMethod(image)[0]
        if sPipeS_opt != None:
            startW, _ = sPipeS_opt
            scndSpacing = int(8 * scndSpacing_u * w / startW)
        else:
            scndSpacing = int(8 * scndSpacing_u)

        scndSpecSpc = scndSpacing - (scndThickness //2)
        primSpecSpc = scndSpacing - ((primThickness) //2)
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
        outImg_wGrid = imageCompositeMaskedMethod(image, gridImage, 0, 0, False, finalMask)[0]
        return (images, sPipeS_opt, outImg_wGrid)


class mwMaskSegment:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "sPipeC": ("MWSIZEPIPE",),
                "invert": ("BOOLEAN", {"default": False}),
                "xPoint": ("INT", {"default": 0, "min": 0, "step": 1}),
                "yPoint": ("INT", {"default": 0, "min": 0, "step": 1}),
                "xRange": ("INT", {"default": 0, "step": 1}),
                "yRange": ("INT", {"default": 0, "step": 1}),
                "increm_feath": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),               
                "feathLft_if": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "feathRgt_if": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "feathTop_if": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "feathBtm_if": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
            },
            "optional": {
                "maskVal": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01})  ###it's optional because it's referenced as a methodFunction in here
            }}
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "mwMS"
    CATEGORY = "mohwag/mask"
    def mwMS(self, sPipeC, invert, xPoint, yPoint, xRange, yRange, increm_feath, feathLft_if, feathRgt_if, feathTop_if, feathBtm_if, maskVal=1.0):
        w, h = sPipeC
        backMask = solidMaskMethod(0.0, w, h)[0]
        xi = xPoint
        xj = xPoint + xRange
        yi = yPoint
        yj = yPoint + yRange
        if xRange > 0:
            x0 = xi
            x1 = xj
        else:
            x0 = xj
            x1 = xi
        if yRange > 0:
            y0 = yi
            y1 = yj
        else:
            y0 = yj
            y1 = yi
        x0 = min(w, x0)
        x1 = max(0, x1)
        y0 = min(h, y0)
        y1 = max(0, y1)
        if x0 == x1 or y0 == y1:
            return (backMask,)
        w0 = x1 - x0
        h0 = y1 - y0
        fLft = increm_feath * feathLft_if
        fRgt = increm_feath * feathRgt_if
        fTop = increm_feath * feathTop_if
        fBtm = increm_feath * feathBtm_if
        frontMask_preFeath = solidMaskMethod(maskVal, w0, h0)[0]
        frontMask = featherMaskMethod(frontMask_preFeath, fLft, fTop, fRgt, fBtm)[0]  #(self, mask, left, top, right, bottom)
        combMask = maskCompositeMethod(backMask, frontMask, x0, y0, "add")[0]
        if invert:
            finMask = invertMaskMethod(combMask)[0]
        else:
            finMask = combMask
        return (finMask,)
mwMaskSegmentMethod = mwMaskSegment().mwMS


class mwMaskSegmentByPS:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "sPipeC": ("MWSIZEPIPE",),
                "psPipe": ("MWPOSSIZEPIPE",),
                "invert": ("BOOLEAN", {"default": False}),
                "increm_feath": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),               
                "feathLft_if": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "feathRgt_if": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "feathTop_if": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "feathBtm_if": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
            },
            "optional": {
                "maskVal": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01}) 
            }}
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "mwMSBPS"
    CATEGORY = "mohwag/mask"
    def mwMSBPS(self, sPipeC, psPipe, invert, increm_feath, feathLft_if, feathRgt_if, feathTop_if, feathBtm_if, maskVal=1.0):
        x, y, w, h = psPipe
        return mwMaskSegmentMethod(sPipeC, invert, x, y, w, h, increm_feath, feathLft_if, feathRgt_if, feathTop_if, feathBtm_if, maskVal)

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
    FUNCTION = "mwMCSS"
    CATEGORY = "mohwag/mask"
    def mwMCSS(self, destination, source, operation):
        outMask = maskCompositeMethod(destination, source, 0, 0, operation)[0]
        return (outMask,)
mwMaskCompSameSizeMethod = mwMaskCompSameSize().mwMCSS


class mwSolidMask:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "sPipeC": ("MWSIZEPIPE",),
                "maskVal": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01}),
            }}
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "mwMCSS"
    CATEGORY = "mohwag/mask"
    def mwMCSS(self, sPipeC, maskVal):
        wC, hC = sPipeC
        outMask = solidMaskMethod(maskVal, wC, hC)[0] #(self, value, width, height)
        return (outMask,)


class mwMaskTweak:
    outTypes = ["as-is", "val1_only", "val0_only"]
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "mask": ("MASK",),
                "operation0": (["none", "multiply", "add"], {"default": "none"}),
                "operation1": (["none", "multiply", "add"], {"default": "none"}),
                "operation2": (["none", "multiply", "add"], {"default": "none"}),
                "operation3": (["none", "multiply", "add"], {"default": "none"}),
                "outputMod": (moh.outTypes,),
            }}
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "mwMT"
    CATEGORY = "mohwag/mask"
    def mwMT(self, mask, operation0, operation1, operation2, operation3, outputMod):
        outMask = mask
        if outputMod == "as-is":
            if operation0 != "none":
                outMask = maskCompositeMethod(outMask, outMask, 0, 0, operation0)[0]
            if operation1 != "none":
                outMask = maskCompositeMethod(outMask, outMask, 0, 0, operation1)[0]
            if operation2 != "none":
                outMask = maskCompositeMethod(outMask, outMask, 0, 0, operation2)[0]
            if operation3 != "none":
                outMask = maskCompositeMethod(outMask, outMask, 0, 0, operation3)[0]
        if outputMod == "val1_only":
            w, h = mwMaskTo_sPipeMethod(mask)[0]
            mask1Val = solidMaskMethod(1.0, w, h)[0]
            outMask = mwMaskCompSameSizeMethod(outMask, mask1Val, "and")[0]
        if outputMod == "val0_only":
            inverted_mask = invertMaskMethod(outMask)[0]   #(self, mask)
            w, h = mwMaskTo_sPipeMethod(mask)[0]
            mask1Val = solidMaskMethod(1.0, w, h)[0]
            inverted_mask_val1 = mwMaskCompSameSizeMethod(inverted_mask, mask1Val, "and")[0]
            outMask = invertMaskMethod(inverted_mask_val1)[0]   #(self, mask)
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
            },
            "optional": {
                "mask_opt": ("MASK",),
                "psPipe_opt": ("MWPOSSIZEPIPE",),
            }}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "mwICM"
    CATEGORY = "mohwag/image"
    def mwICM(self, destination, source, mask_opt = None, psPipe_opt = None):
        if psPipe_opt == None:
            x, y = 0, 0
        else:
            x, y, w, h = psPipe_opt

        if mask_opt == None:
            if psPipe_opt == None:
                w, h = mwImageTo_sPipeMethod(source)[0]
            mask = solidMaskMethod(1.0, w, h)[0]  #(self, value, width, height)
        else:
            mask = mask_opt
        return imageCompositeMaskedMethod(destination, source, x, y, False, mask)


class mwImageScale:    #### doesn't need batch wrap ######
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "image": ("IMAGE",),
            "upscale_method": (moh.upscale_methods,),
            "sPipeOut": ("MWSIZEPIPE",),
            }}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "mwIS"
    CATEGORY = "mohwag/image"
    def mwIS(self, image, upscale_method, sPipeOut):
        w, h = sPipeOut
        output = imageScaleMethod(image, upscale_method, w, h, "disabled")[0]   #(self, image, upscale_method, width, height, crop)
        return (output,)


class mwImageBranch:    ############### DO I NEED THIS?
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "images": ("IMAGE",),
            }}
    RETURN_TYPES = ("IMAGE","IMAGE",)
    RETURN_NAMES = ("images_pt", "image0")
    FUNCTION = "mwISTC"
    CATEGORY = "mohwag/image"
    def mwISTC(self, images):
        image0 = imageFromBatchMethod(images, 0, 1)[0]  #self, image, batch_index, length
        return (images, image0)
    

class mwImageCrop:   #### doesn't need batch wrap ######
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
mwImageCropMethod = mwImageCrop().mwIC  #(self, image, psPipe)


class mwImageCropwParams:   #### doesn't need batch wrap ######
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "images": ("IMAGE",),
                #"unitVal": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),               
                "deltaLft_u": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "deltaRgt_u": ("INT", {"default": 0, "min": -1024, "max": 0, "step": 1}),
                "deltaTop_u": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "deltaBtm_u": ("INT", {"default": 0, "min": -1024, "max": 0, "step": 1}),
            },
            "optional": {
                "sPipeS_opt": ("MWSIZEPIPE",),
            }}
    RETURN_TYPES = ("IMAGE", "MWPOSSIZEPIPE", "MWSIZEPIPE", "MWSIZEPIPE")
    RETURN_NAMES = ("image", "psPipe_crpd", "sPipeS_updtd", "sPipeC_updtd")
    FUNCTION = "mwICWP"
    CATEGORY = "mohwag/image"
    def mwICWP(self, images, deltaLft_u, deltaRgt_u, deltaTop_u, deltaBtm_u, sPipeS_opt = None):
        image = images
        wC, hC = mwImageTo_sPipeMethod(image[0])[0]
        if sPipeS_opt != None:
            wS, _ = sPipeS_opt
            unitVal = int(8 * (wC / wS))
        else:
            unitVal = 8
        xCrp = int(deltaLft_u * unitVal)
        yCrp = int(deltaTop_u * unitVal)
        wCrp = int(wC - xCrp + (deltaRgt_u * unitVal))
        hCrp = int(hC - yCrp + (deltaBtm_u * unitVal))
        wS_new = int(8 * (wCrp / unitVal))
        hS_new = int(8 * (hCrp / unitVal))
        output = imageCropMethod(image, wCrp, hCrp, xCrp, yCrp)[0]  ##(self, image, width, height, x, y)
        out_psPipe_crpd = (xCrp, yCrp, wCrp, hCrp)
        out_sPipeS_updtd = (wS_new, hS_new)
        out_sPipeC_updtd = (wCrp, hCrp)
        return (output, out_psPipe_crpd, out_sPipeS_updtd, out_sPipeC_updtd)


class mwImageConform_StartSizeXL:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "image": ("IMAGE",),
            "upscale_method": (moh.upscale_methods,),
            }}
    RETURN_TYPES = ("IMAGE", "MWSIZEPIPE")
    RETURN_NAMES = ("image", "sPipeS_updtd")
    FUNCTION = "mwICSSXL"
    CATEGORY = "mohwag/image"
    def mwICSSXL(self, image, upscale_method):
        aspectRatList = [0.25, 0.26, 0.27, 0.28, 0.32, 0.33, 0.35, 0.4, 0.42, 0.48, 0.5, 0.52, 0.57, 0.6, 0.68, 0.72, 0.78, 0.82, 0.88, 0.94, 1, 1.07, 1.13, 1.21, 1.29, 1.38, 1.46, 1.67, 1.75, 2, 2.09, 2.4, 2.5, 2.89, 3, 3.11, 3.63, 3.75, 3.88, 4]
        targetW_list = [512, 512, 512, 512, 576, 576, 576, 640, 640, 704, 704, 704, 768, 768, 832, 832, 896, 896, 960, 960, 1024, 1024, 1088, 1088, 1152, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048]
        targetH_list = [2048, 1984, 1920, 1856, 1792, 1728, 1664, 1600, 1536, 1472, 1408, 1344, 1344, 1280, 1216, 1152, 1152, 1088, 1088, 1024, 1024, 960, 960, 896, 896, 832, 832, 768, 768, 704, 704, 640, 640, 576, 576, 576, 512, 512, 512, 512]

        actualW, actualH = mwImageTo_sPipeMethod(image)[0]
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
                "squareSize": ("INT", {"default": 224, "min": 0, "max": 10000, "step": 8}),
            }}
    CATEGORY = "mohwag/image"
    FUNCTION = "MWB"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGES", "MASKS")
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

class mwBatch_squareHelp:# Zuellni
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "input_dir": ("STRING",{"default": folder_paths.get_input_directory()}),
                "inputImg_cnfrmMult": ([8, 16, 32, 64], {"default": 8}),
                "squareSize": ("INT", {"default": 224, "min": 0, "max": 10000, "step": 8}),

            }}
    CATEGORY = "mohwag/image"
    FUNCTION = "MWBSH"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    def MWBSH(self, input_dir, inputImg_cnfrmMult, squareSize):
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

        images_uncut = []
        for image in pil_images:
            image = TF.to_tensor(image)
            image[:3, image[3, :, :] == 0] = 0

            image = [image]
            image = torch.stack(image)
            image = image.permute(0,2,3,1)
            image = image[:, :, :, :3]
            images_uncut.append(image)

        for i in range(len(images_uncut)):
            w, h = mwImageTo_sPipeMethod(images_uncut[i])[0]
            wTarget = inputImg_cnfrmMult * (w // inputImg_cnfrmMult)
            hTarget = inputImg_cnfrmMult * (h // inputImg_cnfrmMult)
            xTarget = (w - wTarget) //2
            yTarget = (h - hTarget) //2

            imgTarget= mwImageCropMethod(images_uncut[i], (xTarget, yTarget, wTarget, hTarget))[0]
            imgScaled = imageScaleMethod(imgTarget, "bicubic", squareSize, squareSize, "disabled")[0]  #(self, image, upscale_method, width, height, crop)

            if i == 0:
                outputImages = imgScaled
            else:
                outputImages = imageBatchMethod(outputImages, imgScaled)[0]
        return (outputImages,)

class mwBatch_AllSections:# Zuellni
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "input_dir": ("STRING",{"default": folder_paths.get_input_directory()}),
                "inputImg_cnfrmMult": ([8, 16, 32, 64], {"default": 8}),
                "squareSize": ("INT", {"default": 224, "min": 0, "max": 10000, "step": 8}),

            }}
    CATEGORY = "mohwag/image"
    FUNCTION = "MWBAS"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    def MWBAS(self, input_dir, inputImg_cnfrmMult, squareSize):
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

        images_uncut = []
        for image in pil_images:
            image = TF.to_tensor(image)
            image[:3, image[3, :, :] == 0] = 0

            image = [image]
            image = torch.stack(image)
            image = image.permute(0,2,3,1)
            image = image[:, :, :, :3]
            images_uncut.append(image)

        images_prepped = []
        for i in range(len(images_uncut)):
            w, h = mwImageTo_sPipeMethod(images_uncut[i])[0]
            wTarget = inputImg_cnfrmMult * (w // inputImg_cnfrmMult)
            hTarget = inputImg_cnfrmMult * (h // inputImg_cnfrmMult)
            xTarget = (w - wTarget) //2
            yTarget = (h - hTarget) //2

            if wTarget == hTarget:
                imgTarget= mwImageCropMethod(images_uncut[i], (xTarget, yTarget, wTarget, hTarget))[0]
                imgScaled = imageScaleMethod(imgTarget, "bicubic", squareSize, squareSize, "disabled")[0]  #(self, image, upscale_method, width, height, crop)
                #images_prepped.append[imgScaled]

                if i == 0:
                    outputImages = imgScaled
                else:
                    outputImages = imageBatchMethod(outputImages, imgScaled)[0]

            elif wTarget < hTarget:
                jcnt = math.ceil(hTarget / wTarget)
                hTarget = wTarget
                yMidpointStep = (h // (jcnt - 1))

                for j in range(jcnt):
                    if j == 0:
                        yTarget = 0
                    elif j == jcnt - 1:
                        yTarget = h - hTarget
                    else:
                        yTarget = (yMidpointStep * j) - (hTarget // 2)

                    imgTarget= mwImageCropMethod(images_uncut[i], (xTarget, yTarget, wTarget, hTarget))[0]
                    imgScaled = imageScaleMethod(imgTarget, "bicubic", squareSize, squareSize, "disabled")[0]
                    #images_prepped.append[imgScaled]

                    if i == 0:
                        if j == 0:
                            outputImages = imgScaled
                        else:
                            outputImages = imageBatchMethod(outputImages, imgScaled)[0]
                    else:
                        outputImages = imageBatchMethod(outputImages, imgScaled)[0]

            else:
                jcnt = math.ceil(wTarget / hTarget)
                wTarget = hTarget
                xMidpointStep = (w // (jcnt - 1))

                for j in range(jcnt):
                    if j == 0:
                        xTarget = 0
                    elif j == jcnt - 1:
                        xTarget = w - wTarget
                    else:
                        xTarget = (xMidpointStep * j) - (wTarget // 2)

                    imgTarget= mwImageCropMethod(images_uncut[i], (xTarget, yTarget, wTarget, hTarget))[0]
                    imgScaled = imageScaleMethod(imgTarget, "bicubic", squareSize, squareSize, "disabled")[0]
                    #images_prepped.append[imgScaled]

                    if i == 0:
                        if j == 0:
                            outputImages = imgScaled
                        else:
                            outputImages = imageBatchMethod(outputImages, imgScaled)[0]
                    else:
                        outputImages = imageBatchMethod(outputImages, imgScaled)[0]

        """ for k in range(len(images_prepped)):
            if k ==0:
                outputImages = images_prepped[0]
            else:
                outputImages = imageBatchMethod(outputImages, images_prepped[k])[0] """
        return (outputImages,)

class mwpsConditioningSetArea():
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
                    "condPos": ("CONDITIONING",),
                    "condNeg": ("CONDITIONING",),
                    "psPipe": ("MWPOSSIZEPIPE",),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                    }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("condPos", "condNeg")
    FUNCTION = "mwCSA"
    CATEGORY = "mohwag/Conditioning"
    def mwCSA(self, condPos, condNeg, psPipe, strength):
        x, y, w, h = psPipe
        strengthMult = strength
        #condPosList = condPos.copy()
        #condNegList = condNeg.copy()
        """ for i in range(len(condPosList)):
            condPosList[i][1]["strength"] = condPosList[i][1].get("strength", 1) * strengthMult
            condPosList[i][1]["area"] = (h//8, w//8, y//8, x//8)
            condPosList[i][1]["set_area_to_bounds"] = False
        for i in range(len(condNegList)):
            condNegList[i][1]["strength"] = condNegList[i][1].get("strength", 1) * strengthMult
            condNegList[i][1]["area"] = (h//8, w//8, y//8, x//8)
            condNegList[i][1]["set_area_to_bounds"] = False """
        cpos = []
        for t in condPos:
            n = [t[0], t[1].copy()]
            n[1]["strength"] = n[1].get("strength", 1) * strengthMult
            n[1]["area"] = (h//8, w//8, y//8, x//8)
            n[1]["set_area_to_bounds"] = False
            cpos.append(n)
        cneg = []
        for t in condNeg:
            n = [t[0], t[1].copy()]
            n[1]["strength"] = n[1].get("strength", 1) * strengthMult
            n[1]["area"] = (h//8, w//8, y//8, x//8)
            n[1]["set_area_to_bounds"] = False
            cneg.append(n)
        return (cpos, cneg)
mwpsConditioningSetAreaMethod = mwpsConditioningSetArea().mwCSA

def psPipeModForLtnt(aimgD, ad0, ad1, amultReq, allDown=False):
    amultReq = amultReq * 8
    ad0 = amultReq * (ad0 // amultReq)
    if allDown:
        ad1 = amultReq * (ad1 // amultReq)
    else:
        ad1 = amultReq * ((ad1 // amultReq) + (ad1 % amultReq > 0))
    amaxD = amultReq * (aimgD // amultReq)
    
    ad0_1 = max(0, ad0)
    ad1_1 = min(amaxD, ad1)
    ad0_f = int(ad0_1)
    ad1_f = int(ad1_1)
    return (ad0_f, ad1_f)

class mwConditioningTuple:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
                    "cond0": ("CONDITIONING",),
                    },
                "optional": {
                    "cond1": ("CONDITIONING",),
                    "cond2": ("CONDITIONING",),
                    "cond3": ("CONDITIONING",),
                    "cond4": ("CONDITIONING",),
                    "cond5": ("CONDITIONING",),                    
                    }}
    RETURN_TYPES = ("CONDTUPLE",)
    RETURN_NAMES = ("condTuple",)
    FUNCTION = "mwCT"
    CATEGORY = "mohwag/Conditioning"
    def mwCT(self, cond0, cond1=None, cond2=None, cond3=None, cond4=None, cond5=None):
        outCondTuple = (cond0,)
        if cond1 != None:
            outCondTuple = outCondTuple + (cond1,)
        if cond2 != None:
            outCondTuple = outCondTuple + (cond2,)
        if cond3 != None:
            outCondTuple = outCondTuple + (cond3,)
        if cond4 != None:
            outCondTuple = outCondTuple + (cond4,)
        if cond5 != None:
            outCondTuple = outCondTuple + (cond5,)
        return (outCondTuple,)

class mwFullPipeTuple:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
                    "fullpipe0": ("MWFULLPIPE",),
                    },
                "optional": {
                    "fullpipe1": ("MWFULLPIPE",),
                    "fullpipe2": ("MWFULLPIPE",),
                    "fullpipe3": ("MWFULLPIPE",),
                    "fullpipe4": ("MWFULLPIPE",),
                    "fullpipe5": ("MWFULLPIPE",),                    
                    }}
    RETURN_TYPES = ("MWFULLPIPETUPLE",)
    RETURN_NAMES = ("fullPipeTuple",)
    FUNCTION = "mwFPT"
    CATEGORY = "mohwag/Conditioning"
    def mwFPT(self, fullpipe0, fullpipe1=None, fullpipe2=None, fullpipe3=None, fullpipe4=None, fullpipe5=None):
        outfullpipeTuple = (fullpipe0,)
        if fullpipe1 != None:
            outfullpipeTuple = outfullpipeTuple + (fullpipe1,)
        if fullpipe2 != None:
            outfullpipeTuple = outfullpipeTuple + (fullpipe2,)
        if fullpipe3 != None:
            outfullpipeTuple = outfullpipeTuple + (fullpipe3,)
        if fullpipe4 != None:
            outfullpipeTuple = outfullpipeTuple + (fullpipe4,)
        if fullpipe5 != None:
            outfullpipeTuple = outfullpipeTuple + (fullpipe5,)
        return (outfullpipeTuple,)



class mwConditioningCnetTileSpecTwo:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
                    "condTuple_Pos": ("CONDTUPLE",),
                    "condTuple_Neg": ("CONDTUPLE",),
                    "control_net": ("CONTROL_NET", ),
                    "image": ("IMAGE", ),
                    "sPipeC": ("MWSIZEPIPE",),
                    "refImgsPixelMultPos": ("FLOAT", {"default": 4.00, "min": 0.00, "max": 25.00, "step": 0.05}),
                    "refImgsPixelMultNeg": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 25.00, "step": 0.05}),
                    "hzOverlap_xpndPct": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "vtOverlap_xpndPct": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "cond_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                    "tile_coords": ("STRING", {"multiline": True}),
                    "cnet_interval_strength": ("STRING", {"multiline": True}),
                    }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LIST")
    RETURN_NAMES = ("condPos", "condNeg", "metrics")
    FUNCTION = "mwCTST"
    CATEGORY = "mohwag/Conditioning"
    def mwCTST(self, condTuple_Pos, condTuple_Neg, control_net, image, sPipeC, refImgsPixelMultPos, refImgsPixelMultNeg, hzOverlap_xpndPct, vtOverlap_xpndPct, cond_strength, tile_coords, cnet_interval_strength):
        wC, hC = sPipeC
        image_upsc = imageScaleMethod(image, "bicubic", wC, hC, "disabled")[0]  #(self, image, upscale_method, width, height, crop)
        coordsList = tile_coords.splitlines()
        for x in ["", " ", "  "]:
            while(x in coordsList):
                coordsList.remove(x)
        for i in range(len(coordsList)):
            coordsList[i] = coordsList[i].split(";", -1)
            for j in range(len(coordsList[i])):
                coordsList[i][j] = coordsList[i][j].split(",", -1)
                for k in range(len(coordsList[i][j])):
                    coordsList[i][j][k] = float(coordsList[i][j][k])
        pointsList = copy.deepcopy(coordsList)
        dimLenUnits = pointsList[0][0]
        pointsList = pointsList[1:]
        for i in range(len(pointsList)):
            for j in range(len(pointsList[i])):
                for k in range(len(pointsList[i][j])):
                    if j == 0:
                        dimMult = wC / dimLenUnits[0]
                    else:
                        dimMult = hC / dimLenUnits[1]
                    pointsList[i][j][k] = pointsList[i][j][k] * dimMult
        orig_sizeList = copy.deepcopy(pointsList)
        for i in range(len(orig_sizeList)):
            orig_sizeList[i] = ((orig_sizeList[i][1][1] - orig_sizeList[i][1][0]) * (orig_sizeList[i][0][1] - orig_sizeList[i][0][0]))**(1/2)

        isList = cnet_interval_strength.splitlines()
        for x in ["", " ", "  "]:
            while(x in isList):
                isList.remove(x)
        intervalList = copy.deepcopy(isList)
        strengthList = copy.deepcopy(isList)
        intervalList = intervalList[0].split(",",-1)
        intervalList = [float(x) for x in intervalList]
        strengthList = strengthList[1].split(",",-1)
        strengthList = [float(y) for y in strengthList]

        countFinalGate = 0
        countPos = 0
        countNeg = 0
        sizeList = []
        check_endPct =[]
        for i in range(len(pointsList)):
            x0_0 = pointsList[i][0][0]
            x1_0 = pointsList[i][0][1]
            x0_1, x1_1 = psPipeModForLtnt(wC, x0_0 - (wC * hzOverlap_xpndPct), x1_0 + (wC * hzOverlap_xpndPct), 1, True)
            y0_0 = pointsList[i][1][0]
            y1_0 = pointsList[i][1][1]
            y0_1, y1_1 = psPipeModForLtnt(hC, y0_0 - (hC * vtOverlap_xpndPct), y1_0 + (hC * vtOverlap_xpndPct), 1, True)
            
            psPipeC_ij = int(x0_1), int(y0_1), int(x1_1 - x0_1), int(y1_1 - y0_1)
            sPipeC_ij = int(x1_1 - x0_1), int(y1_1 - y0_1)

            start_width, start_height, cropW, cropH, targetW, targetH = sdxl_cond_sizes(sPipeC_ij, refImgsPixelMultPos)
            condPos = condTuple_Pos[i]
            condPos_w_whInfo = node_helpers.conditioning_set_values(condPos, {"width": start_width, "height": start_height,
                                                                            "crop_w": cropW, "crop_h": cropH,
                                                                            "target_width": targetW, "target_height": targetH})

            start_width, start_height, cropW, cropH, targetW, targetH = sdxl_cond_sizes(sPipeC_ij, refImgsPixelMultNeg)
            condNeg = condTuple_Neg[i]
            condNeg_w_whInfo = node_helpers.conditioning_set_values(condNeg, {"width": start_width, "height": start_height,
                                                                            "crop_w": cropW, "crop_h": cropH,
                                                                            "target_width": targetW, "target_height": targetH})

            pnCond_w_whInfo_areaInfo_i = mwpsConditioningSetAreaMethod(condPos_w_whInfo, condNeg_w_whInfo, psPipeC_ij, cond_strength)
            boundedImage_i = mwImageCropMethod(image_upsc, psPipeC_ij)[0]
            size_i = ((sPipeC_ij[0] * sPipeC_ij[1])**(1/2))
            sizeList.append(size_i)
            for j in range(len(strengthList)):
                if j == 0:
                    startAdd = 0.0
                else:
                    startAdd = 0.001
                cnetStrtPct_j = intervalList[j] + startAdd
                cnetEndPct_j = intervalList[j+1]
                cnetStrength_j = strengthList[j]
                for k in range(len(pnCond_w_whInfo_areaInfo_i)):
                    for m in range(len(pnCond_w_whInfo_areaInfo_i[k])):
                        stepStrtPct_km = float(pnCond_w_whInfo_areaInfo_i[k][m][1].get("start_percent", 0.0))
                        if stepStrtPct_km == None:
                            stepStrtPct_km = 0.0
                        theStrtPct_jkm = max(cnetStrtPct_j, stepStrtPct_km)
                        stepEndPct_km = float(pnCond_w_whInfo_areaInfo_i[k][m][1].get("end_percent", 1.0))
                        if stepEndPct_km == None:
                            stepEndPct_km = 1.0
                        theEndPct_jkm = min(cnetEndPct_j, stepEndPct_km)
                        if theEndPct_jkm > theStrtPct_jkm:
                            countFinalGate += 1
                            cond_bfCnet_ikm = conditioningSetTimestepRangeMethod([pnCond_w_whInfo_areaInfo_i[k][m]], theStrtPct_jkm, theEndPct_jkm)[0]
                            cond_ijkm = controlNetApplyAdvancedMethod(cond_bfCnet_ikm, cond_bfCnet_ikm, control_net, boundedImage_i, cnetStrength_j, theStrtPct_jkm, theEndPct_jkm)[0]
                            check_endPct.append(theEndPct_jkm)
                            if k == 0:
                                if countPos == 0:
                                    condPosF = cond_ijkm
                                else:
                                    condPosF = condPosF + cond_ijkm
                                countPos += 1
                            elif k == 1:
                                if countNeg == 0:
                                    condNegF = cond_ijkm
                                else:
                                    condNegF = condNegF + cond_ijkm
                                countNeg += 1
        cntPos = len(condPosF)
        cntNeg = len(condNegF)
        sizeList = [int(x) for x in sizeList]
        lrgstSize = max(sizeList)
        avgSize = sum(sizeList) / len(sizeList)
        metrics_out = int(lrgstSize), int(avgSize), cntPos, cntNeg, sizeList
        return (condPosF, condNegF, metrics_out)


class mwConditioningCnetTileSpec():
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
                    "condPos": ("CONDITIONING",),
                    "condNeg": ("CONDITIONING",),
                    "control_net": ("CONTROL_NET", ),
                    "image": ("IMAGE", ),
                    "sPipeC": ("MWSIZEPIPE",),
                    "refImgsPixelMultPos": ("FLOAT", {"default": 4.00, "min": 0.00, "max": 25.00, "step": 0.05}),
                    "refImgsPixelMultNeg": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 25.00, "step": 0.05}),
                    "hzOverlap_xpndPct": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "vtOverlap_xpndPct": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "cond_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                    "tile_coords": ("STRING", {"multiline": True}),
                    "cnet_interval_strength": ("STRING", {"multiline": True}),
                    }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LIST")
    RETURN_NAMES = ("condPos", "condNeg", "metrics")
    FUNCTION = "mwCTS"
    CATEGORY = "mohwag/Conditioning"
    def mwCTS(self, condPos, condNeg, control_net, image, sPipeC, refImgsPixelMultPos, refImgsPixelMultNeg, hzOverlap_xpndPct, vtOverlap_xpndPct, cond_strength, tile_coords, cnet_interval_strength):
        wC, hC = sPipeC
        image_upsc = imageScaleMethod(image, "bicubic", wC, hC, "disabled")[0]  #(self, image, upscale_method, width, height, crop)
        coordsList = tile_coords.splitlines()
        for x in ["", " ", "  "]:
            while(x in coordsList):
                coordsList.remove(x)
        for i in range(len(coordsList)):
            coordsList[i] = coordsList[i].split(";", -1)
            for j in range(len(coordsList[i])):
                coordsList[i][j] = coordsList[i][j].split(",", -1)
                for k in range(len(coordsList[i][j])):
                    coordsList[i][j][k] = float(coordsList[i][j][k])
        pointsList = copy.deepcopy(coordsList)
        dimLenUnits = pointsList[0][0]
        pointsList = pointsList[1:]
        for i in range(len(pointsList)):
            for j in range(len(pointsList[i])):
                for k in range(len(pointsList[i][j])):
                    if j == 0:
                        dimMult = wC / dimLenUnits[0]
                    else:
                        dimMult = hC / dimLenUnits[1]
                    pointsList[i][j][k] = pointsList[i][j][k] * dimMult
        orig_sizeList = copy.deepcopy(pointsList)
        for i in range(len(orig_sizeList)):
            orig_sizeList[i] = ((orig_sizeList[i][1][1] - orig_sizeList[i][1][0]) * (orig_sizeList[i][0][1] - orig_sizeList[i][0][0]))**(1/2)
        #lrgstSize = max(orig_sizeList)
        #avgSize = sum(orig_sizeList) / len(orig_sizeList)

        isList = cnet_interval_strength.splitlines()
        for x in ["", " ", "  "]:
            while(x in isList):
                isList.remove(x)
        intervalList = copy.deepcopy(isList)
        strengthList = copy.deepcopy(isList)
        intervalList = intervalList[0].split(",",-1)
        intervalList = [float(x) for x in intervalList]
        strengthList = strengthList[1].split(",",-1)
        strengthList = [float(y) for y in strengthList]

        countFinalGate = 0
        countPos = 0
        countNeg = 0
        sizeList = []
        check_endPct =[]
        for i in range(len(pointsList)):
            x0_0 = pointsList[i][0][0]
            x1_0 = pointsList[i][0][1]
            #w0 = x1_0 - x0_0
            x0_1, x1_1 = psPipeModForLtnt(wC, x0_0 - (wC * hzOverlap_xpndPct), x1_0 + (wC * hzOverlap_xpndPct), 1, True)
            y0_0 = pointsList[i][1][0]
            y1_0 = pointsList[i][1][1]
            #h0 = x1_0 - x0_0
            y0_1, y1_1 = psPipeModForLtnt(hC, y0_0 - (hC * vtOverlap_xpndPct), y1_0 + (hC * vtOverlap_xpndPct), 1, True)
            
            psPipeC_ij = int(x0_1), int(y0_1), int(x1_1 - x0_1), int(y1_1 - y0_1)
            sPipeC_ij = int(x1_1 - x0_1), int(y1_1 - y0_1)

            start_width, start_height, cropW, cropH, targetW, targetH = sdxl_cond_sizes(sPipeC_ij, refImgsPixelMultPos)
            condPos_w_whInfo = node_helpers.conditioning_set_values(condPos, {"width": start_width, "height": start_height,
                                                                            "crop_w": cropW, "crop_h": cropH,
                                                                            "target_width": targetW, "target_height": targetH})

            start_width, start_height, cropW, cropH, targetW, targetH = sdxl_cond_sizes(sPipeC_ij, refImgsPixelMultNeg)
            condNeg_w_whInfo = node_helpers.conditioning_set_values(condNeg, {"width": start_width, "height": start_height,
                                                                            "crop_w": cropW, "crop_h": cropH,
                                                                            "target_width": targetW, "target_height": targetH})

            pnCond_w_whInfo_areaInfo_i = mwpsConditioningSetAreaMethod(condPos_w_whInfo, condNeg_w_whInfo, psPipeC_ij, cond_strength)
            boundedImage_i = mwImageCropMethod(image_upsc, psPipeC_ij)[0]
            size_i = ((sPipeC_ij[0] * sPipeC_ij[1])**(1/2))
            sizeList.append(size_i)
            for j in range(len(strengthList)):
                if j == 0:
                    startAdd = 0.0
                else:
                    startAdd = 0.001
                cnetStrtPct_j = intervalList[j] + startAdd
                cnetEndPct_j = intervalList[j+1]
                cnetStrength_j = strengthList[j]
                for k in range(len(pnCond_w_whInfo_areaInfo_i)):
                    for m in range(len(pnCond_w_whInfo_areaInfo_i[k])):
                        stepStrtPct_km = float(pnCond_w_whInfo_areaInfo_i[k][m][1].get("start_percent", 0.0))
                        if stepStrtPct_km == None:
                            stepStrtPct_km = 0.0
                        theStrtPct_jkm = max(cnetStrtPct_j, stepStrtPct_km)
                        stepEndPct_km = float(pnCond_w_whInfo_areaInfo_i[k][m][1].get("end_percent", 1.0))
                        if stepEndPct_km == None:
                            stepEndPct_km = 1.0
                        theEndPct_jkm = min(cnetEndPct_j, stepEndPct_km)
                        if theEndPct_jkm > theStrtPct_jkm:
                            countFinalGate += 1
                            cond_bfCnet_ikm = conditioningSetTimestepRangeMethod([pnCond_w_whInfo_areaInfo_i[k][m]], theStrtPct_jkm, theEndPct_jkm)[0]
                            cond_ijkm = controlNetApplyAdvancedMethod(cond_bfCnet_ikm, cond_bfCnet_ikm, control_net, boundedImage_i, cnetStrength_j, theStrtPct_jkm, theEndPct_jkm)[0]
                            check_endPct.append(theEndPct_jkm)
                            if k == 0:
                                if countPos == 0:
                                    condPosF = cond_ijkm
                                else:
                                    condPosF = condPosF + cond_ijkm
                                countPos += 1
                            elif k == 1:
                                if countNeg == 0:
                                    condNegF = cond_ijkm
                                else:
                                    condNegF = condNegF + cond_ijkm
                                countNeg += 1
        cntPos = len(condPosF)
        cntNeg = len(condNegF)

        """ aa = len(condPos_w_whInfo[0])
        ab = len(condNeg_w_whInfo[0])
        ac = len(pnCond_w_whInfo_areaInfo_i[0])
        ad = len(pnCond_w_whInfo_areaInfo_i[0][0])
        ae = len(pnCond_w_whInfo_areaInfo_i[1][0])
        af = len(cond_bfCnet_ikm[0])
        ag = len([cond_ijkm][0][0])
        ah = len(cond_ijkm)
        ai = len(cond_ijkm[0]) """
        #ah = len(cond_ijkm[1])

        sizeList = [int(x) for x in sizeList]
        lrgstSize = max(sizeList)
        avgSize = sum(sizeList) / len(sizeList)

        metrics_out = int(lrgstSize), int(avgSize), cntPos, cntNeg, sizeList
        #metrics_out = cntPos, cntNeg, len(condPos_w_whInfo), len(condNeg_w_whInfo), len(pnCond_w_whInfo_areaInfo_i[0]), len(pnCond_w_whInfo_areaInfo_i[1]), len(cond_bfCnet_ikm), len(cond_ijkm[0]), len(cond_ijkm[1])
        #metrics_out = countFinalGate, (j, k, m), (len(condPosF), len(condNegF)), (aa, ab, ac, ad, ae, af, ag, ah, ai)
        return (condPosF, condNegF, metrics_out)
      

class mwConditioningAddAesthScore():
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
                    "condPos": ("CONDITIONING",),
                    "condNeg": ("CONDITIONING",),
                    "ascore_pos": ("INT", {"default": -1, "min": -1, "max": 1024, "step": 1}),
                    "ascore_neg": ("INT", {"default": -1, "min": -1, "max": 1024, "step": 1}),
                    }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("condPos", "condNeg")
    FUNCTION = "mwCAAS"
    CATEGORY = "mohwag/Conditioning"
    def mwCAAS(self, condPos, condNeg, ascore_pos, ascore_neg):
        condPosList = condPos.copy()
        condNegList = condNeg.copy()
        if ascore_pos != -1:
            for i in range(len(condPosList)):
                condPosList[i][1]["aesthetic_score"] = ascore_pos
        if ascore_neg != -1:
            for i in range(len(condNegList)):
                condNegList[i][1]["aesthetic_score"] = ascore_neg
        return (condPosList, condNegList)

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

class mwFullPipe_modelEdit:
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

class mwCkpt_ckptEdit:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": { 
                    "mwCkpt": ("MWCKPT", ),
                    },
                "optional": {
                    "model": ("MODEL",),
                    "clip": ("CLIP",),
                    "vae": ("VAE",),
                }}
    RETURN_TYPES = ("MWCKPT",)
    RETURN_NAMES = ("mwCkpt",)
    FUNCTION = "mwCCE"

    CATEGORY = "mohwag/FullPipe_BuildLoadMerge"
    def mwCCE(self, mwCkpt, model=None, clip=None, vae=None):
        out_model, out_clip, out_vae = mwCkpt

        if model != None:
            out_model = model
        if clip != None:
            out_clip = clip
        if vae != None:
            out_vae - vae
        
        out_mwCkpt = out_model, out_clip, out_vae
        return (out_mwCkpt,)

class mwCkpt_ckptMerge:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
                    "mwCkpt1": ("MWCKPT",),
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
        return {"required": {
                    "mwFullPipe": ("MWFULLPIPE",),
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


class mwCkpt_loraMerge:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": { "mwCkpt": ("MWCKPT",),
                              "mwLora": ("MWLORA",),
                              "strength": ("FLOAT", {"default": 0, "min": -10.0, "max": 10.0, "step": 0.04}),
                              }}
    RETURN_TYPES = ("MWCKPT",)
    RETURN_NAMES = ("mwCKpt",)
    FUNCTION = "mwCPLM"
    CATEGORY = "mohwag/FullPipe_BuildLoadMerge"

    def mwCPLM(self, mwCkpt, mwLora, strength):
        if strength == 0:
            return (mwCkpt,)
        #mwCkpt1, seed, sampler_name, scheduler = mwFullPipe
        vaeAlways = mwCkpt[2]
        model1, clip1, _ = mwCkpt
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model1, clip1, mwLora, strength, strength)
        mwCkptNew = model_lora, clip_lora, vaeAlways
        #mwFullPipeNew = mwCkptNew, seed, sampler_name, scheduler
        return (mwCkptNew,)

class mwCkpt_loraStackMerge:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": { "mwCkpt": ("MWCKPT",),
                              "strength0": ("FLOAT", {"default": 0, "min": -10.0, "max": 10.0, "step": 0.04}),
                              "strength1": ("FLOAT", {"default": 0, "min": -10.0, "max": 10.0, "step": 0.04}),
                              "strength2": ("FLOAT", {"default": 0, "min": -10.0, "max": 10.0, "step": 0.04}),
                              "strength3": ("FLOAT", {"default": 0, "min": -10.0, "max": 10.0, "step": 0.04}),
                              },
                "optional": { "mwLora0": ("MWLORA",),
                              "mwLora1": ("MWLORA",),
                              "mwLora2": ("MWLORA",),
                              "mwLora3": ("MWLORA",),
                              }}
    RETURN_TYPES = ("MWCKPT",)
    RETURN_NAMES = ("mwCKpt",)
    FUNCTION = "mwCPLSM"
    CATEGORY = "mohwag/FullPipe_BuildLoadMerge"

    def mwCPLSM(self, mwCkpt, mwLora0, strength0, strength1, strength2, strength3, mwLora1 = None, mwLora2 = None, mwLora3 = None):
        model_lora, clip_lora, vaeAlways = mwCkpt

        if mwLora0 != None and strength0 !=0:
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model_lora, clip_lora, mwLora0, strength0, strength0)
        if mwLora1 != None and strength1 !=0:
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model_lora, clip_lora, mwLora1, strength1, strength1)
        if mwLora2 != None and strength2 !=0:
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model_lora, clip_lora, mwLora2, strength2, strength2)
        if mwLora3 != None and strength3 !=0:
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model_lora, clip_lora, mwLora3, strength3, strength3)
        mwCkptNew = model_lora, clip_lora, vaeAlways
        return (mwCkptNew,)

class mwLtntPipe_Create:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "ltnt": ("LATENT", ),
            "sPipeS": ("MWSIZEPIPE",)
            }}

    RETURN_TYPES = ("MWLTNTPIPE",)
    RETURN_NAMES = ("ltntPipe",)
    FUNCTION = "mwLPC"
    CATEGORY = "mohwag/LatentPipe"
    def mwLPC(self, ltnt, sPipeS):
        initialW, initialH = sPipeS
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
        return {
            "required": {
                "image": ("IMAGE", ),
            },
            "optional": {
                "sPipeS_opt": ("MWSIZEPIPE",),
                "mwCkpt_opt": ("MWCKPT",),
                "mwFullPipe_opt": ("MWFULLPIPE",),
            }}
    RETURN_TYPES = ("MWLTNTPIPE",)
    RETURN_NAMES = ("ltntPipe",)
    FUNCTION = "mwLPC"
    CATEGORY = "mohwag/LatentPipe"
    def mwLPC(self, image, sPipeS_opt = None, mwCkpt_opt = None, mwFullPipe_opt = None):
        wC, hC = mwImageTo_sPipeMethod(image)[0]

        if sPipeS_opt != None:
            wS, hS = sPipeS_opt
        else:
            wS, hS = wC, hC

        if mwCkpt_opt != None:
            _, _, vaeAlways = mwCkpt_opt
        else:        
            mwCkptFP, _, _, _ = mwFullPipe_opt
            _, _, vaeAlways = mwCkptFP

        ltnt = {"samples":vaeAlways.encode(image[:,:,:,:3])}

        return ((ltnt, wS, hS, wC, hC),)
    

class mwImage_VertSqzExpnd:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "images": ("IMAGE", ),
                "yTop": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "yBtm": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "sqzDelta_u": ("INT", {"default": 0, "min": -1024, "max": 1024, "step": 1}),
                "scaleMethod": (moh.upscale_methods,),
            },
            "optional": {
                "sPipeS_opt": ("MWSIZEPIPE",),
            }
            }
    RETURN_TYPES = ("IMAGE", "MWIVSEPIPE", "MWSIZEPIPE", "MWSIZEPIPE")
    RETURN_NAMES = ("image", "ivsePipe", "sPipeS_updtd", "sPipeC")
    FUNCTION = "mwIVSEBAT"
    CATEGORY = "mohwag/manip"

    def mwIVSE(self, image, sqzDelta_u, scaleMethod, sqzRngH, sqzdH, y1, y2_0, y3_0, x1, y2_1, y3_1):
        if y1 > 0:
            imgTop = imageCropMethod(image, x1, y1, 0, 0)[0]
        imgMdlSqzd = imageScaleMethod(imageCropMethod(image, x1, sqzRngH, 0, y1)[0], scaleMethod, x1, sqzdH, "disabled")[0]
        if y2_0 < y3_0:
            imgBtm = imageCropMethod(image, x1, y3_0 - y2_0, 0, y2_0)[0]
        if sqzDelta_u < 0:
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
        return addBtmImg

    def mwIVSEBAT(self, images, yTop, yBtm, sqzDelta_u, scaleMethod, sPipeS_opt = None):
        inputW, inputH = mwImageTo_sPipeMethod(images[0])[0]

        if sPipeS_opt != None:
            sPipeStart = sPipeS_opt
        else:
            sPipeStart = inputW, inputH

        if sqzDelta_u == 0:
            return (images, (images, sPipeStart, yTop), sPipeStart, (inputW, inputH))

        startW, startH = sPipeStart

        unitVal = int(8 * inputW / startW)
        yTopMod = int(unitVal * (yTop / 8))
        yBtmMod = int(unitVal * (yBtm / 8))

        sqzVertRng_units = int((yBtmMod - yTopMod) // unitVal)

        sqzRngH = int(sqzVertRng_units * unitVal)
        sqzdH_units = sqzVertRng_units + sqzDelta_u
        sqzdH = int(sqzdH_units * unitVal)
        sqzDeltaH = sqzdH - sqzRngH
        y1 = int(yTopMod)
        y2_0 = int(yBtmMod)
        y3_0 = int(inputH)
        x1 = int(inputW)
        y2_1 = int(yTopMod + sqzdH)
        y3_1 = y3_0 + sqzDeltaH

        #########################
        imgCnt = images.shape[0]

        for i in range(imgCnt):
            ###########################
            if imgCnt == 1:
                output_images = self.mwIVSE(images, sqzDelta_u, scaleMethod, sqzRngH, sqzdH, y1, y2_0, y3_0, x1, y2_1, y3_1)   ########## (self, image, sqzDelta_u, scaleMethod, sqzRngH, sqzdH, y1, y2_0, y3_0, x1, y2_1, y3_1)
                #############################
            else:
                for i in range(imgCnt):
                    imgi = imageFromBatchMethod(images, i, 1)[0]
                    output_imgi = self.mwIVSE(imgi, sqzDelta_u, scaleMethod, sqzRngH, sqzdH, y1, y2_0, y3_0, x1, y2_1, y3_1)   ##########
                    if i == 0:
                        output_images = output_imgi
                    else:
                        output_images = imageBatchMethod(output_images, output_imgi)[0]
            #####################

        TotDeltaH_units = sqzDelta_u
        TotDeltaW_units = 0
        TotDeltaW = int(unitVal * TotDeltaW_units)
        TotDeltaH = int(unitVal * TotDeltaH_units)
        outputW = int(inputW + TotDeltaW)
        outputH = int(inputH + TotDeltaH)
        outputS = outputW, outputH
        newStartW = int(startW + (8 * TotDeltaW_units))
        newStartH = int(startH + (8 * TotDeltaH_units))
        newSPipeStart = newStartW, newStartH

        out_ivsePipe = (output_images, newSPipeStart, yTop)   ############
        return (output_images, out_ivsePipe, newSPipeStart, outputS)   ######################

mwImageVertSqzExpndMethod = mwImage_VertSqzExpnd().mwIVSEBAT  #self, image, yTop, yBtm, sqzDelta_u, scaleMethod, sPipeS_opt = None

class mwImage_VertSqzExpndStack:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "ivsePipe": ("MWIVSEPIPE", ),
            "yTop": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "sqzDelta_u": ("INT", {"default": 0, "min": -1024, "max": 1024, "step": 1}),
            "scaleMethod": (moh.upscale_methods,),
            }}
    RETURN_TYPES = ("IMAGE", "MWIVSEPIPE", "MWSIZEPIPE", "MWSIZEPIPE")
    RETURN_NAMES = ("image", "ivsePipe", "sPipeS_updtd", "sPipeC")
    FUNCTION = "mwIVSES"
    CATEGORY = "mohwag/manip"
    def mwIVSES(self, ivsePipe, yTop, sqzDelta_u, scaleMethod):
        image, sPipeStart, yBtm = ivsePipe
        return mwImageVertSqzExpndMethod(image, yTop, yBtm, sqzDelta_u, scaleMethod, sPipeStart)

class mwImageTaper_1vert:
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

        out_ivt4Pipe = imgFinal, y1, xbl, xbm0, xbr, xbmd
        return (imgFinal, out_ivt4Pipe)

mwImageTaper1VertMethod = mwImageTaper_1vert().mwIVTFO

class mwImageTaperStack_1vert:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
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
    def mwIVTFOP(self, ivt4Pipe, y1, xbl, xbm0, xbr, xbmd, upscaleMethod):
        image, y0, xtl, xtm0, xtr, xtmd = ivt4Pipe
        out_taper = mwImageTaper1VertMethod(image, y0, xtl, xtm0, xtr, y1, xbl, xbm0, xbr, xtmd, xbmd, upscaleMethod)[0]
        out_ivt4Pipe = y1, xbl, xbm0, xbr, xbmd
        return(out_taper, out_ivt4Pipe)


class mwImageTaper_2vert:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "images": ("IMAGE", ),
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
    RETURN_NAMES = ("images", "ivt6Pipe")
    FUNCTION = "mwIVTSIBAT"
    CATEGORY = "mohwag/manip"

    def mwIVTSI(self, image, y0, upscaleMethod, midH, interL, slopeL, interML0, slopeML0, interMR0, slopeMR0, interR, slopeR, interML1, slopeML1, interMR1, slopeMR1):
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
        return imgFinal

    def mwIVTSIBAT(self, images, y0, xtl, xtml0, xtmr0, xtr, y1, xbl, xbml0, xbmr0, xbr, xtmld, xtmrd, xbmld, xbmrd, upscaleMethod):
        xtml1 = xtml0 + xtmld
        xtmr1 = xtmr0 + xtmrd
        xbml1 = xbml0 + xbmld
        xbmr1 = xbmr0 + xbmrd

        modImage = Image.fromarray(np.clip(255. * images[0].cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
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

        imgCnt = images.shape[0]
        if imgCnt == 1:
            output_images = self.mwIVTSI(images, y0, upscaleMethod, midH, interL, slopeL, interML0, slopeML0, interMR0, slopeMR0, interR, slopeR, interML1, slopeML1, interMR1, slopeMR1)   ##########

        else:
            for i in range(imgCnt):
                imgi = imageFromBatchMethod(images, i, 1)[0]
                output_imgi = self.mwIVTSI(imgi, y0, upscaleMethod, midH, interL, slopeL, interML0, slopeML0, interMR0, slopeMR0, interR, slopeR, interML1, slopeML1, interMR1, slopeMR1)   ##########
                if i == 0:
                    output_images = output_imgi
                else:
                    output_images = imageBatchMethod(output_images, output_imgi)[0]

        out_ivt6Pipe = output_images, y1, xbl, xbml0, xbmr0, xbr, xbmld, xbmrd
        return (output_images, out_ivt6Pipe)


mwImageTaper2VertMethod = mwImageTaper_2vert().mwIVTSIBAT

class mwImageTaperStack_2vert:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            #"image": ("IMAGE",),
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
    def mwIVTSIP(self, ivt6Pipe, y1, xbl, xbml0, xbmr0, xbr, xbmld, xbmrd, upscaleMethod):
        image, y0, xtl, xtml0, xtmr0, xtr, xtmld, xtmrd = ivt6Pipe
        out_taper = mwImageTaper2VertMethod(image, y0, xtl, xtml0, xtmr0, xtr, y1, xbl, xbml0, xbmr0, xbr, xtmld, xtmrd, xbmld, xbmrd, upscaleMethod)[0]
        out_ivt6Pipe = out_taper, y1, xbl, xbml0, xbmr0, xbr, xbmld, xbmrd
        return(out_taper, out_ivt6Pipe)


class mwLtntPipeBranch1: #WAS_Latent_Size_To_Number
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "ltntPipe": ("MWLTNTPIPE", ),
            }}
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent", "sPipeS", "sPipeC")
    FUNCTION = "mwLPBO"
    CATEGORY = "mohwag/LatentPipe"
    def mwLPBO(self, ltntPipe):
        ltnt, initialW, initialH, outputW, outputH = ltntPipe
        return (ltnt, (initialW, initialH), (outputW, outputH))
    

class mwLtntPipe_View: #WAS_Latent_Size_To_Number    ###############################################################
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


def condRow_createAvgList(astrng:str):
    leftMarker = "<<<"
    rightMarker = ">>>"
    RepeatMarker = "<<>>"
    specThingsSeparator = ";"
    specThingWgtSeparator = ","

    splt_L_MR = astrng.split(leftMarker, 1)
    if len(splt_L_MR) == 1:
        return ([astrng], [1.0], [1.0])
    spltL = splt_L_MR[0]
    spltMR = splt_L_MR[1]
    splt_M_R = spltMR.split(rightMarker,1)
    spltM = splt_M_R[0]
    spltR = splt_M_R[1]
    spltR_repeat = spltR.split(RepeatMarker,-1)
    spltEntries = spltM.split(specThingsSeparator,-1)

    spltEntriesThing = []
    spltEntriesWgt = []
    for x in spltEntries:
        spltThing = x.split(specThingWgtSeparator,1)
        spltEntriesThing.append(spltThing[0])
        spltEntriesWgt.append(spltThing[1])
    spltEntriesWgtFlt = []
    wgtTotal = 0
    for x in spltEntriesWgt:
        fltThing = float(x)
        spltEntriesWgtFlt.append(fltThing)
        wgtTotal = wgtTotal + fltThing
    spltEntriesWgtFin = []
    for x in spltEntriesWgtFlt:
        spltEntriesWgtFin.append(x/wgtTotal)
    spltEntriesWgtFinAccom = []
    spltEntriesWgtFinRoll = 0
    spltEntriesWgtFinRollNext = 0
    for i in range(len(spltEntriesWgtFin) - 1):
        spltEntriesWgtFinRoll = spltEntriesWgtFinRoll + spltEntriesWgtFin[i]
        spltEntriesWgtFinRollNext = spltEntriesWgtFinRoll + spltEntriesWgtFin[i+1]
        accomThing = spltEntriesWgtFinRoll / spltEntriesWgtFinRollNext
        spltEntriesWgtFinAccom = spltEntriesWgtFinAccom + [accomThing]
    finTemplate = [spltL] + spltR_repeat
    finList = []
    for x in spltEntriesThing:
        finPart = ""
        for i in range(len(finTemplate) - 1):
            finPart = finPart + finTemplate[i] + x
        finPart = finPart + finTemplate[-1]
        finList = finList + [finPart]
    spltEntriesWgtFinAccom = spltEntriesWgtFinAccom + [0.0]
    return (finList, spltEntriesWgtFin, spltEntriesWgtFinAccom)

def prepForCond_oneSide(bprompt:list):
    aresult = []
    for x in bprompt:
        getRslt = condRow_createAvgList(x)
        rsltText = getRslt[0]
        rsltVal = getRslt[2]
        rowRslt = [rsltText, rsltVal]
        aresult.append(rowRslt)
    return aresult


""" def condList_createLR(ablock:str, adelim:str) -> list[str]:
    alist = ablock.splitlines()
    for x in ["", " ", "  "]:
        while(x in alist):
            alist.remove(x)
    acondListL = []
    acondListR = []
    for x in alist:
        athingy = x.split(adelim)
        acondListL.append(athingy[0])
        acondListR.append(athingy[-1])
    return (acondListL, acondListR) """

def condList_createLR(ablock:str, adelim:str, rmvSectionBlank=False) -> list[str]:  #### 0,0 ####
    alist = ablock.splitlines()
    for x in ["", " "]:#for x in ["", " ", "  "]:
        while(x in alist):
            alist.remove(x)
    acondListL = []
    acondListR = []
    for x in alist:
        athingy = x.split(adelim)
        acondListL.append(athingy[0])
        acondListR.append(athingy[-1])
    if rmvSectionBlank:
        for x in ["", " "]:
            while(x in acondListL):
                acondListL.remove(x)
            while(x in acondListR):
                acondListR.remove(x)        
    return (acondListL, acondListR)

def prepForCond(aprompt:str):
    acondListL, acondListR = condList_createLR(aprompt,"||", True)
    return prepForCond_oneSide(acondListL), prepForCond_oneSide(acondListR)



def func_mwCondPrepXLSide(aclip, amwCondS, awidth, aheight, acrop_w, acrop_h, atarget_width, atarget_height):
    bcondType, bcondPrepdList = amwCondS
    lenPrepdList = len(bcondPrepdList)
    if bcondType == "commaSeparated":
        fullTextList = []
        for h in range(lenPrepdList):
            #fullTextList.append(bcondPrepdList[h][0])
            fullTextList = fullTextList + bcondPrepdList[h][0]
        textConcF = ", ".join(map(str, fullTextList))
        outCond = clipTextEncodeSDXL_Method(aclip, awidth, aheight, acrop_w, acrop_h, atarget_width, atarget_height, textConcF, textConcF)[0] #(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l)
        return outCond
    elif bcondType == "condConcat":
        aCondList = []
        for i in range(lenPrepdList):
            activeList = bcondPrepdList[i]
            activeTextList = activeList[0]
            iterCnt = len(activeTextList)
            if iterCnt == 1:
                rowText = activeList[0][0]
                rowCond = clipTextEncodeSDXL_Method(aclip, awidth, aheight, acrop_w, acrop_h, atarget_width, atarget_height, rowText, rowText)[0]
            else:
                activeWgtList = activeList[1]                
                aSubCondList = []
                for j in range(iterCnt):
                    subRowText = activeTextList[j]
                    subRowCond = clipTextEncodeSDXL_Method(aclip, awidth, aheight, acrop_w, acrop_h, atarget_width, atarget_height, subRowText, subRowText)[0]
                    aSubCondList.append(subRowCond)
                avgCond = aSubCondList[0]
                for k in range(iterCnt - 1):
                    avgCond = conditioningAverageMethod(avgCond, aSubCondList[k+1], activeWgtList[k])[0]
                rowCond = avgCond
            aCondList.append(rowCond)
        outCond = aCondList[0]
        if lenPrepdList != 1:
            for l in range(lenPrepdList - 1):
                outCond = conditioningConcatMethod(outCond, aCondList[l+1])[0]
        return outCond
    else:
        return None


def func_mwCondPrepXL(aclip, amwCondLR, awidth, aheight, acrop_w, acrop_h, atarget_width, atarget_height):
    apromptL_condType, apromptR_condType, apromptCmb_condType, apromptCmb_wgtR, acondPrepdListL, acondPrepdListR = amwCondLR

    if apromptCmb_wgtR != 0:
        finCondR = func_mwCondPrepXLSide(aclip, (apromptR_condType, acondPrepdListR), awidth, aheight, acrop_w, acrop_h, atarget_width, atarget_height)
    if apromptCmb_wgtR != 1:
        finCondL = func_mwCondPrepXLSide(aclip, (apromptL_condType, acondPrepdListL), awidth, aheight, acrop_w, acrop_h, atarget_width, atarget_height)

    if apromptCmb_wgtR == 1:
        finCondFin = conditioningSetAreaStrengthMethod(finCondR, 1)[0]
    elif apromptCmb_wgtR == 0:
        finCondFin = conditioningSetAreaStrengthMethod(finCondL, 1)[0]
    else:
        if apromptCmb_condType == "condComb":
            finCondL_wgtd = conditioningSetAreaStrengthMethod(finCondL, 1 - apromptCmb_wgtR)[0]
            finCondR_wgtd = conditioningSetAreaStrengthMethod(finCondR, apromptCmb_wgtR)[0]
            finCondFin = conditioningCombineMethod(finCondL_wgtd, finCondR_wgtd)[0]
        elif apromptCmb_condType == "condAvg":
            finCondFin = conditioningAverageMethod(finCondL, finCondR, 1 - apromptCmb_wgtR)[0]
        elif apromptCmb_condType == "condConcat":
            finCondFin = conditioningConcatMethod(finCondL, finCondR)[0]
        else:
            None

    return finCondFin

class mwCondXLa:
    iPrompt_CondTypes = ["commaSeparated", "condConcat"]
    cmbPrompt_CondTypes = ["condAvg", "condConcat", "condComb"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "clip": ("CLIP", ),
            "sPipe_actual": ("MWSIZEPIPE",),
            "promptL_condType": (moh.iPrompt_CondTypes, {"default": "commaSeparated"}),
            "promptR_condType": (moh.iPrompt_CondTypes, {"default": "commaSeparated"}),
            "promptCmb_condType": (moh.cmbPrompt_CondTypes, {"default": "condAvg"}),
            "promptCmb_wgtR": ("FLOAT", {"default": 1.00, "min": 0, "max": 1, "step": 0.05}),
            "refImgsPixelMult": ("FLOAT", {"default": 1.00, "min": 0, "max": 25, "step": 0.25}),  
            "condText": ("STRING", {"multiline": True}),
            }}
    RETURN_TYPES = ("CONDITIONING", "CLIP", "MWSIZEPIPE", "STRING")
    RETURN_NAMES = ("conditioning", "clip", "sPipe_actual", "condText")
    FUNCTION = "mwCXLa"
    CATEGORY = "mohwag/Conditioning"
    def mwCXLa(self, clip, sPipe_actual, promptL_condType, promptR_condType, promptCmb_condType, promptCmb_wgtR, refImgsPixelMult, condText):
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
        actualPx = startW * startH
        refImgsPx = actualPx * refImgsPixelMult
        refImgRat = startW / startH
        refImgsH = (refImgsPx / refImgRat)**(1/2)
        refImgsW = refImgsPx / refImgsH
        start_width = int(round(refImgsW,0))
        start_height = int(round(refImgsH,0))

        #GET TARGET SIZE AND CROP
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

        condPrepdListL, condPrepdListR = prepForCond(condText)
        condReturn = func_mwCondPrepXL(clip, (promptL_condType, promptR_condType, promptCmb_condType, promptCmb_wgtR, condPrepdListL, condPrepdListR), start_width, start_height, cropW, cropH, int(targetW), int(targetH))

        return (condReturn, clip, sPipe_actual, condText)




class mwCondXLadv:
    iPrompt_CondTypes = ["commaSeparated", "condConcat"]
    cmbPrompt_CondTypes = ["condAvg", "condConcat", "condComb"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "clip": ("CLIP", ),
            "sPipe_actual": ("MWSIZEPIPE",),
            "prompt0_LorR": (["L", "R"],),
            "prompt0_condType": (moh.iPrompt_CondTypes, {"default": "commaSeparated"}),
            "prompt1_LorR": (["L", "R"],),
            "prompt1_condType": (moh.iPrompt_CondTypes, {"default": "commaSeparated"}),
            "promptCmb_condType": (moh.cmbPrompt_CondTypes, {"default": "condAvg"}),
            "promptCmb_wgt1": ("FLOAT", {"default": 1.00, "min": 0, "max": 1, "step": 0.05}),
            "refImgsPixelMult": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 25.00, "step": 0.05}),
            "stop_at_clip_layer": ("INT", {"default": -2, "min": -24, "max": -1, "step": 1}),
            "TimeStepRange_startStep": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
            "TimeStepRange_endStep": ("INT", {"default": 48, "min": 0, "max": 256, "step": 1}),
            "TimeStepRange_steps": ("INT", {"default": 48, "min": 0, "max": 256, "step": 1}),
            "condText": ("STRING", {"multiline": True}),
            }}
    RETURN_TYPES = ("CONDITIONING", "MWCONDXLADVPIPE")
    RETURN_NAMES = ("conditioning", "condXLadvPipe")
    FUNCTION = "mwCXLadv"
    CATEGORY = "mohwag/Conditioning"
    def mwCXLadv(self, clip, sPipe_actual, prompt0_LorR, prompt0_condType, prompt1_LorR, prompt1_condType, promptCmb_condType, promptCmb_wgt1, refImgsPixelMult, stop_at_clip_layer, TimeStepRange_startStep, TimeStepRange_endStep, TimeStepRange_steps, condText):

        aspectRatList = [0.25, 0.26, 0.27, 0.28, 0.32, 0.33, 0.35, 0.4, 0.42, 0.48, 0.5, 0.52, 0.57, 0.6, 0.68, 0.72, 0.78, 0.82, 0.88, 0.94, 1, 1.07, 1.13, 1.21, 1.29, 1.38, 1.46, 1.67, 1.75, 2, 2.09, 2.4, 2.5, 2.89, 3, 3.11, 3.63, 3.75, 3.88, 4]
        targetW_list = [512, 512, 512, 512, 576, 576, 576, 640, 640, 704, 704, 704, 768, 768, 832, 832, 896, 896, 960, 960, 1024, 1024, 1088, 1088, 1152, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048]
        targetH_list = [2048, 1984, 1920, 1856, 1792, 1728, 1664, 1600, 1536, 1472, 1408, 1344, 1344, 1280, 1216, 1152, 1152, 1088, 1088, 1024, 1024, 960, 960, 896, 896, 832, 832, 768, 768, 704, 704, 640, 640, 576, 576, 576, 512, 512, 512, 512]
        
        #TimeStepRange_startStep = 0
        
        """ actualW, actualH = sPipe_actual

        #GET START IMAGE SIZE
        ##start ratio and size before upscaling
        rats = round((np.exp(1))**(np.log(actualW / actualH) * (2/3)),2)
        testListrs = [abs(round(x - rats,3)) for x in aspectRatList]
        testMinrs = min(testListrs)
        testMinLocrs = testListrs.index(testMinrs)
        startW = targetW_list[testMinLocrs]
        startH = targetH_list[testMinLocrs]

        ##start ratio and size final (i.e. w/ upscaling)
        actualPx = startW * startH
        refImgsPx = actualPx * refImgsPixelMult
        refImgRat = startW / startH
        refImgsH = (refImgsPx / refImgRat)**(1/2)
        refImgsW = refImgsPx / refImgsH
        start_width = int(round(refImgsW,0))
        start_height = int(round(refImgsH,0))

        #GET TARGET SIZE AND CROP
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
            cropW = int(64) """

        start_width, start_height, cropW, cropH, targetW, targetH = sdxl_cond_sizes(sPipe_actual, refImgsPixelMult)

        condPrepdListL, condPrepdListR = prepForCond(condText)
        if prompt0_LorR == "L":
            condPrepdList0 = condPrepdListL
        else:
            condPrepdList0 = condPrepdListR
        if prompt1_LorR == "L":
            condPrepdList1 = condPrepdListL
        else:
            condPrepdList1 = condPrepdListR

        modClip = clipSetLastLayerMethod(clip, stop_at_clip_layer)[0]  #(self, clip, stop_at_clip_layer)
        condGen = func_mwCondPrepXL(modClip, (prompt0_condType, prompt1_condType, promptCmb_condType, promptCmb_wgt1, condPrepdList0, condPrepdList1), start_width, start_height, cropW, cropH, int(targetW), int(targetH))
        
        if TimeStepRange_startStep == 0:
            condStart = 0  #TimeStepRange_startStep / TimeStepRange_steps
        else:
            condStart = (TimeStepRange_startStep / TimeStepRange_steps) #+ 0.001
        condEnd = min(1.000, (TimeStepRange_endStep / TimeStepRange_steps))
        condReturn = conditioningSetTimestepRangeMethod(condGen, condStart, condEnd)[0] #(self, conditioning, start, end)
        
        out_condXLadvPipe = (condReturn, clip, sPipe_actual, condText, TimeStepRange_endStep, TimeStepRange_steps)
        
        return (condReturn, out_condXLadvPipe)

class mwCondXLadvStack:
    iPrompt_CondTypes = ["commaSeparated", "condConcat"]
    cmbPrompt_CondTypes = ["condAvg", "condConcat", "condComb"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "condXLadvPipe": ("MWCONDXLADVPIPE", ),
            "prompt0_LorR": (["L", "R"],),
            "prompt0_condType": (moh.iPrompt_CondTypes, {"default": "commaSeparated"}),
            "prompt1_LorR": (["L", "R"],),
            "prompt1_condType": (moh.iPrompt_CondTypes, {"default": "commaSeparated"}),
            "promptCmb_condType": (moh.cmbPrompt_CondTypes, {"default": "condAvg"}),
            "promptCmb_wgt1": ("FLOAT", {"default": 1.00, "min": 0, "max": 1, "step": 0.05}),
            "refImgsPixelMult": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 25.00, "step": 0.05}),
            "stop_at_clip_layer": ("INT", {"default": -2, "min": -24, "max": -1, "step": 1}),
            "TimeStepRange_endStep": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
            },
            "optional": {
                "TSR_startStep_opt": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "condText_opt": ("STRING", {"multiline": True}),
            }}
    RETURN_TYPES = ("CONDITIONING", "MWCONDXLADVPIPE")
    RETURN_NAMES = ("conditioning", "condXLadvPipe")
    FUNCTION = "mwCXLadvS"
    CATEGORY = "mohwag/Conditioning"
    def mwCXLadvS(self, condXLadvPipe, prompt0_LorR, prompt0_condType, prompt1_LorR, prompt1_condType, promptCmb_condType, promptCmb_wgt1, refImgsPixelMult, stop_at_clip_layer, TimeStepRange_endStep, TSR_startStep_opt = None, condText_opt = None):
        
        condReturnPipe, clip, sPipe_actual, condTextPipe, TimeStepRange_startStepPipe, TimeStepRange_steps = condXLadvPipe

        if condText_opt == None:
            condText = condTextPipe
        else:
            condText = condText_opt

        if TSR_startStep_opt == None:
            TimeStepRange_startStep = TimeStepRange_startStepPipe
        else:
            TimeStepRange_startStep = TSR_startStep_opt
        
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
        actualPx = startW * startH
        refImgsPx = actualPx * refImgsPixelMult
        refImgRat = startW / startH
        refImgsH = (refImgsPx / refImgRat)**(1/2)
        refImgsW = refImgsPx / refImgsH
        start_width = int(round(refImgsW,0))
        start_height = int(round(refImgsH,0))

        #GET TARGET SIZE AND CROP
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

        condPrepdListL, condPrepdListR = prepForCond(condText)
        if prompt0_LorR == "L":
            condPrepdList0 = condPrepdListL
        else:
            condPrepdList0 = condPrepdListR
        if prompt1_LorR == "L":
            condPrepdList1 = condPrepdListL
        else:
            condPrepdList1 = condPrepdListR

        modClip = clipSetLastLayerMethod(clip, stop_at_clip_layer)[0]  #(self, clip, stop_at_clip_layer)
        condGen = func_mwCondPrepXL(modClip, (prompt0_condType, prompt1_condType, promptCmb_condType, promptCmb_wgt1, condPrepdList0, condPrepdList1), start_width, start_height, cropW, cropH, int(targetW), int(targetH))
        
        condStart = TimeStepRange_startStep / TimeStepRange_steps
        condEnd = min(1.000, (TimeStepRange_endStep / TimeStepRange_steps) + 0.001)
        condReturnSingle = conditioningSetTimestepRangeMethod(condGen, condStart, condEnd)[0] #(self, conditioning, start, end)
        
        condReturn = conditioningCombineMethod(condReturnSingle, condReturnPipe)[0]

        out_condXLadvPipe = (condReturn, clip, sPipe_actual, condText, TimeStepRange_endStep, TimeStepRange_steps)
        
        return (condReturn, out_condXLadvPipe)


class mwCondListDictKeys:
    iPrompt_CondTypes = ["commaSeparated", "condConcat"]
    cmbPrompt_CondTypes = ["condAvg", "condConcat", "condComb"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {
            "cond": ("CONDITIONING", ),
            }}
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("dictList",)
    FUNCTION = "mwCLDK"
    CATEGORY = "mohwag/Conditioning"
    def mwCLDK(self, cond):
        dictList_out = []
        for i in range(len(cond)):
            dictList_i = list(cond[i][1].keys())
            dictList_out.append(dictList_i)
        return (dictList_out,)


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

    RETURN_TYPES = ("MWLTNTPIPE", "LATENT", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("ltntPipe", "latent", "image", "stepEnd", "steps")
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
        #startS = startW, startH
        return (ltntPipe, ltntOut[0], imgOut[0], stepEnd, steps)


mwFullPipe_KSAStartMethod = mwFullPipe_KSAStart().mwFPKSAS   #(self, fullPipe, positive, negative, cfg, stepEnd, steps, seed_deltaVsOrig, add_noise, return_with_leftover_noise, startS, denoise=1.0):

""" class mwFullPipeTuple_KSAStart2:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
    @classmethod
    def INPUT_TYPES(moh):
        return {"required":
                    {"fullPipeTuple": ("MWFULLPIPETUPLE",),
                    "positiveTuple": ("CONDTUPLE", ),
                    "negativeTuple": ("CONDTUPLE", ),
                    "cfg": ("FLOAT", {"default": 8.0, "min": -100.0, "max": 100.0, "step":0.5, "round": 0.01}),
                    #"stepEnd": ("INT", {"default": 48, "min": 0, "max": 10000}),
                    "steps": ("INT", {"default": 48, "min": 1, "max": 10000}),
                    "seed_deltaVsOrig": ("INT", {"default": 0, "min": 0, "max": 10000, "defaultBehavior": "input"}),
                    "add_noise": (["enable", "disable"], ),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "startS": ("MWSIZEPIPE",),
                    "list_FP_Cond_StepEnd": ("STRING", {"multiline": True}),
                    },
                }

    RETURN_TYPES = ("MWLTNTPIPE", "LATENT", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("ltntPipe", "latent", "image", "stepEnd", "steps")
    FUNCTION = "mwFPTKSAST"
    CATEGORY = "mohwag/Sampling"

    def mwFPTKSAST(self, fullPipeTuple, positiveTuple, negativeTuple, cfg, steps, seed_deltaVsOrig, add_noise, return_with_leftover_noise, startS, list_FP_Cond_StepEnd):

        schedList = list_FP_Cond_StepEnd.splitlines()
        for x in ["", " ", "  "]:
            while(x in schedList):
                schedList.remove(x)

        schedRef = [y.split(" ",-1) for y in schedList]
        schedRefLen = len(schedRef)

        for i in range(schedRefLen):
            for j in range(len(schedRef[i])):
                schedRef[i][j] = int(schedRef[i][j])        
        
        if schedRefLen > 1:
            int_return_with_leftover_noise = "enable"
        else:
            int_return_with_leftover_noise = return_with_leftover_noise

        arun = mwFullPipe_KSAStartMethod(fullPipeTuple[schedRef[0][0]], positiveTuple[schedRef[0][1]], negativeTuple[schedRef[0][1]], cfg, schedRef[0][2], steps, seed_deltaVsOrig, add_noise, int_return_with_leftover_noise, startS)
        
        if schedRefLen == 1:
            ltntPipeOut, ltntOut, imgOut, stepEnd, steps = arun
        else:
            ltntPipeOut, _, _, stepEnd, _ = arun
        
        #progImages = imgOut

        if schedRefLen > 1:
            add_noise = "disable" 
            aoldMult8 = 8
            amult8 = 8
            for i in range(schedRefLen):
                if i > 0:
                    if len(schedRef[i]) == 4:
                        amult8 = schedRef[i][3]
                    else:
                        amult8 = aoldMult8

                    if amult8 > aoldMult8:
                        stepEnd = stepEnd - 1

                    if i < schedRefLen - 1:
                        arun = mwFullPipe_KSAMethod(fullPipeTuple[schedRef[i][0]], ltntPipeOut, positiveTuple[schedRef[i][1]], negativeTuple[schedRef[i][1]], cfg, stepEnd, schedRef[i][2], steps, seed_deltaVsOrig, add_noise, int_return_with_leftover_noise, amult8, "bicubic")
                        #(self, fullPipe, ltntPipe, positive, negative, cfg, stepStart, stepEnd, steps, seed_deltaVsOrig, add_noise, return_with_leftover_noise, multOver8, upscale_method, denoise=1.0):
                        ltntPipeOut, _, _, stepEnd, _ = arun
                        #progImages = imageBatchMethod(progImages, imgOut)[0]
                        aoldMult8 = amult8
                    else:
                        int_return_with_leftover_noise = return_with_leftover_noise
                        arun = mwFullPipe_KSAMethod(fullPipeTuple[schedRef[i][0]], ltntPipeOut, positiveTuple[schedRef[i][1]], negativeTuple[schedRef[i][1]], cfg, stepEnd, schedRef[i][2], steps, seed_deltaVsOrig, add_noise, int_return_with_leftover_noise, amult8, "bicubic")
                        #(self, fullPipe, ltntPipe, positive, negative, cfg, stepStart, stepEnd, steps, seed_deltaVsOrig, add_noise, return_with_leftover_noise, multOver8, upscale_method, denoise=1.0):
                        ltntPipeOut, ltntOut, imgOut, stepEnd, _ = arun
                        #progImages = imageBatchMethod(progImages, imgOut)[0]
                        #aoldMult8 = amult8                        
                   
        return (ltntPipeOut, ltntOut, imgOut, stepEnd, steps) #, progImages)

class mwFullPipeTuple_KSAStart3:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
    @classmethod
    def INPUT_TYPES(moh):
        return {"required":
                    {"fullPipeTuple": ("MWFULLPIPETUPLE",),
                    "positiveTuple": ("CONDTUPLE", ),
                    "negativeTuple": ("CONDTUPLE", ),
                    "cfg": ("FLOAT", {"default": 8.0, "min": -100.0, "max": 100.0, "step":0.5, "round": 0.01}),
                    #"stepEnd": ("INT", {"default": 48, "min": 0, "max": 10000}),
                    "steps": ("INT", {"default": 48, "min": 1, "max": 10000}),
                    "seed_deltaVsOrig": ("INT", {"default": 0, "min": 0, "max": 10000, "defaultBehavior": "input"}),
                    "add_noise": (["enable", "disable"], ),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "startS": ("MWSIZEPIPE",),
                    "list_FP_Cond_StepEnd": ("STRING", {"multiline": True}),
                    "extraStepsOnZoom": ("INT", {"default": 0, "min": 0, "max": 64, "defaultBehavior": "input"}),
                    "return_with_noise_intermed": (["disable", "enable"], ),
                    },
                }

    RETURN_TYPES = ("MWLTNTPIPE", "LATENT", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("ltntPipe", "latent", "image", "stepEnd", "steps")
    FUNCTION = "mwFPTKSASTH"
    CATEGORY = "mohwag/Sampling"

    def mwFPTKSASTH(self, fullPipeTuple, positiveTuple, negativeTuple, cfg, steps, seed_deltaVsOrig, add_noise, return_with_leftover_noise, startS, list_FP_Cond_StepEnd, extraStepsOnZoom, return_with_noise_intermed):

        schedList = list_FP_Cond_StepEnd.splitlines()
        for x in ["", " ", "  "]:
            while(x in schedList):
                schedList.remove(x)

        schedRef = [y.split(" ",-1) for y in schedList]
        schedRefLen = len(schedRef)

        for i in range(schedRefLen):
            for j in range(len(schedRef[i])):
                schedRef[i][j] = int(schedRef[i][j])        
        
        if schedRefLen > 1:
            int_return_with_leftover_noise = "enable"
        else:
            int_return_with_leftover_noise = return_with_leftover_noise

            #def mwFPKSAS(self, fullPipe, positive, negative, cfg, stepEnd, steps, seed_deltaVsOrig, add_noise, return_with_leftover_noise, startS, denoise=1.0):

        arun = mwFullPipe_KSAStartMethod(fullPipeTuple[schedRef[0][0]], positiveTuple[schedRef[0][1]], negativeTuple[schedRef[0][1]], cfg, schedRef[0][2], steps, seed_deltaVsOrig, add_noise, return_with_noise_intermed, startS)
        
        if schedRefLen == 1:
            ltntPipeOut, ltntOut, imgOut, stepEnd, steps = arun   #ltntPipe = runLtntOut[0], startW, startH, outputW, outputH
        else:
            ltntPipeOut, _, _, stepEnd, _ = arun
        
        #progImages = imgOut

        if schedRefLen > 1:
            add_noise = "disable" 
            aoldMult8 = 8
            amult8 = 8
            for i in range(schedRefLen):
                if i > 0:
                    if len(schedRef[i]) == 4:
                        amult8 = schedRef[i][3]
                    else:
                        amult8 = aoldMult8

                    if amult8 > aoldMult8:
                        stepEnd = stepEnd - extraStepsOnZoom

                    if i < schedRefLen - 1:
                        arun = mwFullPipe_KSAMethod(fullPipeTuple[schedRef[i][0]], ltntPipeOut, positiveTuple[schedRef[i][1]], negativeTuple[schedRef[i][1]], cfg, stepEnd, schedRef[i][2], steps, seed_deltaVsOrig, add_noise, return_with_noise_intermed, amult8, "bicubic")
                        #(self, fullPipe, ltntPipe, positive, negative, cfg, stepStart, stepEnd, steps, seed_deltaVsOrig, add_noise, return_with_leftover_noise, multOver8, upscale_method, denoise=1.0):
                        ltntPipeOut, _, _, stepEnd, _ = arun
                        #progImages = imageBatchMethod(progImages, imgOut)[0]
                        aoldMult8 = amult8
                    else:
                        int_return_with_leftover_noise = return_with_leftover_noise
                        arun = mwFullPipe_KSAMethod(fullPipeTuple[schedRef[i][0]], ltntPipeOut, positiveTuple[schedRef[i][1]], negativeTuple[schedRef[i][1]], cfg, stepEnd, schedRef[i][2], steps, seed_deltaVsOrig, add_noise, int_return_with_leftover_noise, amult8, "bicubic")
                        #(self, fullPipe, ltntPipe, positive, negative, cfg, stepStart, stepEnd, steps, seed_deltaVsOrig, add_noise, return_with_leftover_noise, multOver8, upscale_method, denoise=1.0):
                        ltntPipeOut, ltntOut, imgOut, stepEnd, _ = arun
                        #progImages = imageBatchMethod(progImages, imgOut)[0]
                        #aoldMult8 = amult8                        
                   
        return (ltntPipeOut, ltntOut, imgOut, stepEnd, steps) #, progImages) """


class mwFullPipeTuple_KSA:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {"required":
                    {"fullPipeTuple": ("MWFULLPIPETUPLE",),
                    "positiveTuple": ("CONDTUPLE", ),
                    "negativeTuple": ("CONDTUPLE", ),
                    "cfg": ("FLOAT", {"default": 8.0, "min": -100.0, "max": 100.0, "step":0.5, "round": 0.01}),
                    #"stepEnd": ("INT", {"default": 48, "min": 0, "max": 10000}),
                    #"steps": ("INT", {"default": 48, "min": 1, "max": 10000}),
                    "seed_deltaVsOrig": ("INT", {"default": 0, "min": 0, "max": 10000, "defaultBehavior": "input"}),
                    "disable_noise_start": ("BOOLEAN", {"default": False}),
                    "force_full_denoise_intermed": ("BOOLEAN", {"default": True}),
                    "disable_noise_intermed": ("BOOLEAN", {"default": False}),
                    "force_full_denoise_end": ("BOOLEAN", {"default": False}),
                    #"addtl_steps_on_zoom": ("INT", {"default": 0, "min": 0, "max": 64}),
                    "zoom_by_img_or_ltnt": (["img", "ltnt"], ),
                    "upscale_method": (moh.upscale_methods,),
                    "list_FP_Cond_StepStart_StepEnd_Steps_Mult8": ("STRING", {"multiline": True}),
                    },
                "optional":
                    {"sPipeS": ("MWSIZEPIPE",),
                    "ltntPipe": ("MWLTNTPIPE", ),
                    },
                }

    RETURN_TYPES = ("MWLTNTPIPE", "LATENT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("ltntPipe", "latent", "image", "progImages")
    FUNCTION = "mwFPTKSA"
    CATEGORY = "mohwag/Sampling"

    #mwFullPipe = mwCkpt, seed, sampler_name, scheduler
    #mwCkptNew = model, clip1, vaeAlways
    #ltntPipe = ltnt, startW, startH, outputW, outputH

    def mwFPTKSA(self, fullPipeTuple, positiveTuple, negativeTuple, cfg, seed_deltaVsOrig, disable_noise_start, force_full_denoise_intermed, disable_noise_intermed, force_full_denoise_end, zoom_by_img_or_ltnt, upscale_method, list_FP_Cond_StepStart_StepEnd_Steps_Mult8, sPipeS=None, ltntPipe=None):
        schedList = list_FP_Cond_StepStart_StepEnd_Steps_Mult8.splitlines()
        for x in ["", " ", "  "]:
            while(x in schedList):
                schedList.remove(x)
        schedRef = [y.split(" ",-1) for y in schedList]
        schedRefLen = len(schedRef)

        for i in range(schedRefLen):
            for j in range(len(schedRef[i])):
                schedRef[i][j] = int(schedRef[i][j])    

        iFullPipe_refnum = schedRef[0][0]
        iFullPipe = fullPipeTuple[iFullPipe_refnum]
        mwCkpt, seed, sampler_name, scheduler = iFullPipe
        model, _, vae = mwCkpt
        seed = seed + seed_deltaVsOrig

        iCond_refnum = schedRef[i][1]
        iCondPos = positiveTuple[iCond_refnum]
        iCondNeg = negativeTuple[iCond_refnum]

        if ltntPipe != None:
            ltnt, startW, startH, outputW, outPutH = ltntPipe
            mult8 = (outputW * 8) // startW
        elif sPipeS != None:
            startW, startH = sPipeS
            outputW, outputH = sPipeS
            ltnt = {"samples":torch.zeros([1, 4, outputH // 8, outputW // 8], device=self.device)}
            mult8 = 8
        else:
            startW = 1024
            startH = 1024
            outputW = 1024
            outputH = 1024
            ltnt = {"samples":torch.zeros([1, 4, outputH // 8, outputW // 8], device=self.device)}
            mult8 = 8

        outImg = vae.decode(ltnt["samples"])
        if mult8 > 8:
            outImgScaled = imageScaleMethod(outImg, upscale_method, startW, startH, "disabled")[0]
        else:
            outImgScaled = outImg

        progImages = outImgScaled
        inImg = outImg
        
        iDisableNoise = disable_noise_start
        iForceFullDenoise = force_full_denoise_intermed

        oldMult8 = mult8

        iFullPipe_refnumOld = iFullPipe_refnum
        iCond_refnumOld = iCond_refnum


        for i in range(schedRefLen):

            iFullPipe_refnum = schedRef[i][0]
            if iFullPipe_refnum != iFullPipe_refnumOld:
                iFullPipe = fullPipeTuple[iFullPipe_refnum]
                model, _, vae = iFullPipe[0]

            iCond_refnum = schedRef[i][1]
            if iCond_refnum != iCond_refnumOld:
                iCondPos = positiveTuple[iCond_refnum]
                iCondNeg = negativeTuple[iCond_refnum]

            stepStart = schedRef[i][2]
            stepEnd = schedRef[i][3]
            steps = schedRef[i][4]

            if len(schedRef[i]) == 6:
                mult8 = schedRef[i][5]
            else:
                mult8 = oldMult8

            if mult8 != oldMult8:
                outputW = (startW * mult8) // 8
                outputH = (startH * mult8) // 8
                if zoom_by_img_or_ltnt == "img":
                    inImg = imageScaleMethod(inImg, upscale_method, outputW, outputH, "disabled")[0]  #(self, image, upscale_method, width, height, crop)
                    ltnt = vaeEncodeMethod(vae, inImg)[0]   #(self, vae, pixels)
                else:
                    ltnt = latentUpscaleMethod(ltnt, upscale_method, outputW, outputH, "disabled")[0]    #(self, samples, upscale_method, width, height, crop)

            if i == 1:
                iDisableNoise = disable_noise_intermed
            if i == schedRefLen - 1:
                iForceFullDenoise = force_full_denoise_end

            ltnt = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, iCondPos, iCondNeg, ltnt, denoise=1, disable_noise=iDisableNoise, start_step=stepStart, last_step=stepEnd, force_full_denoise=iForceFullDenoise)[0]
            outImg = vae.decode(ltnt["samples"])

            if mult8 > 8:
                outImgScaled = imageScaleMethod(outImg, upscale_method, startW, startH, "disabled")[0]
            else:
                outImgScaled = outImg

            progImages = imageBatchMethod(progImages, outImgScaled)[0]   #(self, image1, image2)

            if i < schedRefLen - 1:
                oldMult8 = mult8
                iFullPipe_refnumOld = iFullPipe_refnum
                iCond_refnumOld = iCond_refnum
                inImg = outImg


        ltntPipeOut = ltnt, startW, startH, outputW, outputH
        return (ltntPipeOut, ltnt, outImg, progImages)


""" class mwFullPipeTuple2_KSA:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
    @classmethod
    def INPUT_TYPES(moh):
        return {"required":
                    {"fullPipeTuple": ("MWFULLPIPETUPLE",),
                    "positiveTuple": ("CONDTUPLE", ),
                    "negativeTuple": ("CONDTUPLE", ),
                    "cfg": ("FLOAT", {"default": 8.0, "min": -100.0, "max": 100.0, "step":0.5, "round": 0.01}),
                    #"stepEnd": ("INT", {"default": 48, "min": 0, "max": 10000}),
                    "steps": ("INT", {"default": 48, "min": 1, "max": 10000}),
                    "seed_deltaVsOrig": ("INT", {"default": 0, "min": 0, "max": 10000, "defaultBehavior": "input"}),
                    "add_noise": (["enable", "disable"], ),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "startS": ("MWSIZEPIPE",),
                    "list_FP_Cond_StepEnd": ("STRING", {"multiline": True}),
                    },
                "optional":
                {"inputImage": ("IMAGE",),
                 "mask": ("MASK",),
                }
                }

    RETURN_TYPES = ("MWLTNTPIPE", "LATENT", "IMAGE", "INT", "INT", "IMAGE")
    RETURN_NAMES = ("ltntPipe", "latent", "image", "stepEnd", "steps", "progImages")
    FUNCTION = "mwFPTTKSAS"
    CATEGORY = "mohwag/Sampling"

    def mwFPTTKSAS(self, fullPipeTuple, positiveTuple, negativeTuple, cfg, steps, seed_deltaVsOrig, add_noise, return_with_leftover_noise, startS, list_FP_Cond_StepEnd):

        schedList = list_FP_Cond_StepEnd.splitlines()
        for x in ["", " ", "  "]:
            while(x in schedList):
                schedList.remove(x)

        schedRef = [y.split(" ",-1) for y in schedList]
        schedRefLen = len(schedRef)

        for i in range(schedRefLen):
            for j in range(len(schedRef[i])):
                schedRef[i][j] = int(schedRef[i][j])        
        
        if schedRefLen > 1:
            int_return_with_leftover_noise = "enable"
        else:
            int_return_with_leftover_noise = return_with_leftover_noise

        arun = mwFullPipe_KSAStartMethod(fullPipeTuple[schedRef[0][0]], positiveTuple[schedRef[0][1]], negativeTuple[schedRef[0][1]], cfg, schedRef[0][2], steps, seed_deltaVsOrig, add_noise, int_return_with_leftover_noise, startS)
        
        ltntPipeOut, ltntOut, imgOut, stepEnd, steps = arun
        
        progImages = imgOut

        add_noise = "disable" 

        aoldMult8 = 8
        amult8 = 8

        if schedRefLen > 1:
            for i in range(schedRefLen):
                if i > 0:
                    if i == schedRefLen:
                        int_return_with_leftover_noise = return_with_leftover_noise
                    if len(schedRef[i]) == 4:
                        amult8 = schedRef[i][3]
                    else:
                        amult8 = aoldMult8

                    if amult8 > aoldMult8:
                        stepEnd = stepEnd - 1

                    arun = mwFullPipe_KSAMethod(fullPipeTuple[schedRef[i][0]], ltntPipeOut, positiveTuple[schedRef[i][1]], negativeTuple[schedRef[i][1]], cfg, stepEnd, schedRef[i][2], steps, seed_deltaVsOrig, add_noise, int_return_with_leftover_noise, amult8, "bicubic")
                    #(self, fullPipe, ltntPipe, positive, negative, cfg, stepStart, stepEnd, steps, seed_deltaVsOrig, add_noise, return_with_leftover_noise, multOver8, upscale_method, denoise=1.0):
                    ltntPipeOut, ltntOut, imgOut, stepEnd, steps = arun
                    progImages = imageBatchMethod(progImages, imgOut)[0]
                    aoldMult8 = amult8
                   
        return (ltntPipeOut, ltntOut, imgOut, stepEnd, steps, progImages) """


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
                    "stepStart": ("INT", {"default": 8, "min": 0, "max": 10000}),
                    "stepEnd": ("INT", {"default": 48, "min": 0, "max": 10000}),
                    "steps": ("INT", {"default": 48, "min": 1, "max": 10000}),
                    "seed_deltaVsOrig": ("INT", {"default": 0, "min": 0, "max": 10000, "defaultBehavior": "input"}),
                    "add_noise": (["enable", "disable"], ),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "multOver8": ("INT", {"default": 8, "min": 0, "max": 64, "step":1}),
                    "upscale_method": (moh.upscale_methods,),
                     },}
    RETURN_TYPES = ("MWLTNTPIPE", "LATENT", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("ltntPipe", "latent", "image", "stepEnd", "steps")
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
        #outputS = outputW, outputH
        return (ltntPipe, runLtntOut[0], imgOut[0], stepEnd, steps)

mwFullPipe_KSAMethod = mwFullPipe_KSA().mwFPKSA   #(self, fullPipe, ltntPipe, positive, negative, cfg, stepStart, stepEnd, steps, seed_deltaVsOrig, add_noise, return_with_leftover_noise, multOver8, upscale_method, denoise=1.0):

class mwFPBranch_all:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }
    RETURN_TYPES = ("MWFULLPIPE", "MWCKPT", "MODEL", "CLIP", "VAE", "INT", comfy.samplers.SAMPLER_NAMES, comfy.samplers.SCHEDULER_NAMES)
    RETURN_NAMES = ("mwFullPipe", "mwCkpt", "model", "clip", "vae", "seed", "sampler_name", "scheduler")
    FUNCTION = "mwFPBA"
    CATEGORY = "mohwag/PipeBranch"
    def mwFPBA(self, mwFullPipe):
        mwCkpt, seed, sampler_name, scheduler = mwFullPipe  #mwCkpt, seed, sampler_name, scheduler
        model, clip, vaeAlways = mwCkpt
        return (mwFullPipe, mwCkpt, model, clip, vaeAlways, seed, sampler_name, scheduler)

class mwFPBranch_ckpt:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }
    RETURN_TYPES = ("MWFULLPIPE", "MWCKPT",)
    RETURN_NAMES = ("mwFullPipe", "mwCkpt",)
    FUNCTION = "mwFPBCP"
    CATEGORY = "mohwag/PipeBranch"
    def mwFPBCP(self, mwFullPipe):
        mwCkpt, _, _, _ = mwFullPipe  #mwCkpt, seed, sampler_name, scheduler
        #model, clip, vaeAlways = mwCkpt
        return (mwFullPipe, mwCkpt,)

class mwFPBranch_model:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }
    RETURN_TYPES = ("MWFULLPIPE", "MODEL",)
    RETURN_NAMES = ("mwFullPipe", "model",)
    FUNCTION = "mwFPBM"
    CATEGORY = "mohwag/PipeBranch"
    def mwFPBM(self, mwFullPipe):
        mwCkpt, _, _, _ = mwFullPipe  #mwCkpt, seed, sampler_name, scheduler
        model, _, _ = mwCkpt
        return (mwFullPipe, model,)

class mwFPBranch_clip:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }
    RETURN_TYPES = ("MWFULLPIPE", "CLIP",)
    RETURN_NAMES = ("mwFullPipe", "clip",)
    FUNCTION = "mwFPBC"
    CATEGORY = "mohwag/PipeBranch"
    def mwFPBC(self, mwFullPipe):
        mwCkpt, _, _, _ = mwFullPipe  #mwCkpt, seed, sampler_name, scheduler
        _, clip, _ = mwCkpt
        return (mwFullPipe, clip,)

class mwFPBranch_vae:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwFullPipe": ("MWFULLPIPE",), }, }
    RETURN_TYPES = ("MWFULLPIPE", "VAE",)
    RETURN_NAMES = ("mwFullPipe", "vae",)
    FUNCTION = "mwFPBV"
    CATEGORY = "mohwag/PipeBranch"
    def mwFPBV(self, mwFullPipe):
        mwCkpt, _, _, _ = mwFullPipe  #mwCkpt, seed, sampler_name, scheduler
        _, _, vaeAlways = mwCkpt
        return (mwFullPipe, vaeAlways,)


class mwCkptBranch_all:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwCkpt": ("MWCKPT",), }, }
    RETURN_TYPES = ("MWCKPT", "MODEL", "CLIP", "VAE",)
    RETURN_NAMES = ("mwCkpt", "model", "clip", "vae",)
    FUNCTION = "mwCPPBA"
    CATEGORY = "mohwag/PipeBranch"
    def mwCPPBA(self, mwCkpt):
        model, clip, vaeAlways = mwCkpt
        return (mwCkpt, model, clip, vaeAlways)

class mwCkptBranch_model:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwCkpt": ("MWCKPT",), }, }
    RETURN_TYPES = ("MWCKPT", "MODEL",)
    RETURN_NAMES = ("mwCkpt", "model",)
    FUNCTION = "mwCPPBM"
    CATEGORY = "mohwag/PipeBranch"
    def mwCPPBM(self, mwCkpt):
        model, _, _ = mwCkpt
        return (mwCkpt, model,)

class mwCkptBranch_clip:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwCkpt": ("MWCKPT",), }, }
    RETURN_TYPES = ("MWCKPT", "CLIP",)
    RETURN_NAMES = ("mwCkpt", "clip",)
    FUNCTION = "mwCPPBC"
    CATEGORY = "mohwag/PipeBranch"
    def mwCPPBC(self, mwCkpt):
        _, clip, _ = mwCkpt
        return (mwCkpt, clip,)

class mwCkptBranch_vae:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwCkpt": ("MWCKPT",), }, }
    RETURN_TYPES = ("MWCKPT", "VAE",)
    RETURN_NAMES = ("mwCkpt", "vae",)
    FUNCTION = "mwCPPBV"
    CATEGORY = "mohwag/PipeBranch"
    def mwCPPBV(self, mwCkpt):
        _, _, vaeAlways = mwCkpt
        return (mwCkpt, vaeAlways,)

class mwCkptBranch2_all:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwCkpt": ("MWCKPT",), }, }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE",)
    RETURN_NAMES = ("model", "clip", "vae",)
    FUNCTION = "mwCPPBTA"
    CATEGORY = "mohwag/PipeBranch"
    def mwCPPBTA(self, mwCkpt):
        model, clip, vaeAlways = mwCkpt
        return (model, clip, vaeAlways)

class mwCkptBranch2_model:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwCkpt": ("MWCKPT",), }, }
    RETURN_TYPES = ( "MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "mwCPPBTM"
    CATEGORY = "mohwag/PipeBranch"
    def mwCPPBTM(self, mwCkpt):
        model, _, _ = mwCkpt
        return (model,)

class mwCkptBranch2_clip:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwCkpt": ("MWCKPT",), }, }
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "mwCPPBTC"
    CATEGORY = "mohwag/PipeBranch"
    def mwCPPBTC(self, mwCkpt):
        _, clip, _ = mwCkpt
        return (clip,)

class mwCkptBranch2_vae:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwCkpt": ("MWCKPT",), }, }
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "mwCPPBTV"
    CATEGORY = "mohwag/PipeBranch"
    def mwCPPBTV(self, mwCkpt):
        _, _, vaeAlways = mwCkpt
        return (vaeAlways,)


class mwCkpt_vaeNone:
    @classmethod
    def INPUT_TYPES(moh):
        return {"required": {"mwCkpt": ("MWCKPT",), }, }
    RETURN_TYPES = ("MWCKPT",)
    RETURN_NAMES = ("mwCkpt",)
    FUNCTION = "mwCVN"
    CATEGORY = "mohwag/PipeBranch"
    def mwCVN(self, mwCkpt):
        model, clip, _ = mwCkpt
        out_mwCkpt = model, clip, None
        return (out_mwCkpt,)

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



class mwText:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "mwTxt"
    CATEGORY = "mohwag/Text"
    def mwTxt(self, text):
        return (text,)


class mwTextConcat:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "text1": ("STRING", {"multiline": True}),
            }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "mwTC"
    CATEGORY = "mohwag/Text"
    def mwTC(self, text, text1):
        out_text = text + text1
        return (out_text,)

class mwTextReplace:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "text1": ("STRING", {"multiline": True}),
            }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "mwTR"
    CATEGORY = "mohwag/Utils"
    def mwTR(self, text, text1):
        findList, ReplaceList = condList_createLR(text1, "^^")
        outText = text
        for x in range(len(findList)):
            outText = outText.replace(findList[x], ReplaceList[x])
        return (outText,)


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
    numRng = int(numMult * (10000 // numMult))
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "numInt": ("INT", {"default": 0, "min": -1 * moh.numRng, "max": moh.numRng, "step": moh.numMult}),
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
    numRng = int(numMult * (10000 // numMult))
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "numInt": ("INT", {"default": 0, "min": -1 * moh.numRng, "max": moh.numRng, "step": moh.numMult}),
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
    numRng = int(numMult * (10000 // numMult))
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "numInt": ("INT", {"default": 0, "min": -1 * moh.numRng, "max": moh.numRng, "step": moh.numMult}),
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
    numRng = int(numMult * (10000 // numMult))
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "numInt": ("INT", {"default": 0, "min": -1 * moh.numRng, "max": moh.numRng, "step": moh.numMult}),
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
    numRng = int(numMult * (10000 // numMult))
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": -1 * moh.numRng, "max": moh.numRng, "step": moh.numMult}),
                "height": ("INT", {"default": 1024, "min": -1 * moh.numRng, "max": moh.numRng, "step": moh.numMult}),
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
                "x256": ("INT", {"default": 0, "min": -16, "max": 16, "step": 1}),
                "x64": ("INT", {"default": 0, "min": -16, "max": 16, "step": 1}),
                "x16": ("INT", {"default": 0, "min": -16, "max": 16, "step": 1}),
                "x8": ("INT", {"default": 0, "min": -8, "max": 8, "step": 1}),
                "x1": ("INT", {"default": 0, "min": -16, "max": 16, "step": 1}),
            }
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "mwNIBI"
    CATEGORY = "mohwag/Int"
    def mwNIBI(self, x256, x64, x16, x8, x1):
        rslt = int((256 * x256) + (64 * x64) + (16 * x16) + (8 * x8) + x1)
        return (rslt,)

class mwNumIntBinaryIsh2:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                #"x1024": ("INT", {"default": 0, "min": -16, "max": 16, "step": 1}),
                "x256": ("INT", {"default": 0, "min": -16, "max": 16, "step": 1}),
                "x64": ("INT", {"default": 0, "min": -16, "max": 16, "step": 1}),
                "x16": ("INT", {"default": 0, "min": -16, "max": 16, "step": 1}),
                "x8": ("INT", {"default": 0, "min": -8, "max": 8, "step": 1}),
                #"x1": ("INT", {"default": 0, "min": -16, "max": 16, "step": 1}),
            }
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "mwNIBIT"
    CATEGORY = "mohwag/Int"
    def mwNIBIT(self, x256, x64, x16, x8):
        rslt = int((256 * x256) + (64 * x64) + (16 * x16) + (8 * x8))
        return (rslt,)
    

class mwsPipeCreate:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": -10000, "max": 10000, "step": 1}),
                "height": ("INT", {"default": 1024, "min": -10000, "max": 10000, "step": 1}),
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



class mwsPipe_MultOver8:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "sPipeS": ("MWSIZEPIPE",),
                "multOver8": ("INT", {"default": 8, "min": 0, "max": 64, "step":1}),
            }
        }
    RETURN_TYPES = ("MWSIZEPIPE", "MWSIZEPIPE", "INT",)
    RETURN_NAMES = ("sPipeS_pt", "sPipeC", "multOver8_pt",)
    FUNCTION = "mwNSP"
    CATEGORY = "mohwag/scaling"

    def mwNSP(self, sPipeS, multOver8):
        initw, inith = sPipeS
        outw = int(multOver8 * initw // 8)
        outh = int(multOver8 * inith // 8)
        sPipeC = outw, outh
        return (sPipeS, sPipeC, multOver8, )


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
'''
class mwMaskBoundingBoxRaw:  ########################################################
    def __init__(self, device="cpu"):
        self.device = device
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "images_mapped": ("IMAGE",),
                "mask_bounding_box": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "deltaLft_units": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "deltaRgt_units": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "deltaTop_units": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "deltaBtm_units": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
            }}
    #RETURN_TYPES = ("MWSIZEPIPE", "MWPOSSIZEPIPE", "MASK", "MASK", "IMAGE")
    #RETURN_NAMES = ("sPipe", "psPipe", "bounded mask", "stndMask", "bounded image")
    RETURN_TYPES = ("IMAGE", "MASK", "MWPOSSIZEPIPE",)
    RETURN_NAMES = ("images_pt", "mask_pt","psPipe",)
    FUNCTION = "mwMBBR"
    CATEGORY = "mohwag/mask"
    def mwMBBR(self, images_mapped, mask_bounding_box, threshold, deltaLft_units, deltaRgt_units, deltaTop_units, deltaBtm_units):
        unitS = 8
        #multReq = 64
        # Get the mask where pixel values are above the threshold
        mask_above_threshold = mask_bounding_box > threshold
        # Compute the bounding box
        non_zero_positions = torch.nonzero(mask_above_threshold)
        if len(non_zero_positions) == 0:
            return (0, 0, 0, 0, 0, 0, torch.zeros_like(mask_bounding_box), torch.zeros_like(images_mapped[0]))

        #imgPrep = Image.fromarray(np.clip(255. * images_mapped[0].cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        #imgW = imgPrep.size[0]
        #imgH = imgPrep.size[1]

        x0i = int(torch.min(non_zero_positions[:, 1]))
        x1i = int(torch.max(non_zero_positions[:, 1]))
        y0i = int(torch.min(non_zero_positions[:, 0]))
        y1i = int(torch.max(non_zero_positions[:, 0]))

        x0_1 = x0i + (deltaLft_units * unitS)
        x1_1 = x1i + (deltaRgt_units * unitS)
        y0_1 = y0i + (deltaTop_units * unitS) 
        y1_1 = y1i + (deltaBtm_units * unitS)

        min_xf = int(x0_1)
        max_xf = int(x1_1)
        min_yf = int(y0_1)
        max_yf = int(y1_1)      

        fin_w = int(max_xf - min_xf)
        fin_h = int(max_yf - min_yf)        

        outputPS = min_xf, min_yf, fin_w, fin_h
        maskOut = imageToMaskMethod(maskToImageMethod(mask_bounding_box)[0], "green")[0]
        return (images_mapped, maskOut, outputPS,)
'''
class mwMaskBoundingBoxRaw:  ########################################################
    def __init__(self, device="cpu"):
        self.device = device
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "images_mapped": ("IMAGE",),
                "mask_bounding_box": ("MASK",),
                "sPipeS": ("MWSIZEPIPE",),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "deltaLft_units": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "deltaRgt_units": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "deltaTop_units": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "deltaBtm_units": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "nextMO8_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "nextMO8_add": ("INT", {"default": 0, "min": -32, "max": 32, "step": 1}),
            }}
    #RETURN_TYPES = ("MWSIZEPIPE", "MWPOSSIZEPIPE", "MASK", "MASK", "IMAGE")
    #RETURN_NAMES = ("sPipe", "psPipe", "bounded mask", "stndMask", "bounded image")
    RETURN_TYPES = ("MWLTNTCNFRMPIPE", "MASK", "INT")
    RETURN_NAMES = ("ltntcnfrmPipe", "mask_pt", "nextMO8")
    FUNCTION = "mwMBBR"
    CATEGORY = "mohwag/mask"
    def mwMBBR(self, images_mapped, mask_bounding_box, sPipeS, threshold, deltaLft_units, deltaRgt_units, deltaTop_units, deltaBtm_units, nextMO8_mult, nextMO8_add):
        wS, _ = sPipeS
        sPipeC = mwImageTo_sPipeMethod(images_mapped[0])[0]
        wC = sPipeC[0]
        unitS = int(8 * wC / wS)
        #multReq = 64
        # Get the mask where pixel values are above the threshold
        mask_above_threshold = mask_bounding_box > threshold
        # Compute the bounding box
        non_zero_positions = torch.nonzero(mask_above_threshold)
        if len(non_zero_positions) == 0:
            return (0, 0, 0, 0, 0, 0, torch.zeros_like(mask_bounding_box), torch.zeros_like(images_mapped[0]))

        #imgPrep = Image.fromarray(np.clip(255. * images_mapped[0].cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        #imgW = imgPrep.size[0]
        #imgH = imgPrep.size[1]

        x0i = int(torch.min(non_zero_positions[:, 1]))
        x1i = int(torch.max(non_zero_positions[:, 1]))
        y0i = int(torch.min(non_zero_positions[:, 0]))
        y1i = int(torch.max(non_zero_positions[:, 0]))

        x0_1 = x0i + (deltaLft_units * unitS)
        x1_1 = x1i + (deltaRgt_units * unitS)
        y0_1 = y0i + (deltaTop_units * unitS) 
        y1_1 = y1i + (deltaBtm_units * unitS)

        min_xf = int(x0_1)
        max_xf = int(x1_1)
        min_yf = int(y0_1)
        max_yf = int(y1_1)      

        fin_w = int(max_xf - min_xf)
        fin_h = int(max_yf - min_yf)        

        nextMO8 = int(((nextMO8_mult * unitS * (1024 / ((fin_w * fin_h)**(1/2)))) + nextMO8_add) // 1)

        psPipeC_out = min_xf, min_yf, fin_w, fin_h
        maskOut = imageToMaskMethod(maskToImageMethod(mask_bounding_box)[0], "green")[0]
        ltntcnfrmpipe_out = (images_mapped, maskOut, psPipeC_out, sPipeS, sPipeC, unitS, nextMO8)
        return (ltntcnfrmpipe_out, maskOut, nextMO8,)


class mwpsScale_LtntConform:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                #"images": ("IMAGE",),
                #"mask": ("MASK",),
                #"psPipe": ("MWPOSSIZEPIPE",),
                #"multOver8": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                #"nextMO8": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "ltntcnfrmPipe": ("MWLTNTCNFRMPIPE",),
                "upscale_method": (moh.upscale_methods,),
            },
            "optional": {
                "mask_opt": ("MASK",),
                "nextMO8_opt": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
            }}
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "MASK", "MASK", "MASK", "MWSIZEPIPE", "MWPOSSIZEPIPE", "MWSIZEPIPE", "MWSIZEPIPE", "MWPOSSIZEPIPE", "MWSIZEPIPE")
    RETURN_NAMES = ("images_pt", "boundedImgs", "images_upsc", "boundedImgs_upsc", "mask_pt", "mask_upsc", "boundedMask_upsc", "sPipeS", "bounded_psPipeS", "bounded_sPipeS", "sPipe_upsc", "bounded_psPipe_upsc", "bounded_sPipe_upsc")
    FUNCTION = "mwPSSLC"
    CATEGORY = "mohwag/scaling"
    def mwPSSLC(self, ltntcnfrmPipe, upscale_method, mask_opt=None, nextMO8_opt=None):

        #(images_mapped, maskOut, psPipeC_out, sPipeS, sPipeC, unitS, nextMO8)
        images, maskPiped, psPipeC, sPipeS, sPipeC, multOver8, nextMO8_piped = ltntcnfrmPipe

        if mask_opt == None:
            mask = maskPiped
        else:
            mask = mask_opt

        if nextMO8_opt == None:
            nextMO8 = nextMO8_piped
        else:
            nextMO8 = nextMO8_opt

        currMult = nextMO8 / multOver8
        baseMult = nextMO8 / 8
        
        #sPipeC = mwImageTo_sPipeMethod(images[0])[0]
        imgWc, imgHc = sPipeC
        x0, y0, w, h = psPipeC
        x1 = x0 + w
        y1 = y0 + h

        imgWu = imgWc * currMult
        imgHu = imgHc * currMult
        
        x0u = x0 * currMult
        x1u = x1 * currMult
        y0u = y0 * currMult
        y1u = y1 * currMult

        imgWuf = int(imgWu)
        imgHuf = int(imgHu)
        sPipeU = imgWuf, imgHuf
        x0uc, x1uc = psPipeModForLtnt(imgWu, x0u, x1u, nextMO8)
        y0uc, y1uc = psPipeModForLtnt(imgHu, y0u, y1u, nextMO8)
 
        x0ucf = int(x0uc)
        x1ucf = int(x1uc)
        y0ucf = int(y0uc)
        y1ucf = int(y1uc)
        wucf = int(x1ucf - x0ucf)
        hucf = int(y1ucf - y0ucf)
        s_ucf = wucf, hucf
        ps_ucf = x0ucf, y0ucf, wucf, hucf

        x0ccf = int(x0uc / currMult)
        #x1cf = int(x1uc / theMult)
        y0ccf = int(y0uc / currMult)
       # y1cf = int(y1uc / theMult)
        wccf = int(wucf / currMult)
        hccf = int(hucf / currMult)
        #s_ccf = wccf, hccf
        ps_ccf = x0ccf, y0ccf, wccf, hccf

        x0bcf = int(x0uc / baseMult)
        #x1cf = int(x1uc / baseMult)
        y0bcf = int(y0uc / baseMult)
       # y1cf = int(y1uc / baseMult)
        wbcf = int(wucf / baseMult)
        hbcf = int(hucf / baseMult)
        s_bcf = wbcf, hbcf
        ps_bcf = x0bcf, y0bcf, wbcf, hbcf

        maskImg = maskToImageMethod(mask)[0]       

        if nextMO8 == multOver8:
            img_u = images
            maskImg_u = maskImg
        else:
            img_u = imageScaleMethod(images, upscale_method, imgWuf, imgHuf, "disabled")[0]
            maskImg_u = imageScaleMethod(maskImg, upscale_method, imgWuf, imgHuf, "disabled")[0]

        boundedImage_c = mwImageCropMethod(images, ps_ccf)[0]
        boundedImage_u = mwImageCropMethod(img_u, ps_ucf)[0]
        boundedMaskImg_u = mwImageCropMethod(maskImg_u, ps_ucf)[0]
        mask_u = imageToMaskMethod(maskImg_u, "green")[0]
        boundedMask_u = imageToMaskMethod(boundedMaskImg_u, "green")[0]
        #("images_pt", "boundedImgs", "images_upsc", "boundedImgs_upsc", "mask_upsc", "boundedMask_upsc", "sPipeS", "bounded_psPipeS", "bounded_sPipeS", "sPipe_upsc", "bounded_psPipe_upsc", "bounded_sPipe_upsc")
        return (images, boundedImage_c, img_u, boundedImage_u, mask, mask_u, boundedMask_u               , sPipeS  , ps_bcf           , s_bcf           , sPipeU      , ps_ucf               , s_ucf)


def conformSizeParams(aimgD, amultReq, ad0, ad1):
    mp0 = (ad0 + ad1)//2
    wid0 = ad1 - ad0
    widRU0 = ((wid0 //amultReq) +(wid0 % amultReq > 0)) * amultReq
    widRU0_half = widRU0 //2
    ad0 = int(mp0 - widRU0_half)
    ad1 = int(mp0 + widRU0_half)
    
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

class mwMaskBoundingBoxRF64:  ########################################################
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
    RETURN_TYPES = ("MWSIZEPIPE", "MWPOSSIZEPIPE", "MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("sPipe", "psPipe", "bounded mask", "stndMask", "bounded image")
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
        stndMask =  imageToMaskMethod(maskToImageMethod(raw_bb)[0], "green") [0] #(self, image, channel = "green")
        raw_img = image_mapped[:,min_yf:max_yf,min_xf:max_xf,:]

        outputS = fin_w, fin_h
        outputPS = min_xf, min_yf, fin_w, fin_h
        return (outputS, outputPS, raw_bb, stndMask, raw_img)


class mwCLIPSeg:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "image": ("IMAGE",),
                        "text": ("STRING", {"multiline": False}),
                        
                     },
                "optional":
                    {
                        "blur": ("FLOAT", {"min": 0, "max": 15, "step": 0.1, "default": 7}),
                        "threshold": ("FLOAT", {"min": 0, "max": 1, "step": 0.01, "default": 0.4}),
                        "dilation_factor": ("INT", {"min": 0, "max": 10, "step": 1, "default": 4}),
                    }
                }
    CATEGORY = "mohwag/mask"
    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("Mask", "stndMask", "Heatmap Mask", "BW Mask")
    FUNCTION = "mwCS"
    def mwCS(self, image: torch.Tensor, text: str, blur: float, threshold: float, dilation_factor: int): # -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recMask, recHeatMapImg, recBW_Img = clipSegMethod(image, text, blur, threshold, dilation_factor)
        
        stndMask = imageToMaskMethod(recBW_Img, "green") [0] #(self, image, channel = "green")
        return recMask, stndMask, recHeatMapImg, recBW_Img


class mwMaskConvert:
    @classmethod
    def INPUT_TYPES(moh):
        return {
            "required": {
                "stndMask": ("MASK",),
            }}
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("specMask",)
    FUNCTION = "mwTMCO"
    CATEGORY = "mohwag/tester"
    def mwTMCO(self, stndMask):
        outMask = stndMask[0]
        return (outMask,)


class mwTester_condStrength:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"cond": ("CONDITIONING", ),}}
    RETURN_TYPES = ("LIST",)
    FUNCTION = "mwTCS"

    CATEGORY = "mohwag/tester"

    def mwTCS(self, cond):
        condCopy = cond.copy()

        strenList = []
        for i in range(len(condCopy)):
            strenList.append(condCopy[i][1].get("strength"))

        return (strenList,)


# taken from https://github.com/pythongosssss/ComfyUI-Custom-Scripts
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

class mwTupleIt:
    @classmethod
    def INPUT_TYPES(s):
        return {"optional": {
                    "any0": (any, ),
                    "any1": (any, ),
                    "any2": (any, ),
                    "any3": (any, ),
                    },
                }
    RETURN_TYPES = ("MWTUPLE",)
    RETURN_NAMES = ("tupled",)
    FUNCTION = "mwTI"
    CATEGORY = "mohwag/Util"

    @staticmethod
    def mwTI(any0=None, any1=None, any2=None, any3=None):
        theList = [any0, any1, any2, any3]
        out = tuple(())
        for i in theList:
            out += (i,)
        return (out,)

class mwTupleItStack:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {
                "tupled": (any, ),
                },
            
                "optional": {
                "any1": (any, ),
                "any2": (any, ),
                "any3": (any, ),
                },}
    RETURN_TYPES = ("MWTUPLE",)
    RETURN_NAMES = ("Tupled",)
    FUNCTION = "mwTIS"
    CATEGORY = "mohwag/Util"

    @staticmethod
    def mwTIS(tupled, any1=None, any2=None, any3=None):
        theList = [any1, any2, any3]
        out = tupled
        for i in theList:
            out += (i,)
        return (out,)


class mwTupleIt_CondTest:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "conditioning1": ("CONDITIONING", { "tooltip": "input conditionings. (Connecting to the input slot increases the number of additional slots.)" }),
                     },
                }

    OUTPUT_TOOLTIPS = ("Combined conditioning", )

    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "mwTICT"

    CATEGORY = "mohwag/Util"

    @staticmethod
    def mwTICT(**kwargs):
        res = []
        for k, v in kwargs.items():
            res += v

        return (res, )
    
class mwTester_CLIPTextEncodeSDXL2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            #"width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            #"height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            #"crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            #"crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            #"target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            #"target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "text_g": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ),
            "text_l": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ),
            }}
    RETURN_TYPES = ("condPart0", 'lencondPart0', "condPart00", "lencondPart00")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    #def encode(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l):
    def encode(self, clip, text_g, text_l):
        tokens = clip.tokenize(text_g)
        tokens["l"] = clip.tokenize(text_l)["l"]

        #tokl = tokens["l"].copy()
        #tokg = tokens["g"].copy()
        #lentokl = len(tokl)
        #lentokg = len(tokg)
        
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        
        condPart0 = cond[0]
        condPart00 = condPart0[0]

        return (condPart0,len(condPart0), condPart00, len(condPart00))


NODE_CLASS_MAPPINGS = {
"mwImageTo_sPipe": mwImageTo_sPipe,
"mwMaskTo_sPipe": mwMaskTo_sPipe,
"mwGridOverlay": mwGridOverlay,
"mwMaskSegment": mwMaskSegment,
"mwMaskSegmentByPS": mwMaskSegmentByPS,
"mwMaskCompSameSize": mwMaskCompSameSize,
"mwSolidMask": mwSolidMask,
"mwMaskTweak": mwMaskTweak,
"mwMaskStack": mwMaskStack,
"mwImageCompositeMasked": mwImageCompositeMasked,
"mwImageScale": mwImageScale,
"mwImageBranch": mwImageBranch,
"mwImageCrop": mwImageCrop,
"mwImageCropwParams": mwImageCropwParams,
"mwImageConform_StartSizeXL": mwImageConform_StartSizeXL,
"mwBatch": mwBatch,
"mwBatch_squareHelp": mwBatch_squareHelp,
"mwBatch_AllSections": mwBatch_AllSections,
"mwpsConditioningSetArea": mwpsConditioningSetArea,
"mwConditioningTuple": mwConditioningTuple,
"mwFullPipeTuple": mwFullPipeTuple,
"mwConditioningCnetTileSpecTwo": mwConditioningCnetTileSpecTwo,
"mwConditioningCnetTileSpec": mwConditioningCnetTileSpec,
"mwConditioningAddAesthScore": mwConditioningAddAesthScore,
"mwFullPipe_Load": mwFullPipe_Load,
"mwFullPipe_modelEdit": mwFullPipe_modelEdit,
"mwCkpt_Load": mwCkpt_Load,
"mwCkpt_modelEdit": mwCkpt_modelEdit,
"mwCkpt_ckptEdit": mwCkpt_ckptEdit,
"mwFullPipe_ckptMerge": mwFullPipe_ckptMerge,
"mwCkpt_ckptMerge": mwCkpt_ckptMerge,
"mwLora_Load": mwLora_Load,
"mwFullPipe_loraMerge": mwFullPipe_loraMerge,
"mwCkpt_loraMerge": mwCkpt_loraMerge,
"mwCkpt_loraStackMerge": mwCkpt_loraStackMerge,
"mwLtntPipe_Create": mwLtntPipe_Create,
"mwLtntPipe_Create2": mwLtntPipe_Create2,
"mwImage_VertSqzExpnd": mwImage_VertSqzExpnd,
"mwImage_VertSqzExpndStack": mwImage_VertSqzExpndStack,
"mwImageTaper_1vert": mwImageTaper_1vert,
"mwImageTaperStack_1vert": mwImageTaperStack_1vert,
"mwImageTaper_2vert": mwImageTaper_2vert,
"mwImageTaperStack_2vert": mwImageTaperStack_2vert,
"mwLtntPipeBranch1": mwLtntPipeBranch1,
"mwLtntPipe_View": mwLtntPipe_View,
"mwCondXLa": mwCondXLa,
"mwCondXLadv": mwCondXLadv,
"mwCondXLadvStack": mwCondXLadvStack,
"mwCondListDictKeys": mwCondListDictKeys,
"mwFullPipe_KSAStart": mwFullPipe_KSAStart,
"mwFullPipeTuple_KSA": mwFullPipeTuple_KSA,
#"mwFullPipeTuple_KSAStart2": mwFullPipeTuple_KSAStart2,
#"mwFullPipeTuple_KSAStart3": mwFullPipeTuple_KSAStart3,
"mwFullPipe_KSA": mwFullPipe_KSA,
"mwFPBranch_all": mwFPBranch_all,
"mwFPBranch_ckpt": mwFPBranch_ckpt,
"mwFPBranch_model": mwFPBranch_model,
"mwFPBranch_clip": mwFPBranch_clip,
"mwFPBranch_vae": mwFPBranch_vae,
"mwCkptBranch_all": mwCkptBranch_all,
"mwCkptBranch_model": mwCkptBranch_model,
"mwCkptBranch_clip": mwCkptBranch_clip,
"mwCkptBranch_vae": mwCkptBranch_vae,
"mwCkptBranch2_all": mwCkptBranch2_all,
"mwCkptBranch2_model": mwCkptBranch2_model,
"mwCkptBranch2_clip": mwCkptBranch2_clip,
"mwCkptBranch2_vae": mwCkptBranch2_vae,
"mwCkpt_vaeNone": mwCkpt_vaeNone,
"mwSchedEdit": mwSchedEdit,
"mwText": mwText,
"mwTextConcat": mwTextConcat,
"mwTextReplace": mwTextReplace,
"mwNumInt": mwNumInt,
"mwNumIntx8": mwNumIntx8,
"mwNumIntx16": mwNumIntx16,
"mwNumIntx32": mwNumIntx32,
"mwNumIntx64": mwNumIntx64,
"mwNumIntx64s": mwNumIntx64s,
"mwStartSizeXL": mwStartSizeXL,
"mwNumIntBinaryIsh": mwNumIntBinaryIsh,
"mwNumIntBinaryIsh2": mwNumIntBinaryIsh2,
"mwsPipeCreate": mwsPipeCreate,
"mwpsPipeCreate": mwpsPipeCreate,
"mwpsPipeCreate2": mwpsPipeCreate2,
"mwsPipeBranch": mwsPipeBranch,
"mwpsPipeBranch": mwpsPipeBranch,
"mwpsPipeBranch2": mwpsPipeBranch2,
"mwsPipe_MultOver8": mwsPipe_MultOver8,
"mwCompScale": mwCompScale,
"mwMaskBoundingBoxRaw": mwMaskBoundingBoxRaw,
"mwpsScale_LtntConform": mwpsScale_LtntConform,
"mwMaskBoundingBoxRF64": mwMaskBoundingBoxRF64,
"mwCLIPSeg": mwCLIPSeg,
"mwMaskConvert": mwMaskConvert,
#"mwTester_ConditioningAverage":mwTester_ConditioningAverage,
#"mwTester_CLIPTextEncodeSDXL": mwTester_CLIPTextEncodeSDXL,
"mwTester_CLIPTextEncodeSDXL2": mwTester_CLIPTextEncodeSDXL2,
#"mwTester_CLIPTextEncode": mwTester_CLIPTextEncode,
"mwTester_condStrength": mwTester_condStrength,
"mwTupleIt": mwTupleIt,
"mwTupleItStack": mwTupleItStack,
"mwTupleIt_CondTest": mwTupleIt_CondTest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
"mwImageTo_sPipe": "sPipeFromImage",  ##################
"mwMaskTo_sPipe": "sPipeFromMask",  ##################
"mwGridOverlay": "GridOverlay",
"mwMaskSegment": "MaskSegment",
"mwMaskSegmentByPS": "MaskSegmentByPS",
"mwMaskCompSameSize": "MaskCompSameSize",
"mwSolidMask": "SolidMask",
"mwMaskTweak": "MaskTweak",
"mwMaskStack": "MaskStack",
"mwImageCompositeMasked": "mwImageCompositeMasked",
"mwImageScale": "ImageScale",
"mwImageBranch": "ImageBranch",
"mwImageCrop": "ImageCrop",
"mwImageCropwParams": "ImageCropwParams",
"mwImageConform_StartSizeXL": "ImageConform_StartSizeXL",
"mwBatch": "Batch",
"mwBatch_squareHelp": "Batch_squareHelp",
"mwBatch_AllSections": "Batch_AllSections",
"mwpsConditioningSetArea": "psConditioningSetArea",
"mwConditioningTuple": "ConditioningTuple",
"mwFullPipeTuple": "FullPipeTuple",
"mwConditioningCnetTileSpecTwo": "ConditioningCnetTileSpecTwo",
"mwConditioningCnetTileSpec": "ConditioningCnetTileSpec",
"mwConditioningAddAesthScore": "ConditioningAddAesthScore",
"mwFullPipe_Load": "FullPipe_Load",
"mwFullPipe_modelEdit": "FullPipe_modelEdit",
"mwCkpt_Load": "Ckpt_Load",
"mwCkpt_modelEdit": "Ckpt_modelEdit",
"mwCkpt_ckptEdit": "Ckpt_ckptEdit",
"mwFullPipe_ckptMerge": "FullPipe_ckptMerge",
"mwCkpt_ckptMerge": "Ckpt_ckptMerge",
"mwLora_Load": "Lora_Load",
"mwFullPipe_loraMerge": "FullPipe_loraMerge",
"mwCkpt_loraMerge": "Ckpt_loraMerge",
"mwCkpt_loraStackMerge": "Ckpt_loraStackMerge",
"mwLtntPipe_Create": "LtntPipe_Create",
"mwLtntPipe_Create2": "LtntPipe_Create2",
"mwImage_VertSqzExpnd": "Image_VertSqzExpnd",
"mwImage_VertSqzExpndStack": "Image_VertSqzExpndStack",
"mwImageTaper_1vert": "ImageTaper_1vert",
"mwImageTaperStack_1vert": "ImageTaperStack_1vert",
"mwImageTaper_2vert": "ImageTaper_2vert",
"mwImageTaperStack_2vert": "ImageTaperStack_2vert",
"mwLtntPipeBranch1": "LtntPipeBranch1",
"mwLtntPipe_View": "LtntPipe_View",
"mwCondXLa": "CondXLa",
"mwCondXLadv": "CondXLadv",
"mwCondXLadvStack": "CondXLadvStack",
"mwCondListDictKeys": "CondListDictKeys",
"mwFullPipe_KSAStart": "FullPipe_KSAStart",
"mwFullPipeTuple_KSA": "FullPipeTuple_KSA",
#"mwFullPipeTuple_KSAStart2": "FullPipeTuple_KSAStart2",
#"mwFullPipeTuple_KSAStart3": "FullPipeTuple_KSAStart3",
"mwFullPipe_KSA": "FullPipe_KSA",
"mwFPBranch_all": "FPBranch_all",
"mwFPBranch_ckpt": "FPBranch_ckpt",
"mwFPBranch_model": "FPBranch_model",
"mwFPBranch_clip": "FPBranch_clip",
"mwFPBranch_vae": "FPBranch_vae",
"mwCkptBranch_all": "CkptBranch_all",
"mwCkptBranch_model": "CkptBranch_model",
"mwCkptBranch_clip": "CkptBranch_clip",
"mwCkptBranch_vae": "CkptBranch_vae",
"mwCkptBranch2_all": "CkptBranch2_all",
"mwCkptBranch2_model": "CkptBranch2_model",
"mwCkptBranch2_clip": "CkptBranch2_clip",
"mwCkptBranch2_vae": "CkptBranch2_vae",
"mwCkpt_vaeNone": "Ckpt_vaeNone",
"mwSchedEdit": "SchedEdit",
"mwText": "Text",
"mwTextConcat": "TextConcat",
"mwTextReplace": "TextReplace",
"mwNumInt": "NumInt",
"mwNumIntx8": "NumIntx8",
"mwNumIntx16": "NumIntx16",
"mwNumIntx32": "NumIntx32",
"mwNumIntx64": "NumIntx64",
"mwNumIntx64s": "NumIntx64s",
"mwStartSizeXL": "StartSizeXL",
"mwNumIntBinaryIsh": "NumIntBinaryIsh",
"mwNumIntBinaryIsh2": "NumIntBinaryIsh2",
"mwsPipeCreate": "sPipeCreate",
"mwpsPipeCreate": "psPipeCreate",
"mwpsPipeCreate2": "psPipeCreate2",
"mwsPipeBranch": "sPipeBranch",
"mwpsPipeBranch": "psPipeBranch",
"mwpsPipeBranch2": "psPipeBranch2",
"mwsPipe_MultOver8": "sPipe_MultOver8",
"mwCompScale": "CompScale",
"mwMaskBoundingBoxRaw": "MaskBoundingBoxRaw",
"mwpsScale_LtntConform": "psScale_LtntConform",
"mwMaskBoundingBoxRF64": "MaskBoundingBoxRF64",
"mwCLIPSeg": "CLIPSeg",
"mwMaskConvert": "MaskConvert",
#"mwTester_ConditioningAverage":"Tester_ConditioningAverage",
#"mwTester_CLIPTextEncodeSDXL": "Tester_CLIPTextEncodeSDXL",
"mwTester_CLIPTextEncodeSDXL2": "Tester_CLIPTextEncodeSDXL2",
#"mwTester_CLIPTextEncode": "Tester_CLIPTextEncode",
"mwTester_condStrength": "Tester_condStrength",
"mwTupleIt": "TupleIt",
"mwTupleItStack": "TupleItStack",
"mwTupleIt_CondTest": "TupleIt_CondTest",
}
