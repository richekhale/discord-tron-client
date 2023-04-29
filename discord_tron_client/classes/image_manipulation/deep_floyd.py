import os
import torch
os.environ['FORCE_MEM_EFFICIENT_ATTN'] = "1"
import sys
from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import dream, style_transfer, super_resolution, inpainting
import torch.nn.functional as F
import random
import torchvision.transforms as T
import numpy as np
import requests
from PIL import Image
import torch
import re
from huggingface_hub import login

login()
device = 'cuda:0'
if_I = IFStageI('IF-I-XL-v1.0', device=device)
if_II = IFStageII('IF-II-L-v1.0', device=device)
if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device)
t5 = T5Embedder(device=device)
prompt = 'ultra close-up color photo portrait of rainbow owl with deer horns in the woods'

count = 4

result = dream(
    t5=t5, if_I=if_I, if_II=if_II, if_III=if_III,
    prompt=[prompt]*count,
    seed=42,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "smart50",
    },
)
if_I.show(result['I'], size=3)
if_I.show(result['II'], size=6)
if_I.show(result['III'], size=14)
