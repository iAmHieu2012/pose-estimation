import os
from os.path import exists, join, basename, splitext

#@markdown ### ▶️ Setup
#@markdown Run this first.


from os import listdir
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from diffusers.utils import load_image
from tqdm import tqdm

#@markdown ### ▶️ Process Folder
#@markdown Folder to process
folder_name = "E:/Code/test/output" #@param {type:"string"}

outputDir = os.path.join(folder_name, 'controlnet')


# import the modules
openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

print("Processing images")
# get the path/directory
for images in tqdm(sorted(os.listdir(folder_name))):
    #print(images)
    # check if the image ends with png or jpg or jpeg
    if (images.endswith(".png") or images.endswith(".jpg")\
        or images.endswith(".jpeg")):

        outfile = os.path.join(outputDir, images)
        #print(outfile)
        image = load_image(os.path.join(folder_name,images))
        image = openpose(image)
        image.save(outfile)


