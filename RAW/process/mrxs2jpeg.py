import os

# script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
# OPENSLIDE_PATH = os.path.join(script_dir, 'openslide', 'bin')

# if hasattr(os, 'add_dll_directory'):
#     with os.add_dll_directory(OPENSLIDE_PATH): 
#         import openslide.open_slide as open_slide
# import openslide.open_slide as open_slide
# from openslide import open_slide
import openslide
import numpy as np
from matplotlib import pyplot as plt
import cv2

def check_blank(img_np):
    image = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edge = cv2.Laplacian(image, cv2.CV_64F)
    return edge.mean()

if __name__ == '__main__':

    target_folder = "/workdir/radish/manhduong/Dataset-Osteosarcoma/Case8"
    outdir = "/workdir/radish/manhduong/images/Case_8"
    os.makedirs(outdir, exist_ok=True)
    cnt = 0

    for slide_name in os.listdir(target_folder):
        if not slide_name.endswith(".mrxs"): continue
        # Path to the slide file
        print("Processing: " + slide_name)
        slide_path = os.path.join(target_folder, slide_name)
        # slide_name = slide_path.split("//")[:-5]

        slide = openslide.open_slide(slide_path)
        slide_dims = slide.dimensions
        # print("CHECK POINT 1")
        # print(slide_dims)

        dims = slide.level_dimensions
        num_levels = len(dims)
        # print("CHECK POINT 2")

        # Example of saving an image
        level = 2
        slide_dim = dims[level]  # level 2
        ori_width, ori_height = dims[0]  # level 2
        print(slide_dim)

        offset_x = int(slide.properties['openslide.bounds-x'])
        offset_y = int(slide.properties['openslide.bounds-y'])
        bounds_width = int(slide.properties['openslide.bounds-width']) // (2**level)
        bounds_height = int(slide.properties['openslide.bounds-height']) // (2**level)

        #end_x = (offset_x + bounds_width) // (2**level)
        #end_y = (offset_y + bounds_height) // (2**level)
        #offset_x = offset_x // (2**level)
        #offset_y = offset_y // (2**level)

        #print(offset_x, offset_y, end_x, end_y)
        desired_x = 0 
        desired_y = 0
        original_image = np.array(
            slide.read_region((desired_x + offset_x, desired_y + offset_y), level, (bounds_width, bounds_height)), dtype='uint8')
        #original_image = cv2.resize(original_image, (original_image.shape[1]//4, original_image.shape[0]//4))
        # print("CHECK POINT 3")

        # Check if the image has an alpha channel (RGBA)
        if original_image.shape[2] == 4:  # Check if the image has 4 channels (RGBA)
            # Extract the alpha channel
            alpha_channel = original_image[:, :, 3]
            rgb_channel = np.max(original_image[:, :, :3], axis=2)

            # Create a mask where the alpha channel is 0 (fully transparent)
            transparent_mask = (alpha_channel == 0)
            background_mask = (rgb_channel == 0)
            # print(transparent_mask.mean)
            mask = np.logical_and(transparent_mask, background_mask)

            # Set the transparent pixels to white (255, 255, 255, 255)
            original_image[mask] = [255, 255, 255, 255]

            # Convert back to RGB by dropping the alpha channel
            original_image = original_image[:, :, :3]

        # print("CHECK POINT 4")
        plt.imsave(os.path.join(
            outdir, slide_name[:-5] + '.png'), 
            original_image)

        cnt += 1

    print(f"[Summary]: {cnt} valid conversion")
