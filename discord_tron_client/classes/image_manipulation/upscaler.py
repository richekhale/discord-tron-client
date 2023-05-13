from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch, gc, logging
from split_image import split
import os
import random


def split_image(im, rows, cols, should_square, should_quiet=False):
    im_width, im_height = im.size
    row_width = int(im_width / cols)
    row_height = int(im_height / rows)
    name = "image"
    ext = ".png"
    name = os.path.basename(name)
    images = []
    if should_square:
        min_dimension = min(im_width, im_height)
        max_dimension = max(im_width, im_height)
        if not should_quiet:
            print("Resizing image to a square...")
            print("Determining background color...")
        bg_color = split.determine_bg_color(im)
        if not should_quiet:
            print("Background color is... " + str(bg_color))
        im_r = Image.new(
            "RGBA" if ext == "png" else "RGB", (max_dimension, max_dimension), bg_color
        )
        offset = int((max_dimension - min_dimension) / 2)
        if im_width > im_height:
            im_r.paste(im, (0, offset))
        else:
            im_r.paste(im, (offset, 0))
        im = im_r
        row_width = int(max_dimension / cols)
        row_height = int(max_dimension / rows)
    n = 0
    for i in range(0, rows):
        for j in range(0, cols):
            box = (
                j * row_width,
                i * row_height,
                j * row_width + row_width,
                i * row_height + row_height,
            )
            outp = im.crop(box)
            outp_path = name + "_" + str(n) + ext
            if not should_quiet:
                print("Exporting image tile: " + outp_path)
            images.append(outp)
            n += 1
    return [img for img in images]


def _resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


def upscale_image(
    pipeline,
    generator,
    image,
    prompt,
    negative_prompt,
    guidance=7,
    steps=50,
    rows=3,
    cols=3,
):
    original_width, original_height = image.size
    max_dimension = max(original_width, original_height)
    tiles = split_image(image, rows, cols, True, False)
    ups_tiles = []
    i = 0
    for x in tiles:
        i = i + 1
        logging.info("Upscaling tile " + str(i) + " of " + str(len(tiles)))
        conditioned_image = _resize_for_condition_image(input_image=x, resolution=1024)
        ups_tile = pipeline(
            prompt="best quality " + prompt,
            negative_prompt="blur, lowres, bad anatomy, bad hands, cropped, worst quality "
            + negative_prompt,
            image=image,
            controlnet_conditioning_image=conditioned_image,
            width=conditioned_image.size[0],
            height=conditioned_image.size[1],
            strength=1.0,
            generator=generator,
            num_inference_steps=32,
        ).images[0]
        torch.cuda.empty_cache()
        gc.collect()
        ups_tiles.append(ups_tile)

    # Determine the size of the merged upscaled image
    total_width = 0
    total_height = 0
    side = 0
    for ups_tile in ups_tiles:
        side = ups_tile.width
        break
    for x in tiles:
        tsize = x.width
        break

    ups_times = abs(side / tsize)
    new_size = (max_dimension * ups_times, max_dimension * ups_times)
    total_width = cols * side
    total_height = rows * side
    logging.info(
        f"New image size: {new_size}, total width: {total_width}, total height: {total_height}"
    )

    # Create a blank image with the calculated size
    merged_image = Image.new("RGB", (total_width, total_height))

    # Paste each upscaled tile into the blank image
    current_width = 0
    current_height = 0
    maximum_width = cols * side
    for ups_tile in ups_tiles:
        merged_image.paste(ups_tile, (current_width, current_height))
        current_width += ups_tile.width
        if current_width >= maximum_width:
            current_width = 0
            current_height = current_height + side

    # Using the center of the image as pivot, crop the image to the original dimension times four
    crop_left = (new_size[0] - original_width * ups_times) // 2
    crop_upper = (new_size[1] - original_height * ups_times) // 2
    crop_right = crop_left + original_width * ups_times
    crop_lower = crop_upper + original_height * ups_times
    final_image = merged_image.crop((crop_left, crop_upper, crop_right, crop_lower))

    # The resulting image should be identical to the original image in proportions / aspect ratio, with no loss of elements.
    # Save the merged image
    return [final_image]
