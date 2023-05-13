from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch, gc, logging
from split_image import split
import os


class ImageSplitter:
    def __init__(self, rows, cols, should_square, padding=0, should_quiet=False):
        self.rows = rows
        self.cols = cols
        self.should_square = should_square
        self.padding = padding
        self.should_quiet = should_quiet

    def split(self, im):
        im_width, im_height = im.size
        row_width = int(im_width / self.cols)
        row_height = int(im_height / self.rows)
        name = "image"
        ext = ".png"
        name = os.path.basename(name)
        images = []
        if self.should_square:
            im, row_width, row_height = self._square_image(im, row_width, row_height)
        return self._crop_images(im, row_width, row_height, name, ext)

    def split_with_padding(self, im):
        im_width, im_height = im.size
        row_width = int(im_width / self.cols)
        row_height = int(im_height / self.rows)

        if self.should_square:
            im, row_width, row_height = self._square_image(im, row_width, row_height)

        padded_images = []
        for i in range(self.rows):
            for j in range(self.cols):
                left = max(0, j * row_width - self.padding)
                upper = max(0, i * row_height - self.padding)
                right = min(im_width, (j + 1) * row_width + self.padding)
                lower = min(im_height, (i + 1) * row_height + self.padding)
                box = (left, upper, right, lower)
                padded_image = im.crop(box)
                padded_images.append(padded_image)

        return padded_images

    def _square_image(self, im, row_width, row_height):
        min_dimension = min(im.width, im.height)
        max_dimension = max(im.width, im.height)
        bg_color = split.determine_bg_color(im)
        im = self._create_new_image(im, bg_color, min_dimension, max_dimension)
        row_width = int(max_dimension / self.cols)
        row_height = int(max_dimension / self.rows)
        return im, row_width, row_height

    def _create_new_image(self, im, bg_color, min_dimension, max_dimension):
        im_r = Image.new(
            "RGBA" if ".png" else "RGB", (max_dimension, max_dimension), bg_color
        )
        offset = int((max_dimension - min_dimension) / 2)
        if im.width > im.height:
            im_r.paste(im, (0, offset))
        else:
            im_r.paste(im, (offset, 0))
        return im_r

    def _crop_images(self, im, row_width, row_height, name, ext):
        images = []
        n = 0
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                box = (
                    j * row_width,
                    i * row_height,
                    j * row_width + row_width,
                    i * row_height + row_height,
                )
                outp = im.crop(box)
                outp_path = name + "_" + str(n) + ext
                if not self.should_quiet:
                    print("Exporting image tile: " + outp_path)
                images.append(outp)
                n += 1
        return images


class ImageResizer:
    @staticmethod
    def resize_for_condition_image(input_image: Image, resolution: int):
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        img = input_image.resize((W, H), resample=Image.LANCZOS)
        return img


class ImageUpscaler:
    def __init__(
        self, pipeline, generator, rows=3, cols=3, padding=64, blend_alpha=0.5
    ):
        self.pipeline = pipeline
        self.generator = generator
        self.rows = rows
        self.cols = cols
        self.padding = padding
        self.blend_alpha = blend_alpha

    def upscale(self, image):
        original_width, original_height = image.size
        max_dimension = max(original_width, original_height)
        splitter = ImageSplitter(self.rows, self.cols, True, self.padding, False)
        tiles = splitter.split_with_padding(image)
        ups_tiles = []
        for idx, tile in enumerate(tiles, start=1):
            logging.info(f"Upscaling tile {idx} of {len(tiles)}")
            conditioned_image = ImageResizer.resize_for_condition_image(
                input_image=tile, resolution=1024
            )
            ups_tile = self._get_upscaled_tile(conditioned_image)
            ups_tiles.append(ups_tile)
        return [
            self._merge_tiles(
                tiles, ups_tiles, max_dimension, original_width, original_height
            )
        ]

    def _get_upscaled_tile(self, conditioned_image):
        return self.pipeline(
            prompt="best quality",
            negative_prompt="blur, lowres, bad anatomy, bad hands, cropped, worst quality",
            image=conditioned_image,
            controlnet_conditioning_image=conditioned_image,
            width=conditioned_image.size[0],
            height=conditioned_image.size[1],
            strength=0.7,
            generator=self.generator,
            num_inference_steps=32,
        ).images[0]

    def _merge_tiles(
        self, tiles, ups_tiles, max_dimension, original_width, original_height
    ):
        side = ups_tiles[0].width
        ups_times = abs(side / tiles[0].width)
        new_size = (max_dimension * ups_times, max_dimension * ups_times)
        total_width = self.cols * side
        total_height = self.rows * side
        logging.info(
            f"New image size: {new_size}, total width: {total_width}, total height: {total_height}"
        )
        merged_image = self._create_blank_image(total_width, total_height)
        merged_image = self._paste_tiles(merged_image, ups_tiles, side)
        return self._crop_final_image(
            merged_image, new_size, original_width, original_height, ups_times
        )

    @staticmethod
    def _create_blank_image(total_width, total_height):
        return Image.new("RGB", (total_width, total_height))

    def _paste_tiles(self, merged_image, ups_tiles, side):
        current_width = 0
        current_height = 0
        maximum_width = self.cols * side

        for idx, ups_tile in enumerate(ups_tiles):
            box = (
                max(0, current_width),
                max(0, current_height),
                min(merged_image.width, current_width + side),
                min(merged_image.height, current_height + side),
            )

            if idx != 0:  # Don't blend for the first tile
                overlap_box = (
                    max(0, current_width - self.padding),
                    max(0, current_height - self.padding),
                    min(merged_image.width, current_width + self.padding),
                    min(merged_image.height, current_height + self.padding),
                )

                prev_tile = merged_image.crop(overlap_box)

                # Calculate overlap box relative to the upscaled tile
                ups_tile_overlap_box = (
                    max(0, self.padding),
                    max(0, self.padding),
                    min(ups_tile.width, self.padding + box[2] - overlap_box[0]),
                    min(ups_tile.height, self.padding + box[3] - overlap_box[1]),
                )

                ups_tile_overlap = ups_tile.crop(ups_tile_overlap_box)

                ups_tile_blend = Image.blend(
                    prev_tile, ups_tile_overlap, self.blend_alpha
                )

                merged_image.paste(ups_tile_blend, overlap_box)

            merged_image.paste(ups_tile, box)

            current_width += side
            if current_width >= maximum_width:
                current_width = 0
                current_height += side

        return merged_image

    @staticmethod
    def _crop_final_image(
        merged_image, new_size, original_width, original_height, ups_times
    ):
        crop_left = (new_size[0] - original_width * ups_times) // 2
        crop_upper = (new_size[1] - original_height * ups_times) // 2
        crop_right = crop_left + original_width * ups_times
        crop_lower = crop_upper + original_height * ups_times
        return merged_image.crop((crop_left, crop_upper, crop_right, crop_lower))
