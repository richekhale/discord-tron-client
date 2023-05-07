import numpy as np
from PIL import Image
import logging, os

class ImageTiler:
class ImageTiler:
    def __init__(self, pil_image: Image, tile_size=64, overlap=8, processing_function=None):
        self.image = pil_image
        self.overlap = overlap
        self.processing_function = processing_function if processing_function else self._default_processing_function
        self.tile_size = tile_size

    def _default_processing_function(self, tile):
        logging.warning(f"No processing function provided for ImageTiler! Returning tile as-is.")
        return tile

    def _resize_image(self):
        w, h = self.image.size
        new_w = ((w + self.tile_size - self.overlap - 1) // (self.tile_size - self.overlap)) * (self.tile_size - self.overlap) + self.tile_size
        new_h = ((h + self.tile_size - self.overlap - 1) // (self.tile_size - self.overlap)) * (self.tile_size - self.overlap) + self.tile_size
        return self.image.resize((new_w, new_h), Image.ANTIALIAS)

    def _split_image(self):
        resized_image = self._resize_image()
        w, h = resized_image.size
        logging.debug(f"Splitting image into tiles of size {self.tile_size}x{self.tile_size}...")
        tiles = []
        for y in range(0, h, (self.tile_size - self.overlap) or 1):
            logging.debug(f'Processing row {y}... ({y/h*100:.2f}%)')
            maxcount = max(self.tile_size - self.overlap, 1)
            for x in range(0, w, maxcount):
                logging.debug(f'Processing tile {x}... ({x/w*100:.2f}%)')
                tile = resized_image.crop((x, y, x+self.tile_size, y+self.tile_size))
                tile_w, tile_h = tile.size
                if tile_h % 64 != 0:
                    logging.debug(f'Crop height {tile_h} is not a multiple of 64! Cropping...')
                    tile_h = (tile_h // 64) * 64
                if tile_w % 64 != 0:
                    logging.debug(f'Crop width {tile_w} is not a multiple of 64! Cropping...')
                    tile_w = (tile_w // 64) * 64
                tile = tile.crop((0, 0, tile_w, tile_h))
                logging.debug(f'Cropped image to {tile.size}. Appending to list.')
                tiles.append(tile)
        return tiles

    def _stitch_tiles(self, tiles, debug_dir=None):
        w, h = self.image.size
        tile_count_x = w // self.tile_size
        tile_count_y = h // self.tile_size
        total_w = self.tile_size * tile_count_x
        total_h = self.tile_size * tile_count_y
        stitched_image = Image.new("RGB", (total_w, total_h))
        logging.debug(f"Stitching {len(tiles)} tiles into image of size {total_w}x{total_h}...")
        logging.debug(f'Image has {tile_count_x} columns and {tile_count_y} rows.')

        tile_idx = 0
        for y in range(0, h, self.tile_size):
            for x in range(0, w, self.tile_size):
                tile = tiles[tile_idx]
                tile_w, tile_h = tile.size

                stitched_image.paste(tile, (x, y))
                tile_idx += 1

        return stitched_image

    async def process_image(self, user_config, scheduler_config, model_id, prompt, side_x, side_y, negative_prompt, steps, debug_dir=None):
        tiles = self._split_image()
        processed_tiles = []
        id = 0
        for tile in tiles:
            id += 1
            processed_tile = await self.processing_function(
                                    user_config=user_config,
                                    scheduler_config=scheduler_config,
                                    model_id=model_id,
                                    prompt=prompt,
                                    side_x=side_x,
                                    side_y=side_y,
                                    negative_prompt=negative_prompt,
                                    steps=steps,
                                    image=tile,
                                    promptless_variation=True
                                    )
            processed_tile.save(os.path.join(debug_dir, f"processed_tile_{id}.png"))
            processed_tiles.append(processed_tile)
        result = self._stitch_tiles(processed_tiles, debug_dir)
        return result
    async def process_debug_images(self, debug_dir):
        if not os.path.exists(debug_dir):
            raise ValueError(f"Debug directory {debug_dir} not found.")

        tiles = []
        for column in sorted(os.listdir(debug_dir)):
            column_dir = os.path.join(debug_dir, column)
            for row in sorted(os.listdir(column_dir)):
                row_dir = os.path.join(column_dir, row)
                if os.path.isdir(row_dir):
                    tile_path = os.path.join(row_dir, "image.png")
                    if os.path.exists(tile_path):
                        tile = Image.open(tile_path)
                        tiles.append(tile)

        result = self._stitch_tiles(tiles, debug_dir)
        result.save(os.path.join(debug_dir, "stitched_result.png"))

        return result
