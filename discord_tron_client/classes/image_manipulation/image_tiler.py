import numpy as np
from PIL import Image
import logging, os

class ImageTiler:
    def __init__(self, pil_image: Image, processing_function=None):
        self.image = pil_image
        self.processing_function = processing_function if processing_function else self._default_processing_function
        self.tile_size = 64

    def _default_processing_function(self, tile):
        logging.warning(f"No processing function provided for ImageTiler! Returning tile as-is.")
        return tile

    def _split_image(self):
        w, h = self.image.size
        logging.debug(f"Splitting image into tiles of size {self.tile_size}x{self.tile_size}...")
        tiles = []
        for y in range(0, h, self.tile_size):
            logging.debug(f'Processing row {y}... ({y/h*100:.2f}%)')
            for x in range(0, w, self.tile_size):
                logging.debug(f'Processing tile {x}... ({x/w*100:.2f}%)')
                tile = self.image.crop((x, y, x+self.tile_size, y+self.tile_size))
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

    async def process_image(self, user_con