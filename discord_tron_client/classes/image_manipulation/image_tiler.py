import numpy as np
from PIL import Image
import logging, os

class ImageTiler:
    def __init__(self, pil_image: Image, tile_size=None, overlap=8, processing_function=None):
        self.image = pil_image
        self.overlap = overlap
        self.processing_function = processing_function if processing_function else self._default_processing_function

        if tile_size is None:
            self.tile_size = self._calculate_tile_size()
        else:
            self.tile_size = tile_size

    def _calculate_tile_size(self):
        w, h = self.image.size
        larger_dim = max(w, h)

        # Divide the larger dimension into 8 equal parts, rounding up to the nearest multiple of 64
        tile_size = ((larger_dim + 7) // 8 + 63) // 64 * 64

        return tile_size

    def _default_processing_function(self, tile):
        logging.warning(f"No processing function provided for ImageTiler! Returning tile as-is.")
        return tile

    def _split_image(self):
        w, h = self.image.size
        logging.debug(f"Splitting image into tiles of size {self.tile_size}x{self.tile_size}...")
        tiles = []
        for y in range(0, h, (self.tile_size - self.overlap) or 1):
            logging.debug(f'Processing row {y}... ({y/h*100:.2f}%)')
            maxcount = max(self.tile_size - self.overlap, 1)
            for x in range(0, w, maxcount):
                logging.debug(f'Processing tile {x}... ({x/w*100:.2f}%)')
                tile = self.image.crop((x, y, x+self.tile_size, y+self.tile_size))
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

    def _blend_seams(self, image1, image2, vertical=True):
        if vertical:
            image1_right = image1.crop((image1.width-self.overlap, 0, image1.width, image1.height))
            image2_left = image2.crop((0, 0, self.overlap, image2.height))
            blend_mask = np.linspace(1, 0, self.overlap).reshape(1, -1)
        else:
            image1_bottom = image1.crop((0, image1.height-self.overlap, image1.width, image1.height))
            image2_top = image2.crop((0, 0, image2.width, self.overlap))
            blend_mask = np.linspace(1, 0, self.overlap).reshape(-1, 1)

        blend_mask = np.repeat(blend_mask[..., np.newaxis], 3, axis=2)
        np_image1 = np.array(image1_right if vertical else image1_bottom).astype(np.float32)
        np_image2 = np.array(image2_left if vertical else image2_top).astype(np.float32)
        blended = np_image1 * blend_mask + np_image2 * (1 - blend_mask)
        blended = blended.astype(np.uint8)

        return Image.fromarray(blended)

    def _stitch_tiles(self, tiles, debug_dir=None):
        w, h = self.image.size
        stitched_tile_size = tiles[0].size
        tile_count_x = (w + stitched_tile_size - self.overlap - 1) // (stitched_tile_size - self.overlap)
        tile_count_y = (h + stitched_tile_size - self.overlap - 1) // (stitched_tile_size - self.overlap)
        total_w = (stitched_tile_size - self.overlap) * (tile_count_x - 1) + stitched_tile_size
        total_h = (stitched_tile_size - self.overlap) * (tile_count_y - 1) + stitched_tile_size
        stitched_image = Image.new("RGB", (total_w, total_h))
        logging.debug(f"Stitching {len(tiles)} tiles into image of size {total_w}x{total_h}...")
        if debug_dir:
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)

        tile_idx = 0
        for y in range(0, h, stitched_tile_size - self.overlap):
            for x in range(0, w, stitched_tile_size - self.overlap):
                tile = tiles[tile_idx]
                tile_w, tile_h = tile.size

                if debug_dir:
                    column_dir = os.path.join(debug_dir, f"{x // (stitched_tile_size - self.overlap)}")
                    row_dir = os.path.join(column_dir, f"{y // (stitched_tile_size - self.overlap)}")
                    if not os.path.exists(row_dir):
                        os.makedirs(row_dir)
                    tile_path = os.path.join(row_dir, "image.png")
                    tile.save(tile_path)

                if x > 0:
                    left_tile = stitched_image.crop((x-self.overlap, y, x, y+tile_h))
                    blend = self._blend_seams(left_tile, tile)
                    stitched_image.paste(blend, (x, y))

                if y > 0:
                    top_tile = stitched_image.crop((x, y-self.overlap, x+tile_w, y))
                    blend = self._blend_seams(top_tile, tile, vertical=False)
                    stitched_image.paste(blend, (x, y))

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
            if os.path.isdir(column_dir):
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