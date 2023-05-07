import cv2
import numpy as np
from PIL import Image

class ImageTiler:
    def __init__(self, pil_image: Image, tile_size=1024, overlap=128, processing_function=None):
        self.image = ImageTiler.pil_to_cv2(pil_image)
        self.tile_size = tile_size
        self.overlap = overlap
        self.processing_function = processing_function if processing_function else self._default_processing_function

    @staticmethod
    def pil_to_cv2(pil_image):
        cv2_image = np.array(pil_image)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
        return cv2_image
    @staticmethod
    def cv2_to_pil(cv2_image):
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv2_image)
        return pil_image
    def _default_processing_function(self, tile):
        # Apply a simple Gaussian blur as the default processing function
        return cv2.GaussianBlur(tile, (5, 5), 0)

    def _split_image(self):
        h, w, _ = self.image.shape
        tiles = []
        for y in range(0, h, self.tile_size - self.overlap):
            for x in range(0, w, self.tile_size - self.overlap):
                tile = self.image[y:y+self.tile_size, x:x+self.tile_size]
                tile_h, tile_w, _ = tile.shape
                if tile_h % 64 != 0:
                    tile_h = (tile_h // 64) * 64
                if tile_w % 64 != 0:
                    tile_w = (tile_w // 64) * 64
                tile = tile[:tile_h, :tile_w]
                tiles.append(tile)
        return tiles

    def _blend_seams(self, image1, image2, vertical=True):
        if vertical:
            image1_right = image1[:, -self.overlap:]
            image2_left = image2[:, :self.overlap]
            blend_mask = np.linspace(1, 0, self.overlap).reshape(1, -1)
        else:
            image1_bottom = image1[-self.overlap:, :]
            image2_top = image2[:self.overlap, :]
            blend_mask = np.linspace(1, 0, self.overlap).reshape(-1, 1)

        blend_mask = cv2.merge([blend_mask] * 3)
        blended = image1_right * blend_mask + image2_left * (1 - blend_mask)

        return blended

    def _stitch_tiles(self, tiles):
        h, w, _ = self.image.shape
        stitched_image = np.zeros_like(self.image)

        tile_idx = 0
        for y in range(0, h, self.tile_size - self.overlap):
            for x in range(0, w, self.tile_size - self.overlap):
                tile = tiles[tile_idx]
                tile_h, tile_w, _ = tile.shape

                if x > 0:
                    left_tile = stitched_image[y:y+tile_h, x-self.overlap:x+tile_w-self.overlap]
                    blend = self._blend_seams(left_tile, tile)
                    stitched_image[y:y+tile_h, x:x+self.overlap] = blend

                if y > 0:
                    top_tile = stitched_image[y-self.overlap:y+tile_h-self.overlap, x:x+tile_w]
                    blend = self._blend_seams(top_tile, tile, vertical=False)
                    stitched_image[y:y+self.overlap, x:x+tile_w] = blend

                stitched_image[y:y+tile_h, x:x+tile_w] = tile
                tile_idx += 1

        return stitched_image

    async def process_image(self, user_config, scheduler_config, model_id, prompt, side_x, side_y, negative_prompt, steps):
        tiles = self._split_image()
        processed_tiles = []
        for tile in tiles:
            processed_tile = ImageTiler.pil_to_cv2(await self.processing_function(
                                    user_config=user_config,
                                    scheduler_config=scheduler_config,
                                    model_id=model_id,
                                    prompt=prompt,
                                    side_x=side_x,
                                    side_y=side_y,
                                    negative_prompt=negative_prompt,
                                    steps=steps,
                                    image=ImageTiler.cv2_to_pil(tile),
                                    promptless_variation=True
                                    ))
            processed_tiles.append(processed_tile)
        result = self._stitch_tiles(processed_tiles)
        return ImageTiler.cv2_to_pil(result)
