import numpy as np
from PIL import Image, ImageFilter

class ImageTiler:
    def __init__(self, pil_image: Image, tile_size=1024, overlap=128, processing_function=None):
        self.image = pil_image
        self.tile_size = tile_size
        self.overlap = overlap
        self.processing_function = processing_function if processing_function else self._default_processing_function

    def _default_processing_function(self, tile):
        # Apply a simple Gaussian blur as the default processing function
        return tile.filter(ImageFilter.GaussianBlur(5))

    def _split_image(self):
        w, h = self.image.size
        tiles = []
        for y in range(0, h, self.tile_size - self.overlap):
            for x in range(0, w, self.tile_size - self.overlap):
                tile = self.image.crop((x, y, x+self.tile_size, y+self.tile_size))
                tile_w, tile_h = tile.size
                if tile_h % 64 != 0:
                    tile_h = (tile_h // 64) * 64
                if tile_w % 64 != 0:
                    tile_w = (tile_w // 64) * 64
                tile = tile.crop((0, 0, tile_w, tile_h))
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

    def _stitch_tiles(self, tiles):
        w, h = self.image.size
        stitched_image = Image.new("RGB", (w, h))

        tile_idx = 0
        for y in range(0, h, self.tile_size - self.overlap):
            for x in range(0, w, self.tile_size - self.overlap):
                tile = tiles[tile_idx]
                tile_w, tile_h = tile.size

                if x > 0:
                    left_tile = stitched_image.crop((x-self.overlap, y, x, y+tile_h))
                    blend = self._blend_seams(left_tile, tile)
                    stitched_image.paste(blend, (x, y))

                if y > 0:
                    top_tile = stitched_image.crop((x, y-self.overlap, x
