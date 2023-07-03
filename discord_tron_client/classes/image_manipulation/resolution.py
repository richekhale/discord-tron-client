import logging
from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.hardware import HardwareInfo

config = AppConfig()
hardware = HardwareInfo()


class ResolutionManager:
    resolutions = [
        # 1:1 aspect ratio
        {"width": 512, "height": 512, "speed_factor": 100},
        {"width": 768, "height": 768, "speed_factor": 30},
        {"width": 1024, "height": 1024, "speed_factor": 30, "default_max": True},
        {"width": 1280, "height": 1280, "speed_factor": 30},
        {"width": 2048, "height": 2048, "speed_factor": 30},
        # 2:3 aspect ratio
        {"width": 1024, "height": 1536, "speed_factor": 30, "default_max": True},
        {"width": 2048, "height": 3072, "speed_factor": 30},
        # Add more 2:3 aspect ratios around 1 MP.
        {"width": 896, "height": 1344, "speed_factor": 30},
        # 3:2 aspect ratio
        {"width": 1152, "height": 768, "speed_factor": 40},
        {"width": 1536, "height": 1024, "speed_factor": 30, "default_max": True},
        {"width": 3072, "height": 2048, "speed_factor": 30},
        # Add more 3:2 aspect ratios around 1 MP.
        {"width": 1024, "height": 680, "speed_factor": 30},
        # 16:9 aspect ratio
        {"width": 1280, "height": 720, "speed_factor": 30},
        {"width": 1920, "height": 1080, "speed_factor": 20, "default_max": True},
        {"width": 3840, "height": 2160, "speed_factor": 20},
        # Add more 16:9 aspect ratios around 1 MP.
        {"width": 1184, "height": 664, "speed_factor": 30},
        # 2.20:1 aspect ratio (Super Panavision 70, used in 2001: A Space Odyssey)
        {"width": 1248, "height": 568, "speed_factor": 30},
    ]

    @staticmethod
    def get_resolutions_with_extra_data(aspectratio=None):
        # Return ResolutionManager.resolutions after adding more information to each row, such as whether we will attention scale that resolution, and its aspect ratio
        for res in ResolutionManager.resolutions:
            if (
                aspectratio is not None
                and ResolutionManager.aspect_ratio(res) != aspectratio
            ):
                continue
            res["aspect_ratio"] = ResolutionManager.aspect_ratio(res)
            res["attention_scale"] = hardware.should_disable_resolution(res)
            if not hasattr(res, "default_max"):
                res["default_max"] = False
        return ResolutionManager.resolutions

    @staticmethod
    def get_resolutions_grouped_by_aspectratio(aspectratio=None):
        resolutions = {}
        for res in ResolutionManager.resolutions:
            if (
                aspectratio is None
                or ResolutionManager.aspect_ratio(res) == aspectratio
            ):
                if ResolutionManager.aspect_ratio(res) not in resolutions:
                    resolutions[ResolutionManager.aspect_ratio(res)] = []
                resolutions[ResolutionManager.aspect_ratio(res)].append(res)
        return resolutions

    @staticmethod
    def get_speed_factor(width, height, scaled_resolutions):
        for res in scaled_resolutions:
            if res["width"] == width and res["height"] == height:
                return int(res["speed_factor"])
        return None

    @staticmethod
    def is_valid_resolution(width, height):
        for res in ResolutionManager.resolutions:
            if res["width"] == width and res["height"] == height:
                return True
        total_pixel_area = width * height
        aspect_ratio = ResolutionManager.aspect_ratio(
            {"width": width, "height": height}
        )
        max_resolution = config.get_max_resolution_by_aspect_ratio(aspect_ratio)
        max_pixel_area = max_resolution["width"] * max_resolution["height"]
        if total_pixel_area > max_pixel_area:
            return False
        return True

    @staticmethod
    def aspect_ratio(resolution_item: dict):
        from math import gcd

        width = resolution_item["width"]
        height = resolution_item["height"]
        # Calculate the greatest common divisor of width and height
        divisor = gcd(width, height)

        # Calculate the aspect ratio
        ratio_width = width // divisor
        ratio_height = height // divisor

        # Return the aspect ratio as a string in the format "width:height"
        return f"{ratio_width}:{ratio_height}"

    # Generation resolutions have to be more carefully selected than resize resolutions.
    @staticmethod
    def nearest_generation_resolution(side_x: int, side_y: int):
        aspect_ratio = ResolutionManager.aspect_ratio(
            {"width": side_x, "height": side_y}
        )
        max_resolution = config.get_max_resolution_by_aspect_ratio(aspect_ratio)
        logging.info(f"Our max resolution config, {max_resolution}")
        requested_pixel_area = int(side_x) * int(side_y)
        max_pixel_area = int(max_resolution["width"]) * int(max_resolution["height"])
        if requested_pixel_area <= max_pixel_area:
            # Total pixel area is under our maximum.
            logging.debug(
                f"Using the provided resolution {side_x}x{side_y} as its requested pixel size {requested_pixel_area} is under our maximum resolution of {max_resolution['width']}x{max_resolution['height']} pixel size of {max_pixel_area}."
            )
            return side_x, side_y
        else:
            logging.info(
                f"Nearest resolution for {side_x}x{side_y} is larger than max resolution {max_resolution} and no better alternative could be found."
            )
            return max_resolution["width"], max_resolution["height"]

    @staticmethod
    def nearest_scaled_resolution(resolution: dict, user_config: dict):
        # We will scale by default, to 4x the requested resolution. Big energy!
        factor = user_config.get("resize", 1)
        if factor == 1 or factor == 0:
            # Do not bother rescaling if it's set to 1 or 0
            return resolution
        aspect_ratio = ResolutionManager.aspect_ratio(resolution)
        max_resolution_config = config.get_max_resolution_by_aspect_ratio(aspect_ratio)

        logging.info(
            f"Resize configuration is set by user factoring at {factor} based on our max resolution config, {max_resolution_config}."
        )

        width = resolution["width"]
        height = resolution["height"]

        new_width = int(width * factor)
        new_height = int(height * factor)
        new_aspect_ratio = ResolutionManager.aspect_ratio(
            {"width": new_width, "height": new_height}
        )

        max_resolution = ResolutionManager.get_highest_resolution(
            aspect_ratio, max_resolution_config
        )

        if ResolutionManager.is_valid_resolution(new_width, new_height):
            if int(new_width) * int(new_height) <= int(max_resolution["width"]) * int(
                max_resolution["height"]
            ):
                logging.info(
                    f"Nearest resolution for AR {aspect_ratio} is {new_width}x{new_height}."
                )
                return {"width": new_width, "height": new_height}
        # Loop through each of ResolutionManager.resolutions by aspect ratio to determine the first resolution that's >= the new resolution
        for res in ResolutionManager.resolutions:
            if (
                ResolutionManager.aspect_ratio(res) == aspect_ratio
                and res["width"] >= new_width
            ):
                logging.info(f"Nearest resolution for AR {aspect_ratio} is {res}.")
                return res

        logging.info(
            f"Nearest resolution for AR {aspect_ratio} is larger than max resolution {max_resolution} and no better alternative could be found."
        )
        return max_resolution

    @staticmethod
    def get_highest_resolution(aspect_ratio: str, max_resolution_config: dict):
        # Calculate the aspect ratio of the input image
        # Filter the resolutions list to only include resolutions with the same aspect ratio as the input image
        filtered_resolutions = [
            r
            for r in ResolutionManager.resolutions
            if ResolutionManager.aspect_ratio(r) == aspect_ratio
        ]

        # Check for a maximum resolution cap in the configuration
        max_res_cap = max_resolution_config.get(aspect_ratio)

        # If there's a cap, filter the sorted resolutions list to only include resolutions below the cap
        if max_res_cap:
            filtered_resolutions = [
                r
                for r in filtered_resolutions
                if r["width"] <= max_res_cap["width"]
                and r["height"] <= max_res_cap["height"]
            ]

        # Return the last (highest) resolution from the sorted list, or None if the list is empty
        return filtered_resolutions[-1] if filtered_resolutions else None

    @staticmethod
    def get_default_maximum(aspect_ratio: str):
        for res in ResolutionManager.resolutions:
            if (
                ResolutionManager.aspect_ratio(res) == aspect_ratio
                and "default_max" in res
                and res["default_max"]
            ):
                return res
        raise ValueError("No default maximum resolution could be ascertained.")

    @staticmethod
    def get_aspect_ratio_and_sides(config, resolution):
        # Current request's aspect ratio
        aspect_ratio = ResolutionManager.aspect_ratio(resolution)
        # Get the maximum resolution for the current aspect ratio
        side_x = config.get_max_resolution_width(aspect_ratio)
        side_y = config.get_max_resolution_height(aspect_ratio)
        logging.info(
            "Aspect ratio "
            + str(aspect_ratio)
            + " has a maximum resolution of "
            + str(side_x)
            + "x"
            + str(side_y)
            + "."
        )
        if resolution["width"] <= side_x and resolution["height"] <= side_y:
            side_x = resolution["width"]
            side_y = resolution["height"]
        return aspect_ratio, side_x, side_y

    @staticmethod
    def fudge_resolution_by_modulo(resolution, modulo: int = 8):
        # Typically we want resolution to be divisible by 8.
        width = resolution["width"]
        height = resolution["height"]
        if width % 8 > 0:
            width = width - (width % 8)
        if height % 8 > 0:
            height = height - (height % 8)
        return {"width": width, "height": height}
