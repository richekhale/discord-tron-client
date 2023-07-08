import re, asyncio, logging
from discord_tron_client.classes.discord_progress_bar import DiscordProgressBar
from discord_tron_client.classes.hardware import HardwareInfo


class TqdmCapture:
    def __init__(self, progress_bar: DiscordProgressBar, loop):
        self.progress_bar = progress_bar
        self.loop = loop
        self.output_file = "/tmp/tqdm_output.txt"
        self.hardware_info = HardwareInfo()
        self.gpu_power_consumption = (
            0.0  # Store GPU power use here, since it's inside the loop.
        )
        # Start below zero so that a progress of zero will begin the bar.
        self.progress = -1

    def write(self, s: str):
        test_string = s.strip()
        if test_string != "":
            with open(self.output_file, "a") as f:
                f.write(s)

            match = re.search(r"\b(\d+)%\|", s)
            if match:
                progress = int(match.group(1))
                if progress <= self.progress:
                    # If we have anything less than what we started with, don't send.
                    return
                self.progress = progress

                if progress >= 50 and progress <= 60:
                    # Record GPU power use around 60% progress.
                    gpu_power_consumption = float(
                        self.hardware_info.get_gpu_power_consumption()
                    )
                    if gpu_power_consumption > self.gpu_power_consumption:
                        # Store the maximum power used rather than a random sample.
                        self.gpu_power_consumption = gpu_power_consumption
                asyncio.run_coroutine_threadsafe(
                    self.progress_bar.update_progress_bar(progress), self.loop
                )

    def flush(self):
        with open(self.output_file, "a") as f:
            f.flush()
