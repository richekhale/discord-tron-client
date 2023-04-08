import re, asyncio, websocket
from discord_tron_client.classes.discord_progress_bar import DiscordProgressBar

class TqdmCapture:
    def __init__(self, progress_bar: DiscordProgressBar, loop):
        self.progress_bar = progress_bar
        self.loop = loop
        self.output_file = "/tmp/tqdm_output.txt"

    def write(self, s: str):
        test_string = s.strip()
        if test_string != "":
            with open(self.output_file, 'a') as f:
                f.write(s)

            match = re.search(r'\b(\d+)%\|', s)
            if match:
                progress = int(match.group(1))
                asyncio.run_coroutine_threadsafe(self.progress_bar.update_progress_bar(progress), self.loop)

    def flush(self):
        with open(self.output_file, 'a') as f:
            f.flush()
