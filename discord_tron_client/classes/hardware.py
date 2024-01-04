import subprocess, torch
import logging, socket
from discord_tron_client.classes.app_config import AppConfig
from diffusers.utils.logging import set_verbosity_warning

set_verbosity_warning()
config = AppConfig()


class HardwareInfo:
    identifier = None

    def __init__(self):
        self.gpu_type = "Unknown type"
        self.cpu_type = "Unknown type"
        self.memory_amount = None
        self.video_memory_amount = None
        self.disk_space_total = None
        self.disk_space_used = None
        self.get_system_capabilities()
        self.get_machine_info()

    def get_register_data(self, worker_id: str):
        return {
            "supported_job_types": self.get_system_capabilities(),
            "hardware_limits": self.get_hardware_limits(),
            "worker_id": worker_id,
        }

    @classmethod
    def get_identifier(cls):
        if HardwareInfo.identifier is not None:
            return HardwareInfo.identifier
        HardwareInfo.identifier = (
            config.get_friendly_name() or HardwareInfo.get_system_hostname()
        )
        import random

        HardwareInfo.identifier = (
            HardwareInfo.identifier + "-" + str(random.randint(0, 2))
        )
        logging.info(
            f"Identifier not found, configuring new one: {HardwareInfo.identifier}"
        )
        return HardwareInfo.identifier

    def should_offload(self):
        if not config.enable_offload():
            return False
        return (
            self.get_video_memory_info() == "Unknown" or self.video_memory_amount < 48
        )

    def should_sequential_offload(self):
        if not config.enable_offload() or not config.enable_sequential_offload():
            return False
        return self.should_offload() and self.video_memory_amount < 10

    def get_system_capabilities(self):
        self.get_gpu_info()
        self.get_cpu_info()
        self.get_memory_total()
        self.get_video_memory_info()
        self.get_disk_space()
        capabilities = {
            "llama": config.is_llama_enabled(),
            "stablelm": config.is_stablelm_enabled(),
            "stablevicuna": config.is_stablevicuna_enabled(),
            "tts_bark": config.is_bark_enabled(),
        }
        if (
            config.enable_diffusion()
            and (
                type(self.video_memory_amount) is float
                or type(self.video_memory_amount) is int
                or self.video_memory_amount.isnumeric()
            )
            and int(self.video_memory_amount) >= 8
        ):
            capabilities["gpu"] = True
            capabilities["variation"] = True
        if int(self.memory_amount) >= 16:
            capabilities["memory"] = True
        if int(self.get_cpu_count()) >= 16:
            capabilities["compute"] = True
        return capabilities

    def get_hardware_limits(self):
        limits = {}
        limits["gpu"] = self.video_memory_amount
        limits["memory"] = self.memory_amount
        limits["cpu"] = self.get_cpu_count()
        return limits

    def get_gpu_info(self):
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
            )
            self.gpu_type = output.decode().strip()
        except:
            try:
                output = subprocess.check_output(["rocm-smi", "--showproductname"])
                self.gpu_type = output.decode().strip()
            except:
                self.gpu_type = "Unknown"

    def get_cpu_info(self):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.strip() and line.startswith("model name"):
                        self.cpu_type = line.strip().split(": ")[1]
                        break
        except:
            self.cpu_type = "Unknown"

    def get_cpu_count(self):
        try:
            with open("/proc/cpuinfo") as f:
                processor_count = 0
                for line in f:
                    if line.strip() and line.startswith("processor"):
                        processor_count += 1
                return processor_count
        except:
            return -1

    def get_memory_total(self):
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        self.memory_amount = int(int(line.split()[1]) / 1024 / 1024)
                        torch.backends.cudnn.benchmark = False
                        # if (int(self.memory_amount)) < 10:
                        #     logging.warn(f"Enabling cuDNN benchmark. Could affect determinism. Obtain a GPU with more than 10G RAM to fix this.")
                        #     torch.backends.cudnn.benchmark = True
                        break
        except:
            self.memory_amount = "Unknown"
        return self.memory_amount

    def get_memory_free(self):
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        self.memory_free = int(int(line.split()[1]) / 1024 / 1024)
                        break
        except:
            self.memory_free = "Unknown"
        return self.memory_free

    def get_video_memory_info(self):
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ]
            )
            self.video_memory_amount = int(output.decode().strip()) / 1024
        except:
            self.video_memory_amount = "Unknown"
        return self.video_memory_amount

    def get_concurrent_pipe_count(self):
        memory_amount = self.get_machine_info()["video_memory_amount"]
        if memory_amount == "Unknown":
            # If we do not know how much vmem we have, that is a bad sign.
            return 1
        gb = int(memory_amount)
        if gb == 8:
            # We have 12GiB per model, essentially.
            return 1
        pipe_count = int(gb / 10)
        logging.debug(
            f"HardwareInfo: {pipe_count} concurrent pipes allowed via {gb}GiB VRAM and 12GiB allocated per pipe."
        )
        return pipe_count

    def get_gpu_power_consumption(self):
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ]
            )
            power_consumption = output.decode().strip()
            return power_consumption
        except Exception as e:
            import traceback

            logging.error(
                f"Caught exception during get_gpu_power_consumption: {e}, traceback: {traceback.format_exc()}"
            )
            return -1

    def get_disk_space(self):
        import psutil
        from discord_tron_client.classes.app_config import AppConfig

        config = AppConfig()
        target_partition = config.config.get("target_partition", "/")
        logging.debug("Looking for disk space info for partition: " + target_partition)
        partitions = psutil.disk_partitions()
        for partition in partitions:
            if partition.device != target_partition:
                continue
            usage = psutil.disk_usage(partition.mountpoint)
            self.disk_space_total = usage.total
            self.disk_space_used = usage.used
            break

    @staticmethod
    def get_system_hostname():
        hostname = socket.gethostname()
        return hostname

    def get_simple_hardware_info(self):
        self.get_machine_info()
        identifier = HardwareInfo.get_identifier()
        return {
            "gpu": self.gpu_type,
            "cpu": self.cpu_type,
            "cpu_count": self.get_cpu_count(),
            "memory_amount": self.memory_amount,
            "video_memory_amount": self.video_memory_amount,
            "hostname": identifier,
        }

    def should_disable_resolution(self, resolution: dict):
        gpu_memory = self.video_memory_amount
        if gpu_memory == "Unknown":
            gpu_memory = 8
        pixel_count = resolution["width"] * resolution["height"]
        memory_to_pixel_ratio = gpu_memory * (1024**3) / pixel_count
        # In practice, an 8GB GPU can handle about 1280x720 which is a ratio of 9320 pixels per GiB.
        resolution_performance_threshold = 9000
        result = memory_to_pixel_ratio < resolution_performance_threshold
        # logging.debug(f"Resolution {resolution} requires {memory_to_pixel_ratio} pixels per GiB of video memory. This resolution is is {'disabled' if result else 'enabled'}.")
        return result

    def get_compute_capability():
        pass

    def get_machine_info(self):
        self.get_gpu_info()
        self.get_cpu_info()
        self.get_cpu_count()
        self.get_memory_total()
        self.get_video_memory_info()
        self.get_disk_space()
        identifier = HardwareInfo.get_identifier()
        return {
            "gpu_type": self.gpu_type,
            "cpu_type": self.cpu_type,
            "cpu_count": self.get_cpu_count(),
            "memory_amount": self.memory_amount,
            "video_memory_amount": self.video_memory_amount,
            "disk_space_total": self.disk_space_total,
            "disk_space_used": self.disk_space_used,
            "hostname": identifier,
        }
