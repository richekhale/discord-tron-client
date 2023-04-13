import subprocess
import logging, socket


class HardwareInfo:
    def __init__(self):
        self.gpu_type = "Unknown type"
        self.cpu_type = "Unknown type"
        self.memory_amount = None
        self.video_memory_amount = None
        self.disk_space_total = None
        self.disk_space_used = None
        self.get_system_capabilities()

    def get_register_data(self, worker_id: str):
        return {
            "supported_job_types": self.get_system_capabilities(),
            "hardware_limits": self.get_hardware_limits(),
            "worker_id": worker_id
        }

    def get_system_capabilities(self):
        self.get_gpu_info()
        self.get_cpu_info()
        self.get_memory_total()
        self.get_video_memory_info()
        self.get_disk_space()
        capabilities = {}
        if self.video_memory_amount >= 8:
            capabilities["gpu"] = True
        if self.memory_amount >= 16:
            capabilities["memory"] = True
        if self.get_cpu_count() >= 16:
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
                        break
        except:
            self.memory_amount = "Unknown"

    def get_memory_free(self):
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        self.memory_amount = int(int(line.split()[1]) / 1024 / 1024)
                        break
        except:
            self.memory_amount = "Unknown"

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

    def get_concurrent_pipe_count(self):
        memory_amount = self.get_memory_total()
        if memory_amount == "Unknown":
            # If we do not know how much vmem we have, that is a bad sign.
            return 1
        gb = int(memory_amount)
        if gb == 8:
            # We have 8GiB per model, essentially.
            return 1
        pipe_count = int(gb / 8)
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
        except:
            import traceback
            logging.error(f"Caught exception during get_gpu_power_consumption: {e}, traceback: {traceback.format_exc()}")
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

    def get_system_hostname(self):
        hostname = socket.gethostname()
        return hostname

    def get_simple_hardware_info(self):
        self.get_machine_info()
        return {
            "gpu": self.gpu_type,
            "cpu": self.cpu_type,
            "cpu_count": self.get_cpu_count(),
            "memory_amount": self.memory_amount,
            "video_memory_amount": self.video_memory_amount,
            "hostname": self.get_system_hostname()
        }

    def get_machine_info(self):
        self.get_gpu_info()
        self.get_cpu_info()
        self.get_cpu_count()
        self.get_memory_total()
        self.get_video_memory_info()
        self.get_disk_space()
        return {
            "gpu_type": self.gpu_type,
            "cpu_type": self.cpu_type,
            "cpu_count": self.get_cpu_count(),
            "memory_amount": self.memory_amount,
            "video_memory_amount": self.video_memory_amount,
            "disk_space_total": self.disk_space_total,
            "disk_space_used": self.disk_space_used,
            "hostname": self.get_system_hostname()
        }
