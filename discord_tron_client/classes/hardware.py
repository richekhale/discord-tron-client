import subprocess
import json, logging

class HardwareInfo:
    def __init__(self):
        self.gpu_type = "Unknown type"
        self.cpu_type = "Unknown type"
        self.memory_amount = None
        self.video_memory_amount = None
        self.disk_space_total = None
        self.disk_space_used = None

    def get_gpu_info(self):
        try:
            output = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
            self.gpu_type = output.decode().strip()
        except:
            try:
                output = subprocess.check_output(["rocm-smi", "--showproductname"])
                self.gpu_type = output.decode().strip()
            except:
                self.gpu_type = "Unknown"

    def get_cpu_info(self):
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if line.strip() and line.startswith('model name'):
                        self.cpu_type = line.strip().split(': ')[1]
                        break
        except:
            self.cpu_type = "Unknown"

    def get_memory_info(self):
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        self.memory_amount = int(int(line.split()[1]) / 1024 / 1024)
                        break
        except:
            self.memory_amount = "Unknown"

    def get_video_memory_info(self):
        try:
            output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
            self.video_memory_amount = int(output.decode().strip()) / 1024
        except:
            self.video_memory_amount = "Unknown"

    def get_disk_space_info(self):
        try:
            output = subprocess.check_output(["df", "-h"])
            for line in output.decode().split('\n'):
                if '/dev/' in line and not line.startswith('tmpfs'):
                    line = line.split()
                    self.disk_space_total = line[1]
                    self.disk_space_used = line[2]
                    break
        except:
            self.disk_space_total = "Unknown"
            self.disk_space_used = "Unknown"

    def get_machine_info(self):
        self.get_gpu_info()
        self.get_cpu_info()
        self.get_memory_info()
        self.get_video_memory_info()
        self.get_disk_space_info()

        return {
            'gpu_type': self.gpu_type,
            'cpu_type': self.cpu_type,
            'memory_amount': self.memory_amount,
            'video_memory_amount': self.video_memory_amount,
            'disk_space_total': self.disk_space_total,
            'disk_space_used': self.disk_space_used
        }