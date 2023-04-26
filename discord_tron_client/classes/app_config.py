# classes/app_config.py

import json, os, logging, traceback
from concurrent.futures import ThreadPoolExecutor

class AppConfig:
    # Class variables
    main_loop = None
    main_api_client = None
    main_pipelinemanager = None
    main_websocket = None
    image_processing_executor = None
    # Initialize the config object.
    def __init__(self):
        from pathlib import Path
        parent = os.path.dirname(Path(__file__).resolve().parent)
        self.parent = parent
        config_path = os.path.join(parent, "config")
        self.config_path = os.path.join(config_path, "config.json")
        self.example_config_path = os.path.join(config_path, "example.json")
        self.auth_ticket_path = os.path.join(config_path, "auth.json")
        self.reload_config()

    def reload_config(self):
        if not os.path.exists(self.config_path):
            with open(self.example_config_path, "r") as example_file:
                example_config = json.load(example_file)
            with open(self.config_path, "w") as config_file:
                json.dump(example_config, config_file, indent=4)
        with open(self.config_path, "r") as config_file:
            self.config = json.load(config_file)

    @classmethod
    def set_loop(cls, loop):
        cls.main_loop = loop
    @classmethod
    def set_api_client(cls, api_client):
        cls.main_api_client = api_client
    @classmethod
    def set_pipeline_manager(cls, pipelinemanager):
        cls.main_pipelinemanager = pipelinemanager
    @classmethod
    def set_websocket(cls, websocket):
        cls.main_websocket = websocket
    @classmethod
    def get_websocket(cls):
        return cls.main_websocket

    @classmethod
    def get_image_worker_thread(cls):
        if cls.get_image_worker_thread is None:
            raise Exception(f"AppConfig.image_processing_executor is not set! Traceback: {traceback.format_stack()}")

    @classmethod
    def set_image_worker_thread(cls):
        if cls.image_processing_executor is not None:
            return cls.image_processing_executor
        config = AppConfig()
        cls.image_processing_executor = ThreadPoolExecutor(max_workers=config.get_concurrent_slots())
        
    @classmethod
    def get_loop(cls):
        return cls.main_loop
    @classmethod
    def get_api_client(cls):
        return cls.main_api_client
    @classmethod
    def get_pipeline_manager(cls):
        return cls.main_pipelinemanager
    def get_log_level(self):
        self.reload_config()
        level = self.config.get("log_level", "INFO")
        result = getattr(logging, level.upper(), "ERROR")
        return result

    # Retrieve the OAuth ticket information.
    def get_auth_ticket(self):
        with open(self.auth_ticket_path, "r") as auth_ticket:
            auth_data = json.load(auth_ticket)
            return auth_data
    def get_tls_key_path(self):
        return self.parent + '/' + self.config.get("websocket_hub", {}).get("server_key_path", 'config/server_key.pem')
    def get_tls_pem_path(self):
        return self.parent + '/' + self.config.get("websocket_hub", {}).get("server_pem_path", 'config/server_cert.pem')
    def get_master_api_key(self):
        return self.config.get("master_api_key", None)

    def get_concurrent_slots(self):
        return self.config.get("concurrent_slots", 1)

    def get_command_prefix(self):
        return self.config.get("cmd_prefix", "+")

    def get_friendly_name(self):
        return self.config.get("friendly_name", False)

    def get_max_resolution_by_aspect_ratio(self, aspect_ratio: str):
        return self.config.get("maxres", {}).get(aspect_ratio, {"width": self.get_max_resolution_width(aspect_ratio=aspect_ratio), "height": self.get_max_resolution_height(aspect_ratio=aspect_ratio)})
    def get_max_resolution_width(self, aspect_ratio: str):
        return self.config.get("maxres", {}).get(aspect_ratio, {}).get("width", 3840)
    def get_max_resolution_height(self, aspect_ratio: str):
        return self.config.get("maxres", {}).get(aspect_ratio, {}).get("height", 2160)

    def get_precision_bits(self):
        return self.config.get("precision_bits", 16)
    def get_cuda_cache_clear_toggle(self):
        return self.config.get("cuda_cache_clear", True)

    def get_web_root(self):
        return self.config.get("web_root", "/var/www/localhost/htdocs")

    def get_master_url(self):
        hostname = str(self.get_websocket_hub_host())
        logging.debug("Websucket hub host: "+hostname)
        return self.config.get("master_url", "https://"+hostname+":5000")

    def verify_master_ssl(self):
        return self.config.get("websocket_hub", {}).get("verify_ssl", False)

    def get_websocket_hub_host(self):
        return self.config.get("websocket_hub", {}).get("host", "localhost")
    def get_websocket_hub_port(self):
        return self.config.get("websocket_hub", {}).get("port", 6789)
    def get_websocket_hub_tls(self):
        return self.config.get("websocket_hub", {}).get("tls", False)
    def get_websocket_config(self):
        protocol="ws"
        if self.get_websocket_hub_tls():
            protocol="wss"
        return {'host': self.get_websocket_hub_host(), 'port': self.get_websocket_hub_port(), 'tls': self.get_websocket_hub_tls(), 'protocol': protocol, 'server_cert_path': self.get_tls_pem_path(), 'server_key_path': self.get_tls_key_path()}

    def get_max_concurrent_uploads(self):
        return self.config.get("max_concurrent_uploads", 8)
    def image_upload_toggle(self):
        return self.config.get("enable_image_uploads", True)

    def get_huggingface_api_key(self):
        return self.config.get("huggingface_api", {}).get("api_key", None)
    def get_huggingface_model_path(self):
        return self.config.get("huggingface_api", {}).get("model_path", "/root/.cache/huggingface/hub")

    def get_discord_api_key(self):
        return self.config.get("discord", {}).get("api_key", None)
    def get_local_model_path(self):
        return self.config["huggingface"].get("local_model_path", None)

    def get_user_config(self, user_id):
        return self.config["users"].get(str(user_id), {})

    def set_user_config(self, user_id, user_config):
        self.config["users"][str(user_id)] = user_config
        with open(self.config_path, "w") as config_file:
            json.dump(self.config, config_file)

    def set_user_setting(self, user_id, setting_key, value):
        user_id = str(user_id)
        if user_id not in self.config["users"]:
            self.config["users"][user_id] = {}
        self.config["users"][user_id][setting_key] = value
        with open(self.config_path, "w") as config_file:
            json.dump(self.config, config_file)

    def get_user_setting(self, user_id, setting_key, default_value=None):
        user_id = str(user_id)
        return self.config["users"].get(user_id, {}).get(setting_key, default_value)

    def get_mysql_user(self):
        return self.config.get("mysql", {}).get("user", "diffusion")
    def get_mysql_password(self):
        return self.config.get("mysql", {}).get("password", "diffusion_pwd")
    def get_mysql_hostname(self):
        return self.config.get("mysql", {}).get("hostname", "localhost")
    def get_mysql_dbname(self):
        return self.config.get("mysql", {}).get("dbname", "diffusion_master")
    
    def is_llama_enabled(self):
        return self.config.get("enable_llama", False)
    def llama_subsystem_type(self):
        return self.config.get("llama_subsystem", "llama.cpp")
    def llama_model_path(self):
        return self.config.get("llama_model_path", "/models/LLaMA")
    def llama_model_default(self):
        return self.config.get("llama_model_default", "7B")
    def llama_model_filename(self):
        return self.config.get("llama_model_filename", "ggml-model-f16.bin")