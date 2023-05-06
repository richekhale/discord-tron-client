from transformers import AutoModelForCausalLM, AutoTokenizer
from discord_tron_client.classes.app_config import AppConfig

config = AppConfig()


class TransformerModelManager:
    def __init__(self):
        self.models = {}
        self.model_path = config.get_huggingface_model_path()

    def get_model(self, model_id):
        if model_id not in self.models:
            self.models[model_id] = AutoModelForCausalLM.from_pretrained(model_id)
        return self.models[model_id]

    def get_tokenizer(self, model_id):
        return AutoTokenizer.from_pretrained(model_id)
