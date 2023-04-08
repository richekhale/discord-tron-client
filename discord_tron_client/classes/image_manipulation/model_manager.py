from transformers import AutoModelForCausalLM, AutoTokenizer

class TransformerModelManager:
    def __init__(self):
        self.models = {}

    def get_model(self, model_id):
        if model_id not in self.models:
            self.models[model_id] = AutoModelForCausalLM.from_pretrained(model_id)
        return self.models[model_id]

    def get_tokenizer(self, model_id):
        return AutoTokenizer.from_pretrained(model_id)