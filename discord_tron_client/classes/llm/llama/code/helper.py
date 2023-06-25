from typing import List, NamedTuple

import torch
import transformers
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import GenerationConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "jordiclive/gpt4all-alpaca-oa-codealpaca-lora-7b"
)

llm_model_name = "decapoda-research/llama-7b-hf"
model = transformers.AutoModelForCausalLM.from_pretrained(
    llm_model_name, torch_dtype=torch.float16
)  # Load Base Model
model.resize_token_embeddings(
    len(tokenizer)
)  # This model repo also contains several embeddings for special tokens that need to be loaded.

model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

lora_weights = "jordiclive/gpt4all-alpaca-oa-codealpaca-lora-7b"
model = PeftModel.from_pretrained(
    model,
    lora_weights,
    torch_dtype=torch.float16,
)  # Load Lora model

model.eos_token_id = tokenizer.eos_token_id
filename = hf_hub_download(
    "jordiclive/gpt4all-alpaca-oa-codealpaca-lora-7b", "extra_embeddings.pt"
)
embed_weights = torch.load(
    filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)  # Load embeddings for special tokens
model.base_model.model.model.embed_tokens.weight[32000:, :] = embed_weights.to(
    model.base_model.model.model.embed_tokens.weight.dtype
).to(
    device
)  # Add special token embeddings


model = model.half().to(device)
generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
)
logging.info(f"temperature; {generation_config.temperature}")


def format_system_prompt(prompt, eos_token="</s>"):
    return "{}{}{}{}".format("<|prompter|>", prompt, eos_token, "<|assistant|>")


def generate(
    prompt, generation_config=generation_config, max_new_tokens=2048, device=device
):
    prompt = format_system_prompt(prompt)  # OpenAssistant Prompt Format expected
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            eos_token_id=2,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    logging.info("Text generated:")
    logging.info(output)
    return output
