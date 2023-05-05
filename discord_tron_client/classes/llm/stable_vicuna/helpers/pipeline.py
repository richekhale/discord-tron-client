import logging, os
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
from string import Template
from threading import Thread

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding, TextIteratorStreamer
from discord_tron_client.classes.app_config import AppConfig

config = AppConfig()
default_model_id = config.stablevicuna_model_default()
def load(model_id = default_model_id):
    auth_token = config.get_huggingface_api_key()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=auth_token,
    )

    offload_folder = os.path.join(config.get_huggingface_model_path(), "offload")
    # We need to create if it doesn't exist:
    if not os.path.exists(offload_folder):
        os.makedirs(offload_folder)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_auth_token=auth_token,
        offload_folder=offload_folder,
    )

    model.eval()
    model.to("cuda")
    max_context_length = model.config.max_position_embeddings
    logging.debug(f"Model context length max: {max_context_length}")
    return tokenizer, model, max_context_length

max_new_tokens = 768
prompt_template = Template("""\
### Human: $human
### Assistant: $bot\
""")

def get_system_prompt(tokenizer):
    system_prompt = "### Assistant: I am StableVicuna, a large language model created by CarperAI. I am here to chat!"
    system_prompt_tokens = tokenizer([f"{system_prompt}\n\n"], return_tensors="pt")
    max_sys_tokens = system_prompt_tokens['input_ids'].size(-1)
    return max_sys_tokens, system_prompt_tokens

def chatml_convert(conversation: list, new_message: str = None):
    """
    Convert the conversation to the desired format.

    Args:
        conversation (list): A list of conversation dictionaries.

    Returns:
        chatml_history (list): A list of formatted conversation strings.
    """
    chatml_history = []
    start_role = conversation[0]["role"]

    for i in range(0, len(conversation), 2):
        if start_role == "user":
            human = conversation[i]["content"]
            if i + 1 < len(conversation):
                bot = conversation[i + 1]["content"]
            else:
                bot = ""
        else:
            bot = conversation[i]["content"]
            if i + 1 < len(conversation):
                human = conversation[i + 1]["content"]
            else:
                human = ""
                
        chatml_history.append(prompt_template.substitute(human=human, bot=bot))
    if new_message is not None:
        chatml_history.append(prompt_template.substitute(human=new_message, bot=""))

    # Lists don't have .join, but:
    return '\n'.join(chatml_history)

def generate(tokenizer, model, user_config, max_tokens=64, temperature=0.7, repeat_penalty=1.1, top_p=0.9, top_k=40, seed=None, prompt = None, user_prompt = None):
    if prompt is None and user_prompt is None:
        raise ValueError("User prompt and raw prompt cannot both be null.")
    if prompt is not None:
        if user_prompt is None:
            user_prompt = "What is the first thing you would like to say to StableVicuna?"
        prompt = chatml_convert(prompt, user_prompt)
    else:      
        prompt = prompt_template.substitute(human=user_prompt, bot="")

    logging.info(f"Generating with user config: {user_config}, user_prompt: {user_prompt}")
    logging.debug(f"Generated prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    if inputs.get("token_type_ids", None) is not None:
        inputs.pop("token_type_ids")
    logging.debug(f"Inputs are {inputs} after CUDA jazz")
    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )
    logging.debug(f"Misc parameters: top_k:{top_k} top_p:{top_p} temperature:{temperature} max_tokens:{max_tokens} repeat_penalty:{repeat_penalty}")
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=user_config.get("max_tokens", max_tokens),
        temperature=user_config.get("temperature", temperature),
        top_p=user_config.get("top_p", top_p),
        top_k=user_config.get("top_k", top_k),
        do_sample=True,
    )
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        new_text = new_text.replace("<br>", "\n")
        generated_text += new_text
    return clean_output(generated_text), len(inputs['input_ids'][0])

def clean_output(text):
    import re
    expr= r"^### Assistant: (.*)"
    matches = re.findall(expr, text, re.MULTILINE)
    if len(matches) > 0:
        return matches[-1]
    return text