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

def bot(tokenizer, history, max_context_length, model):
    history = history or []
    max_sys_tokens, system_prompt_tokens = get_system_prompt(tokenizer)
    # Inject prompt formatting into the history
    prompt_history = []
    for human, bot in history:
        if bot is not None:
            bot = bot.replace("<br>", "\n")
            bot = bot.rstrip()
        prompt_history.append(
            prompt_template.substitute(
                human=human, bot=bot if bot is not None else "")
        )

    msg_tokens = tokenizer(
        "\n\n".join(prompt_history).strip(),
        return_tensors="pt",
        add_special_tokens=False  # Use <BOS> from the system prompt
    )

    # Take only the most recent context up to the max context length and prepend the
    # system prompt with the messages
    max_tokens = -max_context_length + max_new_tokens + max_sys_tokens
    inputs = BatchEncoding({
        k: torch.concat([system_prompt_tokens[k], msg_tokens[k][:, max_tokens:]], dim=-1)
        for k in msg_tokens
    }).to('cuda')
    # Remove `token_type_ids` b/c it's not yet supported for LLaMA `transformers` models
    if inputs.get("token_type_ids", None) is not None:
        inputs.pop("token_type_ids")
    text = model.generate(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=1.0,
        temperature=1.0
    )
    return text


def user(user_message, history):
    return "", history + [[user_message, None]]

def chatml_convert(conversation: list):
    """
    Convert the conversation to the desired format.

    Args:
        conversation (list): A list of conversation dictionaries.

    Returns:
        chatml_history (str): Conversation string.
    """
    stablevicuna_history = []
    for i in range(len(conversation)):
        if i % 2 == 0:
            human = conversation[i]["content"]
            if i + 1 < len(conversation):
                bot = conversation[i + 1]["content"]
            else:
                bot = ""
            stablevicuna_history.append(prompt_template.substitute(human=human, bot=bot))
    return stablevicuna_history.join("\n")


def generate(tokenizer, model, user_prompt, user_config, max_tokens=64, temperature=0.7, repeat_penalty=1.1, top_p=0.9, top_k=40, seed=None):
    logging.info(f"Generating with user config: {user_config}, user_prompt: {user_prompt}")
    system_prompt = "### Assistant: I am StableVicuna, a large language model created by CarperAI. I am here to chat!"
    prompt = prompt_template.substitute(human=user_prompt, bot="")
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