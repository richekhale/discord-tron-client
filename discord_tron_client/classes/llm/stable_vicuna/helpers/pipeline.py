import logging, os
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
    return max_sys_tokens

def bot(tokenizer, history, max_context_length, max_sys_tokens, system_prompt_tokens, model):
    history = history or []

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

    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=1.0,
        temperature=1.0,
    )
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        # Process out the prompt separator
        new_text = new_text.replace("<br>", "\n")
        if "###" in new_text:
            new_text = new_text.split("###")[0]
            partial_text += new_text.strip()
            history[-1][1] = partial_text
            break
        else:
            # Filter empty trailing new lines
            if new_text == "\n":
                new_text = new_text.strip()
            partial_text += new_text
            history[-1][1] = partial_text
        yield history
    return partial_text


def user(user_message, history):
    return "", history + [[user_message, None]]