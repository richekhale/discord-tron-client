import torch, re, logging
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

system_prompt = "<|SYSTEM|>"
more_system_prompt = """
# StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""
system_prompt = str(system_prompt) + str(more_system_prompt)

def generate(tokenizer, model, user_prompt, user_config, max_tokens = 64, temperature = 0.7, repeat_penalty = 1.1, top_p = 0.9, top_k = 40):
    prompt = f"{system_prompt}<|USER|>{user_prompt}<|ASSISTANT|>"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    tokens = model.generate(
        **inputs,
        max_new_tokens=user_config.get("max_tokens", max_tokens),
        temperature=user_config.get("temperature", temperature),
        top_p=user_config.get("top_p", top_p),
        top_k=user_config.get("top_k", top_k),
        do_sample=True,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()])
    )
    output = tokenizer.decode(tokens[0], skip_special_tokens=False)
    return clean_output(output), len(tokens)
    
def clean_output(output: str):
    # Remove "prompt" and any preceeding text from "output":
    beginning_token = "<\|ASSISTANT\|>"
    end_token = "<\|endoftext\|>"
    # Retrieve everything BETWEEN (non-inclusive) beginning and end tokens:
    search = re.search(f"{beginning_token}(.*){end_token}", output, flags=re.DOTALL)
    if search is not None and hasattr(search, "group"):
        output = search.group(1)
    print(f"Output: {output}")
    logging.debug(f"Search result: {search}")
    return output

def load(model_name = '7b'):
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-tuned-alpha-" + str(model_name))
    model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-tuned-alpha-" + str(model_name))
    model.half().cuda()
    return tokenizer, model