import torch, re
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList



class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

def generate(tokenizer, model, user_prompt, user_config, max_tokens = 64, temperature = 0.7, repeat_penalty = 1.1, top_p = 0.9, top_k = 40):
    prompt = f"{system_prompt}<|USER|>{user_prompt}<|ASSISTANT|>"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    tokens = model.generate(
    **inputs,
    max_new_tokens=max_tokens,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    do_sample=True,
    stopping_criteria=StoppingCriteriaList([StopOnTokens()])
    )
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return clean_output(output, user_prompt)
    
def clean_output(output: str, prompt: str):
    # Remove "prompt" and any preceeding text from "output":
    output = output.replace(prompt, "")
    output = re.sub(r"(<\|USER\|>)(.*)(<\|ASSISTANT\|>)", "", output)
    output = re.sub(r"(<\|SYSTEM\|>)(.*)(<\|USER\|>)", "", output)
    return output    

def load(model_name = '7b'):
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-tuned-alpha-" + str(model_name))
    model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-tuned-alpha-" + str(model_name))
    model.half().cuda()
    return tokenizer, model