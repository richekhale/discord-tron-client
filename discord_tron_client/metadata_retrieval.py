from huggingface_hub import get_hf_file_metadata, hf_hub_url, model_info

repo_id = "ptx0/terminus-xl-gamma-training"
url = hf_hub_url(
    repo_id=repo_id,
    filename="unet/diffusion_pytorch_model.safetensors"
)
print(f"URL: {url}\n")
metadata = get_hf_file_metadata(url)
model_info = model_info(repo_id)
print(f"\n -> model_info: {model_info} \n")
# Split the last_modified into two sections split by + and use the first
last_modified = str(model_info.last_modified).split("+")[0]
print(f"Model last updated date: {last_modified}")
print(f"Commit hash: {metadata.commit_hash}")
print(f"Size: {metadata.size}")