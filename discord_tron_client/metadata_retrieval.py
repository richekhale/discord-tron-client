from huggingface_hub import get_hf_file_metadata, hf_hub_url

url = hf_hub_url(
    repo_id="ptx0/terminus-xl-gamma-training",
    filename="unet/diffusion_pytorch_model.safetensors"
)
print(f"URL: {url}\n")
metadata = get_hf_file_metadata(url)
print(f"Commit hash: {metadata.commit_hash}")
print(f"Size: {metadata.size}")