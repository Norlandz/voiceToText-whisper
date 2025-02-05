# import os
# import torch
# # from transformers import WhisperProcessor, WhisperForConditionalGeneration
# # from huggingface_hub import hf_hub_ls
# # Load model directly
# from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
#
#
#
# # --------------------
# # Configuration (Change these if needed)
# # --------------------
#
# # Hugging Face model ID
# MODEL_ID = "openai/whisper-medium"
#
# # Optional: Mirror base URL (if needed)
# MIRROR_URL = "https://mirrors.tuna.tsinghua.edu.cn"  # Example: Tsinghua Mirror
# # Set to None to avoid using a mirror
# # MIRROR_URL = None # comment out the mirror if not needed
#
# # Output file location for the single .pt file
# OUTPUT_PT_FILE = "./whisper_medium.pt"  # Output to current folder
#
#
# # --------------------
# # Code
# # --------------------
# def download_model_with_mirror(model_id, mirror_url, output_pt_file):
#     """Downloads the model from huggingface hub with or without a mirror."""
#
#     # Setup Environment variable for a mirror if necessary
#     if mirror_url:
#         os.environ["HF_ENDPOINT"] = mirror_url
#         print(f"Using Mirror URL: {mirror_url}")
#
#     # Load model components
#     # processor = WhisperProcessor.from_pretrained(model_id)
#     # model = WhisperForConditionalGeneration.from_pretrained(model_id)
#     processor = AutoProcessor.from_pretrained("openai/whisper-medium")
#     model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-medium")
#
#     # Save single .pt file.
#     model.eval()  # Set model to eval mode.
#     torch.save(model.state_dict(), output_pt_file)
#     print(f"Model saved to: {output_pt_file}")
#
# if __name__ == "__main__":
#     download_model_with_mirror(MODEL_ID, MIRROR_URL, OUTPUT_PT_FILE)


from huggingface_hub import hf_hub_download

# repo_id = "lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF"
# filename = "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
repo_id = "lmstudio-community/Qwen2.5-7B-Instruct-1M-GGUF"
filename = "Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf"
# endpoint = "http://hf-mirror.com"

# Download the file
file_path = hf_hub_download(repo_id=repo_id, filename=filename)  # , endpoint=endpoint)
print(f"File downloaded to: {file_path}")
