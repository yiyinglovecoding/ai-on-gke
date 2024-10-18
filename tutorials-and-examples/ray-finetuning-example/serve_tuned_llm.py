# ray job submit  --working-dir . --runtime-env-json='{"pip": ["IPython", "peft", "accelerate>=0.16.0", "transformers>=4.26.0", "torch", "google-cloud-storage"]}' --address http://localhost:8265 -- python serve_llm.py

import ray 
import pandas as pd
from ray import serve
from starlette.requests import Request
import os

model_id = "google/gemma-7b"
revision = "float16"
access_token = "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"
bucket_name = "ray-llm-bucket"
# gcs_dir = "yiyingzhang/gemma-7b-test/TorchTrainer_2024-10-14_11-34-30/TorchTrainer_f750a_00000_0_2024-10-14_11-34-37/checkpoint_000002/checkpoint/"
gcs_dir = "gemma-7b-it-zyy-test/TorchTrainer_2024-10-10_15-52-20/TorchTrainer_52635_00000_0_2024-10-10_15-52-27/checkpoint_000010/checkpoint/"
local_dir = "./zyy/gemma-7b"

ray.init(
    address="ray://fine-tune-gemma-ray-cluster-head-svc:10001",
    runtime_env={
        "pip": [
            "IPython",
            "peft",
            "accelerate>=0.16.0",
            "transformers>=4.26.0",
            "torch",
            "google-cloud-storage",
        ]
    })

serve.shutdown()
serve.start(detached=False, http_options={'host':"0.0.0.0"})

@serve.deployment(ray_actor_options={"num_gpus": 1})
class LLMDeployment:
    def __init__(self):
        from google.cloud import storage

        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
            print(f"creating local dir {local_dir}")

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = list(storage_client.list_blobs(bucket_name, prefix=gcs_dir))
        print(f"len(blobs) = {len(blobs)}")
        for blob in blobs:
            filename = blob.name.replace(gcs_dir, "")
            if filename == "":
                continue
            blob = bucket.blob(blob.name)
            destination_file_name = local_dir + '/' + filename
            blob.download_to_filename(destination_file_name)
                
            print(
                "Downloaded storage object {} from bucket {} to local file {}.".format(
                    blob.name, bucket_name, destination_file_name
                )
            )
        
        def list_files(directory):
            """Lists all files and directories in the specified directory."""
            try:
                items = os.listdir(directory)
                for item in items:
                    print(item)
            except FileNotFoundError:
                print(f"Directory not found: {directory}")

        list_files(local_dir)

        from transformers import AutoTokenizer
        from peft import AutoPeftModelForCausalLM
        import torch

        self.model = AutoPeftModelForCausalLM.from_pretrained(
            local_dir,
            local_files_only=True,
            # trust_remote_code=True,
            # revision=revision,
            # torch_dtype=torch.float16,
            device_map="auto",  # automatically makes use of all GPUs available to the Actor
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_dir,
            local_files_only=True,
            # trust_remote_code=True,
            # model_max_length=256,
        )

    def generate(self, text: str) -> pd.DataFrame:
        print(f"text in generate is {text}")
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(
            self.model.device)

        gen_tokens = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=512,
        )
        return pd.DataFrame(
            self.tokenizer.batch_decode(gen_tokens), columns=["responses"]
        )

    async def __call__(self, http_request: Request) -> str:
        json_request: str = await http_request.json()
        prompts = []
        for prompt in json_request:
            text = prompt["text"]
            if isinstance(text, list):
                prompts.extend(text)
            else:
                prompts.append(text)
        return self.generate(prompts)
    
deployment = LLMDeployment.bind()
serve.run(deployment)

import requests

prompt = {
    "instruction": "Can you tell me something about GKE?",
    "context": "I want to deploy a LLM on GKE.",
}

def format_dolly_inference(sample):
    instruction = f"### Instruction\n{sample['instruction']}"
    context = f"### Context\n{sample['context']}" if "context" in sample else None
    response = f"### Answer\n"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
    return prompt

text =  format_dolly_inference(prompt)

print(f"text is \n{text}")

sample_input = {"text" : text}
output = requests.post("http://fine-tune-gemma-ray-cluster-head-svc:8000/", json=[sample_input]).json()
print(output)
