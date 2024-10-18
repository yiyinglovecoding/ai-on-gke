import ray 

ray.init(
    address="ray://fine-tune-gemma-ray-cluster-head-svc:10001",
    runtime_env={
        "pip": [
            "IPython",
            "peft",
            "boto3==1.26",
            "botocore==1.29", 
            "datasets==2.16.1",
            "fastapi==0.96",
            "accelerate>=0.16.0",
            "transformers>=4.26.0",
            "bitsandbytes==0.43.1",
            "numpy<1.24",  # remove when mlflow updates beyond 2.2
            "torch",
        ]
    }
)

import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"expandable_segments:True"
model_id = "google/gemma-7b"
revision = "float16"
access_token = "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"
use_gpu = True
num_workers = 1
batch_size = 3
output_dir = "yiyingzhang/gemma-7b"

from ray import serve
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling

serve.start(detached=False, http_options={'host':"0.0.0.0"})

def train_func(config):
    torch.cuda.empty_cache()
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    print(f"torch version is: {torch.version.cuda}")
    
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from huggingface_hub import login
    
    datasets = load_dataset("databricks/databricks-dolly-15k", split='train').train_test_split(0.1)
    train_dataset = datasets['train']
    eval_dataset = datasets['test']

    login(token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=access_token, model_max_length=256)
    
    def format_dolly(sample):
        instruction = f"### Instruction\n{sample['instruction']}"
        context =  f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
        response = f"### Answer\n{sample['response']}"
        # join all the parts together
        prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
        return prompt

    # template dataset to add prompt to each sample
    def template_dataset(sample):
        sample["text"] = f"{format_dolly(sample)}{tokenizer.eos_token}"
        return sample

    train_dataset = train_dataset.map(template_dataset, remove_columns=list(train_dataset.features))
    eval_dataset = eval_dataset.map(template_dataset, remove_columns=list(eval_dataset.features))

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
        
    small_train_dataset = (
        train_dataset.select(range(100)).map(tokenize_function, batched=True)
    )
    small_eval_dataset = (
        eval_dataset.select(range(100)).map(tokenize_function, batched=True)
    )

    print(f"small_train_dataset[3]: {small_train_dataset[3]}")

	# create LoRA config
    lora_config = LoraConfig(
        r=6,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",  # automatically makes use of all GPUs available to the Actor
        quantization_config=quantization_config,
        token=access_token,
    )
        
    # add LoRA adapter layer to the base model
    # model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=batch_size,
        learning_rate=config.get("learning_rate", 2e-4),
        num_train_epochs=config.get("epochs", 1),
        weight_decay=config.get("weight_decay", 0.01),
        push_to_hub=False,
        max_grad_norm=0.3,   
        warmup_ratio=0.03,
        # max_steps=max_steps_per_epoch * config.get("epochs", 2),
        disable_tqdm=True,  # declutter the output a little
        no_cuda=not use_gpu,  # you need to explicitly set no_cuda if you want CPUs
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        data_collator=data_collator,
    )

    callback = ray.train.huggingface.transformers.RayTrainReportCallback()
    trainer.add_callback(callback)
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
    trainer.train()
    print(f"Saving model to: {output_dir}") 
    trainer.save_model()
    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    print("Model saved successfully!")
    print(os.listdir(output_dir))  # Verify the saved files
    torch.cuda.empty_cache()

import ray.train.huggingface.transformers
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig

scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)
ray_trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=RunConfig(
        storage_path="gs://ray-llm-bucket/yiyingzhang/gemma-7b",
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="eval_loss",
            checkpoint_score_order="min",
        ),
    ),
)
result = ray_trainer.fit()

# Print available checkpoints
for checkpoint, metrics in result.best_checkpoints:
    print("Loss", metrics["loss"], "checkpoint", checkpoint)


