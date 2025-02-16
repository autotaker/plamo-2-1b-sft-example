"""
train.py.sampleを参考に、create_dataset.pyで作成したデータセットの学習と検証のコードを書いて
"""

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import datasets
import json
import os
from trl import DataCollatorForCompletionOnlyLM
from trl import SFTConfig, SFTTrainer

def main():

    # Load model and tokenizer
    model_name = "pfnet/plamo-2-1b"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(
        "cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load datasets
    train_data_path = os.path.join(os.path.dirname(__file__), "dataset.train.jsonl")
    val_data_path = os.path.join(os.path.dirname(__file__), "dataset.validate.jsonl")


    def load_jsonl(path):
        data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data


    train_data = load_jsonl(train_data_path)
    val_data = load_jsonl(val_data_path)

    train_dataset = datasets.Dataset.from_list(train_data)
    val_dataset = datasets.Dataset.from_list(val_data)

    # Setup data collator
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=tokenizer.encode(" Answer:\n", add_special_tokens=False),
        tokenizer=tokenizer,
    )

    # Training configuration
    sft_args = SFTConfig(
        output_dir="./outputs",
        evaluation_strategy="steps",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.3,
        logging_steps=10,
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        report_to="wandb",  # Change to wandb
        bf16=True,
        max_seq_length=1024,
        gradient_checkpointing=True,
        deepspeed="./deepspeed_config.json",
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
