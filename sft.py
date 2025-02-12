from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import datasets
import string
from trl import DataCollatorForCompletionOnlyLM
from trl import SFTConfig, SFTTrainer


INSTRUCTION_TEMPLATE = string.Template(
    """### Question:
${input}

### Answer:
${response}<|plamo:eos|>
"""
)


def formatting_func(examples):
    output_texts = []
    for i in range(len(examples['instruction'])):
        text = INSTRUCTION_TEMPLATE.substitute(input=examples['instruction'][i], response=examples['output'][i])
        output_texts.append(text)
    return output_texts


def main():
    model_name = "pfnet/plamo-2-1b"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    dataset = datasets.load_dataset("kunishou/databricks-dolly-15k-ja")
    train_dataset = dataset["train"].filter(lambda data: data["input"] == "").select(range(2000))

    print("Train dataset")
    print(train_dataset)
    print(train_dataset[0])

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=tokenizer.encode(" Answer:\n", add_special_tokens=False),
        tokenizer=tokenizer
    )


    sft_args = SFTConfig(
        output_dir="./outputs",
        evaluation_strategy="no",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        #num_train_epochs=1.0,
        num_train_epochs=0.1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.3,
        logging_steps=10,
        save_strategy="epoch",
        report_to="tensorboard",
        bf16=True,
        max_seq_length=1024,
        gradient_checkpointing=True,
        deepspeed='./deepspeed_config.json',
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()