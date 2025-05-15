# src/fine_tune.py

import argparse
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    AdapterTrainer,
    LoRAConfig,
    BertConfig,
)
from datasets import load_dataset

def main(args):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("glue", "sst2")
    encoded = dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=128), batched=True)
    encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    if args.method == 'full':
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    elif args.method == 'adapters':
        from transformers.adapters import AdapterConfig, AdapterType
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        config = AdapterConfig.load("houlsby")
        model.add_adapter("sst2_adapter", config=config)
        model.train_adapter("sst2_adapter")

    elif args.method == 'lora':
        from peft import get_peft_model, LoraConfig, TaskType
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )
        model = get_peft_model(model, lora_config)

    else:
        raise ValueError(f"Método {args.method} no reconocido")

    training_args = TrainingArguments(
        output_dir=f"./results_{args.method}",
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="no",
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_dir='./logs'
    )

    trainer_class = AdapterTrainer if args.method == 'adapters' else Trainer

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["full", "adapters", "lora"], required=True)
    parser.add_argument("--rank", type=int, default=4, help="Rank de LoRA (solo aplica si method=lora)")
    args = parser.parse_args()
    main(args)
