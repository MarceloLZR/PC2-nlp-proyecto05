# src/pretrain.py

import argparse
from transformers import BertTokenizer, BertForPreTraining, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

def main(args):
    # Cargar dataset
    dataset = load_dataset('text', data_files={'train': args.dataset})['train']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenización con truncamiento
    def tokenize(example):
        return tokenizer(example['text'], truncation=True, padding='max_length', max_length=128)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=['text'])

    # Modelo de preentrenamiento con MLM + NSP
    model = BertForPreTraining.from_pretrained('bert-base-uncased')

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # Configuración de entrenamiento
    training_args = TrainingArguments(
        output_dir='./results_pretrain',
        evaluation_strategy="no",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        save_steps=5000,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Ruta al archivo de texto con Wikipedia")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    main(args)
