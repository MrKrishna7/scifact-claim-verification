import os
import sys
import numpy as np
import torch
from typing import List, Dict
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback,
    Trainer,
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.data_utils import (
    load_corpus,
    load_claims,
    build_nli_pairs,
    LABEL2ID,
    ID2LABEL,
    print_label_distribution,
)

def build_dataset(pairs: List[Dict], tokenizer, max_length: int = 512) -> Dataset:

    class NLIDataset(Dataset):
        def __init__(self, pairs: List[Dict], tokenizer, max_length: int):
            self.pairs = pairs
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, idx: int) -> Dict:
            pair = self.pairs[idx]
            encoding = self.tokenizer(
                pair["premise"],
                pair["hypothesis"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )


            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": torch.tensor(pair["label"], dtype=torch.long)
            }

    return NLIDataset(pairs, tokenizer, max_length)

def compute_metrics(eval_pred) -> Dict:

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    macro_f1 = f1_score(labels, preds, average="macro")
    accuracy = accuracy_score(labels, preds)

    # per-class F1
    per_class = f1_score(labels, preds, average=None, labels=[0, 1, 2])
    contradict_f1 = float(per_class[0]) if len(per_class) > 0 else 0.0
    support_f1 = float(per_class[1]) if len(per_class) > 1 else 0.0
    nei_f1 = float(per_class[2]) if len(per_class) > 2 else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "contradict_f1": contradict_f1,
        "support_f1": support_f1,
        "nei_f1": nei_f1
    }

class UnweightedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def load_fresh_model(model_name: str) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
    )
    model.config.label2id = dict(LABEL2ID)
    model.config.id2label = dict(ID2LABEL)
    model.config.problem_type = "single_label_classification"
    return model

def make_training_args(save_dir: str, epochs: int, batch_size: int, lr: float) -> TrainingArguments:
    return TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none"
    )

def run_training(
    trainer_cls,
    model,
    training_args: TrainingArguments,
    train_dataset,
    dev_dataset,
    tokenizer,
    save_dir: str
) -> Dict:
    """
    Returns the final dev evaluation results dict.
    """
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"Starting training — Unweighted")
    trainer.train()

    # load_best_model_at_end swaps the model object,
    # so we must set config on trainer.model (not the original model variable).
    trainer.model.config.label2id = dict(LABEL2ID)
    trainer.model.config.id2label = dict(ID2LABEL)
    trainer.model.config.problem_type = "single_label_classification"

    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nUnweighted model saved to {save_dir}")
    print(f"\nFinal dev evaluation :")
    results = trainer.evaluate()
    for k, v in results.items():
            print(f"  {k}: {v:.4f}")

    return results

def train(max_length: int, epochs: int, batch_size: int, lr: float) -> Dict:

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    print("Loading corpus...")
    corpus = load_corpus(os.path.join(data_dir, "corpus.jsonl"))

    print("Loading claims...")
    train_claims = load_claims(os.path.join(data_dir, "claims_train.jsonl"))
    dev_claims = load_claims(os.path.join(data_dir, "claims_dev.jsonl"))

    print("Building NLI pairs...")
    train_pairs = build_nli_pairs(train_claims, corpus, use_gold_sentences=True)
    dev_pairs = build_nli_pairs(dev_claims, corpus, use_gold_sentences=True)

    print(f"\nTrain pairs : {len(train_pairs)}")
    print(f"Dev pairs   : {len(dev_pairs)}")
    print_label_distribution(train_pairs)

    model_name = "cross-encoder/nli-deberta-v3-base"
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = build_dataset(train_pairs, tokenizer, max_length=max_length)
    dev_dataset = build_dataset(dev_pairs, tokenizer, max_length=max_length)

    base_save = os.path.join(os.path.dirname(__file__), "saved")
    save_dir_unweighted = os.path.join(base_save, "verifier_unweighted")


    unweighted_results = run_training(
        trainer_cls=UnweightedTrainer,
        model=load_fresh_model(model_name),
        training_args=make_training_args(
            save_dir_unweighted,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr
        ),
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        tokenizer=tokenizer,
        save_dir=save_dir_unweighted
            )
    return unweighted_results

if __name__ == "__main__":
    epochs = 5
    batch_size = 16
    lr = 2e-5
    max_length = 512

    train(max_length=max_length, epochs=epochs, batch_size=batch_size, lr=lr)