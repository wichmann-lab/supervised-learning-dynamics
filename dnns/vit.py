import numpy as np
from PIL import Image as PILImage
import os

from datasets import Dataset, DatasetDict, Features, ClassLabel, Image, Value
import evaluate
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from datasets import load_metric
import torch

import helper as h

cwd = os.getcwd()

prepared_dataset = h.load_data()

# Define the collate function
def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    filenames = [item['filename'] for item in batch]

    return {
        'pixel_values': pixel_values,
        'labels': labels,
        'filenames': filenames
    }

# Load the metric
metric = load_metric("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

# Access the label names
labels = prepared_dataset['train'].features['label'].names

# Define a customized trainer class to logg all predictions
class CustomTrainer(Trainer):
    def __init__(self, run_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_id = run_id
        self.epoch_predictions = []
        self.test_paths = [os.path.join(cwd, 'dataset', 'test', f'testset_{i+1}') for i in range(6)]

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(pixel_values=inputs['pixel_values'], labels=inputs['labels'])
        #outputs = model(**inputs)
        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        labels = inputs['labels']
        filenames = inputs['filenames']

        # Log each prediction
        for i, pred in enumerate(predictions):
            prediction_log = {
                'run': self.run_id,
                'epoch': int(self.state.epoch)+1,
                'phase': 'train' if self.model.training else 'test',
                'image_name': filenames[i],
                'ground_truth': labels[i].item(),
                'prediction': pred.item()
            }
            self.log_prediction(prediction_log)

        return (loss, outputs) if return_outputs else loss

    def log_prediction(self, prediction_log):
        import csv
        filename = f"predictions_run_{self.run_id}.csv"
        filepath = os.path.join(self.args.output_dir, filename)
        with open(filepath, 'a', newline='') as csvfile:
            fieldnames = ['run', 'epoch', 'phase', 'image_name', 'ground_truth', 'prediction']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow(prediction_log)

    def evaluate(self, epoch):
        # Get the test dataset for the current epoch
        test_dataset_name = f"test_{epoch+1}"
        test_dataset = self.args.eval_dataset[test_dataset_name]

        # Evaluate the model on the current test dataset
        eval_output = super().evaluate(test_dataset)
        return eval_output

# Train the model for 20 runs
for run in range(1,21):

    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    processor = ViTImageProcessor.from_pretrained(model_name_or_path)

    output_dir = os.path.join(cwd, 'vit', f'run_{run}')
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="steps",
        num_train_epochs=6,
        save_steps=9,
        eval_steps=9,
        logging_steps=9,
        learning_rate=0.0001,
        save_total_limit=6,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=False,
    )

    trainer = CustomTrainer(
        run_id=run,
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_dataset["train"],
        eval_dataset=prepared_dataset,
        tokenizer=processor,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
