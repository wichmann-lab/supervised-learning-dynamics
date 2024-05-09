import numpy as np
from PIL import Image as PILImage
import os

from datasets import Dataset, DatasetDict, Features, ClassLabel, Image, Value
import evaluate
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer, IntervalStrategy
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

# Define a customized trainer class to logg all predictions and to evaluate on multiple test sets
class CustomTrainer(Trainer):
    def __init__(self, run_id, eval_datasets=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_id = run_id
        self.epoch_predictions = []
        self.test_paths = [os.path.join(cwd, 'dataset', 'test', f'testset_{i+1}') for i in range(6)]
        self.eval_datasets = eval_datasets
        self.current_epoch = 1


    def compute_loss(self, model, inputs, return_outputs=False):

        # Check if we need to increment the epoch based on the progress of training steps
        step_in_epoch = (self.state.global_step - 1) % self.args.eval_steps + 1
        if step_in_epoch == 1 and self.state.global_step > 1:
            self.current_epoch += 1

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
                'epoch': self.current_epoch,
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

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Override the default evaluation dataset every time evaluate is called
        if eval_dataset is None:
            epoch = int(self.state.epoch)  # Assuming self.state.epoch starts at 0
            dataset_key = f'test_{epoch}'  # Fetch the dataset for the current epoch
            eval_dataset = self.eval_datasets.get(dataset_key)
            if eval_dataset is None:
                raise ValueError(f"No dataset found for key {dataset_key}")

        return super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )

#set up a dict with 20 models freshly initialized because of strange behavior when not doing so

models = {}
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
for run in range(1, 21):
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )
    models[run] = model


# Train the model for 20 runs
for run in range(1,21):

    output_dir = os.path.join(cwd, 'vit', f'run_{run}')
    os.makedirs(output_dir, exist_ok=True)

    # Use the pre-initialized model
    model = models[run]

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
        eval_datasets={
        'test_1': prepared_dataset['test_1'],
        'test_2': prepared_dataset['test_2'],
        'test_3': prepared_dataset['test_3'],
        'test_4': prepared_dataset['test_4'],
        'test_5': prepared_dataset['test_5'],
        'test_6': prepared_dataset['test_6'],
    },
        tokenizer=processor,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
