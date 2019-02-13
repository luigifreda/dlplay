# ******************************************************************************
# This file is part of dlplay
# 
# Copyright (C) Luigi Freda <luigi dot freda at gmail dot com>
# 
# dlplay is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# dlplay is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with dlplay. If not, see <http://www.gnu.org/licenses/>.
# ******************************************************************************
from datasets import load_dataset
from transformers import AutoTokenizer

from datasets import load_dataset
from transformers import AutoTokenizer

from transformers import AutoModelForSequenceClassification

import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer

from dlplay.paths import DATA_DIR, RESULTS_DIR


# from https://huggingface.co/docs/transformers/training#train-with-pytorch-trainer
if __name__ == "__main__":

    share_to_hub = False
    if share_to_hub:
        from huggingface_hub import login

        login()

    # Fine-tuning adapts a pretrained model to a specific task with a smaller specialized dataset.
    # This approach requires far less data and compute compared to training a model from scratch,
    # which makes it a more accessible option for many users.

    # Transformers provides the Trainer API, which offers a comprehensive set of training features,
    # for fine-tuning any of the models on the Hub.

    # This guide will show you how to fine-tune a model with Trainer to classify Yelp reviews.

    # Start by loading the Yelp Reviews dataset and preprocess (tokenize, pad, and truncate) it for training. Use map to preprocess the entire dataset in one step.

    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset = dataset.map(tokenize, batched=True)

    # Fine-tune on a smaller subset of the full dataset to reduce the time it takes.
    # The results won’t be as good compared to fine-tuning on the full dataset,
    # but it is useful to make sure everything works as expected first before committing to training on the full dataset.

    small_train = dataset["train"].shuffle(seed=42).select(range(1000))
    small_eval = dataset["test"].shuffle(seed=42).select(range(1000))

    # Trainer is an optimized training loop for Transformers models, making it easy to start training
    # right away without manually writing your own training code. Pick and choose from a wide range
    # of training features in TrainingArguments such as gradient accumulation, mixed precision,
    # and options for reporting and logging training metrics.

    # Load a model and provide the number of expected labels (you can find this information
    # on the Yelp Review dataset card).

    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-cased", num_labels=5
    )
    print(f"model: {model}")

    # Some weights of BertForSequenceClassification were not initialized from the model checkpoint
    # at google-bert/bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
    # You should probably TRAIN this model on a down-stream task to be able to use it
    # for predictions and inference.

    # With the model loaded, set up your training hyperparameters in TrainingArguments.
    # Hyperparameters are variables that control the training process - such as the learning rate,
    # batch size, number of epochs - which in turn impacts model performance.
    # Selecting the correct hyperparameters is important and you should experiment with them
    # to find the best configuration for your task.

    # For this guide, you can use the default hyperparameters which provide a good baseline
    # to begin with. The only settings to configure in this guide are where to save the checkpoint,
    # how to evaluate model performance during training, and pushing the model to the Hub.

    # Trainer requires a function to compute and report your metric. For a classification task,
    # you’ll use evaluate.load to load the accuracy function from the Evaluate library.
    # Gather the predictions and labels in compute to calculate the accuracy.

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # convert the logits to their predicted class
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Set up TrainingArguments with where to save the model and when to compute accuracy during training. The example below sets it to "epoch", which reports the accuracy at the end of each epoch. Add push_to_hub=True to upload the model to the Hub after training.

    training_args = TrainingArguments(
        output_dir=f"{RESULTS_DIR}/saved_models/yelp_review_classifier",
        eval_strategy="epoch",
        push_to_hub=share_to_hub,
    )

    # Create a Trainer instance and pass it the model, training arguments, training and test datasets, and evaluation function. Call train() to start training.

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()

    if share_to_hub:
        trainer.push_to_hub()
