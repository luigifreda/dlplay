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
import numpy as np
import torch
import datasets
from transformers import AutoImageProcessor, DefaultDataCollator
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
import evaluate

from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

from dlplay.paths import DATA_DIR, RESULTS_DIR

# from https://huggingface.co/docs/transformers/tasks/image_classification
#      https://huggingface.co/tasks/image-classification
if __name__ == "__main__":

    # Image classification assigns a label or class to an image.
    # Unlike text or audio classification, the inputs are the pixel values that comprise an image.
    # There are many applications for image classification, such as detecting damage after a natural disaster, monitoring crop health,
    # or helping screen medical images for signs of disease.

    # This guide illustrates how to:
    # Fine-tune ViT on the Food-101 dataset to classify a food item in an image.

    # NOTE: to share the model, you need to be signed in to Hugging Face.
    share_model = False
    if share_model:
        from huggingface_hub import notebook_login

        notebook_login()

    # Start by loading a smaller subset of the Food-101 dataset from the 🤗 Datasets library.
    # This will give you a chance to experiment and make sure everything works before spending more time training on the full dataset.
    food = datasets.load_dataset("food101", split="train[:5000]")
    # NOTE: split="train[:5000]" - This specifies which portion of the dataset to load:
    # "train" refers to the training split of the dataset
    # [:5000] is a slice notation that means "take only the first 5000 samples from the training set"

    # Split the dataset’s train split into a train and test set with the train_test_split method:
    food = food.train_test_split(test_size=0.2)

    # Then take a look at an example:
    print(food["train"][0])
    # {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x7F52AFC8AC50>, 'label': 79}

    # Each example in the dataset has two fields:
    # image: a PIL image of the food item
    # label: the label class of the food item

    # To make it easier for the model to get the label name from the label id, create a dictionary
    # that maps the label name to an integer and vice versa:
    labels = food["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Now you can convert the label id to a label name:
    print(id2label[str(79)])
    # 'prime_rib'

    # Preprocess
    # The next step is to load a ViT image processor to process the image into a tensor:
    checkpoint = "google/vit-base-patch16-224-in21k"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)

    # Apply some image transformations to the images to make the model more robust against overfitting.
    # Here you’ll use torchvision’s transforms module, but you can also use any image library you like.

    # Prepare the preprocessor.
    # Crop a random part of the image, resize it, and normalize it with the image mean and standard deviation:
    normalize = Normalize(
        mean=image_processor.image_mean, std=image_processor.image_std
    )
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    # Then create a preprocessing function to apply the transforms and return the pixel_values - the inputs to the model - of the image:
    def transforms(examples):
        examples["pixel_values"] = [
            _transforms(img.convert("RGB")) for img in examples["image"]
        ]
        del examples["image"]
        return examples

    # To apply the preprocessing function over the entire dataset, use 🤗 Datasets with_transform method.
    # The transforms are applied on the fly when you load an element of the dataset:
    food = food.with_transform(transforms)

    # Now create a batch of examples using DefaultDataCollator. Unlike other data collators in 🤗 Transformers,
    # the DefaultDataCollator does not apply additional preprocessing such as padding.
    data_collator = DefaultDataCollator()

    # Evaluate
    # Including a metric during training is often helpful for evaluating your model’s performance. You can quickly load an evaluation method with the 🤗 Evaluate library. For this task, load the accuracy metric (see the 🤗 Evaluate quick tour to learn more about how to load and compute a metric):
    accuracy = evaluate.load("accuracy")

    # Then create a function that passes your predictions and labels to compute to calculate the accuracy.
    # This will be used at training time
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    # Train
    # You’re ready to start training your model now! Load ViT with AutoModelForImageClassification.
    # This is a generic model class that will be instantiated as one of the model classes of the library
    # (with a image classification head) when created with the from_pretrained() class method or the from_config() class method.
    # Specify the number of labels along with the number of expected labels, and the label mappings:
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # At this point, only three steps remain:
    # 1. Define your training hyperparameters in TrainingArguments. It is important you don’t remove unused columns
    #    because that’ll drop the image column. Without the image column, you can’t create pixel_values.
    #    Set remove_unused_columns=False to prevent this behavior! The only other required parameter is
    #    output_dir which specifies where to save your model. You’ll push this model to the Hub by setting
    #    push_to_hub=True (you need to be signed in to Hugging Face to upload your model).
    #    At the end of each epoch, the Trainer will evaluate the accuracy and save the training checkpoint.
    # 2. Pass the training arguments to Trainer along with the model, dataset, tokenizer, data collator,
    #    and compute_metrics function.
    # 3. Call train() to finetune your model.

    training_args = TrainingArguments(
        output_dir=f"{RESULTS_DIR}/saved_models/my_awesome_food_model",  # Where to save model checkpoints and logs
        remove_unused_columns=False,  # Keep all dataset columns (useful for custom datasets)
        eval_strategy="epoch",  # Evaluate model after each epoch
        save_strategy="epoch",  # Save model checkpoint after each epoch
        learning_rate=5e-5,  # Learning rate for optimizer (5 × 10⁻⁵)
        per_device_train_batch_size=16,  # Batch size per GPU/device for training
        gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps (effective batch size = 16×4=64)
        per_device_eval_batch_size=16,  # Batch size per GPU/device for evaluation
        num_train_epochs=3,  # Total number of training epochs
        warmup_ratio=0.1,  # Warm up learning rate for first 10% of training steps
        logging_steps=10,  # Log training info every 10 steps
        load_best_model_at_end=True,  # Load the best model (based on metric) at end of training
        metric_for_best_model="accuracy",  # Use accuracy to determine the best model
        push_to_hub=share_model,  # Whether to push final model to Hugging Face Hub
    )
    # Key Benefits of This Configuration:
    # Effective Batch Size: 16 × 4 = 64 (good for stable training)
    # Regular Checkpoints: Saves progress every epoch
    # Best Model Selection: Automatically loads the most accurate model
    # Learning Rate Warmup: Prevents early training instability
    # Frequent Logging: Monitor training progress every 10 steps

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=food["train"],
        eval_dataset=food["test"],
        processing_class=image_processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if share_model:
        # Push the model to the Hub
        trainer.push_to_hub()

    # save the model to the local directory
    # trainer.save_model(f"{DATA_DIR}/my_awesome_food_model/final_model")
