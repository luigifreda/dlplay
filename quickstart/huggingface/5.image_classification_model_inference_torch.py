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

from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

from dlplay.utils.conversions import to_numpy_uint_image
from dlplay.paths import DATA_DIR, RESULTS_DIR

import json
import cv2

# from https://huggingface.co/docs/transformers/tasks/image_classification
#      https://huggingface.co/tasks/image-classification
if __name__ == "__main__":

    # Image classification assigns a label or class to an image.
    # Unlike text or audio classification, the inputs are the pixel values that comprise an image.
    # There are many applications for image classification, such as detecting damage after a natural disaster, monitoring crop health,
    # or helping screen medical images for signs of disease.

    # This guide illustrates how to:
    # Use your fine-tuned model for inference. (run the previous script to train the model)

    # Great, now that you’ve fine-tuned a model, you can use it for inference!

    # Load an image you’d like to run inference on:
    # NOTE: here we use the validation set to test the model (for training, we used the train set)
    ds = datasets.load_dataset("food101", split="validation[:10]")
    # show image and label
    image = ds["image"][0]
    print(f"image: {image}")

    # show image
    # image.show()

    np_image = to_numpy_uint_image(image)
    np_rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    print(f"np_image: {np_image.shape}")
    cv2.imshow("image", np_rgb_image)

    saved_results_dir = f"{RESULTS_DIR}/saved_models/my_awesome_food_model"

    # The simplest way to try out your finetuned model for inference is to use it in a pipeline().
    # Instantiate a pipeline for image classification with your model, and pass your image to it:
    classifier = pipeline(
        "image-classification",
        model=f"{saved_results_dir}/checkpoint-189",
    )
    pipeline_results = classifier(image)
    json_results = json.dumps(pipeline_results, indent=4)
    print(f"pipeline_results: {json_results}")

    # You can also manually replicate the results of the pipeline if you’d like:

    # Load an image processor to preprocess the image and return the input as PyTorch tensors:
    image_processor = AutoImageProcessor.from_pretrained(
        f"{saved_results_dir}/checkpoint-189"
    )
    inputs = image_processor(image, return_tensors="pt")

    # Pass your inputs to the model and return the logits:
    model = AutoModelForImageClassification.from_pretrained(
        f"{saved_results_dir}/checkpoint-189"
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted label with the highest probability, and use the model’s id2label mapping to convert it to a label:
    predicted_label_id = logits.argmax(-1).item()
    print(f"predicted_label_id: {predicted_label_id}")
    predicted_label = model.config.id2label[predicted_label_id]
    print(f"predicted_label: {predicted_label}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
