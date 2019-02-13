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
from datasets import load_dataset, Audio
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

import torch
from torch.utils.data import DataLoader, Dataset


# https://huggingface.co/docs/datasets/en/quickstart
if __name__ == "__main__":

    # 1. Load the MInDS-14 dataset by providing the load_dataset() function with the dataset name,
    # dataset configuration (not all datasets will have a configuration), and a dataset split:
    dataset = load_dataset("PolyAI/minds14", "en-US", split="train")

    # 2. Next, load a pretrained Wav2Vec2 model and its corresponding feature extractor from the
    # 🤗 Transformers library.
    # It is totally normal to see a warning after you load the model about some weights not being initialized.
    # This is expected because you are loading this model checkpoint for training with another task.
    model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    # 3. The MInDS-14 dataset card indicates the sampling rate is 8kHz, but the Wav2Vec2 model was pretrained
    # on a sampling rate of 16kHZ. You’ll need to upsample the audio column with the cast_column() function
    # and Audio feature to match the model’s sampling rate.
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # 4. Create a function to preprocess the audio array with the feature extractor, and truncate and pad
    # the sequences into tidy rectangular tensors. The most important thing to remember is to call the
    # audio array in the feature extractor since the array - the actual speech signal - is the model input.
    # Use the map() function to speed up processing by applying the function to batches of examples in the dataset.
    def preprocess_function(data_examples):
        # audio_arrays = [x.get_all_samples().data for x in data_examples["audio"]]
        audio_arrays = [x["array"] for x in data_examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            padding=True,
            max_length=100000,
            truncation=True,
        )
        return inputs

    dataset = dataset.map(preprocess_function, batched=True)

    # 5. Use the rename_column() function to rename the intent_class column to labels,
    # which is the expected input name in Wav2Vec2ForSequenceClassification:
    dataset = dataset.rename_column("intent_class", "labels")

    # 6. Set the dataset format according to the machine learning framework you’re using.

    ## ==============================
    ## PyTorch
    ## ==============================

    dataset.set_format(type="torch", columns=["input_values", "labels"])
    dataloader = DataLoader(dataset, batch_size=4)

    # 7. Start training with your machine learning framework!
    # Check out the 🤗 Transformers audio classification guide for an end-to-end example of how to train a model
    # on an audio dataset.
