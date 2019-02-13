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

from transformers import AutoModelForSequenceClassification, AutoTokenizer

import torch
from torch.utils.data import DataLoader

# from https://huggingface.co/docs/datasets/en/quickstart#nlp
if __name__ == "__main__":

    # Text needs to be tokenized into individual tokens by a tokenizer. For the quickstart,
    # you’ll load the Microsoft Research Paraphrase Corpus (MRPC) training dataset to train
    # a model to determine whether a pair of sentences mean the same thing.

    # 1. Load the MRPC dataset by providing the load_dataset() function with the dataset name,
    # dataset configuration (not all datasets will have a configuration), and dataset split:
    dataset = load_dataset("nyu-mll/glue", "mrpc", split="train")

    # 2. Next, load a pretrained BERT model and its corresponding tokenizer from the 🤗 Transformers library.
    # It is totally normal to see a warning after you load the model about some weights not being initialized.
    # This is expected because you are loading this model checkpoint for training with another task.
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # 3. Create a function to tokenize the dataset, and you should also truncate and pad the text into
    # tidy rectangular tensors. The tokenizer generates three new columns in the dataset:
    # input_ids, token_type_ids, and an attention_mask. These are the model inputs.
    # Use the map() function to speed up processing by applying your tokenization function to batches of examples in the dataset:
    def encode(examples):
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            padding="max_length",
        )

    dataset = dataset.map(encode, batched=True)
    dataset[0]
    {
        "sentence1": 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
        "sentence2": 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .',
        "label": 1,
        "idx": 0,
        "input_ids": [
            101,
            7277,
            2180,
            5303,
            4806,
            1117,
            1711,
            117,
            2292,
            1119,
            1270,
            107,
            1103,
            7737,
            107,
            117,
            1104,
            9938,
            4267,
            12223,
            21811,
            1117,
            2554,
            119,
            102,
            11336,
            6732,
            3384,
            1106,
            1140,
            1112,
            1178,
            107,
            1103,
            7737,
            107,
            117,
            7277,
            2180,
            5303,
            4806,
            1117,
            1711,
            1104,
            9938,
            4267,
            12223,
            21811,
            1117,
            2554,
            119,
            102,
            0,
            0,
            ...,
        ],
        "token_type_ids": [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            ...,
        ],
        "attention_mask": [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            ...,
        ],
    }

    # 4. Rename the label column to labels, which is the expected input name in BertForSequenceClassification:
    dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)

    # 5. Set the dataset format according to the machine learning framework you’re using.
    # Use the with_format() function to set the dataset format to torch and specify the columns you want to format.
    # This function applies formatting on-the-fly.
    # After converting to PyTorch tensors, wrap the dataset in torch.utils.data.DataLoader:

    dataset = dataset.select_columns(
        ["input_ids", "token_type_ids", "attention_mask", "labels"]
    )
    dataset = dataset.with_format(type="torch")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
