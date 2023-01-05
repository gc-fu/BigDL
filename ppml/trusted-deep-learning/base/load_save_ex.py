#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import torch.nn as nn
from bigdl.nano.pytorch.patching import patch_encryption
from bigdl.ppml.kms.client import generate_primary_key, generate_data_key, get_data_key_plaintext
from datasets.load import load_from_disk
import argparse
from transformers import BertTokenizer, BertModel, AdamW

parser = argparse.ArgumentParser(description="PyTorch PERT Example")
parser.add_argument("--local-only", action="store_true", default=False,
                    help="If set to true, then load model from disk")
parser.add_argument("--model-path", type=str, default="/ppml/model",
                    help="Where to load model")
parser.add_argument("--dataset-path", type=str, default="/ppml/dataset",
                    help="Where to load original dataset")


# python3 load_save_ex.py --local-only --model-path /ppml/model --dataset-path /ppml/save-datasets/train/
args = parser.parse_args()


# Define APPID and APIKEY in os.environment
APPID = os.environ.get('APPID')
APIKEY = os.environ.get('APIKEY')


encrypted_primary_key_path = ""
encrypted_data_key_path = ""

EHSM_IP = os.environ.get('ehsm_ip')
EHSM_PORT = os.environ.get('ehsm_port', "9000")

if args.local_only:
    checkpoint = args.model_path
    tokenizer = BertTokenizer.from_pretrained(
        checkpoint, model_max_length=512, local_files_only=True)
else:
    checkpoint = 'hfl/chinese-pert-base'
    tokenizer = BertTokenizer.from_pretrained(checkpoint, model_max_length=512)

# prepare environment
def prepare_env():
    if APPID is None or APIKEY is None or EHSM_IP is None:
        print("Please set environment variable APPID, APIKEY, ehsm_ip!")
        exit(1)
    generate_primary_key(EHSM_IP, EHSM_PORT)
    global encrypted_primary_key_path
    encrypted_primary_key_path = "./encrypted_primary_key"
    generate_data_key(EHSM_IP, EHSM_PORT, encrypted_primary_key_path, 32)
    global encrypted_data_key_path
    encrypted_data_key_path = "./encrypted_data_key"
    patch_encryption()

# Get a key from kms that can be used for encryption/decryption
def get_key():
    return get_data_key_plaintext(EHSM_IP, EHSM_PORT, encrypted_primary_key_path, encrypted_data_key_path)

def save_encrypted_dataset(dataset_path, save_path, secret_key):
    dataset = load_from_disk(dataset_path, keep_in_memory=True)
    # This will save the encrypted dataset into disk
    torch.save(dataset, save_path, encryption_key = secret_key)

class Dataset(torch.utils.data.Dataset):
    # data_type is actually split, so that we can define dataset for train set/validate set
    def __init__(self, data_path, key):
        self.data = self.load_data(data_path, key)

    def load_data(self, data_path, key):
        #tmp_dataset = load_dataset(path='seamew/ChnSentiCorp', split=data_type)
        tmp_dataset = torch.load(data_path, decryption_key = key)
        Data = {}
        # So enumerate will return a index, and  the line?
        # line is a dict, including 'text', 'label'
        for idx, line in enumerate(tmp_dataset):
            sample = line
            Data[idx] = sample

        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch_samples):
    batch_text = []
    batch_label = []
    for sample in batch_samples:
        batch_text.append(sample['text'])
        batch_label.append(int(sample['label']))
    # The tokenizer will make the data to be a good format for our model to understand
    X = tokenizer(
        batch_text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X, y





def main():
    prepare_env()
    # This is only safe in sgx environment, where the memory can not be read
    secret_key = get_key()

    encrypted_dataset_path = "/ppml/encryption_dataset.pt"

    # Assume we are in customer environment (which is safe and trusted)
    save_encrypted_dataset(args.dataset_path, encrypted_dataset_path, secret_key)

    # Now we have the encrypted dataset, we can safely distribute it into
    # untrusted environments.

    # load the encrypted dataset back and ready for training 
    train_dataset = Dataset(encrypted_dataset_path, secret_key)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    print("[INFO]Data get loaded successfully", flush=True)


if __name__ == "__main__":
    main()
