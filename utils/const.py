"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

constants
"""
from pytorch_pretrained_bert import BertTokenizer


IMG_DIM = 2048
IMG_LABEL_DIM = 1601
BUCKET_SIZE = 8192


tokenizer = BertTokenizer.from_pretrained(
    'bert-base-cased', do_lower_case=False
)
PAD_TOKEN = tokenizer.convert_tokens_to_ids(["[PAD]"])[0]
