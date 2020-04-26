from collections import defaultdict
import json
import os
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils.logger import LOGGER
from utils.const import PAD_TOKEN

from pdb import set_trace as bp


class VizWizDataset(Dataset):

    def __init__(self, data_dir, split):
        
        self.data_dir = data_dir
        self.split = split

        self.img_dir = os.path.join(self.data_dir, "img", self.split)
        self.ann_dir = os.path.join(self.data_dir, "ann", self.split)

        self.img_names = [
            img_name_with_ext.split(".")[0]
            for img_name_with_ext in os.listdir(self.img_dir)
        ]

        self.pre_dir = os.path.join(self.data_dir, "pre", self.split)
        self.pre_vis_feat_dir = os.path.join(self.pre_dir, "visual_features")
        self.pre_q_tok_dir = os.path.join(self.pre_dir, "question_tokens")
        self.pre_ans_tok_dir = os.path.join(self.pre_dir, "answer_tokens")

        LOGGER.info("Created {} set with {} examples".format(split, len(self)))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):

        img_name = self.img_names[idx]

        pre_vis_feat_path = os.path.join(self.pre_vis_feat_dir, img_name + ".npy")
        img_feat = np.load(pre_vis_feat_path)

        pre_q_tok_path = os.path.join(self.pre_q_tok_dir, img_name + ".npy")
        q_tok = np.load(pre_q_tok_path)
        attn_mask = np.ones_like(q_tok, dtype=np.int32)

        pre_ans_tok_path = os.path.join(self.pre_ans_tok_dir, img_name + ".npy")
        ans_tok = np.load(ans_tok)
        answerable = ans_tok[0][0]
        answer_type = ans_tok[0][1]
        answers = ans_tok[1:, :]
        bp()
        pad_mask = answers == -1
        answers[pad_mask] = PAD_TOKEN

        return img_name, img_feat, q_tok, attn_mask, answers, answerable, answer_type


def vizwiz_collate(*inputs):
    
    img_names, img_feats, qs_tok, attn_masks, answers_tok, answerables, answers_type = inputs

    max_num_feats = max([len(img_feat) for img_feat in img_feats])
    all_feat = []
    for img_feat in img_feats:
        num_feats = len(img_feat)
        num_diff = max_num_feats - num_feats
        img_feat = torch.FloatTensor(img_feat)
        img_feat_dim = img_feat.shape[1]
        padding = torch.zeros((num_diff, img_feat_dim), dtype=torch.float32)
        img_feat = torch.cat([img_feat, padding], dim=0)
        all_feat.append(img_feat)
    img_feats = torch.FloatTensor(all_feat)

    qs_tok = torch.LongTensor(qs_tok)
    qs_tok = pad_sequence(qs_tok, batch_first=True, padding_value=PAD_TOKEN)

    attn_masks = torch.LongTensor(attn_masks)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    attn_padding = torch.ones((len(attn_masks), img_feats.shape[1]), dtype=torch.int32)
    attn_masks = torch.cat([attn_masks, attn_padding], dim=1)

    position_ids = torch.arange(0, qs_tok.size(1), dtype=torch.long
                                ).unsqueeze(0)
    position_ids = torch.stack([position_ids] * len(img_feats), dim=0)

    all_answers = []
    max_answer_len = max([answer_tok.shape[1] for answer_tok in answers_tok])
    for answer_tok in answers_tok:
        answer_len = answer_tok.shape[1]
        len_diff = max_answer_len - answer_len
        answer_feat = torch.FloatTensor(answer_feat)
        num_answers = len(answer_feat)
        padding = torch.ones((num_answers, len_diff), dtype=torch.float32) * PAD_TOKEN
        answer_tok = torch.cat([answer_tok, padding], dim=1)
        all_answers.append(answer_tok)
    answers_tok = torch.LongTensor(all_answers)

    answerables = torch.LongTensor(answerables)

    answers_type = torch.LongTensor(answers_type)

    return {
        "img_names"      : img_names,
        "img_feats"      : img_feats,
        "qs_tok"         : q_toks,
        "attn_masks"     : attn_masks,
        "position_ids"   : position_ids,
        "answers_tok"    : answers_tok,
        "answerables"    : answerables,
        "answers_type"   : answers_type
    }

