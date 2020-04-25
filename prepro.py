"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

preprocess NLVR annotations into LMDB
"""
import argparse
import json
import os
from os.path import exists

from cytoolz import curry
from tqdm import tqdm
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
import cv2 as cv

from data.data import open_lmdb

# import mask rcnn modules
import sys
sys.path.append('maskrcnn/')
from maskrcnn_benchmark.config import cfg
from maskrcnn.demo.predictor import COCODemo

from pdb import set_trace as bp

@curry
def bert_tokenize(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char
            continue
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids


def process_nlvr2(jsonl, db, tokenizer, missing=None):
    id2len = {}
    txt2img = {}  # not sure if useful
    for line in tqdm(jsonl, desc='processing NLVR2'):
        example = json.loads(line)
        id_ = example['identifier']
        img_id = '-'.join(id_.split('-')[:-1])
        img_fname = (f'nlvr2_{img_id}-img0.npz', f'nlvr2_{img_id}-img1.npz')
        if missing and (img_fname[0] in missing or img_fname[1] in missing):
            continue
        input_ids = tokenizer(example['sentence'])
        if 'label' in example:
            target = 1 if example['label'] == 'True' else 0
        else:
            target = None
        txt2img[id_] = img_fname
        id2len[id_] = len(input_ids)
        example['input_ids'] = input_ids
        example['img_fname'] = img_fname
        example['target'] = target
        db[id_] = example
    return id2len, txt2img

def extract_visual_features(img, visual_model):
    vis, bbox, features = visual_model.run_on_opencv_image(img)
    return features, vis

def process_vizwiz(ann, tokenizer, image_dir, output_dir):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    id2len = {}
    text2img = {}
    visual_features_dir = os.path.join(output_dir, "visual_features")
    if not os.path.exists(visual_features_dir):
        os.mkdir(visual_features_dir)
    questions_tokens_dir = os.path.join(output_dir, "question_tokens")
    if not os.path.exists(questions_tokens_dir):
        os.mkdir(questions_tokens_dir)
    answer_tokens_dir = os.path.join(output_dir, "answer_tokens")
    if not os.path.exists(answer_tokens_dir):
        os.mkdir(answer_tokens_dir)

    # load visual feature extractor
    config_file = "maskrcnn/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.MASK_ON", False])
    visual_model = COCODemo(cfg, confidence_threshold=0.2)

    for example_index, example in enumerate(tqdm(ann, desc=f"Saving features and tokens to {output_dir}")):

        # save visual features
        img_name_with_ext = example["image"]
        img_name = img_name_with_ext.split(".")[0]
        img = cv.imread(os.path.join(image_dir, img_name_with_ext))
        visual_features, vis = extract_visual_features(img, visual_model)
        vis_feat_save_path = os.path.join(visual_features_dir, img_name + ".npy")
        np.save(open(vis_feat_save_path, "wb"), visual_features)
        cv.imwrite(img_name_with_ext, vis)

        # tokenize question
        question_tokens = tokenizer(example["question"])
        quest_tok_save_path = os.path.join(questions_tokens_dir, img_name + ".npy")
        np.save(open(quest_tok_save_path, "wb"), question_tokens)

        # tokenize answers
        answers_text = example["answers"]
        answers_tok = [tokenizer(answer_text["answer"]) for answer_text in answers_text]
        max_answer_len = max([len(answer_tok) for answer_tok in answers_tok])
        assert max_answer_len > 0
        max_answer_len = 2 if max_answer_len < 2 else max_answer_len
        answers_tok_np = -np.ones((len(answers_tok) + 1, max_answer_len), dtype=np.int32)
        # add in other label information for answer in first row
        answers_tok_np[0][0] = example["answerable"]
        answer_type = example["answer_type"]
        if answer_type == "unanswerable":
            answers_tok_np[0][1] = 0
        elif answer_type == "yes/no":
            answers_tok_np[0][1] = 1
        elif answer_type == "other":
            answers_tok_np[0][1] = 2
        else:
            raise ValueError("Unidentified answer type")
        for answer_tok, row_index in zip(answers_tok, range(1, len(answers_tok_np))):
            answers_tok_np[row_index, :len(answer_tok)] = answer_tok
        answer_tok_save_path = os.path.join(answer_tokens_dir, img_name + ".npy")
        np.save(open(answer_tok_save_path, "wb"), answers_tok_np)

def main(opts):
    if not exists(opts.output):
        os.makedirs(opts.output)
    # else:
    #     raise ValueError('Found existing DB. Please explicitly remove '
    #                      'for re-processing')
    meta = vars(opts) 
    meta['tokenizer'] = opts.toker
    toker = BertTokenizer.from_pretrained(
        opts.toker, do_lower_case='uncased' in opts.toker)
    tokenizer = bert_tokenize(toker)
    meta['UNK'] = toker.convert_tokens_to_ids(['[UNK]'])[0]
    meta['CLS'] = toker.convert_tokens_to_ids(['[CLS]'])[0]
    meta['SEP'] = toker.convert_tokens_to_ids(['[SEP]'])[0]
    meta['MASK'] = toker.convert_tokens_to_ids(['[MASK]'])[0]
    meta['v_range'] = (toker.convert_tokens_to_ids('!')[0],
                       len(toker.vocab))
    with open(f'{opts.output}/meta.json', 'w') as f:
        json.dump(vars(opts), f, indent=4)

    if opts.dataset == "nvlr2":
        open_db = curry(open_lmdb, opts.output, readonly=False)
        with open_db() as db:
                with open(opts.annotation) as ann:
                        if opts.missing_imgs is not None:
                            missing_imgs = set(json.load(open(opts.missing_imgs)))
                        else:
                            missing_imgs = None
                        id2lens, txt2img = process_nlvr2(ann, db, tokenizer, missing_imgs)

        with open(f'{opts.output}/id2len.json', 'w') as f:
            json.dump(id2lens, f)
        with open(f'{opts.output}/txt2img.json', 'w') as f:
            json.dump(txt2img, f)
    else:
        train_ann_path = os.path.join(opts.annotation, "train.json")
        train_img_dir = os.path.join(opts.img_dir, "train")
        train_output_dir = f'{opts.output}/train/'

        with open(train_ann_path, "r") as ann_file:
            ann = json.load(ann_file)
            process_vizwiz(ann, tokenizer, train_img_dir, train_output_dir)

        val_ann_path = os.path.join(opts.annotation, "val.json")
        val_img_dir = os.path.join(opts.img_dir, "val")
        val_output_dir = f'{opts.output}/val/'

        with open(val_ann_path) as ann_file:
            ann = json.load(ann_file)
            process_vizwiz(ann, tokenizer, val_img_dir, val_output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', required=True,
                        help='annotation JSON')
    parser.add_argument("--img_dir", required=True, help="img dir")
    parser.add_argument('--missing_imgs',
                        help='some training image features are corrupted')
    parser.add_argument('--output', required=True,
                        help='output dir of DB')
    parser.add_argument('--toker', default='bert-base-cased',
                        help='which BERT tokenizer to used')
    parser.add_argument("--dataset", default="vizwiz")
    args = parser.parse_args()
    main(args)
