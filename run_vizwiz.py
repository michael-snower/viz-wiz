"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER finetuning for NLVR2
"""
import argparse
import os
from os.path import exists, join
from time import time

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from apex import amp
from horovod import torch as hvd

from tqdm import tqdm

from data.vizwiz import VizWizDataset, vizwiz_collate
from model.vizwiz import VizWizModel
from optim import get_lr_sched
from optim.misc import build_optimizer

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from utils.const import IMG_DIM

from pdb import set_trace as bp


def main(opts):

    device = torch.device("cuda")
    LOGGER.info(
        "16-bits training: {}".format(opts.fp16))

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))

    set_random_seed(opts.seed)

    LOGGER.info("Loading data from {}".format(opts.data_dir))

    # create datasets
    train_set = VizWizDataset(opts.data_dir, split="train")
    val_set = VizWizDataset(opts.data_dir, split="val")

    # data loaders
    train_dataloader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=opts.train_batch_size,
        num_workers=opts.n_workers,
        collate_fn=vizwiz_collate
    )
    val_dataloader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=opts.train_batch_size,
        num_workers=opts.n_workers,
        collate_fn=vizwiz_collate
    )

    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint)
    else:
        checkpoint = {}
    model = VizWizModel.from_pretrained(opts.model_config, state_dict=checkpoint,
                                     img_dim=IMG_DIM)
    model.init_type_embedding()
    model.to(device)
    # make sure every process has same model parameters in the beginning\
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    model, optimizer = amp.initialize(model, optimizer,
                                      enabled=opts.fp16, opt_level='O2')

    # setup logging
    save_training_meta(opts)
    TB_LOGGER.create(join(opts.output_dir, 'log'))
    pbar = tqdm(total=opts.num_train_steps)
    model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
    if not os.path.exists(join(opts.output_dir, 'results')):
        os.makedirs(join(opts.output_dir, 'results'))
    add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))


    LOGGER.info(f"***** Running training *****")
    LOGGER.info("  Num examples = %d", len(train_dataloader.dataset))
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    running_loss = RunningMeter('loss')
    model.train()
    n_examples = 0
    n_epoch = 0
    start = time()
    global_step = 0

    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()

    while True:

        for step, batch in enumerate(train_dataloader):

            input_ids = batch['qs_tok'].to(device)
            img_feats = batch["img_feats"].to(device)
            attn_masks = batch["attn_masks"].to(device)
            position_ids = batch["position_ids"].to(device)
            answerable_targets = batch["answerables"].to(device)
            n_examples += input_ids.size(0)

            loss = model(
                input_ids=input_ids,
                position_ids=position_ids,
                img_feat=img_feats, 
                attn_masks=attn_masks,
                gather_index=None,
                answerable_targets=answerable_targets,
                compute_loss=True
            )
            loss = loss.mean()

            delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
            with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale
                                ) as scaled_loss:
                scaled_loss.backward()
                # if not delay_unscale:
                #     # gather gradients from every processes
                #     # do this before unscaling to make sure every process uses
                #     # the same gradient scale
                #     grads = [p.grad.data for p in model.parameters()
                #              if p.requires_grad and p.grad is not None]
                    #all_reduce_and_rescale_tensors(grads, float(1))

            running_loss(loss.item())

            if (step + 1) % opts.gradient_accumulation_steps == 0:
                global_step += 1

                # learning rate scheduling
                lr_this_step = get_lr_sched(global_step, opts)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                #losses = all_gather_list(running_loss)
                TB_LOGGER.add_scalar('loss', running_loss.val, global_step)
                TB_LOGGER.step()

                # update model params
                if opts.grad_norm != -1:
                    grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                                opts.grad_norm)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

                if global_step % 100 == 0:
                    # monitor training throughput
                    tot_ex = n_examples
                    ex_per_sec = int(tot_ex / (time()-start))
                    LOGGER.info(f'Step {global_step}: '
                                f'{tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s')
                    TB_LOGGER.add_scalar('perf/ex_per_s',
                                         ex_per_sec, global_step)

                if global_step % opts.valid_steps == 0:

                    LOGGER.info(
                        f"Step {global_step}: start running "
                        f"evaluation on val..."
                    )
                    log, results = validate(model, val_dataloader)
                    model_saver.save(model, global_step)

            if global_step >= opts.num_train_steps:
                break

        if global_step >= opts.num_train_steps:
            break

        n_epoch += 1
        LOGGER.info(f"Step {global_step}: finished {n_epoch} epochs")

    LOGGER.info(
        f"Step {global_step}: start running "
        f"evaluation on val..."
    )
    log, results = validate(model, val_dataloader)
    model_saver.save(model, f'{global_step}_final')


@torch.no_grad()
def validate(model, val_loader):
    model.eval()
    val_loss = 0
    tot_score = 0
    n_ex = 0
    st = time()
    results = []
    for i, batch in enumerate(val_loader):
        qids = batch['qids']
        targets = batch['targets']
        del batch['targets']
        del batch['qids']
        scores = model(**batch, targets=None, compute_loss=False)
        loss = F.cross_entropy(scores, targets, reduction='sum')
        val_loss += loss.item()
        tot_score += (scores.max(dim=-1, keepdim=False)[1] == targets
                      ).sum().item()
        answers = ['True' if i == 1 else 'False'
                   for i in scores.max(dim=-1, keepdim=False
                                       )[1].cpu().tolist()]
        results.extend(zip(qids, answers))
        n_ex += len(qids)
    val_loss = sum(all_gather_list(val_loss))
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    tot_time = time()-st
    val_loss /= n_ex
    val_acc = tot_score / n_ex
    val_log = {f'valid/{split}_loss': val_loss,
               f'valid/{split}_acc': val_acc,
               f'valid/{split}_ex_per_s': n_ex/tot_time}
    model.train()
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc*100:.2f}")
    return val_log, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir")
    parser.add_argument("--model_config",
                        default=None, type=str,
                        help="json file for model architecture")
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained model")
    parser.add_argument("--model", default='answerable',
                        choices=['answerable'],
                        help="choose from model architecture")
    parser.add_argument("--predict_type", action="store_true")

    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')

    # training parameters
    parser.add_argument("--train_batch_size",
                        default=4096, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size",
                        default=4096, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--valid_steps",
                        default=1000,
                        type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps",
                        default=100000,
                        type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adam',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+', type=float,
                        help="beta for adam optimizer")
    parser.add_argument("--dropout",
                        default=0.1,
                        type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm",
                        default=0.25,
                        type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps",
                        default=4000,
                        type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for.")

    # device parameters
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)

    # if exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not "
    #                      "empty.".format(args.output_dir))

    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
