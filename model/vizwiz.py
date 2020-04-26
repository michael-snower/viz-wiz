"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for NLVR2 model
"""
import torch
from torch import nn
from torch.nn import functional as F

from .model import UniterPreTrainedModel, UniterModel
from .attention import MultiheadAttention
from utils.const import PAD_TOKEN
from utils.logger import LOGGER

from pdb import set_trace as bp


def count_trainable_params(mod):
    return sum([p.numel() for p in mod.parameters() if p.requires_grad is True])


class AttentionPool(nn.Module):
    """ attention pooling layer """
    def __init__(self, hidden_size, drop=0.0):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())
        self.dropout = nn.Dropout(drop)

    def forward(self, input_, mask=None):
        """input: [B, T, D], mask = [B, T]"""
        score = self.fc(input_).squeeze(-1)
        if mask is not None:
            mask = mask.to(dtype=input_.dtype) * -1e4
            score = score + mask
        norm_score = self.dropout(F.softmax(score, dim=1))
        output = norm_score.unsqueeze(1).matmul(input_).squeeze(1)
        return output


class VizWizModel(UniterPreTrainedModel):
    """ Finetune UNITER for NLVR2
        (paired format with additional attention layer)
    """
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)

        # fully connected stem
        self.fc1 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob))
        self.fc2 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob))

        # answerable head
        self.answerable_attn = MultiheadAttention(config.hidden_size,
                                config.num_attention_heads,
                                config.attention_probs_dropout_prob)
        self.answerable_pool = AttentionPool(config.hidden_size,
                                       config.attention_probs_dropout_prob)
        self.answerable_output = nn.Linear(config.hidden_size, 2)

        # loss functions
        self.answerable_loss = nn.CrossEntropyLoss()

        self.apply(self.init_weights)
        LOGGER.info("Built UniterForVizWiz model with {:,d} trainable params".format(
            count_trainable_params(self)
        ))

    def init_type_embedding(self):
        new_emb = nn.Embedding(3, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embeddings.token_type_embeddings\
                .weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        new_emb.weight.data[2, :].copy_(emb)
        self.uniter.embeddings.token_type_embeddings = new_emb

    def forward(self, input_ids, position_ids, img_feat,
                attn_masks, gather_index, answerable_targets,
                img_type_ids=None, img_pos_feat=None, compute_loss=True):

        sequence_output = self.uniter(
            input_ids=input_ids, 
            position_ids=position_ids,
            img_feat=img_feat, 
            img_pos_feat=img_pos_feat,
            attention_mask=attn_masks, 
            gather_index=gather_index,
            output_all_encoded_layers=False,
            img_type_ids=img_type_ids
        )

        hidden_state = self.fc1(sequence_output)
        hidden_state = self.fc2(hidden_state)

        # answerable head
        inverse_attn_masks = attn_masks == 0
        answerable_attention = self.answerable_attn(
            query=hidden_state, 
            key=hidden_state,
            value=hidden_state,
            key_padding_mask=inverse_attn_masks
        )
        answerable_pool = self.answerable_pool(answerable_attn, inverse_attn_masks)
        answerable_logits = self.answerable_output(answerable_pool)

        if compute_loss is True:
            answerable_loss = self.answer_loss(answerable_logits, answerable_targets)
            return answerable_loss
        else:
            return answerable_logits
