# Implement the pretrain class of the BLIP2 Q-former model
# This implementation uses the LAVIS/lavis/models/blip2_models/blip2_qformer.py from Salesforce as a reference
# It is modified to work with the huggingface implementation of the BLIP2 Q-former model


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import logging
from transformers import Blip2QFormerConfig

from .blip_2_m import Blip2QFormerModel
from .bert_tools import BertEmbeddings, BertOnlyMLMHead


class BLIP2QformerPretrain(nn.Module):
    "pretrain q-former from BLIP2"

    def __init__(self,
                 processor,
                 visual_encoder,
                 qformer_config,
                 qformer_device,
                 freeze_vit=True,
                 num_query_tokens=32,
                 embed_dim=256,
                 max_txt_len=32,
                 lm_reduction="mean", ):
        super().__init__()

        self.config = qformer_config
        self.qformer_device = qformer_device

        # add a special token [DEC] for the language modeling mode
        self.tokenizer = processor.tokenizer
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})

        self.visual_encoder = visual_encoder

        if freeze_vit:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            self.visual_encoder.eval()
            logging.info("Visual encoder is frozen")
        self.ln_vision = nn.LayerNorm(visual_encoder.num_features, eps=qformer_config.layer_norm_eps)

        # the qformer model
        self.Qformer = Blip2QFormerModel(qformer_config, num_query_tokens)

        # get query embeddings
        self.query_length = num_query_tokens
        self.query_embeds = nn.Parameter(torch.zeros(1, num_query_tokens, qformer_config.hidden_size))

        # embedding layer for text embeddings
        self.embeddings = BertEmbeddings(qformer_config)

        # vision and text projection for image-text contrastive learning
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        # head for image-text matching
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        # temperature parameter for contrastive learning
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.cls = BertOnlyMLMHead(qformer_config)
        self.max_txt_len = max_txt_len
        self.lm_reduction = lm_reduction
        self.lm_loss_fct = CrossEntropyLoss(reduction=lm_reduction, label_smoothing=0.1)

    def forward(self, samples):
        image = samples["image"]
        text = samples["text_input"]

        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.qformer_device)

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.qformer_device)
        text_embeds = self.embeddings(input_ids=text_tokens["input_ids"])

        query_embeds = self.query_embeds.expand(image_embeds.shape[0], -1, -1)

        # ============== Image-text Contrastive =================== #
        qformer_itc_output = self.Qformer(
            query_embeds=query_embeds,
            mode='itc',
            text_embeds=text_embeds,
            attention_mask=text_tokens["attention_mask"],
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            use_cache=True,
        )

        last_hidden_state = qformer_itc_output.last_hidden_state
        query_output = last_hidden_state[:, :self.query_length, :]
        text_output = last_hidden_state[:, self.query_length:, :]

        image_feats = F.normalize(self.vision_proj(query_output), dim=-1)
        text_feats = F.normalize(self.text_proj(text_output[:, 0, :]), dim=-1)

        # compute image-text contrastive loss
        sim_q2t = torch.matmul(image_feats.unsqueeze(1),
                               text_feats.unsqueeze(-1)).squeeze()  # [batch_size, batch_size, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size, num_query_tokens]
        sim_t2q = torch.matmul(text_feats.unsqueeze(1).unsqueeze(1), image_feats.permute(0, 2, 1)).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size]

        bs = image_embeds.size(0)
        targets = torch.arange(bs, dtype=torch.int64).to(self.qformer_device)
        loss_itc = (
                           F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                           + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
                   ) / 2

        # ============== Image-text Matching ===================#
        text_input_ids_world = text_tokens.input_ids
        text_attention_mask_world = text_tokens.attention_mask
        image_embeds_world = image_embeds

        with torch.no_grad():
            sim_t2i[:, :bs].fill_diagonal_(-10000)
            sim_i2t[:, :bs].fill_diagonal_(-10000)

            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)

        # select a negative image for each text; use hard negatives
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])
        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )
        text_embdes_all = self.embeddings(input_ids=text_ids_all)

        image_embeds_all = torch.cat([image_embeds, image_embeds_neg, image_embeds], dim=0)  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(image.device)

        query_embeds_itm = self.query_embeds.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_embeds_itm.size()[:-1], dtype=torch.long).to(image.device)
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        qformer_itm_output = self.Qformer(
            query_embeds=query_embeds_itm,
            mode='itm',
            text_embeds=text_embdes_all,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = qformer_itm_output.last_hidden_state[:, : query_embeds_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        # ================= Image Grounded Text Generation ======================== #
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100)
        decoder_text_embeds = self.embeddings(input_ids=decoder_input_ids)

        query_atts = torch.ones(query_embeds.size()[:-1], dtype=torch.long).to(image.device)
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        qformer_lm_output = self.Qformer(
            query_embeds=query_embeds,
            mode='lm',
            text_embeds=decoder_text_embeds,
            attention_mask=attention_mask,
            # encoder_hidden_states=image_embeds,
            # encoder_attention_mask=image_atts,
            past_key_values=qformer_itc_output.past_key_values,
            return_dict=True,
        )

        sequence_output = qformer_lm_output.last_hidden_state[:, query_embeds.shape[1]:, :]
        prediction_scores = self.cls(sequence_output)

        # we are doing next-token prediction; shift prediction scores and input ids by one
        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        lm_loss = self.lm_loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        if self.lm_reduction == "none":
            lm_loss = lm_loss.view(prediction_scores.size(0), -1).sum(1)

        # ===================== Total Loss ======================== #
        loss = loss_itc + loss_itm + lm_loss
        return loss, loss_itc, loss_itm, lm_loss
