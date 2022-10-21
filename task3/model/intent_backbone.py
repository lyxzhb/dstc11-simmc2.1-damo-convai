import torch
from torch import nn
from transformers import RobertaModel

from .auxiliary import IntentHead, DisamHead, IntentSubHead

class IntentModel(nn.Module):
    ''' 将整体结构封装为一个模型'''

    def __init__(self, args):
        super(IntentModel, self).__init__()
        self.encoder = RobertaModel.from_pretrained(args.backbone)

        if args.add_special_tokens:
            self.encoder.resize_token_embeddings(args.len_tokenizer)
            self.encoder.vocab_size = args.len_tokenizer

        self.intent_head = IntentHead(self.encoder.config.hidden_size)


    def forward(self, input_ids, attention_mask):
        enc_pooler_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output

        intent_logits = self.intent_head(enc_pooler_output)

        return intent_logits


class IntentSubModel(nn.Module):
    ''' 将整体结构封装为一个模型'''

    def __init__(self, args):
        super(IntentSubModel, self).__init__()
        self.encoder = RobertaModel.from_pretrained(args.backbone)

        if args.add_special_tokens:
            self.encoder.resize_token_embeddings(args.len_tokenizer)
            self.encoder.vocab_size = args.len_tokenizer

        self.intent_head = IntentSubHead(self.encoder.config.hidden_size)


    def forward(self, input_ids, attention_mask):
        enc_pooler_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output

        intent_logits = self.intent_head(enc_pooler_output)

        return intent_logits
    
    
class DisamModel(nn.Module):
    ''' 将整体结构封装为一个模型'''

    def __init__(self, args):
        super(DisamModel, self).__init__()
        self.encoder = RobertaModel.from_pretrained(args.backbone)

        if args.add_special_tokens:
            self.encoder.resize_token_embeddings(args.len_tokenizer)
            self.encoder.vocab_size = args.len_tokenizer

        self.disam_head = DisamHead(self.encoder.config.hidden_size)


    def forward(self, input_ids, attention_mask):
        enc_pooler_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output

        intent_logits = self.disam_head(enc_pooler_output)

        return intent_logits