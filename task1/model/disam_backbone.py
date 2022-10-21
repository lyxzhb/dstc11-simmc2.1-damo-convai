import torch
from torch import nn
from transformers import RobertaModel

from .auxiliary import DisamTypeHead

class DisamModel(nn.Module):
    ''' 将整体结构封装为一个模型'''

    def __init__(self, args):
        super(DisamModel, self).__init__()
        self.encoder = RobertaModel.from_pretrained(args.backbone)

        if args.add_special_tokens:
            self.encoder.resize_token_embeddings(args.len_tokenizer)
            self.encoder.vocab_size = args.len_tokenizer

        self.disam_head = DisamTypeHead(self.encoder.config.hidden_size)
        self.CELoss = nn.CrossEntropyLoss()  # 多分类

    def forward(self, input_ids, attention_mask, disam_label):
        
        enc_pooler_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output

        disam_logits = self.disam_head(enc_pooler_output)
        loss = self.CELoss(disam_logits, disam_label)
        
        return loss
    
    def evaluate(self, input_ids, attention_mask, disam_label, active_scene):
        
        batch_size = input_ids.size(0)
        
        enc_pooler_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output

        disambiguation_logits = self.disam_head(enc_pooler_output)     
        disambiguation_pred = disambiguation_logits.argmax(dim=1).tolist()
        
        output = []
        
        for idx in range(batch_size):
            output.append({
                'target_label': disam_label[idx].item(),
                'pred_label': disambiguation_pred[idx],
                'active_scene': active_scene[idx]
            })
        
        return output