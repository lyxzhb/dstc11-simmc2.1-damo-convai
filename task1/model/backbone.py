import torch
from torch import nn
from transformers import LongformerModel

from .auxiliary import (BoxEmbedding, 
                        NoCorefHead, 
                        FashionEncoderHead, 
                        FurnitureEncoderHead, 
                        DisambiguationHead, 
                        IntentHead)
from .auxiliary import (FashionTypeHead,
                        FashionAvaliableSizeHead,
                        FashionBrandHead,
                        FashionColorHead,
                        FashionCustomerReviewHead,
                        FashionPatternHead,
                        FashionPriceHead,
                        FashionSizeHead,
                        FashionSleeveLengthHead)
from .auxiliary import (FurnitureBrandHead,
                        FurnitureColorHead,
                        FurnitureCustomerRatingHead,
                        FurnitureMaterialHead,
                        FurniturePriceHead,
                        FurnitureTypeHead)

from .auxiliary import (DisamAllHead,
                        DisamTypeHead)


class VLBertModelWithDisam(nn.Module):
    ''' 将整体结构封装为一个模型'''

    def __init__(self, args):
        super(VLBertModelWithDisam, self).__init__()
        self.encoder = LongformerModel.from_pretrained(args.backbone)

        if args.add_special_tokens:
            self.encoder.resize_token_embeddings(args.len_tokenizer)
            self.encoder.vocab_size = args.len_tokenizer

        self.box_embedding = BoxEmbedding(self.encoder.config.hidden_size)

        self.fashion_enc_head = FashionEncoderHead(self.encoder.config.hidden_size)
        self.furniture_enc_head = FurnitureEncoderHead(self.encoder.config.hidden_size)
        
        self.CELoss = nn.CrossEntropyLoss()  # 多分类
        self.BCELoss = nn.BCEWithLogitsLoss()  # 多个2分类，并且不需要经过Sigmod


    def forward(self, enc_input, enc_attention_mask, boxes, misc):
        inputs_embeds = self.encoder.embeddings(enc_input)
        batch_size = inputs_embeds.size(0)

        for b_idx in range(batch_size):  # in a batch
            box_embedded = self.box_embedding(torch.tensor(boxes[b_idx]).to(inputs_embeds.device))  # (num_obj_per_line, d_model)
            for obj_idx in range(len(misc[b_idx])):
                pos = misc[b_idx][obj_idx]['pos']
                inputs_embeds[b_idx][pos] += box_embedded[obj_idx]
        
        enc_last_state = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=enc_attention_mask,
        ).last_hidden_state
        
        # 4. Encoder 端的Misc Loss
        misc_loss = 0
        for b_idx in range(batch_size):  # in a batch
            is_fashion = misc[b_idx][0]['is_fashion']
            coref_label = [misc[b_idx][obj_idx]['coref_label'] for obj_idx in range(len(misc[b_idx]))]  # (num_obj)  0 or 1
            
            if is_fashion:
                size_label = [misc[b_idx][obj_idx]['misc_labels']['size'] for obj_idx in range(len(misc[b_idx]))]  # (num_obj)
                available_sizes_label = [misc[b_idx][obj_idx]['misc_labels']['available_sizes'] for obj_idx in range(len(misc[b_idx]))]  # (num_obj, 6)
                brand_label = [misc[b_idx][obj_idx]['misc_labels']['brand'] for obj_idx in range(len(misc[b_idx]))]
                color_label = [misc[b_idx][obj_idx]['misc_labels']['color'] for obj_idx in range(len(misc[b_idx]))]
                pattern_label = [misc[b_idx][obj_idx]['misc_labels']['pattern'] for obj_idx in range(len(misc[b_idx]))]
                sleeve_length_label = [misc[b_idx][obj_idx]['misc_labels']['sleeve_length'] for obj_idx in range(len(misc[b_idx]))]
                asset_type_label = [misc[b_idx][obj_idx]['misc_labels']['asset_type'] for obj_idx in range(len(misc[b_idx]))]
                type_label = [misc[b_idx][obj_idx]['misc_labels']['type']for obj_idx in range(len(misc[b_idx]))]
                price_label = [misc[b_idx][obj_idx]['misc_labels']['price'] for obj_idx in range(len(misc[b_idx]))]
                customer_review_label = [misc[b_idx][obj_idx]['misc_labels']['customer_review'] for obj_idx in range(len(misc[b_idx]))]
            else:
                brand_label = [misc[b_idx][obj_idx]['misc_labels']['brand'] for obj_idx in range(len(misc[b_idx]))]  # (num_obj)
                color_label = [misc[b_idx][obj_idx]['misc_labels']['color'] for obj_idx in range(len(misc[b_idx]))]
                materials_label = [misc[b_idx][obj_idx]['misc_labels']['materials'] for obj_idx in range(len(misc[b_idx]))]
                type_label = [misc[b_idx][obj_idx]['misc_labels']['type'] for obj_idx in range(len(misc[b_idx]))]
                price_label = [misc[b_idx][obj_idx]['misc_labels']['price'] for obj_idx in range(len(misc[b_idx]))]
                customer_review_label = [misc[b_idx][obj_idx]['misc_labels']['customer_review'] for obj_idx in range(len(misc[b_idx]))]
            
            for obj_idx in range(len(misc[b_idx])):
                pos = misc[b_idx][obj_idx]['pos']
                # hidden_concat: (num_obj, 2*model)
                if obj_idx == 0:
                    hidden_concat = torch.reshape(enc_last_state[b_idx][pos:pos+2], (1, -1))
                else:
                    hidden_concat = torch.cat([hidden_concat, torch.reshape(enc_last_state[b_idx][pos:pos+2], (1, -1))], dim=0)

            if is_fashion:
                coref, size, available_sizes, brand, color, pattern, sleeve_length, asset_type, type_, price, customer_review = self.fashion_enc_head(hidden_concat)  # (num_obj, num_logits)
                
                loss_per_line = 8 * self.CELoss(coref, torch.tensor(coref_label, dtype=torch.long).to(inputs_embeds.device)) + \
                                    self.CELoss(color, torch.tensor(color_label, dtype=torch.long).to(inputs_embeds.device)) + \
                                    self.CELoss(pattern, torch.tensor(pattern_label, dtype=torch.long).to(inputs_embeds.device)) + \
                                    self.CELoss(type_, torch.tensor(type_label, dtype=torch.long).to(inputs_embeds.device)) + \
                                    self.CELoss(size, torch.tensor(size_label, dtype=torch.long).to(inputs_embeds.device)) + \
                                    self.BCELoss(available_sizes, torch.tensor(available_sizes_label, dtype=torch.float32).to(inputs_embeds.device)) + \
                                    self.CELoss(brand, torch.tensor(brand_label, dtype=torch.long).to(inputs_embeds.device)) + \
                                    self.CELoss(sleeve_length, torch.tensor(sleeve_length_label, dtype=torch.long).to(inputs_embeds.device)) + \
                                    self.CELoss(asset_type, torch.tensor(asset_type_label, dtype=torch.long).to(inputs_embeds.device)) + \
                                    self.CELoss(price, torch.tensor(price_label, dtype=torch.long).to(inputs_embeds.device)) + \
                                    self.CELoss(customer_review, torch.tensor(customer_review_label, dtype=torch.long).to(inputs_embeds.device))
            else:
                coref, brand, color, materials, type_, price, customer_review = self.furniture_enc_head(hidden_concat)  # (num_obj, num_logits)
                
                loss_per_line = 8 * self.CELoss(coref, torch.tensor(coref_label, dtype=torch.long).to(inputs_embeds.device)) + \
                                    self.CELoss(color, torch.tensor(color_label, dtype=torch.long).to(inputs_embeds.device)) + \
                                    self.CELoss(materials, torch.tensor(materials_label, dtype=torch.long).to(inputs_embeds.device)) + \
                                    self.CELoss(type_, torch.tensor(type_label, dtype=torch.long).to(inputs_embeds.device)) + \
                                    self.CELoss(brand, torch.tensor(brand_label, dtype=torch.long).to(inputs_embeds.device)) + \
                                    self.CELoss(price, torch.tensor(price_label, dtype=torch.long).to(inputs_embeds.device)) + \
                                    self.CELoss(customer_review, torch.tensor(customer_review_label, dtype=torch.long).to(inputs_embeds.device))

            misc_loss += loss_per_line
        misc_loss /= batch_size

        return misc_loss


    def evaluate(self, enc_input, enc_attention_mask, boxes, misc, disam_label = None):
        '''  评测任务1在VLBERT模型上的表现效果'''

        batch_size = len(misc)
        
        inputs_embeds = self.encoder.embeddings(enc_input)
        
        for b_idx in range(batch_size):  # in a batch
            box_embedded = self.box_embedding(torch.tensor(boxes[b_idx]).to(inputs_embeds.device))  # (num_obj_per_line, d_model)
            for obj_idx in range(len(misc[b_idx])):
                pos = misc[b_idx][obj_idx]['pos']
                inputs_embeds[b_idx][pos] += box_embedded[obj_idx]
        
        enc_last_state = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=enc_attention_mask
        ).last_hidden_state    
        
        # disam_all_pred = self.disam_all_head(enc_last_state[:, 2, :]).argmax(dim=1).tolist()
        # disam_type_pred = self.disam_type_head(enc_last_state[:, 3, :]).argmax(dim=1).tolist()
        
        n_true_objects, n_pred_objects, n_correct_objects = 0, 0, 0
        
        output_pred_list = []
        
        for b_idx in range(batch_size):  # in a batch
            
            if disam_label is not None and disam_label[b_idx] == 0:
                output_pred_list.append([])
                continue
                
            for obj_idx in range(len(misc[b_idx])):
                pos = misc[b_idx][obj_idx]['pos']
                
                if obj_idx == 0:
                    hidden_concat = torch.reshape(enc_last_state[b_idx][pos:pos+2], (1, -1)) # hidden_concat: (num_obj, 2*model)
                else:
                    hidden_concat = torch.cat([hidden_concat, torch.reshape(enc_last_state[b_idx][pos:pos+2], (1, -1))], dim=0)

            is_fashion = misc[b_idx][0]['is_fashion']
            
            coref_label = torch.tensor([misc[b_idx][obj_idx]['coref_label'] for obj_idx in range(len(misc[b_idx]))]).to(inputs_embeds.device)  # (num_obj)  0 or 1
            n_true_objects += coref_label.sum().item()
            
                
            if is_fashion:
                coref, size, available_sizes, brand, color, pattern, sleeve_length, asset_type, type_, price, customer_review = self.fashion_enc_head(hidden_concat)  # (num_obj, num_logits)
                n_pred_objects += coref.argmax(dim=1).sum().item()
                n_correct_objects += torch.logical_and(coref.argmax(dim=1), coref_label).int().sum().item() # 1. or 0.
            else:
                coref, brand, color, materials, type_, price, customer_review = self.furniture_enc_head(hidden_concat)  # (num_obj, num_logits)
                n_pred_objects += coref.argmax(dim=1).sum().item()
                n_correct_objects += torch.logical_and(coref.argmax(dim=1), coref_label).int().sum().item() # 1. or 0.
            
            # 记录预测的Object List
            pred_coref = coref.argmax(dim=1).squeeze().tolist()
            pred_object_index = []
            for obj_idx in range(len(misc[b_idx])):
                if pred_coref[obj_idx] == 1:
                    pred_object_index.append(misc[b_idx][obj_idx]['object_index'])
            output_pred_list.append(pred_object_index)
            
        return  n_true_objects, n_pred_objects, n_correct_objects, output_pred_list

    