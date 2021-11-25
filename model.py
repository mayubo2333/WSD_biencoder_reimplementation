import os
import ipdb
import torch
import torch.nn as nn
from transformers import BertModel

from tqdm import tqdm
from WSD_biencoder.utils import padding_sent


class BertForWSD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_encoder = BertModel.from_pretrained(config.model_name_or_path)
        self.gloss_encoder = BertModel.from_pretrained(config.model_name_or_path)
        self.fct = nn.CrossEntropyLoss()
        
    
    def load(self, path):
        new_state_dict = torch.load(path)
        old_state_dict = self.state_dict()
        for name in old_state_dict:
            if name not in new_state_dict:
                print("{} is not initialized in model".format(name))
            else:
                old_state_dict[name] = new_state_dict[name]
        self.load_state_dict(old_state_dict)
        
    
    def save(self):
        torch.save(self.state_dict(), os.path.join(self.config.output_dir, 'best_model.ckpt'))
         
        
    def forward(
        self,
        sent_token_ids,
        sent_mask_ids,
        start_locs,
        end_locs,
        sense_token_ids_list,
        sense_mask_ids_list,
        labels=None,
        key_list=None,
    ):
        
        text_encoder_outputs = self.text_encoder(sent_token_ids, sent_mask_ids)[0]
        
        if key_list is None:
            candidate_sense_nums = [sense_token_ids.size(0) for sense_token_ids in sense_token_ids_list]
            first_sense_locs = list()
            curr = 0
            for sense_num in candidate_sense_nums:
                first_sense_locs.append(curr)
                curr += sense_num
            first_sense_locs.append(curr)
            
            all_sense_token_ids = torch.cat(sense_token_ids_list, dim=0)
            all_sense_mask_ids = torch.cat(sense_mask_ids_list, dim=0)
            gloss_encoder_outputs = self.gloss_encoder(all_sense_token_ids, all_sense_mask_ids)[0]
        
        total_loss = list()
        score_list = list()
        for i in range(sent_token_ids.size(0)):
            text_output = text_encoder_outputs[i]         # [L, H]
            w = torch.mean(text_output[start_locs[i]:end_locs[i]], dim=0).unsqueeze(-1)       # [H, 1]
            
            if key_list is None:
                gloss_output = gloss_encoder_outputs[first_sense_locs[i]:first_sense_locs[i+1]]         # [N, L, H]
                s = gloss_output[:, 0, :]                                           # [N, H]
            else:
                key = key_list[i]
                s = self.gloss_embedding[key].to(self.config.device)
            
            score = torch.mm(s, w).squeeze(-1).unsqueeze(0)     #[1, N]  
     
            if labels is not None:
                label = labels[i].unsqueeze(0)
                loss = self.fct(score, label)
                total_loss.append(loss)
            score_list.append(score.cpu()) 

        if total_loss:
            return torch.mean(torch.stack(total_loss)), score_list
        else:
            return score_list
        
        
    def load_gloss_embedding(self, path):
        self.gloss_embedding = torch.load(path)
        
        
    def cache_gloss_embedding(self, processor, output_path):
        sense_token_ids_dict = dict()
        sense_attention_mask_dict = dict()
        for key, sense_list in tqdm(processor.wn_sense_dict.items()):
            sense_token_ids_dict[key] = list()
            sense_attention_mask_dict[key] = list()
            
            for _, definition, _ in sense_list:
                sense_enc = processor.tokenizer(definition)
                sense_token_ids, sense_attention_mask = padding_sent(
                    sense_enc["input_ids"], sense_enc["attention_mask"],
                    processor.args.max_gloss_length, processor.tokenizer.pad_token_id,
                )
                
                sense_token_ids_dict[key].append(sense_token_ids) 
                sense_attention_mask_dict[key].append(sense_attention_mask)
            
            sense_token_ids_dict[key] = torch.LongTensor(sense_token_ids_dict[key])
            sense_attention_mask_dict[key] = torch.LongTensor(sense_attention_mask_dict[key])
            
        gloss_embedding_dict = dict()
        for key in tqdm(processor.wn_sense_dict):
            input_ids = sense_token_ids_dict[key].to(self.config.device)
            attention_mask = sense_attention_mask_dict[key].to(self.config.device)
            gloss_encoder_output = self.gloss_encoder(input_ids, attention_mask)[0]
            gloss_embedding = gloss_encoder_output[:, 0, :]    #[N, H]
            gloss_embedding_dict[key] = gloss_embedding.detach().cpu()
            # del gloss_encoder_output, gloss_embedding
            torch.cuda.empty_cache()
            
        torch.save(gloss_embedding_dict, output_path)
             
        
class BertForWSD_Siamse(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = BertModel.from_pretrained(config.model_name_or_path)
        self.fct = nn.CrossEntropyLoss()
        
    
    def load(self, path):
        new_state_dict = torch.load(path)
        old_state_dict = self.state_dict()
        for name in old_state_dict:
            if name not in new_state_dict:
                print("{} is not initialized in model".format(name))
            else:
                old_state_dict[name] = new_state_dict[name]
        self.load_state_dict(old_state_dict)
        
    
    def save(self):
        torch.save(self.state_dict(), os.path.join(self.config.output_dir, 'best_model.ckpt'))
         
        
    def forward(
        self,
        sent_token_ids,
        sent_mask_ids,
        start_locs,
        end_locs,
        sense_token_ids_list,
        sense_mask_ids_list,
        labels=None,
    ):
        
        candidate_sense_nums = [sense_token_ids.size(0) for sense_token_ids in sense_token_ids_list]
        first_sense_locs = list()
        curr = 0
        for sense_num in candidate_sense_nums:
            first_sense_locs.append(curr)
            curr += sense_num
        first_sense_locs.append(curr)
         
        all_sense_token_ids = torch.cat(sense_token_ids_list, dim=0)
        all_sense_mask_ids = torch.cat(sense_mask_ids_list, dim=0)
        
        text_encoder_outputs = self.encoder(sent_token_ids, sent_mask_ids)[0]
        gloss_encoder_outputs = self.encoder(all_sense_token_ids, all_sense_mask_ids)[0]
        
        total_loss = list()
        score_list = list()
        for i in range(sent_token_ids.size(0)):
            text_output = text_encoder_outputs[i]         # [L, H]
            gloss_output = gloss_encoder_outputs[first_sense_locs[i]:first_sense_locs[i+1]]         # [N, L, H]
            
            w = torch.mean(text_output[start_locs[i]:end_locs[i]], dim=0).unsqueeze(-1)       # [H, 1]
            s = gloss_output[:, 0, :]                                           # [N, H]
            score = torch.mm(s, w).squeeze(-1).unsqueeze(0)     #[1, N]  
     
            if labels is not None:
                label = labels[i].unsqueeze(0)
                loss = self.fct(score, label)
                total_loss.append(loss)
            score_list.append(score.cpu())       

        if total_loss:
            return torch.mean(torch.stack(total_loss)), score_list
        else:
            return score_list      
            