import os
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from utils import log_sum_exp


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
        
        text_encoder_outputs = self.text_encoder(sent_token_ids, sent_mask_ids)[0]
        gloss_encoder_outputs = self.gloss_encoder(all_sense_token_ids, all_sense_mask_ids)[0]
        
        total_loss = list()
        score_list = list()
        for i in range(sent_token_ids.size(0)):
            text_output = text_encoder_outputs[i]         # [L, H]
            gloss_output = gloss_encoder_outputs[first_sense_locs[i]:first_sense_locs[i+1]]         # [N, L, H]
            
            w = torch.mean(text_output[start_locs[i]:end_locs[i]], dim=0).unsqueeze(-1)       # [H, 1]
            s = gloss_output[:, 0, :]                                           # [N, H]
            if self.config.normalize:
                s = F.normalize(s, dim=1)
                w = F.normalize(w, dim=0)
            if not self.config.loss_self:
                score = torch.mm(s, w).squeeze(-1).unsqueeze(0)     #[1, N]  
            else:
                score = torch.mm(s, w).squeeze(-1)         
            
            if labels is not None:
                if not self.config.loss_self:
                    label = labels[i].unsqueeze(0)
                    loss = self.fct(score, label)
                else:
                    loss = -score[labels[i]] + log_sum_exp(score)
                total_loss.append(loss)
            score_list.append(score.cpu())
            

        if total_loss:
                return torch.mean(torch.stack(total_loss)), score_list
        else:
            return score_list
        
        
    def save(self):
        text_model_output_dir = os.path.join(self.config.output_dir, 'checkpoint_text')
        os.makedirs(text_model_output_dir, exist_ok=True)
        self.text_encoder.save_pretrained(text_model_output_dir)
        
        gloss_model_output_dir = os.path.join(self.config.output_dir, 'checkpoint_gloss')
        os.makedirs(gloss_model_output_dir, exist_ok=True)
        self.gloss_encoder.save_pretrained(gloss_model_output_dir)