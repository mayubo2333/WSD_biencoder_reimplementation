import os
import re
import torch
import random
import numpy as np


pos_converter = {'NOUN':'n', 'PROPN':'n', 'VERB':'v', 'AUX':'v', 'ADJ':'a', 'ADV':'r'}

#copy from https://github.com/facebookresearch/wsd-biencoders/blob/aee2f0ae985ffbadeb37192b018592dbfa8db494/wsd_models/util.py#L18
def generate_key(lemma, pos):
    if pos in pos_converter.keys():
        pos = pos_converter[pos]
    key = '{}+{}'.format(lemma, pos)
    return key


# copy from https://github.com/facebookresearch/wsd-biencoders/blob/aee2f0ae985ffbadeb37192b018592dbfa8db494/wsd_models/util.py#L115
def load_data(datapath, name):
    text_path = os.path.join(datapath, '{}.data.xml'.format(name))
    gold_path = os.path.join(datapath, '{}.gold.key.txt'.format(name))

    #load gold labels 
    gold_labels = {}
    with open(gold_path, 'r', encoding="utf8") as f:
        for line in f:
            line = line.strip().split(' ')
            instance = line[0]
            #this means we are ignoring other senses if labeled with more than one 
            #(happens at least in SemCor data)
            key = line[1]
            gold_labels[instance] = key

    #load train examples + annotate sense instances with gold labels
    sentences = []
    s = []
    with open(text_path, 'r', encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line == '</sentence>':
                sentences.append(s)
                s=[]
                	
            elif line.startswith('<instance') or line.startswith('<wf'):
                word = re.search('>(.+?)<', line).group(1)
                lemma = re.search('lemma="(.+?)"', line).group(1) 
                pos =  re.search('pos="(.+?)"', line).group(1)

                #clean up data
                word = re.sub('&apos;', '\'', word)
                lemma = re.sub('&apos;', '\'', lemma)
                
                sense_inst = -1
                sense_label = -1
                if line.startswith('<instance'):
                    sense_inst = re.search('instance id="(.+?)"', line).group(1)
                    #annotate sense instance with gold label
                    sense_label = gold_labels[sense_inst]
                s.append((word, lemma, pos, sense_inst, sense_label))
    
    return sentences


# copy from https://github.com/facebookresearch/wsd-biencoders/blob/aee2f0ae985ffbadeb37192b018592dbfa8db494/wsd_models/util.py#L50
def load_wn_senses(path):   
    wn_senses = {}
    with open(path, 'r', encoding="utf8") as f:
        for line in f:
            line = line.strip().split('\t')
            lemma = line[0]
            pos = line[1]
            senses = line[2:]
            
            key = generate_key(lemma, pos)
            wn_senses[key] = senses
    return wn_senses


def padding_sent(input_ids, attention_mask, max_length, pad_token_id, pad_mask_token_id=0, assert_message="Padding failed"):
    if len(input_ids)> max_length:
        raise AssertionError(assert_message)
    
    input_ids.extend([pad_token_id]*(max_length-len(input_ids)))
    attention_mask.extend([pad_mask_token_id]*(max_length-len(attention_mask)))
        
    return input_ids, attention_mask


def generate_word_id_to_char_id(word_list):
    word_id_to_char_id = list()
    curr = 0
    for word in word_list:
        word_id_to_char_id.append(curr)
        curr += len(word)+1
    return word_id_to_char_id


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    
def get_pred_label(score_list):
    pred_list = list()
    for score in score_list:
        score = np.array(score)
        pred_list.append(score.argmax())
    return pred_list


def log_sum_exp(scores):
    assert(scores.dim()==1)
    s_max = scores.max()
    return s_max + (scores-s_max).exp().sum().log()


def test_load_data():
    file_path = "./WSD_Evaluation_Framework/Training_Corpora/SemCor"
    name = "semcor"
    sentences = load_data(file_path, name)
    return sentences


def test_load_wn_senses():
    file_path = "./WSD_Evaluation_Framework/Data_Validation/candidatesWN30.txt"
    wn_senses = load_wn_senses(file_path)
    return wn_senses
    

if __name__ == "__main__":
    # # test load_data
    # sentences = test_load_data()
    # print(sentences[56])
    
    # test load_wn_senses
    wn_senses = test_load_wn_senses()
    print(wn_senses['love+v'])