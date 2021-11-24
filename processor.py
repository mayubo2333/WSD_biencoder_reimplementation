import os
import ipdb
import logging
from tqdm import tqdm
from nltk.corpus import wordnet as wn

import torch
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from utils import generate_key, generate_word_id_to_char_id, load_data, load_wn_senses, padding_sent
from utils import test_load_data

logger = logging.getLogger(__name__)


class Sentence:
    def __init__(self, sent):
        self.word_list = list()
        self.lemma_list = list()
        self.pos_list = list()
        self.sense_dict = None
        self.index_dict = None
        self._parse_sent(sent)
    
    
    def _parse_sent(self, sent):
        idx_list, sense_list = list(), list()
        
        for word in sent:
            self.word_list.append(word[0])
            self.lemma_list.append(word[1])
            self.pos_list.append(word[2])
            idx_list.append(word[3])
            sense_list.append(word[4])
   
        self.sense_dict = {idx:sense for idx, sense in zip(idx_list, sense_list) if idx!=-1} 
        self.index_dict = {idx:i for i, idx in enumerate(idx_list) if idx!=-1}
        
        assert(-1 not in self.sense_dict and -1 not in self.index_dict)
        assert("-1" not in self.sense_dict and "-1" not in self.index_dict)
        
        
    def __repr__(self):
        s = ''
        s += 'word_list\n'; s += str(self.word_list); s += '\n'
        s += 'lemma_list\n'; s += str(self.lemma_list); s += '\n'
        s += 'pos_list\n'; s += str(self.pos_list); s += '\n'
        s += 'sense_dict\n'; s += str(self.sense_dict); s += '\n'
        s += 'index_dict\n'; s += str(self.index_dict); s += '\n'
        return s


class Feature:
    def __init__(
        self,
        idx=None,
        sense_lemma=None,
        sent_text=None,
        sent_token_ids=None,
        sent_mask_ids=None,
        sense_text_list=None,
        sense_token_ids_list=None,
        sense_mask_ids_list=None,
        location=None,
        label=None,
        sense_name_list=None,
    ):
        self.idx = idx
        self.sense_lemma = sense_lemma
        self.sent_text = sent_text
        self.sent_token_ids = sent_token_ids
        self.sent_mask_ids = sent_mask_ids
        self.sense_text_list = sense_text_list
        self.sense_token_ids_list = sense_token_ids_list
        self.sense_mask_ids_list = sense_mask_ids_list
        self.location = location
        self.label = label
        self.sense_name_list = sense_name_list
    
    
    def __repr__(self):
        s = ''
        s += 'idx:'; s += str(self.idx); s += '\n'
        s += 'sense_lemma:'; s += str(self.sense_lemma); s += '\n'
        s += 'sent_text\n'; s += str(self.sent_text); s += '\n'
        s += 'sent_token_ids\n'; s += str(self.sent_token_ids); s += '\n'
        s += 'sent_mask_ids\n'; s += str(self.sent_mask_ids); s += '\n'
        s += 'sense_text_list\n'; s += str(self.sense_text_list); s += '\n'
        s += 'sense_token_ids_list\n'; s += str(self.sense_token_ids_list); s += '\n'
        s += 'sense_mask_ids_list\n'; s += str(self.sense_mask_ids_list); s += '\n'
        s += 'start:{}\tend:{}\n'.format(self.location[0], self.location[1])
        s += 'label:'; s += str(self.label); s += '\n'
        return s
    

class WSD_Dataset(Dataset):
    def __init__(self, features):
        super().__init__()
        self.features = features
        
        
    def __len__(self):
        return len(self.features)
    
    
    def __getitem__(self, idx):
        return self.features[idx]
 
 
    def collate_fn(self, batch):
        sent_token_ids = torch.LongTensor([f.sent_token_ids for f in batch])
        sent_mask_ids = torch.LongTensor([f.sent_mask_ids for f in batch])
        start_locs = torch.LongTensor([f.location[0] for f in batch])
        end_locs = torch.LongTensor([f.location[1] for f in batch])
        labels = torch.LongTensor([f.label for f in batch])
        
        sense_token_ids_list = [torch.LongTensor(f.sense_token_ids_list) for f in batch]
        sense_mask_ids_list = [torch.LongTensor(f.sense_mask_ids_list) for f in batch]

        return sent_token_ids, sent_mask_ids, \
                start_locs, end_locs, labels, \
                sense_token_ids_list, sense_mask_ids_list

    
class WSD_Processor:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        
        wn_sense_dict = load_wn_senses(self.args.wn_path)
        self.wn_sense_dict = dict()
        logger.info("Generate sense gloss information")
        for key, sense_list in tqdm(wn_sense_dict.items()):
            self.wn_sense_dict[key] = list()
            for sense in sense_list:
                definition = wn.lemma_from_key(sense).synset().definition()
                self.wn_sense_dict[key].append((sense, definition))
                     
    
    def create_examples(self, sent_list):
        examples = []
        logger.info("Create examples")
        for sent in tqdm(sent_list):
            example = Sentence(sent)
            examples.append(example)
        logger.info("Create {} examples in all".format(len(examples))) 
        return examples
    
    
    def convert_examples_to_features(self, examples):
        features = []
        logger.info("Convert features")
        for example in tqdm(examples):
            sent_text = " ".join(example.word_list)
            word_id_to_char_id = generate_word_id_to_char_id(example.word_list)
            text_enc = self.tokenizer(sent_text)
            try:
                sent_token_ids, sent_mask_ids = padding_sent(
                    text_enc["input_ids"], text_enc["attention_mask"],
                    self.args.max_text_length, self.tokenizer.pad_token_id,
                    )
            except:
                logger.info("Encoded text with length {} exceeds max length constraints!".format(len(text_enc["input_ids"])))
                continue
            
            for idx in example.sense_dict:
                sense = example.sense_dict[idx]
                word_loc = example.index_dict[idx]
                char_loc = word_id_to_char_id[word_loc]
                next_char_loc = word_id_to_char_id[word_loc+1] if word_loc<len(word_id_to_char_id)-1 else -1
                
                lemma, pos = example.lemma_list[word_loc], example.pos_list[word_loc]
                sense_candidate_list = self.wn_sense_dict[generate_key(lemma, pos)]
                sense_name_list = [sense_candidate[0] for sense_candidate in sense_candidate_list]
                sense_text_list = [sense_candidate[1] for sense_candidate in sense_candidate_list]
                label = sense_name_list.index(sense)
            
                start = text_enc.char_to_token(char_loc)
                end = text_enc.char_to_token(next_char_loc) if next_char_loc >=0 else len(sense_token_ids)
                location = (start, end)
                
                sense_token_ids_list, sense_mask_ids_list = list(), list()
                for sense_text in sense_text_list:
                    sense_enc = self.tokenizer(sense_text)
                    sense_token_ids, sense_mask_ids = padding_sent(
                        sense_enc["input_ids"], sense_enc["attention_mask"],
                        self.args.max_gloss_length, self.tokenizer.pad_token_id,
                    )       # Glosses are relatively short. No need to check length.
                    sense_token_ids_list.append(sense_token_ids)
                    sense_mask_ids_list.append(sense_mask_ids)
                
                feature = Feature(idx=idx, sense_lemma=lemma,\
                    sent_text=sent_text, sent_token_ids=sent_token_ids, sent_mask_ids=sent_mask_ids,
                    sense_text_list=sense_text_list, sense_token_ids_list=sense_token_ids_list, sense_mask_ids_list=sense_mask_ids_list,
                    location=location, label=label, sense_name_list=sense_name_list
                    ) 
                features.append(feature)

        logger.info("Convert {} features in all".format(len(features)))
        return features
    
    
    def convert_features_to_dataset(self, features):
        dataset = WSD_Dataset(features)
        return dataset
    
    
    def generate_dataloader(self, set_type):
        assert (set_type in ['train', 'dev', 'test'])
        if set_type=='train':
            file_path = self.args.train_file
            file_name = self.args.train_dataset_name
        elif set_type=='dev':
            file_path = self.args.dev_file
            file_name = self.args.dev_dataset_name
        else:
            file_path = self.args.test_file
            file_name = self.args.test_dataset_name
        
        sent_list = load_data(file_path, file_name)
        examples = self.create_examples(sent_list)
        features = self.convert_examples_to_features(examples)
        dataset = self.convert_features_to_dataset(features)
        logger.info("{} {} samples are prepared.".format(len(dataset), set_type))
        if set_type != 'train':
            # Note that DistributedSampler samples randomly
            dataset_sampler = SequentialSampler(dataset)
        else:
            dataset_sampler = RandomSampler(dataset)
        batch_size = self.args.train_batch_size if set_type=='train' else self.args.infer_batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=dataset_sampler, collate_fn=dataset.collate_fn)
        # ipdb.set_trace()
        return examples, features, dataloader
                   
    
    def test_create_examples(self):
        sent_list = test_load_data()
        examples = self.create_examples(sent_list)
        return examples
    
    
    def test_convert_features(self):
        examples = self.test_create_examples()
        features = self.convert_examples_to_features(examples[:1000])
        
        def test_single_feature(feature):
            print('----------------------')
            print(feature)
            start, end = feature.location
            print("Check sense location: {}, {}\n".format(
                self.tokenizer.decode(feature.sent_token_ids[start:end]),
                feature.sense_lemma,
            ))
            print("Check sense retrieval:{}\n".format(feature.sense_text_list))
            print("Check sense encode:")
            for sense_token in feature.sense_token_ids_list:
                print(self.tokenizer.decode(sense_token))
            print('-----------------------')
             
        # ipdb.set_trace()
        for random_id in [130]:
            test_single_feature(features[random_id])    
       
       
if __name__ == "__main__":
    from easydict import EasyDict as edict
    args = edict(
        train_file="./WSD_Evaluation_Framework/Training_Corpora/SemCor",
        train_dataset_name='semcor',
        wn_path="./WSD_Evaluation_Framework/Data_Validation/candidatesWN30.txt",
        max_text_length=256,
        max_gloss_length=128,
        batch_size=4,
    )
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    
    processor = WSD_Processor(args, tokenizer)
    # processor.test_convert_features()
    processor.generate_dataloader('train')