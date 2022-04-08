import numpy as np
import torch
import time
from collections import defaultdict


class SentenceData:
    """
    Thic class contains data for a tagged sentence
    
    Attributes
    ----------
    subwords : numpy array of strings
        The subwords of a sentence. Mostly whole words, but punctuation marks as well
    pos_tag_nums: numpy array of integers
        The numbers representing part-of-speech tags. These numbers are read from a dictionary.
    chunk_tags: numpy array of strings
        Abbreviations for the chunk tag of a given subword.
    
    """
    
    def __init__(self, sentence_str, tag_dict):
        """
        Arguments:
        sentence_str : string
            A multi-line string which containing words and their pos nad chung tags.
        """
        lines = sentence_str.splitlines()
        n = len(lines)
        subwords, pos_tags, chunk_tags = [], [], []
        
        for l in lines:
            sw, pos_tag, ch_tag = l.split()
            
            subwords.append(sw)
            pos_tags.append(pos_tag)
            chunk_tags.append(ch_tag)
            
        self.subwords = np.array(subwords)
        self.pos_tag_nums = np.array(list(map(lambda x: tag_dict[x], pos_tags)))
        self.chunk_tags = np.array(chunk_tags)
        
    def __str__(self):
        return '_'.join(self.subwords)
        
    def get_subwords(self):
        return self.subwords
    
    def get_tag_nums(self):
        return self.pos_tag_nums
    
    def chunk_tags(self):
        return self.chunk_tags
    
    def tokenize(self, tokenizer):
        """
        Arguments:
        tokenizer
            A BERT tokenizer.
        
        Returns two values: 
            --  a list of tokens produced by the tokenizer.
                The  first element is 101 <- [CLS], the last element is [SEP] indicating the sentence boundaries.
                These tokens were not in the dataset, but we included their numerical value, because BERT uses them.
            --  a list of integers depicting which token corresponds to which word in the sentence.
                If the i-th token t_i = j, it means the i-th token corresponds to the j-th token in the sentence.
                It is needed, since the tokenizer may separate a word into multiple subwords each one with a token
                (if the word was not in the vocabulary of he tokenizer).
                The  first and last elements are None, since the tokenizer uses a special token which indicates the start and end of the sentence.
        """
        tokens = [101]
        #numbers which word the given token corresponds to
        token_word_idx = [None]
        
        for i, w in enumerate(self.subwords):
            cur_tokens = tokenizer.encode(w, is_split_into_words=False, add_special_tokens=False)
            tokens+=cur_tokens
            for j in range(len(cur_tokens)):
                token_word_idx.append(i)
                
        tokens.append(102)
        token_word_idx.append(None)
        
        return tokens, token_word_idx
            
        

#tags = {'CC':0,'CD':1,'DT':2,'EX':3,'FW':4,'IN':5,'JJ':6,'JJR':7,'JJS':8,'LS':9,'MD':10,'NN':11,'NNS':12,'NNP':13,'NNPS':14,'PDT':15,'POS':16,'PRP':17,'PRP$':18,'RB':19,'RBR':20,'RBS':21,'RP':22,'SYM':23,'TO':24,'UH':25,'VB':26,'VBD':27,'VBG':28,'VBN':29,'VBP':30,'VBZ':31,'WDT':32,'WP':33,'WP$':34,'WRB':35}

#https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
tag_list = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']

def get_tag_dict(tag_list):
    tag_dict = defaultdict(lambda: 0)
    for i, tag in enumerate(tag_list):
        #start from 1, misc. tags get the number 0
        tag_dict[tag] = i+1
        
    return tag_dict
        

def get_tags(file_path):
    """Extract all types of POS tags and chunk tags used in the file.
    
    Arguments:
    file_path -- string form for the file path where the tagged data are
    
    The sentences are separated with empty lines. Format example:
        Unilab NNP B-NP
        new JJ B-NP
        markets NNS I-NP
        . . O

        In IN B-PP
        Los NNP B-NP
    
    Useful descriptions:
    https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    https://arxiv.org/pdf/cs/0009008.pdf
    """
    pos_tags = set()
    chunk_tags = set()
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for l in list(filter(lambda x: x.strip() != "", lines)):
            parts = l.split()
            if (parts[2] != 'O'):
                pos_tags.add(parts[1])
                chunk_tags.add(parts[2])
            
    print(f"POS tags found: {pos_tags}")
    print(f"Chunk tags found: {sorted(list(chunk_tags))}")
    
    return pos_tags, chunk_tags

def onehot_encode(n, size):
    out = np.zeros(size)
    out[n] = 1
    return np.squeeze(out)

#THE PENN TREEBANK: AN OVERVIEW
#http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.8216&rep=rep1&type=pdf
#Chunking - description
#http://www.iro.umontreal.ca/~felipe/IFT6010-Automne2001/conll.html
def get_sentences(file_path):
    """
    
    """
    
    sentences = []
    tags = get_tag_dict(tag_list)
    
    with open(file_path, 'r') as f:
            
        data_set = f.read()
        
        tagged_sentences = data_set.split("\n\n")
        
        for sentence_str in tagged_sentences:
            sentences.append(SentenceData(sentence_str, tags))
                
    return sentences
    
    
#https://stackoverflow.com/questions/63413414/is-there-a-way-to-get-the-location-of-the-substring-from-which-a-certain-token-h
def test_sentence(sentence_data):
    
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
    model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
    config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-cased', output_attention=True, foo=False)
    
    word_list = sentence_data.get_subwords()
    tokens, ids = sentence_data.tokenize(tokenizer)

    #https://pytorch.org/docs/stable/generated/torch.no_grad.html
    segments_ids = [1]*len(tokens)  
 
    print(f"tokens: {tokens} ids: {ids}")
    
    tokens_tensor = torch.tensor([tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    with torch.no_grad():
        
        #outputs =  model(tokens_tensor,segments_tensors,  output_hidden_states=True)
        outputs =  model(tokens_tensor,token_type_ids = None,  output_hidden_states=True)
        print(f"type of outputs.hidden_states : {type(outputs.hidden_states)} -> length: {len(outputs.hidden_states)}")
        #print(outputs.hidden_states)
        print("hidden states: ")
        for i, s in enumerate(outputs.hidden_states):
            print(f"state no. {i} -> type: {type(s)} -> shape: {s.shape}]")

        w = outputs.hidden_states[0][0][0][0]
        print(f"{ w } -> {type(w)}")
        
#https://discuss.huggingface.co/t/output-attention-true-after-downloading-a-model/907
def produce_hidden_states(layer_num, sentences, tokenizer,  model, first_sentence=0):
    
    saved_data = []
    
    with torch.no_grad():
        for sentence_num, sentence in enumerate(sentences):
            if ((sentence_num+first_sentence)%100 == 0):
                print(f"Sentence noumber: {sentence_num+first_sentence}")
            word_list = sentence.get_subwords()
            tokens, token_word_ids = sentence.tokenize(tokenizer)
            #print(f"token_word_ids num: {token_word_ids}")
            pos_tags = sentence.get_tag_nums()
            
            tokens_tensor = torch.tensor([tokens])
        
            outputs =  model(tokens_tensor,token_type_ids = None,  output_hidden_states=True)
            
            #print(f"token_word_ids: {token_word_ids}")
            #print(f"pos_tags: {list(pos_tags)}")
            for token_num, word_num in enumerate(token_word_ids):
                subword_attentions = []
                for layer_num, state in enumerate(outputs.hidden_states):
                
                
                    
                    
                    attention = state[0][token_num]
                    
                    subword_attentions.append(attention)
                    
                #not saving 'start tokens' and 'end tokens'
                
                if word_num is not None:
                    #print(f"layer num: {layer_num} state shape: {state.shape}")
                    saved_data.append((sentence_num+first_sentence, word_num, pos_tags[word_num], subword_attentions))
            #print(f"token_word_ids num: {token_word_ids}")
            
    #print(f"saved_data: {saved_data}")
    return saved_data
    
    #print(len(word_list), len(tokens))
    #print(word_list)
    #print(tokens)

#https://pageperso.lis-lab.fr/benoit.favre/pstaln/09_embedding_evaluation.html

#def pos_tag_data():
#    """
#    """
#def generate_embeddings():

def write_hidden_states(data, dest_folder, txt_format=False):

    with open(f"{dest_folder}/word_data.txt", 'w') as save_file:
        for sw_d in data:
            sentence_num, word_num, pos_tag, hidden_states = sw_d
            #print(f"{sentence_num} {word_num} {pos_tag}")
            save_file.write(f"{sentence_num} {word_num} {pos_tag}\n")
    
    if txt_format:
        
        for layer_num in range(13):
            path = f"{dest_folder}/hidden_states_{layer_num:02}.txt"
            with open(path, 'a') as txt_file:
                for i, sw_d in enumerate(data):
                    hidden_state = sw_d[3][layer_num]
                    txt_file.writeln(np.array(hidden_state))

            
    else:
        for layer_num in range(13):
            layer_dict = {}
            for i, sw_d in enumerate(data):
                hidden_state = sw_d[3][layer_num]
                layer_dict[i] = hidden_state
            torch.save(layer_dict, f"{dest_folder}/hidden_states_{layer_num:02}.pt")
        
def read_hidden_states(layer_num, source_folder):
    """Read hidden states for given layer from file.
    
    Arguments:
    layer_num -- number of the layer (for 0 the layer is the initial embedding)
    """
    states = torch.load(f"{source_folder}/hidden_states_{layer_num:02}.pt")
    
    return states
        
    


get_tags("conll2000/train.txt")


"""
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-cased', output_attention=True)
model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased', config=config)


sentences = get_sentences("conll2000/test.txt")
print(f"number of sentences: {len(sentences)}")

print("Start measuring time")
start = time.time()


dt = produce_hidden_states(sentences, tokenizer, model)
write_hidden_states(dt, "test_data")
end = time.time()
print(end - start)
"""



tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-cased', output_attention=True)
model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased', config=config)

sentences = get_sentences("conll2000/train.txt")

print(f"number of sentences: {len(sentences)}")
print("Start measuring time")
start = time.time()

dt = []
for i in range(5):
    dt+=produce_hidden_states(sentences[i*2000:(i+1)*2000], tokenizer, model)
write_hidden_states(dt, "train_data")
end = time.time()
print(end - start)

#print(dt)

#TODO save to 12 different files using torhc save
