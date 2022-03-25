import numpy as np
import torch


#tags = {'CC':0,'CD':1,'DT':2,'EX':3,'FW':4,'IN':5,'JJ':6,'JJR':7,'JJS':8,'LS':9,'MD':10,'NN':11,'NNS':12,'NNP':13,'NNPS':14,'PDT':15,'POS':16,'PRP':17,'PRP$':18,'RB':19,'RBR':20,'RBS':21,'RP':22,'SYM':23,'TO':24,'UH':25,'VB':26,'VBD':27,'VBG':28,'VBN':29,'VBP':30,'VBZ':31,'WDT':32,'WP':33,'WP$':34,'WRB':35}

#https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
tag_list = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']

def get_tag_dict(tag_list):
    tag_dict = dict()
    for i, tag in enumerate(tag_list):
        #start from 1
        tag_dict[tag] = i+1
        
    return tag_dict
        
tags = get_tag_dict(tag_list)
        

def get_tags(file_path):
    """
    Extract all types of POS tags and chunk tags used in the file
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

def onehot_encode(n, size):
    out = np.zeros(size)
    out[n] = 1
    return np.squeeze(out)

#http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.8216&rep=rep1&type=pdf
#http://www.iro.umontreal.ca/~felipe/IFT6010-Automne2001/conll.html
def get_sentences(file_path):
    sentences = []
    
    cur_words = []
    cur_pos_tags = []
    cur_pos_tags_encoded = []
    sentence_str = ""
    
    
    with open(file_path, 'r') as f:
            
        lines = f.readlines()
        
        for l in lines:
            if l.strip() == "":
                sentences.append((sentence_str, cur_words, cur_pos_tags, cur_pos_tags_encoded))
                cur_words = []
                cur_pos_tags = []
                cur_pos_tags_encoded = []
                sentence_str = ""
            else:
                #print(l)
                word, pos_tag, chunk_tag = l.split()
                if (sentence_str != "" and word not in [",", "."]):
                    sentence_str+=" "
                sentence_str+=word
                
                cur_words.append(word)
                cur_pos_tags.append(pos_tag)
                
                
                if(pos_tag not in tags):
                    encoded_tag = onehot_encode(0, 37)
                else:
                    encoded_tag = onehot_encode(tags[pos_tag], 37)
                    
                cur_pos_tags_encoded.append(encoded_tag)
                
    return sentences
            
    print(f"POS tags found: {pos_tags}")
    print(f"Chunk tags found: {sorted(list(chunk_tags))}")
    
    
#https://stackoverflow.com/questions/63413414/is-there-a-way-to-get-the-location-of-the-substring-from-which-a-certain-token-h
def test_sentence(sentence_data):
    
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
    model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
    config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-cased', output_attention=True, foo=False)
    
    word_list = sentence_data[1]
    #indexed_tokens = tokenizer.encode(word_list, is_split_into_words=True, add_special_tokens=True)
    
    #special token - sentence start
    tokens = [[101]]
    
    for w in word_list:
        cur_tokens = tokenizer.encode(w, is_split_into_words=False, add_special_tokens=False)
        tokens.append(cur_tokens)
        
    #special token - sentence end
    tokens.append([102])
    
    print(len(word_list), len(tokens))
    print(word_list)
    print(tokens)

#https://pageperso.lis-lab.fr/benoit.favre/pstaln/09_embedding_evaluation.html

#def pos_tag_data():
#    """
#    """


get_tags("conll2000/train.txt")

#TODO load train pos_tag_data

sentences = get_sentences("conll2000/test.txt")


print(sentences[10])
print("<><><><><><><><><><><><>")
test_sentence(sentences[0])
