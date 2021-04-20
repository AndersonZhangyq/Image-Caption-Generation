import torch
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu

from vocabulary import Vocabulary
from config import *

import string
from collections import OrderedDict
import operator

from sklearn.metrics.pairwise import cosine_similarity


def read_lines(filepath):
    """ Open the ground truth captions into memory, line by line. 
    Args:
        filepath (str): the complete path to the tokens txt file
    """
    file = open(filepath, 'r')
    lines = []

    while True: 
        # Get next line from file 
        line = file.readline() 
        if not line: 
            break
        lines.append(line.strip())
    file.close()
    return lines


def parse_lines(lines):
    """
    Parses token file captions into image_ids and captions.
    Args:
        lines (str list): str lines from token file
    Return:
        image_ids (int list): list of image ids, with duplicates
        cleaned_captions (list of lists of str): lists of words
    """
    image_ids = []
    cleaned_captions = []

    # QUESTION 1.1
    for line in lines:
        splited = line.split("\t")
        image_id = splited[0][:-6]
        caption = splited[1]
        caption = caption.translate(caption.maketrans(string.punctuation, ' '*len(string.punctuation)))
        caption = " ".join([word.lower() for word in caption.split() if word.isalpha()])
        image_ids.append(image_id)
        cleaned_captions.append(caption)


    return image_ids, cleaned_captions


def build_vocab(cleaned_captions):
    """ 
    Parses training set token file captions and builds a Vocabulary object
    Args:
        cleaned_captions (str list): cleaned list of human captions to build vocab with

    Returns:
        vocab (Vocabulary): Vocabulary object
    """

    # QUESTION 1.1
    # TODO collect words
    word_count = {}
    for caption in cleaned_captions:
        for word in caption.split():
            if word not in word_count:
                word_count[word] = 0
            word_count[word] += 1


    # create a vocab instance
    vocab = Vocabulary()

    # add the token words
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # TODO add the rest of the words from the cleaned captions here
    # vocab.add_word('word')
    for word, n in word_count.items():
        if n > 3:
            vocab.add_word(word)


    return vocab



def decode_caption(sampled_ids, vocab):
    """ 
    Args:
        sampled_ids (int list): list of word IDs from decoder
        vocab (Vocabulary): vocab for conversion
    Return:
        predicted_caption (str): predicted string sentence
    """
    predicted_caption = ""


    # QUESTION 2.1
    words = []
    pad_token_id = vocab.word2idx['<pad>']
    start_token_id = vocab.word2idx['<start>']
    end_token_id = vocab.word2idx['<end>']
    unk_token_id = vocab.word2idx['<unk>']
    for word_id in sampled_ids:
        if word_id == start_token_id and len(words) == 0:
            continue
        if word_id == end_token_id:
            break
        words.append(vocab.idx2word[word_id])
    predicted_caption = " ".join(words)

    return predicted_caption


"""
We need to overwrite the default PyTorch collate_fn() because our 
ground truth captions are sequential data of varying lengths. The default
collate_fn() does not support merging the captions with padding.

You can read more about it here:
https://pytorch.org/docs/stable/data.html#dataloader-collate-fn. 
"""
def caption_collate_fn(data):
    """ Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 224, 224).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 224, 224).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length from longest to shortest.
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # merge images (from tuple of 3D tensor to 4D tensor).
    # if using features, 2D tensor to 3D tensor. (batch_size, 256)
    images = torch.stack(images, 0) 

    # merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths


def calculate_bleu(image_id_candidate_reference):
    image_id_bleu = OrderedDict()
    for image_id, v in image_id_candidate_reference.items():
        predict = v['predicted']  # type: str
        ref_captions = v['ref']   # type: list[str]
        predict_splited = predict.split()
        ref_captions_splited = [t.split() for t in ref_captions]
        bleu = sentence_bleu(ref_captions_splited, predict_splited)
        image_id_bleu[image_id] = bleu
        print(f"{image_id}: {bleu}")
    sorted_tuples = sorted(image_id_bleu.items(), key=operator.itemgetter(1), reverse=True)
    image_id_bleu = OrderedDict(sorted_tuples)
    print("Average blue: {}".format(sum(image_id_bleu.values()) / len(image_id_bleu)))
    torch.save(image_id_bleu, "image_id_bleu.pt")
    length = len(image_id_bleu)
    image_ids = list(image_id_bleu.keys())
    # high bleu score sample
    seleted_id = image_ids[length // 8]
    print("bleu: {}, {}".format(image_id_bleu[seleted_id], image_id_candidate_reference[seleted_id]))
    # low bleu score sample
    seleted_id = image_ids[length // 8 * 7]
    print("bleu: {}, {}".format(image_id_bleu[seleted_id], image_id_candidate_reference[seleted_id]))


def calculate_cosine_similarity(image_id_candidate_reference):
    image_id_cos_sim = OrderedDict()
    for image_id, v in image_id_candidate_reference.items():
        predict = v['predicted']  # type: str
        ref_captions = v['ref']   # type: list[str]
        predict_splited = predict.split()
        predict_embed_vector = 
        ref_captions_splited = [t.split() for t in ref_captions]


if __name__ == "__main__":
    image_id_candidate_reference = torch.load('image_id_candidate_reference.pt')
    calculate_bleu(image_id_candidate_reference)