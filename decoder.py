"""
COMP5623M Coursework on Image Caption Generation


python decoder.py


"""

import torch
import numpy as np

import torch.nn as nn
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image


from datasets import Flickr8k_Images, Flickr8k_Features
from models import DecoderRNN, EncoderCNN
from utils import *
from config import *

from tqdm import tqdm

# if false, train model; otherwise try loading model from checkpoint and evaluate
EVAL = True


# reconstruct the captions and vocab, just as in extract_features.py
lines = read_lines(TOKEN_FILE_TRAIN)
image_ids, cleaned_captions = parse_lines(lines)
vocab = build_vocab(cleaned_captions)


# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# initialize the models and set the learning parameters
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(device)


if not EVAL:

    # load the features saved from extract_features.py
    print(len(lines))
    features = torch.load('features.pt', map_location=device)
    print("Loaded features", features.shape)

    features = features.repeat_interleave(5, 0)
    print("Duplicated features", features.shape)

    dataset_train = Flickr8k_Features(
        image_ids=image_ids,
        captions=cleaned_captions,
        vocab=vocab,
        features=features,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64, # change as needed
        shuffle=True,
        num_workers=0, # may need to set to 0
        collate_fn=caption_collate_fn, # explicitly overwrite the collate_fn
    )


    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)

    print(len(image_ids))
    print(len(cleaned_captions))
    print(features.shape)


#########################################################################
#
#        QUESTION 1.3 Training DecoderRNN
# 
#########################################################################

    # TODO write training loop on decoder here


    # for each batch, prepare the targets using this torch.nn.utils.rnn function
    # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
    lengths = len(vocab)
    max_epoch = 5
    print_freq = 50
    for epoch in range(max_epoch):
        losses = []
        for batch_idx, (image_features, captions, lengths) in enumerate(train_loader):
            image_features = image_features.to(device)
            captions = captions.to(device)
            optimizer.zero_grad()
            output = decoder(image_features, captions, lengths)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())
            if batch_idx % print_freq == 0:
                print(f"\tEpoch {epoch + 1} / {max_epoch}: Batch {batch_idx} Loss {losses[-1]}")
        print(f"Epoch {epoch + 1}: Mean Loss {sum(losses) / len(losses)}")


    # save model after training
    decoder_ckpt = torch.save(decoder, "decoder.ckpt")



# if we already trained, and EVAL == True, reload saved model
else:

    data_transform = transforms.Compose([ 
        transforms.Resize(224),     
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),   # using ImageNet norms
                             (0.229, 0.224, 0.225))])


    test_lines = read_lines(TOKEN_FILE_TEST)
    test_image_ids, test_cleaned_captions = parse_lines(test_lines)


    # load models
    encoder = EncoderCNN().to(device)
    decoder = torch.load("decoder.ckpt").to(device)
    encoder.eval()
    decoder.eval() # generate caption, eval mode to not influence batchnorm



#########################################################################
#
#        QUESTION 2.1 Generating predictions on test data
# 
#########################################################################


    # TODO define decode_caption() function in utils.py
    image_id_candidate_reference = {} # type: dict[str, dict[str, list[str]]]
    for image_id, ref_caption in zip(test_image_ids, test_cleaned_captions):
        if image_id not in image_id_candidate_reference:
            image_id_candidate_reference[image_id] = {"predicted": None, "ref": []}
        image_id_candidate_reference[image_id]['ref'].append(ref_caption)
    output = []
    with torch.no_grad():
        for image_id in tqdm(list(image_id_candidate_reference.keys())):
            path = IMAGE_DIR + str(image_id) + ".jpg"
            image = Image.open(open(path, 'rb'))
            image = data_transform(image).to(device)

            image_features = encoder(image.unsqueeze(0)) # add batch_size dim
            word_ids = decoder.sample(image_features.view(1, -1))
            predicted_caption = decode_caption(word_ids.squeeze().tolist(), vocab)
            image_id_candidate_reference[image_id]['predicted'] = predicted_caption
            output.append("\t".join([image_id, predicted_caption]))
    torch.save(image_id_candidate_reference, "image_id_candidate_reference.pt")
    print("\n".join(output))
    with open("caption_generated.txt", "w+") as f:
        f.write("\n".join(output))


#########################################################################
#
#        QUESTION 2.2-3 Caption evaluation via text similarity 
# 
#########################################################################


    # Feel free to add helper functions to utils.py as needed,
    # documenting what they do in the code and in your report

    calculate_bleu(image_id_candidate_reference)

    # TODO
    calculate_cosine_similarity(image_id_candidate_reference)
