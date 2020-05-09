import json
import os
import pathlib
import torch
from transformers import *
from helper_functions import natural_sort_key
import sys

'''
Those are the following fields of the json review
['conference', 'title', 'abstract', 'histories', 'reviews', 'authors', 'accepted', 'id']

Each review has the following fields (the numbers are counters):

    {'IS_META_REVIEW': 7270,
    'comments': 7270,
    'DATE': 6500,
    'TITLE': 6500,
    'OTHER_KEYS': 6500,
    'RECOMMENDATION': 2642,
    'REVIEWER_CONFIDENCE': 2618,
    'SUBSTANCE': 248,
    'RECOMMENDATION_UNOFFICIAL': 168,
    'ORIGINALITY': 516,
    'IS_ANNOTATED': 974,
    'SOUNDNESS_CORRECTNESS': 388,
    'CLARITY': 428,
    'MEANINGFUL_COMPARISON': 154,
    'IMPACT': 296,
    'APPROPRIATENESS': 56}

    Observations:
        comments is the text of the review
        every reviews that is not a meta review has a title and date

'''


class DataLoader:
    ROOT = pathlib.Path(__file__).parent.parent
    # HEAD and TAIL are used for truncating the middle of the review
    HEAD = 128
    TAIL = 384
    MAXLEN = 512

    def __init__(self, device, full_reviews=False, meta_reviews=False, conference='iclr_2017',
                 model_class=BertModel, tokenizer_class=BertTokenizer, pretrained_weights='bert-base-cased',
                 truncate_policy='right'):

        # features from the review used
        self.full_reviews = full_reviews
        self.meta_reviews = meta_reviews
        self.conference = conference

        # Different transformer models options
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.pretrained_weights = pretrained_weights
        self.truncate_policy = truncate_policy  # By default it truncates from right, i.e., the end of the review
        # https://stackoverflow.com/questions/58636587/how-to-use-bert-for-long-text-classification -
        # https://arxiv.org/pdf/1905.05583.pdf : truncation policy

        # Construct pretrained transformer model
        # TODO there is a lowercasing probably
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights)
        self.model.eval()


        # get file names
        train_path = self.ROOT / ('data/PeerRead/data/' + conference) / 'train/reviews/'
        test_path = self.ROOT / ('data/PeerRead/data/' + conference) / 'test/reviews/'
        dev_path = self.ROOT / ('data/PeerRead/data/' + conference) / 'dev/reviews/'

        train_files = [os.path.join(train_path, file) for file in os.listdir(train_path) if file.endswith('.json')]
        test_files = [os.path.join(test_path, file) for file in os.listdir(test_path) if file.endswith('.json')]
        dev_files = [os.path.join(dev_path, file) for file in os.listdir(dev_path) if file.endswith('.json')]

        print(len(train_files), len(dev_files), len(test_files),
              len(train_files) / (len(train_files) + len(dev_files) + len(test_files)))

        self.files = train_files + test_files + dev_files

        # There is a 0.8-0.1-0.1 split on the data. I am merging the data so we can decide on a different split.

        # Peer reviews from paper
        self.paper_reviews = []
        # self._read_full_reviews()
        self.embeddings_from_reviews = []
        self.device = device
        self.labels = []

        # path for saving embeddings matrices
        self.path = self.ROOT / 'data/embeddings/' / self.conference / 'pre_trained' / self.get_dir_name()

    def get_dir_name(self):
        return '_'.join([self.model_class.__name__, self.pretrained_weights, 'truncate-from-' + self.truncate_policy])

    def _create_embeddings(self):
        self.model.to(self.device)
        cnt = 0
        for reviews in self.paper_reviews:
            # embeddings = []
            # for review in paper:
            #     print(review)
            #     input_ids = torch.tensor([self.tokenizer.encode(review, max_length=512)])
            #     with torch.no_grad():
            #         _last_hidden_states, classification_token_state = self.model(input_ids)
            #     embeddings.append(classification_token_state)
            # embeddings = torch.cat(embeddings)
            # embeddings = embeddings.view((1, len(reviews), -1))
            # self.embeddings_from_reviews.append(embeddings)
            if self.truncate_policy == 'right':
                batch_input_ids = self.tokenizer.batch_encode_plus(reviews, max_length=self.MAXLEN,
                                                                   pad_to_max_length=True, return_tensors='pt')
            elif self.truncate_policy == 'left':
                # left is bugged.
                batch_input_ids = self.tokenizer.batch_encode_plus(reviews,
                                                                   pad_to_max_length=True, return_tensors='pt')
                for key, tensor in batch_input_ids.items():
                    # batch_input_ids[key] = tensor[:, -self.MAXLEN:]
                    head = tensor[:, :1]
                    tail = tensor[:, -self.MAXLEN-1:]
                    batch_input_ids[key] = torch.cat([head, tail], dim=1)
            elif self.truncate_policy == 'mid':
                batch_input_ids = self.tokenizer.batch_encode_plus(reviews,
                                                                   pad_to_max_length=True, return_tensors='pt')
                for key, tensor in batch_input_ids.items():
                    head = tensor[:, :self.HEAD]
                    tail = tensor[:, -self.TAIL:]
                    batch_input_ids[key] = torch.cat([head, tail], dim=1)
            else:
                raise Exception('Invalid truncation policy. Choose from: mid, left, right')
            # print(batch_input_ids)
            for key, tensor in batch_input_ids.items():
                batch_input_ids[key] = tensor.to(self.device)
            with torch.no_grad():
                reviews_embeddings = self.model(**batch_input_ids)

            # --clear from GPU memory--
            for tensor in batch_input_ids.values():
                del tensor
            pooled_embeddings = reviews_embeddings[1].cpu()
            for emb in reviews_embeddings:
                del emb
            torch.cuda.empty_cache()
            # -------------------------

            self.embeddings_from_reviews.append(pooled_embeddings)
            cnt += 1
            print(cnt)
        return self.embeddings_from_reviews

    def _read_full_reviews(self):
        for i, file in enumerate(self.files):
            with open(file) as json_file:
                full_reviews = json.load(json_file)
                if self.full_reviews:
                    self.paper_reviews.append(full_reviews)
                else:
                    reviews_for_specific_paper = []
                    for review in full_reviews['reviews']:
                        if self.meta_reviews or not review['IS_META_REVIEW']:
                            reviews_for_specific_paper.append(review)

                    self.paper_reviews.append(reviews_for_specific_paper)
        return self.paper_reviews

    def _read_reviews_only_text(self):
        for i, file in enumerate(self.files):
            with open(file) as json_file:
                full_reviews = json.load(json_file)
                if self.full_reviews:
                    self.paper_reviews.append(full_reviews)
                else:
                    reviews_for_specific_paper = []
                    for review in full_reviews['reviews']:
                        if self.meta_reviews or not review['IS_META_REVIEW']:
                            reviews_for_specific_paper.append(review['comments'])

                    self.paper_reviews.append(reviews_for_specific_paper)
        return self.paper_reviews

    def _get_full_review_stats(self):
        if not self.full_reviews:
            total_reviews = 0
            for paper in self.paper_reviews:
                for _ in paper:
                    total_reviews += 1

            print('Total Papers: ', len(self.paper_reviews))
            print('Total Reviews: ', total_reviews)
            return

        reviews_structures = []
        total_reviews = 0
        meta_reviews = 0
        field_counters = {}

        for i, file in enumerate(self.paper_reviews):
            print(i)
            for j in range(len(self.paper_reviews[i]['reviews'])):
                total_reviews += 1
                for key in self.paper_reviews[i]['reviews'][j]:
                    if key not in field_counters:
                        field_counters[key] = 1
                    else:
                        field_counters[key] += 1
                if self.paper_reviews[i]['reviews'][j]['IS_META_REVIEW']:
                    meta_reviews += 1
                if not self.paper_reviews[i]['reviews'][j]['IS_META_REVIEW'] and \
                        'TITLE' not in self.paper_reviews[i]['reviews'][j]:
                    print(self.paper_reviews[i]['reviews'][j]['comments'])
                review_struct = list(self.paper_reviews[i]['reviews'][j].keys())
                review_struct.sort()
                if review_struct not in reviews_structures:
                    reviews_structures.append(review_struct)
        print(reviews_structures)
        print(len(reviews_structures))
        print('Total Papers: ', len(self.paper_reviews))
        print('Total Reviews: ', total_reviews)
        print('Meta Reviews: ', meta_reviews)
        print(field_counters)

    def get_embeddings_from_reviews(self):
        self._read_reviews_only_text()
        self._get_full_review_stats()
        return self._create_embeddings()

    def write_embeddings_to_file(self):
        # path = self.ROOT / 'data/embeddings/' / self.conference / 'pre_trained' / (self.model_class.__name__ + '_' +
        #                                                                            self.pretrained_weights)
        print(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        for idx, reviews in enumerate(self.embeddings_from_reviews):
            torch.save(reviews, self.path / ('paper_' + str(idx) + '.pt'))

    def read_embeddigns_from_file(self):
        files = [os.path.join(self.path, file) for file in os.listdir(self.path) if file.endswith('.pt')]
        # natural ordering sort so when I keep the order of the papers from PeerRead. This preserves the order for
        # reading the labels.
        files.sort(key=natural_sort_key)
        for file in files:
            self.embeddings_from_reviews.append(torch.load(file))
        return self.embeddings_from_reviews

    def read_labels(self):
        for i, file in enumerate(self.files):
            with open(file) as json_file:
                full_reviews = json.load(json_file)
            self.labels.append(full_reviews['accepted'])
        self.labels = torch.tensor(self.labels, dtype=torch.float)
        return self.labels


if __name__ == '__main__':
    GPU = True
    if GPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)
    for policy in sys.argv[1:]:
        r = DataLoader(device, truncate_policy=policy)
        r.get_embeddings_from_reviews()
        r.write_embeddings_to_file()

# GPU = True
# device_idx = 0
# if GPU:
#     device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
# else:
#     device = torch.device("cpu")
# print(device)
#
# r = DataLoader(device, truncate_policy='left')
# r.get_embeddings_from_reviews()
# r.write_embeddings_to_file()
# # r.read_embeddigns_from_file()
#
#
# # REVIEWS = read_full_reviews()
# #
# # get_full_review_stats(REVIEWS)
#

