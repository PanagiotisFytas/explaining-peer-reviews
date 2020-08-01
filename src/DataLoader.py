import json
import os
import pathlib
import torch
from transformers import *
from helper_functions import natural_sort_key
import sys
import nltk
from nltk.corpus import stopwords
import spacy
import numpy as np
import operator
from sklearn.datasets import load_svmlight_file
import language_check

'''
0       get_most_recent_reference_year
1       get_num_references
2       get_num_refmentions
3       get_avg_length_reference_mention_contexts
4       abstract_contains_deep
5       abstract_contains_neural
6       abstract_contains_embedding
7       abstract_contains_outperform
8       abstract_contains_novel
9       abstract_contains_state_of_the_art
10      abstract_contains_state-of-the-art
11      get_num_recent_references
12      get_num_ref_to_figures
13      get_num_ref_to_tables
14      get_num_ref_to_sections
15      get_num_uniq_words
16      get_num_sections
17      get_avg_sentence_length
18      get_contains_appendix
19      proportion_of_frequent_words
20      get_title_length
21      get_num_authors
22      get_num_ref_to_equations
23      get_num_ref_to_theorems
'''

# features_to_use = [1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23]
features_to_use = [12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23]

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
try:
    stopwords_en = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    stopwords_en = stopwords.words('english')

stopwords_en = stopwords.words('english')
stopwords_kept = ['am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
                  'does', 'did', 'doing', 'no', 'nor', 'not', 'can', 'will', 'don', "don't", 'should', "should've",
                  'll', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
                  "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'mightn', "mightn't", 'mustn',
                  "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
                  "weren't", 'won', "won't", 'wouldn', "wouldn't"]


class DataLoader:
    # ROOT = pathlib.Path(__file__).parent.parent
    DATA_ROOT = pathlib.Path(os.environ['DATA'])
    # HEAD and TAIL are used for truncating the middle of the review
    HEAD = 128
    TAIL = 384
    MAXLEN = 512
    BATCH_SIZE = 200
    SCIBERT_PATH = str(DATA_ROOT / 'scibert_scivocab_uncased')

    def __init__(self, device, full_reviews=False, meta_reviews=False, conference='iclr_2017',
                 model_class=BertModel, tokenizer_class=BertTokenizer, pretrained_weights='bert-base-uncased',
                 truncate_policy='right', final_decision='include', allow_empty=True, remove_duplicates=True, 
                 remove_stopwords=False):
        """
        :param device: 'cuda' or 'cpu'
        :param full_reviews:
        :param meta_reviews:
        :param conference:
        :param model_class:
        :param tokenizer_class:
        :param pretrained_weights:
        :param truncate_policy:
        :param final_decision: 'include', 'exclude', 'only'
        :param allow_empty
        """
        # features from the review used
        self.full_reviews = full_reviews
        self.meta_reviews = meta_reviews
        self.final_decision = final_decision
        if final_decision not in ['include', 'exclude', 'only']:
            print(final_decision)
            raise Exception('Wrong final_decision value')
        self.allow_empty = allow_empty
        self.conference = conference
        self.remove_duplicates = remove_duplicates
        self.remove_stopwords = remove_stopwords
        if remove_stopwords:
            self.stopwords = list(set(stopwords_en) - set(stopwords_kept))

        # Different transformer models options
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        if pretrained_weights == 'scibert_scivocab_uncased':
            self.pretrained_weights = self.SCIBERT_PATH
            self.pretrained_weights_name = pretrained_weights
        else:
            self.pretrained_weights = pretrained_weights
            self.pretrained_weights_name = pretrained_weights

        self.truncate_policy = truncate_policy  # By default it truncates from right, i.e., the end of the review
        # https://stackoverflow.com/questions/58636587/how-to-use-bert-for-long-text-classification -
        # https://arxiv.org/pdf/1905.05583.pdf : truncation policy

        # Construct pretrained transformer model
        # TODO there is a lowercasing probably
        self.tokenizer = tokenizer_class.from_pretrained(self.pretrained_weights)
        self.model = model_class.from_pretrained(self.pretrained_weights)
        self.model.eval()

        # get file names
        train_path = self.DATA_ROOT / ('PeerRead/data/' + conference) / 'train/reviews/'
        test_path = self.DATA_ROOT / ('PeerRead/data/' + conference) / 'test/reviews/'
        dev_path = self.DATA_ROOT / ('PeerRead/data/' + conference) / 'dev/reviews/'

        train_files = [os.path.join(train_path, file) for file in os.listdir(train_path) if file.endswith('.json')]
        test_files = [os.path.join(test_path, file) for file in os.listdir(test_path) if file.endswith('.json')]
        dev_files = [os.path.join(dev_path, file) for file in os.listdir(dev_path) if file.endswith('.json')]

        print(len(train_files), len(dev_files), len(test_files),
              len(train_files) / (len(train_files) + len(dev_files) + len(test_files)))

        self.files = test_files + dev_files + train_files

        # There is a 0.8-0.1-0.1 split on the data. I am merging the data so we can decide on a different split.
        # print(self.files)
        # Peer reviews from paper
        self.paper_reviews = []
        # self.read_full_reviews()
        self.embeddings_from_reviews = []
        self.device = device
        self.labels = []
        self.abstracts = []
        self.abstract_embeddings = []
        # only used for perreview dataloaders
        self.peer_review_to_paper_ids = []

        # path for saving embeddings matrices
        self.path = self.DATA_ROOT / 'embeddings/' / self.conference / 'pre_trained' / self.get_dir_name()

    def get_dir_name(self):
        if self.final_decision == 'include':
            ret = '_'.join([
                self.model_class.__name__,
                self.pretrained_weights_name,
                'truncate-from-' + self.truncate_policy
            ])
        else:
            ret = '_'.join([
                self.model_class.__name__,
                self.pretrained_weights_name,
                'truncate-from-' + self.truncate_policy,
                'final-decision-' + self.final_decision
            ])
        if not self.allow_empty:
            ret += '_not-allow-empty'
        if self.remove_duplicates:
            ret += '_remove-duplicates'
        if self.remove_stopwords:
            ret += '_remove-stopwords'
        return ret

    def reviews_to_embeddings(self, reviews):
        if self.remove_stopwords:
            for idx, review in enumerate(reviews):
                rev = review
                rev = ' '.join([
                    word for word in rev.split() if word.lower() not in (self.stopwords)
                ])
                reviews[idx] = rev
        if self.truncate_policy == 'right':
            batch_input_ids = self.tokenizer.batch_encode_plus(reviews, max_length=self.MAXLEN,
                                                               pad_to_max_length=True, return_tensors='pt')
        elif self.truncate_policy == 'left':
            # left is bugged.
            batch_input_ids = self.tokenizer.batch_encode_plus(reviews,
                                                               pad_to_max_length=True, return_tensors='pt')
            for key, tensor in batch_input_ids.items():
                # batch_input_ids[key] = tensor[:, -self.MAXLEN:]
                head = tensor[:, :2]
                tail = tensor[:, -self.MAXLEN - 2:]
                batch_input_ids[key] = torch.cat([head, tail], dim=1)
        elif self.truncate_policy == 'mid':
            batch_input_ids = self.tokenizer.batch_encode_plus(reviews,
                                                               pad_to_max_length=True, return_tensors='pt')
            for key, tensor in batch_input_ids.items():
                head = tensor[:, :self.HEAD]
                tail = tensor[:, -self.TAIL:]
                batch_input_ids[key] = torch.cat([head, tail], dim=1)
        elif self.truncate_policy == 'half':
            batch_input_ids = self.tokenizer.batch_encode_plus(reviews,
                                                               pad_to_max_length=True, return_tensors='pt')
            for key, tensor in batch_input_ids.items():
                head = tensor[:, :int(self.MAXLEN / 2)]
                tail = tensor[:, -int(self.MAXLEN / 2):]
                batch_input_ids[key] = torch.cat([head, tail], dim=1)
        else:
            raise Exception('Invalid truncation policy. Choose from: mid, left, right, half')
        # print(batch_input_ids)
        for key, tensor in batch_input_ids.items():
            batch_input_ids[key] = tensor.to(self.device)
        with torch.no_grad():
            reviews_embeddings = self.model(**batch_input_ids)

        # --clear from GPU memory--
        for tensor in batch_input_ids.values():
            del tensor
        embeddings = reviews_embeddings[1].cpu()
        for emb in reviews_embeddings:
            del emb
        torch.cuda.empty_cache()
        return embeddings
        # -------------------------

    def _create_embeddings(self):
        self.model.to(self.device)
        cnt = 0
        for reviews in self.paper_reviews:
            embeddings = self.reviews_to_embeddings(reviews)
            self.embeddings_from_reviews.append(embeddings)
            cnt += 1
            print(cnt)
        return self.embeddings_from_reviews

    def exclude(self, review):
        # exclude empty reviews and final decision if specified
        return (not self.allow_empty and not review['comments'])\
               or (self.final_decision == 'exclude' and review.get('TITLE') == 'ICLR committee final decision')

    def read_full_reviews(self):
        """
        :returns both the text of the reviews and the metadata of the reviews
        """
        for i, file in enumerate(self.files):
            with open(file) as json_file:
                full_reviews = json.load(json_file)
                if self.full_reviews:
                    self.paper_reviews.append(full_reviews)
                else:
                    reviews_for_specific_paper = []
                    for review in full_reviews['reviews']:
                        if self.exclude(review):
                            continue
                        if self.final_decision == 'only' and review.get('TITLE') == 'ICLR committee final decision':
                            reviews_for_specific_paper.append(review['comments'])
                        elif self.meta_reviews or not review['IS_META_REVIEW']:
                            reviews_for_specific_paper.append(review)

                    self.paper_reviews.append(reviews_for_specific_paper)
        return self.paper_reviews

    """
    :returns only the text of the reviews (comments attribute of the json)
    """
    def read_reviews_only_text(self):
        for i, file in enumerate(self.files):
            with open(file) as json_file:
                full_reviews = json.load(json_file)
                if self.full_reviews:
                    self.paper_reviews.append(full_reviews)
                else:
                    reviews_for_specific_paper = []
                    for review in full_reviews['reviews']:
                        if self.exclude(review):
                            continue
                        if self.final_decision == 'only':
                            if review.get('TITLE') == 'ICLR committee final decision':
                                if not self.remove_duplicates or review['comments'] not in reviews_for_specific_paper:
                                    reviews_for_specific_paper.append(review['comments'])
                        elif self.meta_reviews or not review['IS_META_REVIEW']:
                            if not self.remove_duplicates or review['comments'] not in reviews_for_specific_paper:
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
        self.read_reviews_only_text()
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

    def read_abstract_embeddings(self):
        abstract_path = self.DATA_ROOT / 'abstract_embeddings' / self.conference / 'pre_trained' / self.get_dir_name()
        try:
            self._read_abstract_embeddigns_from_file(abstract_path)
        except FileNotFoundError:
            self.read_abstracts_text()
            self._create_abstract_embeddings()
            self._write_abstract_embeddings(abstract_path)
        return self.abstract_embeddings

    def read_abstracts_text(self):
        for i, file in enumerate(self.files):
            with open(file) as json_file:
                full_reviews = json.load(json_file)
            self.abstracts.append(full_reviews['abstract'])
        # self.abstracts = torch.tensor(self.abstracts, dtype=torch.float)
        return self.abstracts

    def read_titles(self):
        titles = []
        for i, file in enumerate(self.files):
            with open(file) as json_file:
                full_reviews = json.load(json_file)
            titles.append(full_reviews['title'])
        # self.abstracts = torch.tensor(self.abstracts, dtype=torch.float)
        return titles

    def _read_abstract_embeddigns_from_file(self, abstract_path):
        files = [os.path.join(abstract_path, file) for file in os.listdir(abstract_path) if file.endswith('.pt') and 'abstract_' in file]
        # natural ordering sort so when I keep the order of the papers from PeerRead. This preserves the order for
        # reading the labels.
        files.sort(key=natural_sort_key)
        for file in files:
            self.abstract_embeddings.append(torch.load(file).view(1, -1))
        print('Shape: ', self.abstract_embeddings[0].shape)
        self.abstract_embeddings = torch.cat(self.abstract_embeddings)
        print('Shape: ', self.abstract_embeddings.shape)
        return self.abstract_embeddings

    def _write_abstract_embeddings(self, abstract_path):
        abstract_path = self.DATA_ROOT / 'abstract_embeddings' / self.conference / 'pre_trained' / self.get_dir_name()
        abstract_path.mkdir(parents=True, exist_ok=True)
        for idx, abstract in enumerate(self.abstract_embeddings):
            torch.save(abstract, abstract_path / ('abstract_' + str(idx) + '.pt'))

    def _create_abstract_embeddings(self):
        self.model.to(self.device)
        cnt = 0
        N = len(self.abstracts)
        print('Total Len: ', N)
        for i in range(0, N, self.BATCH_SIZE):
            abstracts = self.abstracts[i:i+self.BATCH_SIZE]
            embeddings = self.reviews_to_embeddings(abstracts)
            self.abstract_embeddings.append(embeddings)
            cnt += 1
            print(cnt)
        self.abstract_embeddings = torch.cat(self.abstract_embeddings)
        print('Shape: ', self.abstract_embeddings.shape)
        return self.abstract_embeddings

    def parsed_pdfs_from_review_files(self):
        parsed_pdfs = []
        for file in self.files:
            parsed_pdfs.append(file.replace('reviews','parsed_pdfs').replace('.json','.pdf.json'))
        return parsed_pdfs

    def read_email_domains(self):
        parsed_pdfs = self.parsed_pdfs_from_review_files()
        email_domains = []
        for file in parsed_pdfs:
            with open(file) as json_file:
                paper = json.load(json_file)
                emails = paper['metadata']['emails']
                domains = [email.split('@')[1] for email in emails]
                email_domains.append(domains)
        return email_domains
        
    def domain_counts(self, domains):
        domain_counts = {}
        for domain_per_paper in domains:
            for domain in domain_per_paper:
                if domain in domain_counts:
                    domain_counts[domain] += 1
                else:
                    domain_counts[domain] = 1
        return dict(sorted(domain_counts.items(), key=operator.itemgetter(1),reverse=True))

    def read_handcrafted_features(self, test_order=True):
        # if test_order=True, read_labels must have been called first
        train_path = self.DATA_ROOT / ('PeerRead/data/' + self.conference) / 'train/dataset'
        test_path = self.DATA_ROOT / ('PeerRead/data/' + self.conference) / 'test/dataset'
        dev_path = self.DATA_ROOT / ('PeerRead/data/' + self.conference) / 'dev/dataset'
        features_file_name = 'features.svmlite_30000_False_True.txt'
        train_data = self.read_features_from_file(train_path / features_file_name)
        test_data = self.read_features_from_file(test_path / features_file_name)
        dev_data = self.read_features_from_file(dev_path / features_file_name)
        
        ids_file_name = 'ids_30000_False_True.tsv'
        train_ids_file = train_path / ids_file_name
        test_ids_file = test_path / ids_file_name
        dev_ids_file = dev_path / ids_file_name

        title_to_idx = self.generate_ids_dict([train_ids_file, test_ids_file, dev_ids_file])

        titles = self.read_titles()

        data = np.concatenate([train_data, test_data, dev_data], axis=0)
        fixed_ids = self.fix_feature_order(title_to_idx, titles)

        if test_order:
            # assert sequence is corrected
            labels_file_name = 'labels_30000_False_True.tsv'
            train_labels_file = train_path / labels_file_name
            test_labels_file = test_path / labels_file_name
            dev_labels_file = dev_path /labels_file_name
            labels = self.read_labels_from_feature_file([train_labels_file, test_labels_file, dev_labels_file])
            labels = torch.tensor(labels, dtype=torch.float)[fixed_ids]
            assert torch.all(torch.eq(labels,
                                      torch.tensor(self.labels, dtype=torch.float)))

        return torch.tensor(data[fixed_ids], dtype=torch.float)
        # self.files = test_files + dev_files + train_files

    @staticmethod
    def read_features_from_file(file):
        sparse_data = load_svmlight_file(str(file))
        data = np.array(sparse_data[0].todense())
        return data[:, features_to_use]

    @staticmethod
    def fix_feature_order(title_to_idx, titles):
        fixed_ids = []
        for title in titles:
            fixed_ids.append(title_to_idx[title])
        return fixed_ids

    @staticmethod
    def generate_ids_dict(files):
        title_to_idx = {}
        idx = 0
        for file in files:
            with open(file) as fp:
                for line in fp:
                    _, title = line.rstrip('\n').split('\t')
                    title_to_idx[title] = idx
                    idx += 1
        return title_to_idx

    @staticmethod
    def read_labels_from_feature_file(files):
        labels_str = ''
        for file in files:
            with open(file) as fp:
                labels_str += fp.readline()
        return np.array(list(map(int, labels_str)))

    def read_paper_text(self):
        parsed_pdfs = self.parsed_pdfs_from_review_files()
        papers_texts = []
        for file in parsed_pdfs:
            with open(file) as json_file:
                paper = json.load(json_file)
                sections = paper['metadata']['sections']
                sections_in_paper = []
                try:
                    for section in sections:
                        if section['heading']:
                            sections_in_paper.append(section['heading'])
                        if section['text']:
                            sections_in_paper.append(section['text'])
                    papers_texts.append(sections_in_paper)
                except TypeError:
                    # No text exists in the paper json
                    # Will dummy value to impute with the average value
                    papers_texts.append([])
        return papers_texts

    def detect_errors(self, papers_texts, abstracts):
        tool = language_check.LanguageTool('en-US')
        errors_per_paper = []
        words_per_paper = []
        ids_to_impute = []
        total_errors = 0
        total_words = 0
        for idx, section_in_paper in enumerate(papers_texts):
            print('IDX ', idx)
            if not section_in_paper:
                print('This will be imputed')
                errors = -1 # dummy value, will be imputed
                words = -1
                ids_to_impute.append(idx)
            else:
                errors = 0
                words = 0
                for text in section_in_paper:
                    errors += len(tool.check(text))
                    total_errors += errors
                    words += len(text.split())
                    total_words += words
                print('Errors ', errors)
                print('Words', words)
            errors_per_paper.append(errors)
            words_per_paper.append(words)
        mean_errors_per_paper = round(total_errors / (len(papers_texts) - len(ids_to_impute)))
        mean_words_per_paper = round(total_words / (len(papers_texts) - len(ids_to_impute)))
        # impute
        for idx in ids_to_impute:
            assert(errors_per_paper[idx] == -1)
            assert(words_per_paper[idx] == -1)
            errors_per_paper[idx] = mean_errors_per_paper
            words_per_paper[idx] = mean_words_per_paper
        # calculate errors in abstract
        abstract_errors = []
        abstract_words = []
        for abstract_text in abstracts:
            abstract_errors.append(len(tool.check(abstract_text)))
            abstract_words.append(len(abstract_text.split()))
        return torch.tensor(errors_per_paper, dtype=torch.float),\
                torch.tensor(abstract_errors, dtype=torch.float),\
                torch.tensor(words_per_paper, dtype=torch.float),\
                torch.tensor(abstract_words, dtype=torch.float)

    def read_errors(self):
        try:
            paper_errors = torch.load(self.DATA_ROOT / ('paper_errors.pt'))
            abstract_errors = torch.load(self.DATA_ROOT / ('abstract_errors.pt'))
            paper_words = torch.load(self.DATA_ROOT / ('paper_words.pt'))
            abstract_words = torch.load(self.DATA_ROOT / ('abstract_words.pt'))
        except FileNotFoundError:
            if self.abstracts:
                abstracts = self.abstracts
            else:
                abstracts = self.read_abstracts_text()
            paper_text = self.read_paper_text()
            paper_errors, abstract_errors, paper_words, abstract_words = self.detect_errors(paper_text, abstracts)
            torch.save(paper_errors, self.DATA_ROOT / ('paper_errors.pt'))
            torch.save(abstract_errors, self.DATA_ROOT / ('abstract_errors.pt'))
            torch.save(paper_words, self.DATA_ROOT / ('paper_words.pt'))
            torch.save(abstract_words, self.DATA_ROOT / ('abstract_words.pt'))

        return paper_errors, abstract_errors, paper_words, abstract_words
        '''
        ## TO read handcrafted_features and errors:
        from DataLoader import DataLoader
        d = DataLoader('cpu')
        d.read_labels().shape
        feat = d.read_handcrafted_features()
        perr, aerr, pwor, awor = d.read_errors()
        '''
    
    def copy_to_peer_review(self, features):
        copied_features = []
        for idx in self.peer_review_to_paper_ids:
            copied_features.append(features[idx])
        return np.stack(copied_features)


class PerReviewDataLoader(DataLoader):
    BATCH_SIZE = 200
    def __init__(self, *args, **kwargs):
        super(PerReviewDataLoader, self).__init__(*args, **kwargs)
        self.path = self.DATA_ROOT / 'per_review_embeddings/' / self.conference / 'pre_trained' / self.get_dir_name()
        self.recommendation_scores = []

    def write_embeddings_to_file(self):
        print(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        for idx, reviews in enumerate(self.embeddings_from_reviews):
            torch.save(reviews, self.path / ('review_' + str(idx) + '.pt'))
        
    def write_scores_to_file(self):
        print(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        torch.save(self.recommendation_scores, self.path / ('recommendation_scores.pt'))

    def read_embeddigns_from_file(self):
        files = [os.path.join(self.path, file) for file in os.listdir(self.path) if file.endswith('.pt') and 'review_' in file]
        # natural ordering sort so when I keep the order of the papers from PeerRead. This preserves the order for
        # reading the labels.
        files.sort(key=natural_sort_key)
        for file in files:
            self.embeddings_from_reviews.append(torch.load(file).view(1, -1))
        print('Shape: ', self.embeddings_from_reviews[0].shape)
        self.embeddings_from_reviews = torch.cat(self.embeddings_from_reviews)
        print('Shape: ', self.embeddings_from_reviews.shape)
        return self.embeddings_from_reviews

    def read_scores_from_file(self):
        file = self.path / ('recommendation_scores.pt')
        self.recommendation_scores = torch.load(file)
        return self.recommendation_scores
    
    def exclude(self, review):
        # exclude empty reviews and final decision if specified
        return (not self.allow_empty and not review['comments'])\
               or ('RECOMMENDATION' not in review)\
               or (self.final_decision == 'exclude' and review.get('TITLE') == 'ICLR committee final decision')

    """
    :returns only the text of the reviews (comments attribute of the json)
    """
    def read_reviews_only_text(self):
        for i, file in enumerate(self.files):
            with open(file) as json_file:
                full_reviews = json.load(json_file)
                reviews_for_specific_paper = []
                for review in full_reviews['reviews']:
                    if self.exclude(review):
                        continue
                    if self.final_decision == 'only':
                        raise Exception('Final decisions should be excluded from this data loader')
                    elif self.meta_reviews or not review['IS_META_REVIEW']:
                        if not self.remove_duplicates or review['comments'] not in reviews_for_specific_paper:
                            reviews_for_specific_paper.append(review['comments'])
                            self.paper_reviews.append(review['comments'])
                            self.peer_review_to_paper_ids.append(i)
                            self.recommendation_scores.append(review['RECOMMENDATION'])
        return self.paper_reviews
    
    def _get_full_review_stats(self):
        pass

    def read_scores(self, task='classification'):
        self.recommendation_scores = torch.tensor(self.recommendation_scores, dtype=torch.float)
        if task == 'classification':
            return (self.recommendation_scores > 5).float()
        else:
            return self.recommendation_scores
        

    def _create_embeddings(self):
        self.model.to(self.device)
        cnt = 0
        N = len(self.paper_reviews)
        print('Total Len: ', N)
        for i in range(0, N, self.BATCH_SIZE):
            reviews = self.paper_reviews[i:i+self.BATCH_SIZE]
            embeddings = self.reviews_to_embeddings(reviews)
            self.embeddings_from_reviews.append(embeddings)
            cnt += 1
            print(cnt)
        self.embeddings_from_reviews = torch.cat(self.embeddings_from_reviews)
        print('Shape: ', self.embeddings_from_reviews.shape)
        return self.embeddings_from_reviews
    
    def tokenise(self, reviews, maxlen=250, overlap=50):
        splitted_reviews = []
        for review in reviews:
            review_widnows = self.get_windows(maxlen=maxlen, overlap=ovelap)
            splitted_reviews.append(review_widnows)
        batch_input_ids = self.tokenizer.batch_encode_plus(reviews, max_length=maxlen,
                                                           pad_to_max_length=True, return_tensors='pt')
        # # print(batch_input_ids)
        # for key, tensor in batch_input_ids.items():
        #     batch_input_ids[key] = tensor.to(self.device)
        # with torch.no_grad():
        #     reviews_embeddings = self.model(**batch_input_ids)

        # # --clear from GPU memory--
        # for tensor in batch_input_ids.values():
        #     del tensor
        # embeddings = reviews_embeddings[1].cpu()
        # for emb in reviews_embeddings:
        #     del emb
        # torch.cuda.empty_cache()
        # return embeddings
        # # -------------------------

    @staticmethod
    def get_windows(text, maxlen=250, overlap=50):
        text_windows = []
        new_tokens_len = (maxlen - overlap)
        if (len(text.split()) // new_tokens_len) > 0:
            number_of_windows = len(text.split()) // new_tokens_len
        else: 
            number_of_windows = 1
        for window_idx in range(number_of_windows):
            if window_idx == 0:
                window = text.split()[:maxlen]
                test_windows.append(" ".join(window))
            else:
                start_idx = window_idx * new_tokens_len
                end_idx = start_idx + maxlen
                window = text.split()[start_idx:end_idx]
                text_windows.append(" ".join(window))
        return text_windows

class PreProcessor:
    nlp = spacy.load('en_core_web_lg')
    def __init__(self, final_decision='only', lemmatise=True, lowercase=True, remove_stopwords=True, punctuation_removal=True):
        self.lemmatise = lemmatise
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.punctuation_removal = punctuation_removal
        if self.remove_stopwords:
            self.stopwords = stopwords.words('english')
        else:
            self.stopwords = []
        self.final_decision = final_decision

    def preprocess(self, reviews):
        processed_reviews = []
        for review in reviews:
            # for meta reviews:
            # print(review)
            if self.final_decision == 'only':
                review = review[0]
            if self.lowercase:
                review = review.lower()
            if self.lemmatise:
                review = [token.lemma_ for token in self.nlp.tokenizer(review)
                          if (not self.punctuation_removal or not (token.is_punct or token.is_space))]
            else:
                review = [token.text for token in self.nlp.tokenizer(review)
                          if (not self.punctuation_removal or not (token.is_punct or token.is_space))]
            review = [word for word in review if word not in self.stopwords]
            processed_reviews.append(review)
            print(review)
        return processed_reviews


class LSTMEmbeddingLoader(DataLoader):
    # ROOT = pathlib.Path(__file__).parent.parent
    DATA_ROOT = pathlib.Path(os.environ['DATA'])
    SCIBERT_PATH = str(DATA_ROOT / 'scibert_scivocab_uncased')
    
    def __init__(self, device, model_class=BertModel, tokenizer_class=BertTokenizer, conference='iclr_2017', 
                 final_decision='only', pretrained_weights='bert-base-uncased', lemmatise=True, lowercase=True, 
                 remove_stopwords=True, punctuation_removal=True):
        
        self.meta_reviews = False
        self.full_reviews = False
        self.remove_duplicates = True
        self.allow_empty = False
        self.truncate_policy = 'right'

        self.lemmatise = lemmatise
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.punctuation_removal = punctuation_removal
        self.preprocessor = PreProcessor(lemmatise=lemmatise, lowercase=lowercase, remove_stopwords=remove_stopwords,
                                         final_decision=final_decision, punctuation_removal=punctuation_removal)
        
        self.conference = conference
        self.final_decision = final_decision

        # Different transformer models options
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        if pretrained_weights == 'scibert_scivocab_uncased':
            self.pretrained_weights = self.SCIBERT_PATH
            self.pretrained_weights_name = pretrained_weights
        else:
            self.pretrained_weights = pretrained_weights
            self.pretrained_weights_name = pretrained_weights

        # Construct pretrained transformer model
        # TODO there is a lowercasing probably
        self.tokenizer = tokenizer_class.from_pretrained(self.pretrained_weights)
        self.model = model_class.from_pretrained(self.pretrained_weights)
        self.model.eval()

        # get file names
        train_path = self.DATA_ROOT / ('PeerRead/data/' + conference) / 'train/reviews/'
        test_path = self.DATA_ROOT / ('PeerRead/data/' + conference) / 'test/reviews/'
        dev_path = self.DATA_ROOT / ('PeerRead/data/' + conference) / 'dev/reviews/'

        train_files = [os.path.join(train_path, file) for file in os.listdir(train_path) if file.endswith('.json')]
        test_files = [os.path.join(test_path, file) for file in os.listdir(test_path) if file.endswith('.json')]
        dev_files = [os.path.join(dev_path, file) for file in os.listdir(dev_path) if file.endswith('.json')]

        print(len(train_files), len(dev_files), len(test_files),
              len(train_files) / (len(train_files) + len(dev_files) + len(test_files)))

        self.files = test_files + dev_files + train_files

        # There is a 0.8-0.1-0.1 split on the data. I am merging the data so we can decide on a different split.
        # print(self.files)
        # Peer reviews from paper
        self.paper_reviews = []
        # self.read_full_reviews()
        self.embeddings_from_reviews = []
        self.device = device
        self.labels = []
        self.scores = []
        self.abstracts = []
        self.abstract_embeddings = []

        # path for saving embeddings matrices
        self.path = self.DATA_ROOT / 'lstm_embeddings/' / self.conference / 'pre_trained' / self.get_dir_name()

    def get_dir_name(self):
        if self.final_decision == 'include':
            ret = '_'.join([
                self.model_class.__name__,
                self.pretrained_weights_name
            ])
        else:
            ret = '_'.join([
                self.model_class.__name__,
                self.pretrained_weights_name,
                'final-decision-' + self.final_decision
            ])
        if self.lemmatise:
            ret += '_lemmatise'
        if self.lowercase:
            ret += '_lowercase'
        if self.remove_stopwords:
            ret += '_stopwrod_removal'
        if self.punctuation_removal:
            ret += '_punctuation_removal'
        return ret

    # def reviews_to_embeddings(self, _):
    #     raise Exception('Not Implemented')

    def tokenised_review_to_embeddings(self, review):
        batch_input_ids = self.tokenizer.batch_encode_plus(review, pad_to_max_length=True, return_tensors='pt')
        for key, tensor in batch_input_ids.items():
            batch_input_ids[key] = tensor.to(self.device)
        with torch.no_grad():
            review_embeddings = self.model(**batch_input_ids)

        # --clear from GPU memory--
        for tensor in batch_input_ids.values():
            del tensor
        embeddings = review_embeddings[1].cpu()
        for emb in review_embeddings:
            del emb
        torch.cuda.empty_cache()
        return embeddings
        # -------------------------

    def _create_embeddings(self):
        self.model.to(self.device)
        cnt = 0
        for review in self.paper_reviews:
            embeddings = self.tokenised_review_to_embeddings(review)
            self.embeddings_from_reviews.append(embeddings)
            cnt += 1
            print(cnt)
        return self.embeddings_from_reviews

    def get_embeddings_from_reviews(self):
        self.read_reviews_only_text()
        self._get_full_review_stats()
        self.paper_reviews = self.preprocessor.preprocess(self.paper_reviews)
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

    def read_average_scores(self, aspect='RECOMMENDATION'):
        mean_scores_per_paper = []
        for i, file in enumerate(self.files):
            with open(file) as json_file:
                full_reviews = json.load(json_file)
                scores_for_specific_paper = []
                reviews_for_specific_paper = [] # to remove duplicates
                for review in full_reviews['reviews']:
                    if aspect not in review:
                        continue
                    if self.meta_reviews or not review['IS_META_REVIEW']:
                        if not self.remove_duplicates or review['comments'] not in reviews_for_specific_paper:
                            scores_for_specific_paper.append(review[aspect])
                            reviews_for_specific_paper.append(review['comments'])
            if not scores_for_specific_paper:
                mean = 2.5
            else:
                mean = np.mean(scores_for_specific_paper)
            mean_scores_per_paper.append(mean)
        return torch.tensor(mean_scores_per_paper, dtype=torch.float)


class LSTMPerReviewDataLoader(LSTMEmbeddingLoader):
    def __init__(self, *args, aspect=None, **kwargs):
        super(LSTMPerReviewDataLoader, self).__init__(*args, **kwargs)
        if not aspect:
            self.path = self.DATA_ROOT / 'lstm_per_review_embeddings'/ self.conference / 'pre_trained' / self.get_dir_name()
        else:
            self.path = self.DATA_ROOT / 'lstm_per_review_embeddings'/ aspect / self.conference / 'pre_trained' / self.get_dir_name()
            
        self.aspect_scores = []
        self.labels = []
        self.aspect = aspect
        self.recommendation_scores = []
        self.peer_review_to_paper_ids = []
    
    def exclude(self, review):
        # exclude empty reviews and final decision if specified
        return (not self.allow_empty and not review['comments'])\
               or ('RECOMMENDATION' not in review)\
               or (self.aspect and self.aspect not in review)\
               or (self.final_decision == 'exclude' and review.get('TITLE') == 'ICLR committee final decision')

    """
    :returns only the text of the reviews (comments attribute of the json)
    """
    def read_reviews_only_text(self):
        for i, file in enumerate(self.files):
            with open(file) as json_file:
                full_reviews = json.load(json_file)
                reviews_for_specific_paper = []
                self.labels.append(full_reviews['accepted'])
                for review in full_reviews['reviews']:
                    if self.exclude(review):
                        continue
                    if self.final_decision == 'only':
                        raise Exception('Final decisions should be excluded from this data loader')
                    elif self.meta_reviews or not review['IS_META_REVIEW']:
                        if not self.remove_duplicates or review['comments'] not in reviews_for_specific_paper:
                            reviews_for_specific_paper.append(review['comments'])
                            self.paper_reviews.append(review['comments'])
                            self.recommendation_scores.append(review['RECOMMENDATION'])
                            self.peer_review_to_paper_ids.append(i)
                            if self.aspect:
                                self.aspect_scores.append(review[self.aspect])

        return self.paper_reviews
    
    def read_aspect_scores(self):
        return  torch.tensor(self.aspect_scores, dtype=torch.float)
    
    def read_labels(self, task='classification'):
        if task == 'classification':
            return (torch.tensor(self.recommendation_scores) > 5).float()
        else:
            return torch.tensor(self.recommendation_scores).float()

if __name__ == '__main__':
    device_idx = input("GPU: ")
    GPU = True
    if GPU:
        device = torch.device("cuda:" + device_idx if torch.cuda.is_available() else "cpu")
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
