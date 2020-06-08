# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with PDTB """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

from ast import literal_eval
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id=None, guid=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.guid = guid

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class SNLIPremProcessor(DataProcessor):
    """Processor for the Stanford NLI dataset, premise-only (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        dup_dict = {}
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = ""
            label = line[-1]
            if dup_dict.get(text_a, 0) < 1:
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                dup_dict[text_a] = 1

        return examples

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode=None,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 vector_ops=False):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    alternative_labels = False
    if isinstance(examples[0].label, list):
        alternative_labels = True
    
    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)

        if vector_ops:
            sent_max_len = round(max_seq_length*0.5)
            if len(tokens_a) > sent_max_len - 2: tokens_a = tokens_a[:sent_max_len-2]
            if len(tokens_b) > sent_max_len - 1: tokens_b = tokens_b[:sent_max_len-1]
            if cls_token_at_end:
                s1 = tokens_a + [sep_token]
                s2 = tokens_b + [sep_token] + [cls_token]
            else:
                s1 = [cls_token] + tokens_a + [sep_token]
                s2 = tokens_b + [sep_token]
            s1_padding_length = sent_max_len - len(s1)
            s2_padding_length = sent_max_len - len(s2)

            if pad_on_left:
                s1_input_ids = ([pad_token] * s1_padding_length) + tokenizer.convert_tokens_to_ids(s1)
                s2_input_ids = ([pad_token] * s2_padding_length) + tokenizer.convert_tokens_to_ids(s2)
                s1_input_mask = ([0 if mask_padding_with_zero else 1] * s1_padding_length) + ([1 if mask_padding_with_zero else 0] * len(s1)) 
                s2_input_mask = ([0 if mask_padding_with_zero else 1] * s2_padding_length) + ([1 if mask_padding_with_zero else 0] * len(s2)) 

                if cls_token_at_end:
                    s1_segment_ids = ([pad_token_segment_id] * s1_padding_length) + ([sequence_a_segment_id] * len(s1)) 
                    s2_segment_ids = ([pad_token_segment_id] * s2_padding_length) + ([sequence_b_segment_id] * len(s2)-1) + [cls_token_segment_id] 
                else:
                    s1_segment_ids = ([pad_token_segment_id] * s1_padding_length) + [cls_token_segment_id] + ([sequence_a_segment_id] * (len(s1)-1)) 
                    s2_segment_ids = ([pad_token_segment_id] * s2_padding_length) + ([sequence_b_segment_id] * len(s2)) 

            else:
                s1_input_ids = tokenizer.convert_tokens_to_ids(s1) + ([pad_token] * s1_padding_length)
                s2_input_ids = tokenizer.convert_tokens_to_ids(s2) + ([pad_token] * s2_padding_length)
                s1_input_mask = ([1 if mask_padding_with_zero else 0] * len(s1)) + ([0 if mask_padding_with_zero else 1] * s1_padding_length)
                s2_input_mask = ([1 if mask_padding_with_zero else 0] * len(s2)) + ([0 if mask_padding_with_zero else 1] * s2_padding_length)

                if cls_token_at_end:
                    s1_segment_ids = ([sequence_a_segment_id] * len(s1)) + ([pad_token_segment_id] * s1_padding_length)
                    s2_segment_ids = ([sequence_b_segment_id] * len(s2)-1) + [cls_token_segment_id] + ([pad_token_segment_id] * s2_padding_length)
                else:
                    s1_segment_ids = [cls_token_segment_id] + ([sequence_a_segment_id] * (len(s1)-1)) + ([pad_token_segment_id] * s1_padding_length)
                    s2_segment_ids = ([sequence_b_segment_id] * len(s2)) + ([pad_token_segment_id] * s2_padding_length)

            input_ids = s1_input_ids + s2_input_ids
            input_mask = s1_input_mask + s2_input_mask
            segment_ids = s1_segment_ids + s2_segment_ids

        else:
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length, "{}, {}".format(len(input_ids), len(s1_input_ids))
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              guid=example.guid))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

processors = {
    "snli": SNLIPremProcessor,
}

