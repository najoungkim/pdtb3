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

    def __init__(self, guid, text_a, text_b=None, label=None, masked_conn=None):
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
        self.masked_conn = masked_conn


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, alt_label_id=None, masked_conn_id=None, conn_id=None, guid=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.alt_label_id = alt_label_id
        self.masked_conn_id = masked_conn_id
        self.conn_id = conn_id
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


class PDTB2Level1Processor(DataProcessor):
    """Processor for the PDTB2 data set, 4-way classification (L1)"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_multi(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples_multi(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test")

    def get_labels(self):
        """See base class."""
        return ["Expansion", "Contingency", "Comparison", "Temporal"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[6]
            text_b = line[7]
            label = line[4]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_examples_multi(self, lines, set_type):
        """Creates examples for the dev and test sets. (multiple correct answers possible) """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label1 = line[4]
            label2 = line[5]
            if label2 == "None" or label2 not in self.get_labels(): label2 = label1
 
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=[label1, label2]))

        return examples

class PDTB2Level2Processor(DataProcessor):
    """Processor for the PDTB data set (11-way, Lin or Ji)"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_multi(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples_multi(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test")

    def get_labels(self):
        """See base class."""
        return ['Temporal.Asynchronous', 'Temporal.Synchrony', 'Contingency.Cause',
                'Contingency.Pragmatic cause', 'Comparison.Contrast', 'Comparison.Concession',
                'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Restatement',
                'Expansion.Alternative','Expansion.List']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = line[6]
            text_b = line[7]
            label = line[4]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_multi(self, lines, set_type):
        """Creates examples for the dev and test sets. (multiple correct answers possible) """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = line[7]
            text_b = line[8]
            label1 = line[4]
            label2 = line[5]
            if label2 == "None" or label2 not in self.get_labels(): label2 = label1
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=[label1, label2]))

        return examples

class PDTB2Level2ProcessorS1(PDTB2Level2Processor):
    """Processor for the PDTB data set (11-way, Lin or Ji)"""

    def _create_examples(self, lines, set_type):
        """Creates examples for the training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = line[6]
            text_b = ""
            label = line[4]
                
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_examples_multi(self, lines, set_type):
        """Creates examples for the dev and test sets. (multiple correct answers possible) """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = line[7]
            text_b = ""
            label1 = line[4]
            label2 = line[5]
            if label2 == "None" or label2 not in self.get_labels(): label2 = label1
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=[label1, label2]))

        return examples

class PDTB2Level2ProcessorS2(PDTB2Level2Processor):
    """Processor for the PDTB data set (11-way, Lin or Ji)"""

    def _create_examples(self, lines, set_type):
        """Creates examples for the training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = ""
            text_b = line[7]
            label = line[4]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_examples_multi(self, lines, set_type):
        """Creates examples for the dev and test sets. (multiple correct answers possible) """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = ""
            text_b = line[8]
            label1 = line[4]
            label2 = line[5]
            if label2 == "None" or label2 not in self.get_labels(): label2 = label1
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=[label1, label2]))

        return examples

class PDTB3Level1Processor(DataProcessor):
    """Processor for the PDTB 3.0 dataset (4-way classification)"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_multi(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples_multi(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test")

    def get_labels(self):
        """See base class."""
        return ["Expansion", "Contingency", "Comparison", "Temporal"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[6]
            text_b = line[7]
            label = line[4]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_examples_multi(self, lines, set_type):
        """Creates examples for dev/test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label1 = line[4]
            label2 = line[5]
            if label2 == "None" or label2 not in self.get_labels(): label2 = label1

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=[label1, label2]))
        return examples

class PDTB3Level2Processor(DataProcessor):
    """Processor for the PDTB 3.0 (14-way)"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_multi(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples_multi(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test")

    def get_labels(self):
        """See base class."""  
        return ['Temporal.Asynchronous', 'Temporal.Synchronous', 'Contingency.Cause',
                'Contingency.Cause+Belief', 'Contingency.Condition', 'Contingency.Purpose',
                'Comparison.Contrast', 'Comparison.Concession',
                'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Equivalence',
                'Expansion.Level-of-detail', 'Expansion.Manner', 'Expansion.Substitution']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = line[6]
            text_b = line[7]
            label = line[4]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_multi(self, lines, set_type):
        """Creates examples for the dev and test sets. (multiple correct answers possible) """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = line[7]
            text_b = line[8]
            label1 = line[4]
            label2 = line[5]
            if label2 == "None" or label2 not in self.get_labels(): label2 = label1
           
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=[label1, label2]))
 
        return examples

class PDTB3Level2ProcessorS1(PDTB3Level2Processor):
    """Processor for the PDTB 3.0 (14-way)"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_multi(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples_multi(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = line[6]
            text_b = ""
            label = line[4]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_multi(self, lines, set_type):
        """Creates examples for the dev and test sets. (multiple correct answers possible) """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = line[7]
            text_b = ""
            label1 = line[4]
            label2 = line[5]
            if label2 == "None" or label2 not in self.get_labels(): label2 = label1
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=[label1, label2]))
 
        return examples

class PDTB3Level2ProcessorS2(PDTB3Level2Processor):
    """Processor for the PDTB 3.0 (14-way classification)"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_multi(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples_multi(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = ""
            text_b = line[7]
            label = line[4]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_multi(self, lines, set_type):
        """Creates examples for the dev and test sets. (multiple correct answers possible) """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = ""
            text_b = line[8]
            label1 = line[4]
            label2 = line[5]
            if label2 == "None" or label2 not in self.get_labels(): label2 = label1
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=[label1, label2]))
 
        return examples

class PDTB3Level2Level3Processor(DataProcessor):
    """Processor for the PDTB 3.0 (18-way)"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_multi(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples_multi(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test")

    def get_labels(self):
        """See base class."""  
        return [
                'Temporal.Asynchronous.Precedence', 'Temporal.Asynchronous.Succession', 'Temporal.Synchronous', 'Contingency.Cause.Reason',
                'Contingency.Cause.Result', 'Contingency.Cause+Belief', 'Contingency.Condition', 
                'Contingency.Purpose', 'Comparison.Contrast', 'Comparison.Concession',
                'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Equivalence',
                'Expansion.Level-of-detail.Arg1-as-detail', 'Expansion.Level-of-detail.Arg2-as-detail',
                'Expansion.Manner.Arg1-as-manner', 'Expansion.Manner.Arg2-as-manner', 'Expansion.Substitution']


    def _create_examples(self, lines, set_type):
        """Creates examples for the training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = line[6]
            text_b = line[7]
            label = line[4]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_multi(self, lines, set_type):
        """Creates examples for the dev and test sets. (multiple correct answers possible) """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = line[7]
            text_b = line[8]
            label1 = line[4]
            label2 = line[5]
            if label2 == "None" or label2 not in self.get_labels(): label2 = label1
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=[label1, label2]))
 
        return examples

class PDTB3Level2Level3ProcessorS1(PDTB3Level2Level3Processor):
    """Processor for the PDTB 3.0 (18-way)"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_multi(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples_multi(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = line[6]
            text_b = ""
            label = line[4]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_multi(self, lines, set_type):
        """Creates examples for the dev and test sets. (multiple correct answers possible) """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = line[7]
            text_b = ""
            label1 = line[4]
            label2 = line[5]
            if label2 == "None" or label2 not in self.get_labels(): label2 = label1
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=[label1, label2]))
 
        return examples

class PDTB3Level2Level3ProcessorS2(PDTB3Level2Level3Processor):
    """Processor for the PDTB 3.0 (18-way)"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_multi(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples_multi(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = ""
            text_b = line[7]
            label = line[4]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_multi(self, lines, set_type):
        """Creates examples for the dev and test sets. (multiple correct answers possible) """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = ""
            text_b = line[8]
            label1 = line[4]
            label2 = line[5]
            if label2 == "None" or label2 not in self.get_labels(): label2 = label1
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=[label1, label2]))
 
        return examples



def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
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

        if output_mode == "classification":
            if alternative_labels:
                label_id = label_map[example.label[0]]
                alt_label_id = label_map[example.label[1]]
            else:
                label_id = label_map[example.label]
                alt_label_id = None

        elif output_mode == "regression":
            label_id = float(example.label)
            alt_label_id = None
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            if alt_label_id is not None:
                logger.info("alternative_label: %s (id = %d)" % (example.label[1], alt_label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              alt_label_id=alt_label_id,
                              guid=example.guid))
    return features

def convert_examples_to_features_lm(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]', pad_token=0,
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
            tokens_b = [mask_token] + tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            raise ValueError("Need text b") 

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
        
        #masked_conn_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.masked_conn))[0]
        mask_idx = tokens.index(mask_token)
        masked_conn_id = tokenizer.convert_tokens_to_ids(tokens[0:mask_idx] + [tokenizer.tokenize(example.masked_conn)[0]] + tokens[mask_idx + 1:])

        assert len(masked_conn_id) == len(input_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)

        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            masked_conn_id = ([pad_token] * padding_length) + masked_conn_id
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            masked_conn_id = masked_conn_id + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length, "{}, {}, {}".format(len(input_ids), max_seq_length, tokens)
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            if alternative_labels:
                label_id = label_map[example.label[0]]
                alt_label_id = label_map[example.label[1]]
            else:
                label_id = label_map[example.label]
                alt_label_id = None

        elif output_mode == "regression":
            label_id = float(example.label)
            alt_label_id = None
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("masked_conn: %s (id = %d)" % (example.masked_conn, masked_conn_id[mask_idx]))
            if alt_label_id is not None:
                logger.info("alternative_label: %s (id = %d)" % (example.label[1], alt_label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              alt_label_id=alt_label_id,
                              masked_conn_id=masked_conn_id,
                              guid=example.guid))
    return features

def convert_examples_to_features_conn(examples, label_list, conn_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]', pad_token=0,
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
    conn_map = {conn : i for i, conn in enumerate(conn_list)}

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
            raise ValueError("Need text b") 

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

        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length, "{}, {}, {}".format(len(input_ids), max_seq_length, tokens)
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            if alternative_labels:
                label_id = label_map[example.label[0]]
                alt_label_id = label_map[example.label[1]]
            else:
                label_id = label_map[example.label]
                alt_label_id = None
            conn_id = conn_map[example.masked_conn.split()[0].lower()]

        elif output_mode == "regression":
            label_id = float(example.label)
            alt_label_id = None
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("conn: %s (id = %d)" % (example.masked_conn.split()[0], conn_id))
            if alt_label_id is not None:
                logger.info("alternative_label: %s (id = %d)" % (example.label[1], alt_label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              alt_label_id=alt_label_id,
                              conn_id=conn_id,
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

def multi_accuracy(preds, labels, alt_labels):
    acc = (preds == labels).astype(int) + ((alt_labels != labels) & (preds == alt_labels)).astype(int)
    return {"acc": acc.mean()}

def multi_acc_and_macro_f1(preds, labels, alt_labels):
    acc = multi_accuracy(preds, labels, alt_labels)['acc']
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "macro_f1": f1,
        "acc_and_f1": (acc + f1) / 2
    }

def compute_metrics(task_name, preds, labels, alt_labels=None):
    assert len(preds) == len(labels)
    if task_name in ["pdtb2_level1", "pdtb3_level1"]:
        return multi_acc_and_macro_f1(preds, labels, alt_labels)
    elif task_name in ["pdtb2_level2", "pdtb2_level2_s1", "pdtb2_level2_s2", "pdtb3_level2", "pdtb3_level2_s1", "pdtb3_level2_s2", "pdtb3_level2_level3", "pdtb3_level2_level3_s1", "pdtb3_level2_level3_s2"]:
        return multi_accuracy(preds, labels, alt_labels)
    else:
        raise KeyError(task_name)

processors = {
    "pdtb2_level1": PDTB2Level1Processor,
    "pdtb3_level1": PDTB3Level1Processor,
    "pdtb2_level2": PDTB2Level2Processor,
    "pdtb2_level2_s1": PDTB2Level2ProcessorS1,
    "pdtb2_level2_s2": PDTB2Level2ProcessorS2,
    "pdtb3_level2": PDTB3Level2Processor,
    "pdtb3_level2_s1": PDTB3Level2ProcessorS1,
    "pdtb3_level2_s2": PDTB3Level2ProcessorS2,
    "pdtb3_level2_level3": PDTB3Level2Level3Processor,
    "pdtb3_level2_level3_s1": PDTB3Level2Level3ProcessorS1,
    "pdtb3_level2_level3_s2": PDTB3Level2Level3ProcessorS2,
}

output_modes = {
    "pdtb2_level1": "classification",
    "pdtb3_level1": "classification",
    "pdtb2_level2": "classification",
    "pdtb2_level2_s1": "classification",
    "pdtb2_level2_s2": "classification",
    "pdtb3_level2": "classification",
    "pdtb3_level2_s1": "classification",
    "pdtb3_level2_s2": "classification",
    "pdtb3_level2_level3": "classification",
    "pdtb3_level2_level3_s1": "classification",
    "pdtb3_level2_level3_s2": "classification",
}

PDTB_TASKS_NUM_LABELS = {
    "pdtb2_level1": 4,
    "pdtb3_level1": 4,
    "pdtb2_level2": 11, 
    "pdtb2_level2_s1": 11, 
    "pdtb2_level2_s2": 11,
    "pdtb3_level2": 14,
    "pdtb3_level2_s1": 14,
    "pdtb3_level2_s2": 14,
    "pdtb3_level2_level3": 18,
    "pdtb3_level2_level3_s1": 18,
    "pdtb3_level2_level3_s2": 18,
}
