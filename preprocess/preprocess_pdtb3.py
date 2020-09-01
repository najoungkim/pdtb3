import re
import os
import random
import argparse
from utils import *

SELECTED_SENSES_PDTB3 = frozenset([
    'Temporal.Asynchronous', 'Temporal.Synchronous', 'Contingency.Cause',
    'Contingency.Cause+Belief', 'Contingency.Condition', 'Contingency.Purpose',
    'Comparison.Contrast', 'Comparison.Concession',
    'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Equivalence',
    'Expansion.Level-of-detail', 'Expansion.Manner', 'Expansion.Substitution'

])

UNSELECTED_SENSES_PDTB3 = frozenset([
    'Comparison.Concession+SpeechAct', 'Comparison.Similarity',
    'Contingency.Cause+SpeechAct', 'Contingency.Condition+SpeechAct',
    'Expansion.Disjunction', 'Expansion.Exception',
    'Comparison', 'Contingency', 'Expansion', 'Temporal', 'Contingency.Negative-condition'

])

SELECTED_SENSES_L3_PDTB3 = frozenset([
    'Temporal.Asynchronous.Precedence', 'Temporal.Asynchronous.Succession',
    'Temporal.Synchronous', 'Contingency.Cause.Reason',
    'Contingency.Cause.Result', 'Contingency.Cause+Belief', 'Contingency.Condition',
    'Contingency.Purpose', 'Comparison.Contrast', 'Comparison.Concession',
    'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Equivalence',
    'Expansion.Level-of-detail.Arg1-as-detail', 'Expansion.Level-of-detail.Arg2-as-detail',
    'Expansion.Manner.Arg1-as-manner', 'Expansion.Manner.Arg2-as-manner', 'Expansion.Substitution'

])


def pdtb3_process_raw_sections(data_path, write_path):
    """Processes raw PDTB 3.0 data from LDC and saves individual sections to file."""

    annotation_path = os.path.join(data_path, 'gold/')
    text_path = os.path.join(data_path, 'raw/')

    for dirname in os.listdir(annotation_path):
        lines_to_write = []
        lines_to_write.append(tab_delimited(['section', 'filename',
                                             'relation_type', 'arg1', 'arg2',
                                             'conn1', 'conn1_sense1', 'conn1_sense2',
                                             'conn2', 'conn2_sense1', 'conn2_sense2']))

        annotation_dir = os.path.join(annotation_path, dirname)
        text_dir = os.path.join(text_path, dirname)
        for filename in os.listdir(annotation_dir):
            with open(os.path.join(annotation_dir, filename), encoding='latin1') as f:
                annotation_data = f.readlines()
            with open(os.path.join(text_dir, filename), encoding='latin1') as f:
                text_data = f.read()

            lines_to_write.extend(process_file(annotation_data, text_data,
                                               dirname, filename))

        if not os.path.exists(write_path):
            os.makedirs(write_path)

        with open(f'{write_path}/{dirname}.tsv', 'w') as f:
            f.writelines(lines_to_write)
            print(f'Wrote Section {dirname}'.format(dirname))


def process_file(annotation_data, text_data, dirname, filename):
    """Processes a single file of annotated examples in PDTB 3.0."""

    lines_to_write = []
    for line in annotation_data:
        data_tuple = process_line(line, text_data)
        if data_tuple:
            conn1, conn1_sense1, conn1_sense2, \
            conn2, conn2_sense1, conn2_sense2, \
            arg1_str, arg2_str, relation_type = data_tuple

            lines_to_write.append(tab_delimited([dirname, filename, relation_type,
                                                 arg1_str, arg2_str,
                                                 conn1, conn1_sense1, conn1_sense2,
                                                 conn2, conn2_sense1, conn2_sense2]))
    return lines_to_write


def process_line(line, text_data):
    """Processes a single line of annotated example in PDTB 3.0."""
    args = line.split('|')
    relation_type = args[0]

    if relation_type != 'Implicit':
        return None

    conn1 = args[7]
    conn1_sense1 = args[8]
    conn1_sense2 = args[9]
    conn2 = args[10]
    conn2_sense1 = args[11]
    conn2_sense2 = args[12]

    arg1_idx = args[14].split(';')
    arg2_idx = args[20].split(';')

    arg1_str = []

    # Arguments may be discontiguous spans.
    for pairs in arg1_idx:
        arg1_i, arg1_j = pairs.split('..')
        arg1 = text_data[int(arg1_i):int(arg1_j)+1]
        arg1_str.append(re.sub('\n', ' ', arg1))

    arg2_str = []
    for pairs in arg2_idx:
        if pairs == '':
            continue
        arg2_i, arg2_j = pairs.split('..')
        arg2 = text_data[int(arg2_i):int(arg2_j)+1]
        arg2_str.append(re.sub('\n', ' ', arg2))

    return (conn1, conn1_sense1, conn1_sense2,
            conn2, conn2_sense1, conn2_sense2,
            ' '.join(arg1_str), ' '.join(arg2_str),
            relation_type)


def pdtb3_make_splits_xval(data_path, write_path, random_sections=False, level='L2'):
    """Creates cross-validation splits.

    Note that this method only creates splits based on 14-way classification.
    That is, it will skip low-count labels even if they are relevant
    for the 4-way classification.

    Args:
        data_path: Path containing the PDTB 3.0 data preprocessed into sections.
        write_path: Path to write the splits to.
        random_sections: Whether to create randomized splits or fixed splits.
    """
    sections = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
                '22', '23', '24']
    dev_sections = []
    test_sections = []
    train_sections = []

    means_d = {'train': 0, 'dev': 0, 'test': 0}

    if not random_sections:
        for i in range(0, 25, 2):
            dev_sections.append([sections[i], sections[(i+1)%25]])
            test_sections.append([sections[(i+23)%25], sections[(i+24)%25]])
            train_sections.append([sections[(i+j)%25] for j in range(2, 23)])
    else:
        seed = 111
        random.seed(seed)
        for i in range(0, 13, 1):
            random.seed(seed + i)
            random.shuffle(sections)
            dev_sections.append(sections[0:2])
            test_sections.append(sections[2:4])
            train_sections.append(sections[4:])

    print(test_sections)

    for fold_no, (dev, test, train) in enumerate(zip(dev_sections[:-1],
                                                     test_sections[:-1],
                                                     train_sections[:-1])):
        label_d = {}
        all_splits = dev + test + train
        assert len(set(all_splits)) == 25

        split_d = {'train': train, 'dev': dev, 'test': test}
        lines_d = {'train': [], 'dev': [], 'test': []}

        for split, sections in split_d.items():
            for section in sections:
                process_section(data_path, section, split, lines_d, label_d, level)

        for split, lines in lines_d.items():
            means_d[split] += len(lines)-1

        print(sorted(label_d.items()))

        # Write to file
        write_path_fold = os.path.join(write_path, f'fold_{fold_no + 1}')
        write_to_file(lines_d, write_path_fold)

    for split, total in means_d.items():
        print(f'Total: {total}')
        print(f'Mean {split}: {total/len(dev_sections[:-1])}')


def process_section(data_path, section, split, lines_d, label_d, level='L2'):
    """Processes a single PDTB section."""
    with open(data_path + '/' + section + '.tsv') as f:
        data = f.readlines()

    for line in data[1:]:
        section, file_no, category, arg1, arg2, \
        conn1, conn1_sense1, conn1_sense2, \
        conn2, conn2_sense1, conn2_sense2 = line.rstrip('\n').split('\t')

        sense1 = (conn1_sense1, conn1)
        sense2 = (conn1_sense2, conn1)
        sense3 = (conn2_sense1, conn2)
        sense4 = (conn2_sense2, conn2)

        # Use list instead of set to preserve order
        sense_list = [sense1, sense2, sense3, sense4]
        if level == 'L2':
            formatted_sense_list = format_sense_l2(sense_list)
        elif level == 'L3':
            formatted_sense_list = format_sense_l3(sense_list)
        else:
            raise ValueError('Level must be L2 or L3')

        # No useable senses
        if not formatted_sense_list:
            continue

        if split == 'train':
            for sense, conn, sense_full in formatted_sense_list:
                lines_d[split].append(tab_delimited([split, section, file_no,
                                                     sense, category,
                                                     arg1, arg2, conn, sense_full]))
                label_d[sense] = label_d.get(sense, 0) + 1

        else:
            if len(formatted_sense_list) == 1:
                formatted_sense_list.append((None, None, None))
            sense_paired = zip(formatted_sense_list[0], formatted_sense_list[1])
            senses, conns, senses_full = sense_paired
            lines_d[split].append(tab_delimited([split, section, file_no,
                                                 senses[0], senses[1], category,
                                                 arg1, arg2, conns[0],
                                                 senses_full[0], conns[1], senses_full[1]]))

            label_d[senses[0]] = label_d.get(senses[0], 0) + 1
            if senses[1] is not None:
                label_d[senses[1]] = label_d.get(senses[1], 0) + 1


def format_sense_l2(sense_list):
    formatted_sense_list = []
    for sense_full, conn in sense_list:
        if sense_full is not None:
            sense = '.'.join(sense_full.split('.')[0:2])
            if (sense not in [s for s, c, sf in formatted_sense_list] and
                    sense in SELECTED_SENSES_PDTB3):
                formatted_sense_list.append((sense, conn, sense_full))
    return formatted_sense_list


def format_sense_l3(sense_list):
    formatted_sense_list = []
    for sense_full, conn in sense_list:
        if sense_full is not None:
            sense_l2 = '.'.join(sense_full.split('.')[0:2])
            if (sense_full not in [s for s, c, sf in formatted_sense_list] and
                    sense_full in SELECTED_SENSES_L3_PDTB3):
                formatted_sense_list.append((sense_full, conn, sense_full))
            elif (sense_l2 not in [s for s, c, sf in formatted_sense_list] and
                  sense_l2 in SELECTED_SENSES_L3_PDTB3):
                formatted_sense_list.append((sense_l2, conn, sense_full))
    return formatted_sense_list


def pdtb3_make_splits_l1(data_path, write_path):
    """Creates a split for L1 classification using specifications from Ji & Eistenstein (2015)."""
    TRAIN = ['02', '03', '04', '05', '06', '07', '08', '09', '10',
             '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    DEV = ['00', '01']
    TEST = ['21', '22']

    label_d = {}

    dev_sections = [DEV]
    test_sections = [TEST]
    train_sections = [TRAIN]

    for (dev, test, train) in zip(dev_sections, test_sections, train_sections):

        split_d = {'train': train, 'dev': dev, 'test': test}
        lines_d = {'train': [], 'dev': [], 'test': []}

        label_d = {}
        for split, sections in split_d.items():
            for section in sections:
                process_section_l1(data_path, section, split, lines_d, label_d)

        # Write to file
        write_to_file(lines_d, write_path)
        print(label_d)


def process_section_l1(data_path, section, split, lines_d, label_d):
    with open(data_path + '/' + section + '.tsv') as f:
        data = f.readlines()

    for line in data[1:]:

        section, file_no, category, arg1, arg2, \
        conn1, conn1_sense1, conn1_sense2, \
        conn2, conn2_sense1, conn2_sense2 = line.rstrip('\n').split('\t')

        sense1 = (conn1_sense1, conn1)
        sense2 = (conn1_sense2, conn1)
        sense3 = (conn2_sense1, conn2)
        sense4 = (conn2_sense2, conn2)

        sense_list = [sense1, sense2, sense3, sense4]
        formatted_sense_list = []
        for sense_full, conn in sense_list:
            if sense_full is not None:
                sense = sense_full.split('.')[0]
                if sense not in [s for s, c, sf in formatted_sense_list] and sense:
                    formatted_sense_list.append((sense, conn, sense_full))

        # Should be at least one sense
        assert formatted_sense_list
        assert len(formatted_sense_list) <= 2, formatted_sense_list
        if len(formatted_sense_list) == 2:
            assert formatted_sense_list[0] != formatted_sense_list[1]

        if split == 'train':
            for sense, conn, sense_full in formatted_sense_list:
                lines_d[split].append(tab_delimited([split, section, file_no,
                                                     sense, category, arg1,
                                                     arg2, conn, sense_full]))
                label_d[sense] = label_d.get(sense, 0) + 1

        else:
            if len(formatted_sense_list) == 1:
                formatted_sense_list.append((None, None, None))
            sense_paired = zip(formatted_sense_list[0], formatted_sense_list[1])
            senses, conns, senses_full = sense_paired
            lines_d[split].append(tab_delimited([split, section, file_no,
                                                 senses[0], senses[1], category,
                                                 arg1, arg2, conns[0], senses_full[0],
                                                 conns[1], senses_full[1]]))

            label_d[senses[0]] = label_d.get(senses[0], 0) + 1
            if senses[1] is not None:
                label_d[senses[1]] = label_d.get(senses[1], 0) + 1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default=None, type=str, required=True,
                        help='Path to a directory containing raw and gold PDTB 3.0 files.\
                              Refer to README.md about obtaining this file.')
    parser.add_argument('--output_dir', default=None, type=str, required=True,
                        help='Path to output directory \
                              where the preprocessed dataset will be stored.')
    parser.add_argument('--split', default=None, type=str, required=True,
                        help='Type of split to create. Should be one of \
                              "L2_xval", "L3_xval" or "L1_ji".')

    args = parser.parse_args()

    # If sections have not been preprocessed, process them first.
    sections_data_dir = os.path.join(args.data_dir, 'sections/')
    if not os.path.exists(sections_data_dir):
        pdtb3_process_raw_sections(args.data_dir, sections_data_dir)

    # Create splits.
    if args.split == 'L2_xval':
        pdtb3_make_splits_xval(sections_data_dir, args.output_dir, level='L2')
    elif args.split == 'L3_xval':
        pdtb3_make_splits_xval(sections_data_dir, args.output_dir, level='L3')
    elif args.split == 'L1_ji':
        pdtb3_make_splits_l1(sections_data_dir, args.output_dir)
    else:
        raise ValueError('--split must be one of "L2_xval", "L3_xval", "L1_ji".')


if __name__ == '__main__':
    main()
