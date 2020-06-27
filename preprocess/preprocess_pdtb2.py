import os
import argparse
from pdtb2 import CorpusReader
from utils import *

SELECTED_SENSES_PDTB2 = set([
    'Temporal.Asynchronous', 'Temporal.Synchrony', 'Contingency.Cause',
    'Contingency.Pragmatic cause', 'Comparison.Contrast', 'Comparison.Concession',
    'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Restatement',
    'Expansion.Alternative', 'Expansion.List'
])


def pdtb2_make_splits_xval(path, write_path):
    """Make 12 cross-validation splits for PDTB 2.0"""

    sections = ['00', '01', '02', '03', '04', '05', '06', '07', '08',
                '09', '10', '11', '12', '13', '14', '15', '16', '17',
                '18', '19', '20', '21', '22', '23', '24']

    dev_sections = []
    test_sections = []
    train_sections = []

    for i in range(0, 25, 2):
        dev_sections.append([sections[i], sections[(i+1)%25]])
        test_sections.append([sections[(i+23)%25], sections[(i+24)%25]])
        train_sections.append([sections[(i+j)%25] for j in range(2, 23)])

    means_d = {'train':0, 'dev':0, 'test': 0}

    pdtb_data = list(CorpusReader(path).iter_data())
    for fold_no, (dev, test, train) in enumerate(zip(dev_sections[:-1],
                                                     test_sections[:-1],
                                                     train_sections[:-1])):
        all_splits = dev + test + train
        assert len(set(all_splits)) == 25

        split_d = {'train': train, 'dev': dev, 'test': test}
        lines_d = {'train': [], 'dev': [], 'test': []}
        label_d = {}

        for corpus in pdtb_data:
            for split, sections in split_d.items():
                if corpus.Relation == 'Implicit' and corpus.Section in sections:
                    sense1 = (corpus.ConnHeadSemClass1, corpus.Conn1)
                    sense2 = (corpus.ConnHeadSemClass2, corpus.Conn1)
                    sense3 = (corpus.Conn2SemClass1, corpus.Conn2)
                    sense4 = (corpus.Conn2SemClass2, corpus.Conn2)

                    # use list instead of set to preserve order
                    sense_list = [sense1, sense2, sense3, sense4]
                    formatted_sense_list = []
                    for sense_full, conn in sense_list:
                        if sense_full is not None:
                            sense = '.'.join(sense_full.split('.')[0:2])
                            if (sense not in [s for s, c, sf in formatted_sense_list] and
                                sense in SELECTED_SENSES_PDTB2):
                                formatted_sense_list.append((sense, conn, sense_full))

                    # No useable senses
                    if len(formatted_sense_list) == 0:
                        continue

                    arg1 = corpus.Arg1_RawText
                    arg2 = corpus.Arg2_RawText

                    if split == 'train':
                        for sense, conn, sense_full in formatted_sense_list:
                            lines_d[split].append(tab_delimited([split, corpus.Section,
                                                                 corpus.FileNumber,
                                                                 sense, corpus.Relation,
                                                                 arg1, arg2,
                                                                 conn, sense_full]))
                            label_d[sense] = label_d.get(sense, 0) + 1

                    else:
                        if len(formatted_sense_list) == 1:
                            formatted_sense_list.append((None, None, None))
                        sense_paired = zip(formatted_sense_list[0], formatted_sense_list[1])
                        senses, conns, senses_full = sense_paired
                        lines_d[split].append(tab_delimited([split, corpus.Section,
                                                             corpus.FileNumber,
                                                             senses[0], senses[1],
                                                             corpus.Relation, arg1, arg2,
                                                             conns[0], senses_full[0],
                                                             conns[1], senses_full[1]]))
                        label_d[senses[0]] = label_d.get(senses[0], 0) + 1
                        if senses[1] is not None:
                            label_d[senses[1]] = label_d.get(senses[1], 0) + 1

                    assert len(formatted_sense_list) <= 2
                    if len(formatted_sense_list) == 2:
                        if formatted_sense_list[0][0] == formatted_sense_list[1][0]:
                            print('redundant!')

        for split, lines in lines_d.items():
            means_d[split] += len(lines)-1

        # Write to file
        write_path_fold = os.path.join(write_path,
                                       'fold_{}'.format(fold_no+1))
        write_to_file(lines_d, write_path_fold)

        print('Cross-validation fold {}'.format(fold_no+1))
        print('Label counts: ', label_d)

        total = 0
        for _, count in label_d.items():
            total += count

        print('Total: ', total)

    for split, total in means_d.items():
        print('Mean {}: {}'.format(split, total/len(dev_sections[:-1])))


def pdtb2_make_splits_single(path, write_path, split_name):
    """
    Make single standard split for PDTB 2.0.
    split_name should be one of 'ji', 'lin', 'patterson'.
    'ji': Split from Ji & Eistenstein (2015), 2-20 train, 0-1 dev, 21-22 test
    'lin': Split from Lin et al. (2009) and dev set as indicated by Qin et al. (2017),
           2-21 train, 22 dev, 23 test
    'patterson': Split from Patterson & Kehler (2013), 2-22 train, 0-1 dev, 23-24 test
    """

    if split_name == 'ji':
        train_sections = ['02', '03', '04', '05', '06', '07', '08',
                          '09', '10', '11', '12', '13', '14', '15',
                          '16', '17', '18', '19', '20']
        dev_sections = ['00', '01']
        test_sections = ['21', '22']

    elif split_name == 'lin':
        train_sections = ['02', '03', '04', '05', '06', '07', '08',
                          '09', '10', '11', '12', '13', '14', '15',
                          '16', '17', '18', '19', '20', '21']
        dev_sections = ['22']
        test_sections = ['23']

    elif split_name == 'patterson':
        train_sections = ['02', '03', '04', '05', '06', '07', '08',
                          '09', '10', '11', '12', '13', '14', '15',
                          '16', '17', '18', '19', '20', '21', '22']
        dev_sections = ['00', '01']
        test_sections = ['23', '24']

    sections = train_sections + dev_sections + test_sections

    lines_d = {'train': [], 'dev': [], 'test': []}
    label_d = {}

    for corpus in CorpusReader(path).iter_data():
        if corpus.Relation == 'Implicit' and corpus.Section in sections:
            sense1 = (corpus.ConnHeadSemClass1, corpus.Conn1)
            sense2 = (corpus.ConnHeadSemClass2, corpus.Conn1)
            sense3 = (corpus.Conn2SemClass1, corpus.Conn2)
            sense4 = (corpus.Conn2SemClass2, corpus.Conn2)

            # use list instead of set to preserve order
            sense_list = [sense1, sense2, sense3, sense4]
            formatted_sense_list = []
            for sense_full, conn in sense_list:
                if sense_full is not None:
                    sense = '.'.join(sense_full.split('.')[0:2])
                    if (sense not in [s for s, c, sf in formatted_sense_list] and
                        sense in SELECTED_SENSES_PDTB2):
                        formatted_sense_list.append((sense, conn, sense_full))

            # No useable senses
            if len(formatted_sense_list) == 0:
                continue

            arg1 = corpus.Arg1_RawText
            arg2 = corpus.Arg2_RawText

            if corpus.Section in train_sections:
                split = 'train'
            elif corpus.Section in dev_sections:
                split = 'dev'
            else:
                split = 'test'

            if split == 'train':
                for sense, conn, sense_full in formatted_sense_list:
                    lines_d[split].append(tab_delimited([split, corpus.Section,
                                                         corpus.FileNumber,
                                                         sense, corpus.Relation,
                                                         arg1, arg2,
                                                         conn, sense_full]))
                    label_d[sense] = label_d.get(sense, 0) + 1

            else:
                if len(formatted_sense_list) == 1:
                    formatted_sense_list.append((None, None, None))
                sense_paired = zip(formatted_sense_list[0], formatted_sense_list[1])
                senses, conns, senses_full = sense_paired
                lines_d[split].append(tab_delimited([split, corpus.Section,
                                                     corpus.FileNumber,
                                                     senses[0], senses[1],
                                                     corpus.Relation, arg1, arg2,
                                                     conns[0], senses_full[0],
                                                     conns[1], senses_full[1]]))

                label_d[senses[0]] = label_d.get(senses[0], 0) + 1
                if senses[1] is not None:
                    label_d[senses[1]] = label_d.get(senses[1], 0) + 1

            assert len(formatted_sense_list) <= 2
            if len(formatted_sense_list) == 2:
                if formatted_sense_list[0][0] == formatted_sense_list[1][0]:
                    raise ValueError('Redundant labels!')

    # Write to file
    write_to_file(lines_d, write_path)
    print('Label counts: ', label_d)


def pdtb2_make_splits_single_l1(path, write_path, split_name):
    """
    Make single standard split for PDTB 2.0, using Level-1 labels (4-way classification).
    split_name should be one of 'ji', 'lin', 'patterson'.
    'ji': Split from Ji & Eistenstein (2015), 2-20 train, 0-1 dev, 21-22 test
    'lin': Split from Lin et al. (2009) and dev set as indicated by Qin et al. (2017),
           2-21 train, 22 dev, 23 test
    'patterson': Split from Patterson & Kehler (2013), 2-22 train, 0-1 dev, 23-24 test
    """
    if split_name == 'ji':
        train_sections = ['02', '03', '04', '05', '06', '07', '08',
                          '09', '10', '11', '12', '13', '14', '15',
                          '16', '17', '18', '19', '20']
        dev_sections = ['00', '01']
        test_sections = ['21', '22']

    elif split_name == 'lin':
        train_sections = ['02', '03', '04', '05', '06', '07', '08',
                          '09', '10', '11', '12', '13', '14', '15',
                          '16', '17', '18', '19', '20', '21']
        dev_sections = ['22']
        test_sections = ['23']

    elif split_name == 'patterson':
        train_sections = ['02', '03', '04', '05', '06', '07', '08',
                          '09', '10', '11', '12', '13', '14', '15',
                          '16', '17', '18', '19', '20', '21', '22']
        dev_sections = ['00', '01']
        test_sections = ['23', '24']

    sections = train_sections + dev_sections + test_sections

    lines_d = {'train': [], 'dev': [], 'test': []}
    label_d = {}

    for corpus in CorpusReader(path).iter_data():
        if corpus.Relation == 'Implicit' and corpus.Section in sections:
            sense1 = (corpus.ConnHeadSemClass1, corpus.Conn1)
            sense2 = (corpus.ConnHeadSemClass2, corpus.Conn1)
            sense3 = (corpus.Conn2SemClass1, corpus.Conn2)
            sense4 = (corpus.Conn2SemClass2, corpus.Conn2)

            # use list instead of set to preserve order
            sense_list = [sense1, sense2, sense3, sense4]
            formatted_sense_list = []
            for sense_full, conn in sense_list:
                if sense_full is not None:
                    sense = sense_full.split('.')[0]
                    if sense not in [s for s, c, sf in formatted_sense_list]:
                        formatted_sense_list.append((sense, conn, sense_full))

            # Should be at least one sense
            assert len(formatted_sense_list) > 0

            arg1 = corpus.Arg1_RawText
            arg2 = corpus.Arg2_RawText

            if corpus.Section in train_sections:
                split = 'train'
            elif corpus.Section in dev_sections:
                split = 'dev'
            else:
                split = 'test'

            if split == 'train':
                for sense, conn, sense_full in formatted_sense_list:
                    lines_d[split].append(tab_delimited([split, corpus.Section,
                                                         corpus.FileNumber,
                                                         sense, corpus.Relation,
                                                         arg1, arg2,
                                                         conn, sense_full]))
                    label_d[sense] = label_d.get(sense, 0) + 1

            else:
                if len(formatted_sense_list) == 1:
                    formatted_sense_list.append((None, None, None))
                sense_paired = zip(formatted_sense_list[0], formatted_sense_list[1])
                senses, conns, senses_full = sense_paired
                lines_d[split].append(tab_delimited([split, corpus.Section,
                                                     corpus.FileNumber,
                                                     senses[0], senses[1],
                                                     corpus.Relation, arg1, arg2,
                                                     conns[0], senses_full[0],
                                                     conns[1], senses_full[1]]))

                label_d[senses[0]] = label_d.get(senses[0], 0) + 1
                if senses[1] is not None:
                    label_d[senses[1]] = label_d.get(senses[1], 0) + 1

            assert len(formatted_sense_list) <= 2
            if len(formatted_sense_list) == 2:
                if formatted_sense_list[0][0] == formatted_sense_list[1][0]:
                    raise ValueError('Redundant labels!')

    # Write to file
    write_to_file(lines_d, write_path)
    print('Label count: ', label_d)


def main():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--data_file', default=None, type=str, required=True,
                        help='Path to .csv formatted PDTB 2.0 file. \
                              Refer to README.md about obtaining this file.')
    parser.add_argument('--output_dir', default=None, type=str, required=True,
                        help='Path to output directory \
                              where the preprocessed dataset will be stored.')
    parser.add_argument('--split', default=None, type=str, required=True,
                        help='What type of split to create. Should be one of "xval", "single".')

    # Optional arguments
    parser.add_argument('--split_name', default=None, type=str,
                        help='Name of the standard split. Used only when \
                              "split" argument is set to "single". \
                              Should be one of "ji", "lin", "patterson".')
    parser.add_argument('--label_type', default='L2', type=str,
                        help='Which label scheme to use. Should be either "L1" or "L2". \
                              Used only when "split" argument is set to "single". Defaults to L2.')

    args = parser.parse_args()

    if args.split == 'xval':
        pdtb2_make_splits_xval(args.data_file, args.output_dir)
    elif args.split == 'single':
        if args.split_name is None:
            raise ValueError('--split_name should be set for standard splits. \
                              Should be one of "ji", "lin", "patterson".')
        if args.label_type == 'L2':
            pdtb2_make_splits_single(args.data_file, args.output_dir, args.split_name)
        elif args.label_type == 'L1':
            pdtb2_make_splits_single_l1(args.data_file, args.output_dir, args.split_name)
        else:
            raise ValueError('--label_type should be either "L1" or "L2".')


if __name__ == '__main__':
    main()
