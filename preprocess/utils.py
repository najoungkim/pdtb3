import os

def tab_delimited(list_to_write):
    """Format list of elements in to a tab-delimited line"""
    list_to_write = [str(x) for x in list_to_write]
    return '\t'.join(list_to_write) + '\n'


def write_to_file(lines_d, write_path):
    """Save splits into file"""
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    for split, lines in lines_d.items():
        write_fname = '{}.tsv'.format(split)
        with open(os.path.join(write_path, write_fname), 'w') as f:
            if split == 'train':
                f.write(tab_delimited(['idx', 'split', 'section', 'file_number',
                                       'label', 'category', 'arg1', 'arg2', 'conn', 'full_sense']))
            else:
                f.write(tab_delimited(['idx', 'split', 'section', 'file_number',
                                       'label1', 'label2', 'category', 'arg1', 'arg2',
                                       'conn1', 'full_sense1', 'conn2', 'full_sense2']))
            for i, line in enumerate(lines):
                line_to_write = '{}\t{}'.format(i, line)
                f.write(line_to_write)


