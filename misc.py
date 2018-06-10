# -*- coding: utf-8 -*-


def argv2line(argv):
    if len(argv) == 0:
        return 'argv should be sys.argv.'
    else:
        line = argv[0]
        for arg in argv[1:]:
            line += ' ' + arg
        return line


def print_args(args):
    for attr_name in dir(args):
        if attr_name[0] != '_':
            print('{} = {}'.format(attr_name, getattr(args, attr_name)))
