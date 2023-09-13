# Copyright (c) 2023, DeepLink.
import re
import json
import os
import csv


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def parase_args():
    import argparse
    parser = argparse.ArgumentParser(description='dipu op capture tool')
    parser.add_argument('--train_log', type=str, default = 'dipu_train.log', help='Log file of model training (environment variable DIPU_DUMP_OP_ARGS=2 is set)')
    parser.add_argument('--out', type=str, default = 'dipu_ops.csv', help='The file to save the captured operator information')

    args = parser.parse_args()
    return args

def get_all_op_from_train_log(log_content):
    ops = re.findall('(?:--\[.*\]: *[\w\d_]+ *)\s(?:[\t ]+[:\w\d_\.]+:.*\s)*', log_content)
    return ops

def extract_op_arg(op):
    args_info = []
    op_name = re.search('(?<=(\-\-\[)).*(?=\])', op).group().strip()
    args = re.findall(f'(?<={op_name}:).*', op)
    for arg in args:
        index = arg.find(':')
        name = arg[0:index].strip()
        attrs = arg[index + 1:]
        attrs = re.sub(', *data_ptr: 0x[\da-f]+', '', attrs)
        args_info.append(name + ':[' + attrs + '] ')

    return args_info

def extract_fallback_op_info(log_content):
    aten_op_names = re.findall('fallback to cpu, name=([\w\d_:\.]+)', log_content)
    aten_op_names = set(aten_op_names)
    op_infos = []
    for name in aten_op_names:
        op_info = dict()
        op_info['aten_name'] = name
        op_info['diopi_fun'] = 'fallback'
        op_info['args'] = 'no'
        op_infos.append(op_info)
    return op_infos


def extract_op_info(op):
    op_info = dict()
    op_name = re.search('(?<=(\-\-\[)).*(?=\])', op).group().strip()
    op_info['aten_name'] = op_name
    op_info['diopi_fun'] = re.search('[\w\d_]+', op[op.find(']:'):]).group().strip()
    op_info['args'] = extract_op_arg(op)
    return op_info

def unique_ops(op_infos):
    op_infos_unique = []
    op_infos_dict = dict()
    for op_info in op_infos:
        op_name = op_info['aten_name']
        if op_name not in op_infos_dict:
            op_infos_dict[op_name] = dict()
            op_infos_dict[op_name]['diopi_fun'] = op_info['diopi_fun']
            op_infos_dict[op_name]['args'] = set()
        op_infos_dict[op_name]['args'].add(str(op_info['args']))
    for name in op_infos_dict.keys():
        for args in op_infos_dict[name]['args']:
            op_info = dict()
            op_info['aten_name'] = name
            op_info['diopi_fun'] = op_infos_dict[name]['diopi_fun']
            op_info['args'] = args
            op_infos_unique.append(op_info)

    return op_infos_unique


def op_capaure(log_content):
    ops = get_all_op_from_train_log(log_content)
    op_infos = []
    for op in ops:
        op_info = extract_op_info(op)
        op_infos.append(op_info)
    return op_infos

def main():
    args = parase_args()
    with open(args.train_log) as train_log:
        file_data = train_log.read()
        op_infos = op_capaure(file_data)
        op_infos += extract_fallback_op_info(file_data)
        op_infos = unique_ops(op_infos)

    if len(op_infos) <= 0:
        return

    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        header = op_infos[0].keys()
        writer.writerow(header)
        rows = []

        for op_info in op_infos:
            r = []
            for key in header:
                r.append(op_info[key])
            rows.append(r)
        writer.writerows(rows)

if __name__ == "__main__":
    main()
