import json
import os
# file for creating vocab for opcodes and oprnd
# also for creating training data for BERT

def src_to_txt(collect_func_src):
    with open(r'train2.txt', 'w') as fp:
        for line_ in collect_func_src:
            # write each item on a new line
            fp.write("%s\n" % line_)
        print('Done Train file!')

vocab = []
file_directory = 'small_sample_2'
n_file = len([name for name in os.listdir(file_directory) if os.path.isfile(os.path.join(file_directory, name))])
print("no of files: ", n_file)
count_file = 1
collect_func_src = []
for file_name in os.listdir(file_directory):
    f = os.path.join(file_directory,file_name)
    if os.path.isfile(f):
        with open(file_directory + "\\" + file_name, 'r') as f:
            data = json.load(f)
        func_list = list(data.values())
        for func in func_list:
            new_src = []
            for inst in func['src']:
                new_inst = inst.split()[0]
                vocab = vocab + [new_inst]
                for op in inst.split()[1:]:
                    if op[0:4] == 'loc_':
                        op = 'location'
                    if op[0] == '_':
                        op = 'func_name'
                    if 'cwe399_gcc_' in op:
                        op = 'func_name'
                    if 'CWE' in op:
                        op = 'func_name'
                    if 'locret_' in op:
                        op = 'locret_'
                    if 'unk_' in op:
                        op = 'unk_'
                    if op[0] == 'a' and op[1].isupper():
                        op = 'uString'
                    vocab = vocab + [op]
                    vocab = list(set(vocab))
                    new_inst = new_inst + " " + op
                new_src.append(new_inst)
                collect_func_src.append(new_inst)
            func["src"] = new_src
        with open(file_directory + "\\" + file_name, 'w') as outfile:
            json.dump(data, outfile)
    print(count_file, " / ", n_file, " done!")
    count_file = count_file + 1
print(vocab)

special_tokens = [
  "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]

vocab = special_tokens + vocab


with open(r'vocab_single_function.txt', 'w') as fp:
    for word in vocab:
        # write each item on a new line
        fp.write("%s\n" % word)
    print('Done')

src_to_txt(collect_func_src)