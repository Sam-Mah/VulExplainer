# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
import os
import re
import itertools
import string

import numpy as np


def normalize(blocks):
    inst_lst = []
    # Flatten the codes of all blocks in a function
    # flat_blocks = list(itertools.chain.from_iterable(blocks))

    for inst in blocks:
        # The heading memory address is removed
        pattern1 = "\A0x\w+"
        inst = re.sub(pattern1, '', ' '.join(inst)).strip()
        # replace consts with const '0Bh'
        pattern2 = "\s\d+.*(?<!\s)$"
        inst = re.sub(pattern2, ' const', inst)
        # replace effective addresses with addr [ebp+Var_C]
        pattern3 = "\[.+\]"
        inst = re.sub(pattern3, ' addr', inst)
        # # Null is inserted at the tail of instructions with length less than three
        # while (len(inst.split()) < 3):
        #     inst += ' null'
        inst_lst.append(inst.lower())

    return inst_lst


def remove_isolated_func(dict, callee):
    flat_callee = list(itertools.chain.from_iterable(callee))
    # dict_keys = dict.keys()

    for itm in list(dict):
        if (itm not in flat_callee and dict[itm]['call'] == False):
            del dict[itm]

    return dict


def pre_process_function_calls(path_f):
    # Opening JSON file
    curr_dir = os.path.join(os.getcwd(), path_f)
    out_dir = "Output"
    for filename in os.listdir(curr_dir):
        if filename.endswith('.json'):
            with open(os.path.join(curr_dir, filename), 'r') as f:
                # Returns JSON object as
                data = json.load(f)
                # Iterating through the json
                dict = {}
                callee = []
                for x in data['functions']:
                    nested_dict = {}
                    # Include the caller functions only
                    nested_dict['name'] = x['name']
                    nested_dict['call'] = x['call']
                    callee.append(x['call'])
                    src_list = []
                    for y in x['blocks']:
                        src_list.append(y['src'])
                    # Normalize the instructions in all blocks and append as a list
                    blocks = normalize(src_list)
                    nested_dict['src'] = blocks
                    dict[x['id']] = nested_dict
                with open(os.path.join(out_dir, filename + "_graph.json"), "w") as fp:
                    json.dump(dict, fp)
                    print(dict)


def bag_of_words_opr_types(opr_list):
    # define o_void        0  // No Operand                           ----------
    # define o_reg         1  // General Register (al, ax, es, ds...) reg
    # define o_mem         2  // Direct Memory Reference  (DATA)      addr
    # define o_phrase      3  // Memory Ref [Base Reg + Index Reg]    phrase
    # define o_displ       4  // Memory Reg [Base Reg + Index Reg + Displacement] phrase+addr
    # define o_imm         5  // Immediate Value                      value
    # define o_far         6  // Immediate Far Address  (CODE)        addr
    # define o_near        7  // Immediate Near Address (CODE)        addr

    bag_words = np.zeros(8)
    for i in range(len(opr_list)):
        if len(opr_list[i]) == 0:
            opr_list[i].append(0)

    try:
        flat_opr_list = np.array(list(itertools.chain.from_iterable(opr_list)))
        values, counts = np.unique(flat_opr_list, return_counts=True)
        if (len(opr_list) != 0):
            for i in range(len(bag_words)):
                bag_words[values] = counts
    except:
        print("Out of bound index")

    return bag_words

def pre_process_CFG(path_f):
    # Opening JSON file
    curr_dir = os.path.join(os.getcwd(), path_f)
    out_dir = r'I:\XAI_Project\Datasets\Data_VulEx\Output_NDSS'

    substring1 = "call"
    substring2 = "bnd"
    # Extracting features from each block
    for filename in os.listdir(curr_dir):
        if filename.endswith('.json'):
            with open(os.path.join(curr_dir, filename), 'r') as f:
                # Returns JSON object as
                data = json.load(f)
                # Iterating through the json files
                dict = {}
                for x in data['functions']:
                    call_temp = x['call']
                    if (len(call_temp) > 1):
                        print(call_temp)
                    for y in x['blocks']:
                        norm_y_src = normalize(y['src'])
                        if (len(norm_y_src) != 0):
                            nested_dict = {'call': {}, 'src': '', 'features': []}
                            nested_dict['features'] = list(bag_of_words_opr_types(y['oprTypes']))
                            # Block calls
                            for itm in y['call']:
                                # Edge feature (call type): 1 (inter-function call), 0 (intra-function call (between blocks))
                                nested_dict['call'][str(x['id']) + ":" + str(x['blocks'][itm]['name'])] = 0
                            # Normalize the instructions in all blocks and append as a list
                            nested_dict['src'] = norm_y_src
                            # Find the caller block
                            flat_src = list(itertools.chain.from_iterable(y['src']))
                            loc = [s for s in flat_src if substring1 in s or substring2 in s]
                            # Function calls
                            if (len(loc) != 0 and len(call_temp) != 0):
                                # Find the callee function
                                d = [d for d in data['functions'] if d['id'] == call_temp[0]][0]
                                del call_temp[0]
                                # Edge feature (call type): 1 (inter-function call), 0 (intra-function call (between blocks))
                                nested_dict['call'][str(d['id']) + ":" + str(d['blocks'][0]['name'])] = 1
                            dict[str(x['id']) + ":" + str(y['name'])] = nested_dict
                        else:
                            print("garbage")
                            # for k,v in dict.items():
                            #     del dict[x['id']]['call'][y['name']]
                # ***********************************************
                with open(os.path.join(out_dir, filename + "_graph.json"), "w") as fp:
                    json.dump(dict, fp)
                    print(dict)


def pre_process_Steven(path_f):
    # Opening JSON file
    curr_dir = os.path.join(os.getcwd(), path_f)
    out_dir = "Output_LiTao"
    for filename in os.listdir(curr_dir):
        if filename.endswith('.json'):
            with open(os.path.join(curr_dir, filename), 'r') as f:
                # Returns JSON object as
                data = json.load(f)
                # print(data)
                # Iterating through the json
                dict = {}
                for x in data['blocks']:
                    nested_dict = {}
                    src_list = []
                    for y in x['ins']:
                        src = str(y['mne']) + " " + ' '.join(y['oprs'])
                        src_list.append(src)
                    # #Normalize the instructions in all blocks and append as a list
                    nested_dict['src'] = src_list
                    dict[x['_id']] = nested_dict
                with open(os.path.join(out_dir, filename + "_graph.json"), "w") as fp:
                    json.dump(dict, fp)
                    print(dict)


def print_Json(path_f):
    # Opening JSON file
    curr_dir = os.path.join(os.getcwd(), path_f)
    out_dir = "Selected_Test"
    for filename in os.listdir(curr_dir):
        with open(os.path.join(curr_dir, filename), 'r') as f:
            # Returns JSON object as
            data = json.load(f)
            # print(data)
            for x in data['functions']:
                # Include the caller functions only
                src_list = []
                src_list_2 = []
                for y in x['blocks']:
                    src_list.append(y['src'])
                    src_list_2.append(y['new_src'])
                print(src_list)
                print(src_list_2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pre_process_CFG(r'I:\XAI_Project\Datasets\Ashita\NDSS18\NDSS18Graph\Output_Preprocessed')
    # pre_process_Steven('C:/Users/Samaneh/XAI_Project/Datasets/LiTao/juliet/juliet/all')
    # print_Json('Selected_20220921')
