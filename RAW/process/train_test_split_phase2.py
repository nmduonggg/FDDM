'''
Simply divide a dict (case/class)_dict into dataset_dict, divide equally on each criteria 
(by class if using class_dict and by case if case_dict)
'''

import os
import json
import random

def random_split(indices, ratio1, ratio2):
    assert (ratio1 + ratio2 <= 1), "sum of ratios must <= 1"
    
    random.shuffle(indices)
    split_point1 = int(len(indices) * ratio1)
    split_point2 = int(len(indices) * (ratio1 + ratio2))
    set1 = indices[:split_point1]
    set2 = indices[split_point1: split_point2]
    set3 = indices[split_point2:]
    
    return set1, set2, set3

def write_to_json(train_indices, valid_indices, test_indices,
                  data = None,
                  path='./dataset_split.json'):
    if data is None:
        data = {
            'train': train_indices,
            'valid': valid_indices,
            'test': test_indices
        }
    for k, v in data.items():
        print(k, len(v))
        
    with open(path, 'w') as f:
        json.dump(data, f)
    return

if __name__=='__main__':
    
    metadata_path = '/home/user01/aiotlab/nmduong/BoneTumor/RAW/REAL_WSIs/_DENOISE2/metadata.json'   # update continuously
    dataset_outpath = '/home/user01/aiotlab/nmduong/BoneTumor/RAW/REAL_WSIs/_DENOISE2/dataset_split.json'
    
    with open(metadata_path, 'r') as f:
        data_list = json.load(f)
       
    # avoid overlap
    start_id = 0
    for i, dt in enumerate(data_list):
        if dt['crop_index'] == 1:
            start_id = i
    # print(cnt)
    data_list = data_list[start_id:]
    
    train_ratio = 0.9
    valid_ratio = 0.1
        
    num_items = len(data_list)
        
    train_indices, valid_indices, test_indices = random_split(list(range(len(data_list))), train_ratio, valid_ratio)
            
    print(len(train_indices), len(valid_indices), len(test_indices))
        
    write_to_json(train_indices, valid_indices, test_indices,
                  path = dataset_outpath)
    
    new_metadata_path = metadata_path.replace('.json', '_reindex.json')
    with open(new_metadata_path, 'w') as f:
        json.dump(data_list, f, indent=4)