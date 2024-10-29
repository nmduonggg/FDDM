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
    
    class_dict_path = '/home/user01/aiotlab/nmduong/BoneTumor/RAW/REAL_WSIs/class_dict_256_case68.json'   # update continuously
    # case_dict_path = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/case_dict_256.json'
    # old_class_dict_path = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/class_dict_256_bkup.json'
    # dataset_dict_path = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/dataset_split_256_by_class_bkup.json'
    
    with open(class_dict_path, 'r') as f:
        case_dict = json.load(f)
        
    train_indices = []
    valid_indices = []
    test_indices = []
    
    train_ratio = 0.9
    valid_ratio = 0.1
        
    for case in case_dict.keys():
        if 'last_index' in case: continue
        case_indices = case_dict[case]
        # old_case_indices = old_case_dict[case]
        
        train_set, valid_set, test_set = random_split(case_indices, train_ratio, valid_ratio)
        
        train_indices += train_set
        valid_indices += valid_set
        test_indices += test_set
        
        # # oversampling: 5
        # if case=='5':
        #     train_indices += train_set * 2
            
        print(case, len(train_set), len(valid_set), len(test_set))
        
    write_to_json(train_indices, valid_indices, test_indices,
                  path = "/home/user01/aiotlab/nmduong/BoneTumor/RAW/REAL_WSIs/dataset_split_256_case68.json")