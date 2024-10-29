import os
import json
import numpy as np
import pandas as pd

def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

# Function to create a row of data for each case and image
def add_row(case, image, pred_values, gt_values):
    row = {
        ("gt-viable"): gt_values.get("viable", None),
        ("gt-necrosis"): gt_values.get("necrosis", None),
        ("gt-fibrosis/hyalination"): gt_values.get("fibrosis/hyalination", None),
        ("gt-hemorrhage/cystic-change"): gt_values.get("hemorrhage/cystic-change", None),
        ("gt-inflammatory"): gt_values.get("inflammatory", None),
        ("gt-non-tumor"): gt_values.get("non-tumor", None),
        ("pred-viable"): pred_values.get("viable", None),
        ("pred-necrosis"): pred_values.get("necrosis", None),
        ("pred-fibrosis/hyalination"): pred_values.get("fibrosis/hyalination", None),
        ("pred-hemorrhage/cystic-change"): pred_values.get("hemorrhage/cystic-change", None),
        ("pred-inflammatory"): pred_values.get("inflammatory", None),
        ("pred-non-tumor"): pred_values.get("non-tumor", None),
    }
    # Append a tuple (case, image, row) to the data list
    data.append((f"{case} - {image}", row))
    
def write2excel(df, exl_path):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(exl_path, engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object. Turn off the default
    # header and index and skip one row to allow us to insert a user defined
    # header.
    df.to_excel(writer, sheet_name='Sheet1', startrow=1, header=False, index=True)

    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']

    # Get the dimensions of the dataframe.
    (max_row, max_col) = df.shape

    # Create a list of column headers, to use in add_table().
    column_settings = []
    column_settings.append({'header': 'slide_name'})
    for header in df.columns:
        column_settings.append({'header': header})

    # Add the table.
    worksheet.add_table(0, 0, max_row, max_col, {'columns': column_settings})

    # Make the columns wider for clarity.
    worksheet.set_column(0, max_col, 20)

    # Close the Pandas Excel writer and output the Excel file.
    writer.close()



if __name__=='__main__':
    gt_path = '/home/manhduong/BoneTumor/RAW/REAL_WSIs/REAL_STATISTICS/gt_dict.json'
    # pred_path = '/home/user01/aiotlab/nmduong/BoneTumor/src/infer/smooth_uni_last/pred_dict.json'
    pred_path = pred_folder = '/home/manhduong/BoneTumor/src/infer/smooth_segformer/pred_dict.json'
    xlsx_path = './smooth_segformerB0.xlsx'
    
    cases = ["Case_6", "Case_8"]
    # Create an empty list to store the rows for the DataFrame
    data = []
    
    gt_data = read_json(gt_path)
    pred_data = read_json(pred_path)
    
    for case, gt_dict in gt_data.items():
        if case not in cases: continue
        if case not in pred_data: continue
        pred_dict = pred_data[case]
        
        for img_name in gt_dict.keys():
            print(img_name)
            gt_values = {k: round(v, 4) for k, v in gt_dict[img_name].items()}
            pred_values = {k: round(v, 4) for k, v in pred_dict[img_name].items()}
            
            add_row(case, img_name, pred_values, gt_values)
            
    # Create a DataFrame with a MultiIndex for the rows (Case, Image)
    df = pd.DataFrame.from_dict(dict(data), orient="index")
    write2excel(df, xlsx_path)
            
        