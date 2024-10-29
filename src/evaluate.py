import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score

color_map = [
    [255, 255, 255],    # background
    [0, 128, 0],    # Viable tumor
    [255, 143, 204],    # Necrosis
    [255, 0, 0],    # Fibrosis/Hyalination
    [0, 0, 0],  # Hemorrhage/ Cystic change
    [165, 42, 42],  # Inflammatory
    [0, 0, 255]]    # Non-tumor tissue

def apply_threshold_mapping(image):
    # Create masks for pixels that are closer to green or pink
    # Initialize the output image with the original image
    tolerance = 50
    
    output = np.ones_like(image)*255 # 2D only
    masks = 0.
    for idx, color in enumerate(color_map):
        color = np.array(color)
        mask = np.all(np.abs(image - color) < tolerance, axis=-1)
        # output[mask] = color
        output[mask] = idx
        if idx==2: print(mask.mean())

    return output

def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def open_img(file):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# def calculate_iou_per_class(pred_mask, gt_mask, class_label):
#     """
#     Calculate IoU for a specific class.
    
#     :param pred_mask: 2D array of predicted mask
#     :param gt_mask: 2D array of ground truth mask
#     :param class_label: Integer representing the class label
#     :return: IoU value for the class
#     """
#     pred_mask=pred_mask.reshape(-1)
#     gt_mask=gt_mask.reshape(-1)
#     pred_class_mask = (pred_mask == class_label)
#     gt_class_mask = (gt_mask == class_label)

#     intersection = np.logical_and(pred_class_mask, gt_class_mask).sum()
#     union = np.logical_or(pred_class_mask, gt_class_mask).sum()
    
#     if union == 0:
#         return float('nan'), union  # Avoid division by zero
#     iou = intersection / union
#     return (iou, union)

# def calculate_precision_recall(pred_mask, gt_mask, class_label):
#     """
#     Calculate Precision and Recall for a specific class.
    
#     :param pred_mask: 2D array of predicted mask
#     :param gt_mask: 2D array of ground truth mask
#     :param class_label: Integer representing the class label
#     :return: precision, recall, average precision (AP)
#     """
#     pred_class_mask = (pred_mask == class_label).astype(int).flatten()
#     gt_class_mask = (gt_mask == class_label).astype(int).flatten()

#     precision, recall, _ = precision_recall_curve(gt_class_mask, pred_class_mask)
#     ap = average_precision_score(gt_class_mask, pred_class_mask)
    
#     return precision, recall, ap

# def evaluate_segmentation(pred_masks, gt_masks, class_labels):
#     """
#     Evaluate segmentation results for multiple classes and images.
    
#     :param pred_masks: List of 2D arrays of predicted masks
#     :param gt_masks: List of 2D arrays of ground truth masks
#     :param class_labels: List of class labels to evaluate
#     :return: Dictionary of IoU, AP, and AR for each class
#     """
#     iou_per_class = {label: [] for label in class_labels}
#     ap_per_class = {label: [] for label in class_labels}
#     ar_per_class = {label: [] for label in class_labels}
    
#     for pred_mask, gt_mask in zip(pred_masks, gt_masks):
#         for class_label in class_labels:
#             # Calculate IoU for the class
#             iou, union = calculate_iou_per_class(pred_mask, gt_mask, class_label)
#             if union==0: continue
#             if not np.isnan(iou):
#                 iou_per_class[class_label].append(iou)
            
#             # Calculate Precision, Recall, and AP for the class
#             precision, recall, ap = calculate_precision_recall(pred_mask, gt_mask, class_label)
#             ap_per_class[class_label].append(ap)
#             ar_per_class[class_label].append(np.mean(recall))  # Average Recall for the current image
    
#     # Compute mAP and mAR by averaging across all images
#     results = {}
#     for class_label in class_labels:
#         results[class_label] = {
#             'IoU': np.nanmean(iou_per_class[class_label]),
#             'AP': np.mean(ap_per_class[class_label]),
#             'AR': np.mean(ar_per_class[class_label])
#         }
    
#     return results

# def compare(y_true, y_pred, num_classes):
#     class_labels = range(num_classes)
#     class_wise_results = evaluate_segmentation(y_pred, y_true, class_labels)
#     results = {
#         'IoU': 0.,
#         'AP': 0.,
#         'AR': 0.}
#     for met in results.keys():
#         for c in class_wise_results.keys():
#             results[met] += class_wise_results[c][met]
#     for met in results.keys():
#         results[met] /= len(class_wise_results.keys())
#     return results

def compare(y_true, y_pred, num_classes):
    """
    Function to calculate IoU, Precision, Recall, and Accuracy for multiclass segmentation.
    
    Parameters:
    y_true: np.ndarray
        Ground truth image with classes.
    y_pred: np.ndarray
        Predicted image with classes.
    num_classes: int
        Number of classes in the segmentation task.
    
    Returns:
    metrics: dict
        A dictionary containing IoU, Precision, Recall, and Accuracy for each class, and the average metrics.
    """
    
    IoU_per_class = []
    Precision_per_class = []
    Recall_per_class = []
    Accuracy_per_class = []
    
    # num_classes -= 1 # skip background
    
    # Loop through each class and calculate metrics
    for class_id in range(0, num_classes):
        # True Positive (TP): predicted correctly as the current class
        TP = np.sum((y_true == class_id) & (y_pred == class_id))
        
        # True Negative (TN): predicted correctly as NOT the current class
        TN = np.sum((y_true != class_id) & (y_pred != class_id))
        
        # False Positive (FP): predicted as the current class, but actually is not
        FP = np.sum((y_true != class_id) & (y_pred == class_id))
        
        # False Negative (FN): actually the current class, but predicted as not
        FN = np.sum((y_true == class_id) & (y_pred != class_id))
        
        # if class_id==0:
        
        #     TP = np.sum(((y_true == 6) | (y_true==0)) & ((y_pred == 6) | (y_pred==0)))
            
        #     # True Negative (TN): predicted correctly as NOT the current class
        #     TN = np.sum(((y_true != 6) & (y_true != 0)) & ((y_pred != 6) & (y_pred != 0)))
            
        #     # False Positive (FP): predicted as the current class, but actually is not
        #     FP = np.sum(((y_true != 6) & (y_true != 0)) & ((y_pred == 6) | (y_pred==0)))
            
        #     # False Negative (FN): actually the current class, but predicted as not
        #     FN = np.sum(((y_true == 6) | (y_pred==0)) & ((y_pred != 6) & (y_pred != 0)))
        
        # Calculate IoU, Precision, Recall, Accuracy
        IoU = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 1
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 1
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 1
        Accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 1
        
        # Append the result for the current class
        IoU_per_class.append(IoU)
        Precision_per_class.append(Precision)
        Recall_per_class.append(Recall)
        Accuracy_per_class.append(Accuracy)
    
    # Calculate mean metrics across all classes
    mean_IoU = np.mean(IoU_per_class)
    mean_Precision = np.mean(Precision_per_class)
    mean_Recall = np.mean(Recall_per_class)
    mean_Accuracy = np.mean(Accuracy_per_class)
    
    metrics = {
        'IoU_per_class': IoU_per_class,
        'Precision_per_class': Precision_per_class,
        'Recall_per_class': Recall_per_class,
        'Accuracy_per_class': Accuracy_per_class,
        'mean_IoU': mean_IoU,
        'mean_Precision': mean_Precision,
        'mean_Recall': mean_Recall,
        'mean_Accuracy': mean_Accuracy
    }
    
    return metrics

if __name__=='__main__':
    gt_folder = '/workdir/radish/manhduong/REAL_STATISTICS'
    # pred_folder = '/home/user01/aiotlab/nmduong/BoneTumor/src/infer/smooth_uni/UNI_lora_cls'
    # pred_folder = '/home/user01/aiotlab/nmduong/BoneTumor/src/infer/smooth_vit/ViT_baseline'
    # pred_folder = '/home/user01/aiotlab/nmduong/BoneTumor/src/infer/smooth_resnet/ResNet_baseline'
    # pred_folder = '/home/user01/aiotlab/nmduong/BoneTumor/src/infer/smooth_uni_last/UNI_lora_cls'
    pred_folder = '/home/manhduong/BoneTumor/src/infer/smooth_segformer/segformer'
    
    
    case_names = [n for n in os.listdir(gt_folder) if os.path.isdir(os.path.join(gt_folder, n))]
    overall_metrics = {}
    cnt = 0
    
    valid_cases = [f"Case_{i}" for i in [6, 8]]
    train_cases = [f"Case_{i}" for i in [1, 2, 3, 4, 5, 7, 9, 10]]
    
    process_cases = valid_cases
    
    for case_name in tqdm(case_names, total=len(case_names)):
        if case_name not in process_cases: continue
        gt_case_folder = os.path.join(gt_folder, case_name)
        pred_case_folder = os.path.join(pred_folder, case_name)
        
        if not os.path.isdir(pred_case_folder): continue
        
        for label_name in os.listdir(gt_case_folder):
            
            if "x8" in label_name:
                image_name = label_name.split("-x8")[0] + '_pred.png'
                gt_name1 = label_name.split("-x8")[0] + '_label.png'
                upsample = True
            else:
                image_name = label_name.split("-labels")[0] + '_pred.png'
                gt_name1 = label_name.split("-labels")[0] + '_label.png'
            
            pred_image_path = os.path.join(pred_case_folder, image_name)
            if not os.path.isfile(pred_image_path): continue
            pred_image = open_img(pred_image_path)
            
            # gt_image_path = os.path.join(gt_case_folder, label_name)
            gt_image_path = os.path.join(pred_case_folder, gt_name1)
            gt_image = open_img(gt_image_path)
            
            pred_image = cv2.resize(pred_image, (gt_image.shape[1], gt_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            pred_image = apply_threshold_mapping(pred_image)
            gt_image = apply_threshold_mapping(gt_image)
            metrics = compare(gt_image, pred_image, len(color_map))
            
            for key in metrics.keys():
                overall_metrics[key] = overall_metrics.get(key, 0) + np.array(metrics[key])
            cnt += 1
            
    print("count:", cnt)
    for k, v in overall_metrics.items():
        print(f"{k}: {v / cnt}")