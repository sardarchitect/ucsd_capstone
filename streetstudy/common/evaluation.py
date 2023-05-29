import numpy as np
import scipy.optimize

def bbox_iou(boxA, boxB):
    """
    Finds IOU for two bounding boxes
    
    Keyword Arguments:
    boxA -- first bounding box with top-left and bottom-right x-y coordinates in a list
    boxB -- second bounding box with top-left and bottom-right x-y coordinates in a list
    
    Return:
    iou -- percent overlap (0 = none, 1 = perfect-fit)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    intersection_width = xB - xA 
    intersection_height = yB - yA
    
    if intersection_width <= 0 or intersection_height <= 0:
        return 0
    
    intersection_area = intersection_width * intersection_height
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou 

def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.01):
    """
    Given sets of ground truth and predicted bounding boxes, determine best possible match.
    
    Keyword Arguments:
    bbox_gt -- bounding boxes for ground truth
    bbox_pred -- bounding boxes for predictions
    
    Return:
    
    """
    
    MIN_IOU = 0
    
    num_gt = len(bbox_gt)
    num_pred = len(bbox_pred)
    
    bbox_gt_np = np.empty((num_gt, 4))
    bbox_pred_np = np.empty((num_pred, 4))
    
    bbox_gt_np[:, 0] = bbox_gt['bbox_center_x'] - (bbox_gt['bbox_width'] / 2)
    bbox_gt_np[:, 1] = bbox_gt['bbox_center_y'] - (bbox_gt['bbox_height'] / 2)
    bbox_gt_np[:, 2] = bbox_gt['bbox_center_x'] + (bbox_gt['bbox_width'] / 2)
    bbox_gt_np[:, 3] = bbox_gt['bbox_center_x'] + (bbox_gt['bbox_height'] / 2)
    
    bbox_pred_np[:, 0] = bbox_pred['bbox_center_x'] - (bbox_pred['bbox_width'] / 2)
    bbox_pred_np[:, 1] = bbox_pred['bbox_center_y'] - (bbox_pred['bbox_height'] / 2)
    bbox_pred_np[:, 2] = bbox_pred['bbox_center_x'] + (bbox_pred['bbox_width'] / 2)
    bbox_pred_np[:, 3] = bbox_pred['bbox_center_x'] + (bbox_pred['bbox_height'] / 2)
    
    iou_matrix = np.zeros((num_gt, num_pred))
    
    for i in range(num_gt):
        for j in range(num_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt_np[i], bbox_pred_np[j])
            
    if num_pred > num_gt:
        diff = num_pred - num_gt
        iou_matrix = np.concatenate((iou_matrix, np.full((diff, num_pred), MIN_IOU)), axis=0)
    if num_gt > num_pred:
        diff = num_gt - num_pred
        iou_matrix = np.concatenate((iou_matrix, np.full((num_gt, diff), MIN_IOU)), axis=1)
        
    idxs_gt, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)
    sel_pred = idxs_pred < num_pred
    idx_pred_actual = idxs_pred[sel_pred]
    idx_gt_actual = idxs_gt[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)
    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label

def precision_recall(bbox_gt, bbox_pred, bbox_pred_labels):
    """
    Returns precision and recall of given bounding boxes
    
    """
    
    total_gt = len(bbox_gt)
    total_pred = len(bbox_pred)

    true_positives = sum(bbox_pred_labels)
    false_positives = abs(total_pred - true_positives)
    false_negatives = abs(total_gt - total_pred)
    
    if total_pred == 0:
        precision = 0
        recall = 0
        return precision, recall
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return precision, recall