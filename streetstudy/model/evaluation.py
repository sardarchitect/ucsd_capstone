import numpy as np
import scipy.optimize

def bbox_iou(boxA, boxB):
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
    '''
    Given sets of ground truth and predicted bounding boxes,
    determine best possible match.
    '''
    num_gt = bbox_gt.shape[0]
    num_pred = bbox_pred.shape[0]
    MAX_DIST = 1.0
    MIN_IOU = 0.0
    
    iou_matrix = np.zeros((num_gt, num_pred))
    
    for i in range(num_gt):
        for j in range(num_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i], bbox_pred[j])
    
    if num_pred > num_gt:
        diff = num_pred - num_gt
        iou_matrix = np.concatenate((iou_matrix, np.full((diff, num_pred), MIN_IOU)), axis=0)
        
    if num_gt > num_pred:
        diff = num_gt - num_pred
        iou_matrix = np.concatenate((iou_matrix, np.full((num_gt, diff), MIN_IOU)), axis=1)
        
    idxs_gt, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)
    if (not idxs_gt.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_gt, idxs_pred]
        
    sel_pred = idxs_pred < num_pred
    idx_pred_actual = idxs_pred[sel_pred]
    idx_gt_actual = idxs_gt[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)
    
    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label

def average_precision(n_true_positive, n_false_positive):
    return n_false_positive / (n_true_positive + n_false_positive)