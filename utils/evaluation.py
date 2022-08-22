"""
    Reference:  Dominik Müller, Adrian Pfleiderer & Frank Kramer. (2022).
                miseval: a metric library for Medical Image Segmentation EVALuation.
                https://github.com/frankkramer-lab/miseval
"""
import numpy as np
def calc_ConfusionMatrix(truth, pred, c=1, dtype=np.int64, **kwargs):
    # Obtain predicted and actual condition
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    # Compute Confusion Matrix
    tp = np.logical_and(pd, gt).sum()
    tn = np.logical_and(not_pd, not_gt).sum()
    fp = np.logical_and(pd, not_gt).sum()
    fn = np.logical_and(not_pd, gt).sum()
    # Convert to desired numpy type to avoid overflow
    tp = tp.astype(dtype)
    tn = tn.astype(dtype)
    fp = fp.astype(dtype)
    fn = fn.astype(dtype)
    # Return Confusion Matrix
    return tp, tn, fp, fn

def calc_DSC_CM(truth, pred, c=1, **kwargs):
    # Obtain confusion mat
    tp, tn, fp, fn = calc_ConfusionMatrix(truth, pred, c)
    # Calculate Dice
    if (2*tp + fp + fn) != 0 : dice = 2*tp / (2*tp + fp + fn)
    else : dice = 0.0
    # Return computed Dice
    return dice

def calc_Specificity_CM(truth, pred, c=1, **kwargs):
    # Obtain confusion matrix
    tp, tn, fp, fn = calc_ConfusionMatrix(truth, pred, c)
    # Calculate specificity
    if (tn + fp) != 0 : spec = (tn) / (tn + fp)
    else : spec = 0.0
    # Return specificity
    return spec

def calc_Sensitivity_CM(truth, pred, c=1, **kwargs):
    # Obtain confusion matrix
    tp, tn, fp, fn = calc_ConfusionMatrix(truth, pred, c)
    # Calculate sensitivity
    if (tp + fn) != 0 : sens = (tp) / (tp + fn)
    else : sens = 0.0
    # Return sensitivity
    return sens

def __calc__(mask, pred, func):
    if np.array(mask).ndim!=3 :# )#len(np.shape(mask)))
        return False
    score_mc = []
    for i in range(len(mask)):
        score_mc.append(func(mask[i], pred[i], c=1))
    return np.average(score_mc)

def calc_DSC(mask, pred):
    return __calc__(mask, pred, func=calc_DSC_CM)

def calc_Specificity(mask, pred):
    return __calc__(mask, pred, func=calc_Specificity_CM)

def calc_Sensitivity(mask, pred):
    return __calc__(mask, pred, func=calc_Sensitivity_CM)
