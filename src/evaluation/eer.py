import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from typing import Tuple, Dict

class EERResult(dict):

    def __iter__(self):
        return iter((self['eer'], self['threshold'], self['curve_data']))

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> EERResult:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    diff = np.abs(fpr - fnr)
    idx = int(np.nanargmin(diff))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    optimal_thresh = float(thresholds[idx])
    try:
        if len(thresholds) > 3 and (not (np.all(fpr == 0) and np.all(fnr == 0))):
            unique_indices = np.unique(thresholds, return_index=True)[1]
            t_sorted = thresholds[unique_indices]
            f_sorted = fpr[unique_indices]
            fn_sorted = fnr[unique_indices]
            diff_sorted = f_sorted - fn_sorted
            if len(diff_sorted) > 1 and np.sign(diff_sorted[0]) != np.sign(diff_sorted[-1]):
                eer_func = interp1d(t_sorted, diff_sorted, kind='linear', fill_value='extrapolate', assume_sorted=False)
                root_thresh = float(brentq(eer_func, float(np.min(t_sorted)), float(np.max(t_sorted))))
                interp_eer = float(interp1d(t_sorted, f_sorted, fill_value='extrapolate', assume_sorted=False)(root_thresh))
                if 0.0 <= interp_eer <= 1.0:
                    eer = interp_eer
                    optimal_thresh = root_thresh
    except Exception:
        pass
    curve_data = {'fpr': [float(x) for x in fpr], 'tpr': [float(x) for x in tpr], 'fnr': [float(x) for x in fnr], 'thresholds': [float(x) for x in thresholds]}
    return EERResult(eer=float(eer), threshold=float(optimal_thresh), curve_data=curve_data)
