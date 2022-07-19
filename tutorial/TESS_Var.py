from typing import Tuple
import numpy as np

def median_multi_segment(y, segment_ind):
    """
    Fits piecewise polyonmials to a list of Segments Seperated by points.
    
    Parameters
    -----------
    
    x: np.ndarray array of the x values 
    y: np.ndarray array of the y values
    segment_ind: array-like list of the indexes of the points which seperate the different segments
    
    Returns
    ----------
    y_median: np.ndarray of the 
    """
    
    #Split y into parts at the segment index locations so we only take those into account for fitting
    y_parts = np.split(y,segment_ind)
    y_median = np.split(np.zeros_like(y),segment_ind) # extra arrays to store the median in
    segments = len(segment_ind) + 1 #The amount of segments is always one more than the borders between each segment
    
    #For each segment calculate the Median
    for i in range(segments):
        current_median = np.median(y_parts[i])
        y_median[i][:] = current_median
    
    #Return array concatenatet array with the medians this gives the corresponding median for each point in the original data.
    return np.concatenate(y_median)



def Calc_window_with_padding(Time : np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    """Calculates the window function of a data which has gaps in between. Where it gives 0 at the points 

    Args:
        Time (np.ndarray): An array of the time dat

    Returns:
        Tuple[np.ndarry,np.ndarray]: _description_
    """
    StepSize = np.median(np.diff(Time))
    Resample = np.arange(Time.min()-3*StepSize,Time.max()+3*StepSize,StepSize)

    gaplist = get_gaps_limits(Time,0.5)

    window = np.zeros_like(Resample,dtype=bool)
    for gap in gaplist:
        bools = (Resample>gap[0]) & (Resample<gap[1])
        window = bools  | window
        
    return window,Resample


def get_gaps_limits(Time,min_gap,padding=0.1):
    diffs = np.diff(Time)
    ind_gap = np.where(diffs>min_gap)[0]
    
    if len(ind_gap) == 0:
        return tuple([(Time.min(),Time.max())])

    first = (Time.min(),Time[ind_gap[0]])
    last = (Time[ind_gap[-1]+1],Time.max())

    lims_list = []
    lims_list.append(first)

    if len(ind_gap) > 1:
        for i in range(ind_gap.size-1):
            iter_tuple = None
            iter_tuple = (Time[ind_gap[i]+1],Time[ind_gap[i+1]])
            lims_list.append(iter_tuple)

    lims_list.append(last)
    
    return tuple(lims_list)
