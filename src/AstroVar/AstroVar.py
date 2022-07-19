from typing import Dict, Tuple
import numpy as np
from PyAstronomy.pyTiming import pyPeriod
from astropy.table import Table
import astropy.units as u
import matplotlib.pyplot as plt



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


def get_gaps_limits(Time : np.ndarray ,min_gap : float) -> tuple:
    """Find the Gaps in a time series data which are bigger than a minimum gap

    Args:
        Time (np.ndarray): The Array with the times of each measurement
        min_gap (float): the minimum gap

    Returns:
        tuple: a tuple of the begining and end of the gaps e.g. (gap_1_start,gap_1_end, gap_2_start, gap_2_end ....)
    """
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



def iter_detrend(Flux : np.ndarray ,Time : np.ndarray, FeatureMask : np.ndarray = None ,periodgrid  : np.ndarray = np.logspace(np.log10(0.25),np.log10(30),1000)) -> Tuple[Table,Dict]:
    
    """The Iterative detrending algorithm 

    Args:
        Flux (np.ndarray): the photometric flux
        Time (np.ndarray): the time at which each measurement was taken
        FeatureMask (np.ndarray): an array in which outliers or other features are masked out
        periodgrid (np.ndarray): a specific periodgrid on which the periodogram is calculated


    Returns:
        Table: A Table including all the important fit parameters for each iterative step
        Dict: A dictionary including additional arrays, like the iterative detrended fluxes for each step, the Periodogram for each Step, the FAP levels and the fitted Sine Curves
    """

    #Prepare the Frequency grid on which the Periodogram is calculated
    periods = periodgrid
    frequency = 1/periods

    if FeatureMask == None:
        FeatureMask = np.ones_like(Flux,dtype = bool)

    flx = Flux[FeatureMask]
    tme = Time [FeatureMask]

    Flux_Rep = flx.copy() #Copy of the Flux for iterative subtracting of the fits  without changing the original Pre Detrended Flux
    Flux_Features = Flux.copy() # Copy of the Flux with Features so we get a detrended view of the features at the end.

    #Preparation of the Table which stores the different Parameters for each 
    tbl_Detrend = Table(names=["Period","P_err","Amplitude","A_err","T0","T0_err","off","off_err","Probability","False Alarm Probability"],units=[u.d,u.d,u.electron/u.s,u.electron/u.s,u.d,u.d,u.electron/u.s,u.electron/u.s,1,1]) #Table
        


    FalseAlarmFlag = False #Flag for stopping the loop once we have reached the threashold that the max of the periodogram is under the false alarmlevel
    FAP_Stop = 0.5 # The False Alarm Probabiliy at which the Iterative detrending Algorithm Stops

    Powers = []
    Fluxes = [Flux_Rep] #List that Tracks the Fluxes from the Start, includes the undetrended Flux so Power[0] is the Corresponding Periodogram to Fluxes[0]
    FAP_Levels = []
    Sines = []

    i = 0
    while FalseAlarmFlag == False:
        
        
        #Create Lomb Scargle Periodogram Class
        clp = pyPeriod.Gls((tme,Flux_Rep), freq = frequency,norm="ZK") #Calculates GLS 

        Periodogram = clp.power # Calculate the Power of the Lomb Scargle Periodogram
        idx = np.argmax(Periodogram) #Find Index of Strongest Power
        p_rep = periods[idx] #Corresponding Period of the Strongest Signal
        


        y_fit = clp.sinmod(tme) # Gives the Fitted Sinus curve to the Variability
        Flux_Rep = Flux_Rep - y_fit # Detrend with the Fit
        Flux_Features -= clp.sinmod(Time) #Detrend the Time Series with Masked out outliers
        
        FAP = clp.FAP(Periodogram[idx])#Get the FAP of the Maximum
        
        Powers.append(Periodogram)
        Sines.append(y_fit)
        Fluxes.append(Flux_Rep)

        #Extract all the necessary parameters from the computed Periodogram which can be used to calculate the fitted sinus.
        amp = clp.hpstat["amp"]
        pe = p_rep*clp.hpstat["f_err"]/clp.hpstat["fbest"]
        Ae = clp.hpstat["amp_err"]
        T0 = clp.hpstat["T0"]
        T0e = clp.hpstat["T0_err"]
        off = clp.hpstat["offset"]
        offe = clp.hpstat["offset_err"]
        
        
        tbl_Detrend.add_row([p_rep,pe,amp,Ae,T0,T0e,off,offe,Periodogram[idx],FAP])
        
        FAP_Power = clp.powerLevel(FAP_Stop) #Get the Probbility of the Periodogram corresponding to the False alarm Porbability set in FAP_Stop
        FAP_Levels.append(FAP_Power)

        # Break Condition strongest signal is below a FAP of 0.5
        if FAP_Power > Periodogram[idx]:
            FalseAlarmFlag = True
            

        i += 1

    Powers = np.array(Powers)
    Fluxes = np.array(Fluxes)
    FAP_Levels = np.array(FAP_Levels)
    Sines = np.array(Sines)

    #Store the Additional Arrays for returning them in a dictionary 
    Additional_Dict = {"Powers":Powers,"Fluxes":Fluxes,"FAP_Levels":FAP_Levels,"Sines":Sines, "Flux_Features": Flux_Features}

    return tbl_Detrend, Additional_Dict


def plot_iterative(Time : np.ndarray, Flux : np.ndarray , Additional_Dict : Dict , tbl_Detrend : Table ,n_single : int,FeatureMask = None , periodgrid  : np.ndarray = np.logspace(np.log10(0.25),np.log10(30),1000)):
    
    n_single = 2 # How many Iteration Steps are plotted

    periods = periodgrid
    tme = Time
    Fluxes = Additional_Dict["Fluxes"]
    Sines = Additional_Dict["Sines"]
    Powers = Additional_Dict["Powers"]
    FAP_Levels = Additional_Dict["FAP_Levels"]
    Flux_Features = Additional_Dict["Flux_Features"]
    if FeatureMask == None:
        FeatureMask = np.ones_like(Flux,dtype = bool)


    #Figure
    fig = plt.figure(constrained_layout = False,figsize=(13,2*(n_single+1)))
    gs1 = fig.add_gridspec(nrows=n_single+1, ncols=2, wspace=0.20,hspace=0.0)
    gs2 = fig.add_gridspec(nrows=n_single+1, ncols=2, hspace=0.7)



    ax_right = []
    ax_left = []
    for i in range(n_single):
    
        a = fig.add_subplot(gs1[i,0])
        b=fig.add_subplot(gs1[i,1])
        ax_left.append(a)
        ax_right.append(b)
        


    axbig = fig.add_subplot(gs2[n_single,:])

    for i in range(n_single):

        Periodogram = Powers[i]
        Detrended = Fluxes[i]
        idx = np.argmax(Periodogram)
        amp = tbl_Detrend[i]["Amplitude"]

        ax_right[i].plot(periods,Periodogram)
        ax_right[i].plot(periods[idx],Periodogram[idx],"o",label=f"Amplitude = {amp:.3} \nPeriod = {periods[idx]:.3}")
        ax_right[i].axhline(FAP_Levels[i],dashes=(2,3),color="r")
        ax_right[i].legend()
        ax_right[i].semilogx()

        ax_left[i].plot(tme,Detrended,".")
        ax_left[i].plot(tme,Sines[i])


    for ax in ax_left:
        ax.set(ylabel="Flux [$e^-/s$]")

    for ax in ax_right:
        ax.set(ylabel="Probability [A.U.]")

    for ax in ax_left[:-1]:
        ax.set_xticks([])

    ax_left[n_single-1].set(xlabel="Time [BJD]")
    ax_right[n_single-1].set(xlabel="Period [d]")


    N = 50
    trend_lc = np.convolve(Flux_Features, np.ones((N,))/N, mode='same')#Calculate a Rolling Average to Check the Las

    axbig.plot(Time,Flux_Features,".")
    axbig.plot(Time[~FeatureMask],Flux_Features[~FeatureMask],"r.")
    axbig.plot(Time,trend_lc,"-")
    axbig.text(0.35,0.85,f"Last repeat after {tbl_Detrend.__len__()} Iterations",transform=axbig.transAxes,fontsize=14)
    axbig.set(xlabel="Time [BJD]")
    

    return fig, (ax,axbig)