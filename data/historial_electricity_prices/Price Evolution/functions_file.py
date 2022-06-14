#%% Libraries 

import os.path
import pypsa
import math
import numpy as np
from numpy.linalg import eig
import pandas as pd

import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.ticker as tick
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter

#%% SavePlot

def SavePlot(fig,path,title):
    """

    Parameters
    ----------
    path : string
        folder path in which the figures needs to be saved in. This needs to be created before
    title: string
        Name of the figure, this is also what the figures is saved as. This does not need to be created

    Returns
    -------
    Nothing

    """
    
    # Check if path exist
    assert os.path.exists(path), "Path does not exist"
    
    # Check if figure is already existing there
    fig.savefig(path+title+".png", bbox_inches='tight')
    
    return fig
    

#%% FilterPrice

def FilterPrice(prices, maxVal):
    """
    Parameters
    ----------
    prices : DataFrame
        Pandas dataframe containing .

    maxVal : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    pricesCopy = prices.copy()
    
    for name in pricesCopy.columns:
        for i in np.arange(len(pricesCopy.index)):
            
            # Check if the current value is larger than the maxVal
            if pricesCopy[name][i] > maxVal:
                
                # If i is the first element
                if i == 0:
                    position = 0
                    value = maxVal + 1
                    
                    # Replace the current value with the first value that is less than the max allowable value
                    while value > maxVal:
                        value = pricesCopy[name][i + position]
                        pricesCopy[name][i] = value
                        position +=1
                
                # If i is the last element
                elif i == (len(pricesCopy.index)-1):
                    pricesCopy[name][i] = pricesCopy[name][i-1]
            
                # Average the current element with its closest neighbouring elements
                else:
                    
                    # Forward looking
                    position = 0
                    valueForward = maxVal + 1
                    while valueForward > maxVal:
                        valueForward = pricesCopy[name][i + position]
                        position +=1
                        
                        # If the end of the array is reached
                        if i + position == (len(pricesCopy.index)-1):
                            valueForward = np.inf
                            break
                    
                    # Backward looking
                    position = 0
                    valueBackward = maxVal + 1
                    while valueBackward > maxVal:
                        valueBackward = pricesCopy[name][i - position]
                        position +=1
                        
                        # If the beginning of the array is reached
                        if i - position == 0:
                            valueBackward = np.inf
                            break
                    
                    
                    # Determine the value to insert into the array
                    value = 0
                    
                    # If the position of the array resulted in being out of bound, the value to insert is determined on only a one of them or the maxVal
                    if valueForward == np.inf and valueBackward == np.inf:
                        value = maxVal
                    
                    # If only one of the val
                    elif valueForward == np.inf:
                        value = valueBackward
                    
                    elif valueBackward == np.inf:
                        value = valueForward
                    
                    else:
                        value = (valueForward + valueBackward) / 2
                    
                    pricesCopy[name][i] = value
    return(pricesCopy)

#%% PriceEvolution

def PriceEvolution(meanPrice, quantileMeanPrice, quantileMinPrice, ylim=0, constraintype="", networktype="green", figsize=[5,4], fontsize=12, dpi=200, title="none"):
    """
    
    Parameters
    ----------
    meanPrice : array of values
        mean value of price agross all cases
    quantileMeanPrice : array of float64
        mean quantiles of price agross all cases.
    quantileMinPrice : list of array of values
        DESCRIPTION.
    ylim : float64, optional
        lower y value. The default is 0.
    constraintype : array of string, optional
        type of constrain. If default then it looks at the amount of cases. The default is "".
    networktype : string, optional
        Greenfield (green) or Brownfield (brown). The default is "green".
    figsize : array of float (size=2), optional
        size of figure. The default is [5,4].
    fontsize : float, optional
        size of font used. The default is 12.
    dpi : float, optional
        quality of figure. The default is 200.
    title : string, optional
        title ontop of figure. The default is "none".

    Returns
    -------
    fig : plt.figure
        Outputs figure to be able to save it with the SavePlot function

    """
    
    
    # Amount of data
    N = len(meanPrice)
    
    # Colors
    color = ["tab:blue","tab:green"]
    
    # alpha
    alpha = [0.2,0.4,0.2]
    
    # empty data
    datas = []
       
    # create dataframe
    meanPrice = pd.DataFrame(meanPrice, columns=["min","mean"])
    quantileMeanPrice = pd.DataFrame(quantileMeanPrice, columns=["0.05","0.25","0.75","0.95"])
    quantileMinPrice = pd.DataFrame(quantileMinPrice, columns=["0.05","0.25","0.75","0.95"])
    
    # label
    if meanPrice["min"].sum() < 1:
        label = ["Mean","Min"]
        meanPrice = meanPrice.drop("min",axis=1)
    else:
        label = label = ["Mean","Min"]
    
    # quantiles labels
    quantilesLabel = ["","25%-75%","5%-95%"]
    
    # create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # plot data
    for i, data in enumerate(reversed(list(meanPrice.columns))):

        # Plot quantiles
        plt.plot(meanPrice[data],color[i],linewidth=2,label=label[i],marker='o', markersize=3)
        for j, k in enumerate(list(quantileMeanPrice.columns)):
            plt.plot(quantileMeanPrice[k],color=color[i],alpha=0) # only there to fix legend "best"
        
            if j != 0:
                if i == 0:
                    quantile1 = quantileMeanPrice[quantileMeanPrice.columns[j-1]]
                    quantile2 = quantileMeanPrice[quantileMeanPrice.columns[j]]
                elif i == 1:
                    quantile1 = quantileMinPrice[quantileMinPrice.columns[j-1]]
                    quantile2 = quantileMinPrice[quantileMinPrice.columns[j]]
                plt.fill_between(range(N), quantile1, quantile2,
                                 label=quantilesLabel[j-1],
                                 color=color[i],
                                 alpha=alpha[j-1])
        for l in range(N): # quantile lines
            for k in range(3):
                if i == 0:
                    plt.plot([l,l],[quantileMeanPrice.iloc[l][k],quantileMeanPrice.iloc[l][k+1]],
                              color=color[i],
                              alpha=alpha[k]+0.1,
                              linestyle=(0,(2,2)))
                elif i == 1:
                    plt.plot([l,l],[quantileMinPrice.iloc[l][k],quantileMinPrice.iloc[l][k+1]],
                              color=color[i],
                              alpha=alpha[k]+0.1,
                              linestyle=(0,(2,2)))
    
    # plot setup
    plt.ylabel("Price [€/MWh]", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylim(ymin=-5)
    if ylim != 0:
        plt.ylim(ymax=ylim)
    plt.grid(alpha=0.15)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1.05), fontsize=fontsize-2)
    
    if networktype == "green":
        if constraintype == "":
            if N == 5:
                plt.xticks(np.arange(N),['Zero', 'Current', '2x Current', '4x Current', '6x Current'], fontsize=fontsize, rotation=-17.5)
            else:
                plt.xticks(np.arange(N),['40%', '50%', '60%', '70%', '80%', '90%', '95%'], fontsize=fontsize)
        else:
            plt.xticks(np.arange(np.size(constraintype)),constraintype, fontsize=fontsize)
    elif networktype == "brown":
        plt.xticks(np.arange(N),["2020","2025","2030","2035","2040","2045","2050"], fontsize=fontsize)
    else: 
        assert False, "choose either green og brown as type"
    
    # title
    if title != "none":
        plt.title(title)
    
    # Show plot
    plt.show(all)
    
    return fig

#%% CompinedPriceEvolution

def CompinedPriceEvolution(meanPrices, quantileMeanPrices, constraint, ymax, title="none", figsize=(12,3), fontsize=12, dpi=200):

    # Create figure
    fig = plt.figure(figsize=figsize,dpi=dpi) # Figure size and quality
    gs = fig.add_gridspec(1, 8) # Grid size

    # Setup gridspace to according figures
    axs = []
    axs.append( fig.add_subplot(gs[0,0:2]) )    # Plot 1: Real prices
    axs.append( fig.add_subplot(gs[0,2:5]) )    # Plot 2: Elec Only prices
    axs.append( fig.add_subplot(gs[0,5:8]) )    # Plot 3: Elec sector coupled prices

    color = ["tab:blue"]
    label = ["Mean"]
    quantilesLabel = ["","25%-75%","5%-95%"]
    alpha = [0.2,0.4,0.2]

    #% Plotting - Inserting real data

    for h in range(3):
        
        meanPrice = pd.DataFrame(np.array(meanPrices[h]))
        quantileMeanPrice = pd.DataFrame(np.array(quantileMeanPrices[h]), columns=["0.05","0.25","0.75","0.95"])
        
        N = len(meanPrice)
        
        i = 0
        
        # Plot quantiles
        axs[h].plot(meanPrice,color[i],linewidth=2,label=label[i],marker='o', markersize=3)
        for j, k in enumerate(list(quantileMeanPrice.columns)):
            axs[h].plot(quantileMeanPrice[k],color=color[i],alpha=0) # only there to fix legend "best"
        
            if j != 0:
                
                quantile1 = quantileMeanPrice[quantileMeanPrice.columns[j-1]]
                quantile2 = quantileMeanPrice[quantileMeanPrice.columns[j]]
                axs[h].fill_between(range(N), quantile1, quantile2,
                                 label=quantilesLabel[j-1],
                                 color=color[i],
                                 alpha=alpha[j-1])
        for j in range(N): # quantile lines
            for k in range(3):
                axs[h].plot([j,j],[quantileMeanPrice.iloc[j][k],quantileMeanPrice.iloc[j][k+1]],
                          color=color[i],
                          alpha=alpha[k]+0.1,
                          linestyle=(0,(2,2)))
        
        # general plot setup
        axs[h].tick_params(axis='both',
                           labelsize=12)
        axs[h].set(ylim=[-5,ymax])
        axs[h].grid(alpha=0.15)
        
        # Only y label on the left figure
        if h == 0:
            axs[h].set_ylabel("Price [€/MWh]", fontsize=fontsize)
        
        # No ticks on the middel and right figure
        if h != 0:
            axs[h].set_yticklabels([])
            
        # Only legend on the right figure
        if h == 2:
            axs[h].legend(loc="upper left", bbox_to_anchor=(1, 1.05), fontsize=fontsize)
        
        # x axis ticks acording to names
        axs[h].set_xticks(range(0,N,1))
        axs[h].set_xticklabels(constraint[h],rotation=45)
        
        if title != "none":
            axs[h].set_title(label=title[h], fontweight="bold", size=fontsize+2) # Title
        
    # Show plot
    plt.show(all)
    
    return fig