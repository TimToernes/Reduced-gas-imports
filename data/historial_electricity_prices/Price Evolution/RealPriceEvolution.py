# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:14:36 2022

@author: jones

"""

#%% Function setup

# Import libraries
import os
import sys
import pypsa
import numpy as np
import pandas as pd
import time
import math

# Timer
t0 = time.time() # Start a timer

# Import functions file
sys.path.append(os.path.split(os.getcwd())[0])
from functions_file import *    

#%% Setup paths

# Directory of files
directory = os.path.split(os.getcwd())[0] + "\\Data\\Real Electricity prices\\"

# Figure path
figurePath = os.getcwd() + "\\Figures\\"

# List of file names (CO2)
filename2019 = directory + "2019\\electricityPrices2019.csv" 
filename2020 = directory + "2020\\electricityPrices2020.csv" 
filename2021 = directory + "2021\\electricityPrices2021.csv" 

filenames = [filename2019,filename2020,filename2021]

#%% Setup constrain names

# List of year names
yearNames = ["2019","2020","2021"]

#%% Quantiles data collection

# Variable to store nodal price mean and standard variation
meanPriceElec = []
stdMeanPriceElec = []
meanPriceElecWC = []
meanPriceElecWT = []
quantileMeanPriceElecC = []
quantileMinPriceElecC = []
quantileMeanPriceElecT = []
quantileMinPriceElecT = []
quantileMeanPriceElecWC = []
quantileMeanPriceElecWT = []

#%% Data collection for 2019, 2020, 2021

for i in range(3):
    # Load real electricity prices
    realPrices = pd.read_csv((filenames[i]), index_col=0)
    realPrices = pd.DataFrame(data=realPrices.values, index=pd.to_datetime(realPrices.index), columns=pd.Index(realPrices.columns))
    
    # Country weight
    sizeWeight = realPrices.sum() / realPrices.sum().sum()
    
    # ----------------------- NP Mean --------------------#
    # --- Elec ---
    # Mean price for country
    minPrice = realPrices.min().mean()
    meanPrice = realPrices.mean().mean()
    
    # weighted average
    meanPriceWC = np.average(np.average(realPrices,axis=0), weights=sizeWeight)
    meanPriceWT = np.average(np.average(realPrices,axis=1,weights=sizeWeight))
    
    # append min, max and mean to matrix
    meanPriceElec.append([minPrice, meanPrice])
    meanPriceElecWC.append([minPrice, meanPriceWC])
    meanPriceElecWT.append([minPrice, meanPriceWT])
    
    
    # ----------------------- NP Quantile (Elec) --------------------#
    # --- Elec ---
    # Mean price for flat country
    quantileMinPriceC = np.quantile(realPrices.min(),[0.05,0.25,0.75,0.95])
    quantileMeanPriceC = np.quantile(realPrices.mean(),[0.05,0.25,0.75,0.95])
    
    # Mean price for flat time
    quantileMinPriceT = np.quantile(realPrices.min(axis=1),[0.05,0.25,0.75,0.95])
    quantileMeanPriceT = np.quantile(realPrices.mean(axis=1),[0.05,0.25,0.75,0.95])  
    
    # Mean price for weight country
    quantileMeanPriceWC = np.quantile(np.average(realPrices,axis=0),[0.05,0.25,0.75,0.95])  
    
    # Mean price for weight time
    quantileMeanPriceWT = np.quantile(np.average(realPrices,axis=1,weights=sizeWeight),[0.05,0.25,0.75,0.95]) 
    
    # append min and mean to matrix (flat country)
    quantileMeanPriceElecC.append(quantileMeanPriceC)
    quantileMinPriceElecC.append(quantileMinPriceC)
    
    # append min and mean to matrix (flat time)
    quantileMeanPriceElecT.append(quantileMeanPriceT)
    quantileMinPriceElecT.append(quantileMinPriceT)
    
    # append min and mean to matrix (weight country)
    quantileMeanPriceElecWC.append(quantileMeanPriceWC)
    
    # append min and mean to matrix (weight time)
    quantileMeanPriceElecWT.append(quantileMeanPriceWT)

#%% Plot quantiles - CO2 constraint

# ----------------------- Price evalution (Elec) (flat country) --------------------#
title =  "Real Elec NP Evolution (flat average across countries)"
fig = PriceEvolution(meanPriceElec,quantileMeanPriceElecC, quantileMinPriceElecC, ylim=140, constraintype=yearNames, networktype="green", title=title, figsize=[6,3], fontsize=16)
title =  ("Real Price" + " - ENP CO2 Evolution")
path = figurePath + "flat average (countries)\\"
SavePlot(fig, path, title)

# ----------------------- Price evalution (Elec) (flat time) --------------------#
title =  "Real Elec NP Evolution (flat average across time)"
fig = PriceEvolution(meanPriceElec,quantileMeanPriceElecT, quantileMinPriceElecT, ylim=240, constraintype=yearNames, networktype="green", title=title, figsize=[6,3], fontsize=16)
title =  ("Real Price" + " - ENP CO2 Evolution")
path = figurePath + "flat average (time)\\"
SavePlot(fig, path, title)

# ----------------------- Price evalution (Elec) (weighted country) --------------------#
title =  "Real Elec NP Evolution (weighted average across countries)"
fig = PriceEvolution(meanPriceElecWC,quantileMeanPriceElecWC, quantileMinPriceElecC, ylim=140, constraintype=yearNames, networktype="green", title=title, figsize=[6,3], fontsize=16)
title =  ("Real Price" + " - ENP CO2 Evolution")
path = figurePath + "weighted average (countries)\\"
SavePlot(fig, path, title)

# ----------------------- Price evalution (Elec) (weighted time) --------------------#
title =  "Real Elec NP Evolution (weighted average across time)"
fig = PriceEvolution(meanPriceElecWT,quantileMeanPriceElecWT, quantileMinPriceElecT, ylim=240, constraintype=yearNames, networktype="green", title=title, figsize=[6,3], fontsize=16)
title =  ("Real Price" + " - ENP CO2 Evolution")
path = figurePath + "weighted average (time)\\"
SavePlot(fig, path, title)

