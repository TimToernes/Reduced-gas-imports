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

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.ticker as tick
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter

# Timer
t0 = time.time() # Start a timer

# Import functions file
sys.path.append(os.path.split(os.getcwd())[0])
from functions_file import *    

#%% Setup paths

# Directory of files
directory = os.path.split(os.getcwd())[0] + "\\Data\\"

# Figure path
figurePath = os.getcwd() + "\\Figures\\"

# List of file names (Elec Only)
filename_ElecOnly = [
                #"postnetwork-elec_only_0.125_0.6.h5",
                #"postnetwork-elec_only_0.125_0.5.h5",
                "postnetwork-elec_only_0.125_0.4.h5",
                "postnetwork-elec_only_0.125_0.3.h5",
                "postnetwork-elec_only_0.125_0.2.h5",
                "postnetwork-elec_only_0.125_0.1.h5",
                "postnetwork-elec_only_0.125_0.05.h5"]

# List of file names (Elec sector coupled)
filename_ElecHeatV2G50 = [
                #"postnetwork-elec_heat_v2g50_0.125_0.6.h5",
                #"postnetwork-elec_heat_v2g50_0.125_0.5.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.4.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.3.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.2.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.1.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.05.h5"]

# List of file names (Elec Real)
filename2019 = directory + "Real Electricity prices\\2019\\electricityPrices2019.csv" 
filename2020 = directory + "Real Electricity prices\\2020\\electricityPrices2020.csv" 
filename2021 = directory + "Real Electricity prices\\2021\\electricityPrices2021.csv" 

filenames_Real = [filename2019,filename2020,filename2021]

#%% Setup constrain names

# List of constraints
#constraints_CO2 = ["40%", "50%", "60%", "70%", "80%", "90%", "95%"]
constraints_CO2 = ["60%", "70%", "80%", "90%", "95%"]
yearNames = ["2019","2020","2021"]

constraint = [yearNames, constraints_CO2, constraints_CO2]

# titles
title = ["Real Prices",
         "Electricity Only",
         "Electricity Sector Coupled"]

#%% Variables to save for each case
meanPrices = [] # flat mean prices
meanPricesWC = [] # weighted agross time mean price
meanPricesWT = [] # weighted agross country mean price
quantileMeanPricesC = [] # flat quantiles across countries
quantileMeanPricesT = [] # flat quantiles across time
quantileMeanPricesWC = [] # weighted quantiles agross countries
quantileMeanPricesWT = [] # weighted quantiles agross time


#%% Data collection for 2019, 2020, 2021

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

for i in range(3):
    # Load real electricity prices
    realPrices = pd.read_csv((filenames_Real[i]), index_col=0)
    realPrices = pd.DataFrame(data=realPrices.values, index=pd.to_datetime(realPrices.index), columns=pd.Index(realPrices.columns))
    
    # Country weight
    sizeWeight = realPrices.sum() / realPrices.sum().sum()
    
    # ----------------------- NP Mean --------------------#
    # --- Elec ---
    # Mean price for country
    meanPrice = realPrices.mean().mean()
    
    # weighted average
    meanPriceWC = np.average(np.average(realPrices,axis=0), weights=sizeWeight)
    meanPriceWT = np.average(np.average(realPrices,axis=1,weights=sizeWeight))
    
    # append min, max and mean to matrix
    meanPriceElec.append(meanPrice)
    meanPriceElecWC.append(meanPriceWC)
    meanPriceElecWT.append(meanPriceWT)
    
    
    # ----------------------- NP Quantile (Elec) --------------------#
    # --- Elec ---
    # Mean price for flat country
    quantileMeanPriceC = np.quantile(realPrices.mean(),[0.05,0.25,0.75,0.95])
    
    # Mean price for flat time
    quantileMeanPriceT = np.quantile(realPrices.mean(axis=1),[0.05,0.25,0.75,0.95])  
    
    # Mean price for weight country
    quantileMeanPriceWC = np.quantile(np.average(realPrices,axis=0),[0.05,0.25,0.75,0.95])  
    
    # Mean price for weight time
    quantileMeanPriceWT = np.quantile(np.average(realPrices,axis=1,weights=sizeWeight),[0.05,0.25,0.75,0.95]) 
    
    # append min and mean to matrix (flat country)
    quantileMeanPriceElecC.append(quantileMeanPriceC)
    
    # append min and mean to matrix (flat time)
    quantileMeanPriceElecT.append(quantileMeanPriceT)
    
    # append min and mean to matrix (weight country)
    quantileMeanPriceElecWC.append(quantileMeanPriceWC)
    
    # append min and mean to matrix (weight time)
    quantileMeanPriceElecWT.append(quantileMeanPriceWT)

# Save values
meanPrices.append(meanPriceElec) # flat mean prices
meanPricesWC.append(meanPriceElecWC)  # weighted agross time mean price
meanPricesWT.append(meanPriceElecWT)  # weighted agross country mean price
quantileMeanPricesC.append(quantileMeanPriceElecC)  # flat quantiles across countries
quantileMeanPricesT.append(quantileMeanPriceElecT)  # flat quantiles across time
quantileMeanPricesWC.append(quantileMeanPriceElecWC)  # weighted quantiles agross countries
quantileMeanPricesWT.append(quantileMeanPriceElecWT)  # weighted quantiles agross time

#%% Quantiles data collection - Elec Only

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

# For loop
for file in filename_ElecOnly:
    # Network
    network = pypsa.Network(directory + "elec_only\\" + file)

    # Get the names of the data
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Country weight
    sizeWeight = network.loads_t.p_set.sum() / network.loads_t.p_set.sum().sum()
    
    # Prices for electricity for each country (restricted to 1000 €/MWh)
    priceElec = FilterPrice(network.buses_t.marginal_price[dataNames], 465)
    

    # ----------------------- NP Mean (Elec) --------------------#
    # --- Elec ---
    # Mean price for country
    meanPrice = priceElec.mean().mean()
    
    # weighted average
    meanPriceWC = np.average(np.average(priceElec,axis=0), weights=sizeWeight)
    meanPriceWT = np.average(np.average(priceElec,axis=1,weights=sizeWeight))
    
    # append min, max and mean to matrix
    meanPriceElec.append(meanPrice)
    meanPriceElecWC.append(meanPriceWC)
    meanPriceElecWT.append(meanPriceWT)
    
    
    # ----------------------- NP Quantile (Elec) --------------------#
    # --- Elec ---
    # Mean price for flat country
    quantileMeanPriceC = np.quantile(priceElec.mean(),[0.05,0.25,0.75,0.95])
    
    # Mean price for flat time
    quantileMeanPriceT = np.quantile(priceElec.mean(axis=1),[0.05,0.25,0.75,0.95])  
    
    # Mean price for weight country
    quantileMeanPriceWC = np.quantile(np.average(priceElec,axis=0),[0.05,0.25,0.75,0.95])  
    
    # Mean price for weight time
    quantileMeanPriceWT = np.quantile(np.average(priceElec,axis=1,weights=sizeWeight),[0.05,0.25,0.75,0.95]) 
    
    # append min and mean to matrix (flat country)
    quantileMeanPriceElecC.append(quantileMeanPriceC)
    
    # append min and mean to matrix (flat time)
    quantileMeanPriceElecT.append(quantileMeanPriceT)
    
    # append min and mean to matrix (weight country)
    quantileMeanPriceElecWC.append(quantileMeanPriceWC)
    
    # append min and mean to matrix (weight time)
    quantileMeanPriceElecWT.append(quantileMeanPriceWT)

# Save values
meanPrices.append(meanPriceElec) # flat mean prices
meanPricesWC.append(meanPriceElecWC)  # weighted agross time mean price
meanPricesWT.append(meanPriceElecWT)  # weighted agross country mean price
quantileMeanPricesC.append(quantileMeanPriceElecC)  # flat quantiles across countries
quantileMeanPricesT.append(quantileMeanPriceElecT)  # flat quantiles across time
quantileMeanPricesWC.append(quantileMeanPriceElecWC)  # weighted quantiles agross countries
quantileMeanPricesWT.append(quantileMeanPriceElecWT)  # weighted quantiles agross time

#%% Quantiles data collection - Elec sector coupled

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

# For loop
for file in filename_ElecHeatV2G50:
    # Network
    network = pypsa.Network(directory + "elec_heat_v2g50\\" + file)

    # Get the names of the data
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Country weight
    sizeWeight = network.loads_t.p_set[dataNames].sum() / network.loads_t.p_set[dataNames].sum().sum()
    
    # Prices for electricity for each country (restricted to 1000 €/MWh)
    priceElec = FilterPrice(network.buses_t.marginal_price[dataNames], 465)
    

    # ----------------------- NP Mean (Elec) --------------------#
    # --- Elec ---
    # Mean price for country
    meanPrice = priceElec.mean().mean()
    
    # weighted average
    meanPriceWC = np.average(np.average(priceElec,axis=0), weights=sizeWeight)
    meanPriceWT = np.average(np.average(priceElec,axis=1,weights=sizeWeight))
    
    # append min, max and mean to matrix
    meanPriceElec.append(meanPrice)
    meanPriceElecWC.append(meanPriceWC)
    meanPriceElecWT.append(meanPriceWT)
    
    
    # ----------------------- NP Quantile (Elec) --------------------#
    # --- Elec ---
    # Mean price for flat country
    quantileMeanPriceC = np.quantile(priceElec.mean(),[0.05,0.25,0.75,0.95])
    
    # Mean price for flat time
    quantileMeanPriceT = np.quantile(priceElec.mean(axis=1),[0.05,0.25,0.75,0.95])  
    
    # Mean price for weight country
    quantileMeanPriceWC = np.quantile(np.average(priceElec,axis=0),[0.05,0.25,0.75,0.95])  
    
    # Mean price for weight time
    quantileMeanPriceWT = np.quantile(np.average(priceElec,axis=1,weights=sizeWeight),[0.05,0.25,0.75,0.95]) 
    
    # append min and mean to matrix (flat country)
    quantileMeanPriceElecC.append(quantileMeanPriceC)
    
    # append min and mean to matrix (flat time)
    quantileMeanPriceElecT.append(quantileMeanPriceT)
    
    # append min and mean to matrix (weight country)
    quantileMeanPriceElecWC.append(quantileMeanPriceWC)
    
    # append min and mean to matrix (weight time)
    quantileMeanPriceElecWT.append(quantileMeanPriceWT)

# Save values
meanPrices.append(meanPriceElec) # flat mean prices
meanPricesWC.append(meanPriceElecWC)  # weighted agross time mean price
meanPricesWT.append(meanPriceElecWT)  # weighted agross country mean price
quantileMeanPricesC.append(quantileMeanPriceElecC)  # flat quantiles across countries
quantileMeanPricesT.append(quantileMeanPriceElecT)  # flat quantiles across time
quantileMeanPricesWC.append(quantileMeanPriceElecWC)  # weighted quantiles agross countries
quantileMeanPricesWT.append(quantileMeanPriceElecWT)  # weighted quantiles agross time

#%% Plot quantiles - Combined Version

# ----------------------- Price evalution (Elec) (flat country) --------------------#
fig = CompinedPriceEvolution(meanPrices,quantileMeanPricesC, constraint, ymax=140, title=title)
figtitle =  ("Combined" + " - ENP CO2 Evolution")
path = figurePath + "flat average (countries)\\"
SavePlot(fig, path, figtitle)

# ----------------------- Price evalution (Elec) (flat time) --------------------#
fig = CompinedPriceEvolution(meanPrices,quantileMeanPricesT, constraint, ymax=240, title=title)
figtitle =  ("Combined" + " - ENP CO2 Evolution")
path = figurePath + "flat average (time)\\"
SavePlot(fig, path, figtitle)

# ----------------------- Price evalution (Elec) (weighted country) --------------------#
fig = CompinedPriceEvolution(meanPricesWC,quantileMeanPricesWC, constraint, ymax=140, title=title)
figtitle =  ("Combined" + " - ENP CO2 Evolution")
path = figurePath + "weighted average (countries)\\"
SavePlot(fig, path, figtitle)

# ----------------------- Price evalution (Elec) (weighted time) --------------------#
fig = CompinedPriceEvolution(meanPricesWT,quantileMeanPricesWT, constraint, ymax=240, title=title)
figtitle =  ("Combined" + " - ENP CO2 Evolution")
path = figurePath + "weighted average (time)\\"
SavePlot(fig, path, figtitle)
    
