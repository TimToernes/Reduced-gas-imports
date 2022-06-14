# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
idx = pd.IndexSlice

from vresutils import Dict
import yaml
snakemake = Dict()
with open('config.yaml', encoding='utf8') as f:
    snakemake.config = yaml.safe_load(f)
color=snakemake.config['plotting']['tech_colors']

plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.titlesize'] = 18
plt.figure(figsize=(10, 20))
gs1 = gridspec.GridSpec(7, 2)
gs1.update(wspace=0.05, hspace=0.1)

version='baseline' #'sensitivity-fixedcosts' #  #

transmission='1.0'
cluster='37'
decay='ex0'

budgets = ['25.7',
           '35.4',
           '45.0',
           '54.7', 
           '64.3', 
           '73.9']

NET=['DAC1', 'gas for industry CC3',
     'process emissions CC2', 'solid biomass for industry CC3',
     'urban central solid biomass CHP CC4',
     'SMR CC3','urban central gas CHP CC4',
     ]

NET_names=['DAC', 'gas for industry CC',
           'process emissions CC', 'solid biomass for industry CC',
           'solid biomass CHP CC',
           'SMR CC', 'gas CHP CC',
           ]

CCUS_names=['sequestered CO2','Fischer-Tropsch', 'Methanation', 'Helmeth']

colors=[color['DAC'],
        'gray', #color['gas for industry'],
        color['process emissions'], 
        'lightgreen', #color['solid biomass for industry'],
        'darkgreen', #color['biogas']
        color['SMR'],
        color['gas'],]

colors2=['darkorange', #color['co2 stored'], 
         'gold', #color['Fischer-Tropsch'], 
         color['Sabatier'], 
         color['helmeth']]

CCUS=[ 'co2 stored','Fischer-Tropsch2', 'Sabatier2', 'helmeth2']


plt.figure(figsize=(10, 20))
gs1 = gridspec.GridSpec(7, 2)
gs1.update(wspace=0.05, hspace=0.1)
budgets=['25.7',
         '25.7-gasconstrained',
         '73.9',
         '73.9-gasconstrained']

title={'25.7':'1.5$^{\circ}$C',
       '25.7-gasconstrained':'1.5$^{\circ}$C- gas limited',
       '73.9':           '2.0$^{\circ}$C',
       '73.9-gasconstrained':'2.0$^{\circ}$C-gas limited',}

for i,budget in enumerate(budgets):
    ax1 = plt.subplot(gs1[i,0])
    ax2 = plt.subplot(gs1[i,1])
    
    balances_df = pd.read_csv('results/version-0.6.0/csvs/supply_energy.csv',
                                  index_col=list(range(3)),
                                  header=list(range(4)))
    if 'gasconstrained' in budget:
        opt= '3H-T-H-B-I-A-solar+p3-dist1-cb{}ex0-gasconstrained'.format(budget.split('-')[0])
    else:
        opt= '3H-T-H-B-I-A-solar+p3-dist1-cb{}ex0'.format(budget)
    
    
    sel = 0.000001*balances_df.loc[idx['co2 stored', ['links','stores'], :],idx[cluster, transmission, opt,:]].droplevel([0,1]) #CO2 -> Mt CO2
    
    # Add one value per year to plot step-wise figures
    data=sel.loc[NET]
    data.columns=[int(x) for x in data.columns.get_level_values(3)]
    for year in range(2020,2055,5):
        for j in range(0,5):
            data[year-2+j]=data[year]
    data=data.reindex(sorted(data.columns), axis=1)
    data.drop(columns=[2018,2019,2020,2051,2052])
    ax1.stackplot(data.columns, data, colors=colors)
    
     # Add one value per year to plot step-wise figures
    data=-sel.loc[CCUS]
    data.columns=[int(x) for x in data.columns.get_level_values(3)]
    for year in range(2020,2055,5):
        for j in range(0,5):
            data[year-2+j]=data[year]
    data=data.reindex(sorted(data.columns), axis=1)
    data.drop(columns=[2018,2019,2020,2051,2052])
    ax2.stackplot(data.columns, data, colors=colors2)
    
    ax1.set_ylabel("MtCO$_2$/a")
    
    ax1.set_xticks([2030, 2040, 2050])
    ax2.set_xticks([2030, 2040, 2050])
    if i!=(len(budgets)-1):
        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
    else:
        ax1.set_xticklabels(['2030', '2040', '2050'])
        ax2.set_xticklabels(['2030', '2040', '2050'])
        
    ax1.set_yticks([0, 200, 400, 600, 800,1000])
    ax2.set_yticks([0, 200, 400, 600, 800, 1000])
    ax2.set_yticklabels([])
    ax1.set_ylim([0, 1100])
    ax2.set_ylim([0, 1100])
    ax1.set_xlim([2020, 2050])
    ax2.set_xlim([2020, 2050])
    ax1.text(2021, 900, title[budget], fontsize=18)


NET=['DAC1', 'gas for industry CC3',
     'process emissions CC2', 'solid biomass for industry CC3',
     'urban central solid biomass CHP CC4',
     'SMR CC3','urban central gas CHP CC4',
     ]

NET_names2=['DAC', 'gas for industry CC',
           'process emissions CC', 'solid biomass for industry CC',
           'solid biomass CHP CC',
           'SMR CC', 'gas CHP CC',
           ]

ax1.legend(fancybox=True, fontsize=16, loc=(-0.1, -1.3), facecolor='white', 
           frameon=True, ncol=1, labels=NET_names)
ax2.legend(fancybox=True, fontsize=16, loc=(0.1, -1.0), facecolor='white', 
           frameon=True, ncol=1, labels=CCUS_names)

plt.savefig('figures/NET.png',
            dpi=600, bbox_inches='tight')