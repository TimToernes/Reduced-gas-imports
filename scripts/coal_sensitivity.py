# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:28:58 2022

@author: au485969
"""
from matplotlib import cm
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

fs = 16 # fontsize
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs

n_header = 4 

cmap = cm.get_cmap('gist_heat', 8)  # matplotlib color palette name, n colors
for i in range(cmap.N):
    rgb = cmap(i)[:3]  # will return rgba, we take only first 3 so we get rgb

colpal1 = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in np.arange(8)]

c_df = pd.read_csv(
        '../results/version-gaslimit3100-3H/csvs/capacities.csv',
        index_col=list(range(2)),
        header=list(range(n_header))
    )

b_df = pd.read_csv(
        '../results/version-gaslimit3100-3H/csvs/supply_energy.csv',
        index_col=list(range(3)),
        header=list(range(n_header))
    )

b_base_1p5C = b_df.T.loc[('37m', '1.0', '3H-T-H-B-I-solar+p3-dist1-cb25.7ex0')].T
b_nogas_1p5C = b_df.T.loc[('37m', '1.0', '3H-T-H-B-I-solar+p3-dist1-cb25.7ex0-gasconstrained')].T

b_base_2C = b_df.T.loc[('37m', '1.0', '3H-T-H-B-I-solar+p3-dist1-cb73.9ex0')].T
b_nogas_2C = b_df.T.loc[('37m', '1.0', '3H-T-H-B-I-solar+p3-dist1-cb73.9ex0-gasconstrained')].T

c_base_1p5C = c_df.T.loc[('37m', '1.0', '3H-T-H-B-I-solar+p3-dist1-cb25.7ex0')].T
c_nogas_1p5C = c_df.T.loc[('37m', '1.0', '3H-T-H-B-I-solar+p3-dist1-cb25.7ex0-gasconstrained')].T

c_base_2C = c_df.T.loc[('37m', '1.0', '3H-T-H-B-I-solar+p3-dist1-cb73.9ex0')].T
c_nogas_2C = c_df.T.loc[('37m', '1.0', '3H-T-H-B-I-solar+p3-dist1-cb73.9ex0-gasconstrained')].T

planning_horizons = [2020,2025,2030,2035,2040,2045,2050]

# b_dict = {'1.5C base':b_base_1p5C,'2C base':b_base_2C,'1.5C gas limit':b_nogas_1p5C,'2C gas limit':b_nogas_2C}
# c_dict = {'1.5C base':c_base_1p5C,'2C base':c_base_2C,'1.5C gas limit':c_nogas_1p5C,'2C gas limit':c_nogas_2C}

b_dict_1p5 = {'1.5C base':b_base_1p5C,'1.5C gas limit':b_nogas_1p5C}
c_dict_1p5 = {'1.5C base':c_base_1p5C,'1.5C gas limit':c_nogas_1p5C}

b_dict_2 = {'1.5C base':b_base_2C,'1.5C gas limit':b_nogas_2C}
c_dict_2 = {'1.5C base':c_base_2C,'1.5C gas limit':c_nogas_2C}

years = ['2025','2030','2035','2040','2045','2050']

#%% Coal consumption
b_df_sens = b_df.T.loc[('37m', '1.0')]
fig,axes = plt.subplots(nrows=1,ncols=2,figsize=[15,6])

labdic = {'-gasconstrained':'8.15 €/MWh (gas limit)',
          '-gasconstrained-costofcoal+c1.2':'9.78 €/MWh (gas limit)',
          '-gasconstrained-costofcoal+c1.5':'12.2 €/MWh (gas limit)',
          '-gasconstrained-costofcoal+c2':'16.3 €/MWh (gas limit)',
          '-gasconstrained-costofcoal+c4':'32.6 €/MWh (gas limit)',
          '-gasconstrained-costofcoal+c10':'81.5 €/MWh (gas limit)'}

temp_dic = {'cb25.7ex0':'1.5C',
            'cb73.9ex0':'2C'}
STR1 = ''
STR2 = ''
coal_df_sens = ((b_df.T.loc[('37m', '1.0','3H-T-H-B-I-solar+p3-dist1-cb25.7ex0' + STR1 + STR2)].T.loc[('coal','generators','coal')] 
                 + b_df.T.loc[('37m', '1.0','3H-T-H-B-I-solar+p3-dist1-cb25.7ex0' + STR1 + STR2)].T.loc[('lignite','generators','lignite')])/1e6).loc[years]

iit = 0
axes[0].plot(coal_df_sens,label='8.15 €/MWh (base)',color=colpal1[iit])
coal_df_sens = ((b_df.T.loc[('37m', '1.0','3H-T-H-B-I-solar+p3-dist1-cb73.9ex0' + STR1 + STR2)].T.loc[('coal','generators','coal')] 
                 + b_df.T.loc[('37m', '1.0','3H-T-H-B-I-solar+p3-dist1-cb73.9ex0' + STR1 + STR2)].T.loc[('lignite','generators','lignite')])/1e6).loc[years]
axes[1].plot(coal_df_sens,color=colpal1[iit])

it = 0
# iit = 0
for temp in ['cb25.7ex0','cb73.9ex0']:
    ax = axes[it]
    iit = 1
    for STR1 in ['-gasconstrained']:
        for STR2 in ['','-costofcoal+c1.2','-costofcoal+c1.5','-costofcoal+c2','-costofcoal+c4','-costofcoal+c10']:
            coal_df_sens = ((b_df.T.loc[('37m', '1.0','3H-T-H-B-I-solar+p3-dist1-'+ temp + STR1 + STR2)].T.loc[('coal','generators','coal')] 
                 + b_df.T.loc[('37m', '1.0','3H-T-H-B-I-solar+p3-dist1-' + temp + STR1 + STR2)].T.loc[('lignite','generators','lignite')])/1e6).loc[years]
            if temp == 'cb25.7ex0':
                ax.plot(coal_df_sens,label=labdic[STR1 + STR2],color=colpal1[iit],ls='--')
            else:
                ax.plot(coal_df_sens,color=colpal1[iit],ls='--')
            iit += 1 
    ax.set_title(temp_dic[temp],fontsize=fs,fontweight='bold')
    it += 1
    
axes[0].set_ylabel('Coal consumption [TWh]')

axes[0].set_ylim([-50,2550])
axes[1].set_ylim([-50,2550])

fig.legend(bbox_to_anchor=(0.6, 0),borderaxespad=0,prop={'size':fs})
fig.savefig('../figures/Coal_consumption_vs_coal_price.pdf', bbox_inches="tight")


#%%
# import pypsa
# n = pypsa.Network('../results/version-gaslimit3100-3H/postnetworks/elec_s370_37m_lv1.0__3H-T-H-B-I-solar+p3-dist1-cb25.7ex0_2020.nc')