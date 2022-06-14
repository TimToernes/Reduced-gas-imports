# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 19:21:06 2022

@author: au485969
"""
# import os
# from matplotlib import cm
import pypsa
# import matplotlib
# from matplotlib.dates import DateFormatter
# import matplotlib.dates as mdates
import numpy as np
import glob
import matplotlib.pyplot as plt
plt.close('all')
import pandas as pd
# from calendar import isleap
import yaml
with open('tech_colors.yaml') as file:
    tech_colors = yaml.safe_load(file)['tech_colors']
import warnings
warnings.filterwarnings("ignore")

def rename_techs(label):
    prefix_to_remove = [
        "residential ",
        "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral "
    ]
    rename_if_contains = [
        "gas boiler",
        "biogas",
        "solar thermal",
        "air heat pump",
        "ground heat pump",
        "resistive heater",
        "Fischer-Tropsch"
    ]
    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "retrofitting": "building retrofitting",
        "battery": "battery storage",
    }
    rename = {
        "solar": "solar PV",
        "Sabatier": "methanation",
        "offwind": "offshore wind",
        "offwind-ac": "offshore wind (AC)",
        "offwind-dc": "offshore wind (DC)",
        "onwind": "onshore wind",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "hydroelectricity",
        "co2 Store": "DAC",
        "co2 stored": "CO2 sequestration",
        "AC": "transmission lines",
        "DC": "transmission lines",
        "B2B": "transmission lines"
    }
    for ptr in prefix_to_remove:
        if label[:len(ptr)] == ptr:
            label = label[len(ptr):]
    for rif in rename_if_contains:
        if rif in label:
            label = rif
    for old,new in rename_if_contains_dict.items():
        if old in label:
            label = new
    for old,new in rename.items():
        if old == label:
            label = new
    return label

def rename_techs_tyndp(tech):
    tech = rename_techs(tech)
    if "CHP" in tech:
        return "CHP"
    if "heat pump" in tech:
        return "heat pump"
    if "solar" in tech:
        return "solar"
    elif tech == "Fischer-Tropsch":
        return "Fischer-Tropsch"
    elif "offshore wind" in tech:
        return "offshore wind"
    else:
        return tech
    
def legend_without_duplicate_labels(fig,axes):
    handles = []
    labels =  []
    for i in range(len(axes)):
        handles_i, labels_i = axes[i].get_legend_handles_labels()
        handles += handles_i
        labels += labels_i
    
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    a = 0.85 if len(unique) > 4 else 0.7
    fig.legend(*zip(*unique),bbox_to_anchor=(a, 0.06),borderaxespad=0,prop={'size':fs},ncol=6)
    
    
fs = 16 # fontsize
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs

#%% Read networks
path = '../results/PyPSA-Eur-Sec-0.6.0/3H_oil_lignite_nonextendable/postnetworks/'
networks = glob.glob(path + '*.nc')

no_scen_dic = {14:'1.5_2020',15:'1.5_2025',
               16:'1.5_2030',17:'1.5_2035',
               18:'1.5_2040',19:'1.5_2045',
               20:'1.5_2050',7:'1.5_2020_gc',
               8:'1.5_2025_gc',9:'1.5_2030_gc',
               10:'1.5_2035_gc',11:'1.5_2040_gc',
               12:'1.5_2045_gc',13:'1.5_2050_gc',
               35:'2_2020',36:'2_2025',
               37:'2_2030',38:'2_2035',
               39:'2_2040',40:'2_2045',
               41:'2_2050',28:'2_2020_gc',
               29:'2_2025_gc',30:'2_2030_gc',
               31:'2_2035_gc',32:'2_2040_gc',
               33:'2_2045_gc',34:'2_2050_gc',
               }
scen_no_dic = {v: k for k, v in no_scen_dic.items()}

year = '2030'
temp = '2'

# gas_side = 'supply'
gas_side = 'consumption'

N = 10 # Number for moving average
important_countries = ['IT','ES','GB','NL','DE','HU']

array_annual_gas = np.zeros([len(important_countries),2])

scens = [temp + '_' + year, temp + '_' + year + '_gc']
scen_dic = {temp + '_' + year:'',
            temp + '_' + year:'',
            temp + '_'  + year + '_gc':'w. gas limit',
            temp + '_'  + year + '_gc':'w. gas limit'}

fig,axes = plt.subplots(nrows=len(important_countries),ncols=2,figsize=[15,16],sharey='row')
s_i = 0
for scen in scens:
    c_i = 0
    n_i = pypsa.Network(networks[scen_no_dic[scen]])
    buses = n_i.buses.loc[n_i.buses.carrier == 'AC'].index
    countries = n_i.buses.loc[n_i.buses.carrier == 'AC'].country.values
    zip_iterator = zip(buses, countries)
    bus_country_dic = dict(zip_iterator)
    country_bus_dic = {v: k for k, v in bus_country_dic.items()}
    country_bus_dic['GB'] = ['GB4 0','GB5 0']
    country_bus_dic['IT'] = ['IT0 0','IT6 0']
    country_bus_dic['DK'] = ['DK0 0','DK1 0']
    country_bus_dic['ES'] = ['ES0 0','ES3 0']
    
    #%% Group by gas consumers
    for country in important_countries:
        
        bus = country_bus_dic[country]
        weighting = 8760/n_i.snapshots.shape[0]
        e_t = n_i.links_t.p1 if gas_side == 'supply' else n_i.links_t.p0                # primary energy consumption time series
        gas_cons = n_i.links.query('bus0 == "EU gas"')                                  # gas consumers
        gas_cons['bus'] = gas_cons['bus1'].str.split(' ',expand=True)[0]
        
        gas_sups = n_i.links.query('bus1 == "EU gas"')                                  # gas suppliers
        gas_sups['bus'] = gas_sups['bus0'].str.split(' ',expand=True)[0]
        
        gas_X = gas_sups if gas_side == 'supply' else gas_cons
        sign = -1 if gas_side == 'supply' else 1
        
        if type(bus) != str:
            gas_n = pd.concat([gas_X.loc[gas_X.bus == bus[0][0:3]],gas_X.loc[gas_X.bus == bus[1][0:3]]]) # national gas consumers/suppliers
            bus = bus[0]
        else:
            gas_n = gas_X.loc[gas_X.bus == bus[0:3]]                                    # national gas consumers/suppliers
        df_gas_t = pd.DataFrame(index=n_i.snapshots)                                    # gas consumption/supply time series dataframe
        for carrier in gas_n.carrier.unique():
            gas_t = e_t[gas_n.carrier[gas_n.carrier == carrier].index].sum(axis=1)      # gas consumption/supply by link carrier
            df_gas_t[carrier] = sign*gas_t*weighting/1e3
        df_gas_t.index = pd.date_range('1/1/2019','1/1/2020',freq='3h')[0:-1]
        
        array_annual_gas[c_i,s_i] = df_gas_t.sum().sum()/1e3
        
        #%% Grouping
        df_gas_t = df_gas_t.groupby(rename_techs_tyndp, axis=1).sum()
        df_gas_t = df_gas_t[df_gas_t.sum()[df_gas_t.sum() > 1].index] # Dropping carrier if sum less than 1 GWh
        
        #%% Plotting
        ax = axes[c_i,s_i]
        colors=[tech_colors[t] for t in list(df_gas_t.columns.get_level_values(0))]
        df_gas_t.rolling(N).mean().plot(kind='area',stacked=True,ax=ax,color=colors,alpha=1,linewidth=0.)
        ax.set_xticklabels(['','Feb','','Apr','','Jun','','Aug','','Oct','','Dec'])
        ax.set_ylabel('GWh')
        ax.set_title(bus_country_dic[bus] + ' ' + scen_dic[scen],fontsize=fs)
        ax.set_ylim([0,ax.get_ylim()[1]])
        if scen[-2:] == 'gc':
            annual_reduction = np.abs(np.round(array_annual_gas[c_i,s_i] - array_annual_gas[c_i,s_i-1])) # in TWh
            annual_reduction_per = (array_annual_gas[c_i,s_i] - array_annual_gas[c_i,s_i-1])/array_annual_gas[c_i,s_i-1]*100 # in percentage
            mp = '' if annual_reduction_per < 0 else '+'
            ax.text(0.35,0.6,mp + str(int(np.round(annual_reduction_per))) + '% (' + str(int(annual_reduction)) + ' TWh)',fontsize=fs,transform=ax.transAxes,color='gray')
        # else:
        #     df_gas_t.sum(axis=1).rolling(3*N).mean().plot(ax=axes[c_i,1],color='gray',linewidth=0.5,label='Baseline')
        c_i += 1
    
    s_i += 1

legend_without_duplicate_labels(fig,axes.flatten())

for ax in axes.flatten():
    ax.legend().remove()
    
fig.subplots_adjust(hspace=0.4)
fig.subplots_adjust(wspace=0.1)
fig.suptitle('Gas consumption: ' + temp + 'C ' + year,fontsize=1.5*fs,y=0.93)

fig.savefig('../figures/Gas_consumption_' + temp + 'C_' + year + '.pdf', bbox_inches="tight")