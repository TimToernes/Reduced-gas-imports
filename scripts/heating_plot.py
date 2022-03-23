# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:17:44 2022
@author: au485969
"""

# import os
import pypsa
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
plt.close('all')
import warnings
warnings.filterwarnings("ignore")
lw = 3
fs = 20 # fontsize
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs

with open('tech_colors.yaml') as file:
    tech_colors = yaml.safe_load(file)['tech_colors']
tech_colors['ambient heat'] = tech_colors['heat pumps']

n_header = 4

b_df = pd.read_csv(
        '../results/version-gaslimit3100-3H/csvs/supply_energy.csv',
        index_col=list(range(3)),
        header=list(range(n_header))
    )

c_df = pd.read_csv(
        '../results/version-gaslimit3100-3H/csvs/capacities.csv',
        index_col=list(range(2)),
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

e_dict = {'1.5C base':b_base_1p5C,'2C base':b_base_2C,'1.5C gas limit':b_nogas_1p5C,'2C gas limit':b_nogas_2C}

c_dict_1p5 = {'1.5C base':c_base_1p5C,'1.5C gas limit':c_nogas_1p5C}
c_dict_2 = {'2C base':c_base_2C,'2C gas limit':c_nogas_2C}

planning_horizons = [2020,2025,2030,2035,2040,2045,2050]
#%% Heating demand INDEX - Remember to add agriculture if new studies are made with PyPSA-Eur-Sec 0.6.0
residential_rural_heat = b_df.loc[('residential rural heat','loads','residential rural heat')] 
residential_urban_decentral_heat = b_df.loc[('residential urban decentral heat','loads','residential urban decentral heat')]

services_rural_heat = b_df.loc[('services rural heat','loads','services rural heat')]
services_urban_decentral_heat = b_df.loc[('services urban decentral heat','loads','services urban decentral heat')]
services_industry_heat = b_df.loc[('services urban decentral heat','loads','low-temperature heat for industry')]

urban_central_heat = b_df.loc[('urban central heat','loads','urban central heat')]
urban_central_industry_heat = b_df.loc[('urban central heat','loads','low-temperature heat for industry')]

total_heat_demand = (residential_rural_heat + residential_urban_decentral_heat + services_rural_heat 
                     + services_urban_decentral_heat + services_industry_heat + urban_central_heat 
                     + urban_central_industry_heat)

heat_demand_dic = {'1.5C base':total_heat_demand.loc[('37m', '1.0', '3H-T-H-B-I-solar+p3-dist1-cb25.7ex0')],
                   '1.5C gas limit':total_heat_demand.loc[('37m', '1.0', '3H-T-H-B-I-solar+p3-dist1-cb25.7ex0')],
                   '2C base':total_heat_demand.loc[('37m', '1.0', '3H-T-H-B-I-solar+p3-dist1-cb25.7ex0')],
                   '2C gas limit':total_heat_demand.loc[('37m', '1.0', '3H-T-H-B-I-solar+p3-dist1-cb73.9ex0')]}
#%% Heating production INDEX
residential_heaters = [] # residential heating
services_heaters = [] # services heating
urban_heaters = [] # urban central heating
for i in b_df.index:
    if 'residential' in i[0]:
        residential_heaters.append(i)
    if 'services' in i[0]: 
        services_heaters.append(i)
    if 'urban central heat' in i[0]:
        urban_heaters.append(i)
heating_dict = {'residential':residential_heaters,'services':services_heaters,'urban_heaters':urban_heaters} #,'rural_heaters':rural_heaters}

#%% Acquire indices of all heating sources
heatpumps = []
gas_boilers = []
gas_chp = []
oil_boilers = []
resistive_heaters = []
biomass_chp = []
DAC_heat = []
Fischer_Tropsch_heat = []
Fuel_cell_heat = []
for i in list(heating_dict.keys()):
    heaters = e_dict['1.5C base'].loc[heating_dict[i]].index
    for k in range(len(heaters)):
        if 'heat pump' in heaters[k][2]:
            heatpumps.append(heaters[k])
        elif 'gas boiler' in heaters[k][2]:
            gas_boilers.append(heaters[k])
        elif 'gas CHP' in heaters[k][2]:
            gas_chp.append(heaters[k])
        elif 'oil boiler' in heaters[k][2]:
            oil_boilers.append(heaters[k])
        elif 'resistive heater' in heaters[k][2]:
            resistive_heaters.append(heaters[k])
        elif 'biomass' in heaters[k][2]:
            biomass_chp.append(heaters[k])
        elif 'DAC' in heaters[k][2]:
            DAC_heat.append(heaters[k])
        elif 'Fischer-Tropsch' in heaters[k][2]:
            Fischer_Tropsch_heat.append(heaters[k])
        elif 'Fuel Cell' in heaters[k][2]:
            Fuel_cell_heat.append(heaters[k])

#%% By source
sum_2020 = np.zeros(4)
sum_2025 = np.zeros(4)
sum_2030 = np.zeros(4)
sum_2035 = np.zeros(4)
sum_2040 = np.zeros(4)
sum_2045 = np.zeros(4)
sum_2050 = np.zeros(4)

leg = {}
lab = {}

color_dict2 = {
               'Gas boilers':tech_colors['gas'],
               'Gas CHP':tech_colors['CHP heat'],
               'Oil boilers':tech_colors['oil'],
               'Heatpumps':tech_colors['ambient heat'],
               'Resistive heaters':tech_colors['resistive heater'],
               'Biomass CHP':tech_colors['solid biomass'],
               'DAC heat':tech_colors['DAC'],
               'Fischer Tropsch heat':tech_colors['Fischer-Tropsch'],
               'Fuel cell heat':tech_colors['H2 Fuel Cell'],
               'Coal boilers': tech_colors['coal']
               }

heating_dict2 = {'Gas boilers':gas_boilers,
                 'Oil boilers':oil_boilers,
                 'Gas CHP':gas_chp,
                 'Heatpumps':heatpumps,
                 'Resistive heaters':resistive_heaters,
                 'Biomass CHP':biomass_chp,
                 'DAC heat':DAC_heat,
                 'Fischer Tropsch heat':Fischer_Tropsch_heat,
                 'Fuel cell heat':Fuel_cell_heat
                 }

fig,axes = plt.subplots(nrows=3,ncols=2,figsize=(14,16))
ax = axes.flatten()
count = 0
wi = 0.9
for ci in list(e_dict.keys()):
    for i in list(heating_dict2.keys()):
        lab[i] = i
        leg[i] = ax[count].bar(1,e_dict[ci].loc[heating_dict2[i]]['2025'][e_dict[ci].loc[heating_dict2[i]]['2025'] > 0].sum()/1e6,bottom=sum_2025[count],color=color_dict2[i],width=wi)
        ax[count].bar(2,e_dict[ci].loc[heating_dict2[i]]['2030'][e_dict[ci].loc[heating_dict2[i]]['2030'] > 0].sum()/1e6,bottom=sum_2030[count],color=color_dict2[i],width=wi)
        ax[count].bar(3,e_dict[ci].loc[heating_dict2[i]]['2035'][e_dict[ci].loc[heating_dict2[i]]['2035'] > 0].sum()/1e6,bottom=sum_2035[count],color=color_dict2[i],width=wi)
        ax[count].bar(4,e_dict[ci].loc[heating_dict2[i]]['2040'][e_dict[ci].loc[heating_dict2[i]]['2040'] > 0].sum()/1e6,bottom=sum_2040[count],color=color_dict2[i],width=wi)
        ax[count].bar(5,e_dict[ci].loc[heating_dict2[i]]['2045'][e_dict[ci].loc[heating_dict2[i]]['2045'] > 0].sum()/1e6,bottom=sum_2045[count],color=color_dict2[i],width=wi)
        ax[count].bar(6,e_dict[ci].loc[heating_dict2[i]]['2050'][e_dict[ci].loc[heating_dict2[i]]['2050'] > 0].sum()/1e6,bottom=sum_2050[count],color=color_dict2[i],width=wi)

        if i == 'Heatpumps':
            envelope = ax[count].step(np.arange(7)-0.5+1,[-heat_demand_dic[ci].values[1]/1e6] + list(-heat_demand_dic[ci].values[1:]/1e6),ls='-',color='k',lw=lw)
            ax[count].axvline(0.5,ls=':',color='k')

        sum_2020[count] += e_dict[ci].loc[heating_dict2[i]]['2020'][e_dict[ci].loc[heating_dict2[i]]['2020'] > 0].sum()/1e6
        sum_2025[count] += e_dict[ci].loc[heating_dict2[i]]['2025'][e_dict[ci].loc[heating_dict2[i]]['2025'] > 0].sum()/1e6
        sum_2030[count] += e_dict[ci].loc[heating_dict2[i]]['2030'][e_dict[ci].loc[heating_dict2[i]]['2030'] > 0].sum()/1e6
        sum_2035[count] += e_dict[ci].loc[heating_dict2[i]]['2035'][e_dict[ci].loc[heating_dict2[i]]['2035'] > 0].sum()/1e6
        sum_2040[count] += e_dict[ci].loc[heating_dict2[i]]['2040'][e_dict[ci].loc[heating_dict2[i]]['2040'] > 0].sum()/1e6
        sum_2045[count] += e_dict[ci].loc[heating_dict2[i]]['2045'][e_dict[ci].loc[heating_dict2[i]]['2045'] > 0].sum()/1e6
        sum_2050[count] += e_dict[ci].loc[heating_dict2[i]]['2050'][e_dict[ci].loc[heating_dict2[i]]['2050'] > 0].sum()/1e6
        
        ax[count].set_xticks(np.arange(7))
        ax[count].set_xticklabels([])
        
        if count % 2 != 0:
            ax[count].set_yticklabels([])

        ax[count].set_ylim([0,4400])
    count += 1
    
ax[0].set_ylabel(r'$\bf{Baseline}$' + '\n Heating [TWh]')
ax[2].set_ylabel(r'$\bf{Gas limit}$' + '\n Heating [TWh]')

demand_2020 = -heat_demand_dic[ci].values[0]/1e6

hist_gas_2019 = 0.39*demand_2020
hist_oil_2019 = 0.15*demand_2020
hist_wood_2019 = 0.254*demand_2020
hist_hp_2019 = 0.107*demand_2020
hist_electricity_2019 = 0.058*demand_2020
hist_coal_2019 = 0.0408*demand_2020

for i in range(len(e_dict.keys())):
    ax_i = ax[i]
    
    ax_i.bar(0,hist_gas_2019,bottom=0,color=color_dict2['Gas boilers'],width=wi)
    ax_i.bar(0,hist_oil_2019,bottom=hist_gas_2019,color=color_dict2['Oil boilers'],width=wi)
    leg_coal = ax_i.bar(0,hist_coal_2019,bottom=hist_gas_2019+hist_oil_2019,color=color_dict2['Coal boilers'],width=wi)
    ax_i.bar(0,hist_hp_2019,bottom=hist_gas_2019 + hist_oil_2019 + hist_coal_2019,color=color_dict2['Heatpumps'],width=wi)
    ax_i.bar(0,hist_electricity_2019,bottom=hist_hp_2019 + hist_gas_2019 + hist_oil_2019+hist_coal_2019,color=color_dict2['Resistive heaters'],width=wi)
    leg_biomass = ax_i.bar(0,hist_wood_2019,bottom=hist_hp_2019 + hist_gas_2019 + hist_oil_2019+hist_coal_2019+hist_electricity_2019,color='#d9b382',width=wi)
    
#%% Heat pump capacities
residential_heaters = [] # residential heating
services_heaters = [] # services heating
urban_heaters = [] # urban central heating
for j in c_df.loc[('links')].index:
    if 'residential' in j: 
        residential_heaters.append(j)
    if 'services' in j: 
        services_heaters.append(j)
    if 'urban central' in j: 
        urban_heaters.append(j)

heating_dict = {'residential':residential_heaters,'services':services_heaters,'urban_heaters':urban_heaters} 
heatpumps = []
gas_boilers = []
gas_chp = []
oil_boilers = []
resistive_heaters = []
biomass_chp = []
DAC_heat = []
Fischer_Tropsch_heat_c = []
Fuel_cell_heat = []
for i in list(heating_dict.keys()):
    heaters = c_dict_1p5['1.5C base'].loc[('links')].loc[heating_dict[i]].index
    for k in range(len(heaters)):
        if 'heat pump' in heaters[k]:
            heatpumps.append(heaters[k])
        elif 'gas boiler' in heaters[k]:
            gas_boilers.append(heaters[k])
        elif 'gas CHP' in heaters[k]:
            gas_chp.append(heaters[k])
        elif 'oil boiler' in heaters[k]:
            oil_boilers.append(heaters[k])
        elif 'resistive heater' in heaters[k]:
            resistive_heaters.append(heaters[k])
        elif 'biomass' in heaters[k]:
            biomass_chp.append(heaters[k])
        elif 'DAC' in heaters[k]:
            DAC_heat.append(heaters[k])
        elif 'Fischer Tropsch' in heaters[k]:
            Fischer_Tropsch_heat_c.append(heaters[k])
        elif 'Fuel cell' in heaters[k]:
            Fuel_cell_heat.append(heaters[k])

years = ['2025','2030','2035','2040','2045','2050']

heaters = {'Gas boilers':gas_boilers,
            'Oil boilers':oil_boilers,
            'Heatpumps':heatpumps,
            'Resistive heaters':resistive_heaters,
            'Biomass CHP':biomass_chp,
            'DAC heat':DAC_heat,
            'Fischer Tropsch heat':Fischer_Tropsch_heat_c,
            'Fuel cell heat':Fuel_cell_heat,
            }

ax_dict = {'1.5':ax[4],'2':ax[5]}
for tj in ['1.5','2']:
    ax_j = ax_dict[tj]
    
    for heater in heaters.keys():
    
        if heater == 'Heatpumps':
            COP = 3
            zord = 2
        else:
            COP = 1
            zord = 1
            
        heater_caps_base_1p5C = c_base_1p5C.loc[('links')].loc[heaters[heater]].sum().loc[years]/1e3*COP
        heater_caps_base_2C = c_base_2C.loc[('links')].loc[heaters[heater]].sum().loc[years]/1e3*COP
        heater_caps_nogas_1p5C = c_nogas_1p5C.loc[('links')].loc[heaters[heater]].sum().loc[years]/1e3*COP
        heater_caps_nogas_2C = c_nogas_2C.loc[('links')].loc[heaters[heater]].sum().loc[years]/1e3*COP
        
        heater_dict = {'1.5':[heater_caps_base_1p5C,heater_caps_nogas_1p5C],
                   '2':[heater_caps_base_2C,heater_caps_nogas_2C]}
        
        ax_j.plot(np.arange(6)+1.5,heater_dict[tj][0],ls='-',marker='o',color=color_dict2[heater],linewidth=lw,zorder=zord)
        ax_j.plot(np.arange(6)+1.5,heater_dict[tj][1],ls='--',marker='o',color=color_dict2[heater],linewidth=lw,zorder=zord)

    ax_j.set_ylim([0,995])
    ax_j.set_xticks(np.arange(7)+0.5)
    ax_j.set_xlim([-0.25,7.25])
    ax_j.set_xticklabels(['Historic','2025','2030','2035','2040','2045','2050'],rotation = 45)
    ax_j.axvline(1,ls=':',color='k')
    if tj == '2':
        ax_j.set_yticklabels([])
    

abc = ['a)','b)','c)','d)','e)','f)']
for k in range(len(ax)):
    ax_k = ax[k]
    ax_k.text(0.05, 0.95,abc[k],
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax_k.transAxes,fontsize=fs,fontweight='bold')
    
ax[4].set_ylabel('Capacities ' + r'$[GW_{heat}]$')

ax_tw = ax[5].twinx()
ax_tw.set_ylim([0,ax[4].get_ylim()[1]/(10*1e-6)/1e6]) # convert GW --> numbers of units --> millions of units
ax_tw.set_ylabel('Mil. heatpump units',fontweight='bold')
ax_tw.yaxis.set_label_coords(1.1,0.3)

ax_tw.yaxis.label.set_color(tech_colors['ambient heat'])
ax_tw.tick_params(axis='y', colors=tech_colors['ambient heat'])

ax_tw.set_yticks([10,20,30,40,50])
ax_tw.set_yticklabels([10,20,30,40,50],fontweight='bold')

ax[0].set_title(r'$\mathbf{1.5^\circ C}$ scenario',fontweight='bold',fontsize=fs)
ax[1].set_title(r'$\mathbf{2^\circ C}$ scenario',fontweight = 'bold',fontsize=fs)

ax[1].annotate('Demand', xy=(5, 3400),  xycoords='data',
                xytext=(0.9, 0.95), textcoords='axes fraction',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='right', verticalalignment='top',
                fontproperties={'size':fs}
                )

ax[5].text(4.5,900,'Baseline',fontsize=fs)
ax[5].text(1.5,730,'Gas limit',fontsize=fs)

fig.legend(list(leg.values()) + [leg_coal,leg_biomass],
           list(lab.values()) + ['Coal boilers','Biomass'], 
           bbox_to_anchor=(0.85, 0.04),borderaxespad=0,prop={'size':fs},ncol=3)

fig.subplots_adjust(hspace=0.05)
fig.subplots_adjust(wspace=0.05)

fig.savefig('../figures/Heating_by_source.pdf', bbox_inches="tight")

#%% Seasonality of heating supply and demand
def plot_series(network, carrier="heat", name="test"):

    n = network.copy()
    assign_location(n)
    assign_carriers(n)

    buses = n.buses.index[n.buses.carrier.str.contains(carrier)]

    supply = pd.DataFrame(index=n.snapshots)
    for c in n.iterate_components(n.branch_components):
        n_port = 4 if c.name=='Link' else 2
        for i in range(n_port):
            supply = pd.concat((supply,
                                (-1) * c.pnl["p" + str(i)].loc[:,
                                                               c.df.index[c.df["bus" + str(i)].isin(buses)]].groupby(c.df.carrier,
                                                                                                                     axis=1).sum()),
                               axis=1)

    for c in n.iterate_components(n.one_port_components):
        comps = c.df.index[c.df.bus.isin(buses)]
        supply = pd.concat((supply, ((c.pnl["p"].loc[:, comps]).multiply(
            c.df.loc[comps, "sign"])).groupby(c.df.carrier, axis=1).sum()), axis=1)

    supply = supply.groupby(rename_techs_tyndp, axis=1).sum()

    both = supply.columns[(supply < 0.).any() & (supply > 0.).any()]

    positive_supply = supply[both]
    negative_supply = supply[both]

    positive_supply[positive_supply < 0.] = 0.
    negative_supply[negative_supply > 0.] = 0.

    supply[both] = positive_supply

    suffix = " charging"

    negative_supply.columns = negative_supply.columns + suffix

    supply = pd.concat((supply, negative_supply), axis=1)

    # 14-21.2 for flaute
    # 19-26.1 for flaute

    start = "2013-01-01"
    stop = "2013-12-31"

    threshold = 10e3

    to_drop = supply.columns[(abs(supply) < threshold).all()]

    if len(to_drop) != 0:
        print("dropping", to_drop)
        supply.drop(columns=to_drop, inplace=True)

    supply.index.name = None

    supply = supply / 1e3

    supply.rename(columns={"electricity": "electric demand",
                           "heat": "heat demand"},
                  inplace=True)
    
    # print(supply['heat demand'])
    supply.columns = supply.columns.str.replace("residential ", "")
    supply.columns = supply.columns.str.replace("services ", "")
    supply.columns = supply.columns.str.replace("urban decentral ", "decentral ")

    preferred_order = pd.Index(["electric demand",
                                "transmission lines",
                                "hydroelectricity",
                                "hydro reservoir",
                                "run of river",
                                "pumped hydro storage",
                                "Fischer-Tropsch",
                                "CHP",
                                "onshore wind",
                                "offshore wind",
                                "solar PV",
                                "solar thermal",
                                "building retrofitting",
                                "ground heat pump",
                                "air heat pump",
                                "resistive heater",
                                "OCGT",
                                "gas boiler",
                                "gas",
                                "natural gas",
                                "methanation",
                                "hydrogen storage",
                                "battery storage",
                                "hot water storage"])

    new_columns = (preferred_order.intersection(supply.columns)
                   .append(supply.columns.difference(preferred_order)))

    supply =  supply.groupby(supply.columns, axis=1).sum()
    fig, ax = plt.subplots()
    fig.set_size_inches((8, 5))

    (supply.loc[start:stop, new_columns]
     .plot(ax=ax, kind="area", stacked=True, linewidth=0.,
           color=[tech_colors[i.replace(suffix, "")]
                  for i in new_columns]))

    handles, labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    new_handles = []
    new_labels = []

    for i, item in enumerate(labels):
        if "charging" not in item:
            new_handles.append(handles[i])
            new_labels.append(labels[i])

    ax.legend(new_handles, new_labels, ncol=3, loc="upper left", frameon=False)
    ax.set_xlim([start, stop])
    ax.set_ylim([-1300, 1900])
    ax.grid(True)
    ax.set_ylabel("Power [GW]")
    fig.tight_layout()
    ax.plot(-supply['heat demand'],color=tech_colors['heat demand'],lw=10,zorder=100)
    
def assign_location(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            if i == -1: continue
            names = ifind.index[ifind == i]
            c.df.loc[names, 'location'] = names.str[:i]
            
def assign_carriers(n):
    if "carrier" not in n.lines:
        n.lines["carrier"] = "AC"
        
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
        "CHP",
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
        # "H2 Electrolysis": "hydrogen storage",
        # "H2 Fuel Cell": "hydrogen storage",
        # "H2 pipeline": "hydrogen storage",
        "battery": "battery storage",
        # "CC": "CC"
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
    if "heat pump" in tech:
        return "heat pump"
    # elif tech in ["H2 Electrolysis", "methanation", "helmeth", "H2 liquefaction"]:
    #     return "power-to-gas"
    # elif tech == "H2":
    #     return "H2 storage"
    # elif tech in ["OCGT", "CHP", "gas boiler", "H2 Fuel Cell"]:
    #     return "gas-to-power/heat"
    if "solar" in tech:
        return "solar"
    elif tech == "Fischer-Tropsch":
        return "Fischer-Tropsch"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif "CC" in tech or "sequestration" in tech:
        return "CCS"
    else:
        return tech
    
n = pypsa.Network('../results/version-gaslimit3100-3H/postnetworks/elec_s370_37m_lv1.0__3H-T-H-B-I-solar+p3-dist1-cb25.7ex0_2035.nc')
plot_series(n, carrier="heat") #, name=suffix)
plt.savefig('../figures/Heating_seasonal_mic_1p5C_2035.pdf', bbox_inches="tight")