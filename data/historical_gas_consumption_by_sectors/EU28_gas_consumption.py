# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 20:49:04 2022

@author: au485969
"""

import os
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt

fs = 16 # fontsize
# plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs

################################ How is gas used? #########################################

#%% Industry
IDEES_files = os.listdir('IDEES_EU28')
xl_file_industry = pd.ExcelFile('IDEES_EU28/' + IDEES_files[0])
xl_file_industry_dict = {sheet_name: xl_file_industry.parse(sheet_name) 
          for sheet_name in xl_file_industry.sheet_names}
df_industry = xl_file_industry_dict['Ind_Summary']
df_industry.set_index('EU28: Industry Summary',inplace=True)
#%% Power sector
xl_file_power = pd.ExcelFile('IDEES_EU28/' + IDEES_files[1])
xl_file_power_dict = {sheet_name: xl_file_power.parse(sheet_name) 
          for sheet_name in xl_file_power.sheet_names}

df_power1 = xl_file_power_dict['Thermal_ElecOnly']
df_power1.set_index('Overview of electricity only thermal power plants',inplace=True)

df_power2 = xl_file_power_dict['Thermal_CHP']
df_power2.set_index('Overview of CHP thermal power plants',inplace=True)

df_power3 = xl_file_power_dict['DistHeat']
df_power3.set_index('Overview of district heating plants',inplace=True)
#%% Residential
xl_file_res = pd.ExcelFile('IDEES_EU28/' + IDEES_files[2])
xl_file_res_dict = {sheet_name: xl_file_res.parse(sheet_name) 
          for sheet_name in xl_file_res.sheet_names}
df_res = xl_file_res_dict['RES_hh_fec']
df_res.set_index('EU28 - Final energy consumption',inplace=True)
#%% Services + Agriculture (tertiary)
xl_file_ter = pd.ExcelFile('IDEES_EU28/' + IDEES_files[3])
xl_file_ter_dict = {sheet_name: xl_file_ter.parse(sheet_name) 
          for sheet_name in xl_file_ter.sheet_names}
df_ter = xl_file_ter_dict['SER_hh_fec']
df_ter.set_index('EU28 - Final energy consumption',inplace=True)
#%% Transport
xl_file_trans = pd.ExcelFile('IDEES_EU28/' + IDEES_files[4])
xl_file_trans_dict = {sheet_name: xl_file_trans.parse(sheet_name) 
          for sheet_name in xl_file_trans.sheet_names}
df_trans = xl_file_trans_dict['TrRoad_ene'] # Road transport 
df_trans.set_index('EU28 - Road transport / energy consumption',inplace=True)

#%% Gas consumptions
industry_energy = df_industry[2015].loc['Natural gas'][0]*11.63/1e3 # TWh
industry_feedstock = df_industry[2015].loc['Natural gas'][1]*11.63/1e3 # TWh
power_elec = df_power1['2015'].loc['Natural gas']*11.63/1e3 # TWh
power_chp = df_power2['2015'].loc['Natural gas']*11.63/1e3 # TWh
power_distheat = df_power3['2015'].loc['Natural gas'][1]*11.63/1e3 # TWh
residential = df_res[2015].loc[['Gas/Diesel oil incl. biofuels (GDO)','Gases incl. biogas']].sum()*11.63/1e3 # TWh
tertiary = df_ter[2015].loc[['Gas/Diesel oil incl. biofuels (GDO)','Gases incl. biogas']].sum()*11.63/1e3 # TWh 
transport = df_trans[2015].loc['Natural gas']*11.63/1e3 # TWh

fig,ax = plt.subplots()

ax.bar(0,industry_energy,label='Industry (Energy)')
ax.bar(0,industry_feedstock,bottom=industry_energy,label='Industry (Feedstock)')
ax.bar(0,power_elec,bottom=industry_energy+industry_feedstock,label='Power (Electricity)')
ax.bar(0,power_chp,bottom=industry_energy+industry_feedstock+power_elec,label='Power (CHP)')
ax.bar(0,power_distheat,bottom=industry_energy+industry_feedstock+power_elec+power_chp,label='District heating')
ax.bar(0,residential,bottom=industry_energy+industry_feedstock+power_elec+power_chp+power_distheat,label='Residential')
ax.bar(0,tertiary,bottom=industry_energy+industry_feedstock+power_elec+power_chp+power_distheat+residential,label='Services + Agriculture')
ax.bar(0,transport,bottom=industry_energy+industry_feedstock+power_elec+power_chp+power_distheat+residential+tertiary,label='Transport')

ax.set_xlim([-1,1])
ax.set_xticks([0])
ax.set_xticklabels(['EU-27 + GB'],fontsize=fs)

lgd = fig.legend(bbox_to_anchor=(1.42, 0.88),borderaxespad=0,prop={'size':fs})
ax.set_ylabel('TWh',fontsize=fs)
ax.set_title('Gas consumption by sector',fontsize=fs)
fig.savefig('Gas_cons_by_sector_refined.jpg', bbox_inches="tight",dpi=300)

#%% To csv
df_gas_sec = pd.DataFrame(index=['Industry (Energy)',
                                 'Industry (Feedstock)',
                                 'Power (Electricity)',
                                 'Power (CHP)',
                                 'Residential',
                                 '  Services',
                                 'Transport',
                                 'District heating plants'
                                 ],columns=['consumption [TWh]'])

df_gas_sec.loc['Industry (Energy)'] = industry_energy
df_gas_sec.loc['Industry (Feedstock)'] = industry_feedstock
df_gas_sec.loc['Power (Electricity)'] = power_elec
df_gas_sec.loc['Power (CHP)'] = power_chp
df_gas_sec.loc['Residential'] = residential
df_gas_sec.loc['  Services'] = tertiary
df_gas_sec.loc['Transport'] = transport
df_gas_sec.loc['District heating plants'] = power_distheat

df_gas_sec.to_csv('../data/gas_cons_by_sector.csv')
