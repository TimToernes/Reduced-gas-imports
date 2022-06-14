#script that
# Does this for one country
#1) reads in netcdf file DONE
    #Navigates to right folder. Goes back, then into the right directory
#2) identifies where the solar + wind data is
    #We know this is in generators
#2b) Plot the electrity demand of the 
#3) If necessary, sums up solar and wind for all countries
#4) Identifies day and/or week with lowest average solar and wind
#5) Ffff

##function

import pypsa
import pandas as pd
# import numpy as np
import yaml
import matplotlib.pyplot as plt
# import os
# from pathlib import Path
from datetime import datetime
import matplotlib.dates as mdates


#The purpose of this script is to make variations on worst_week.py script. One variation is when we use 2050 instead

# ---------------------------------
# ------- Settings ---------------- 
mav = 32 # number of neighbors for moving average
year = 2050
t = 2 # temperature increase sceanario. Can be either 2 or 1.5.
# dates = ["2013-01-13","2013-01-20"]
dates = ["2013-01-01","2013-12-31"]
# ------------------------------------
tdic = {'1.5':'cb25.7','2':'cb73.9'}

# gas constrained
n2 = pypsa.Network()
n2.import_from_netcdf("../results/PyPSA-Eur-Sec-0.6.0/3H_oil_lignite_nonextendable/postnetworks/elec_s370_37_lv1.0__3H-T-H-B-I-A-solar+p3-dist1-" + tdic[str(t)] + "ex0-gasconstrained_" + str(year) + ".nc")

n = pypsa.Network()
# gas unconstrained
n.import_from_netcdf("../results/PyPSA-Eur-Sec-0.6.0/3H_oil_lignite_nonextendable/postnetworks/elec_s370_37_lv1.0__3H-T-H-B-I-A-solar+p3-dist1-" + tdic[str(t)] + "ex0_" + str(year) + ".nc")

fs = 18
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs

#We are most interested in the indicies 117:173, when there is the least amount of solar

with open('./tech_colors.yaml') as file:
    tech_colors = yaml.safe_load(file)['tech_colors']

#THESE ARE THE THINGS GIVING ES0 0 ENERGY

#NO stores

def rename_techs(label):

    prefix_to_remove = [
        "residential ",
        "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral "
    ]

    rename_if_contains = [ #currently, if one of these is in a technology, it just substitutes
        "biomass CHP",
        "gas CHP",
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
        "B2B": "transmission lines",
        'uranium':'nuclear',
        'coal CC': 'coal',
        'ES0 0': 'electric demand'
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


#Generators: on and offshore wind, ror, solar
def make_plot_Spain():

    ESGenBus = n.generators.query('bus == "ES0 0"').index #The bus ES0 0 has generators, links that feed into it, links that feed out of it, storage units 
    ESgens = n.generators_t.p[ESGenBus].groupby(n.generators.carrier, axis = 1 ).sum()

    #ESgens['wind_pv'] = ESgens.sum(axis = 1) - ESgens['ror']


    #Links--where ES0 0 is bus 1, show with p0

    ESLinkBus1 = n.links.query('bus1 == "ES0 0"').index
    ESlinks1 = n.links_t.p0[ESLinkBus1].groupby(n.links.carrier,axis=1).sum()




    #THESE ARE THE THINGS THAT COULD EITHER GIVE OR TAKE ENERGY AWAY FROM ES0 0

    #Storage Units #I think I will split this up--positive values for one, negative values for another
    #PHS can be positive or negative, but hydro can only be positive
    ESStUnBus = n.storage_units.query('bus == "ES0 0"').index
    ESStUns = n.storage_units_t.p[ESStUnBus].groupby(n.storage_units.carrier, axis = 1).sum()
    #These are the positive things
    ESStUnsPlus = ESStUns.copy()
    ESStUnsPlus['PHS'] = ESStUnsPlus.apply(lambda row: row['PHS'] if row['PHS'] > 0 else 0, axis = 1)


    #This is the dataframe which has everything that we need for generation
    ESsupply = pd.concat([ESgens, ESlinks1, ESStUnsPlus], axis = 1)


    #These are the negative things
    ESStUnsNeg = ESStUns.copy()
    ESStUnsNeg = ESStUnsNeg.drop('hydro', axis = 1)
    ESStUnsNeg['PHS'] = ESStUnsNeg.apply(lambda row: row['PHS'] if row['PHS'] < 0 else 0, axis = 1)



    #THESE ARE THE THINGS TAKING AWAY ENERGY FROM ES0 0

    #links--where ES0 0 is bus 0, show with p0


    ESLinkBus = n.links.query('bus0 == "ES0 0"').index
    ESlinks = n.links_t.p0[ESLinkBus].groupby(n.links.carrier,axis=1).sum()


    #Demand of ES0 0

    ESDemand = pd.read_csv("../results/PyPSA-Eur-Sec-0.6.0/3H_oil_lignite_nonextendable/csvs/Spain_demand.csv")#This is the same thing as ES0 0 loads_t.p (plus ES3 0)
    ESDemand = ESDemand.drop(['name', "ES3 0"], axis = 1)

    EStotdem = pd.concat([ESStUnsNeg, ESlinks, ESDemand], axis = 1)


    #Here, I want to add 
    hours_in_2030 = pd.date_range('2030-01-01','2030-12-31 23:00', freq='h')
    hours_in_2030 = hours_in_2030[::3]

    ESsupply.index = hours_in_2030
    EStotdem.index = hours_in_2030



    techs = ['offwind-ac', 'offwind-dc', 'onwind', 'ror', 'solar', 'CCGT', 'DC',
        'H2 Fuel Cell', 'OCGT', 'battery discharger', 'coal', 'coal CC',
        'lignite', 'nuclear', 'oil', 'urban central gas CHP',
        'urban central gas CHP CC', 'urban central solid biomass CHP',
        'urban central solid biomass CHP CC', 'PHS', 'hydro']


    ESsupply = ESsupply.groupby([rename_techs(t)for t in techs], axis = 1).sum()
    ESsupply.rename(columns={'offshore wind':'offshore wind','offshore wind (AC)':'offshore wind','offshore wind (DC)':'offshore wind'},inplace=True)
    ESsupply.rename(columns = {'offshore wind': 'wind', 'onshore wind': 'wind', 'CCGT': 'gas', 'OCGT': 'gas'}, inplace = True)
    ESsupply= ESsupply.groupby(level=0, axis = 1).sum()

    coldrop = [col for col in ESsupply.columns if ESsupply[col].max() < 300]
    ESsupply = ESsupply.drop(['transmission lines'], axis = 1)
    ESsupply = ESsupply.drop(coldrop, axis = 1)




    techs2 = ['PHS', 'DC', 'H2 Electrolysis', 'battery charger',
        'electricity distribution grid', 'helmeth',
        'residential rural resistive heater',
        'residential urban decentral air heat pump',
        'residential urban decentral resistive heater',
        'services rural resistive heater',
        'services urban decentral air heat pump',
        'services urban decentral resistive heater',
        'urban central air heat pump', 'urban central resistive heater',
        'ES0 0']
    EStotdem = EStotdem.groupby([rename_techs(t)for t in techs2], axis = 1).sum()
    EStotdem= EStotdem.groupby(level=0).sum()

    coldrop2 = [col for col in EStotdem.columns if EStotdem[col].max() < 300]
    EStotdem = EStotdem.drop(coldrop2, axis = 1)

    ESsupply = ESsupply.iloc[:, ::-1]

    ESsupply = ESsupply/1000

    fig, ax = plt.subplots(figsize = (14, 8))
    

    #x = ESsupply.index.values.astype('datetime64[D]')
    #x = range(len(ESsupply['CHP']))
    x = ESsupply.index



    higheridx = 173
    loweridx = higheridx - 56
    #ax.grid(True)

    ax.stackplot(x[loweridx:higheridx], *[ts[loweridx:higheridx] for col, ts in ESsupply.iteritems()], 
    labels = [columnname for (columnname, columndata) in ESsupply.iteritems()],
    colors=[tech_colors[t] for t in list(ESsupply.columns.get_level_values(0))]
    )

    #This part of the plot shows where the energy is being used
    #ax.stackplot(x[117:163], *[-ts[117:163] for col, ts in EStotdem.iteritems()], labels = [columnname for (columnname, columndata) in EStotdem.iteritems()], colors=[tech_colors[t] for t in list(EStotdem.columns.get_level_values(0))])


    monthyearFmt = mdates.DateFormatter('%d %b')
    # locator = mdates.AutoDateLocator()
    # monthyearFmt = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_formatter(monthyearFmt)
    ax.legend(fontsize = fs, ncol = 3, bbox_to_anchor=(0.15, -0.2), loc = 'upper left', borderaxespad=0)
    ax.set_ylabel("Electricity Source (GW)")
    #ax.set_xlabel("Day in 2030")
    ax.set_title("Electricity source in Spain for the week of least solar generation", fontsize = fs)
    fig.subplots_adjust(wspace=0.05,hspace=0.05)




    plt.tight_layout()
    plt.savefig("../figures/worstweekSpain3.pdf")
    plt.show()
#make_plot_Spain()


def make_plot_any(bus):

    #ESGenBus = n.generators.query('bus =="' + bus+ '"').index
    ESGenBus = n.generators[n.generators['bus'].str.startswith("DK")].index
    ESgens = n.generators_t.p[ESGenBus].groupby(n.generators.carrier, axis = 1 ).sum()

    #ESgens['wind_pv'] = ESgens.sum(axis = 1) - ESgens['ror']


    #Links--where ES0 0 is bus 1, show with p0

    ESLinkBus1 = n.links[n.links['bus1'].str.startswith("DK")].index
    ESlinks1 = n.links_t.p0[ESLinkBus1].groupby(n.links.carrier,axis=1).sum()




    #THESE ARE THE THINGS THAT COULD EITHER GIVE OR TAKE ENERGY AWAY FROM ES0 0

    #Storage Units #I think I will split this up--positive values for one, negative values for another
    #PHS can be positive or negative, but hydro can only be positive

    #These are the positive things



    #This is the dataframe which has everything that we need for generation
    ESsupply = pd.concat([ESgens, ESlinks1], axis = 1)


    #These are the negative things




    #THESE ARE THE THINGS TAKING AWAY ENERGY FROM ES0 0

    #links--where ES0 0 is bus 0, show with p0


    ESLinkBus = n.links[n.links['bus0'].str.startswith("DK")].index
    ESlinks = n.links_t.p0[ESLinkBus].groupby(n.links.carrier,axis=1).sum()


    #Demand of ES0 0

    ESDemand = n.loads_t.p[bus]

    EStotdem = pd.concat([ESlinks, ESDemand], axis = 1)




    #Here, I want to add 
    hours_in_2030 = pd.date_range('2030-01-01','2030-12-31 23:00', freq='h')
    hours_in_2030 = hours_in_2030[::3]

    ESsupply.index = hours_in_2030
    EStotdem.index = hours_in_2030


    print(ESsupply.columns)
    techs = ESsupply.columns


    ESsupply = ESsupply.groupby([rename_techs(t)for t in techs], axis = 1).sum()
   
    ESsupply.rename(columns={'offshore wind':'offshore wind','offshore wind (AC)':'offshore wind','offshore wind (DC)':'offshore wind'},inplace=True)
    ESsupply.rename(columns = {'offshore wind': 'wind', 'onshore wind': 'wind', 'CCGT': 'gas', 'OCGT': 'gas'}, inplace = True)
    ESsupply= ESsupply.groupby(level=0, axis = 1).sum()

    # coldrop = [col for col in ESsupply.columns if ESsupply[col].max() < 300]
    # ESsupply = ESsupply.drop(['transmission lines'], axis = 1)
    # ESsupply = ESsupply.drop(coldrop, axis = 1)




    techs2 = EStotdem.columns
    EStotdem = EStotdem.groupby([rename_techs(t)for t in techs2], axis = 1).sum()
    EStotdem= EStotdem.groupby(level=0).sum()

    coldrop2 = [col for col in EStotdem.columns if EStotdem[col].max() < 300]
    EStotdem = EStotdem.drop(coldrop2, axis = 1)

    ESsupply = ESsupply.iloc[:, ::-1]


    fig, ax = plt.subplots(figsize = (14, 8))
    

    #x = ESsupply.index.values.astype('datetime64[D]')
    #x = range(len(ESsupply['CHP']))
    x = ESsupply.index



    higheridx = 2820
    loweridx = higheridx - 56
    #ax.grid(True)

    ax.stackplot(x[loweridx:higheridx], *[ts[loweridx:higheridx] for col, ts in ESsupply.iteritems()], 
    labels = [columnname for (columnname, columndata) in ESsupply.iteritems()],
    colors=[tech_colors[t] for t in list(ESsupply.columns.get_level_values(0))]
    )

    #This part of the plot shows where the energy is being used
    #ax.stackplot(x[117:163], *[-ts[117:163] for col, ts in EStotdem.iteritems()], labels = [columnname for (columnname, columndata) in EStotdem.iteritems()], colors=[tech_colors[t] for t in list(EStotdem.columns.get_level_values(0))])


    monthyearFmt = mdates.DateFormatter('%d %b')
    # locator = mdates.AutoDateLocator()
    # monthyearFmt = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_formatter(monthyearFmt)
    ax.legend(fontsize = fs, ncol = 3, bbox_to_anchor=(0.1, -0.2), loc = 'upper left', borderaxespad=0)
    ax.set_ylabel("Production MWh)")
    #ax.set_xlabel("Day in 2030")
    ax.set_title("Electricity produced in Denmark for the lowest solar plus wind week", fontsize = fs)
    fig.subplots_adjust(wspace=0.05,hspace=0.05)




    plt.tight_layout()
    #plt.savefig("../figures/worstweekSpainwind.pdf")
    plt.show()
#make_plot_any("DK0 0")



def plot_series2(network, carrier="heat", name="test"):

    n = network.copy()
    assign_location(n)
    assign_carriers(n)

    buses = n.buses.index[n.buses.carrier.str.contains(carrier)]

    supply = pd.DataFrame(index=n.snapshots)
    for c in n.iterate_components(n.branch_components):
        # n_port = 4 if c.name=='Link' else 2
        for i in [1]: #range(n_port):
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
    stop = "2013-12-15"

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

    preferred_order = pd.Index([#"electric demand",
                                #"transmission lines",
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
    #ax.plot(-supply['heat demand'],color=tech_colors['heat demand'],lw=10,zorder=100)
    plt.show()
  


def plot_series3(network, carrier="heat", name="test"):
    plt.rcdefaults()
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

    start = "2013-02-19"
    stop = "2013-02-26"

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
    supply.columns = supply.columns.str.replace("residential ", "")
    supply.columns = supply.columns.str.replace("services ", "")
    supply.columns = supply.columns.str.replace("urban decentral ", "decentral ")

    preferred_order = pd.Index(["electric demand",
                                "transmission lines",
                                "hydroelectricity",
                                "hydro reservoir",
                                "run of river",
                                "pumped hydro storage",
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
    #ax.set_xlim([start, stop])
    #ax.set_ylim([-1300, 1900])
    ax.grid(True)
    ax.set_ylabel("Power [GW]")
    fig.tight_layout()
    plt.show()


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
    elif tech == "gas boiler":
        return "gas"
    elif "solar" in tech:
        return "solar"
    # elif tech == "Fischer-Tropsch":
    #     return "power-to-liquid"
    elif "offshore wind" in tech:
        return "offshore wind"
    # elif "CC" in tech or "sequestration" in tech:
    #     return "CCS"
    else:
        return tech


#modified so it is over the entire year
def plot_series(network, ax, dates,carrier="heat", mav=1,name="test"):

    n = network.copy()#First, copy network
    assign_location(n)
    assign_carriers(n)#in case there is no carrier, put carrier

    buses = n.buses.index[n.buses.carrier.str.contains(carrier)]#list of all buses--but what do I need to do for electricity

    supply = pd.DataFrame(index=n.snapshots)
    for c in n.iterate_components(n.branch_components):#for the lines, links, transformers
        n_port = 4 if c.name=='Link' else 2 # what is n_por
        # if c.name == 'Link':
        for i in range(n_port):
            supply = pd.concat((supply,
                                (-1) * c.pnl["p" + str(i)].loc[:,
                                                               c.df.index[c.df["bus" + str(i)].isin(buses)]].groupby(c.df.carrier,
                                                                                                                     axis=1).sum()),#what is pnl
                               axis=1)
                                                                                                                     
        # if c.name == 'Link':
        #     break                                                                                        
                                                                                                             
    for c in n.iterate_components(n.one_port_components):
        comps = c.df.index[c.df.bus.isin(buses)]
        supply = pd.concat((supply, ((c.pnl["p"].loc[:, comps]).multiply(
            c.df.loc[comps, "sign"])).groupby(c.df.carrier, axis=1).sum()), axis=1)

    # watersupply = supply.loc[:, supply.columns.str.contains("CHP")]
    # print(watersupply)
    # print(watersupply.sum())
    # print(watersupply.sum().sum())
    supply = supply.groupby(rename_techs_tyndp, axis=1).sum()#renames techs

    # watersupply = supply.loc[:, supply.columns.str.contains("CHP")]
    # print(watersupply)
    # print(watersupply.sum())

    both = supply.columns[(supply < 0.).any() & (supply > 0.).any()]

    positive_supply = supply[both]
    negative_supply = supply[both]

    positive_supply[positive_supply < 0.] = 0.
    negative_supply[negative_supply > 0.] = 0.

    print(negative_supply)
    supply[both] = positive_supply

    # suffix = " charging"

    # negative_supply.columns = negative_supply.columns + suffix
    # supply = pd.concat((supply, negative_supply), axis=1) #I commented this out because I want to get rid of 


    # 14-21.2 for flaute
    # 19-26.1 for flaute

    start = "2013-01-13" #dates[0]

    start = datetime.strptime(start, "%Y-%m-%d")
    stop = "2013-01-20" # dates[1]
    stop = datetime.strptime(stop, "%Y-%m-%d")

    threshold = 10e3

    supply.rename(columns={"electricity": "electric demand",
                           "heat": "heat demand"},
                  inplace=True)


    # heatdem = pd.DataFrame()

    #heatdem['heat demand'] = supply['heat demand']

    to_drop = supply.columns[(supply < threshold).all()]



    if len(to_drop) != 0:
        print("dropping", to_drop)
        supply.drop(columns=to_drop, inplace=True)

    #supply = pd.concat((supply, heatdem['heat demand']), axis = 1)
        

    supply.index.name = None



    supply = supply / 1e3



    supply.columns = supply.columns.str.replace("residential ", "")#gets rid of some prefixes
    supply.columns = supply.columns.str.replace("services ", "")
    supply.columns = supply.columns.str.replace("urban decentral ", "decentral ")

    preferred_order = pd.Index([#"electric demand",
                                "onshore wind",
                                "offshore wind",
                                "solar",
                                
                                #"transmission lines",
                                "hydroelectricity",#uses
                                "hydro reservoir",
                                "run of river",
                                "pumped hydro storage",
                                "Fischer-Tropsch",
                                "gas boiler",
                                "gas,"
                                "natural gas",
                                
                                "resistive heater",
                                
                                
                                "biomass CHP",
                                "gas CHP",
                                "CHP",
                                
                                "solar PV",#uses
                                "solar thermal",
                                "building retrofitting",
                                "ground heat pump",
                                "air heat pump",
                                
                                "OCGT",
                                
                                # "gas",
                                
                                "methanation",
                                "hydrogen storage",
                                "battery storage",#uses
                                ])
                                #"hot water storage"])

    # m = supply.lt(0).all()
    # supply = supply.loc[:, ~m]

    # new_columns = (preferred_order.intersection(supply.columns)#what is the purpose of this? To only use the columns we want?
    #                .append(supply.columns.difference(preferred_order)))


    #Removing oil boiler between gas and gas CHP --adam 23/5
    if 'gas' in n.name:
        new_columns = ['gas', 'gas CHP', 'heat pump','resistive heater','biomass CHP', 'hot water storage']
    else:
        new_columns = ['gas', 'gas CHP', 'heat pump','resistive heater', 'hot water storage']
        
    supply =  supply.groupby(supply.columns, axis=1).sum()

    #We pass an ax instead

    # fig, ax = plt.subplots()
    # fig.set_size_inches((8, 5))
 
    #This is for the year
    supply.set_index(pd.date_range('2013-01-01','2014-01-01',freq='3H')[0:-1],inplace=True)

    #This is for the 'worst week'
    # supply.set_index(pd.date_range('2013-01-13','2013-01-20',freq='3H')[0:-1],inplace=True)



    #ax.plot(-supply['heat demand'], color = tech_colors['heat demand'], lw = 2)


    # cols = ['resistive heater', 'gas CHP', 'gas', 'heat pump', 'hot water storage','oil boiler']
    # cols = ['resistive heater', 'gas CHP', 'gas', 'heat pump','oil boiler']
    suffix = ' charging'

    #We use this for yearly
    # supply[new_columns].loc[start:stop].rolling(mav).mean().plot(ax=ax, kind="area", stacked=True, linewidth=0.,
    #     color=[tech_colors[i.replace(suffix, "")]
    #             for i in new_columns])

    #We use this for weekly
    supply[new_columns].loc[start:stop].plot(ax=ax, kind="area", stacked=True, linewidth=0.,
        color=[tech_colors[i.replace(suffix, "")]
                for i in new_columns])



    



    # supply['heat demand'].loc[start:stop].plot(ax = ax, kind = 'line', linewidth = 10, color = tech_colors['heat demand'])

    # print(heatdem)
    # print(supply)

    # (ax.plot(-heatdem['heat demand'].loc[start:stop],color=tech_colors['heat demand'],lw=3,zorder=100))
    
    handles, labels = ax.get_legend_handles_labels()
    #print(labels)

    # handles = [handles[0]] + [handles[1]] + [handles[3]]#for some reason heat demand was still in here so I had to avoid it
    # labels = [labels[0]] + [labels[1]] + [labels[3]]
    #print([item for item in labels][0] + [item for item in handles][2])


    handles.reverse()
    labels.reverse()

    new_handles = []
    new_labels = []

    for i, item in enumerate(labels):
        if "charging" not in item:
            new_handles.append(handles[i])
            new_labels.append(labels[i])


    #ax.set_xticklabels([])
    #ax.set_x
    monthyearFmt = mdates.DateFormatter('%m')

    
    ax.xaxis.set_major_formatter(monthyearFmt)
    ax.xaxis.set_tick_params(rotation = 0)

    
    ax.legend(new_handles, new_labels, ncol=3, loc="upper left", frameon=False)
    ax.legend().set_visible(False)
    ax.set_xlim([start, stop])
    ax.set_ylim([0, 1000])
    ax.grid(True)
    ax.set_ylabel("Power [GW]")
    #ax.set_xlabel("Day in January")

    if carrier == 'heat':
        ax.set_title("Heat supply (aggregate) during highest average natural gas use")
    else:
        ax.set_title("Electricity supply (aggregate) during highest average natural gas use")

    fig.tight_layout()
    #print(supply)



    #plt.plot(supply.index.to_pydatetime(), supply['heat demand'])
    #supply['heat demand'].plot(color=tech_colors['heat demand'],lw=10,zorder=100, x_compat = True)
    #print(supply)

    #supply['gasavg'] = supply['OCGT'].rolling(window = 56).mean()

    #ax.plot
    
    # fig.savefig('../figures/Aggregate_heat2Cwoconstraint.pdf')

    # plt.show()
# fig, ax = plt.subplots()
# plot_series(n2, ax, carrier = 'heat')

def plot_series_year(network, ax, carrier="heat", name="test"):

    n = network.copy()#First, copy network
    assign_location(n)
    assign_carriers(n)#in case there is no carrier, put carrier

    buses = n.buses.index[n.buses.carrier.str.contains(carrier)]#list of all buses--but what do I need to do for electricity

    supply = pd.DataFrame(index=n.snapshots)
    for c in n.iterate_components(n.branch_components):#for the lines, links, transformers
        n_port = 4 if c.name=='Link' else 2# what is n_port
        for i in range(n_port):
            supply = pd.concat((supply,
                                (-1) * c.pnl["p" + str(i)].loc[:,
                                                               c.df.index[c.df["bus" + str(i)].isin(buses)]].groupby(c.df.carrier,
                                                                                                                     axis=1).sum()),#what is pnl
                               axis=1)

    for c in n.iterate_components(n.one_port_components):
        comps = c.df.index[c.df.bus.isin(buses)]
        supply = pd.concat((supply, ((c.pnl["p"].loc[:, comps]).multiply(
            c.df.loc[comps, "sign"])).groupby(c.df.carrier, axis=1).sum()), axis=1)

    supply = supply.groupby(rename_techs_tyndp, axis=1).sum()#renames techs

    both = supply.columns[(supply < 0.).any() & (supply > 0.).any()]

    positive_supply = supply[both]
    negative_supply = supply[both]

    positive_supply[positive_supply < 0.] = 0.
    negative_supply[negative_supply > 0.] = 0.

    supply[both] = positive_supply

    suffix = " charging"

    negative_supply.columns = negative_supply.columns + suffix
    supply = pd.concat((supply, negative_supply), axis=1) #I commented this out because I want to get rid of 

    # 14-21.2 for flaute
    # 19-26.1 for flaute

    start = "2013-01-01"

    start = datetime.strptime(start, "%Y-%m-%d")
    stop = "2013-12-31"
    stop = datetime.strptime(stop, "%Y-%m-%d")

    threshold = 10e3

    supply.rename(columns={"electricity": "electric demand",
                           "heat": "heat demand"},
                  inplace=True)


    #heatdem = pd.DataFrame()

    #heatdem['heat demand'] = supply['heat demand']

    to_drop = supply.columns[(supply < threshold).all()]



    if len(to_drop) != 0:
        print("dropping", to_drop)
        supply.drop(columns=to_drop, inplace=True)

    # supply = pd.concat((supply, heatdem['heat demand']), axis = 1)
        

    supply.index.name = None



    supply = supply / 1e3



    supply.columns = supply.columns.str.replace("residential ", "")#gets rid of some prefixes
    supply.columns = supply.columns.str.replace("services ", "")
    supply.columns = supply.columns.str.replace("urban decentral ", "decentral ")

    preferred_order = pd.Index(["electric demand",
                                "onshore wind",
                                "offshore wind",
                                "solar",
                                
                                "transmission lines",
                                "hydroelectricity",#uses
                                "hydro reservoir",
                                "run of river",
                                "pumped hydro storage",
                                "Fischer-Tropsch",
                                "CHP",
                                
                                "solar PV",#uses
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
                                "battery storage",#uses
                                
                                
                                "hot water storage"])

    # m = supply.lt(0).all()
    # supply = supply.loc[:, ~m]

    new_columns = (preferred_order.intersection(supply.columns)#what is the purpose of this? To only use the columns we want?
                   .append(supply.columns.difference(preferred_order)))


    supply =  supply.groupby(supply.columns, axis=1).sum()


    #We pass an ax instead

    # fig, ax = plt.subplots()
    # fig.set_size_inches((8, 5))
 

    supply.set_index(pd.date_range('2013-01-01','2014-01-01',freq='3H')[0:-1],inplace=True)



    #ax.plot(-supply['heat demand'], color = tech_colors['heat demand'], lw = 2)


    ax2 = supply[new_columns].loc[start:stop].plot(ax=ax, kind="area", stacked=True, linewidth=0.,
        color=[tech_colors[i.replace(suffix, "")]
                for i in new_columns])


    



    # supply['heat demand'].loc[start:stop].plot(ax = ax, kind = 'line', linewidth = 10, color = tech_colors['heat demand'])

    # print(heatdem)
    # print(supply)

    # (ax.plot(-heatdem['heat demand'].loc[start:stop],color=tech_colors['heat demand'],lw=3,zorder=100))
    
    handles, labels = ax.get_legend_handles_labels()
    #print(labels)

    # handles = [handles[0]] + [handles[1]] + [handles[3]]#for some reason heat demand was still in here so I had to avoid it
    # labels = [labels[0]] + [labels[1]] + [labels[3]]
    #print([item for item in labels][0] + [item for item in handles][2])


    handles.reverse()
    labels.reverse()

    new_handles = []
    new_labels = []

    for i, item in enumerate(labels):
        if "charging" not in item:
            new_handles.append(handles[i])
            new_labels.append(labels[i])


    #ax.set_xticklabels([])
    #ax.set_x
    monthyearFmt = mdates.DateFormatter('%d')

    
    ax.xaxis.set_major_formatter(monthyearFmt)
    ax.xaxis.set_tick_params(rotation = 0)

    
    ax.legend(new_handles, new_labels, ncol=3, loc="upper left", frameon=False)
    ax.set_xlim([start, stop])
    ax.set_ylim([0, 2000])
    ax.grid(True)
    ax.set_ylabel("Power [GW]")
    #ax.set_xlabel("Day in January")

    if carrier == 'heat':
        ax.set_title("Heat supply (aggregate) during highest average natural gas use")
    else:
        ax.set_title("Electricity supply (aggregate) during highest average natural gas use")

    fig.tight_layout()
    #print(supply)



def plot_series_test(network, ax, carrier="heat", name="test"):

    n = network.copy()#First, copy network
    assign_location(n)
    assign_carriers(n)#in case there is no carrier, put carrier

    buses = n.buses.index[n.buses.carrier.str.contains(carrier)]#list of all buses--but what do I need to do for electricity

    supply = pd.DataFrame(index=n.snapshots)
    for c in n.iterate_components(n.branch_components):#for the lines, links, transformers
        n_port = 4 if c.name=='Link' else 2# what is n_port
        for i in range(n_port):
            supply = pd.concat((supply,
                                (-1) * c.pnl["p" + str(i)].loc[:,
                                                               c.df.index[c.df["bus" + str(i)].isin(buses)]].groupby(c.df.carrier,
                                                                                                                     axis=1).sum()),#what is pnl
                               axis=1)

    for c in n.iterate_components(n.one_port_components):
        comps = c.df.index[c.df.bus.isin(buses)]
        supply = pd.concat((supply, ((c.pnl["p"].loc[:, comps]).multiply(
            c.df.loc[comps, "sign"])).groupby(c.df.carrier, axis=1).sum()), axis=1)

    supply = supply.groupby(rename_techs_tyndp, axis=1).sum()#renames techs

    both = supply.columns[(supply < 0.).any() & (supply > 0.).any()]

    positive_supply = supply[both]
    negative_supply = supply[both]

    positive_supply[positive_supply < 0.] = 0.
    negative_supply[negative_supply > 0.] = 0.

    supply[both] = positive_supply

    suffix = " charging"

    negative_supply.columns = negative_supply.columns + suffix
    supply = pd.concat((supply, negative_supply), axis=1) #I commented this out because I want to get rid of 

    # 14-21.2 for flaute
    # 19-26.1 for flaute

    start = "2013-01-13"

    start = datetime.strptime(start, "%Y-%m-%d")
    stop = "2013-01-20"
    stop = datetime.strptime(stop, "%Y-%m-%d")

    threshold = 10e3

    supply.rename(columns={"electricity": "electric demand",
                           "heat": "heat demand"},
                  inplace=True)


    heatdem = pd.DataFrame()

    #heatdem['heat demand'] = supply['heat demand']

    to_drop = supply.columns[(supply < threshold).all()]



    if len(to_drop) != 0:
        print("dropping", to_drop)
        supply.drop(columns=to_drop, inplace=True)

    #supply = pd.concat((supply, heatdem['heat demand']), axis = 1)
        

    supply.index.name = None



    supply = supply / 1e3



    supply.columns = supply.columns.str.replace("residential ", "")#gets rid of some prefixes
    supply.columns = supply.columns.str.replace("services ", "")
    supply.columns = supply.columns.str.replace("urban decentral ", "decentral ")

    preferred_order = pd.Index(["electric demand",
                                "onshore wind",
                                "offshore wind",
                                "solar",
                                
                                "transmission lines",
                                "hydroelectricity",#uses
                                "hydro reservoir",
                                "run of river",
                                "pumped hydro storage",
                                "Fischer-Tropsch",
                                "CHP",
                                
                                "solar PV",#uses
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
                                "battery storage",#uses
                                
                                
                                "hot water storage"])

    # m = supply.lt(0).all()
    # supply = supply.loc[:, ~m]

    new_columns = (preferred_order.intersection(supply.columns)#what is the purpose of this? To only use the columns we want?
                   .append(supply.columns.difference(preferred_order)))


    supply =  supply.groupby(supply.columns, axis=1).sum()



 

    supply.set_index(pd.date_range('2013-01-01','2014-01-01',freq='3H')[0:-1],inplace=True)



    #ax.plot(-supply['heat demand'], color = tech_colors['heat demand'])


    ax2 = supply[new_columns].loc[start:stop].plot(ax=ax, kind="area", stacked=True, linewidth=0.,
        color=[tech_colors[i.replace(suffix, "")]
                for i in new_columns])


    



    # supply['heat demand'].loc[start:stop].plot(ax = ax, kind = 'line', linewidth = 10, color = tech_colors['heat demand'])

    # print(heatdem)
    # print(supply)

    # (ax.plot(-heatdem['heat demand'].loc[start:stop],color=tech_colors['heat demand'],lw=3,zorder=100))
    
    handles, labels = ax.get_legend_handles_labels()
    #print(labels)

    # handles = [handles[0]] + [handles[1]] + [handles[3]]#for some reason heat demand was still in here so I had to avoid it
    # labels = [labels[0]] + [labels[1]] + [labels[3]]
    #print([item for item in labels][0] + [item for item in handles][2])


    handles.reverse()
    labels.reverse()

    new_handles = []
    new_labels = []

    for i, item in enumerate(labels):
        if "charging" not in item:
            new_handles.append(handles[i])
            new_labels.append(labels[i])


    #ax.set_xticklabels([])
    #ax.set_x
    monthyearFmt = mdates.DateFormatter('%d')

    ax.xaxis.set_major_formatter(monthyearFmt)
    ax.xaxis.set_tick_params(rotation = 0)

    
    ax.legend(new_handles, new_labels, ncol=3, loc="upper left", frameon=False)
    ax.set_xlim([start, stop])
    ax.set_ylim([0, 2000])
    ax.grid(True)
    ax.set_ylabel("Power [GW]")
    ax.set_xlabel("Day in January")

    if carrier == 'heat':
        ax.set_title("Heat supply (aggregate) during highest average natural gas use")
    else:
        ax.set_title("Electricity supply (aggregate) during highest average natural gas use")

    fig.tight_layout()
    #print(supply)



    #plt.plot(supply.index.to_pydatetime(), supply['heat demand'])
    #supply['heat demand'].plot(color=tech_colors['heat demand'],lw=10,zorder=100, x_compat = True)
    #print(supply)

    #supply['gasavg'] = supply['OCGT'].rolling(window = 56).mean()

    #ax.plot
    
    fig.savefig('../figures/Aggregate_heat2Cwoconstraint.pdf')

    plt.show()


def plot_series_elec(network, ax, dates, carrier="AC", mav=1, name="test"):

    n = network.copy()#First, copy network
    assign_location(n)
    assign_carriers(n)#in case there is no carrier, put carrier

    buses = n.buses.index[n.buses.carrier.str.contains(carrier)]#list of all buses--but what do I need to do for electricity

    supply = pd.DataFrame(index=n.snapshots)
    for c in n.iterate_components(n.branch_components):#for the lines, links, transformers
        # n_port = 4 if c.name=='Link' else 2# what is n_port
        # for i in range(n_port):#can either be 4 or 2
        #     supply = pd.concat((supply,#concats from previous #looking at p0, p1, p2, etc
        #                         (-1) * c.pnl["p" + str(i)].loc[:,#looking at columns
        #                                                        c.df.index[c.df["bus" + str(i)].isin(buses)]].groupby(c.df.carrier,#looking at bus0, bus1, etc
        #                                                                                                              axis=1).sum()),#what is pnl
        #                        axis=1)

        for i in [1]: #range(2):
            supply = pd.concat((supply,
                                (-1) * c.pnl["p" + str(i)].loc[:,
                                                               c.df.index[c.df["bus" + str(i)].isin(buses)]].groupby(c.df.carrier,
                                                                                                                     axis=1).sum()),
                               axis=1)

    for c in n.iterate_components(n.one_port_components):
        comps = c.df.index[c.df.bus.isin(buses)]
        supply = pd.concat((supply, ((c.pnl["p"].loc[:, comps]).multiply(
            c.df.loc[comps, "sign"])).groupby(c.df.carrier, axis=1).sum()), axis=1)


    # loadselec = n.loads.index[n.loads.carrier.str.contains('electricity')]
    # demand = n.loads_t.p[loadselec]
    # demand= demand.sum(axis = 1)
    
    # truedemand = pd.DataFrame()
    # truedemand['electric demand'] = -demand

    # print(truedemand)
    # chpsupply = supply.loc[:, supply.columns.str.contains("CHP")]
    # print(chpsupply)
    # print(chpsupply.sum())

    # watersupply = supply.loc[:, supply.columns.str.contains("water")]
    # print(watersupply)
    # print(watersupply.sum())

    supply.drop(columns=['AC','DC'],inplace=True) # Europe-aggregate (one node = "no transmission")

    supply = supply.groupby(rename_techs_tyndp, axis=1).sum()#renames techs
    both = supply.columns[(supply < 0.).any() & (supply > 0.).any()]

    positive_supply = supply[both]
    negative_supply = supply[both]

    positive_supply[positive_supply < 0.] = 0.
    negative_supply[negative_supply > 0.] = 0.

    supply[both] = positive_supply

    suffix = " charging"

    negative_supply.columns = negative_supply.columns + suffix
    supply = pd.concat((supply, negative_supply), axis=1)

    start = "2013-01-13" #dates[0]
    start = datetime.strptime(start, "%Y-%m-%d")
    stop =  "2013-01-20"#dates[1]
    stop = datetime.strptime(stop, "%Y-%m-%d")

    threshold = 10e3

    to_drop = supply.columns[(supply < threshold).all()]

    if len(to_drop) != 0:
        print("dropping", to_drop)
        supply.drop(columns=to_drop, inplace=True)

    supply.rename(columns={"electricity": "electric demand", 'electricity distribution grid': 'electricity distribution grid charging',
                           "heat": "heat demand"},
                  inplace=True)

    # print(supply)
    # elecdem = pd.DataFrame()

    # print(supply)
    # print(supply.columns)
    # print(supply[supply.columns[(supply < 0.).any()]])
    # print(supply.[supply.columns.sum() < 0])

    # elecdem['electric demand'] = supply['electricity distribution grid charging']
    #print(supply)

    # supply = pd.concat((supply, truedemand['electric demand']), axis = 1)
    
    # print(supply)
    supply.index.name = None

    supply = supply / 1e3

    supply.columns = supply.columns.str.replace("residential ", "")#gets rid of some prefixes
    supply.columns = supply.columns.str.replace("services ", "")
    supply.columns = supply.columns.str.replace("urban decentral ", "decentral ")

    preferred_order = pd.Index([#"electric demand",
                                "onshore wind",
                                "offshore wind",
                                "solar",
                                
                                "transmission lines",
                                "hydroelectricity",#uses
                                "hydro reservoir",
                                "run of river",
                                "pumped hydro storage",
                                "Fischer-Tropsch",
                                "biomass CHP",
                                "gas CHP",    
                                "CHP",
                            
                                "solar PV",#uses
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
                                "battery storage",#uses
                                "hot water storage"])

    # m = supply.lt(0).all()
    # supply = supply.loc[:, ~m]

    new_columns = (preferred_order.intersection(supply.columns)#what is the purpose of this? To only use the columns we want?
                   .append(supply.columns.difference(preferred_order)))


    supply =  supply.groupby(supply.columns, axis=1).sum()


    #We pass an ax instead
    # fig, ax = plt.subplots()
    # fig.set_size_inches((8, 5))
 
    # This is the yearly index
    supply.set_index(pd.date_range('2013-01-01','2014-01-01',freq='3H')[0:-1],inplace=True)

    #This is the 'worst week'
    # supply.set_index(pd.date_range('2013-01-13','2013-01-20',freq='3H')[0:-1],inplace=True)



    #ax.plot(-supply['electric demand'], color = tech_colors['electric demand'], lw = 2)


    #for year
    # ax2 = supply[new_columns].loc[start:stop].rolling(mav).mean().plot(ax=ax, kind="area", stacked=True, linewidth=0.,
    #     color=[tech_colors[i.replace(suffix, "")]
    #             for i in new_columns])


    # for week
    ax2 = supply[new_columns].loc[start:stop].plot(ax=ax, kind="area", stacked=True, linewidth=0.,
        color=[tech_colors[i.replace(suffix, "")]
                for i in new_columns])



    # supply['heat demand'].loc[start:stop].plot(ax = ax, kind = 'line', linewidth = 10, color = tech_colors['heat demand'])

    # print(heatdem)
    # print(supply)

    # (ax.plot(-heatdem['heat demand'].loc[start:stop],color=tech_colors['heat demand'],lw=3,zorder=100))
    
    handles, labels = ax.get_legend_handles_labels()
    #print(labels)

    # handles = [handles[0]] + [handles[1]] + [handles[3]]#for some reason heat demand was still in here so I had to avoid it
    # labels = [labels[0]] + [labels[1]] + [labels[3]]
    #print([item for item in labels][0] + [item for item in handles][2])


    handles.reverse()
    labels.reverse()

    new_handles = []
    new_labels = []

    for i, item in enumerate(labels):
        if "charging" not in item:
            new_handles.append(handles[i])
            new_labels.append(labels[i])


    #ax.set_xticklabels([])
    #ax.set_x
    monthyearFmt = mdates.DateFormatter('%d')

    ax.xaxis.set_major_formatter(monthyearFmt)
    ax.xaxis.set_tick_params(rotation = 0)

    ax.legend(new_handles, new_labels, ncol=3, loc="upper left", frameon=False)
    ax.legend().set_visible(False)
    ax.set_xlim([start, stop])
    ax.set_ylim([0, 1500])
    ax.grid(True)
    ax.set_ylabel("Power [GW]")
    ax.set_xlabel("Day in January")

    if carrier == 'heat':
        ax.set_title("Heat supply (aggregate) during highest average natural gas use")
    else:
        ax.set_title("Electricity supply (aggregate) during highest average natural gas use")

    fig.tight_layout()
    #print(supply)



    #plt.plot(supply.index.to_pydatetime(), supply['heat demand'])
    #supply['heat demand'].plot(color=tech_colors['heat demand'],lw=10,zorder=100, x_compat = True)
    #print(supply)

    #supply['gasavg'] = supply['OCGT'].rolling(window = 56).mean()

    #ax.plot
    
    #fig.savefig('../figures/Aggregate_heat2Cwoconstraint.pdf')

    # plt.show()



def plot_series_elec_year(network, ax, carrier="heat", name="test"):

    n = network.copy()#First, copy network
    assign_location(n)
    assign_carriers(n)#in case there is no carrier, put carrier


    buses = n.buses.index[n.buses.carrier.str.contains(carrier)]#list of all buses--but what do I need to do for electricity

    supply = pd.DataFrame(index=n.snapshots)
    for c in n.iterate_components(n.branch_components):#for the lines, links, transformers
        n_port = 4 if c.name=='Link' else 2# what is n_port
        for i in range(n_port):
            supply = pd.concat((supply,
                                (-1) * c.pnl["p" + str(i)].loc[:,
                                                               c.df.index[c.df["bus" + str(i)].isin(buses)]].groupby(c.df.carrier,
                                                                                                                     axis=1).sum()),#what is pnl
                               axis=1)

    for c in n.iterate_components(n.one_port_components):
        comps = c.df.index[c.df.bus.isin(buses)]
        supply = pd.concat((supply, ((c.pnl["p"].loc[:, comps]).multiply(
            c.df.loc[comps, "sign"])).groupby(c.df.carrier, axis=1).sum()), axis=1)

    supply = supply.groupby(rename_techs_tyndp, axis=1).sum()#renames techs

    print(supply.columns)
    both = supply.columns[(supply < 0.).any() & (supply > 0.).any()]

    positive_supply = supply[both]
    negative_supply = supply[both]

    positive_supply[positive_supply < 0.] = 0.
    negative_supply[negative_supply > 0.] = 0.

    supply[both] = positive_supply

    suffix = " charging"

    negative_supply.columns = negative_supply.columns + suffix
    supply = pd.concat((supply, negative_supply), axis=1)



    start = "2013-01-01"
    start = datetime.strptime(start, "%Y-%m-%d")
    stop = "2014-01-01"
    stop = datetime.strptime(stop, "%Y-%m-%d")

    threshold = 10e3

    supply.rename(columns={"electricity": "electric demand", 'electricity distribution grid': 'electricity distribution grid charging',
                           "heat": "heat demand"},
                  inplace=True)

   
    elecdem = pd.DataFrame()
    


    elecdem['electric demand'] = supply['electricity distribution grid charging']
    #print(supply)



    to_drop = supply.columns[(supply < threshold).all()]



    if len(to_drop) != 0:
        print("dropping", to_drop)
        supply.drop(columns=to_drop, inplace=True)

    supply = pd.concat((supply, elecdem['electric demand']), axis = 1)
        
    print(supply)
    supply.index.name = None



    supply = supply / 1e3



    supply.columns = supply.columns.str.replace("residential ", "")#gets rid of some prefixes
    supply.columns = supply.columns.str.replace("services ", "")
    supply.columns = supply.columns.str.replace("urban decentral ", "decentral ")

    preferred_order = pd.Index(["electric demand",
                                "onshore wind",
                                "offshore wind",
                                "solar",
                                
                                "transmission lines",
                                "hydroelectricity",#uses
                                "hydro reservoir",
                                "run of river",
                                "pumped hydro storage",
                                "Fischer-Tropsch",
                                "CHP",
                                
                                "solar PV",#uses
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
                                "battery storage",#uses
                                
                                
                                "hot water storage"])

    # m = supply.lt(0).all()
    # supply = supply.loc[:, ~m]

    new_columns = (preferred_order.intersection(supply.columns)#what is the purpose of this? To only use the columns we want?
                   .append(supply.columns.difference(preferred_order)))


    supply =  supply.groupby(supply.columns, axis=1).sum()


    #We pass an ax instead
    # fig, ax = plt.subplots()
    # fig.set_size_inches((8, 5))
 

    supply.set_index(pd.date_range('2013-01-01','2014-01-01',freq='3H')[0:-1],inplace=True)



    #ax.plot(-supply['electric demand'], color = tech_colors['electric demand'], lw = 2)


    ax2 = supply[new_columns].loc[start:stop].plot(ax=ax, kind="area", stacked=True, linewidth=0.,
        color=[tech_colors[i.replace(suffix, "")]
                for i in new_columns])




    



    # supply['heat demand'].loc[start:stop].plot(ax = ax, kind = 'line', linewidth = 10, color = tech_colors['heat demand'])

    # print(heatdem)
    # print(supply)

    # (ax.plot(-heatdem['heat demand'].loc[start:stop],color=tech_colors['heat demand'],lw=3,zorder=100))
    
    handles, labels = ax.get_legend_handles_labels()
    #print(labels)

    # handles = [handles[0]] + [handles[1]] + [handles[3]]#for some reason heat demand was still in here so I had to avoid it
    # labels = [labels[0]] + [labels[1]] + [labels[3]]
    #print([item for item in labels][0] + [item for item in handles][2])


    handles.reverse()
    labels.reverse()

    new_handles = []
    new_labels = []

    for i, item in enumerate(labels):
        if "charging" not in item:
            new_handles.append(handles[i])
            new_labels.append(labels[i])

    #ax.set_x

    ax.legend(new_handles, new_labels, ncol=3, loc="upper left", frameon=False)
    ax.set_xlim([start, stop])

    print(ax.get_xticklabels())
    monthyearFmt = mdates.DateFormatter('%b')
    print(ax.get_xticklabels())

    ax.xaxis.set_major_formatter(monthyearFmt)
    ax.xaxis.set_tick_params(rotation = 45)
    ax.set_ylim([0, 2000])
    ax.grid(True)
    ax.set_ylabel("Power [GW]")
    # ax.set_xlabel("Day in January")

    if carrier == 'heat':
        ax.set_title("Heat supply (aggregate) during highest average natural gas use")
    else:
        ax.set_title("Electricity supply (aggregate) during highest average natural gas use")

    fig.tight_layout()
    #print(supply)




def assign_location(n):#looking at one port components(gen, load, storageunit, store) as well as branch (line, link, tranform)
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)#looking for space at 4th? idk
        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            if i == -1: continue
            names = ifind.index[ifind == i]
            c.df.loc[names, 'location'] = names.str[:i]
            
def assign_carriers(n):#why 
    if "carrier" not in n.lines:
        n.lines["carrier"] = "AC"#why
#plot_series_elec(n, carrier = "AC")


#%%
if __name__ == "__main__":
    fig,ax = plt.subplots(2,2,figsize=(14,11),sharex=True,sharey='row')
    ax = ax.flatten()

    #right now: ax
    
    plot_series_elec(n, ax[0], dates,carrier = 'AC',mav=mav)
    plot_series_elec(n2, ax[1], dates,carrier = 'AC',mav=mav)

    plot_series(n, ax[2], dates,carrier = 'heat',mav=mav)
    plot_series(n2, ax[3], dates,carrier = 'heat',mav=mav)

    # plot_series_elec_year(n15, ax[2], carrier = 'AC')
    # plot_series_elec_year(n2, ax[3], carrier = 'AC')


    ax[0].set_title(r"$\bf{Baseline}$",fontweight="bold",fontsize=fs)
    ax[0].set_ylabel(r"$\bf{Electricity [GW]}$" + '\n' + "Power")
    

    ax[1].set_title(r"$\bf{Gas\;Limit}$",fontweight="bold",fontsize=fs)
    ax[2].set_ylabel(r"$\bf{Heat [GW]}$" + '\n' + "Power")

    ax[2].set_title("")
    # ax[2].set_xticks("")


    #use ax 1 and 3 legend
    ax[3].set_title("")
    handles1, labels1 = ax[1].get_legend_handles_labels()
    # handles1 = handles1[::-1]
    # labels1 = labels1[::-1]
    # print(labels1)
    handles3, labels3 = ax[3].get_legend_handles_labels()
    # handles3 = [handles3[0], handles3[1], handles3[3], handles3[4], handles3[6]]
    # labels3 = [labels3[0],labels3[1], labels3[3], labels3[4], labels3[6]]

    handles3 = [handles3[0],  handles3[2], handles3[3], handles3[5]]
    labels3 = [labels3[0],labels3[2], labels3[3], labels3[5]]


    # handles3 = handles3[::-1]
    # handles3 = handles3[:-3] + [handles3[-1]]

    # labels3 = labels3[::-1]
    # labels3 = labels3[:-3] + [labels3[-1]]

    handles = handles1 + handles3
    labels = labels1 + labels3

    fig.legend(handles, labels, prop={'size':fs}, ncol=3, loc = (0.2, 0.02))

    # prop={'size':fs}

    fig.suptitle(str(t) + r"$\bf{^\circ C \; scenarios\; in\; }$" + str(year), fontweight = 'bold', fontsize = fs)
    
    #This needs to stay
    fig.tight_layout(rect = [0, 0.25, 1, 1])

    # ax[2].set_xticklabels(['Jan','','Mar','','May','','Jul','','Sep','','Nov','',''])
    # ax[3].set_xticklabels(['Jan','','Mar','','May','','Jul','','Sep','','Nov','',''])
    
    # ax[2].set_xticklabels(['',14,15,16,17,18,19,20,''])
    # ax[3].set_xticklabels(['',14,15,16,17,18,19,20,''])
    # ax[3].set_xlabel('Jan')
    # ax[2].set_xlabel('Jan')
    fig.savefig("../figures/aggregate_heat_and_elec_2CCompare" + str(year) + '_' + dates[0] + '_' + dates[1] + "week.pdf")

    plt.show()




#make_plot_Spain()


#plot_series(n, carrier="heat")
    #####_----------------------------Trash-----------------------------_###########


    # es2030gen = esgen[esgen.columns[esgen.columns.str.endswith("2030")|esgen.columns.str.endswith("ror")]] 

    # #very long because I want to select wind, relevant solar, and ror
    # relgen = es2030gen[es2030gen.columns[es2030gen.columns.str.contains("offwind")|es2030gen.columns.str.contains("onwind")|es2030gen.columns.str.endswith("solar-2030")|es2030gen.columns.str.endswith("ror")]]


    # relgen.columns.str.contains("wind")
    # #n.loads.loc[n.loads["name"].str.startswith("ES")]
    # relgen['wind'] = relgen.loc[:, relgen.columns.str.contains("wind")].sum(axis = 1)
    # relgen['solar'] = relgen.loc[:, relgen.columns.str.contains('solar')].sum(axis = 1)
    # relgen['ror'] = relgen.loc[:, relgen.columns.str.contains('ror')].sum(axis = 1)

# relgen['windavg'] = relgen['wind'].rolling(window = 56).mean()
# relgen['solaravg'] = relgen['solar'].rolling(window = 56).mean()
# relgen['roravg'] = relgen['ror'].rolling(window = 56).mean()

# esdem = pd.read_csv("../results/PyPSA-Eur-Sec-0.6.0/Nuclear_decommission_years_biomass_boiler/csvs/Spain_demand.csv")
# esdem['tot_demand'] = esdem['ES0 0'] + esdem['ES3 0']

# relgen['tot_demand'] = esdem['tot_demand']


# plt.plot(relgen["solar"][117:173])
# plt.plot(relgen['tot_demand'][117:173])
# plt.show()


#n.links.query('bus3 == "ES0 0"')
# n.links_t.p0[ES_busses].groupby(n.links.carrier,axis=1).sum()
#n.generators_t.p[ES_busses].groupby(n.generators.carrier, axis = 1 ).sum()
#ES_busses = n.generators.query('bus == "ES0 0"').index
#create electricity demand time series: ES0 0 and ES3 0.