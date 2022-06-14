#Adam Dvorak, 10 June

#This is a function that:
    #Groups the dataframe by historical years and future years
    #Sums up the years for historical and future by country
    #Imports shapefiles
    #Plots on map of Europe the amount of historical and current coal used



import pypsa
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime
import matplotlib.dates as mdates

#2030 constrained
n = pypsa.Network()
n.import_from_netcdf("../results/PyPSA-Eur-Sec-0.6.0/3H_oil_lignite_nonextendable/postnetworks/elec_s370_37_lv1.0__3H-T-H-B-I-A-solar+p3-dist1-cb73.9ex0-gasconstrained_2030.nc")
n.links.query('carrier == "coal"').to_csv("../data/2030")
#2025 constrained
n1 = pypsa.Network()
n1.import_from_netcdf("../results/PyPSA-Eur-Sec-0.6.0/3H_oil_lignite_nonextendable/postnetworks/elec_s370_37_lv1.0__3H-T-H-B-I-A-solar+p3-dist1-cb73.9ex0-gasconstrained_2025.nc")
#2030 unconstrained
n2 = pypsa.Network()
n2.import_from_netcdf("../results/PyPSA-Eur-Sec-0.6.0/3H_oil_lignite_nonextendable/postnetworks/elec_s370_37_lv1.0__3H-T-H-B-I-A-solar+p3-dist1-cb73.9ex0_2030.nc")
#2025 unconstrained
n3 = pypsa.Network()
n3.import_from_netcdf("../results/PyPSA-Eur-Sec-0.6.0/3H_oil_lignite_nonextendable/postnetworks/elec_s370_37_lv1.0__3H-T-H-B-I-A-solar+p3-dist1-cb73.9ex0_2025.nc")


n = pd.read_csv("../data/coalinstalled_2c_gasconstrained_datatable.csv")


n['country'] = n.apply(lambda row: row['bus1'][0:2], axis = 1)
n2 = n[['name', 'country', 'p_nom', 'build_year', 'p_nom_opt']]
n3 = n2.query('build_year == 2030') 

n_oldgen = n.query('build_year < 2020')
n_oldgroupgen = n_oldgen.groupby("country").sum()

n_newgen = n.query('build_year > 2020')
n_newgroupgen = n_newgen.groupby("country").sum()
n_filtered= n_newgroupgen.query('p_nom_opt > 1')




