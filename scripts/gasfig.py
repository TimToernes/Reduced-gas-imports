#This file will h

import pypsa

n = pypsa.Network()

n.import_from_netcdf("results/version-gaslimit-3H/postnetworks/elec_s370_37m_lv1.0__3H-T-H-B-I-solar+p3-dist1-cb25.7ex0_2020.nc")

print(n.generators)