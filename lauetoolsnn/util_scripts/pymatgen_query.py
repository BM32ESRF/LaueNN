# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 11:55:35 2022

@author: PURUSHOT

A query script to get materials unti cell information from PYMATGEN to index unknown alloy/materials

Idea: Query with known elemental composition of the alloy and get all possible alloys
and then train a NN with all the combinations (preferably with low HKL limit like 3)
and then hope that it will index one of the possible alloy components
"""
#% PYMATGEN project
from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifParser
import numpy as np
import _pickle as cPickle


## MY api key to access materialsproject database
pymatgen_api_key = "87iNakVJgbTSqn3A"

#Mg2Si
#Al-Fe-Mn
# criteria = {"elements": {"$in": ["Si", "Fe", "Cu", "Mn", "Mg", "Al", "Cr"]}}#, "$all": ["O"]}, "nelements":{'$lt':2}}

criteria = {"elements": ["Si", "Fe", "Cu", "Mn", "Mg", "Al", "Cr"]}#, "$all": ["O"]}, "nelements":{'$lt':2}}

properties = ["cif", "pretty_formula", "spacegroup.symbol"]

# criteria = {"pretty_formula": "GaN"}

with MPRester(api_key=pymatgen_api_key) as mpr:
    results = mpr.query(criteria=criteria,properties=properties)

# #or
# with MPRester(api_key=pymatgen_api_key) as mpr:
#     results = mpr.get_entries_in_chemsys("Si-Mn-Fe-Cu-Mg-Al-Cr")

#or
with MPRester(pymatgen_api_key) as mpr:
    results = mpr.query(criteria='Mn-Fe-Al', properties=properties)
#%%
## extract unit cell from the query 
a,b,c,alp,bet,gam = [],[],[],[],[],[]
lattice_params_cif = np.zeros((len(results), 6))
sg = []
for k, i in enumerate(results):
    print("Space group is ",i['spacegroup.symbol'])
    sg.append(i['spacegroup.symbol'])
    for j in i['cif'].split("\n"):
        if j.startswith('_cell_length_a'):
            a.append(j.split(" ")[-1])
            lattice_params_cif[k,0] = j.split(" ")[-1]
        if j.startswith('_cell_length_b'):
            b.append(j.split(" ")[-1])
            lattice_params_cif[k,1] = j.split(" ")[-1]
        if j.startswith('_cell_length_c'):
            c.append(j.split(" ")[-1])
            lattice_params_cif[k,2] = j.split(" ")[-1]
        if j.startswith('_cell_angle_alpha'):
            alp.append(j.split(" ")[-1])
            lattice_params_cif[k,3] = j.split(" ")[-1]
        if j.startswith('_cell_angle_beta'):
            bet.append(j.split(" ")[-1])
            lattice_params_cif[k,4] = j.split(" ")[-1]
        if j.startswith('_cell_angle_gamma'):
            gam.append(j.split(" ")[-1])
            lattice_params_cif[k,5] = j.split(" ")[-1]

structures = []
for material in results:
    structures.append(CifParser.from_string(material["cif"]).get_structures()[0])

material_info = np.zeros((len(structures),8))
formula = []
group_name = []
for i, ijk in enumerate(structures):
    material_info[i,0] = ijk.get_space_group_info()[1]
    material_info[i,1] = ijk._lattice.a
    material_info[i,2] = ijk._lattice.b
    material_info[i,3] = ijk._lattice.c
    material_info[i,4] = ijk._lattice.alpha
    material_info[i,5] = ijk._lattice.beta
    material_info[i,6] = ijk._lattice.gamma
    material_info[i,7] = ijk._lattice.volume
    formula.append(ijk.formula)
    group_name.append(ijk.get_space_group_info()[0])

unique_list, index_unique = np.unique(material_info[:,0], return_index=True)
material_info_unique = material_info[index_unique, :]
formula_unique = [formula[i] for i in index_unique]
group_name_unique = [group_name[i] for i in index_unique]
print(len(np.unique(material_info[:,0])))

## dump a pickle file with all the data
save_data = False
if save_data:
    with open("exhaustive_list_space_group.pickle", "wb") as output_file:
        cPickle.dump([index_unique, material_info_unique, formula_unique, \
                      group_name_unique, material_info, formula, group_name], output_file) 


















