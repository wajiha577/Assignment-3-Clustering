#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:31:09 2023

@author: Wajiha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy.polynomial.polynomial import polyfit
import os
import scipy.optimize as opt


def create_2d_cluster(df):
    x = df['Urban Population']
    y = df['CO2 Emission']
    kmeans = KMeans(n_clusters=4, random_state=0)
    df['cluster'] = kmeans.fit_predict(df[['Urban Population', 
                                                     'CO2 Emission']])
    # get centroids
    centroids = kmeans.cluster_centers_
    cen_x = [i[0] for i in centroids] 
    cen_y = [i[1] for i in centroids]
    ## add to df
    df['cen_x'] = df.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2],
                                            3: cen_x[3]})
    df['cen_y'] = df.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2],
                                            3: cen_y[3]})
    # define and map colors

    colors = ['#DF2020', '#81DF20', '#2095DF', "#FFF"]
    df['c'] = df.cluster.map({0:colors[0], 1:colors[1], 2:colors[2], 
                                        3: colors[3]})

    fig, ax = plt.subplots()

    ax.scatter(x, y, 
                c=df.c, alpha = 0.6, s=10)
    ax.set_facecolor('black')
    plt.xlabel("Urban Population")
    plt.ylabel("CO2 Emission")

    plt.scatter(df['cen_x'], df['cen_y'], 10, "purple", marker="d",)
    b, m = polyfit(x, y, 1)
    plt.plot(x, b + m * x, '--')
    
    plt.plot([x.mean()]*2, [0,3e7], color='#ddd', lw=0.5, linestyle='--')
    plt.plot([0,3e9], [y.mean()]*2, color='#ddd', lw=0.5, linestyle='--')
    
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)

    ax.plot(x, p(x), label='Fit', linewidth=0.2)
    
    
    
def create_3d_cluster(data):
    x_element = 'Urban Population'
    y_element = 'CO2 Emission'
    z_element = 'Energy Usage'
    x = data[x_element]
    y = data[y_element]
    z = data[z_element]
    colors = ['#DF2020', '#81DF20', '#2095DF']
    kmeans = KMeans(n_clusters=3, random_state=0)
    data['cluster'] = kmeans.fit_predict(data[[x_element, y_element, 
                                               z_element]])
    data['c'] = data.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})
    fig = plt.figure(figsize=(26,6))
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(x, y, z, c=data.c, s=15)
    ax.set_xlabel(x_element)
    ax.set_ylabel(y_element)
    ax.set_zlabel(z_element)
    
    
    

directory = os.path.dirname(__file__)
df_urban_pop = pd.read_csv(os.path.join(directory, 'urban_pop.csv'))
df_co2 = pd.read_csv(os.path.join(directory, 'co2_emission.csv'))
df_energy_use = pd.read_csv(os.path.join(directory, 'energy_usage.csv'))

sub_df_urban_pop = df_urban_pop[['1980', '1990', '2000', '2010', '2020']]
# pd.plotting.scatter_matrix(sub_df_urban_pop, figsize=(8, 5), s=5, alpha=0.8)
selected_year = '2014'

df_urban_pop = df_urban_pop[df_urban_pop[selected_year].notna()]
df_co2 = df_co2[df_co2[selected_year].notna()]
df_energy_use = df_energy_use[df_energy_use[selected_year].notna()]

df_urban_pop2014 = df_urban_pop[["Country Name", "Country Code", 
                                 selected_year]].copy()
df_co22014 = df_co2[["Country Name", "Country Code", selected_year]].copy()
df_energy2014 = df_energy_use[["Country Name", "Country Code", 
                               selected_year]].copy()

df_2014 = pd.merge(df_urban_pop2014, df_co22014, on="Country Name", how="outer")

df_2014 = df_2014.dropna()

df_2014 = df_2014.rename(columns = {"2014_x":"Urban Population", 
                                  "2014_y":"CO2 Emission"})

print(df_2014.corr())

create_2d_cluster(df_2014)
print(df_energy2014)
df_2014 = pd.merge(df_2014, df_energy2014, on = "Country Name", how = "outer")
df_2014 = df_2014.dropna()
df_2014 = df_2014.rename(columns = {"2014":"Energy Usage"})
create_3d_cluster(df_2014)

plt.show()
















