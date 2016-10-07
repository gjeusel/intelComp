#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# # Make the graphs a bit prettier, and bigger
# pd.set_option('display.mpl_style', 'default')
# pd.set_option('display.width', 5000)
# pd.set_option('display.max_columns', 60)

def main():
    """Main program : Count the number of Spawn per pokemon"""
    parser = argparse.ArgumentParser(description='Count the number of Spawn per pokemon')

    #ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])

    parser.add_argument('pokemon_datas', type=str, action='store',
            metavar='<pokemon_datas>', help='pokemon.csv path')

    args = parser.parse_args()  # de type <class 'argparse.Namespace'>

    print 'Reading ' + args.pokemon_datas + ' ...'
    spawns = pd.read_csv(args.pokemon_datas)
    # Schema : s2_id,s2_token,num,name,lat,lng,encounter_ms,disppear_ms
    #     s2_id and s2_token reference Google's S2 spatial area library.
    #     num represents pokemon pokedex id
    #     encounter_ms represents time of scan
    #     disappear_ms represents time this encountered mon will despawn

    # print spawns[['name', 'encounter_ms']][:5]
    # print spawns['name'].value_counts()[:10]

    pokemon_counts = spawns['name'].value_counts()
    pokemon_counts[:10].plot(kind='bar')

    # plt.show() # show all figures

    # ax_count = pokemon_counts[:100].plot(kind='bar')
    # fig = ax_count.get_figure()
    # plt.show()
    # fig.savefig('fooPokemon.png')


if __name__ == '__main__':
    main()
