#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, re
import argparse
import pandas as pd

def main():
    """Main program : reduce the pokemon datas"""
    parser = argparse.ArgumentParser(description='Reduce the pokemon datas, selecting pokemon names')

# ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])

    parser.add_argument('pokemon_datas', type=str, action='store',
            metavar='<pokemon_datas>', help='pokemon.csv path')

    parser.add_argument('-p', '--pokemon', dest='pok_name', default="Pikachu", type=str, action='store',
            metavar='<pok_name>', help="Name of the selected pokemon to keep [default: Pikachu]")

    parser.add_argument('-o', '--output', dest='outfname', default="output.csv", type=str, action='store',
            metavar='<outfname>', help="Name of the output txt file [default: output.csv]")

    args = parser.parse_args()  # de type <class 'argparse.Namespace'>

    print 'Reading ' + args.pokemon_datas + ' ...'
    spawns = pd.read_csv(args.pokemon_datas)

    print 'Selecting spawns of ' + args.pok_name + ' ...'
    spawn_one = spawns[spawns["name"] == "Pikachu"]

    print 'Writting csv ...'
    spawn_one.to_csv(args.outfname, index=False)


if __name__ == '__main__':
    main()
