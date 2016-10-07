#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def write_df_to_png(dataset, fout):
    template = r'''\documentclass[preview]{{standalone}}
    \usepackage{{booktabs}}
    \begin{{document}}
    {}
    \end{{document}}
    '''

    with open("tmp.tex", 'wb') as f:
        f.write(template.format(dataset.to_latex()))

    # subprocess.call(['pdflatex', "tmp.tex"])
    # subprocess.call(['convert', '-density', '300', "tmp.pdf", '-quality', '90', fout])


def main():
    """Main program : Count the number of Spawn per pokemon"""
    parser = argparse.ArgumentParser(description='Count the number of Spawn per pokemon')

    #ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])

    parser.add_argument('dataset', type=str, action='store',
            metavar='<dataset>', help='dataset path')

    args = parser.parse_args()  # de type <class 'argparse.Namespace'>

    print 'Reading ' + args.dataset + ' ...'
    dataset = pd.read_excel(args.dataset)
    dataset.columns = [ " X1 Relative Compactness      " ,\
                        " X2 Surface Area              " ,\
                        " X3 Wall Area                 " ,\
                        " X4 Roof Area                 " ,\
                        " X5 Overall Height            " ,\
                        " X6 Orientation               " ,\
                        " X7 Glazing Area              " ,\
                        " X8 Glazing Area Distribution " ,\
                        " y1 Heating Load              " ,\
                        " y2 Cooling Load              " ,\
                      ]

    table_describe = dataset.describe()

    cm = sns.light_palette("green", as_cmap=True)
    table_describe.style.background_gradient(cmap=cm)


    # df.style.set_caption('Colormaps, with a caption.')\
    #         .background_gradient(cmap=cm)

    # write_df_to_png(table_describe, "energ_describe.png")

    # table_describe_X = table_describe.iloc[:, 0:7]
    # print table_describe_X
    # table_describe_X.to_html("statistic_desc.html")
    # table_describe_Y = table_describe.iloc[:, 8:9]
    # print table_describe_Y
    # table_describe_Y.to_html("statistic_desc.html")




    dataset.hist()

    # print spawns[['name', 'encounter_ms']][:5]
    # print spawns['name'].value_counts()[:10]

    # pokemon_counts = spawns['name'].value_counts()
    # pokemon_counts[:10].plot(kind='bar')

    # plt.show() # show all figures

    # ax_count = pokemon_counts[:100].plot(kind='bar')
    # fig = ax_count.get_figure()
    plt.show()
    # fig.savefig('fooPokemon.png')


if __name__ == '__main__':
    main()
