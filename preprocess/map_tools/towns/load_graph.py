#!/usr/bin/python3

# ==============================================================================
# -- COPYRIGHT -----------------------------------------------------------------
# ==============================================================================

# Copyright (c) 2022 SLED Lab, EECS, University of Michigan.
# authors: Martin Ziqiao Ma (marstin@umich.edu),
#          Cristian-Paul Bara (cpbara@umich.edu),
#          Owen Yidong Huang (owenhji@umich.edu),
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



from __future__ import print_function

import argparse
import sys
import weakref
import threading
import json
from queue import Queue
import socket
import time
import carla
import os
import networkx as nx


def get_junction(G, junction_id):
    node_list = []
    for node in G.nodes:
        if G.nodes[node]["opendriveid"]["junction_id"] == junction_id:
            node_list.append(node)
    return node_list


def get_road(G, road_id):
    node_list = []
    for node in G.nodes:
        if G.nodes[node]["opendriveid"]["road_id"] == road_id:
            node_list.append(node)
    return node_list


def game_loop(args):
    """ Main loop for agent"""
    with open(args.town+".json","r") as f:
        data=json.load(f)
    G=nx.adjacency_graph(data,directed=True)
    print(get_junction(G,1352))
    print(get_road(G,0))
    for node in get_road(G,0):
        print(G.nodes[node])
        print(G.edges(node))
        edges=G.edges(node)
        for edge in edges:
            print(G.edges[edge])
            # print(G.nodes[edge[1]])
                
        
# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--town',
        help='townname of the graph',
        dest='town',)

    args = argparser.parse_args()

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()