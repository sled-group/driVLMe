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
from networkx.readwrite import json_graph
import networkx as nx


def game_loop(args):
    """ Main loop for agent"""
    world = None
    tot_target_reached = 0
    num_min_waypoints = 21

    client = carla.Client(args.host, args.port)
    client.set_timeout(101.0)

    client.load_world(args.town)
    carlamap = client.get_world().get_map()
    roadlist=[]
    junction_list=[]
    topology=carlamap.get_topology()
    node_list=[]

    def generate_transform(point):
        transform={}
        transform["location"]={}
        transform["rotation"]={}
        transform["location"]["x"]=point.transform.location.x
        transform["location"]["y"]=point.transform.location.y
        transform["location"]["z"]=point.transform.location.z
        transform["rotation"]["pitch"]=point.transform.rotation.pitch
        transform["rotation"]["yaw"]=point.transform.rotation.yaw
        transform["rotation"]["roll"]=point.transform.rotation.roll
        return transform

    def generate_opendrive(point):
        opendrive={}
        opendrive["road_id"]=point.road_id
        opendrive["junction_id"]=point.get_junction().id if point.is_junction else 0
        opendrive["lane_id"]=point.lane_id
        return opendrive
    
    def append_node(point,node_list):
        opendrive=generate_opendrive(point)
        for node in node_list:
            if (opendrive["road_id"]==node[1]["opendriveid"]["road_id"])&(
                opendrive["lane_id"]==node[1]["opendriveid"]["lane_id"]):
                return node[0]
                
        start=point.next_until_lane_end(0.02)[-1]
        end=point.previous_until_lane_start(0.02)[-1]
        length=abs(start.s-end.s)
        start_transform=generate_transform(start)
        end_transform=generate_transform(end)
        i=len(node_list)
        node_list.append((i,{"start":start_transform,"end":end_transform,"opendriveid":opendrive,"length":length,"have_left":False,"have_right":False}))
        return i
    node_list=[]
    edge_list=[]
    for point1,point2  in topology:
        node1=append_node(point1,node_list)
        node2=append_node(point2,node_list)
        edge_list.append((node1,node2,{"type":"concat"}))
        
    print(len(node_list))
        
        
    G=nx.DiGraph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)
    print(len(G.nodes))
    print(len(G.edges))
    for node1 in G.nodes:
        for node2 in G.nodes:
            if (G.nodes[node1]["opendriveid"]["road_id"]==G.nodes[node2]["opendriveid"]["road_id"])and  (
                abs(G.nodes[node1]["opendriveid"]["lane_id"])-abs(G.nodes[node2]["opendriveid"]["lane_id"])==1)and  (
                abs(G.nodes[node1]["opendriveid"]["lane_id"]-G.nodes[node2]["opendriveid"]["lane_id"])==1):
                G.add_edge(node1,node2,type="left_lane_change")
                G.nodes[node1]["have_left"]=True
            if (G.nodes[node1]["opendriveid"]["road_id"]==G.nodes[node2]["opendriveid"]["road_id"])and  (
                abs(G.nodes[node1]["opendriveid"]["lane_id"])-abs(G.nodes[node2]["opendriveid"]["lane_id"])==-1)and  (
                abs(G.nodes[node1]["opendriveid"]["lane_id"]-G.nodes[node2]["opendriveid"]["lane_id"])==1):
                G.add_edge(node1,node2,type="right_lane_change")
                G.nodes[node1]["have_right"]=True
    for node1 in G.nodes:
        other_lane=-100*G.nodes[node1]["opendriveid"]["lane_id"]
        dest_node=-1
        for node2 in G.nodes:

            if (G.nodes[node1]["opendriveid"]["road_id"]==G.nodes[node2]["opendriveid"]["road_id"]) and(
                abs(G.nodes[node2]["opendriveid"]["lane_id"])<abs(other_lane))and  (
                G.nodes[node1]["opendriveid"]["lane_id"]*G.nodes[node2]["opendriveid"]["lane_id"]<0):
                other_lane=G.nodes[node2]["opendriveid"]["lane_id"]
                dest_node=node2
        if dest_node>=0:
            G.add_edge(node1,dest_node,type="uturn")

    print(len(G.edges))
    data=json_graph.adjacency_data(G)
    print(data)
    with open(args.town+".json","w")as f:
        json.dump(data,f)

                
        
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