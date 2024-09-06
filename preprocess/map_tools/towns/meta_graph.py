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
    def load_waypoint(point):
        location=point.transform.location
        rotation=point.transform.rotation
        waypoint={}
        waypoint["Transform"]={}
        waypoint["Transform"]["Location"]={}
        waypoint["Transform"]["Rotation"]={}
        waypoint["Transform"]["Location"]['x']=location.x
        waypoint["Transform"]["Location"]['y']=location.y
        waypoint["Transform"]["Location"]['z']=location.z
        waypoint["Transform"]["Rotation"]["pitch"]=rotation.pitch
        waypoint["Transform"]["Rotation"]["yaw"]=rotation.yaw
        waypoint["Transform"]["Rotation"]["roll"]=rotation.roll
        waypoint["OpenDriveID"]={}
        waypoint["OpenDriveID"]["road_id"]=point.road_id
        waypoint["OpenDriveID"]["lane_id"]=point.lane_id
        waypoint["OpenDriveID"]["junction_id"]=point.get_junction().id if point.is_junction else -1

        waypoint["s"]=point.s
        return waypoint

    client = carla.Client(args.host, args.port)
    client.set_timeout(101.0)

    client.load_world(args.town)
    carlamap = client.get_world().get_map()
    print(len(carlamap.get_topology()))
    junctionidlist=[]
    junction_list=[]
    for (point1,point2) in carlamap.get_topology():
        if (point1.road_id==35)or(point2.road_id==35):
            print(load_waypoint(point1))
            print(load_waypoint(point2))
        if point1.is_junction:
            if point1.get_junction().id not in junctionidlist:
                junctionidlist.append(point1.get_junction().id)
                junction_list.append(point1.get_junction())
        if point2.is_junction:
            if point2.get_junction().id not in junctionidlist:
                junctionidlist.append(point2.get_junction().id)
                junction_list.append(point2.get_junction())
    node_list=[(junction.id,{"x_axis":junction.bounding_box.location.x,
        "y_axis":junction.bounding_box.location.y,
        "id":junction.id,
        "obj":junction,
        "neighbor_edge":set([])}) for junction in junction_list]
    G=nx.DiGraph()
    G.add_nodes_from(node_list)
    print(list(G.nodes))
    for idx in G.nodes:
        junction=G.nodes[idx]["obj"]
        waypoints=junction.get_waypoints(carla.LaneType.Driving)
        for  (point1,point2) in waypoints:
            next_list=point2.next(0.01)
            if (len(next_list)>0):
                if (not next_list[0].is_junction):
                    G.nodes[idx]["neighbor_edge"].add(next_list[0].road_id)
                    tmp=next_list[0]
                    start=tmp
                    ids=[]
                    length=0
                    while not (tmp.is_junction):
                        waypoint=tmp.next_until_lane_end(0.01)[-1]
                        # print(waypoint.road_id,waypoint.lane_id)
                        ids.append(waypoint.road_id)
                        length+=max(waypoint.s,tmp.s)
                        tmp=waypoint.next(0.01)[0]
                    G.add_edge(idx,tmp.get_junction().id,ids=ids,length=length)
                    if idx in [1682, 1082, 1191, 1205, 1469, 1654, 1736]:
                        print(tmp.get_junction().id)
                        print(ids,length)
        if idx in [1682, 1082, 1191, 1205, 1469, 1654, 1736]:
            print(G.nodes[idx]["neighbor_edge"])
            print(idx)
    for idx in G.nodes:
        G.nodes[idx]["obj"]=0
        G.nodes[idx]["neighbor_edge"]=list(G.nodes[idx]["neighbor_edge"])

    print(len(G.edges))
    data=json_graph.adjacency_data(G)
    # print(data)
    with open(args.town+"_meta.json","w")as f:
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
