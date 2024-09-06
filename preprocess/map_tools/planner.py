import networkx as nx
import pandas as pd
import numpy as np
import json
import math
import turn_angles

class RoutePlanner:

    def __init__(self, town_name, config, trajectory_csv=None):
        # Load town map file
        self.town_name = town_name
        town_data_filepath = './towns/' + town_name + '.json'
        with open(town_data_filepath, 'r') as f:
            self.town_data = json.load(f)

        meta_town_data_filepath = '../map/' + town_name + '_meta.json'
        with open(meta_town_data_filepath, 'r') as f:
            self.meta_town_data = json.load(f)
        self.G = nx.node_link_graph(self.meta_town_data)

        # Create a networkx graph based on the town_data json file
        self.mapx = nx.readwrite.adjacency_graph(self.town_data['map'], directed=True)
        self.meta_map = nx.readwrite.adjacency_graph(self.town_data['meta_map'], directed=True)

        # Load config (for landmark locations, etc.)
        self.config = config

    def plan_route(self, landmark_name, vehicle_road_id, v_x, v_y, v_yaw):
        # Get landmark position
        landmarks = self.config['environment']['landmarks']
        found = False
        for landmark in landmarks:
            if landmark['name'] == landmark_name:
                found = True
                break
        if not found:
            print(f'Landmark {landmark_name} is not defined')
            return None
        landmark_road_id = landmark['wp']['OpenDriveID']['road_id']
        landmark_junction_id = landmark['wp']['OpenDriveID']['junction_id']

        # Compute junctions near to vehicle
        vehicle_nearby_junctions = []
        for node in self.town_data['meta_map']['adjacency']:
            for sub_node in node:
                if vehicle_road_id in sub_node['ids']:
                    vehicle_nearby_junctions.append(sub_node['id'])

        # Determine which junction the vehicle is facing
        unit_vec_facing = np.array([math.cos(v_yaw), math.sin(v_yaw)])
        x_vals = []
        y_vals = []
        for junction in vehicle_nearby_junctions:
            j_x = None
            j_y = None
            for node in self.town_data['meta_map']['nodes']:
                if node['id'] == junction:
                    j_x = node['x_axis']
                    j_y = node['y_axis']
                    break
            if j_x is not None:
                x_vals.append(j_x)
                y_vals.append(j_y)
            else:
                raise Exception(f'Junction ID {junction} is not recognized')
        
        x_diff = x_vals[0] - x_vals[1]
        y_diff = y_vals[0] - y_vals[1]
        unit_vec_junction_diff = np.array([x_diff, y_diff])
        dot = np.dot(unit_vec_facing, unit_vec_junction_diff)

        if dot > 0:
            facing_junction = vehicle_nearby_junctions[0]
            prev_junction = vehicle_nearby_junctions[1]
        else:
            facing_junction = vehicle_nearby_junctions[1]
            prev_junction = vehicle_nearby_junctions[0]
        
        # Compute shortest path
        path = nx.shortest_path(self.meta_map, source=facing_junction, target=landmark_junction_id)

        # Compute turn directions and road names
        direction_str = []
        path = [prev_junction] + path
        for i in range(len(path)-2):
            junction_start = path[i]
            junction_mid = path[i+1]
            junction_end = path[i+2]
            prev_road = self.G.edges[junction_start, junction_mid]
            post_road = self.G.edges[junction_mid, junction_end]
            turn_angle = post_road['start_angle'] - prev_road['end_angle']
            print(turn_angle)
            junction_tuple = (junction_start, junction_mid, junction_end)
            direction = get_turn_direction(turn_angle, intersection_tuple=junction_tuple, town=self.town_name)

            street_name = None
            for j, street in enumerate(self.town_data['streets']['junction_id']):
                if (junction_mid in street) and (junction_end in street):
                    street_name = self.config['environment']['street_names'][j]
                    break

            turn_str = f'{direction} onto {street_name}'
            direction_str.append(turn_str)

        print(path)
        print(vehicle_nearby_junctions)
        print(landmark_junction_id)
        print(direction_str)


def get_turn_direction(angle, intersection_tuple=None, town=None):

    if intersection_tuple is not None and town is not None:
        intsersection_triples = turn_angles.intersection_triples[town]
        if intersection_tuple in intsersection_triples.keys():
            return intsersection_triples[intersection_tuple]

    if (angle > 30) and (angle < 150):
        return "right"
    elif (angle >= 150) and (angle < 210):
        return "U turn"
    elif (angle >= 210) and (angle < 330):
        return "left"
    elif (angle <= 30) or (angle > 330):
        return "straight"


