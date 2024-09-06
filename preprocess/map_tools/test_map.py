import pandas as pd
import planner
import json

town_name = 'Town03'
config_filepath = './example/config.json'
with open(config_filepath, 'r') as f:
    config = json.load(f)
trajectory_csv = './example/trajectory.csv'

# Get vehicle position
if trajectory_csv is not None:
    trajectories = pd.read_csv(trajectory_csv)
frame = 6
row = trajectories.loc[trajectories['frame'] == frame]
road_id = int(row.iloc[0][' road_id'])
yaw = int(row.iloc[0][' yaw'])
x = row.iloc[0][' x']
y = row.iloc[0][' y']
vehicle_road_id = int(row.iloc[0][' road_id'])

tmap = planner.RoutePlanner(town_name, config, trajectory_csv=trajectory_csv)
tmap.plan_route('house', vehicle_road_id, x, y, yaw)
