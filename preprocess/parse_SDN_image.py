from llm_interface import LLMInterface
import warnings
import map_tools.map_tool as map_tool
import json
import openai
from tqdm import tqdm
import os

from decord import VideoReader, cpu
import cv2
import shutil

def create_video(image_filenames, output_file):
    # Create a VideoWriter object
    frame_size = (960, 540)  # Example frame size (width, height)
    frame_rate = 5  # Example frame rate
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is the codec for .mp4
    video = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)

    for filename in image_filenames:
        img = cv2.imread(filename)
        img = cv2.resize(img, frame_size)  
        
        video.write(img)

    video.release()

class SDN_Tester:
    
    def __init__(self, log_folder_path, model, llm):
        self.log_folder_path = log_folder_path
        self.id = log_folder_path.split("/")[-1]
        self.model = model

        self.cur_timestep = 0

        self.eval_goal_updates = False
        self.eval_physical_actions = True

        if self.eval_physical_actions:
            self.total_physical_actions = 0
            self.physical_action_slot_types_correct = 0
            self.physical_action_slot_vals_correct = 0

        annotated_log_path = os.path.join(self.log_folder_path , 'annotated_log.json')
        with open(annotated_log_path, 'r') as f:
            self.annotated_log = json.load(f)

        self.llm = llm
        self.conversations=[]

    # Loads the following from the log folder:
    # - Knowledge, including street names and known landmarks
    # - Map topology
    def load_initial(self):
        config_path = os.path.join(self.log_folder_path , 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        map_name = config['environment']['map']
        
        trajectory_csv = os.path.join(self.log_folder_path ,'trajectory.csv')
        turn_plan_path = os.path.join(self.log_folder_path ,'turn_plan.json')
        with open(turn_plan_path, 'r') as f:
            turn_plan = json.load(f)
        additional_path = os.path.join(self.log_folder_path ,"out",'additional.json')
        with open(additional_path, 'r') as f:
            additional = json.load(f)
        

    # Loads the following for a particular timestep:
    # - RGB image associated with the timestep
    # - Annotated log data that occurred during the timestep
    def process_next_timestep(self):
        elem = self.annotated_log[self.cur_timestep]

        # print(f'FRAME: {elem["frame"]}')
        self.generate(elem)

        self.cur_timestep += 1


    def generate(self, elem):
        if (elem['type'] == 'PhysicalAction') or (elem['type'] == 'DialogueMove'):
            self.conversations.append([])
            self.total_physical_actions += 1
            # TODO: add slot val evaluation
            #self.physical_action_slot_vals_correct
            frame = elem['frame']
            root = f"dorothie_train_videos/{self.id}_{frame}.mp4"
            output_file =  f"dorothie_test_videos/{self.id}_{frame}.mp4"
            rgb_dir = os.path.join(self.log_folder_path ,"out",'rgb')
            if os.path.exists(output_file):
                try:
                    
                    vr = VideoReader(output_file, ctx=cpu(0))
                    return
                except:
                    pass                
            #     print(output_file)
            image_filenames = []
            for i in range(40):
                if frame - i*2<0:
                    break
                if os.path.exists(os.path.join(rgb_dir,f"{frame - i*2}.png")):
                    image_filenames.append(os.path.join(rgb_dir,f"{frame - i*2}.png"))
            # print(self.log_folder_path,image_filenames)
            image_filenames.reverse()
            create_video(image_filenames, output_file)
    # Provides the information from a timestep to the model and
    # removes information that the model should not access
    def give_timestep_to_model(self):
        pass

    # Gets the total number of timesteps
    def get_total_timesteps(self):
        return len(self.annotated_log)

    # Tests the model on one particular log folder
    def test_on_log(self):
        # Get the initial data and provide it to the model
        self.load_initial()

        # Loop through timesteps
        total_timesteps = self.get_total_timesteps()
        while self.cur_timestep < total_timesteps:
            self.process_next_timestep()
        return len(self.conversations)

# if __name__ == '__main__':
#     split="train"
#     root = f"/nfs/turbo/coe-chaijy/owenhji/SDN_final/{split}/"
#     model=6
#     dataset=0
#     for log_fold in os.listdir(root):
#         log_path = os.path.join(root,log_fold)
#         sdn_tester = SDN_Tester(log_path, model, 'cmd')
#         dataset+=sdn_tester.test_on_log()
#     print(dataset)


def generate_video(log_fold):
    split="test"
    root = f"/nfs/turbo/coe-chaijy/owenhji/SDN_final/{split}/"
    model=6
    log_path = os.path.join(root,log_fold)
    sdn_tester = SDN_Tester(log_path, model, 'cmd')
    out = sdn_tester.test_on_log()
    return out
import multiprocessing
from tqdm.contrib.concurrent import process_map 
if __name__ == '__main__':
    split="test"
    root = f"/nfs/turbo/coe-chaijy/owenhji/SDN_final/{split}/"
    dataset=[]
    num_processes = multiprocessing.cpu_count()
    print(num_processes)
    results=process_map(generate_video, os.listdir(root), max_workers=num_processes)
    print(sum(results))