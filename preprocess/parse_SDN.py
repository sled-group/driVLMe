from llm_interface import LLMInterface
import warnings
import map_tools.map_tool as map_tool
import json
import openai
import os
import argparse

class SDN_Tester:
    
    def __init__(self,split, log_folder_path, model, llm):
        self.split = split
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
        self.actions=[]
        self.answers=[]

    # Loads the following from the log folder:
    # - Knowledge, including street names and known landmarks
    # - Map topology
    def load_initial(self):
        config_path = os.path.join(self.log_folder_path , 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        map_name = config['environment']['map']
        print(map_name)
        self.is_hidden = (map_name=="Town02")
        
        trajectory_csv = os.path.join(self.log_folder_path ,'trajectory.csv')
        turn_plan_path = os.path.join(self.log_folder_path ,'turn_plan.json')
        with open(turn_plan_path, 'r') as f:
            turn_plan = json.load(f)
        additional_path = os.path.join(self.log_folder_path ,"out",'additional.json')
        with open(additional_path, 'r') as f:
            additional = json.load(f)
        
        town_map = map_tool.TownMap(map_name, config, trajectory_csv=trajectory_csv)

        assets = {}
        for elem in config['environment']['landmarks']:
            assets[elem['asset']]=elem['name']
        print(assets)
        self.interface = LLMInterface(assets,turn_plan,additional, town_map, llm=self.llm)
        self.counter=0

    # Loads the following for a particular timestep:
    # - RGB image associated with the timestep
    # - Annotated log data that occurred during the timestep
    def process_next_timestep(self):
        elem = self.annotated_log[self.cur_timestep]

        # print(f'FRAME: {elem["frame"]}')

        if elem['type'] == 'DialogueMove':
            if elem['from'] == 'dorothy':
                self.interface.receive_dialogue(elem['utterance_gt'])

        if self.eval_goal_updates:
            self.eval_goal_update(elem)
        self.generate(elem)
        if elem['type'] == 'DialogueMove':
            if elem['from'] == 'wizard':
                self.interface.inject_llm_dialogue(elem['utterance_gt'])

        if elem['type'] == 'PhysicalAction':
            self.interface.inject_physical_action(elem)

        self.cur_timestep += 1

    def eval_goal_update(self, elem):
        if elem['type'] == 'BeliefUpdate' and elem['val']['act'] == 'GoalUpdate':
            try:
                destination = elem['val']['slot_val'][-1]['landmark']['asset']

                self.interface.evaluate()

                if destination.lower() in self.interface.current_goal:
                    num_correct_goal_updates += 1
            except Exception as e:
                warnings.warn('some sort of issue')
                print(e)

    def generate(self, elem):
        frame = elem['frame']
        # print(self.log_folder_path,frame)
        if str(frame) not in self.interface.additional["data"]:
            return
        if ((elem['type'] == 'DialogueMove') and (elem['from'] == 'wizard')):
            # TODO: add slot val evaluation
            #self.physical_action_slot_vals_correct
            if self.split =="train":
                output_content = {"unseen":self.is_hidden,'id': f"{self.id}_{frame}", 'video': f"{self.id}_{frame}.pkl", 'conversations': []}
            else:
                output_content = {"unseen":self.is_hidden,'id': f"{self.id}_{frame}", 'video': f"{self.id}_{frame}.mp4", 'conversations': []}
                if not os.path.exists(f"dorothie_test_videos/{self.id}_{frame}.mp4"):
                    print(f"dorothie_test_videos/{self.id}_{frame}.mp4")
                
            prompt = self.interface.caption_prompt(frame)
            output_content['conversations'].append({'from': 'human', 'value': f"<video>\n{prompt}"})
            prompt = self.interface.caption(frame)
            output_content['conversations'].append({'from': 'gpt', 'value': prompt})
            prompt = self.interface.plan_prompt(frame)
            output_content['conversations'].append({'from': 'human', 'value': prompt})
            prompt = self.interface.plan_out(frame)
            output_content['conversations'].append({'from': 'gpt', 'value': prompt})
            
            
            
            prompt = self.interface.dialogue_question(frame)
            output_content['conversations'].append({'from': 'human', 'value': prompt})
            prompt = self.interface.dialogue_ans(frame,elem)
            output_content['conversations'].append({'from': 'gpt', 'value': prompt})
            self.conversations.append(output_content)

            
        if elem['type'] == 'PhysicalAction':
            self.total_physical_actions += 1
            # TODO: add slot val evaluation
            #self.physical_action_slot_vals_correct
            frame = elem['frame']
            act = elem['val']['act']
            slot_type = elem['val']['slot_type']
            slot_val = elem['val']['slot_val']
            if self.split =="train":
                output_content = {"unseen":self.is_hidden,'id': f"{self.id}_{frame}", 'video': f"{self.id}_{frame}.pkl", 'conversations': []}
            else:
                output_content = {"unseen":self.is_hidden,'id': f"{self.id}_{frame}", 'video': f"{self.id}_{frame}.mp4", 'conversations': []}
                if not os.path.exists(f"dorothie_test_videos/{self.id}_{frame}.mp4"):
                    print(f"dorothie_test_videos/{self.id}_{frame}.mp4")
            prompt = self.interface.caption_prompt(frame)
            output_content['conversations'].append({'from': 'human', 'value': f"<video>\n{prompt}"})
            prompt = self.interface.caption(frame)
            output_content['conversations'].append({'from': 'gpt', 'value': prompt})
            prompt = self.interface.plan_prompt(frame)
            output_content['conversations'].append({'from': 'human', 'value': prompt})
            prompt = self.interface.plan_out(frame)
            output_content['conversations'].append({'from': 'gpt', 'value': prompt})




            prompt,direction = self.interface.question(frame)
            output_content['conversations'].append({'from': 'human', 'value': prompt})
            if act == 'SpeedChange':
                
                output_content['conversations'].append({'from': 'gpt', 'value': f"F. Keep on current Lane"})
                output_content['conversations'].append({'from': 'human', 'value': "Are you willing to change the current speed? \n A. Accelerate\n B. Decelerate \n C. Keep current speed"})
                if  slot_val>0:
                    output_content['conversations'].append({'from': 'gpt', 'value': f"A. Accelerate"})
                else:
                    output_content['conversations'].append({'from': 'gpt', 'value': f"B. Decelerate"})
            elif act == 'Start':
                output_content['conversations'].append({'from': 'gpt', 'value': f"A. Start"})
            elif act == 'Stop':
                output_content['conversations'].append({'from': 'gpt', 'value': f"B. Stop"})
            elif act == 'UTurn':
                output_content['conversations'].append({'from': 'gpt', 'value': f"E. Make U turn"})
            elif act == 'LaneSwitch':
                output_content['conversations'].append({'from': 'gpt', 'value': f"C. Change lane"})
                output_content['conversations'].append({'from': 'human', 'value': "Which direction are you switching to?\n A. Change to right lane\n B. Change to left lane"})
                if slot_val == 2:
                    ans = "A. Change to right lane"
                else:
                    ans = "B. Change to left lane"
                output_content['conversations'].append({'from': 'gpt', 'value': ans})
            elif act == "JTurn":
                angle_diff=slot_val % 360
                # print(angle_diff)
                if (angle_diff > 30) and (angle_diff <150):
                    ans=("C. Turn Left")
                elif (angle_diff > 150) and (angle_diff <210):
                    ans=("D. Make U turn")
                elif (angle_diff > 210) and (angle_diff <330):
                    ans=("B. Turn Right")
                elif ((angle_diff <30) or  (angle_diff >330)) :
                    ans=("A. Go Straight")
                else:
                    print(slot_val)
                # print(direction,ans)
                output_content['conversations'].append({'from': 'gpt', 'value': f"D. Make turns at intersection"})
                output_content['conversations'].append({'from': 'human', 'value': "Which direction are you turning to?\n A. Go Straight\n B. Turn Right\n C. Turn Left\n D. Make U turn"})
                output_content['conversations'].append({'from': 'gpt', 'value': ans})
            else: 
                print(act)
            self.actions.append(output_content)

        if (elem['type'] == 'DialogueMove'):
            return
            self.counter+=1
            # if self.interface.car_is_stopped:
            #     return
            # if self.counter%4!=0:
            #     return
            frame = elem['frame']
            if self.split =="train":
                output_content = {"unseen":self.is_hidden,'id': f"{self.id}_{frame}", 'video': f"{self.id}_{frame}.pkl", 'conversations': []}
            else:
                output_content = {"unseen":self.is_hidden,'id': f"{self.id}_{frame}", 'video': f"{self.id}_{frame}.mp4", 'conversations': []}
                if not os.path.exists(f"dorothie_test_videos/{self.id}_{frame}.mp4"):
                    print(f"dorothie_test_videos/{self.id}_{frame}.mp4")
            prompt = self.interface.caption_prompt(frame)
            output_content['conversations'].append({'from': 'human', 'value': f"<video>\n{prompt}"})
            prompt = self.interface.caption(frame)
            output_content['conversations'].append({'from': 'gpt', 'value': prompt})
            prompt = self.interface.plan_prompt(frame)
            output_content['conversations'].append({'from': 'human', 'value': prompt})
            prompt = self.interface.plan_out(frame)
            output_content['conversations'].append({'from': 'gpt', 'value': prompt})
            prompt,direction = self.interface.question(frame)
            output_content['conversations'].append({'from': 'human', 'value': prompt})
            
            
            output_content['conversations'].append({'from': 'gpt', 'value': f"F. Keep on current Lane"})
            output_content['conversations'].append({'from': 'human', 'value': "Are you willing to change the current speed? \n A. Accelerate\n B. Decelerate \n C. Keep current speed"})
            output_content['conversations'].append({'from': 'gpt', 'value': f"C. Keep current speed"})
            self.actions.append(output_content)
            

            # except Exception as e:
                # print(e)

    # Provides the information from a timestep to the model and
    # removes information that the model should not access
    def give_timestep_to_model(self):
        pass

    # Gets the total number of timesteps
    def get_total_timesteps(self):
        return len(self.annotated_log)

    # Finalizes analysis of model over entire log data
    def complete_analysis(self):
        if self.eval_physical_actions:
            slot_type_accuracy = self.physical_action_slot_types_correct / self.total_physical_actions
            slot_val_accuracy = self.physical_action_slot_vals_correct / self.total_physical_actions
            print(f'Total physical actions = {self.total_physical_actions}')
            print(f'Slot type accuracy = {slot_type_accuracy}')
            print(f'Slot val accuracy = {slot_val_accuracy}')


    # Tests the model on one particular log folder
    def test_on_log(self):
        # Get the initial data and provide it to the model
        self.load_initial()

        # Loop through timesteps
        total_timesteps = self.get_total_timesteps()
        while self.cur_timestep < total_timesteps:
            self.process_next_timestep()
        print(len(self.conversations))
        return self.conversations, self.actions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SDN Tester')
    parser.add_argument('--root', type=str, help='Root directory of log files',default="/nfs/turbo/coe-chaijy/owenhji/SDN_final/")
    args = parser.parse_args()
    
    split="train"
    root = os.path.join(args.root,split)
    model=6
    dataset=[]
    for log_fold in os.listdir(root):
        log_path = os.path.join(root,log_fold)
        sdn_tester = SDN_Tester(split,log_path, model, 'cmd')
        conversation, action=sdn_tester.test_on_log()
        dataset+=conversation
        dataset+=action
    print(len(dataset))
    with open(f"datasets/DriVLMe_sft_data.json","w") as f:
        json.dump(dataset,f)

    
    split="test"
    root = os.path.join(args.root,split)
    model=6
    conversations=[]
    actions=[]
    unseen_c,unseen_a=0,0
    seen_c,seen_a=0,0
    for log_fold in os.listdir(root):
        log_path = os.path.join(root,log_fold)
        sdn_tester = SDN_Tester(split,log_path, model, 'cmd')
        conversation, action=sdn_tester.test_on_log()
        if sdn_tester.is_hidden:
            unseen_c+=len(conversation)
            unseen_a+=len(action)
        else:
            seen_c+=len(conversation)
            seen_a+=len(action)
            
        conversations+=conversation
        actions+=action
    print(unseen_c,unseen_a,seen_c,seen_a)
    print(len(conversations),len(actions))
    with open(f"datasets/SDN_{split}_conversations.json","w") as f:
        json.dump(conversations,f)
    with open(f"datasets/SDN_{split}_actions.json","w") as f:
        json.dump(actions,f)