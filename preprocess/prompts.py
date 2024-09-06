import string

header = '''
You are ChauffeurGPT. You are responsible for safely piloting a car according to the instructions of a passenger.
You must communicate with the passenger and make high-level decisions regarding the current navigational goals.

'''

only_mc = '''
Please ONLY output the letter corresponding to your choice.
Do not output any text except for the letter corresponding to your choice
'''

#############################################################################################
#################################### High-level prompts #####################################
#############################################################################################

high_level_current_goal = '''
The current goal is {goal}
'''

high_level_decisions = '''
You can output one of the following choices:
(A) SET NEW GOAL: change the existing goal
(B) SPEAK: ouput dialogue for the passenger
'''

def print_dialogue(llm_interface):
    prompt = "DIALOGUE HISTORY:\n"
    for utterance in llm_interface.dialogue_history[-20:]:
        prompt += utterance
        prompt += "\n"

    return prompt

def get_high_level_prompt(llm_interface):
    prompt = header
    prompt += print_dialogue(llm_interface)
    prompt += high_level_current_goal.format(goal=llm_interface.current_goal)
    prompt += high_level_decisions
    prompt += only_mc

    return prompt

def print_map(llm_interface, frame):
    prompt = "CURRENT MAP:\n"
    prompt += llm_interface.town_map.load_map_at_frame(frame)
    prompt += '\n'
    prompt += llm_interface.town_map.get_vehicle_string()
    prompt += '\nLandmark labels:\n'
    prompt += llm_interface.town_map.get_landmark_string()
    prompt += '\nStreet names:\n'
    prompt += llm_interface.town_map.get_streetname_string()

    return prompt


#############################################################################################
############################## Low-level goal-setting prompts ###############################
#############################################################################################

low_level_goal_setting = '''
You have been tasked with choosing the new navigational goal.
The vehicle will take the fastest route to this navigational goal.

The current goal is {goal}

You can choose a new goal from among these options:
'''

goal_choice = "({letter}) {potential_goal}\n"

def get_low_level_goal_setting(llm_interface):
    prompt = header
    prompt += print_dialogue(llm_interface)
    prompt += low_level_goal_setting.format(goal=llm_interface.current_goal)
    goals = {}
    for i, asset in enumerate(llm_interface.assets):
        asset_name = asset['type']
        letter = string.ascii_uppercase[i]
        prompt += goal_choice.format(letter=letter, potential_goal=asset_name)
        goals[letter] = asset_name

    prompt += only_mc

    return prompt, goals

#############################################################################################
################################ Low-level dialogue prompts #################################
#############################################################################################

low_level_dialogue = '''
You have been tasked with communicating with the human passenger.
Please take this opportunity to provide the passenger with information
or to ask them a question.
'''

def get_low_level_dialogue(llm_interface):
    prompt = header
    prompt += low_level_dialogue

    return prompt

#############################################################################################
################################## Physical action prompts ##################################
#############################################################################################

low_level_phys_action_stopped = '''
You have been tasked with choosing the next physical action.
Your car is currently stopped, you must choose to start.
You can choose a new navigaional action from among these options:
'''

low_level_nav_action_started = '''
You have been tasked with choosing the next physical action.
Your current speed is {car_speed} kph, you shouldn't choose start now.
You must make a full stop at each intersection.
You can choose a new navigaional action from among these options:
'''

low_level_speed_action='''
You can choose a new speed action from among these options:
'''
caption_q='''
Describe what you see:
'''
obstacle_prompt='''
There is a obstacle {type} in front of me, the distance is {dist}
'''
lan_num_prompt='''
I'm on the {num} lane from the left of the road.
'''
lan_change_prompt=["I'm not able to change lane.","I'm only able to change to right lane.", "I'm only able to change to left lane.", "I'm able to change to both right and left lane"]

dist_far_prompt='''
I am far from the end of the road. I don't need to make a decision for turning now.
'''
dist_near_prompt='''
I am near the end of the road. I don't need to make a decision for turning now.
'''
dist_at_prompt='''
I am at the end of the road, I need to stop if there is a red light, or make a decision to turn left, turn right or go straight now.
'''
traffic_ligh_prompt='''
There is a traffic light {dist} meters from me, showing {state}.
'''
sign_prompt='''
There is a {name} {dist} meters from me, showing {state}.
'''

action_choice = "({letter}) {potential_action}\n"

actions = ["Start", "Stop","Change lane",  "Make turns at intersection", "Make U turn","Change Speed"]
sign_state = ["Red","Yellow","Green", "Off", "Unknown"]

def caption_prompt(llm_interface, frame):

    prompt = header
    prompt += '\n'
    prompt += caption_q
    
    return prompt

def print_plan(llm_interface,frame):
    prompt = 'Current plan for the next intersections:\n'
    if llm_interface.turn_plan[str(frame)][1]==None:
        return "",None
    else:
        direction = llm_interface.turn_plan[str(frame)][0][0]
        for plan in llm_interface.turn_plan[str(frame)][0]:
            prompt += plan + '\n'
    # return "",None
    return prompt,direction

def get_header(llm_interface, frame):
    prompt = print_dialogue(llm_interface)
    prompt += '\n'
    prompt = print_action_history(llm_interface)
    prompt += '\n'
    
    return prompt
  
plan_q_prompt='''
You have a planning tool that you can plan your path to the destination. You can call it by plan(dest), and it will return you a plan to get to your destination. If you don't have a destination in your mind, you can return plan(None). Your last goal was: {last_goal}
'''  
plan_q_prompt_without='''
You have a planning tool that you can plan your path to the destination. You can call it by plan(dest), and it will return you a plan to get to your destination. If you don't have a destination in your mind, you can return plan(None). 
'''  
def plan_prompt(llm_interface, frame):
    prompt = ""
    prompt += header
    prompt += '\n'
    prompt += get_header(llm_interface, frame)
    prompt +=plan_q_prompt_without
    return prompt

def plan_out(llm_interface, frame):
    if (llm_interface.turn_plan[str(frame)][1]!=None):
        # print(llm_interface.turn_plan[str(frame)])
        name = llm_interface.turn_plan[str(frame)][1][0]
        if (name.split("_")[0] in llm_interface.assets):
            llm_interface.last_plan = f'{llm_interface.assets[(name.split("_")[0])]}'
            return f'plan({llm_interface.assets[(name.split("_")[0])]})'
    return 'plan(None)'
    
    
def get_caption(llm_interface, frame):
    data=llm_interface.additional["data"][str(frame)]
    lan_num = abs(data["lane"][0])
    
    if int(data["dist"])>10:
        prompt=dist_far_prompt
    elif int(data["dist"])>5:
        prompt=dist_near_prompt
    else:
        prompt = dist_at_prompt
    prompt += '\n'
    prompt += lan_change_prompt[data["lane"][1]]
    prompt += '\n'
    prompt += lan_num_prompt.format(num = lan_num)
    prompt += '\n'
    
    for sign in data["lights"]:
        if "state" in sign:
            prompt += traffic_ligh_prompt.format(dist = int(sign["dist"]), state=sign_state[sign["state"]])
        else:
            prompt += sign_prompt.format(name = sign["name"],dist = int(sign["dist"]), state=sign["value"])
        prompt += '\n'
            
    if frame in llm_interface.additional["object"]:
        obstacle = llm_interface.additional["object"][str(frame)]
        prompt += obstacle_prompt.format(type = obstacle["type"],dist = int(obstacle["dist"]))
        prompt += '\n'
    # print(prompt)
    return prompt
    
def dialogue_question(llm_interface, frame):
    prompt, _ = print_plan(llm_interface, frame)
    prompt+="\n"
    prompt+="Please fisrt predict the type of output conversation and then reply to the PASSENGER:"
    return prompt
    
def dialogue_ans(llm_interface, frame,elem):
    prompt = f'type: {elem["act"]["move"]} \n output:'
    prompt += elem['utterance_gt']
    return prompt
    
    
def get_action_question(llm_interface, frame):
    stopped=False
    prompt,direction = print_plan(llm_interface, frame)
    if llm_interface.car_is_stopped:
        speed_actions = actions
        prompt += low_level_phys_action_stopped
    else:
        stopped=True
        speed_actions = actions
        prompt += low_level_nav_action_started.format(car_speed=llm_interface.car_speed)
    
    for i, action in enumerate(actions):
        letter = string.ascii_uppercase[i]
        prompt += action_choice.format(letter=letter, potential_action=action)
        
    # prompt += low_level_speed_action
    # for i, action in enumerate(speed_actions):
    #     letter = string.ascii_uppercase[i]
    #     prompt += action_choice.format(letter=letter, potential_action=action)

    return prompt,direction

def get_followup_jturn(llm_interface):
    prompt = 'You chose to make a regular turn.\n'
    yaw = llm_interface.town_map.yaw
    prompt += f'You are currently facing at an angle of {yaw} degrees. '
    prompt += 'In this context, 0 degrees would be facing directly right. '
    prompt += 'A positive angle 0 < x < 180 would be facing up.\n'
    prompt += 'What angle would you like to add to this angle?\n'
    prompt += 'Provide ONLY an integer value between -180 and 180\n'
    prompt += 'Do NOT output anything except for this number.\n'

    return prompt

def get_followup_speedchange(llm_interface):
    prompt = 'You chose to make a speed change.\n'
    speed = llm_interface.car_speed
    prompt += f'You are currently traveling at {speed} kph.\n'
    prompt += 'Which of the following would you like to do?\n'
    prompt += '(A) Increase your speed\n'
    prompt += '(B) Decrease your speed\n'

    prompt += only_mc

    return prompt



def print_action_history(llm_interface):
    prompt = 'PHYSICAL ACTION HISTORY:\n'
    for action in llm_interface.phys_action_history[-10:]:
        prompt += action + '\n'
    return prompt

#############################################################################################
###################################### Other prompts ########################################
#############################################################################################

# Lower-level prompts
navigation_block = '''For navigational purposes, you have access to a planner function that you may call in the following manner:
```Python
loc1 = Landmark('Home') # this is the location of 'home'
loc2 = Intersection('Broadway', 'Baits') # this is the intersection between two roads named 'Broadway' and 'Baits'
loc3 = StreetSegment('Broadway', loc1, loc2) # this is the location of a street segment between loc1 and loc2
Planner(loc1, [loc2, loc3]) # this planning function maps a route from the current position to loc1 while avoiding loc2 and loc3
```
Here is a sample output of the planner function. It lists all intersections and landmarks that will be passed along the way:
```Python
Planner(loc1, [])
>>> [['uturn'], ['straight', Intersection('Baits', 'Hubbard')], ['right'], ['straight', Landmark('Home')]]
```
'''

dialogue_history_header = '''The following is the history, which contains dialogue between yourself (ChaufferGPT) and the passenger. 
It also contains records you have made regarding road conditions:'''

sample_dialogue_history = '''
PASSENGER: can you take me to Arbys?
CHAUFFEURGPT: yes
PASSENGER: actually nevermind. can you take me to Wendys?
CHAUFFEURGPT: no worries. yes, I can
NOTE: there is construction blocking the intersection of Hayward and Baits
PASSENGER: actually, I changed my mind again. can you take me to Arbys?
'''

other_stuff = '''
This is the plan that is currently saved into your navigational history:
```Python
goal = Landmark('Arbys')
Planner(goal, [])
>>> [['straight', Intersection('Fuller', 'Hubbard')], ['left'], ['straight', Landmark('Arbys')]]
```

Here is your current location:
```Python
loc1 = Intersection('Fuller', 'Draper')
loc2 = Intersection('Fuller', 'Hubbard')
current_location = StreetSegment('Fuller', loc1, loc2)
facing_towards = loc2
```

Here is a description of your current visual input:
There is a road in front of you with an ambulance that is blocking both lanes.

You must now produce an ouput of the following form. You do not need to output all output types. You do not need to output them in any specific order either.
OUTPUTTYPE1
Your outputs for output type 1
OUTPUTTYPE2
Your outputs for output type 2
'''