import prompts
import openai
import string

verbose = True

class LLMInterface:

    def __init__(self, assets, turn_plan,additional, town_map, llm='cmd'):
        self.llm = llm # By default, queries user input through CMD rather than an actual LLM
        self.requires_evalutation = True # if True, LLM needs to be queried at next time step
        self.new_goal = False
        self.assets = assets

        # State data
        self.additional=additional
        self.turn_plan=turn_plan
        self.current_goal = None
        self.dialogue_history = [] # List of all dialogue history
        self.phys_action_history = []
        self.town_map = town_map
        self.car_is_stopped = True
        self.car_speed = 25
        self.last_plan=None
        
        
    '''
        Prompts the LLM to adjust to an environmental change
    '''
    def evaluate(self):
        if self.requires_evalutation:
            result = self.prompt_high_level()
            letter = parse_mc_response(result)
            if letter == 'A' or letter == 'B':
                self.goal_setting_prompt()
                self.new_goal = True
            #elif letter == 'B':
                #self.dialogue_prompt()
            else:
                raise Exception(result)

            self.requires_evalutation = False

    '''
        Compiles the high-level prompt
    '''
    def prompt_high_level(self):
        prompt = prompts.get_high_level_prompt(self)

        return self.comm_w_llm(prompt)

    def goal_setting_prompt(self):
        prompt, goals = prompts.get_low_level_goal_setting(self)
        response = self.comm_w_llm(prompt)

        letter = parse_mc_response(response)

        self.current_goal = goals[letter]

        return response
    
    def caption_prompt(self, frame):
        prompt = prompts.caption_prompt(self, frame)
        return prompt
    def plan_prompt(self, frame):
        prompt = prompts.plan_prompt(self, frame)
        return prompt
    def plan_out(self, frame):
        prompt = prompts.plan_out(self, frame)
        return prompt

    def physical_action_prompt(self, frame):
        prompt = prompts.get_low_level_phys_action(self, frame)
        return prompt

    def caption(self, frame):
        prompt = prompts.get_caption(self, frame)
        return prompt
    def question(self, frame):
        prompt,direction = prompts.get_action_question(self, frame)
        return prompt,direction
    def dialogue_question(self, frame):
        prompt = prompts.dialogue_question(self, frame)
        return prompt
    def dialogue_ans(self, frame,elem):
        prompt = prompts.dialogue_ans(self, frame,elem)
        return prompt

    def speedchange_slot_val(self):
        prompt = prompts.get_followup_speedchange(self)
        return prompt

    def jturn_slot_val(self):
        prompt = prompts.get_followup_jturn(self)
        return prompt

    def dialogue_prompt(self):
        prompt = prompts.get_low_level_dialogue(self)
        response = self.comm_w_llm(prompt)
        formatted_response = 'CHAUFFEURGPT: {response}'.format(response=response)
        self.dialogue_history.append(formatted_response)

        return response

    def inject_llm_dialogue(self, dialogue):
        formatted_response = 'CHAUFFEURGPT: {dialogue}'.format(dialogue=dialogue)
        self.dialogue_history.append(formatted_response)

    def inject_physical_action(self, elem):
        frame = elem['frame']
        act = elem['val']['act']
        slot_type = elem['val']['slot_type']
        slot_val = elem['val']['slot_val']
        if act == 'SpeedChange':
            if slot_val == 5:
                ans="Accelerate"
                self.car_speed+=5
            else:
                ans='Reduce Speed'
                self.car_speed-=5
            
            self.phys_action_history.append( f"At frame {frame}: {ans}")
        elif act == 'Start':
            self.phys_action_history.append( f"At frame {frame}: Start")
            self.car_is_stopped = False
        elif act == 'Stop':
            self.phys_action_history.append(f"At frame {frame}: Stop")
            self.car_is_stopped = True
        elif act == 'UTurn':
            self.phys_action_history.append(f"At frame {frame}: Make U turn")
        elif act == 'LaneSwitch':
            if slot_val == 90:
                ans = "Change to right lane"
            else:
                ans = "Change to left lane"
            self.phys_action_history.append(f"At frame {frame}: {ans}")
        elif act == "JTurn":
            angle_diff=slot_val
            if (angle_diff > 30) and (angle_diff <150):
                ans=("Turn Right")
            elif (angle_diff > 150) and (angle_diff <210):
                ans=("Make U turn")
            elif (angle_diff > 210) and (angle_diff <330):
                ans=("Turn Left")
            elif ((angle_diff <30) or  (angle_diff >330)) :
                ans=("Keep Lane")
            else:
                print(slot_val)
            self.phys_action_history.append(f"At frame {frame}: {ans}")
        else:
            print(act)



    '''
        This function actually sends the already-created prompt
        to the LLM and receives the output
    '''
    def comm_w_llm(self, prompt, response=None, second_prompt=None):
        if self.llm == 'cmd':
            return receive_cmd_input(prompt, response=response, second_prompt=second_prompt)
        elif self.llm == 'gpt4':
            return receive_gpt4_input(prompt, response=response, second_prompt=second_prompt)
        elif self.llm == 'always_a':
            print(prompt)
            return 'D'
        else:
            raise Exception('Code not compatible with the LLM {llm} yet'.format(llm=self.llm))

    '''
        Adds string from the human representing a new communication
    '''
    def receive_dialogue(self, dialogue):
        formatted_dialogue = 'PASSENGER: {dialogue}'.format(dialogue=dialogue)
        self.dialogue_history.append(formatted_dialogue)
        self.requires_evalutation = True

def receive_cmd_input(prompt, response=None, second_prompt=None):
    print('------------------------------------------------------------------------')
    if response is None:
        print(prompt)
    else:
        print(second_prompt)
    print('------------------------------------------------------------------------')
    val = input("Input: ")
    return val


def receive_gpt4_input(prompt, response=None, second_prompt=None):
    if response is None:
        messages=[{"role": "user", "content": prompt}]
    else:
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
            {"role": "user", "content": second_prompt}
        ]

    gpt_response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=messages,
        temperature = 0
    )

    response_text = gpt_response['choices'][0]['message']['content']

    if verbose:
        print('------------------------------------------------------------------------')
        print(prompt)
        print('GPT4_RESPONSE:')
        print(response_text)
        print('------------------------------------------------------------------------')

    return response_text

def parse_mc_response(response):
    for letter in string.ascii_uppercase:
        if letter in response:
            return letter
    
    raise Exception('LLM did not return a valid character')