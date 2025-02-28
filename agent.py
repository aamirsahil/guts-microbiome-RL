from AI import AI
import numpy as np

# Agent class
class Agent():
    def __init__(self, food_pop : dict = None, chemicals : dict = None, component_pop : dict = None,
                    kick_resilience : dict = None, kick_satisfaction : float = 100,
                    action_size : int = 5, state_size : int = 5,
                    eat_default_time : float = 10, AI_data : dict = None, 
                    food_consume_rate : int = 1, chemical_decay_rate : float = 0.01) -> None:
        """
        Agent class\n
        
        Parameters:\n
            food_pop = (dict) {"food_1" : (float)amount, ... } How much of each food substance is in gut\n
            chemicals = (dict) {"chemical_1" : (float) amount, ... } How much of each chemical is in gut (act as state)\n
            component_pop = (dict) {"comp_1" : (float)amount, ... } How much of each comp substance is in gut\n
            kick_resilience = (dict) {"chemical_1" : (float) resilience, ... } represents kick resilience towards each chemical\n
            kick_satisfaction = (flaot) represents diminishing return from kick\n
            action_size = (int) possible action size\n
            state_size = (int) possible state size\n
            eat_default_time = (float) fixed eat time\n
            food_consume_rate = (int) chunk of food consumed\n
            chemical_decay_rate = (float) rate of decay of chemical\n
            ai_data = (dict)\n
                {\n
                    "learning_rate" : (float)?, "alpha" : (float)?, "gamma" : (float)?, "exp_rate" : (float)?, "model_loc" : None}\n  
                }\n
        """
        self.food_pop = food_pop
        self.component_pop = component_pop
        self.chemicals = chemicals
        self.kick_resilience = kick_resilience
        self.kick_satisfaction = kick_satisfaction
        self.eat_time_default = eat_default_time
        self.food_consume_rate = food_consume_rate
        self.chemical_decay_rate = chemical_decay_rate
        self.ai = AI(state_size=state_size, action_size=action_size, **AI_data)
        # keep track of when to eat
        self.eat_time = eat_default_time
        # keep track of action
        self.action = 0
        # keep track of reward
        self.reward = 0
        # keep track of current state
        if AI_data["NN_data"]["type"] == "dense":
            self.curr_state = np.zeros((1, state_size))
            self.next_state = np.zeros((1, state_size))
        elif AI_data["NN_data"]["type"] == "lstm":
            self.curr_state = np.zeros((AI_data["NN_data"]["lstm_memory"], state_size))
            self.next_state = np.zeros((AI_data["NN_data"]["lstm_memory"], state_size))

    def _setState(self, chemical_list : list = None, state : np.ndarray = None) -> np.ndarray:
        """
        Convert chemical poulation data to state.\n

        Parameter:\n
            chemical_list = (list) List of all chemical names.
            state = (ndarray) state of system
        """
        for index, chemical in enumerate(chemical_list):
            if chemical in self.chemicals:
                state[0][index] = self.chemicals[chemical]
        state = np.roll(state, -1)
        
        return state

    def setCurrState(self, chemical_list : list = None) -> None:
        """
        set current state.\n

        Parameter:\n
            chemical_list = (list) List of all chemical names.
        """
        self.curr_state = self._setState(chemical_list=chemical_list, state=self.curr_state)

    def setNextState(self, chemical_list : list = None) -> None:
        """
        set next state.\n

        Parameter:\n
            chemical_list = (list) List of all chemical names.
        """
        self.next_state = self._setState(chemical_list=chemical_list, state=self.curr_state)
    
    def getAction(self) -> int:
        """
        Agent takes best action for current state.\n

        Parameter:\n
            state = (list) Current state of agent.\n
        
        Return:\n
            action = (int) index of action taken by the agent
        """
        self.action = self.ai.decide(self.curr_state)
        return self.action
    
    def _setComponentPop(self, components : dict = None) -> None:
        """
        Update component population\n

        Parameter:\n
            components = (dict) {"comp_1" : (float) ratio, ... ] Components in food consumed.\n
        """
        for component, ratio in components.items():
            self.component_pop[component] = self.component_pop.get(component, 0) + self.food_consume_rate * ratio

    def eat(self, food : str = None, food_dict : dict = None) -> None:
        """
        Agent eats at set intervals, food pop increase at a rate\n

        Parameter:\n
            food = (str) food currently consumed
            food_dict = (dict) {"comp_1" : (flaot) ratio, ... } Food compponent makeup.\n
        """
        self.food_pop[food] = self.food_pop.get(food, 0) + self.food_consume_rate
        self._setComponentPop(components = food_dict)
        
        # reset the eat time
        self.eat_time = self.eat_time_default

    def decayChemicals(self) -> None:
        """
        After every step chemical decays
        """
        for chemical in self.chemicals:
            self.chemicals[chemical] *= self.chemical_decay_rate
    
    def setChemicals(self, chemicals_produced : dict = None) -> None:
        """
        Set chemical population after bacteria produces it.

        Parameter:\n
            chemical_produced = (dict) {"chemical_1" : (float) amount, ... }
        """
        for chemical in chemicals_produced:
            self.chemicals[chemical] = self.chemicals.get(chemical, 0) + chemicals_produced[chemical]

    def findReward(self, chemical_list : list = None, chemical_profile : dict = None):
        """
        """
        # reward depends on change in chemical
        chemical_before = self._stateToChem(state=self.curr_state[0], chemical_list=chemical_list)
        chemical_after = self._stateToChem(state=self.next_state[0], chemical_list=chemical_list)
        # calculates reward
        reward = 0
        for chemical in chemical_list:
            chemical_diff_factor = (chemical_after.get(chemical, 0) - chemical_before.get(chemical, 0))/self.kick_resilience[chemical]
            kick = chemical_profile[chemical]["kick"]
            reward += (kick*chemical_diff_factor)
        # set total reward
        self.reward = reward
    
    def _stateToChem(self, state : np.ndarray = None, chemical_list : list = None) -> dict:
        """
        Find chemicals in gut when state is given.

        Parameters:\n
            state : (np.ndarray) state of the agent\n
            chemical_list : (list) list of chemicals\n
        Retrn:\n
            chemical_dict = (dict) {"chem_1" : (float) amount, ...}
        """
        chemical_dict = {}
        for idx, chemical in enumerate(chemical_list):
            chemical_dict[chemical] = state[idx]
        return chemical_dict