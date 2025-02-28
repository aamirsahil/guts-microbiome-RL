from AI import AI
import numpy as np

# Bacteria class
class Bacteria:
    def __init__(self, id : int = 0, pop : float = 100,
                 comp_like : dict = None, chemical_produce : list = None, chemical_inducer : dict = None,
                 chemical_produce_rate : float = 1, eat_rate : float = 0.01, speed : float = 1,
                 state_size : int = 5, action_size : int = 5, increased_production_rate : float = 1, AI_data : dict = None) -> None:
        """
        Bacteria class.

        Parameters:\n
            id = (int) unique tag for the bacteria\n
            pop = (float) Population of the bacteria\n
            comp_like = (dict) {"comp_1" : (float) like_rate, ... } Component liked by the bacteria\n
            chemical_produce = (list) chemicals produced by the bacteria\n
            chemical_inducer = (dict) {"chem_1" : (float) effect, ... } chemicals that affect in bacteria growth\n
            chemicals_produce_rate = (float) Rate at which chemicals are produced(in case of no ai)\n
            eat_rate = (float) Rate at which food is consumed.\n
            speed = (float) Food access speed\n
            state_size = (int)
            action_size = (int)
            AI_data = (dict)\n
                {\n
                    "learning_rate" : (float)?, "alpha" : (float)?, "gamma" : (float)?, "exp_rate" : (float)?, "model_loc" : None}\n  
                }\n
        """
        self.id = id
        self.pop = pop
        self.pre_pop = 0
        self.comp_like = comp_like
        self.chemical_produce = chemical_produce
        self.chemical_inducer = chemical_inducer
        self.chemical_produce_rate = chemical_produce_rate
        # growth parameters(logarithmic model)
        self.eat_rate = eat_rate
        # chemical induced speed
        self.speed = speed
        # ai parameters
        if AI_data == None:
            self.ai = None
        else:
            self.ai = AI(state_size=state_size, action_size=action_size, **AI_data)
        # keep track of action
        self.action = 0
        self.increased_chemical_production_rate = increased_production_rate
        # keep track of reward
        self.reward = 0
        # keep track of current state
        if AI_data["NN_data"]["type"] == "dense":
            self.curr_state = np.zeros((1, state_size))
            self.next_state = np.zeros((1, state_size))
        elif AI_data["NN_data"]["type"] == "lstm":
            self.curr_state = np.zeros((AI_data["NN_data"]["lstm_memory"], state_size))
            self.next_state = np.zeros((AI_data["NN_data"]["lstm_memory"], state_size))

    def growth(self, comp_pop, total_bacteria_pop) -> dict:
        """
        Bacteria population model:\n
        After every time step bacteria population changes according to logarithmic model.

        Parameter:\n
            comp_pop : (dict) {"comp_1" : pop}

        Return:\n
            comp_pop : (dict) {"comp_1" : pop} reducing the food conumer anount.
        """

        # the size of the colony determine access to food 
        access_to_food = self.pop/total_bacteria_pop
        
        # total_amount eatable food -> initialized to 0.01 to showcase potential for growth
        total_food = 0.01
        for comp, rate in self.comp_like.items():
            if comp in comp_pop:
                total_food += comp_pop[comp]*rate
                comp_pop[comp] -= self.pop*self.eat_rate
                comp_pop[comp] = max(comp_pop[comp], 0)
    
        # total number of bacteria the current food can accomodate
        capacity = access_to_food*total_food/self.eat_rate

        # change the populatio of the bacteria based on log growth
        change_pop = self.eat_rate*self.pop*(1 - self.pop/capacity)
        
        self.pre_pop = self.pop
        self.pop += change_pop
        # set to minimum of 0.01 to show that bacteria is never completely gone
        self.pop = max(self.pop, 0.01)

        return comp_pop

    def _setState(self, component_list : list = None, component_pop : dict = None, state : np.ndarray = None) -> np.ndarray:
        """
        Convert chemical poulation data to state.\n

        Parameter:\n
            chemical_list = (list) List of all chemical names.
            state = (ndarray) state of system
        """
        for index, component in enumerate(component_list):
            if component in component_pop:
                state[0][index] = component_pop[component]
        state = np.roll(state, -1)
        
        return state

    def setCurrState(self, component_list : list = None, component_pop : dict = None) -> None:
        """
        set current state.\n

        Parameter:\n
            chemical_list = (list) List of all chemical names.
        """
        self.curr_state = self._setState(component_list=component_list, component_pop=component_pop, state=self.curr_state)

    def setNextState(self, component_list : list = None, component_pop : dict = None) -> None:
        """
        set next state.\n

        Parameter:\n
            chemical_list = (list) List of all chemical names.
        """
        self.next_state = self._setState(component_list=component_list, component_pop=component_pop, state=self.next_state)

    def recatToChemical(self, chemicals) -> None:
        """
        Bacteria speed changes based on inducer chemicals.

        Parameter:\n
            chemicals : (dict) {"chemical" : (float)amount, ... } Chemical distribution in the system currently.
        """
        for chemical, value in self.chemical_inducer.items():
            self.speed += chemicals.get(chemical, 0)*value
    
    def produceChemicals(self) -> dict:
        """
        After every time step bacteria produce chemicals(based on RL or not)\n
        
        Return:\n
             = (dict) {"chemical" : (float), ...} Chemicals produced by the bacteria\n
        """
        production_rate = np.ones(len(self.chemical_produce))*self.chemical_produce_rate
        if self.ai != None:
            self.action = self.ai.decide(self.curr_state)
            production_rate[self.action] += self.increased_chemical_production_rate
        
        # keep track of chemicals produced
        chemicals = {}
        for chemical, rate in zip(self.chemical_produce, production_rate):
            chemicals[chemical] = self.pop * rate
        return chemicals
    
    def findReward(self, component_list : list = None) -> None:
        """
        """
        # calculates reward
        reward = self.pop - self.pre_pop
        # set total reward
        self.reward = reward