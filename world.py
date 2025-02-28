import numpy as np
from agent import Agent
from bacteria import Bacteria
import csv

# world class
class World:
    def __init__(self, no_bacteria : int = 5, no_components : int = 5,
                 no_food : int = 5, no_chemical : int = 5,
                 component_list : list = None, food_list : list = None, chemical_list : list = None,
                 food_profile : dict = None, chemical_profile : dict = None,
                 bacteria_data : list = None, agent_data : dict = None) -> None:
        """
        World class\n
        
        Parameters:\n
            no_bacteria = (int) Total varity of bacterias\n
            no_components = (int) Total varity of components\n
            no_food = (int) Total varity of food\n
            no_chemical = (int) Total varity of chemical\n
            food_list = (list) ["food_1", ... ] List of food names\n
            component_list = (list) ["comp_1", ... ] List of component names\n
            chemical_list = (list) ["chem_1", ... ] List of chemical names\n
            food_profile = (dict) {"food_1" : (dict) {"comp_1" : (float) ratio, ...}, ... } Profile of food\n
            chemical_profile = (dict) {"chemical_1" : (dict) {"kick" : (float) value, ...}, ...} Profile of chemical\n
            bacteria_data = (list) [(dict) {"idx" : (int), ...}, ...] Bacteria initial data\n
            agent_data = (dict) {"food_pop" : (dict) {}, ...} Agent initial data\n
        """
        self.time = 0
        # number of each elements
        self.no_bacteria = no_bacteria
        self.no_components = no_components
        self.no_food = no_food
        self.no_chemical = no_chemical
        # list of these elements for easy reference
        self.component_list = component_list
        self.food_list = food_list
        self.chemical_list = chemical_list
        # profile of these elements
        self.food_profile = food_profile
        self.chemical_profile = chemical_profile
        # define bacteria colonies and agent
        self.bacterias = [ Bacteria(**bacteria_data[i]) for i in range(self.no_bacteria) ]        
        self.agent = Agent(**agent_data)
        self.reward = 0
    
    def step(self, filename : str = None) -> None:
        """
        Function inside the main simulation loop.
        """
        # log the data
        self._log(filename)
        # set the state of the agent
        self.agent.setCurrState(chemical_list=self.chemical_list)
        # sort bacteria action according to speed value
        self._sortBacteria()
        # Agent eats when eat_time becomes 0
        if self.agent.eat_time == 0:
            action = self.agentAct()
            # bacteria acts
            self.bacteriaAct()
            self.agent.setNextState(chemical_list=self.chemical_list)
            self.agentTrain(action)
        else:
            # Bacteria acts
            self.bacteriaAct()
            self.agent.setNextState(chemical_list=self.chemical_list)
        # chemical decays
        self.agent.decayChemicals()
        # time moves forward
        self.time += 1
        self.agent.eat_time -= 1
    
    def agentAct(self) -> int:
        # state in gut before taking action
        action = self.agent.getAction()
        food, food_dict = self._actionToFood(action=action)
        self.agent.eat(food=food, food_dict=food_dict)

        return action
    
    def agentTrain(self, action):
        # state and food in gut after taking action
        self.agent.findReward(chemical_list=self.chemical_list, chemical_profile=self.chemical_profile)
        self.agent.ai.train(reward=self.agent.reward, state=self.agent.curr_state, next_state=self.agent.next_state, action=action)

    def bacteriaTrain(self, bacteria : Bacteria, action : int) -> None:
        # state and food in gut after taking action
        bacteria.findReward(component_list=self.component_list)
        bacteria.ai.train(reward=bacteria.reward, state=bacteria.curr_state, next_state=bacteria.next_state, action=action)

    def bacteriaAct(self):
        # current total number of bacterias
        total_bacteria_pop = sum([bacteria.pop for bacteria in self.bacterias])
        # each bacteria act according to its speed
        for bacteria in self.bacterias:
            bacteria.setCurrState(component_list=self.component_list, component_pop=self.agent.component_pop)
            # bacteria grow/decay accordance with food availablity
            self.agent.component_pop = bacteria.growth(comp_pop=self.agent.component_pop, total_bacteria_pop=total_bacteria_pop)
            # bacteria produce chemicals and agent updates its chem pop
            chemical_produced = bacteria.produceChemicals()
            self.agent.setChemicals(chemicals_produced=chemical_produced)
            # increase or decrease speed
            bacteria.recatToChemical(self.agent.chemicals)
            # bacteria learns
            bacteria.setNextState(component_list=self.component_list, component_pop=self.agent.component_pop)
            self.bacteriaTrain(bacteria=bacteria, action=bacteria.action)
    
    def _sortBacteria(self) -> None:
        """
        Sort bacteria list as according to there speed value
        """
        self.bacterias.sort(reverse=True, key=lambda x: x.speed)

    def _chemToState(self, chemicals):
        index = 0
        state = np.zeros((1, self.no_chemical))
        for chemical in self.chemical_list:
            if chemical in chemicals:
                state[0][index] = chemicals[chemical]
            index += 1
        # state[0] = self._normalize(state[0])
        return state

    def _actionToFood(self, action : int = 0) -> tuple:
        """
        Returns the foods profile of corresponding to actions.
        
        Paramter:\n
            action = (int) action taken.
        Return:\n
            food_dict = (dict) {"comp_1" : (float) ratio, ... } component makeup.
        """
        food = self.food_list[action]
        food_dict = self.food_profile[food]

        return food, food_dict
    
    def _signOf(self, num):
        if num > 0:
            return +1
        elif num < 0:
            return -1
        else: return 0

    def _normalize(self, state):
        state = np.exp(state)/np.sum(np.exp(state))
        return state

    def _setPlotData(self, plot_time, plot_bacteria_pop, plot_component_pop, plot_food_pop, plot_chemical_pop, plot_reward):
        plot_time.append(self.time)
        
        for i in range(self.no_bacteria):
            plot_bacteria_pop[i].append(self.bacterias[i].pop)
        for i, comp in enumerate(self.component_list):
            plot_component_pop[i].append(self.agent.component_pop.get(comp, 0))
        for i, chem in enumerate(self.chemical_list):
            plot_chemical_pop[i].append(self.agent.chemicals.get(chem, 0))
        plot_food_pop.append(self.agent.action)
        plot_reward.append(self.reward)
    
    def _log(self, filename):
        data = []
        print("############################################")
        print(f"Time : {self.time}")
        data.append(self.time)
        print("Bacterias:")
        for i in range(self.no_bacteria):
            print("------------------------------")
            print(f"id : {self.bacterias[i].id}")
            print(f"population : {self.bacterias[i].pop}")
            print(f"speed : {self.bacterias[i].speed}")
            print(f"reward : {self.bacterias[i].reward}")
            data.append(self.bacterias[i].id)
            data.append(self.bacterias[i].pop)
            data.append(self.bacterias[i].speed)
            data.append(self.bacterias[i].reward)
        print("------------------------------")
        print("Agent:")
        print(f"Component in guts :{self.agent.component_pop}")
        print(f"Chemicals in system: {self.agent.chemicals}")
        # print(f"State of system: {self.agent.chemicals}")
        print(f"Time till eat time : {self.agent.eat_time}")
        print(f"Exploration : {self.agent.ai.exploration}")
        print(f"Reward : {self.agent.reward}")
        for comp in self.component_list:
            data.append(self.agent.component_pop.get(comp, 0))
        for chem in self.chemical_list:
            data.append(self.agent.chemicals.get(chem, 0))
        data.append(self.agent.eat_time)
        data.append(self.agent.ai.exploration)
        data.append(self.agent.reward)
        print("############################################")
            
        # writing to csv file 
        with open(filename, 'a') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile)
                
            # writing the fields
            csvwriter.writerow(data)