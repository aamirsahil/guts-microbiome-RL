from world import World
from tqdm import tqdm
import pandas as pd
from plot import plotFig
import csv

def start(data, total_time, filename):
    # plot data
    world = World(**data)
    createLogFile(filename)
    
    for time in range(total_time):
        world.step(filename)

def createLogFile(filename):
    # field names 
    fields = ['Time', 'B1_id', 'B1_pop', 'B1_speed', 'B1_reward',
               'B2_id', 'B2_pop', 'B2_speed', 'B2_reward',
               'comp1_pop', 'comp2_pop', 'comp3_pop', 'comp4_pop',
               'chem1_pop', 'chem2_pop',
               'time_till_eat', 'exploration', 'reward']
        
    # writing to csv file 
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile)
            
        # writing the fields
        csvwriter.writerow(fields)

# main function
def main():
    # system
    bacteria_num = 2
    component_num = 4
    food_num = 5
    chemical_num = 5
    
    action_size = food_num
    state_size = chemical_num

    food_list = [f"food-{(i+1):02d}" for i in range(food_num)]
    chemical_list = [f"chem-{(i+1):02d}" for i in range(chemical_num)]
    component_list = [f"comp-{(i+1):02d}" for i in range(component_num)]

    data = {
        "no_bacteria" : bacteria_num,
        "no_components" : component_num,
        "no_food" : food_num,
        "no_chemical" : chemical_num,

        # How much craving for a particular component this chemical adds to
        "chemical_profile" : {
            "chem-01" : {"kick" : 10.0},
            "chem-02" : {"kick" : 3.0},
            "chem-03" : {"kick" : 1.0},
            "chem-04" : {"kick" : 1.0},
            "chem-05" : {"kick" : 1.0},
        },
        # The components that make up the food
        "food_profile" : {
            "food-01": {"comp-01": 100, "comp-02": 0, "comp-03": 0, "comp-04" : 0},
            "food-02": {"comp-01": 0, "comp-02": 100, "comp-03": 0, "comp-04" : 0},
            "food-03": {"comp-01": 0, "comp-02": 0, "comp-03": 100, "comp-04" : 0},
            "food-04": {"comp-01": 0, "comp-02": 0, "comp-03": 0, "comp-04" : 100},
            "food-05": {"comp-01": 0, "comp-02": 0, "comp-03": 50, "comp-04" : 50},
        },

        "food_list" : food_list,
        "chemical_list" : chemical_list,
        "component_list" : component_list,

        # Data on what each bacteria eats and produces
        "bacteria_data" : [
            {"id" : 1, "pop" : 100, "comp_like" : {"comp-01" : 1.0}, "chemical_produce" : ["chem-01", "chem-02"], "chemical_inducer" : {},
             "state_size" : component_num, "action_size" : 2,
             "AI_data" : {"learning_rate" : 0.01, "alpha" : 0.1, "gamma" : 0.95, "exploration_decay" : 0.99, "model_loc" : None, "NN_data" : {"type" : "dense"}}},
            {"id" : 2, "pop" : 100, "comp_like" : {"comp-02" : 1.0}, "chemical_produce" : ["chem-02"], "chemical_inducer" : {"chem-01" : -1.0},
             "state_size" : component_num, "action_size" : 1,
             "AI_data" : {"learning_rate" : 0.01, "alpha" : 0.1, "gamma" : 0.95, "exploration_decay" : 0.99, "model_loc" : None, "NN_data" : {"type" : "dense"}}},
        ],
        # Initial state of agent and kick(decrease in craving) it gets from 
        "agent_data" : 
            {
            "food_pop" : {}, "component_pop" : {}, "chemicals" : {},
            "kick_resilience" : {"chem-01": 10, "chem-02": 50, "chem-03": 200, "chem-04" : 10, "chem-05" : 500, "chem-06": 10, "chem-07": 50, "chem-08": 200, "chem-09" : 10, "chem-10" : 500},
            "kick_satisfaction" : 50,
            "state_size" : state_size, "action_size" : action_size,
            "eat_default_time" : 10,
            "AI_data" : {"learning_rate" : 0.01, "alpha" : 0.1, "gamma" : 0.95, "exploration_decay" : 0.99, "model_loc" : None, "NN_data" : {"type" : "dense"}}
            },
    }

    # simulation start
    total_time = 2000
    filename = "001.csv"
    start(data, total_time, filename)
    
    # system.agent.model.save("./data/model.h5")
    print("done")

# Start
if __name__=="__main__":
    main()