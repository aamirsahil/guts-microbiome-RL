import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    filename = "001.csv"
    df = pd.read_csv(filename)
    print(df['B1_speed'])
    # plt.plot(df['B1_speed'])
    # plt.plot(df['B2_speed'])
    # plt.show()

    # df['B1_reward'].plot()
    # df['B2_reward'].plot()
    # df['reward'].plot()
    # plt.show()
    # plot speed
    
    # plot reward

if __name__=="__main__":
    main()

def plotFig(bacteria_num, chemical_num, food_num, plot_time, plot_bacteria_pop, data,
             plot_component_pop, plot_chemical_pop, total_time, action_size, plot_food_pop, plot_reward):
    # bacteria plot
    labels = [f"Bacteria{i+1}" for i in range(bacteria_num)]
    xlabel = "Time"
    ylabel = "Bacteria population"
    title = "Bacteria population"
    save_loc = f"./data/bacteriaPlot_bacterianum{bacteria_num}_chemicalnum{chemical_num}_foodnum{food_num}.png"
    plotStack(plot_time, plot_bacteria_pop, xlabel, ylabel, title, labels, save_loc)
    # food plot
    labels = data["component_list"]
    xlabel = "Time"
    ylabel = "Component amount"
    title = "Component amount"
    save_loc = f"./data/componentPlot_bacterianum{bacteria_num}_chemicalnum{chemical_num}_foodnum{food_num}.png"
    plotStack(plot_time, plot_component_pop, xlabel, ylabel, title, labels, save_loc)
    # # chem plot
    labels = data["chemical_list"]
    xlabel = "Time"
    ylabel = "Chemical amount"
    title = "Chemical amount"
    save_loc = f"./data/chemicalPlot_bacterianum{bacteria_num}_chemicalnum{chemical_num}_foodnum{food_num}.png"
    plotStack(plot_time, plot_chemical_pop, xlabel, ylabel, title, labels, save_loc)
    # # chem plot
    xlabel = "Time"
    ylabel = "Reward amount"
    title = "Reward amount"
    save_loc = f"./data/rewardPlot_bacterianum{bacteria_num}_chemicalnum{chemical_num}_foodnum{food_num}.png"
    plotNormal(plot_time, plot_reward, xlabel, ylabel, title, save_loc)
    # food/action plot
    turns = 10
    food_per_turn = np.zeros((total_time//turns, action_size))
    for turn, actions in enumerate(food_per_turn):
        for action, value in enumerate(actions):
            startIdx = turn*turns
            endIdx = startIdx + turns
            food_per_turn[turn][action] = plot_food_pop[startIdx:endIdx].count(action)
    labels = data["food_list"]
    xlabel = "Time"
    ylabel = f"Food in last {turns} turn"
    title = "Food consumption"
    save_loc = f"./data/foodPlot_bacterianum{bacteria_num}_chemicalnum{chemical_num}_foodnum{food_num}.png"
    # plot_time = np.linspace(0, total_time, total_time//turns)
    # plotStack(plot_time, food_per_turn, xlabel, ylabel, title, labels, save_loc)
    print(food_per_turn)

# Stack plotting graphs
def plotStack(x, y, xlabel, ylabel, title, labels, save_loc):
    # Stackplot with X, Y, colors value
    plt.stackplot(x, y, labels=labels)
    showPlot(xlabel, ylabel, title, save_loc)

def plotNormal(x, y, xlabel, ylabel, title, save_loc):
    plt.plot(x, y)
    showPlot(xlabel, ylabel, title, save_loc)

def showPlot(xlabel, ylabel, title, save_loc):
    plt.xlabel(xlabel)
    # No of hours
    plt.ylabel(ylabel)
    # Title of Graph
    plt.title(title)
    plt.legend()
    # Displaying Graph
    plt.savefig(save_loc)
    plt.show()