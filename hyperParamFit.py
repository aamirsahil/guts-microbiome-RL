from bayes_opt import BayesianOptimization
from main import main

# Define the RL objective function to be optimized
def rl_objective_function(h0, h1, h2, h3, h4, h5):
    # Set the RL hyperparameters based on x and y
    kick_satisfaction = h0
    eat_time = h1
    learning_rate = h2
    alpha = h3
    gamma = h4
    exp_decay = h5

    # Initialize and train the RL agent with the hyperparameters
    evaluation_metric = main(kick_satisfaction,eat_time,learning_rate,alpha,gamma,exp_decay)

    # Return the negative evaluation metric (BayesianOptimization minimizes the objective function)
    return -evaluation_metric

# Define the parameter bounds for optimization
parameter_bounds = {
    'h0': (0, 10),
    'h1': (1, 10),
    'h2': (0.001, 1),
    'h3': (0.01, 1),
    'h4': (0, 1),
    'h5': (0, 1)
}

# Create an instance of the BayesianOptimization class
bayes_optimizer = BayesianOptimization(f=rl_objective_function, pbounds=parameter_bounds)

# Perform Bayesian optimization
bayes_optimizer.maximize(init_points=2, n_iter=5)

# Access the best hyperparameters and the corresponding evaluation metric
best_hyperparameters = bayes_optimizer.max['params']
best_evaluation_metric = -bayes_optimizer.max['target']

print(best_evaluation_metric)
print(best_hyperparameters)