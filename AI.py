# AI class
import numpy as np
import tensorflow as tf

class AI:
    def __init__(self, state_size : int = 5, action_size : int = 5,
                 learning_rate : float = 0.01, alpha : float = 0.1, gamma : float = 0.9, 
                 exploration : float = 1.0, exploration_decay : float = 0.94, NN_data : dict = None, model_loc : str = None) -> None:
        """
        AI class.

        Parameters:\n
            state_size = (int) num of input layer nodes\n
            action_size = (int) num of output layer nodes\n
            learning_rate = (float) rate of NN learning\n
            alpha = (float) rate of RL learning\n
            gamma = (float) significance of previous reward\n
            exploration = (float) determines exploration\n
            exploration_decay = (float) rate of exploration decay\n
            NN_data = (dict) {"name" : dense} type of neural_net\n
            model_loc = (str) for importing model eg : "./data/model.h5"\n
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration
        self.exploration_decay = exploration_decay
        if model_loc == None:
            if NN_data["type"] == "dense":
                self.model = self._build_dense()
            elif NN_data["type"] == "LSTM":
                self.model = self._build_LSTM(**NN_data)
        else:
            self.model = tf.keras.models.load_model(model_loc)

    def _build_dense(self) -> tf.keras.Model:
        """
        Build a densely connected model
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(10, input_shape = (self.state_size,) ))
        model.add(tf.keras.layers.Dense(10, activation = 'relu' ))
        model.add(tf.keras.layers.Dense(self.action_size, activation = 'softmax' ))
        model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate))
        
        return model

    def _build_LSTM(self, lstm_memory : int = 2) -> tf.keras.Model:
        """
        Build LSTM

        Parameters:\n
            lstm_memory = (int) number of previous states kept in track.
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(units=128, input_shape=(lstm_memory, self.state_size)))
        model.add(tf.keras.layers.Dense(units=self.action_size, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def _act(self, state : list = None) -> list:
        """
        Calculate q values\n
        
        Parameters\n
            state = (list) current state\n
        Return\n
            q_values = (list) values of action\n
        """
        q_values = self.model.predict(state)
        return q_values[0]
    
    def _findAllMax(self, list_ : list = None) -> list:
        """
        Find the index of all max elements.

        Parameters:\n
            list_ = (list) list of numbers\n
        return:\n
            all_max_idx = (list) list of index of all max elements.
        """
        max_ = np.max(list_)
        all_max_idx = [idx for idx,value in enumerate(list_) if value == max_]
        return all_max_idx

    def decide(self, state : list = None) -> int:
        """
        Decides what action to take\n
        
        Parameters\n
            state = (list) current state\n
        Return\n
            action = (int) index of food to eat after a exploration/exploitation consideration\n
        """
        # calculate q values for given state
        q_values = self._act(state=state)
        r = np.random.random()
        # index of action to take
        action = 0
        if r < self.exploration:
            # exploration
            action = np.random.choice(range(self.action_size))
        else:
            # exploitation
            action = np.random.choice(self._findAllMax(q_values))
        # reduce exploration rate
        self.exploration *= self.exploration_decay
        
        return action

    def train(self, reward : float = 0, state : list = None, 
              next_state : list = None, action : int = 0):
        """
        Trains the NN

        Parameters:\n
            reward = (float) reward from getting to new state\n
            state = (list) state before action was taken\n
            next_state = (list) state after the action was taken\n
            action = (int) current action\n
        """
        q_values = self.model.predict(state)
        next_q_values = self.model.predict(next_state)

        # Bellmann equation
        q_values[0][action] = (1 - self.alpha)*q_values[0][action] + self.alpha*(reward + self.gamma * np.max(next_q_values[0]))

        # update the NN weights
        self.model.fit(state, q_values, epochs=10, verbose=None)