# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        
        # action is (pickup, drop) & action space is list of all possible actions including (0,0)
        self.action_space = [(i,j) for i in range(1,m+1) for j in range(1,m+1) if i!=j]
        self.action_space.insert(0, (0,0))
        
        # state is (location, time, day) & state space is list of all possible states
        self.state_space = [(i,j,k) for i in range(1,m+1) for j in range(0,t) for k in range(0,d)]
        
        # For the initial state, pick one state randomly out of all state spaces
        self.state_init = self.state_space[random.randrange(len(self.state_space))]

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        
        # Create location array of zeros and update the respective location value to 1
        X = np.zeros(m)
        X[state[0]-1] = 1
        
        # Create time of day array....
        T = np.zeros(t)
        T[state[1]] = 1
        
        # Create day of the week array....
        D = np.zeros(d)
        D[state[2]] = 1
        
        # Concatenate all to create the encoded state
        state_encod = np.concatenate((X,T,D), axis = None)

        return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        # Locations are represented by values from 1 to 5
        location = state[0]
        
        L = 0
        if location == 1:
            L = 2
        elif location == 2:
            L = 12
        elif location == 3:
            L = 4
        elif location == 4:
            L = 7
        elif location == 5:
            L = 8
        
        requests = np.random.poisson(L)

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]
        
        # Add the action (0,0) and it's index
        possible_actions_index.append(0)
        actions.append((0,0))

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        if action == (0,0):
            reward = -C
            
        else:       
        
            location, time, day = state
            pick_loc, drop_loc = action

            time_ItoP = Time_matrix[location-1][pick_loc-1][time][day]

            # If the time taken to go from initial position to pick up is more than zero, then time and day will change
            new_time, new_day = self.new_time_day(time, day, time_ItoP)
            
            time_PtoD = Time_matrix[pick_loc-1][drop_loc-1][new_time][new_day]

            reward = (R*time_PtoD)-(C*(time_PtoD+time_ItoP))
        
        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        location, time, day = state
        pick_loc, drop_loc = action
        
        if action == (0,0):
            # Adding 1 hour for unused time
            new_time, new_day = self.new_time_day(time, day, 1)
                        
            next_state = (location, new_time, new_day)
            
        else:
            new_location = drop_loc
            
            # new time and day for initial position to pick up position
            time_ItoP = Time_matrix[location-1][pick_loc-1][time][day]
            
            new_time, new_day = self.new_time_day(time, day, time_ItoP)
            
            # new time and day after the drop
            time_PtoD = Time_matrix[pick_loc-1][drop_loc-1][new_time][new_day]
            
            new_time, new_day = self.new_time_day(new_time, new_day, time_PtoD)
            
            next_state = (new_location, new_time, new_day)
        
        
        return next_state

    
    # Utility function which takes in old time, old day, change in time (delta t) and return the new time and day
    def new_time_day(self, time, day, dt):
        
        new_time = int((time+dt)%24)
        day_diff = int((time+dt)//24)
        new_day = int((day+day_diff)%7)
        
        return new_time, new_day


    def reset(self):
        return self.action_space, self.state_space, self.state_init
