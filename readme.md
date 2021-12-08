#CS 271 Sokoban Game Group 17

###members: Yeung, Calvin, Jeng-Shiun, Ho, Shen, Daolun

##Instruction:
run the program: 

    python3 main.py [input_file_path]

output:

    [solution length] [solution]



Parameters can be modified:

- reward: environment.py line 20 
- epsilon greedy/UCB switch: [qlearning.py] def select_action(self, state, greedy=True)
    + epsilon greedy if greedy = True 
    + UCB if greedy = False
    
- epsilon(with epsilon greedy switch on): [environment.py] def get_epsilon(self)
- delta(heuristic will not take effect if 0): [environment.py] def get_delta(self, state, action)
- learning_rate: [environment.py] def get_learning_rate(self, state, action)


