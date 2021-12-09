# CS 271 Sokoban Game Group 17

### Members: 
- Yeung, Calvin
- Jeng-Shiun, Ho
- Shen, Daolun

## Instructions:
To run the program: 

    python3 main.py [input_file_path]

Output format:

    [solution length] [solution]



Parameters that can be modified:
- reward: environment.py BASIC_REWARD 
  + 'SPACE': step on a space or empty target
  + 'INFEASIBLE': if infeasible action(for future reward observation only)
  + 'ON_TARGET': box push to a target
  + 'ON_SPACE': box push to a space
      + 'BOX_BY_BOX': box's new position has an adjacent wall(s)
    + 'BOX_BY_WALL': box's new position has an adjacent wall(s)
  + 'OFF_TARGET': push a box off target
  + 'DEADLOCK': deadlock reached
  + 'GOAL': goal reached
- epsilon greedy/UCB switch: [qlearning.py](qlearning.py) def select_action(self, state, greedy=True)
    + epsilon greedy if greedy = True 
    + UCB if greedy = False
    
- epsilon(with epsilon greedy switch on): [environment.py](environment.py) def get_epsilon(self)
- delta(heuristic will not take effect if 0): [environment.py](environment.py) def get_delta(self, state, action)
- learning_rate: [environment.py](environment.py) def get_learning_rate(self, state, action)


