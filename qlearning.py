import environment
import heuristics
import random
from copy import deepcopy

class QLearner:
    def __init__(self, state) -> None:
        self.state = state # initial state
        self.q_table = {}
        self.f_table = {}
        self.discount_factor = 0.8
        self.learning_rate = 0.5
        self.epsilon = 0.5

        self.distance_table = heuristics.get_distance_table(self.state)
        self.heuristics = [heuristics.EMMHeuristic(self.distance_table), heuristics.AgentBoxHeuristic(self.distance_table)]
        self.h_weight = [2, 1] # relative importance of heuristics

        self.max_episode_length = 500

    def heuristic(self, state):
        h_val = 0
        for i, h in enumerate(self.heuristics):
            h_val += self.h_weight[i] * h.heuristic(state)
        return h_val

    def select_action(self, state):
        feasible_actions = environment.get_feasible_actions(state)
        epsilon = self.get_epsilon(state)

        # apply exploration with epsilon-greedy
        if random.random() <= epsilon:
            return random.choice(feasible_actions)
        else:
            max_action = None
            max_val = -10e10
            h = self.heuristic(state)
            for a in feasible_actions:
                delta = self.get_delta(state, a)
                s_a = environment.step(state, a)
                delta_h = h - self.heuristic(s_a)
                val = (1 - delta) * self.get_q_value(state, a) + delta * delta_h

                if val > max_val:
                    max_val = val
                    max_action = a
            return max_action

    def update_q_value(self, state, action, new_state, reward):
        learning_rate = self.get_learning_rate(state, action)
        q_val = self.get_q_value(state, action)
        max_q = self.get_max_q(new_state)
        q_val += learning_rate * (reward + self.discount_factor * max_q - q_val)
        self.q_table[(state, action)] = q_val

    def update_f_value(self, state, action):
        if (state, action) in self.f_table:
            self.f_table[(state, action)] += 1
        else:
            self.f_table[(state, action)] = 1

    def get_max_q(self, state):
        feasible_actions = environment.get_feasible_actions(state)
        q_vals = [self.get_q_value(state, action) for action in feasible_actions]
        return max(q_vals)

    def get_q_value(self, state, action):
        return self.q_table[(state, action)] if (state, action) in self.q_table else -25

    def get_state_action_frequency(self, state, action):
        if (state, action) in self.f_table:
            return self.f_table[(state, action)]
        return 0

    def get_state_frequency(self, state):
        f = 0
        for a in environment.get_feasible_actions(state):
            f += self.get_state_action_frequency(state, a)
        return f

    def get_epsilon(self, state):
        #f = self.get_state_frequency(state)
        #return 1 / (2 * (f + 1))
        return self.epsilon

    def get_delta(self, state, action):
        f = self.get_state_action_frequency(state, action)
        return 1 / (f + 1)

    def get_learning_rate(self, state, action):
        #f = self.get_state_action_frequency(state, action)
        #return 1 / (2 * (f + 1))
        return self.learning_rate

    def learn(self, episodes):
        for i in range(episodes):
            state = deepcopy(self.state)
            episode_length = 0
            goal_found = False
            deadlock = False
            if i % 10 == 0:
                self.epsilon = 0.5 / (i/10+1)
            while episode_length < self.max_episode_length:
                #print(state.map)
                if environment.is_goal(state):
                    self.learning_rate *= 0.9
                    goal_found = True
                    break
                elif environment.is_deadlock(state):
                    deadlock = True
                    break
                
                action = self.select_action(state)
                new_state = environment.step(state, action)
                reward = environment.get_reward(state, action, new_state)

                self.update_f_value(state, action)
                self.update_q_value(state, action, new_state, reward)
                state = new_state

                episode_length += 1

            print(f"Episode {i+1}, length={episode_length}, goal_found={goal_found}, deadlock={deadlock}, max_q={self.get_max_q(state)}")
