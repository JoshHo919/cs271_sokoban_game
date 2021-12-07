import environment
import heuristics
import random
import numpy as np
from copy import copy

class QLearner:
    def __init__(self, state) -> None:
        self.state = state # initial state
        self.q_table = {}
        self.f_table = {}
        self.discount_factor = 0.96
        self.learning_rate = 0.5
        self.epsilon = 0.1
        self.t = 1 # total time step

        self.distance_table = heuristics.get_distance_table(self.state)
        self.heuristics = [heuristics.EMMHeuristic(self.distance_table), heuristics.AgentBoxHeuristic(self.distance_table)]
        self.h_weight = [2, 1] # relative importance of heuristics

        self.max_episode_length = 1000

    def heuristic(self, state):
        h_val = 0
        for i, h in enumerate(self.heuristics):
            h_val += self.h_weight[i] * h.heuristic(state)
        return h_val

    def select_action(self, state, greedy=False):
        feasible_actions = environment.get_feasible_actions(state)
        # if state.test:
        #     print(state.map)
        #     print(feasible_actions)
        #     state.test = False
        
        if greedy:
            epsilon = self.epsilon
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
                    delta_h = 50 * (h - self.heuristic(s_a))
                    val = (1 - delta) * self.get_q_value(state, a) + delta * delta_h

                    if val > max_val:
                        max_val = val
                        max_action = a
        else:
            c = 1.5
            max_action = None
            max_val = -10e10
            h = self.heuristic(state)
            for a in feasible_actions:
                delta = self.get_delta(state, a)
                s_a = environment.step(state, a)
                delta_h = 20 * (h - self.heuristic(s_a))
                ucb = self.get_q_value(state, a)
                if self.get_state_action_frequency(state, a) > 0:
                    ucb += c * np.sqrt(np.log(self.t) / self.get_state_action_frequency(state, a))
                val = (1 - delta) * ucb + delta * delta_h
                #print(ucb)
                #val = ucb

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
        return self.q_table[(state, action)] if (state, action) in self.q_table else 0

    def get_state_action_frequency(self, state, action):
        if (state, action) in self.f_table:
            return self.f_table[(state, action)]
        return 0

    def get_state_frequency(self, state):
        f = 0
        for a in environment.get_feasible_actions(state):
            f += self.get_state_action_frequency(state, a)
        return f

    def get_epsilon(self, goal_found_list):
        lookback = 50
        epsilon = 0.05
        recent_goal_rate = sum(goal_found_list[-lookback:]) / lookback
        if len(goal_found_list) >= lookback and recent_goal_rate >= 0.5:
            epsilon /= 5 ** ((recent_goal_rate-0.4)*10)
        return epsilon

    def get_delta(self, state, action):
        return 1 / (self.t / 10 + 1)

    def get_learning_rate(self, state, action):
        f = self.get_state_action_frequency(state, action)
        return 1 / (1.2 * (f/2 + 0.5))

    def backtracking_update(self, state_actions):
        for s, a in state_actions[::-1]:
            s_a = environment.step(s, a)
            r = environment.get_reward(s, a, s_a)
            self.update_q_value(s, a, s_a, r)

    def learn(self, episodes, display=True):
        shortest_solution = []
        goal_found_list = []

        for i in range(episodes):
            state = copy(self.state)
            goal_found = False
            deadlock = False
            states = []
            actions = []
            self.epsilon = self.get_epsilon(goal_found_list)
            new_state_actions = 0
            for step in range(self.max_episode_length):
                # exit if goal or deadlock is reached
                if environment.is_goal(state):
                    # update q-values of path if goal is reached
                    if len(actions) < len(shortest_solution) or len(shortest_solution) == 0:
                        shortest_solution = actions
                    goal_found = True
                    break
                elif len(actions) and environment.is_deadlock(state, actions[-1]):
                    # print(state.map)
                    deadlock = True
                    break

                action = self.select_action(state)
                states.append(state)
                actions.append(action)
                
                if (state, action) not in self.q_table:
                    new_state_actions += 1

                new_state = environment.step(state, action)
                reward = environment.get_reward(state, action, new_state)

                self.update_f_value(state, action)
                self.update_q_value(state, action, new_state, reward)
                state = new_state
                self.t += 1
            if goal_found:
                break
            #self.backtracking_update(list(zip(states, actions)))

            if display:
                print(f"Episode {i+1}, length={step}, deadlock={deadlock}, max_q={self.get_max_q(self.state)}, new_state_action_ratio={new_state_actions/step}")

            if False:
                for s in states[-1:]:
                    print(s.map)
                    pass
                s_a = environment.step(states[-1], actions[-1])
                print(s_a.map)
        if display:
            print(f"Shortest solution has length {len(shortest_solution)}: {shortest_solution}")
        return i, len(shortest_solution)