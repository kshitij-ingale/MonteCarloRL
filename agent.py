"""
Monte Carlo implementation for Gym environment of Blackjack
"""
import gym, argparse, os
import numpy as np
from config import Training_parameters, Directories


class QTable:
    def __init__(self, num_states=32 * 11 * 2, num_actions=2):
        # self.Q_values = np.random.rand(num_states, num_actions)
        self.Q_values = np.zeros((num_states, num_actions))
        self.num_actions = num_actions
        self.state_map = {}
        self.pointer = 0
        self.policy = np.zeros((num_states, num_actions))
        self.visits = np.zeros((num_states, num_actions))
        self.epsilon = Training_parameters.epsilon
        self.discount = Training_parameters.discount

    def get_value_function(self, rewards):
        rewards_array = np.array(rewards)
        rewards_array_cumsum = rewards_array.cumsum()
        value_function = [rewards_array_cumsum[-1]]
        value_function.extend((rewards_array_cumsum[-1] - rewards_array_cumsum)[:-1])
        # Discount rewards
        return [value_function[i]*(self.discount**i) for i in range(len(value_function))]

    def update(self, episode_returns):
        states, actions, rewards = zip(*episode_returns)
        first_occurrence = set()
        value_function = self.get_value_function(rewards)
        for i in range(len(states)):
            if (states[i], actions[i]) not in first_occurrence:
                first_occurrence.add((states[i], actions[i]))
                state_index = self.get_index(states[i])
                self.Q_values[state_index, actions[i]] = self.Q_values[state_index, actions[i]] * self.visits[
                    state_index, actions[i]] + value_function[i]
                self.visits[state_index, actions[i]] += 1
                self.Q_values[state_index, actions[i]] /= self.visits[state_index, actions[i]]

        for i in range(len(states)):
            state_index = self.get_index(states[i])
            self.policy[state_index, :] = self.epsilon / self.num_actions
            self.policy[state_index, np.argmax(self.Q_values[state_index, :])] += 1 - self.epsilon

    def get_index(self, state):
        """
        Fetch index of state for Q-values array and assign index if not present
        :param state: environment state of agent
        :return: index corresponding to the state
        """
        if state not in self.state_map:
            self.state_map[state] = self.pointer
            self.pointer += 1
        return self.state_map[state]

    def get_action(self, state):
        """
        Obtain action as per epsilon-greedy policy
        :param state: environment state of agent
        :return: action to be taken from current state as per current policy
        """
        if np.sum(self.policy[self.get_index(state), :]) == 0:
            return np.random.choice(np.arange(self.num_actions))
        return np.random.choice(np.arange(self.num_actions), p=self.policy[self.get_index(state), :])

    def save_current_arrays(self):
        np.savez(Directories.arrays, qvalues=self.Q_values, statemap=self.state_map)

    def load_arrays(self):
        arrays = np.load(Directories.arrays,allow_pickle=True)
        self.Q_values = arrays['qvalues']
        self.state_map = arrays['statemap']


class MonteCarlo:
    def __init__(self, parameters):
        self.Q_store = QTable()
        if parameters.test_decision:
            self.Q_store.load_arrays()
        self.env = gym.make(parameters.environment_name)
        self.train_episodes = parameters.train_episodes

    def generate_episode(self):
        done = False
        state = self.env.reset()
        episode_data = []
        while not done:
            action = self.Q_store.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            episode_data.append((state, action, reward))
        return episode_data

    def train(self):
        for i in range(self.train_episodes):
            episode_returns = self.generate_episode()
            self.Q_store.update(episode_returns)
        self.Q_store.save_current_arrays()

    def greedy_policy(self, q_values):
        """
        Returns action as per greedy policy
        :param q_values: Q-values for the possible actions
        :return: Action selected by agent as per greedy policy on Q_values
        """
        return np.argmax(q_values)

    def test(self, test_episodes=100):
        rewards = []
        for test_episode in range(test_episodes):
            curr_episode_reward = 0
            state = self.env.reset()
            done = False
            while not done:
                if state not in self.Q_store.state_map:
                    # If state not encountered while training use random action
                    print('Unknown state encountered, using random action')
                    action = self.env.action_space.sample()
                else:
                    action = self.greedy_policy(self.Q_store.Q_values[self.Q_store.state_map[state], :])
                next_state, reward, done, _ = self.env.step(action)
                curr_episode_reward += reward
                if done:
                    state = self.env.reset()
                else:
                    state = next_state
            rewards.append(curr_episode_reward)
        rewards = np.array(rewards)
        return np.mean(rewards), np.std(rewards)

def parse_arguments():
    """Parse command line arguments using argparse"""
    parser = argparse.ArgumentParser(description='Train Monte Carlo agent for Blackjack')
    parser.add_argument('--t', dest='test_decision', action='store_true', help='Test the agent')
    parser.add_argument('--e', dest='environment_name', default='Blackjack-v0', type=str,
                        help='Gym Environment')
    parser.add_argument('--ep', dest='train_episodes', default=10000, type=int,
                        help='Number of Training episodes')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    if not args.test_decision:
        MCAgent = MonteCarlo(args)
        MCAgent.train()
    else:
        if os.path.exists(Directories.arrays):
            MCAgent = MonteCarlo(args)
        else:
            raise FileNotFoundError('Q-values and state-map arrays not found')
    print(MCAgent.test())


if __name__ == '__main__':
    main()

