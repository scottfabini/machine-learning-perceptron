#!/usr/bin/env python3

"""
Homework 6 for CS545 Machine Learning at Portland State University

Perform experiments on Q-Learning
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import random
import math
import sys
import getopt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from enum import Enum
from Board import Board



class Main:

    episodes_N = 5000
    test_episodes_N = 5000
    steps_M = 200
    reward_can = 10
    reward_crash = -5
    reward_pick_up_empty = -1
    eta = 0.2
    gamma = 0.9
    epsilon = 1
    final_epsilon = .1
    epsilon_reduction_rate = 0.01
    epsilon_episodes = 50

    

    def main(self, argv):
        self.getConfiguration(argv)
        board = Board(self.steps_M, self.reward_can, self.reward_crash, self.reward_pick_up_empty, self.eta, self.gamma)
        ## What to calculate: average sum-of-rewards per episode, standard deviation
        mean_training_rewards_per_100_episodes = np.empty(int(self.episodes_N / 100))

        for n in range(int(self.episodes_N / 100)):
            print("---Episode {0}---".format(n * 100))
            training_rewards = np.empty(100)
            for i in range(100):
                board.shuffleBoard()
                if i != 0 and i % self.epsilon_episodes == 0 and self.epsilon > self.final_epsilon:
                    self.epsilon -= self.epsilon_reduction_rate
                training_rewards[i] = board.runEpisode(i, self.epsilon, True)
            mean_training_rewards_per_100_episodes[n] = np.mean(training_rewards)


        print("Training rewards: \n{0}".format(mean_training_rewards_per_100_episodes))
        print("Std. Deviation: \n{0}".format(np.std(mean_training_rewards_per_100_episodes)))
        plt = self.plot_results(mean_training_rewards_per_100_episodes, 1)

        print("Begin Test Data")
        self.epsilon = 0.1
        mean_test_rewards_per_100_episodes = np.empty(int(self.test_episodes_N / 100))
        for n in range(int(self.test_episodes_N / 100)):
            print("---Episode {0}---".format(n * 100))
            test_rewards = np.empty(100)
            for i in range(100):
                board.shuffleBoard()
                test_rewards[i] = board.runEpisode(i, self.epsilon, False)
            mean_test_rewards_per_100_episodes[n] = np.mean(test_rewards)
                
        print("Test rewards: \n{0}".format(mean_test_rewards_per_100_episodes))
        print("Std. Deviation: \n{0}".format(np.std(mean_test_rewards_per_100_episodes)))
        plt2 = self.plot_results(mean_test_rewards_per_100_episodes, 2)

        plt.show()
        plt2.show()

    
    def plot_results(self, results, figure_number):
        plt.figure(figure_number)
        plt.plot(np.arange(len(results)), results, color='black',
                 lw=2, label='sum')
        plt.xlim([0.0, len(results)-1])
        plt.ylim([min(0, min(results)), max(results)+1 ])
        plt.xlabel('x * 100 Episodes')
        plt.ylabel('Average Sum of Rewards')
        plt.title('')
        plt.legend(loc="lower right")
        return plt

    def getConfiguration(self, argv):
        # process command line arguments
        try:
            opts, args = getopt.getopt(argv[1:], "N:M:r:h", ["eta=", "gamma=", "initial_epsilon=", "final_epsilon=", "accuracy-file=", "confusion-file="])
        except getopt.GetoptError:
            print('Main.py -N <episodes> -M <steps in episode> -r <Reward for can> --eta=0.5 --gamma=0.9 --initial-epsilon=1 --final-epsilon=0.1')
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print('Usage: Main.py -N <episodes> -M <steps in episode> -nu=0.5 -gamma=0.9')
                sys.exit()
            elif opt == "-N":
                self.episodes_N = int(arg)
            elif opt == "-M":
                self.steps_M = int(arg)
            elif opt == "-r":
                self.reward_can = int(arg)
            elif opt == "--eta":
                self.eta = float(arg)
            elif opt == "--gamma":
                self.gamma = float(arg)
            elif opt == "--initial-epsilon":
                self.epsilon = float(arg)
            elif opt == "--final-epsilon":
                self.final_epsilon = float(arg)
            elif opt == "--accuracy-file":
                self.accuracy_file = './results/' + arg
            elif opt == "--confusion-file":
                self.confusion_file = './results/' + arg

        print('Configuration: \n'
              'Episodes: {0}\n' 'Steps: {1}\n'
              'Can Reward: {2}\n' 'Learning Rate (eta): {3}\n'
              'Discount Factor (gamma): {4}\n'
              'Training Epsilon Range: {5}-{6}\n' 
              .format(self.episodes_N, self.steps_M, self.reward_can, self.eta, self.gamma,
                      self.epsilon, self.final_epsilon))



if __name__ == "__main__":
    sys.exit(Main().main(sys.argv))