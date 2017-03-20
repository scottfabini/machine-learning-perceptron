#!/usr/bin/env python3
import pandas as pd
import numpy as np
import random
from enum import Enum
import os

# Enumeration of available actions
class Action(Enum):
    P = 0
    N = 1
    S = 2
    E = 3
    W = 4

# Enumeration of available states
class State(Enum):
    Empty = 0
    Wall = 1
    Can = 2

class Board:

    # Constructor initialization
    def __init__(self, steps_M, 
                      reward_can, reward_crash, reward_pick_up_empty,
                      nu, gamma):
        self.board = pd.DataFrame([[State.Empty for x in range(12)] for y in range(12)])
        self.board[0] = State.Wall
        self.board[11] = State.Wall
        self.board.iloc[0] = State.Wall
        self.board.iloc[11] = State.Wall
        self.robot = (1,1)
        self.qMatrix = np.zeros((5 ** 5, 5))
        self.steps_M = steps_M
        self.reward_can = reward_can
        self.reward_crash = reward_crash
        self.reward_pick_up_empty = reward_pick_up_empty
        self.nu = nu
        self.gamma = gamma
        self.move_penalty = 0
        self.move_score = np.zeros(len(Action))
        self.qMatrixValues = [None] * len(Action)
        self.sensed_state = np.zeros(len(Action))


    # Randomize the board state.
    def shuffleBoard(self):
        for (i,j), element in np.ndenumerate(self.board):
            if i != 0 and i != 11 and j != 0 and j != 11:
                if random.random() > 0.5:
                    val = State.Can
                else:
                    val = State.Empty
                self.board.set_value(i, j, val)
        i = random.randint(1,10)
        j = random.randint(1,10)
        self.robot = (i, j)

    # Run the episode. Choose an action to take, perform the action,
    # and update the QMatrix (if updateQ == True).
    def runEpisode(self, episode, epsilon, updateQ):
        reward = np.zeros(self.steps_M)
        for i in range(self.steps_M):
            currentState = self.senseCurrentState(self.robot)

            # choose current action
            if 1 - epsilon < random.random():
                currentAction = random.choice(list(Action)) 
            else:
                currentAction = self.maximizeAction()

            # perform the action
            if  currentAction == Action.P and self.board.get_value(self.robot[0], self.robot[1]) == State.Can:
                self.board.set_value(self.robot[0], self.robot[1], State.Empty)
                reward[i] = self.reward_can
                nextState = self.senseCurrentState(self.robot)

            elif currentAction == Action.P and self.board.get_value(self.robot[0], self.robot[1]) == State.Empty:
                reward[i] = self.reward_pick_up_empty
                nextState = self.senseCurrentState(self.robot)

            elif self.isWall(self.robot, currentAction):
                reward[i] = self.reward_crash
                nextState = self.senseCurrentState(self.robot)
            
            else:
                self.robot = self.moveTo(self.robot, currentAction)            
                reward[i] = self.move_penalty
                nextState = self.senseCurrentState(self.robot)
            # update the qMatrix (if this is training data)
            if updateQ:
                self.updateQMatrix(reward[i], currentState, currentAction, nextState)
        return np.sum(reward)

    # Update the qMatrix according to the Q-learning algorithm.
    def updateQMatrix(self, reward, currentState, currentAction, nextState):
        for i, action in enumerate(Action):
            self.qMatrixValues[i] = self.qMatrix[nextState, action.value]
        if np.argmax(self.qMatrixValues) == 0 and self.qMatrixValues[0] == 0:
            nextAction = random.randint(0, 4)
        else: 
            nextAction =  np.argmax(self.qMatrixValues)
        self.qMatrix[currentState, currentAction.value] = self.qMatrix[currentState, currentAction.value] + self.nu * (reward + self.gamma * self.qMatrix[nextState, nextAction] - self.qMatrix[currentState, currentAction.value])

    # Query the QMatrix to determine the best action to take.
    def maximizeAction(self):
        for i, action in enumerate(Action):
            current_state = self.senseCurrentState(self.robot)
            self.move_score[i] = self.qMatrix[current_state, i]
        if np.argmax(self.move_score) == 0 and self.move_score[0] == 0:
            return self.reverseIndex(random.randint(0, 4))
        return self.reverseIndex(np.argmax(self.move_score))

    # Query the board for the current state of the robot at the given location.
    def senseCurrentState(self, location):
        for i, action in enumerate(Action):
                self.sensed_state[i] = 3 ** i * self.detect(location, action)
        return int(np.sum(self.sensed_state))


    

    '''  Begin Helper Functions '''

    # Display the board
    def displayBoard(self):
        for (i,j), element in np.ndenumerate(self.board):
            if i == self.robot[0] and j == self.robot[1]:
                print("*", end = ' ')
            else:
                print(element.value, end=' ')
            if j == self.board.shape[1] - 1:
                print()
        print("\n\n")

    # Detect the state of the adjacent sqare
    def detect(self, robot, action):
        move = self.moveTo(robot, action)
        sensed_value = self.board.get_value(move[0], move[1]).value
        return sensed_value


    # Helper function to convert the move Enum into an index in the qMatrix
    def reverseIndex(self, moveIndex):
        if moveIndex == 0:
            return Action.P
        elif moveIndex == 1:
            return Action.N
        elif moveIndex == 2:
            return Action.S
        elif moveIndex == 3:
            return Action.E
        elif moveIndex == 4:
            return Action.W

    # Move the robot to a new location based on the action
    def moveTo(self, robot, action):
        if action == Action.P:
            move = (0, 0)
        elif action == Action.N:
            move = (-1, 0)
        elif action == Action.S:
            move = (1, 0)
        elif action == Action.E:
            move = (0, 1)
        elif action == Action.W:
            move = (0, -1)
        return tuple(map(sum, zip(robot, move)))

    # Detect if the robot will run into a wall on the nextMove
    def isWall(self, robot, nextMove):
        i, j = self.moveTo(self.robot, nextMove)
        if self.board.get_value(i, j) == State.Wall:
            return True
        else:
            return False

    # Sum the state of the board for debug
    def sum(self):
        sum = 0
        for (i,j), element in np.ndenumerate(self.board):            
            sum += element.value
        return sum
