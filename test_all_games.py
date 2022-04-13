""""
    This is a Regression Test Suite to automatically test all games.
    Each test plays two quick games using an untrained neural network
    (randomly initialized) against a random player.
"""

import unittest

import Arena

from MCTS import MCTS

from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.TicTacToePlayers import *
from tictactoe.keras.NNet import NNetWrapper as TicTacToeNNet

from connect4.Connect4Game import Connect4Game
from connect4.Connect4Players import *
from connect4.keras.NNet import NNetWrapper as Connect4NNet

import numpy as np
from utils import dotdict


class TestAllGames(unittest.TestCase):

    @staticmethod
    def execute_game_test(game, neural_net):
        rp = RandomPlayer(game).play

        args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
        mcts = MCTS(game, neural_net(game), args)
        n1p = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

        arena = Arena.Arena(n1p, rp, game)
        print(arena.playGames(2, verbose=False))

    def test_tictactoe(self):
        self.execute_game_test(TicTacToeGame(), TicTacToeNNet)

    def test_connect4(self):
        self.execute_game_test(Connect4Game(), Connect4NNet)


if __name__ == '__main__':
    unittest.main()
