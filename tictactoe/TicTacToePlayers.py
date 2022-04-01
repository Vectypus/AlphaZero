import numpy as np


class RandomPlayer():
    """
    Random player for the game of TicTacToe.

    Author: Evgeny Tyurin, github.com/evg-tyurin
    Date: Jan 5, 2018.

    Modified by Victor Taillieu, github.com/Vectypus

    Based on the OthelloPlayers by Surag Nair.
    """

    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)

        action = np.random.randint(self.game.getActionSize())
        while not valids[action]:
            action = np.random.randint(self.game.getActionSize())

        return action


class HumanTicTacToePlayer():
    """
    Human-ineracting players for the game of TicTacToe.

    Author: Evgeny Tyurin, github.com/evg-tyurin
    Date: Jan 5, 2018.

    Modified by Victor Taillieu, github.com/Vectypus

    Based on the OthelloPlayers by Surag Nair.
    """

    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)

        for i in range(len(valids)):
            if valids[i]:
                print(i, end=' ')
        print()

        while True:
            action = int(input("Move: "))

            if valids[action]:
                break
            else:
                print("Invalid move")

        return action
