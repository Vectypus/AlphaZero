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


class OneStepLookaheadTicTacToePlayer():
    """
    Simple player who always takes a win if presented, or blocks a loss if obvious, otherwise is random.
    """

    def __init__(self, game):
        self.game = game
        self.player_num = 1

    def play(self, board):
        valid_moves = self.game.getValidMoves(board, self.player_num)
        win_move_set = set()
        fallback_move_set = set()
        stop_loss_move_set = set()
        for move, valid in enumerate(valid_moves):
            if not valid:
                continue
            if self.player_num == self.game.getGameEnded(*self.game.getNextState(board, self.player_num, move)):
                win_move_set.add(move)
            if -self.player_num == self.game.getGameEnded(*self.game.getNextState(board, -self.player_num, move)):
                stop_loss_move_set.add(move)
            else:
                fallback_move_set.add(move)

        if len(win_move_set) > 0:
            ret_move = np.random.choice(list(win_move_set))
        elif len(stop_loss_move_set) > 0:
            ret_move = np.random.choice(list(stop_loss_move_set))
        elif len(fallback_move_set) > 0:
            ret_move = np.random.choice(list(fallback_move_set))
        else:
            raise Exception("No valid moves!")

        return ret_move
