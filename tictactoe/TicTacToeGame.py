import numpy as np
from Game import Game

from .TicTacToeLogic import Board


class TicTacToeGame(Game):
    """
    Game class implementation for the game of TicTacToe.

    Author: Evgeny Tyurin, github.com/evg-tyurin
    Date: Jan 5, 2018.

    Modified by Victor Taillieu, github.com/Vectypus

    Based on the OthelloGame by Surag Nair.
    """

    def __init__(self, n=3):
        self.n = n

    def getInitBoard(self):
        # Return initial board (numpy)
        b = Board(self.n)
        return np.array(b.squares)

    def getBoardSize(self):
        return (self.n, self.n)

    def getActionSize(self):
        # n * n squares + filled board case
        return self.n * self.n + 1

    def getNextState(self, board, player, action):
        # If player takes action on board, return next (board,player)
        # Action must be a valid move
        if action == self.n * self.n:
            # Can't move, filled board
            return (board, -player)

        b = Board(self.n)
        b.squares = np.copy(board)

        # Execute the move
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, player)

        return (b.squares, -player)

    def getValidMoves(self, board, player):
        # Return a binary vector of size actionSize
        valids = [0] * self.getActionSize()
        b = Board(self.n)
        b.squares = np.copy(board)
        legalMoves = b.get_legal_moves()

        if len(legalMoves) == 0:
            # Can't move, filled board
            valids[-1] = 1
        else:
            for x, y in legalMoves:
                valids[self.n * x + y] = 1

        return np.array(valids)

    def getGameEnded(self, board, player):
        """
        Check game status.

        Returns:
               0: not ended
               1: player 'O' won
              -1: player 'X' won
            1e-4: draw
        """

        b = Board(self.n)
        b.squares = np.copy(board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0
        # Draw
        return 1e-4

    def getCanonicalForm(self, board, player):
        # Return board if player==1, else -board
        return player * board

    def getSymmetries(self, board, pi):
        # List board symmetries: mirror, rotational
        assert(len(pi) == self.getActionSize())
        pi_board = np.reshape(pi[:-1], self.getBoardSize())
        symmetries = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                symmetries += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return symmetries

    def stringRepresentation(self, board):
        return board.tostring()

    @staticmethod
    def display(board):
        """
        Display the board to the console.
        """

        n = board.shape[0]

        for x in range(n):
            print(' ', end='')
            for y in range(n):
                piece = board[x][y]
                if piece == 1:
                    print('0', end=' ')
                elif piece == -1:
                    print('X', end=' ')
                else:
                    print(' ', end=' ')

                if y < n - 1:
                    print('|', end=' ')
                else:
                    print()

            if x < n - 1:
                print("---|---|---")
            else:
                print()
