import numpy as np
from Game import Game

from .Connect4Logic import Board


class Connect4Game(Game):
    """
    Connect4 Game class implementing the alpha-zero-general Game interface.
    """

    def __init__(self, height=6, width=7, win_length=4, np_pieces=None):
        Game.__init__(self)
        self._base_board = Board(height, width, win_length, np_pieces)

    def getInitBoard(self):
        return self._base_board.np_pieces

    def getBoardSize(self):
        return (self._base_board.height, self._base_board.width)

    def getActionSize(self):
        return self._base_board.width

    def getNextState(self, board, player, action):
        """Returns a copy of the board with updated move, original board is unmodified."""
        b = self._base_board.with_np_pieces(np_pieces=np.copy(board))
        b.add_stone(action, player)
        return b.np_pieces, -player

    def getValidMoves(self, board, player):
        "Any zero value in top row in a valid move"
        return self._base_board.with_np_pieces(np_pieces=board).get_valid_moves()

    def getGameEnded(self, board, player):
        b = self._base_board.with_np_pieces(np_pieces=board)
        winstate = b.get_win_state()
        if winstate.is_ended:
            if winstate.winner is None:
                # draw has very little value.
                return 1e-4
            elif winstate.winner == player:
                return +1
            elif winstate.winner == -player:
                return -1
            else:
                raise ValueError('Unexpected winstate found: ', winstate)
        else:
            # 0 used to represent unfinished game.
            return 0

    def getCanonicalForm(self, board, player):
        # Flip player from 1 to -1
        return board * player

    def getSymmetries(self, board, pi):
        """Board is left/right board symmetric"""
        return [(board, pi), (board[:, ::-1], pi[::-1])]

    def stringRepresentation(self, board):
        return board.tostring()

    @staticmethod
    def display(board):
        for x in range(board.shape[0]):
            print(' ', end='')
            for y in range(board.shape[1]):
                piece = board[x][y]
                if piece == 1:
                    print('0', end=' ')
                elif piece == -1:
                    print('X', end=' ')
                else:
                    print('.', end=' ')

                if y == board.shape[1] - 1:
                    print()

            if x == board.shape[0] - 1:
                print()
