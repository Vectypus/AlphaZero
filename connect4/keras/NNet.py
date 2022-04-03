import os

import numpy as np
from NeuralNet import NeuralNet
from utils import dotdict

from .Connect4NNet import Connect4NNet as onnet

args = dotdict({
    'lr': 1e-3,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 128,
    'num_residual_layers': 20
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        # self.nnet.model.summary()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        Parameters:
            examples: list of examples of form (board, pi, v)
        """

        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)

        self.nnet.model.fit(
            x=input_boards,
            y=[target_pis, target_vs],
            batch_size=args.batch_size,
            epochs=args.epochs
        )

    def predict(self, board):
        """
        Parameters:
            board: board as numpy array
        """

        # Prepare input
        board = board[np.newaxis, :, :]

        pi, v = self.nnet.model.predict(board)

        return pi[0], v[0]

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.h5"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(f"Checkpoint Directory does not exist! Making directory {folder}")
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists!")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.h5"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path '{filepath}'")
        self.nnet.model.load_weights(filepath)
