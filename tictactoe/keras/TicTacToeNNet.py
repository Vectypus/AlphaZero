from keras.models import Model
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, Input, Reshape)
from tensorflow.keras.optimizers import Adam


class TicTacToeNNet():
    """
    NeuralNet for the game of TicTacToe.

    Author: Evgeny Tyurin, github.com/evg-tyurin
    Date: Jan 5, 2018.

    Modified by Victor Taillieu, github.com/Vectypus

    Based on the OthelloNNet by SourKream and Surag Nair.
    """

    def __init__(self, game, args):
        # Game parameters
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        # Neural network model
        self.input_boards = Input(shape=(self.board_x, self.board_y))            # batch_size * 3 * 3

        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)    # batch_size * 3 * 3 * 1
        conv1 = Activation("relu")(
            BatchNormalization(axis=3)(                                          # batch_size * 3 * 3 * num_channels
                Conv2D(args.num_channels, 3, padding="same")(x_image)
            )
        )
        conv2 = Activation("relu")(
            BatchNormalization(axis=3)(                                          # batch_size * 3 * 3 * num_channels
                Conv2D(args.num_channels, 3, padding="same")(conv1)
            )
        )
        conv3 = Activation("relu")(
            BatchNormalization(axis=3)(                                          # batch_size * 3 * 3 * num_channels
                Conv2D(args.num_channels, 3, padding="same")(conv2)
            )
        )
        conv4 = Activation("relu")(
            BatchNormalization(axis=3)(                                          # batch_size * 1 * 1 * num_channels
                Conv2D(args.num_channels, 3, padding="valid")(conv3)
            )
        )
        flat = Flatten()(conv4)                                                  # batch_size * num_channels
        fc1 = Dropout(args.dropout)(                                             # batch_size * 1024
            Activation("relu")(
                BatchNormalization(axis=1)(
                    Dense(1024)(flat)
                )
            )
        )
        fc2 = Dropout(args.dropout)(                                             # batch_size * 512
            Activation("relu")(
                BatchNormalization(axis=1)(
                    Dense(512)(fc1)
                )
            )
        )

        self.pi = Dense(self.action_size, activation="softmax", name="pi")(fc2)  # batch_size * 10
        self.v = Dense(1, activation="tanh", name='v')(fc2)                      # batch_size * 1

        self.model = Model(
            inputs=self.input_boards,
            outputs=[self.pi, self.v]
        )
        self.model.compile(
            loss=["categorical_crossentropy", "mean_squared_error"],
            optimizer=Adam(args.lr)
        )
