from keras.activations import relu
from keras.models import Model
from keras.layers import (Add, BatchNormalization, Conv2D, Dense, Flatten,
                          Input, Reshape)
from tensorflow.keras.optimizers import Adam


def relu_bn(inputs):
    relu1 = relu(inputs)
    bn = BatchNormalization()(relu1)
    return bn


def residual_block(x, filters, kernel_size=3):
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(x)

    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    y = BatchNormalization()(y)
    out = Add()([x, y])
    out = relu(out)

    return out


def value_head(input):
    conv1 = Conv2D(kernel_size=1,
                   strides=1,
                   filters=1,
                   padding="same")(input)

    bn1 = BatchNormalization()(conv1)
    bn1_relu = relu(bn1)

    flat = Flatten()(bn1_relu)

    dense1 = Dense(256)(flat)
    dn_relu = relu(dense1)

    dense2 = Dense(256)(dn_relu)

    return dense2


def policy_head(input):
    conv1 = Conv2D(kernel_size=2,
                   strides=1,
                   filters=1,
                   padding="same")(input)
    bn1 = BatchNormalization()(conv1)
    bn1_relu = relu(bn1)
    flat = Flatten()(bn1_relu)

    return flat


class Connect4NNet():
    def __init__(self, game, args):
        # Game parameters
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        # Neural Net
        # Inputs
        self.input_boards = Input(shape=(self.board_x, self.board_y))
        inputs = Reshape((self.board_x, self.board_y, 1))(self.input_boards)

        bn1 = BatchNormalization()(inputs)
        conv1 = Conv2D(args.num_channels, kernel_size=3, strides=1, padding="same")(bn1)
        t = relu_bn(conv1)

        for _ in range(args.num_residual_layers):
            t = residual_block(t, filters=args.num_channels)

        self.pi = Dense(self.action_size, activation="softmax", name="pi")(policy_head(t))
        self.v = Dense(1, activation="tanh", name='v')(value_head(t))

        self.model = Model(
            inputs=self.input_boards,
            outputs=[self.pi, self.v]
        )
        self.model.compile(
            loss=["categorical_crossentropy", "mean_squared_error"],
            optimizer=Adam(args.lr)
        )
