import logging

import click
import coloredlogs

from Coach import Coach
from connect4.Connect4Game import Connect4Game
from connect4.keras.NNet import NNetWrapper as Connect4NNet
from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.keras.NNet import NNetWrapper as TicTacToeNNet
from utils import dotdict

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


@click.command()
@click.argument('game', type=click.Choice(['tictactoe', 'connect4']))
@click.option(
    '-iters', '--iterations',
    default=3, show_default=True,
    help='Number of iterations'
)
@click.option(
    '-eps', '--episodes',
    default=25, show_default=True,
    help='Number of self-play games per iteration'
)
@click.option(
    '-mcts', '--mcts-simulations',
    default=25, show_default=True,
    help='Number of MCTS simulations per turn'
)
@click.option(
    '--checkpoint',
    type=click.Path(file_okay=False),
    default='./temp/',
    help='Path to save checkpoints'
)
@click.option(
    '--load-model',
    is_flag=True,
    help='Load model from checkpoint'
)
@click.option(
    '--load-model-file',
    type=(str, str),
    default=('./checkpoints', 'checkpoint.h5'),
    help='Folder and file name of model to load'
)
@click.option(
    '--temperature-threshold',
    default=15, show_default=True,
    help='Number of moves in an episode before the temperature changes from 1 to 0'
)
@click.option(
    '--update-threshold',
    default=0.6, show_default=True,
    help='Minimal win rate to accept new network'
)
@click.option(
    '--cpuct',
    default=1, show_default=True,
    help='Level of exploration'
)
@click.option(
    '--games-compare',
    default=40, show_default=True,
    help='Number of games to play to determine if new network is accepted'
)
@click.option(
    '--iters-for-train',
    default=20, show_default=True,
    help='Number of iterations used for training'
)
@click.option(
    '--max-queue-length',
    default=200000, show_default=True,
    help='Maximum number of game examples for training'
)
def main(game, iterations, episodes, mcts_simulations, checkpoint, load_model, load_model_file, temperature_threshold,
         update_threshold, cpuct, games_compare, iters_for_train, max_queue_length):
    args = dotdict({
        'numIters': iterations,
        'numEps': episodes,
        'numMCTSSims': mcts_simulations,
        'checkpoint': checkpoint,
        'load_model': load_model,
        'load_folder_file': load_model_file,
        'tempThreshold': temperature_threshold,
        'updateThreshold': update_threshold,
        'cpuct': cpuct,
        'arenaCompare': games_compare,
        'numItersForTrainExamplesHistory': iters_for_train,
        'maxlenOfQueue': max_queue_length
    })

    if game == "tictactoe":
        Game = TicTacToeGame
        nn = TicTacToeNNet
    elif game == "connect4":
        Game = Connect4Game
        nn = Connect4NNet

    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
