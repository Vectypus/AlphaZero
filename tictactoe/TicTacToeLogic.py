class Board():
    """
    Board class for the game of TicTacToe.

    Default board size is 3x3.
    Board data: 1='O', -1='X', 0=empty
    Squares are stored and manipulated as (x,y) tuples.

    Author: Evgeny Tyurin, github.com/evg-tyurin
    Date: Jan 5, 2018.

    Modified by Victor Taillieu, github.com/Vectypus

    Based on the board for the game of Othello by Eric P. Nichols.
    """

    def __init__(self, n=3):
        """
        Set up initial board configuration.
        """

        self.n = n

        # Create the empty board array
        self.squares = [None] * self.n
        for i in range(self.n):
            self.squares[i] = [0] * self.n

    def __getitem__(self, index):
        """
        Add [][] indexer syntax for Board.
        """

        return self.squares[index]

    def get_legal_moves(self):
        """
        Return the list of all legal moves.
        """

        moves = []

        # Get all the empty squares
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    moves.append((x, y))
        return moves

    def has_legal_moves(self):
        """
        Check for legal moves.
        """

        return len(self.get_legal_moves()) != 0

    def is_win(self, player):
        """
        Check if the given player has a triplet.

        Parameters:
            player: 1='O', -1='X'
        """

        # Check for vertical triplet
        for y in range(self.n):
            count = 0
            for x in range(self.n):
                if self[x][y] == player:
                    count += 1
            if count == self.n:
                return True

        # Check for horizontal triplet
        for x in range(self.n):
            count = 0
            for y in range(self.n):
                if self[x][y] == player:
                    count += 1
            if count == self.n:
                return True

        # Check for diagonal triplet
        count = 0
        for i in range(self.n):
            if self[i][i] == player:
                count += 1
        if count == self.n:
            return True

        count = 0
        for i in range(self.n):
            if self[i][self.n - i - 1] == player:
                count += 1
        if count == self.n:
            return True

        return False

    def execute_move(self, move, player):
        """
        Perform the given move on the board.

        Parameters:
            move: square (x, y)
            player: 1='O', -1='X'
        """

        x, y = move
        # Check if square is empty
        assert self[x][y] == 0

        self[x][y] = player
