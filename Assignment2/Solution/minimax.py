import sys
import numpy as np
from copy import deepcopy
from math import inf as infinity

def find_box(x):
    coor = []

    if x % 3 == 0:
        coor.append(x)
        coor.append(x + 3)
    if x % 3 == 1:
        coor.append(x - 1)
        coor.append(x + 2)
    else:
        coor.append(x - 2)
        coor.append(x + 1)
    return coor[0], coor[1]


class Board:
    def __init__(self, file_name=None, grid=None):
        self.grid = np.zeros((9, 9))
        self.legal_run = False
        self.legal_moves = None
        self.eval_v = -infinity

        if file_name is not None:
            self.read_world(file_name)
        else:
            self.grid = deepcopy(grid)

    def read_world(self, file_name):
        with open(file_name, "r") as f:
            for i, line in enumerate(f):
                vals = np.array(line.split())
                self.grid[i] = vals

    def print_board(self):
        print(self.grid)
        print()

    def get_legal(self):
        if self.legal_run:
            return self.legal_moves
        self.legal_run = True
        legal_pos_x, legal_pos_y = np.where(self.grid == 0)
        num_pos = legal_pos_x.shape[0]
        legal_moves = {(legal_pos_x[i], legal_pos_y[i]): [1, 2, 3, 4, 5, 6, 7, 8, 9] for i in range(num_pos)}
        for i in range(num_pos):
            x, y = legal_pos_x[i], legal_pos_y[i]
            x0, x1 = find_box(x)
            y0, y1 = find_box(y)
            box = self.grid[x0:x1, y0:y1]
            for num in range(1, 10):
                if (num in self.grid[x]) or (num in self.grid[:, y] or (num in box)):
                    legal_moves[x, y].remove(num)
            if len(legal_moves[x, y]) == 0:
                del legal_moves[x, y]

        self.legal_moves = legal_moves
        return self.legal_moves

class MaxPlayer:
    def __init__(self):
        self.node_count = 1

    def eval(self, level):
        # If level is an even number, MAX Player wins (Means the last move was made by the MAX Player)
        eval_v = -1
        if level % 2 == 0:
            eval_v = 1

        return eval_v

    def search(self, board, board_level=1):
        legal_moves = board.get_legal()  # Number of legal moves left

        if len(legal_moves) == 0:
            # If no more legal moves (game ended)
            return self.eval(board_level)
        else:
            # For empty cells that we can write legal numbers in
            for x, y in legal_moves:
                # For each legal number we can write, create a new state
                for val in legal_moves[x, y]:
                    b = Board(grid=board.grid)
                    self.node_count += 1
                    b.grid[x, y] = val
                    board.eval_v = max(board.eval_v, self.search(b, board_level + 1))
            return board.eval_v

    def does_win(self, board):
        board.eval_v = max(board.eval_v, self.search(board))
        return board.eval_v


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Not enough arguments")
        sys.exit(-1)

    filename = sys.argv[1]
    b = Board(filename)
    # b.print_board()
    p = MaxPlayer()
    res = p.does_win(b)
    if res == 1:
    	print(res)
    else:
    	print(2)
    # print(p.node_count)
    # print("Search end")