# Read: https://towardsdatascience.com/monte-carlo-tree-search-an-introduction-503d8c04e168

# Based on: https://www.baeldung.com/java-monte-carlo-tree-search

from tictactoe import Board
import random

board = Board(dimensions=(3, 3), x_in_a_row=3)

'''
Need to build out tree

Each tree has n number of nodes

Each node has one parent

Each node has visit iterations and number of wins

'''

class Node:
    def __init__(self, rep="", parent=None):
        self.parent = parent
        self.children = []
        self.visit_iterations = 0
        self.number_wins = 0
        self.str = rep

    def __str__(self):
        return self.str

'''
Ex Tree
'''

head = Node(rep=str(board), parent=None)

last_node = head
while board.result() == None: # while game not finished
    board.push(random.choice([move for move in board.possible_moves()]))
    child = Node(rep=str(board), parent=last_node)
    last_node.children.append(child)
    last_node = child

# print random rollout
curr_node = head
while True:
    if len(curr_node.children) > 0:
        child = random.choice(curr_node.children)
        print(child)
        print("\n"*3 + "-"*10 + "\n"*3)
        curr_node = child
    else:
        break

if board.result() == 1:
    print("X Won") # alternate: has_x_won = board.has_won(tictactoe.X)
elif board.result() == 2:
    print("O Won")
elif board.result() == 0:
    print("Draw")
