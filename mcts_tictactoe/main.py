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

Notes from Alg4Dm:

Avoids exponential complexity via running m sims from current state

Each of these sims estimates value of Q(s, a) action value function for each node

Good starting exploration strategy: UCB exploration heuristic -- balances exploration vs exploitation

UCB: Q(s, a) + c * sqrt ( log N(s) / N(s, a) )

N(s) = summation over actions N(s, a) (parent vs child visit count)

where c is exploration parameter (frequency of unexplored actions)
That second part is called the exploration bonus

Low N(s, a) causes explosion in bonus making it favored

argmax(UCB) determines action to take

At each time step, we increment visit count N(s, a) and update Q(s, a)

If we reach maximum depth or a state we haven't yet explored:

    - if unexplored: initialize N(s, a) and Q(s, a) to zero
    - figure out how to estimate a value for this (through some rollout of policy ex. sparse sampling -- this is independent of your next action)

N is a dictionary for N[(s, a)], Q is a dictionary for Q[(s, a)]

Once you estimate one child, you move on to next one from the same base root node based on UCB

Once you sample all child nodes and get values, your exploration parameter is dampened and you just choose the best move by UCB

You expand another rollout from THE SAME ROOT NODE following the maximized UCB path. Run a new rollout and update both visitation count and Q (average)

Keep doing this and keep updating valuation for next node till you meet a constraint of some sort and you stop

Progressive widening where you incrementally increase depth

You run this at every time step to update based on opponent move

In 2 opponent game, assume opponent makes random move?


'''



class Node:
    def __init__(self, rep="", parent=None):
        self.parent = parent
        self.children = []
        self.N = 0 # number of visits
        self.W = 0 # number of wins
        self.str = rep

    def __str__(self):
        return self.str


def UCB_heuristic(node):
    pass

# TODO: evaluation function on node

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
        child.N += 1
        print(child)
        print("\n"*3 + "-"*10 + "\n"*3)
        curr_node = child
    else:
        break

if board.result() == 1:
    print("X Won") # alternate: has_x_won = board.has_won(tictactoe.X)
    curr_node.W += 1
elif board.result() == 2:
    print("O Won")
elif board.result() == 0:
    print("Draw")
