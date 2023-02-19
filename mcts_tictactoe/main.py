# based around Alg4DM book

'''
Not a perfect MCTS, just a notepad for vtol surveyor planner. this first version has a couple big bugs, fixed in other file



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

# TODO: run tests and find possible bugs 

from tictactoe import Board
import numpy as np
import random
import copy
import math
import time

class Node:
    def __init__(self, rep, parent=None, move=None):
        self.parent = parent
        self.children = []
        self.N = 0.0 # number of visits
        self.Q = 0 # mean value of node
        self.board = rep
        self.move = move

    def __str__(self):
        return str(self.board)


agent_wins = 0
for j in range(500):
    board = Board(dimensions=(3, 3), x_in_a_row=3)
    c=2 * 2**0.5 # exploration parameter

    while board.result() == None: # while game not finished
        rollout_count = 0
        board.push(random.choice([move for move in board.possible_moves()])) # player move (X)
        if board.result() != None: break
        # print("Player move: \n", board)

        curr = Node(rep=board, parent=None)

        for move in board.possible_moves():
            new_board = copy.deepcopy(board)
            new_board.push(move)
            curr.children.append(Node(rep=new_board, parent=curr, move=move))

        timestamp = time.time()

        while time.time() - timestamp < 0.2: # x seconds for computer to make move

            curr.N += 1
            UCB_heuristics = [child.Q + (1e+12 if child.N == 0 else c * math.sqrt(math.log(curr.N) / child.N)) for child in curr.children] # Q(s, a) + c * sqrt ( log N(s) / N(s, a) )
            # Run a rollout
            selected_child = curr.children[np.argmax(UCB_heuristics)]
            selected_child.N += 1
            rollout_board = copy.deepcopy(selected_child.board)
            while rollout_board.result() == None: # while game not finished
                for i in range(2): # first opponent move, then agent move
                    if rollout_board.result() == None:
                        rollout_board.push(random.choice([move for move in rollout_board.possible_moves()]))
                # print(rollout_board)

            # [win, lose, draw] -> reward of [1, -1, 0]
            reward = {1:-1, 2:1, 0:0}[rollout_board.result()]
            # selected_child.Q += selected_child.Q * selected_child.N / (selected_child.N + 1) + reward / (selected_child.N + 1)
            selected_child.Q += (reward - selected_child.Q) / selected_child.N
            # (1 + 2 + 3) / 3 + 4 / 4

            rollout_count += 1
        # print("%d rollouts performed" % rollout_count)

        move = curr.children[np.argmax([child.Q for child in curr.children])].move
        board.push(move)
        # print(board)
        # print("\n"*3 + "-"*10 + "\n"*3)


    if board.result() == 1:
        # print("X Won") # alternate: has_x_won = board.has_won(tictactoe.X)
        agent_wins -= 1
    elif board.result() == 2:
        # print("O Won")
        agent_wins += 1
    elif board.result() == 0:
        # print("Draw")
        agent_wins -= 0.2

    # print("\n\n\n", board)

    print("Games Played: ", j, "; Win Differential: ", agent_wins)

'''

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

'''
