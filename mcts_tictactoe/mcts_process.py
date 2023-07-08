import random
from multiprocessing import Process, Event, Manager, Value
import copy
from env import StateFilterEntropy
from node import Node
import time
import numpy as np
import math
import ctypes
import requests
import json
import cv2
import torch

class MCTSProcess(Process):
    
    def __init__(self, proc_num, policy_dict, value_dict, child_value_dict, curr, model, device, mode="inference", pm_in_train=False, z_scr_val=True, console_viz=False, tree_viz=False):
        Process.__init__(self, daemon=True)
        self.event = Event()
        self.policy_dict = policy_dict
        self.value_dict = value_dict
        self.child_value_dict = child_value_dict
        self.cmd_action = Value(ctypes.c_int, -1)
        self.proc_num = proc_num
        self.curr = copy.deepcopy(curr) # root node
        self.console_viz = console_viz
        self.tree_viz = tree_viz
        self.model = model
        self.device = device
        self.mode = mode
        self.pm_in_train = pm_in_train
        self.z_scr_val = z_scr_val
        if self.z_scr_val:
            with open('val_stats.json', 'r') as fp:
                tmp_json = json.load(fp)
                self.val_stats = {int(k):list(v) for k,v in tmp_json.items()}
                print("Val Stats Loaded:", self.val_stats)

    def log(self, *args):
        if self.console_viz:
            for arg in args:
                print(arg, flush=True)

    """
    TODO:
    def finished(self):
        if termination criteria met, return true
    If all processes are finished, then select move and go on
    Main should not select when these end based on time, process should move on when a certain number of visits are made from root
    """


    def run(self):
        if self.curr.board.finished():
            return
        self.log('Worker process running...', self.proc_num)
        for act in [action for action in StateFilterEntropy.Actions if action not in [
                child.action for child in self.curr.children]]:
            new_env = copy.deepcopy(self.curr.board) # TODO: double copy memory hogging
            new_env.step(act)
            self.curr.children.append(Node(new_env, parent=self.curr, action=act))
            # print("\n\n\n\n" + "Expanding" + "\n\n\n\n")

        # timestamp = time.time()
        # max_search_time = 80
        # cover 10% of search tree till termination before executing move; not
        # counting precomputed stats
        # rollout_suggested_amt = (search_pctg := 0.1) * (4 ** self.curr.board.moves_left())
        # print("Rollout Suggested Amount", rollout_suggested_amt)

        rollout_count = 0

        # while rollout_count < rollout_suggested_amt and (time.time() - timestamp) < max_search_time:
        while True:
            last_node = self.curr

            '''
            Select
            '''

            log_msg = "\n"
            log_msg += "Process Num: " + str(self.proc_num) + " Rollout Count: " + str(rollout_count) + "\n"
            curr_counter = last_node.board.depth()

            while len(last_node.children) > 0:
                unvisited = [x for x in set([i for i in StateFilterEntropy.Actions]) if x not in set([child.action for child in last_node.children])] # unvisited actions

                for action in unvisited:
                    new_env = copy.deepcopy(last_node.board)
                    new_env.step(action)

                    last_node.children.append(
                        Node(new_env, parent=last_node, action=action))

                # Ensures children are sorted as UP, DOWN, RIGHT, LEFT
                last_node.children.sort(key=lambda val: {StateFilterEntropy.Actions.UP:0, StateFilterEntropy.Actions.DOWN:1, StateFilterEntropy.Actions.RIGHT:2, StateFilterEntropy.Actions.LEFT:3}[val.action])

                self.model.eval()
                board = copy.copy(last_node.board.get())
                board[last_node.board.pose()[0]][last_node.board.pose()[1]] = -1 # current position

                board = torch.as_tensor(
                    [np.expand_dims(board, 0)], dtype=torch.float).to(self.device)

                other = torch.as_tensor(
                    [[last_node.board.moves_left(), 0, 0, 0, 0]], dtype=torch.float).to(self.device)

                out = np.array(self.model(board, other)[0][0].tolist())

                out = (noise_frac:=(0.5 if self.pm_in_train else 0.3)) * np.array(np.random.dirichlet([alpha:=32 / (num_moves:=4)] * num_moves), dtype=np.float32) + (1-noise_frac) * out # dirichlet noise # TODO: modulate noise_frac based on whether policy model in training

                p_dict = dict(zip([act for act in StateFilterEntropy.Actions], out.tolist()))

                UCB_heuristics = [child.Q + (1 if self.mode=="train" and not self.pm_in_train else p_dict[child.action]) * (MAX_INT := 1e+12 if child.N == 0 else (cpuct:= (0.5*2**0.5 if not self.pm_in_train else 0.5*2**0.5) * math.sqrt(last_node.N) / (1 + child.N))) for child in last_node.children] # TODO: modulate cpuct (exploration) based on whether policy model in training

                """
                UCB_heuristics = [child.Q + p_dict[child.action] *
                        (MAX_INT := 1e+12 if child.N == 0 else ((c := 3 if rollout_count < 28 else 2) * 2**0.5) * math.sqrt(math.log(last_node.N) / child.N)) for child in last_node.children]
                """

                log_msg += "Actions" + str([child.action for child in last_node.children]) + "\n"
                log_msg += "UCB heuristics" + str(UCB_heuristics) + "\n"
                log_msg += "Q Value" + str([child.Q for child in last_node.children]) + "\n"
                log_msg += "Visitation" + str([child.N for child in last_node.children]) + "\n"
                log_msg += "P Dict" + str(p_dict) + "\n\n\n"

                # Selection of heuristicsfocus node based on UCB
                last_node = last_node.children[np.argmax(UCB_heuristics)]

            self.log(log_msg)

            returns = None

            if self.mode == "inference":
                '''
                Value Estimate
                '''
                if not last_node.board.finished():
                    for action in [i for i in StateFilterEntropy.Actions]:
                        new_env = copy.deepcopy(last_node.board)
                        new_env.step(action)
                        last_node.children.append(
                            Node(new_env, parent=last_node, action=action))

                    # self.model.eval()
                    board = copy.copy(last_node.board.get())
                    board[last_node.board.pose()[0]][last_node.board.pose()[1]] = -1 # current position

                    board = torch.as_tensor(
                        [np.expand_dims(board, 0)], dtype=torch.float).to(self.device)

                    other = torch.as_tensor(
                        [[last_node.board.moves_left(), 0, 0, 0, 0]], dtype=torch.float).to(self.device)

                    returns = float(self.model(board, other)[1][0]) # if game is over, replace with that
                else:
                    returns = last_node.board.returns() if not self.z_scr_val else (last_node.board.returns() - self.val_stats[last_node.board.depth()][0])/self.val_stats[last_node.board.depth()][1]
            else:

                '''
                Rollout
                '''
                # Run random-ish policy rollout till termination (no value network)
                prob_back = 0.1

                opp_action = {StateFilterEntropy.Actions.UP: StateFilterEntropy.Actions.DOWN, StateFilterEntropy.Actions.DOWN: StateFilterEntropy.Actions.UP, StateFilterEntropy.Actions.RIGHT: StateFilterEntropy.Actions.LEFT, StateFilterEntropy.Actions.LEFT: StateFilterEntropy.Actions.RIGHT}

                while not last_node.board.finished():
                    action_space = [i for i in StateFilterEntropy.Actions]
                    weights = [prob_back if action == opp_action[last_node.action] else (1 - prob_back)/3 for action in action_space]
                    assert abs(sum(weights) - 1) < 0.01
                    chosen_action = np.random.choice(action_space, 1, p=weights)[0]
                    new_env = copy.deepcopy(last_node.board)
                    new_env.step(chosen_action)
                    child = Node(new_env, parent=last_node, action=chosen_action)
                    last_node.children.append(child)
                    last_node = child
                    """
                    weights = []
                    action_space = [i for i in StateFilterEntropy.Actions]
                    for action in action_space:
                        new_env = copy.deepcopy(last_node.board)
                        new_env.step(action)
                        last_node.children.append(
                            Node(new_env, parent=last_node, action=action))
                        weights.append(prob_back if action == opp_action[last_node.action] else (1 - prob_back)/3)
                        # Choose new last_node from random policy rollout
                    # last_node.action low probability repeat
                    assert abs(sum(weights) - 1) < 0.02
                    last_node = np.random.choice(last_node.children, 1, p=weights)[0]
                    """
                returns = last_node.board.discounted_returns(curr_counter)

            '''
            Backpropagate
            '''
            while True:
                last_node.N += 1
                last_node.Q += (returns - last_node.Q) / last_node.N
                if last_node.parent is None:
                    break
                last_node = last_node.parent

            rollout_count += 1
            # self.log("\n\n\n\nRollout Finished: ", rollout_count, "\n\n\n\n")

            '''
            Build Viz Tree
            '''

            if self.tree_viz:
                self.log("Starting Encoding")
                self.graph_data = []
                self.node_count = 1
                label = [[0, "🔴"]]

                def recursive_encode(node):
                    if len(node.children) == 0:
                        return
                    for child in node.children:
                        if child.N > 0:
                            label.append([child.id_c, "🔴"])
                        else:
                            label.append([child.id_c, " "])
                        self.graph_data.append([node.id_c, child.id_c])
                        recursive_encode(child)
                        self.node_count += 1

                recursive_encode(self.curr)
                # print(graph_data)
                try:
                    s1 = requests.get(
                        "http://127.0.0.1:8050/graph/" +
                        str(self.node_count) + ";" + json.dumps(self.graph_data) +
                        ";" + json.dumps(label),
                        headers={"content-type": "application/x-www-form-urlencoded"})
                except Exception as e:
                    self.log("Tree Viz Failed")


            '''
            This thread receives stop command
            This thread sends back visitation
            It receives commanded action
            This thread sends back finished
            '''
            if self.event.is_set():

                '''
                Update MCTS child visitation
                '''
                # self.log("VISITATION", [child.N for child in self.curr.children])
                self.policy_dict[self.proc_num] = [child.N for child in self.curr.children]
                self.value_dict[self.proc_num] = self.curr.Q
                self.child_value_dict[self.proc_num] = [child.Q for child in self.curr.children]
                self.event.clear()
                self.event.wait()
                # self.log("COMMANDED ACTION:", self.cmd_action.value)

                '''
                Reset root node
                '''
                selected_child = self.curr.children[self.cmd_action.value]
                selected_child.parent = None
                selected_child.action = None
                self.curr = selected_child
                rollout_count = 0
                # env.step([action for action in StateFilterEntropy.Actions][self.cmd_action.value])
                # self.log("Action Set")
                self.event.clear()
                '''
                while self.event.is_set(): # main thread clears once action cmd determined
                    pass

                # TODO: apply cmd_action
                self.cmd_action = -1
                rollout_count = 0
                '''


            # print(len(sorted(np.unique(np.array([item for sublist in graph_data for item in sublist])))) == node_count, len(np.unique(np.array([item for sublist in graph_data for item in sublist]))), node_count)

        self.log('Worker closing down', self.proc_num)
