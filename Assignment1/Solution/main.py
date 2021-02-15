# Uğur Ali Kaplan
# 150170042


import numpy as np
from copy import deepcopy
from directions import *
from itertools import combinations_with_replacement, permutations, count
from queue import Queue, LifoQueue, PriorityQueue
import sys
from time import perf_counter_ns

entry = count(0, 1)
MAX_AGENTS = 9
sample_pool = ["U"] * MAX_AGENTS + ["D"] * MAX_AGENTS + ["L"] * MAX_AGENTS + ["R"] * MAX_AGENTS + ["P"] * MAX_AGENTS

# Mapping from string to direction constants
move_map = {
    "U": U,
    "D": D,
    "R": R,
    "P": P,
    "L": L
}

# Define Classes


class State:
    def __init__(self, agents=list(), goals=list(), st_name=None, heuristic=0, parent=None, step_cost=0, level=0):
        self.agents = deepcopy(agents)
        self.goals = deepcopy(goals)
        self.st_name = st_name
        self.heuristic = heuristic
        self.parent = parent
        self.children = list()
        self.step_cost = step_cost
        self.level = level

    def check_success(self):
        """
        Check if all agents accessed their goals
        :return: True if all agents accessed their goals, False otherwise
        """
        for i in range(len(self.agents)):
            if not (self.agents[i] == self.goals[i]).all():
                return False
        return True

    def __eq__(self, st):
        agents = st.agents

        if len(agents) != len(self.agents):
            return False

        for i in range(len(agents)):
            if not (self.agents[i] == agents[i]).all():
                return False

        return True


class prioritizedState:
    def __init__(self, state):
        self.state = state
        self.heuristic = state.heuristic
        self.step_cost = state.step_cost
        self.level = state.level

    def __eq__(self, st2):
        return (self.heuristic + self.step_cost) == (st2.heuristic + st2.step_cost)

    def __neq__(self, st2):
        return (self.heuristic + self.step_cost) != (st2.heuristic + st2.step_cost)

    def __le__(self, st2):
        return (self.heuristic + self.step_cost) <= (st2.heuristic + st2.step_cost)

    def __ge__(self, st2):
        return (self.heuristic + self.step_cost) >= (st2.heuristic + st2.step_cost)

    def __gt__(self, st2):
        return (self.heuristic + self.step_cost) > (st2.heuristic + st2.step_cost)

    def __lt__(self, st2):
        return (self.heuristic + self.step_cost) < (st2.heuristic + st2.step_cost)


class World:
    def __init__(self, file):
        """
        Initializes the maze and the agents.
        :param file: Input file name
        """

        # File to read
        self.file = file

        # These will be filled in in the read_world function
        self.rows, self.columns, self.root = 0, 0, State(st_name="Begin")  # agents is a list of agent instances
        self.walls = list()  # Store wall positions
        self.empty_count = 0  # Store wall positions
        self.read_world()  # Read and fill in the attributes

    def read_world(self):
        """
        Reads the world information, writes the walls and empty spaces into memory
        :return:
        """
        with open(self.file, "r") as f:
            self.rows, self.columns, agent_no = map(int, f.readline().split())
            # Create goals and agents lists with the "agent_no" number of elements
            goals = [0] * agent_no
            agents = [0] * agent_no

            # Iterate through the given maze
            for row, line in enumerate(f):
                for col, val in enumerate(line.split()):
                    if val == "W":
                        self.walls.append((int(row), int(col)))
                    elif val == "E":
                        self.empty_count += 1
                    elif val[0] == "G":
                        # For G1, fill in the goals[0]; For G2, fill in the goals[1] and so on
                        goals[int(val[-1]) - 1] = np.array([int(row), int(col)])
                    elif val[0] == "A":
                        # For A1, fill in the agents[0]; For A2, fill in the agents[1] and so on
                        agents[int(val[-1]) - 1] = np.array([int(row), int(col)])

            for i in range(len(agents)):
                self.root.agents.append(agents[i])
                self.root.goals.append(goals[i])

    def change_world(self, file):
        """
        Calls the constructor again
        :param file: Filename for the new world
        :return:
        """
        self.__init__(file)

    def check_no_collision(self, next_st_name, cur_state):
        """
        Given the current state and the next possible state, checks if there is an illegal move
        :return: False if there is collision, True if no collision
        """
        next_st_name = list(next_st_name)  # Split the string into characters
        cur_pos_l = list()
        next_pos = list()  # It will store the next positions of the agents
        for i in range(len(next_st_name)):
            cur_pos = cur_state.agents[i]
            cur_pos_l.append(tuple(cur_pos))
            new_pos = tuple(cur_pos + move_map[next_st_name[i]])  # Move the agent
            if new_pos in self.walls or new_pos in next_pos:
                return False  # Collision if another agent moved there or there is a wall
            next_pos.append(new_pos)

        # Any illegal swaps?
        count_p = max(1, next_st_name.count("P"))
        if len(set(cur_pos_l).intersection(set(next_pos))) > count_p:
            return False

        return True  # No Collision

    # noinspection PyMethodMayBeStatic
    def calculate_heuristic(self, next_st_name, cur_state):
        """
        Calculates the heuristic using Manhattan distance
        :param next_st_name: State to be evaluated
        :param cur_state: State object
        :return:
        """
        moves = [move_map[s] for s in list(next_st_name)]  # Move values
        total_cost = 0  # Total cost of the state

        for i in range(len(moves)):
            # Calculate Manhattan Distance
            dist = np.sum(np.abs(cur_state.goals[i] - (cur_state.agents[i] + moves[i])))
            total_cost += dist

        return total_cost

    def generate_states(self, cur_state):
        """
        Given an state object, creates the next possible states
        :param cur_state:
        :return:
        """

        # Generate all possible next states
        all_state_names = ["".join(x) for x in set(permutations(sample_pool, len(cur_state.agents)))]

        # If new states are valid (no collisions), add them into a list
        valid_state_names = [''.join(s) for s in all_state_names if self.check_no_collision(s, cur_state)]

        # Create State objects from the names
        for state_name in valid_state_names:
            moves = [move_map[s] for s in state_name]
            new_state = State(agents=cur_state.agents,
                              goals=cur_state.goals,
                              st_name=state_name,
                              heuristic=self.calculate_heuristic(state_name, cur_state),
                              parent=cur_state,
                              step_cost=cur_state.step_cost+np.sum(np.abs(moves)),
                              level=cur_state.level+1)

            new_state.agents = [new_state.agents[i] + moves[i] for i in range(len(new_state.agents))]
            cur_state.children.append(new_state)

        # Also return the generated valid State objects
        return cur_state.children

    def bfs(self):
        # Create a queue and add the root
        frontier = Queue()
        frontier.put(self.root)
        # Analysis
        # max_nodes = frontier.qsize()
        # start_t = perf_counter_ns()

        # BFS Algorithm Start
        explored = list()
        solution = None
        while not frontier.empty():
            # Analysis
            # max_nodes = max(max_nodes, frontier.qsize())
            cur_state = frontier.get()
            if cur_state in explored:
                continue

            if cur_state.check_success():
                solution = cur_state
                break

            self.generate_states(cur_state)

            for state in cur_state.children:
                frontier.put(state)

            explored.append(cur_state)

        # Analysis
        # end_t = perf_counter_ns()

        path = list()
        while solution is not None:
            path.append(solution.st_name)
            solution = solution.parent

        print(f"{len(path) - 1} {len(self.root.agents)}")
        for i in range(len(path) - 2, -1, -1):
            print(path[i])

        # Analysis
        # print(f"BFS Generated Nodes: {len(explored) + frontier.qsize()}")
        # print(f"Number of Expanded Nodes: {len(explored)}")
        # print(f"Maximum number of nodes kept in memory: {max_nodes}")
        # print(f"Running Time: {end_t - start_t}")

    def dfs(self):
        # Create a stack and add the root
        frontier = LifoQueue()
        frontier.put(self.root)

        # Generate the first children
        self.generate_states(self.root)

        # # Analysis
        # max_nodes = frontier.qsize()
        # start_t = perf_counter_ns()

        # DFS Algorithm Start
        explored = list()
        solution = None
        while not frontier.empty():
            # Analysis
            # max_nodes = max(max_nodes, frontier.qsize())
            cur_state = frontier.get()
            if cur_state in explored:
                continue

            if cur_state.check_success():
                solution = cur_state
                break

            self.generate_states(cur_state)

            for state in cur_state.children:
                frontier.put(state)

            explored.append(cur_state)

        path = list()
        while solution is not None:
            path.append(solution.st_name)
            solution = solution.parent

        # Analysis
        # end_t = perf_counter_ns()

        print(f"{len(path) - 1} {len(self.root.agents)}")
        for i in range(len(path) - 2, -1, -1):
            print(path[i])

        # Analysis
        # print(f"DFS Generated Nodes: {len(explored) + frontier.qsize()}")
        # print(f"Number of Expanded Nodes: {len(explored)}")
        # print(f"Maximum number of nodes kept in memory: {max_nodes}")
        # print(f"Running Time: {end_t - start_t}")

    def a_star(self, tie_breaker="h"):
        """
        A* algorithm to find the solution.
        :param tie_breaker: "h" for using the heuristic cost, "s" for selecting the one with maximum amount of moves
        :return:
        """
        # Create a priority queue and add the root
        frontier = PriorityQueue()
        # Elements in PriorityQueue is in the form of (priority_number, data)
        if tie_breaker == "h":
            frontier.put((prioritizedState(self.root), self.root.heuristic))
        elif tie_breaker == "l":
            frontier.put((prioritizedState(self.root), self.root.heuristic + self.root.level))

        # Generate the first children
        self.generate_states(self.root)
        # Analysis
        # max_nodes = frontier.qsize()
        # start_t = perf_counter_ns()

        # A* Algorithm Start
        explored = list()
        solution = None
        abc = 0
        while not frontier.empty():
            # Analysis
            # max_nodes = max(max_nodes, frontier.qsize())

            cur_state = frontier.get()[0].state

            # Analysis
            # abc += 1
            # if abc % 1000:
            #     print(f"Manhattan Distance: {cur_state.heuristic}, Step Cost: {cur_state.step_cost}, State Level: {cur_state.level}")

            if cur_state.check_success():
                if solution is None:
                    solution = cur_state

                item = frontier.get()

                if item[0].level < cur_state.level:
                    frontier.put(item)
                    continue
                else:
                    break

            self.generate_states(cur_state)

            for state in cur_state.children:
                if state in explored:
                    continue
                if tie_breaker == "h":
                    frontier.put((prioritizedState(state), state.heuristic))
                elif tie_breaker == "l":
                    frontier.put((prioritizedState(state), state.heuristic + state.level))

            explored.append(cur_state)

        path = list()
        while solution is not None:
            path.append(solution.st_name)
            solution = solution.parent

        end_t = perf_counter_ns()

        print(f"{len(path) - 1} {len(self.root.agents)}")
        for i in range(len(path) - 2, -1, -1):
            print(path[i])

        # Analysis
        # print(f"A* Generated Nodes: {len(explored) + frontier.qsize()}")
        # print(f"Number of Expanded Nodes: {len(explored)}")
        # print(f"Maximum number of nodes kept in memory: {max_nodes}")
        # print(f"Running Time: {end_t - start_t}")


if __name__ == '__main__':
    # argc = len(sys.argv)
    # if argc < 2:
    #     print("No input file")
    #     exit(0)
    #
    # in_file = sys.argv[1]
    # out_file = "output.txt"
    # if argc >= 3:
    #     out_file = sys.argv[2]
    #
    # # in_file = "input_3.txt"
    # # out_file = "maze_3.txt"
    #
    # sys.stdout = open(out_file, "w")
    # # w = World(in_file)
    # # w.bfs()
    # # del w
    # #
    # # w = World(in_file)
    # # w.dfs()
    # # del w
    # #
    # # w = World(in_file)
    # # w.a_star("h")
    # # del w
    #
    # w = World(in_file)
    # w.a_star("l")
    # del w
    #
    # sys.stdout.close()
    print("You should not use main.py to run the algorithms.\nUse the specific files for different algorithms!")
