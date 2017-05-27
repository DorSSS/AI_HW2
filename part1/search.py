# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

class SearchNode:
    '''
    This class represents a node in the search graph
    '''

    def __init__(self, state, action=None, cost=0, parent_node=None):
        self.state = state
        self.action = action
        self.cost = cost
        self.parent_node = parent_node

    def get_path_from_root(self):
        path = [self]
        if self.parent_node is not None:
            path = self.parent_node.get_path_from_root() + path
        return path

    def get_actions_chain(self):
        '''
        :return: list of actions from the root to this node (including)
         note that the first element is not included because the action that leads to the first node (i.e. the
         initial state) is always None.
        '''
        return [node.action for node in self.get_path_from_root()][1:]

    def __eq__(self, other):
        return self.state == other.state

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.state)
                
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    stack = util.Stack()
    initial_node = SearchNode(problem.getStartState())
    stack.push(initial_node)
    visited_nodes = set()
    while not stack.isEmpty():
        curr_node = stack.pop()
        if curr_node not in visited_nodes:
            visited_nodes.add(curr_node)

            if problem.isGoalState(curr_node.state):
                return curr_node.get_actions_chain()

            for next_state, next_act, next_cost in problem.getSuccessors(curr_node.state):
                stack.push(SearchNode(next_state, next_act, next_cost, curr_node))
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    queue = util.Queue()
    initial_node = SearchNode(problem.getStartState())
    queue.push(initial_node)
    visited_nodes = set()
    while not queue.isEmpty():
        curr_node = queue.pop()
        if curr_node not in visited_nodes:
            visited_nodes.add(curr_node)
            
            if problem.isGoalState(curr_node.state):
                return curr_node.get_actions_chain()

            for next_state, next_act, next_cost in problem.getSuccessors(curr_node.state):
                queue.push(SearchNode(next_state, next_act, next_cost, curr_node))
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    priority_queue = util.PriorityQueue()
    initial_node = SearchNode(problem.getStartState())
    priority_queue.push(initial_node, 0)
    visited_nodes = set()

    while not priority_queue.isEmpty():
        curr_node = priority_queue.pop()

        if curr_node not in visited_nodes:
            visited_nodes.add(curr_node)

            if problem.isGoalState(curr_node.state):
                return curr_node.get_actions_chain()

            for next_state, next_act, next_cost in problem.getSuccessors(curr_node.state):
                next_node = SearchNode(next_state, next_act, next_cost + curr_node.cost, curr_node)
                priority_queue.push(next_node, next_node.cost)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    problem = problem #type: SearchProblem
    explored_nodes = [] 
    queue = util.PriorityQueue()
    current_node = SearchNode(problem.getStartState())
    current_node.cost_from_root = 0
    while not (problem.isGoalState(current_node.state)):
        explored_nodes.append(current_node)
        for next_state, next_act, next_cost in problem.getSuccessors(current_node.state):
            next_node = SearchNode(next_state,next_act, next_cost, current_node)
            next_node.cost_from_root = current_node.cost_from_root + next_node.cost
            if (not next_node in explored_nodes) and (not next_node.state in queue.heap):
                is_in_queue = False
                for queue_node in queue.heap:
                    if next_node == queue_node[2]:
                        is_in_queue = True  #find if not already in queue
                        break
                if (is_in_queue == False):
                    queue.push(next_node, next_node.cost_from_root + heuristic(next_state,problem))
                elif queue_node[0] > next_node.cost_from_root + heuristic(next_state,problem):
                    queue.update(next_node, next_node.cost_from_root + heuristic(next_state,problem))
        current_node = queue.pop()

    return current_node.get_actions_chain()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch