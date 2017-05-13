# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method.

        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Do not change this method.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        oldFood = currentGameState.getFood()
        for ghost_state in newGhostStates:
            if ghost_state.scaredTimer > 1:
                continue
            if manhattanDistance(newPos, ghost_state.getPosition()) < 2 :
                return -1

        x, y = currentGameState.getWalls().asList()[-1]
        board_size = x + y
        bonus = (len(oldFood.asList()) - len(newFood.asList()))

        closest_food = board_size
        for food in oldFood.asList():
          closest_food = min(closest_food, manhattanDistance(food, newPos))

        food_heu = 1.0/(closest_food+1)
        score = successorGameState.getScore()
        if score>0:
          score_heu = 1 - 1.0/score
        elif score<0:
          score_heu = 1.0/((score-1)**2)
        else:
          score_heu = 0

        best_score = int( 100*food_heu + 50*score_heu + 100*bonus)
        return best_score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 1)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        best_state = float("-inf") # starting from pacmen, maximum
        bestAction = []
        agent = 0 #pacmen
        actions = gameState.getLegalActions(agent)

        for action in actions:
          successor = gameState.generateSuccessor(agent, action)
          #start with ghost 1 - random choice
          temp_state = minimax_state(0, gameState.getNumAgents(), successor, self.depth, self.evaluationFunction) 
          if temp_state > best_state:
            best_state = temp_state
            bestAction = action
        return bestAction
        
def minimax_state(agent, agentsCount, state, depth, evalFunc):
      
      agentList = range(agentsCount)

      if depth <= 0 or state.isWin() or state.isLose():
        return evalFunc(state)
        
      if agent == 0: #pacmen
        best_state = float("-inf") #max
      else:
        best_state = float("inf") #min
              
      actions = state.getLegalActions(agent)
      successors = [state.generateSuccessor(agent, action) for action in actions]
      for successor in successors:
        if agent == 0: # agent is pacmen
          best_state = max(best_state, minimax_state(agentList[agent+1], agentsCount, successor, depth, evalFunc))
        elif agent == agentList[-1]: # agent is the last ghost
          best_state = min(best_state, minimax_state(agentList[0], agentsCount, successor, depth - 1, evalFunc))
        else: # agent is middle ghost 
          best_state = min(best_state, minimax_state(agentList[agent+1], agentsCount, successor, depth, evalFunc))
      
      return best_state

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        
        


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

################################################
#    IGNORE THE CODE BELOW - DON'T CHANGE IT
################################################

def betterEvaluationFunction(currentGameState):
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

