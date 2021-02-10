#!/usr/bin/python

import numpy as np
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math
from queue import Queue

# Throughout this file, ASP means adversarial search problem.

# Constants that define the winning and losing scores of the game
WINNING_SCORE = 100000000
LOSING_SCORE = -100000000
# Constants defining the player spaces, opponent spaces, and articulation points
STARTING_SPACE = 0
PLAYER_SPACE = 1
OPPONENT_SPACE = -1
ARTICULATION = -2
BATTLEFRONT = -3


class StudentBot:
    """ Write your student bot here"""

    def __init__(self):
        self.BOT_NAME = "SmartBot v1"
        self.game_count = 0

    def compute_distances(self, state, ptm, asp):
        """
        Computes the distances for each location on the board, for the given player. Tested and functional
        Inputs:
            - state: A starting state for the voronoi computation
            - ptm: Index for the player to move
        Returns:
            - Distances: An array where every reachable point has a distance, all others are set to 0
        """
        board = np.array(state.board)
        distances = np.zeros((board.shape[0], board.shape[1]))
        frontier = Queue()
        # Adding the state to the frontier
        frontier.put(state)
        # Executing a breadth-first search to populate the board with distances
        while not frontier.empty():
            curr_state = frontier.get()
            board = np.array(curr_state.board)
            loc = curr_state.player_locs[ptm]
            # Getting a list of possible move actions to take
            possibilities = list(TronProblem.get_safe_actions(board, loc))
            # Adding items to the frontier, if they have not yet been visited
            for action in possibilities:
                possible_state = asp.transition(curr_state, action)
                # Setting to the current player for ptm, to ensure that the same player keeps moving
                possible_state.ptm = ptm
                # Getting the new location
                new_loc = possible_state.player_locs[ptm]
                # Ignoring if this state has already been visited
                if distances[new_loc[0], new_loc[1]] != 0:
                    continue
                distances[new_loc[0], new_loc[1]] = distances[loc[0], loc[1]] + 1
                # Adding to the frontier
                frontier.put(possible_state)
        return distances

    def compute_articulations(self, voronoi_region, location):
        """
        Using a NumPy array, computes the size of the "chamber" that the bot is currently in.
        The chamber is a region that is cut off by a series of articulation points. In addition, returns
        a list of articulation points and a visited list to allow for additional searching into the additional
        chambers
        Inputs:
            - voronoi_region: A board where the player's voronoi regions are labeled with a "1",
              the opponent's voronoi regions are labeled with a "-1", and all unreachable squares are
              labeled with np.inf
            - location: The player location
        Outputs:
            - visited: A set of visited cell locations
            - num_in_chamber: Number of spaces in the player's current chamber
            - articulations: A set of articulation points. Each point is represented as a (row, col) tuple
            - voronoi_region: The voronoi region with the articulation point represented by the respective labelling
        """
        # Frontie and visited set to conduct the BFS. Articulation set to keep track of all articulations
        frontier = Queue()
        visited = set()
        articulations = set()
        frontier.put((location[0], location[1]))
        # Counter to keep track of the size of the current chamber size
        num_in_chamber = 0
        # Setting the player position to be the player position constant
        voronoi_region[location[0], location[1]] = STARTING_SPACE
        # Conducting the BFS
        while not frontier.empty():
            curr_location = frontier.get()
            row = curr_location[0]
            col = curr_location[1]
            # Ignoring if this state has already been visited
            if (row, col) in visited:
                continue
            # Adding the current location tuple to the visited set
            visited.add((row, col))
            # If we are at the starting location, add all neighbors and continue counting
            if voronoi_region[row][col] == STARTING_SPACE:
                # Handling the special case when the current location is itself an articulation point
                if (
                    voronoi_region[row - 1][col] == np.Inf
                    and voronoi_region[row + 1][col] == np.Inf
                ):
                    # Checking if the right and left spaces are free spaces. If so, they are articulations and are added as such
                    if voronoi_region[row][col - 1] == PLAYER_SPACE:
                        voronoi_region[row][col - 1] = ARTICULATION
                        articulations.add((row, col - 1))
                        visited.add((row, col - 1))
                    if voronoi_region[row][col + 1] == PLAYER_SPACE:
                        voronoi_region[row][col + 1] = ARTICULATION
                        articulations.add((row, col + 1))
                        visited.add((row, col + 1))
                    continue
                if (
                    voronoi_region[row][col - 1] == np.Inf
                    and voronoi_region[row][col + 1] == np.Inf
                ):
                    # Checking if the up and down spaces are free spaces. If so, they are articulations and are added as such
                    if voronoi_region[row - 1][col] == PLAYER_SPACE:
                        voronoi_region[row - 1][col] = ARTICULATION
                        articulations.add((row - 1, col))
                        visited.add((row - 1, col))
                    if voronoi_region[row + 1][col] == PLAYER_SPACE:
                        voronoi_region[row + 1][col] = ARTICULATION
                        articulations.add((row + 1, col))
                        visited.add((row + 1, col))
                    continue
                # Otherwise, adding all neighbors as usual
                if voronoi_region[row - 1][col] == PLAYER_SPACE:
                    frontier.put((row - 1, col))
                if voronoi_region[row + 1][col] == PLAYER_SPACE:
                    frontier.put((row + 1, col))
                if voronoi_region[row][col - 1] == PLAYER_SPACE:
                    frontier.put((row, col - 1))
                if voronoi_region[row][col + 1] == PLAYER_SPACE:
                    frontier.put((row, col + 1))
                continue
            # Checking if the current location represents an articulaton
            if (
                voronoi_region[row - 1][col] == np.Inf
                and voronoi_region[row + 1][col] == np.Inf
            ):
                # Setting to the appropriate label, adding to the articulation list, and continuing
                voronoi_region[row][col] = ARTICULATION
                articulations.add((row, col))
                continue
            if (
                voronoi_region[row][col - 1] == np.Inf
                and voronoi_region[row][col + 1] == np.Inf
            ):
                # Setting to the appropriate label, adding to the articulation list, and continuing
                voronoi_region[row][col] = ARTICULATION
                articulations.add((row, col))
                continue
            # Reaching this code indicates that we are at a new location within the player's current chamber.
            # We increment the counter, and check each neighbor to see if they should be added to the BFS
            num_in_chamber += 1
            if voronoi_region[row - 1][col] == PLAYER_SPACE:
                frontier.put((row - 1, col))
            if voronoi_region[row + 1][col] == PLAYER_SPACE:
                frontier.put((row + 1, col))
            if voronoi_region[row][col - 1] == PLAYER_SPACE:
                frontier.put((row, col - 1))
            if voronoi_region[row][col + 1] == PLAYER_SPACE:
                frontier.put((row, col + 1))
        # Returning the designated items
        return visited, num_in_chamber, articulations, voronoi_region

    def compute_tc_statistic(
        self, voronoi_region, articulations, visited, num_in_chamber
    ):
        """
        Computes the tree of chambers statistic, using a map that has been prelabeled with articulation points
        Inputs:
            - voronoi_region: Prelabeled numpy array of articulation points and free spaces
            - articulations: A set of articulation points
            - visited: A set of visited points
            - num_in_chamber: The number of points in the player's component region; not blocked off by any articulation points
        Returns:
            - TC Statistic: A number representing the tree of chambers statistic
        """
        max_tc_statistic = num_in_chamber
        # Iterating through every location, conducting a BFS, and comparing against max_tc_statistic to see if this is a larger value
        for cut_point in articulations:
            # Removing this from the visited list, since it will already have been added
            visited.remove(cut_point)
            row = cut_point[0]
            col = cut_point[1]
            curr_tc_statistic = num_in_chamber
            # Set representing the frontier
            frontier = Queue()
            # Adding the current point to the frontier
            frontier.put((row, col))
            # Iterating through every element in the frontier
            while not frontier.empty():
                curr_location = frontier.get()
                row = curr_location[0]
                col = curr_location[1]
                # Ignoring if this state has already been visited
                if (row, col) in visited:
                    continue
                # Adding the current location tuple to the visited set
                visited.add((row, col))
                # Reaching this indicates that this a new free space, we increment the curr_tc_statistic accordingly
                curr_tc_statistic += 1
                # Checking each neighbor. If it is a free space, we add it. If it is a opponent space, we are in a battlefront region and set the flag accordingly
                if voronoi_region[row - 1][col] == PLAYER_SPACE:
                    frontier.put((row - 1, col))
                if voronoi_region[row + 1][col] == PLAYER_SPACE:
                    frontier.put((row + 1, col))
                if voronoi_region[row][col - 1] == PLAYER_SPACE:
                    frontier.put((row, col - 1))
                if voronoi_region[row][col + 1] == PLAYER_SPACE:
                    frontier.put((row, col + 1))
            # We check if the TC statistic should be expanded. If so, it is expanded
            if curr_tc_statistic > max_tc_statistic:
                max_tc_statistic = curr_tc_statistic
        # Returning the maximum value for the tree of chambers heuristic, and the boolean determining whether the graph is connected
        return max_tc_statistic

    def compute_voronoi(self, state, ptm, asp):
        """
        Given a state representation, this function computes the Voronoi heuristic function.
        This function always receives a state for which the opponent is about to move, and so this must
        be reset prior to calling compute distances
        Inputs:
            - state: A Tron state
            - ptm: A player to move, should always be the PTM of the user when called within this class
        Output:
            - voronoi_score: The voronoi score for this state
        """
        # Converting such that the current player moving is always ptm, and computing the first Voronoi distance
        state.ptm = ptm
        student_dist = self.compute_distances(state, ptm, asp)
        # Converting such that the opponent is always ptm, and computing the second Voronoi distance
        if ptm == 0:
            opp_ptm = 1
        else:
            opp_ptm = 0
        state.ptm = opp_ptm
        opp_dist = self.compute_distances(state, opp_ptm, asp)
        # Converting all zero's to infinity to prevent false comparisons
        student_dist[student_dist == 0] = np.Inf
        opp_dist[opp_dist == 0] = np.Inf
        # Computing the Voronoi regions
        student_closer = np.where((student_dist) < opp_dist, PLAYER_SPACE, student_dist)
        student_closer = np.where(
            (student_dist >= opp_dist) & (student_dist < np.Inf),
            OPPONENT_SPACE,
            student_closer,
        )
        opp_closer = np.where(opp_dist < student_dist, PLAYER_SPACE, opp_dist)
        opp_closer = np.where(
            (opp_dist >= student_dist) & (opp_dist < np.Inf), OPPONENT_SPACE, opp_closer
        )
        # Getting the locations of the two players
        student_position = state.player_locs[ptm]
        opponent_position = state.player_locs[opp_ptm]
        # Computing the articulation points, and retrieving the visited set, the num_in_chamber, the articulation set, and the voronoi region for each player
        (
            student_visited,
            student_num_in_chamber,
            student_articulations,
            student_voronoi_region,
        ) = self.compute_articulations(student_closer, student_position)
        (
            opp_visited,
            opp_num_in_chamber,
            opp_articulations,
            opp_voronoi_region,
        ) = self.compute_articulations(opp_closer, opponent_position)

        # Computing the Tree of Chambers heuristic value for each player
        student_tc = self.compute_tc_statistic(
            student_voronoi_region,
            student_articulations,
            student_visited,
            student_num_in_chamber,
        )
        opp_tc = self.compute_tc_statistic(
            opp_voronoi_region,
            opp_articulations,
            opp_visited,
            opp_num_in_chamber,
        )
        return student_tc - opp_tc

    def max_value_ab_cutoff(self, asp, state, ptm, alpha, beta, cutoff_ply):
        """
        Returns the maximum value of possible child states that can be transitioned
        to from the current passed in state parameter, while pruning children using alpha/beta pruning
        and stopping using the cutoff_ply
        Input: asp - an AdversarialSearchProblem
            state - A GameState representing where we are moving from
            ptm - A number representing the index of the player currently making a move (this is always the user bot)
            alpha - A value for the best possible result that a max node above this current max node
            can achieve. Will be updated depending on whether this current node has a higher value
            beta - A value for the best possible result that a min node above this current max node
            can achieve. If the current max node produces a result that is "worse" for this min_node,
            we can immediately prune neighbors
            cutoff_ply - Value indicating how many additional layers deep we should explore
        Output: A value, representing the maximum value of all possible child states
        """
        # Base case test, checking if we are at a terminal state
        if asp.is_terminal_state(state):
            result = asp.evaluate_state(state)
            # Returning WINNING_SCORE if it is a winning state
            if result[ptm] == 1:
                return WINNING_SCORE
            # Otherwise we return the LOSING_SCORE
            else:
                return LOSING_SCORE
        # Checking if we are at a cutoff level, if so we must compute the voronoi heuristic and return
        if cutoff_ply == 0:
            return self.compute_voronoi(state, ptm, asp)
        # Decrementing the ply
        cutoff_ply = cutoff_ply - 1
        # Setting the minimum child value to be infinity, for comparison purposes
        maximum_child_value = float("-inf")
        # Iterating through all possible child actions
        for action in asp.get_available_actions(state):
            transitioned_state = asp.transition(state, action)
            # Getting the min value of this node, since it is a min node
            transitioned_state_value = self.min_value_ab_cutoff(
                asp, transitioned_state, ptm, alpha, beta, cutoff_ply
            )
            # Checking if this value is the maximum value. If so, we set the maximum_child_value
            # accordingly
            if transitioned_state_value > maximum_child_value:
                maximum_child_value = transitioned_state_value
                # If this is the "best" child we have seen so far, also checking if we can prune siblings/this
                # branch is irrelevant. If not, updating the value of alpha and reiterating through the loop
                if maximum_child_value >= beta:
                    return maximum_child_value
                # If this branch is not rendered irrelevant, updating alpha to be the max of the current node
                # value and itself
                alpha = max(alpha, maximum_child_value)
        # Returning the value of the minimum node
        return maximum_child_value

    def min_value_ab_cutoff(self, asp, state, ptm, alpha, beta, cutoff_ply):
        """
        Returns the minimum value of possible child states that can be transitioned
        to from the current passed in state parameter. Cutoff ply is used to limit the depth
        to which the algorithm searches, at which point the eval_function is applied to return
        an estimate of the value at that node
        Input: asp - an AdversarialSearchProblem
            state - A GameState representing where we are moving from
            ptm - A number representing the index of the player currently making a move
            alpha - The best possible value a max node above the current node can achieve, regardless of
            the current node outcome. If this node is lower than the best outcome for the max node, siblings are pruned
            beta - The best possible outcome for a min node above the current node can achieve. Will update depending
            on the current node's value
            cutoff_ply - Value indicating how many additional layers deep we should explore
        Output: A value, representing the minimum value of all possible child states
        """
        # Base case test, checking if we are at a terminal state
        if asp.is_terminal_state(state):
            result = asp.evaluate_state(state)
            # Returning WINNING_SCORE if it is a winning state
            if result[ptm] == 1:
                return WINNING_SCORE
            # Otherwise we return the LOSING_SCORE
            else:
                return LOSING_SCORE
        # Checking if we are at a cutoff level, if so we must compute the voronoi heuristic and return
        if cutoff_ply == 0:
            return self.compute_voronoi(state, ptm, asp)
        # Decrementing the ply
        cutoff_ply = cutoff_ply - 1
        # Setting the minimum child value to be infinity, for comparison purposes
        minimum_child_value = float("inf")
        # Iterating through all possible child actions
        for action in asp.get_available_actions(state):
            transitioned_state = asp.transition(state, action)
            # Getting the max value of this node, since it is a max node
            transitioned_state_value = self.max_value_ab_cutoff(
                asp, transitioned_state, ptm, alpha, beta, cutoff_ply
            )
            # Checking if this value is the minimum value. If so, we set the minimum_child_value
            # accordingly
            if transitioned_state_value < minimum_child_value:
                minimum_child_value = transitioned_state_value
                # If this value is a minimum value, we check if it is so low that the branch is to be eliminated.
                # If not, we update beta and continue through the loop
                if minimum_child_value <= alpha:
                    return minimum_child_value
                # If this branch is not rendered irrelevant, we set beta to be the minimum of this new minimum
                # child value and itself, and then continue
                beta = min(beta, minimum_child_value)
        # Returning the value of the minimum node
        return minimum_child_value

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}

        To get started, you can get the current
        state by calling asp.get_start_state()
        Utilizes the AB cutoff algorithm to search the state space and retrieve the proper action
        """
        # Starting position of the AI Bot
        state = asp.get_start_state()
        # Retrieving the board and our player index
        ptm = state.ptm
        # Retrieving the current location of the player
        start_location = state.player_locs[ptm]
        # Getting the list of move actions to take
        possibilities = list(TronProblem.get_safe_actions(state.board, start_location))
        # Parameters to be used in the AB cutoff algorithm
        max_value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        max_action = None
        # Getting the number of empty spaces, used to determine the ply
        ply_board = np.array(state.board)
        ply_board = np.where(ply_board == " ", 1, 0)
        num_spaces = np.sum(ply_board)
        # Cutoff ply increases as the board fills up
        if num_spaces < 20:
            cutoff_ply = 5
        elif num_spaces < 60:
            cutoff_ply = 4
        elif num_spaces < 100:
            cutoff_ply = 3
        elif num_spaces < 140:
            cutoff_ply = 2
        else:
            cutoff_ply = 1
        # Returning the action that has the maximum value, according to the AB cutoff algorithm
        # We don't handle terminal states in the first step, as we are only looking at safe actions
        for action in possibilities:
            new_state = asp.transition(state, action)
            # Calling minimum value to get the minimum value of these child states
            transitioned_state_value = self.min_value_ab_cutoff(
                asp, new_state, ptm, alpha, beta, cutoff_ply
            )
            # Checking to see if this is the best action so far
            if transitioned_state_value > max_value:
                max_value = transitioned_state_value
                max_action = action
            # Setting the alpha parameter for pruning in child nodes
            alpha = max(alpha, max_value)
        # Returning the best action
        return max_action

    def cleanup(self):
        """
        Input: None
        Output: None

        This function will be called in between
        games during grading. You can use it
        to reset any variables your bot uses during the game
        (for example, you could use this function to reset a
        turns_elapsed counter to zero). If you don't need it,
        feel free to leave it as "pass"
        """
        self.game_count += 1
        print(self.game_count)


class RandBot:
    """Moves in a random (safe) direction"""

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if possibilities:
            return random.choice(possibilities)
        return "U"

    def cleanup(self):
        pass


class WallBot:
    """Hugs the wall"""

    def __init__(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def cleanup(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if not possibilities:
            return "U"
        decision = possibilities[0]
        for move in self.order:
            if move not in possibilities:
                continue
            next_loc = TronProblem.move(loc, move)
            if len(TronProblem.get_safe_actions(board, next_loc)) < 3:
                decision = move
                break
        return decision


class StudentVoronoiBot:
    """ Write your student bot here"""

    def compute_distances(self, state, ptm, asp):
        """
        Computes the distances for each location on the board, for the given player
        Inputs:
            - state: A starting state for the voronoi computation
            - ptm: Index for the player to move
        Returns:
            - Distances: An array where every reachable point has a distance, all others are set to 0
        """
        board = np.array(state.board)
        distances = np.zeros((board.shape[0], board.shape[1]))
        frontier = Queue()
        # Adding the state to the frontier
        frontier.put(state)
        # Executing a breadth-first search to populate the board with distances
        while not frontier.empty():
            curr_state = frontier.get()
            board = np.array(curr_state.board)
            loc = curr_state.player_locs[ptm]
            # Getting a list of possible move actions to take
            possibilities = list(TronProblem.get_safe_actions(board, loc))
            # Adding items to the frontier, if they have not yet been visited
            for action in possibilities:
                possible_state = asp.transition(curr_state, action)
                # Setting to the current player for ptm, to ensure that the same player keeps moving
                possible_state.ptm = ptm
                # Getting the new location
                new_loc = possible_state.player_locs[ptm]
                # Ignoring if this state has already been visited
                if distances[new_loc[0], new_loc[1]] != 0:
                    continue
                distances[new_loc[0], new_loc[1]] = distances[loc[0], loc[1]] + 1
                # Adding to the frontier
                frontier.put(possible_state)
        return distances

    def compute_voronoi(self, state, ptm, asp):
        """
        Given a state representation, this function computes the Voronoi heuristic function.
        This function always receives a state for which the opponent is about to move, and so this must
        be reset prior to calling compute distances
        Inputs:
            - state: A Tron state
            - ptm: A player to move, should always be the PTM of the user when called within this class
        Output:
            - voronoi_score: The voronoi score for this state
        """
        # Converting such that the current player moving is always ptm, and computing the first Voronoi distance
        state.ptm = ptm
        student_dist = self.compute_distances(state, ptm, asp)
        # Converting such that the opponent is always ptm, and computing the second Voronoi distance
        if ptm == 0:
            opp_ptm = 1
        else:
            opp_ptm = 0
        state.ptm = opp_ptm
        opp_dist = self.compute_distances(state, opp_ptm, asp)
        # Converting all zero's to infinity to prevent false comparisons
        student_dist[student_dist == 0] = np.Inf
        opp_dist[opp_dist == 0] = np.Inf
        # Computing respective Voronoi scores for the two players
        student_closer = student_dist < opp_dist
        opp_closer = opp_dist < student_dist
        student_voronoi = np.sum(student_closer)
        opp_voronoi = np.sum(opp_closer)
        # Difference between the student and the opponent score
        return student_voronoi - opp_voronoi

    def max_value_ab_cutoff(self, asp, state, ptm, alpha, beta, cutoff_ply):
        """
        Returns the maximum value of possible child states that can be transitioned
        to from the current passed in state parameter, while pruning children using alpha/beta pruning
        and stopping using the cutoff_ply
        Input: asp - an AdversarialSearchProblem
            state - A GameState representing where we are moving from
            ptm - A number representing the index of the player currently making a move (this is always the user bot)
            alpha - A value for the best possible result that a max node above this current max node
            can achieve. Will be updated depending on whether this current node has a higher value
            beta - A value for the best possible result that a min node above this current max node
            can achieve. If the current max node produces a result that is "worse" for this min_node,
            we can immediately prune neighbors
            cutoff_ply - Value indicating how many additional layers deep we should explore
        Output: A value, representing the maximum value of all possible child states
        """
        # Base case test, checking if we are at a terminal state
        if asp.is_terminal_state(state):
            result = asp.evaluate_state(state)
            # Returning WINNING_SCORE if it is a winning state
            if result[ptm] == 1:
                return WINNING_SCORE
            # Otherwise we return the LOSING_SCORE
            else:
                return LOSING_SCORE
        # Checking if we are at a cutoff level, if so we must compute the voronoi heuristic and return
        if cutoff_ply == 0:
            return self.compute_voronoi(state, ptm, asp)
        # Decrementing the ply
        cutoff_ply = cutoff_ply - 1
        # Setting the minimum child value to be infinity, for comparison purposes
        maximum_child_value = float("-inf")
        # Iterating through all possible child actions
        for action in asp.get_available_actions(state):
            transitioned_state = asp.transition(state, action)
            # Getting the min value of this node, since it is a min node
            transitioned_state_value = self.min_value_ab_cutoff(
                asp, transitioned_state, ptm, alpha, beta, cutoff_ply
            )
            # Checking if this value is the maximum value. If so, we set the maximum_child_value
            # accordingly
            if transitioned_state_value > maximum_child_value:
                maximum_child_value = transitioned_state_value
                # If this is the "best" child we have seen so far, also checking if we can prune siblings/this
                # branch is irrelevant. If not, updating the value of alpha and reiterating through the loop
                if maximum_child_value >= beta:
                    return maximum_child_value
                # If this branch is not rendered irrelevant, updating alpha to be the max of the current node
                # value and itself
                alpha = max(beta, maximum_child_value)
        # Returning the value of the minimum node
        return maximum_child_value

    def min_value_ab_cutoff(self, asp, state, ptm, alpha, beta, cutoff_ply):
        """
        Returns the minimum value of possible child states that can be transitioned
        to from the current passed in state parameter. Cutoff ply is used to limit the depth
        to which the algorithm searches, at which point the eval_function is applied to return
        an estimate of the value at that node
        Input: asp - an AdversarialSearchProblem
            state - A GameState representing where we are moving from
            ptm - A number representing the index of the player currently making a move
            alpha - The best possible value a max node above the current node can achieve, regardless of
            the current node outcome. If this node is lower than the best outcome for the max node, siblings are pruned
            beta - The best possible outcome for a min node above the current node can achieve. Will update depending
            on the current node's value
            cutoff_ply - Value indicating how many additional layers deep we should explore
        Output: A value, representing the minimum value of all possible child states
        """
        # Base case test, checking if we are at a terminal state
        if asp.is_terminal_state(state):
            result = asp.evaluate_state(state)
            # Returning WINNING_SCORE if it is a winning state
            if result[ptm] == 1:
                return WINNING_SCORE
            # Otherwise we return the LOSING_SCORE
            else:
                return LOSING_SCORE
        # Checking if we are at a cutoff level, if so we must compute the voronoi heuristic and return
        if cutoff_ply == 0:
            return self.compute_voronoi(state, ptm, asp)
        # Decrementing the ply
        cutoff_ply = cutoff_ply - 1
        # Setting the minimum child value to be infinity, for comparison purposes
        minimum_child_value = float("inf")
        # Iterating through all possible child actions
        for action in asp.get_available_actions(state):
            transitioned_state = asp.transition(state, action)
            # Getting the max value of this node, since it is a max node
            transitioned_state_value = self.max_value_ab_cutoff(
                asp, transitioned_state, ptm, alpha, beta, cutoff_ply
            )
            # Checking if this value is the minimum value. If so, we set the minimum_child_value
            # accordingly
            if transitioned_state_value < minimum_child_value:
                minimum_child_value = transitioned_state_value
                # If this value is a minimum value, we check if it is so low that the branch is to be eliminated.
                # If not, we update beta and continue through the loop
                if minimum_child_value <= alpha:
                    return minimum_child_value
                # If this branch is not rendered irrelevant, we set beta to be the minimum of this new minimum
                # child value and itself, and then continue
                beta = min(beta, minimum_child_value)
        # Returning the value of the minimum node
        return minimum_child_value

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}

        To get started, you can get the current
        state by calling asp.get_start_state()
        Utilizes the AB cutoff algorithm to search the state space and retrieve the proper action
        """
        # Starting position of the AI Bot
        state = asp.get_start_state()
        # Retrieving the board and our player index
        ptm = state.ptm
        # Retrieving the current location of the player
        start_location = state.player_locs[ptm]
        # Getting the list of move actions to take
        possibilities = list(TronProblem.get_safe_actions(state.board, start_location))
        # Parameters to be used in the AB cutoff algorithm
        max_value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        max_action = None
        cutoff_ply = 2
        # Returning the action that has the maximum value, according to the AB cutoff algorithm
        # We don't handle terminal states in the first step, as we are only looking at safe actions
        for action in possibilities:
            new_state = asp.transition(state, action)
            # Calling minimum value to get the minimum value of these child states
            transitioned_state_value = self.min_value_ab_cutoff(
                asp, new_state, ptm, alpha, beta, cutoff_ply
            )
            # Checking to see if this is the best action so far
            if transitioned_state_value > max_value:
                max_value = transitioned_state_value
                max_action = action
            # Setting the alpha parameter for pruning in child nodes
            alpha = max(alpha, max_value)
        # Returning the best action
        return max_action

    def cleanup(self):
        """
        Input: None
        Output: None

        This function will be called in between
        games during grading. You can use it
        to reset any variables your bot uses during the game
        (for example, you could use this function to reset a
        turns_elapsed counter to zero). If you don't need it,
        feel free to leave it as "pass"
        """
        pass