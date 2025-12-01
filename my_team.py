import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='BalancedOffensiveAgent', second='BalancedDefensiveAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # fallback to nearest home if very few food left
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 1.0}


class BalancedOffensiveAgent(ReflexCaptureAgent):
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        # distance to nearest food
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # capsules
        capsules = self.get_capsules(successor)
        if capsules:
            min_capsule_dist = min([self.get_maze_distance(my_pos, cap) for cap in capsules])
            features['distance_to_capsule'] = min_capsule_dist

        # avoid ghosts
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        if ghosts:
            dists = [self.get_maze_distance(my_pos, g.get_position()) for g in ghosts]
            features['ghost_distance'] = min(dists)

        # carry food: if carrying 2 or more, try to go home
        features['return_home'] = 0
        if my_state.num_carrying >= 2 or (ghosts and min(dists) <= 2):
            home_pos = self.start
            features['return_home'] = self.get_maze_distance(my_pos, home_pos)

        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1, 'distance_to_capsule': -1,
                'ghost_distance': 2, 'return_home': -10}


class BalancedDefensiveAgent(ReflexCaptureAgent):
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # on defense
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # chase invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        # patrol: if no invaders, go near center of your half
        if len(invaders) == 0:
            center = (game_state.data.layout.width // 2, game_state.data.layout.height // 2)
            features['patrol_distance'] = self.get_maze_distance(my_pos, center)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'invader_distance': -10, 'on_defense': 100,
                'stop': -100, 'reverse': -2, 'patrol_distance': -1}
