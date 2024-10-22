from collections import defaultdict
import json
import numpy as np
from operator import itemgetter
import random
import requests
from sklearn.ensemble import GradientBoostingClassifier
import time
import xml.etree.ElementTree as ET

BAD_GAMES_FACTOR = 1
MIN_GAMES_SHOWN = 50
MIN_CONFIDENCE = 0.6


numerical_feature_names = [
    'minplayers',
    'maxplayers',
    'minplaytime',
    'maxplaytime',
    'minage',
    'usersrated',
    'average',
    'stddev',
    'averageweight',
    'min_suggested_n_players',
    'max_suggested_n_players',
]
numerical_feature_set = set(numerical_feature_names)


class Game:
    def __init__(self, game_dict, game_id):
        self.id = game_id
        self.name = game_dict['name']
        self.best_n_players_list = game_dict['best_n_players_list']
        # When the BGG play time is a number (rather than a range), the max play time is 0 in the data - fix this.
        self.min_play_time = min(game_dict['minplaytime'], game_dict['maxplaytime'])
        self.max_play_time = max(game_dict['minplaytime'], game_dict['maxplaytime'])
        self.users_rated = game_dict['usersrated']

        self.numerical_features = {}
        for f in numerical_feature_names:
            if f in game_dict:
                self.numerical_features[f] = game_dict[f]
        # These two are not directly in the data
        self.numerical_features['min_suggested_n_players'] = min(self.best_n_players_list)
        self.numerical_features['max_suggested_n_players'] = max(self.best_n_players_list)
        # These two are in the data, but need the fix described above
        self.numerical_features['minplaytime'] = self.min_play_time
        self.numerical_features['maxplaytime'] = self.max_play_time

        self.boolean_features = set(
            ['category: ' + category for category in game_dict['categories']] +
            ['mechanic: ' + mechanic for mechanic in game_dict['mechanics']] + 
            ['family: ' + family for family in game_dict['families'] if not family.startswith('Game: ')] +
            [f'Works well with {n_players} players' for n_players in range(2, 6) if n_players in self.best_n_players_list]
        )

        # People don't enjoy getting lots of 18xx in their ratings, but there are so few of those, they aren't naturally
        # removed for people who haven't rated any.
        # Solution: make them all the same game family, so that you get at most one.
        # (this means they will not be recommended for people who did rate them highly, either)
        if 'Series: 18xx' in game_dict['families']:
            game_dict['families'].append('Game: 18xx')

        self.game_families = [family for family in game_dict['families'] if family.startswith('Game: ')]
        self.equivalences = set(game_dict['implementations']) | set(game_dict['integrations'])


class GameRecommender:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        with open(f'{data_dir}/feature_display_names.json') as f:
            self.feature_display_names = json.load(f)
        with open(f'{data_dir}/category_key.json') as f:
            self.category_key = json.load(f)
        self.load_and_process_games()
        self.add_game_families_to_equivalences()
        self.boolean_feature_names = list(set.union(*[game.boolean_features for game in self.games.values()]))
        self.all_feature_names = numerical_feature_names + self.boolean_feature_names
        
    def load_and_process_games(self):
        with open(f"{self.data_dir}/top_bgg_games_data.json") as f:
            games_dict = json.load(f)
        self.games = {game_id: Game(game_dict, game_id) for game_id, game_dict in games_dict.items()}

    def add_game_families_to_equivalences(self):
        game_families = defaultdict(set)
        for game_id, game in self.games.items():
            for family in game.game_families:
                game_families[family].add(game_id)
        for game_id, game in self.games.items():
            for family in game.game_families:
                game.equivalences |= game_families[family]
            game.equivalences.discard(game_id)

    def load_user_ratings(self, username):
        url = f"https://boardgamegeek.com/xmlapi2/collection?username={username}&stats=1&rated=1"
        
        for _ in range(5):
            response = requests.get(url)
            match response.status_code:
                case 200:
                    break
                case 202:
                    time.sleep(10)
                case c:
                    raise Exception(f"Error {c} fetching data from BGG.  Please try again later.")

        if response.status_code != 200:
            raise Exception("BGG is taking too long to respond. Please try again in a few minutes.")

        ratings = self.parse_bgg_ratings(response)
        self.ratings = {key: value for key, value in ratings.items() if key in self.games}

        if len(ratings) == 0:
            raise Exception(f"User {username} has not rated any games in the BGG top 10000.  Please check name for typos.")

        self.good_weights = self.calc_good_weights(self.ratings)
        self.threshold = min(self.ratings[gid] for gid in self.good_weights)
        self.bad_weights = self.calc_bad_weights(self.good_weights)

    def parse_bgg_ratings(self, response):
        # Parse the XML response
        ratings = {}
        root = ET.fromstring(response.content)
        for item in root.findall('item'):
            game_id = item.get('objectid')
            rating = item.find('stats').find('rating').get('value')
            try:
                ratings[game_id] = float(rating)
            except ValueError:
                pass
        return ratings

    def define_possible_games(self, ratings, from_bgg_top_ranked, min_players, max_players, min_length, max_length):
        excluded_game_ids = set()
        for game_id in ratings:
            self.exclude_equivalences(game_id, excluded_game_ids)
        self.possible_game_ids = [
            game_id for game_id in list(self.games.keys())[:from_bgg_top_ranked] 
            if game_id not in excluded_game_ids
            and any(min_players <= n <= max_players for n in self.games[game_id].best_n_players_list)
            and self.games[game_id].min_play_time <= max_length
            and self.games[game_id].max_play_time >= min_length
        ] 

    def calc_good_weights(self, ratings):
        sorted_rated_game_ids, sorted_ratings = zip(*sorted(ratings.items(), key=itemgetter(1), reverse=True))
        like_threshold = min(7.99, sorted_ratings[int(len(ratings) * 0.3)])
        like_threshold = min(like_threshold, sorted_ratings[0] - 0.01)
        excluded_rated_game_ids = set()
        high_ratings = {}
        for game_id in sorted_rated_game_ids:
            if ratings[game_id] <= like_threshold:
                break
            if game_id in excluded_rated_game_ids:
                continue
            self.exclude_equivalences(game_id, excluded_rated_game_ids)
            high_ratings[game_id] = ratings[game_id]
        max_rating = max(high_ratings.values())
        good_weights = {g: 2 ** (r - max_rating) for g, r in high_ratings.items()}
        return good_weights

    def calc_bad_weights(self, good_weights):
        excluded_game_ids_for_training = set()
        for game_id in good_weights:
            self.exclude_equivalences(game_id, excluded_game_ids_for_training)

        # each game without a high rating is "bad", weighted according to how well known it is.
        bad_weights = {gid: self.games[gid].users_rated for gid in self.games if gid not in excluded_game_ids_for_training}
        return bad_weights

    def importance(self, weight_pair):
        w_good, w_bad = weight_pair[1], weight_pair[2]
        diff = w_good - w_bad
        log_ratio = np.log((w_good + 1e-8) / (w_bad + 1e-8)) / 8
        if abs(diff) < abs(log_ratio):
            return diff
        return log_ratio

    def calc_important_features(self):
        numerical_feature_weight_pairs = self.calc_numerical_feature_weight_pairs(self.good_weights, self.bad_weights)
        boolean_feature_weight_pairs = self.calc_boolean_feature_weight_pairs(self.good_weights, self.bad_weights)
        feature_weight_pairs = sorted(numerical_feature_weight_pairs + boolean_feature_weight_pairs, key=self.importance)
        good_features = self.finalize_important_features(reversed(feature_weight_pairs[-20:]))
        bad_features = self.finalize_important_features(feature_weight_pairs[:20])
        return good_features, bad_features

    def calc_boolean_weights(self, game_weights: dict):
        total_weight = sum(game_weights.values())
        boolean_weights = defaultdict(int)
        for game_id, weight in game_weights.items():
            game = self.games[game_id]
            for feature in game.boolean_features:
                boolean_weights[feature] += weight
        boolean_weights = {feature: weight / total_weight for feature, weight in boolean_weights.items()}
        return boolean_weights

    def calc_boolean_feature_weight_pairs(self, good_weights, bad_weights):
        good_boolean_weights = self.calc_boolean_weights(good_weights)
        bad_boolean_weights = self.calc_boolean_weights(bad_weights)
        boolean_feature_weight_pairs = []
        for feature in set(good_boolean_weights) | set(bad_boolean_weights):
            good_mean = good_boolean_weights.get(feature, 0)
            bad_mean = bad_boolean_weights.get(feature, 0)
            boolean_feature_weight_pairs.append((feature, good_mean, bad_mean))
        return boolean_feature_weight_pairs

    def calc_numerical_feature_weight_pairs(self, good_weights: dict, bad_weights: dict):
        total_good_weight = sum(good_weights.values())
        total_bad_weight = sum(bad_weights.values())
        base_feature_weight_pairs = []
        for feature in numerical_feature_names:
            values = sorted(set(self.games[gid].numerical_features[feature] for gid in good_weights))
            best = None
            worst = None
            best_score = worst_score = 0
            for value in values:
                good_with = sum(weight for gid, weight in good_weights.items() if self.games[gid].numerical_features[feature] >= value) / total_good_weight
                bad_with = sum(weight for gid, weight in bad_weights.items() if self.games[gid].numerical_features[feature] >= value) / total_bad_weight
                score = good_with - bad_with
                if score > best_score:
                    best = value, good_with, bad_with
                    best_score = score
                good_without = sum(weight for gid, weight in good_weights.items() if self.games[gid].numerical_features[feature] > value) / total_good_weight
                bad_without = sum(weight for gid, weight in bad_weights.items() if self.games[gid].numerical_features[feature] > value) / total_bad_weight
                score = good_without - bad_without
                if score < worst_score:
                    worst = value, good_without, bad_without
                    worst_score = score
            feature_name = self.feature_display_names[feature]
            if abs(best_score) > abs(worst_score):
                value, good, bad = best
                bad_weight = sum(weight for gid, weight in bad_weights.items() if self.games[gid].numerical_features[feature] >= value) / total_bad_weight
                if bad_weight < 0.5:
                    name = f"{feature_name} of {self.format_number(value, feature_name)} or more"
                    base_feature_weight_pairs.append((name, good, bad))
                else:
                    name = f"{feature_name} below {self.format_number(value, feature_name)}"
                    base_feature_weight_pairs.append((name, 1-good, 1-bad))
            else:
                value, good, bad = worst
                bad_weight = sum(weight for gid, weight in bad_weights.items() if self.games[gid].numerical_features[feature] > value) / total_bad_weight
                if bad_weight < 0.5:
                    name = f"{feature_name} above {self.format_number(value, feature_name)}"
                    base_feature_weight_pairs.append((name, good, bad))
                else:
                    name = f"{feature_name} of {self.format_number(value, feature_name)} or less"
                    base_feature_weight_pairs.append((name, 1-good, 1-bad))
        return base_feature_weight_pairs

    def finalize_important_features(self, feature_weight_pairs):
        player_count_shown = False
        playing_time_shown = False
        finalized_features = []
        for feature, good, bad in feature_weight_pairs:
            display = True
            if feature[3:14] == ". play time":
                if playing_time_shown:
                    display = False
                else:
                    playing_time_shown = True
            if feature[3:19] == ". player count (":
                if player_count_shown:
                    display = False
                else:
                    player_count_shown = True           
            if display and abs(self.importance((feature, good, bad))) > 0.06:
                if feature[:8] in ('category', 'mechanic', 'family: '):
                    category, name = feature.split(': ', 1)
                    category = 'boardgame' + category
                    category_id = self.category_key[category].get(name)
                    if category_id is None:
                        feature = name
                    else:
                        url = f'https://boardgamegeek.com/{category}/{category_id}'
                        feature = f'<a href="{url}" target="_blank">{name}</a>'
                finalized_features.append((feature, round(good*100), round(bad*100))) 
        return finalized_features

    def recommend_games(self, from_top_ranked, min_players, max_players, min_length, max_length):
        self.define_possible_games(self.ratings, from_top_ranked, min_players, max_players, min_length, max_length)
        possible_games_by_feature = {feature: [self.extract_feature(self.games[game_id], feature) for game_id in self.possible_game_ids]
                                    for feature in self.all_feature_names}
        cumulative_probs = self.calc_cumulative_probs(possible_games_by_feature)
        return self.finalize_recommendations(self.possible_game_ids, cumulative_probs)

    def calc_cumulative_probs(self, possible_games_by_feature):
        SAMPLE_SIZE = 12000
        cumulative_probs = []
        bad_weight_keys = list(self.bad_weights.keys())
        bad_weight_values = list(self.bad_weights.values())
        total_good_weight = sum(self.good_weights.values())
        for _ in range(int(SAMPLE_SIZE / total_good_weight) + 1):
            good_game_ids = [g for g, weight in self.good_weights.items() if random.random() < weight]
            bad_game_ids = random.choices(bad_weight_keys, bad_weight_values, k=round(len(good_game_ids) * BAD_GAMES_FACTOR))
            training_game_ids = good_game_ids + bad_game_ids
            training_boolean_features = list(set.union(*[self.games[game_id].boolean_features for game_id in training_game_ids]))
            training_feature_names = numerical_feature_names + training_boolean_features
            training_game_features = [[self.extract_feature(self.games[game_id], feature_name)
                                    for feature_name in training_feature_names]
                                    for game_id in training_game_ids]
            training_classes = [1] * len(good_game_ids) + [0] * len(bad_game_ids)
            grad = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3)
            grad.fit(training_game_features, training_classes)
            possible_game_features = list(zip(*[possible_games_by_feature[feature] for feature in training_feature_names]))
            probs = grad.predict_proba(possible_game_features)
            cumulative_probs.append([p[1] for p in probs])
        cumulative_probs = np.array(cumulative_probs)    
        return cumulative_probs

    def finalize_recommendations(self, possible_game_ids, cumulative_probs):
        mean_probs = np.mean(cumulative_probs, axis=0)
        probs_and_ids = sorted(zip(mean_probs, possible_game_ids), reverse=True)
        recommendations = []
        repeated_game_ids = set()
        for p, game_id in probs_and_ids:
            if p < MIN_CONFIDENCE and len(recommendations) >= MIN_GAMES_SHOWN:
                break
            if game_id not in repeated_game_ids:
                self.exclude_equivalences(game_id, repeated_game_ids)
                name = self.games[game_id].name
                recommendations.append((game_id, name, round(p*100)))
        return recommendations

    def extract_feature(self, game: Game, feature_name):
        if feature_name in numerical_feature_set:
            return game.numerical_features[feature_name]
        return feature_name in game.boolean_features

    def exclude_equivalences(self, game_id, excluded_game_ids: set):
        if game_id in excluded_game_ids:
            return
        if game_id not in self.games:
            return
        excluded_game_ids.add(game_id)
        for equivalence in self.games[game_id].equivalences:
            self.exclude_equivalences(equivalence, excluded_game_ids)

    def format_number(self, num, feature_name):
        return f"{num:.2f}" if int(num) != num else f"{int(num)}{' minutes' if '. play time' in feature_name else ''}"
