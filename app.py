"""
To do:

Option to exclude owned game.
waiting message
check mobile
How to handle missing playing time?  currently it's 0, which means it's removed when asking for minimum time.

"""

from flask import Flask, render_template, request

from game_recommender import GameRecommender

app = Flask(__name__)

# recommender = GameRecommender('/Users/yaron/workspace/games/bgg_ratings/recommenders/app/data')
recommender = GameRecommender('data')

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method != 'POST':
        return render_template('index.html')

    username = request.form['username']
    from_top_ranked = int(request.form['from_top_ranked'])
    min_players = int(request.form['min_players'])
    max_players = int(request.form['max_players'])
    min_length = int(request.form['min_length'])
    max_length = int(request.form['max_length'])
    if max_length == 300:
        max_length = 30000
    
    try:
        recommender.load_user_ratings(username)
    except Exception as e:
        error_message = str(e)
        return render_template('index.html', error=error_message)
    
    good_features, bad_features = recommender.calc_important_features()

    recommendations = recommender.recommend_games(from_top_ranked, min_players, max_players, min_length, max_length)

    return render_template('results.html', 
                           n_high_ratings=len(recommender.good_weights),
                           threshold=recommender.threshold,
                           from_top_ranked=from_top_ranked,
                           min_players=(min_players if min_players > 1 else None),
                           max_players=(max_players if max_players < 12 else None),
                           min_length=(min_length if min_length > 0 else None),
                           max_length=(max_length if max_length < 300 else None),
                           good_features=good_features,
                           bad_features=bad_features,
                           recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)