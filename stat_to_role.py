import pandas as pd
import joblib

def season_convert(s):
    if s == 12 or s == 17:
        return -1
    elif s == 13:
        return 1
    else:
        return 0
    
def kd_check(kd, counted_kd, deaths):
    if abs(kd - counted_kd) < 1:
        return counted_kd
    elif not deaths:
        return kd
    else: 
        return counted_kd

async def role_by_stats(**kwargs):
    base_stats_miss = [k for k in ["kills", "deaths", "kd", "counted_kd", "avg", "is_ranked"] if k not in kwargs]
    rank_stats_miss = [k for k in ["rank_kills", "rank_deaths", "rank_kd", "rank_counted_kd", "rank_avg", "rank_games", "season"] if k not in kwargs]
    season_stats_miss = [k for k in ["season_kills", "season_deaths", "season_kd", "season_counted_kd", "season_avg"] if k not in kwargs]

    if base_stats_miss or (rank_stats_miss and season_stats_miss):
        return 'error', 'error'
    else:
        is_ranked = kwargs.get('is_ranked')

        ALL_COLUMNS = ["kills", "deaths", "kd", "avg", "is_ranked", 
                       "season", "rank_games", 
                       "rank_kills", "rank_deaths", "rank_kd", "rank_avg",
                       "rank_s1", "rank_s2", 
                       "season_kills", "season_deaths", "season_kd", "season_avg", ]

        df = pd.DataFrame([kwargs], columns=ALL_COLUMNS)

        df['kd'] = kd_check(kwargs.get('kd'), kwargs.get('counted_kd'), kwargs.get('deaths'))
        if is_ranked:
            df['rank_kd'] = kd_check(kwargs.get('rank_kd'), kwargs.get('rank_counted_kd'), kwargs.get('rank_deaths'))
            rank_map = {'unknown': 0,'bronze': 1, 'silver': 2, 'gold': 3, 'platinum': 4, 'diamond': 5, 'master': 6, 'predator': 7}
            df['rank_s1'] = df['rank_s1'].map(rank_map).fillna(0).astype(int)
            df['rank_s2'] = df['rank_s2'].map(rank_map).fillna(0).astype(int)
            df['season'] = df['season'].apply(season_convert)
            rank = str(max(df['rank_s1'][0], df['rank_s2'][0]))
        else:
            df['season_kd'] = kd_check(kwargs.get('season_kd'), kwargs.get('season_counted_kd'), kwargs.get('season_deaths'))
            rank = '0'

        model = joblib.load('skill_pred_model/lgbm_model.pkl')
        skill = str(model.predict(df)[0])

        return skill, rank