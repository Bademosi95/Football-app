#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd


# In[54]:


df = pd.read_html('https://fbref.com/en/comps/Big5/Big-5-European-Leagues-Stats#all_league_summary', 
                    attrs={"id":"big5_table"})[0]


# In[55]:


df.head()


# In[56]:


df = df.dropna(subset=['Rk'])


# In[57]:


df.head()


# In[58]:


import numpy as np
from scipy.stats import poisson


# In[59]:


league_data = df = df.dropna(subset=['Rk'])


# In[106]:


# Precompute per-match averages for league-wide attack and defense
league_averages = (
    league_data.groupby('Country', as_index=False)
    .agg(
        # Calculate averages by dividing sum of each stat by sum of matches played
        avg_gf=('GF', lambda x: x.sum() / league_data.loc[x.index, 'MP'].sum()),
        avg_ga=('GA', lambda x: x.sum() / league_data.loc[x.index, 'MP'].sum()),
        avg_xg=('xG', lambda x: x.sum() / league_data.loc[x.index, 'MP'].sum()),
        avg_xga=('xGA', lambda x: x.sum() / league_data.loc[x.index, 'MP'].sum()),
        avg_xgd=('xGD', lambda x: x.sum() / league_data.loc[x.index, 'MP'].sum()),
        avg_xgdpg=('xGD/90', lambda x: x.sum() / league_data.loc[x.index, 'MP'].sum())
    )
)


# In[160]:


Squad = league_data.merge(league_averages, on='Country')
Squad['attack_strength'] = Squad['GF'] / Squad['MP'] / Squad['avg_gf']
Squad['defense_strength'] = Squad['GA'] / Squad['MP'] / Squad['avg_ga']
Squad['expected_attack_strength'] = Squad['xG'] / Squad['MP'] / Squad['avg_xg']
Squad['expected_defence_strength'] = Squad['xGA'] / Squad['MP'] / Squad['avg_xga']
Squad['expected_gd'] = Squad['xGD'] / Squad['MP'] / Squad['avg_xgd']
Squad['expected_gdpg'] = Squad['xGD/90'] / Squad['MP'] / Squad['avg_xgdpg']


# In[156]:


# --- Prediction Model ---
def expected_goals(home_team: str, away_team: str, Country: str):
    home = teams[(teams['Squad'] == home_team) & (teams['Country'] == Country)].iloc[0]
    away = teams[(teams['Squad'] == away_team) & (teams['Country'] == Country)].iloc[0]

# Compute expected goals using Poisson model
    Country_avg = league_averages[league_averages['Country'] == Country].iloc[0]
    home_gf = home['attack_strength'] * away['defense_strength'] * Country_avg['avg_gf']
    away_gf = away['attack_strength'] * home['defense_strength'] * Country_avg['avg_gf']
    return home_gf, away_gf
    


# In[157]:


def expected_gconceded (home_team: str, away_team: str, Country: str):
    home = teams[(teams['Squad'] == home_team) & (teams['Country'] == Country)].iloc[0]
    away = teams[(teams['Squad'] == away_team) & (teams['Country'] == Country)].iloc[0]

# Compute expected goals using Poisson model
    Country_avg = league_averages[league_averages['Country'] == Country].iloc[0]
    home_ga = home['defense_strength'] * away['attack_strength'] * Country_avg['avg_ga']
    away_ga = away['defense_strength'] * home['attack_strength'] * Country_avg['avg_ga']
    return home_ga, away_ga


# In[150]:


def expected_gdpg (home_team: str, away_team: str, Country: str):
    home = teams[(teams['Squad'] == home_team) & (teams['Country'] == Country)].iloc[0]
    away = teams[(teams['Squad'] == away_team) & (teams['Country'] == Country)].iloc[0]

# Compute expected goals using Poisson model
    Country_avg = league_averages[league_averages['Country'] == Country].iloc[0]
    home_xgdpg = home['expected_gdpg'] * away['expected_gdpg'] * Country_avg['avg_xgdpg']
    away_xgdpg = away['expected_gdpg'] * away['expected_gdpg'] * Country_avg['avg_xgdpg']
    return home_xgpg, away_gdpg


# In[170]:


def score_prob_matrix(home_gf: float, away_gf: float, max_goals: int = 10):
    # Generate matrix of score probabilities up to max_goals
    prob_matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            # Fixed syntax error by using proper line continuation with parentheses
            prob_matrix[i, j] = poisson.pmf(i, home_gf) * poisson.pmf(j, away_gf)
    return prob_matrix


# In[172]:


def match_outcomes(home_team: str, away_team: str, Country: str):
    hg, ag = expected_goals(home_team, away_team, Country)
    hac, aac = expected_gconceded(home_team, away_team, Country)
    hxgpg, axgpg = expected_goals(home_team, away_team, Country)
    P = score_prob_matrix(hg, ag)

    # Probabilities
    p_home_win = np.tril(P, -1).sum()
    p_draw = np.trace(P)
    p_away_win = np.triu(P, 1).sum()
    p_both_score = 1 - (P.sum(axis=1)[0] + P.sum(axis=0)[0]) + P[0,0]
    exp_goals = (P * np.arange(P.shape[0])[:,None]).sum() + (P * np.arange(P.shape[1])[None,:]).sum()
    return {
        'home_goals': hg,
        'away_goals': ag,
        'home_goals_against':hac,
        'away_goals_against':aac,
        'home_xgpg':hxgpg,
        'away_xgpg':axgpg,
        'p_home_win': p_home_win,
        'p_draw': p_draw,
        'p_away_win': p_away_win,
        'p_both_score': p_both_score,
        'expected_total_goals': exp_goals
    }


# In[173]:


# --- Interactive Selection ---
def run_cli():
    print("=== Dynamic Bet Analysis ===")
    league = input("Enter league (e.g. Premier League): ")
    home = input("Home team: ")
    away = input("Away team: ")
    stats = match_outcomes(home, away, league)
    print(f"Expected goals: {home} {stats['home_goals']:.2f} - {stats['away_goals']:.2f} {away}")
    print(f"Expected goals against: {home} {stats['home_goals_against']:.2f} - {stats['away_goals_against']:.2f} {away}")
    print(f"Expected xgpd: {home} {stats['home_xgpg']:.2f} - {stats['away_xgpg']:.2f} {away}")
    print(f"Probability home win: {stats['p_home_win']:.2%}")
    print(f"Probability draw: {stats['p_draw']:.2%}")
    print(f"Probability away win: {stats['p_away_win']:.2%}")
    print(f"Probability both teams score: {stats['p_both_score']:.2%}")
    print(f"Expected total goals: {stats['expected_total_goals']:.2f}")

if __name__ == '__main__':
    run_cli()


# In[ ]:




