import pandas as pd
import statsmodels.formula.api as sm
import statsmodels.api as sma
from pathlib import Path

def forward_selected(data, response, remaining, prev=[]):
    """
        based upon algorithm found at: https://planspace.org/20150423-forward_selection_with_statsmodels/
    """
    selected = []
    prv = []
    current_score, best_new_score = 0.05, 0.05
    best_formula = ''
    starting_formula = "{response} ~ {prev}{selected}"

    for i in range(0,len(prev)):
        prv.append("*".join(prev[i]))
    if len(prv) > 0:
        previous = "*".join(prv)
    else:
        previous = '1'

    while remaining and current_score == best_new_score:
        current_score = 0.05
        scores_with_candidates = []
        sel = starting_formula.format(response=response, selected='*'.join(selected), prev=previous)
        s_file = "models/"+sel.replace(response+" ~ ","")+'.pickle'
        s_file = s_file.replace('*', '')
        if Path(s_file).exists():
            sel_model = sma.load(s_file)
        else:
            if sel == best_formula:
                sel_model = best_model
            else:
                sel_model = sm.ols(sel, data).fit()
                sel_model.save(s_file)
        print("testing base: {}".format(sel))
        if previous == "1" or previous == "":
            previous = ""
        else:
            if previous[:1] != "*":
                previous = previous+"*"
        for candidate in remaining:
            formula = starting_formula.format(response=response, selected='*'.join(selected + [candidate]), prev=previous)
            f_file = "models/"+formula.replace(response+" ~ ","")+'.pickle'
            f_file = f_file.replace('*','')
            if Path(f_file).exists():
                model = sma.load(f_file)
            else:
                model = sm.ols(formula, data).fit()
                model.save(f_file)
            print("testing addition: {}".format(formula))
            prf = sma.stats.anova_lm(sel_model,model)['Pr(>F)'].loc[1]
            print("testing addition: {} result: {}".format(formula, prf))
            scores_with_candidates.append((prf, candidate, model, formula))
        scores_with_candidates.sort()
        best_new_score, best_candidate, best_model, best_formula = scores_with_candidates.pop(0)
        if best_new_score < current_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            best_model.save('best_model.pickle')
    if previous[:1] != "*" and len(selected) == 0:
        previous = previous[:-1]
    formula = starting_formula.format(response=response, selected='*'.join(selected), prev=previous)
    model = sm.ols(formula, data).fit()
    return model, formula, selected


d = pd.read_csv("DA_Clean NCSA Reserves_4.14.18-FINAL.csv")
d['ReleaseMonth'] = d['ReleaseMonth'].astype('str')
d['ReleaseYear'] = d['ReleaseYear'].astype('str')

result, f, selected = forward_selected(d,'ReservesLevel',['Platform', 'Region', 'ReleaseMonth', 'Channel', 'Edition', 'RelativeWeek', 'GameType'])
# result, f, selected = forward_selected(d,'ReservesLevel',['RelativeWeek','GameType','ReleaseYear'], [['Platform','Region','ReleaseMonth','Channel','Edition']])

print(f)
print(selected)
print(result.summary())