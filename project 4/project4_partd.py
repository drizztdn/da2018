import pandas as pd
import statsmodels.formula.api as sm
import statsmodels.api as sma

def backward_selected(data, response, remaining, prev=[]):
    """
        based upon algorithm found at: https://planspace.org/20150423-forward_selection_with_statsmodels/
    """
    remain = remaining[:]
    selected = []
    prv = []
    current_score, best_new_score = 0.05, 0.05
    starting_formula = "{response} ~ {prev}{selected}"

    for i in range(0,len(prev)):
        prv.append("*".join(prev[i]))
    if len(prv) > 0:
        previous = "+".join(prv)
        if len(previous) > 1:
            previous = previous + '+'
    else:
        previous = '1'

    while remain and current_score == best_new_score:
        current_score = 0.05
        scores_with_candidates = []
        sel = starting_formula.format(response=response, selected='*'.join(remain), prev=previous)
        sel_model = sm.ols(sel, data).fit()
        print("testing base: {}".format(sel))
        for candidate in remain:
            s = remain[:]
            s.remove(candidate)
            if len(s) == 0 and previous.endswith('+'):
                previous = previous[:-1]
            formula = starting_formula.format(response=response, selected='*'.join(s), prev=previous)
            model = sm.ols(formula, data).fit()
            print("testing removal: {}".format(formula))
            prf = sma.stats.anova_lm(model,sel_model)['Pr(>F)'].loc[1]
            scores_with_candidates.append((prf, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remain.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    if previous[:1] != "+" and len(selected) == 0:
        previous = previous[:-1]
    for s in selected:
        remaining.remove(s)
    formula = starting_formula.format(response=response, selected='*'.join(remaining), prev=previous)
    model = sm.ols(formula, data).fit()
    return model, formula, remaining


d = pd.read_csv("DA_Clean NCSA Reserves_4.14.18-FINAL.csv")
d['ReleaseMonth'] = d['ReleaseMonth'].astype('str')
d['ReleaseYear'] = d['ReleaseYear'].astype('str')

result, f, selected = backward_selected(d,'ReservesLevel',['Region','Channel','Edition','Platform','ReleaseYear','ReleaseMonth','RelativeWeek','GameType'])

print(f)
print(selected)
print(result.summary())