import pandas as pd
import statsmodels.formula.api as sm
import statsmodels.api as sma
from pathlib import Path

file = open('backwards_output2.txt','w')

def backward_selected(data, response, remaining, prev=[]):
    """
        based upon algorithm found at: https://planspace.org/20150423-forward_selection_with_statsmodels/
    """
    remain = remaining[:]
    selected = []
    prv = []
    best_formula = ''
    current_score, best_new_score = 0.05, 0.05
    starting_formula = "{response} ~ {selected}"

    for i in range(0,len(prev)):
        prv.append("+".join(prev[i]))
    if len(prv) > 0:
        previous = "+".join(prv)
        if len(previous) > 1:
            previous = previous + '+'
    else:
        previous = '1'

    while remain and current_score == best_new_score:
        current_score = 0.05
        scores_with_candidates = []
        sel = starting_formula.format(response=response, selected='+'.join(remain), prev=previous)
        s_file = "b_models/" + sel.replace(response + " ~ ", "") + '.pickle'
        s_file = s_file.replace('+', '')
        if Path(s_file).exists():
            sel_model = sma.load(s_file)
        else:
            if sel == best_formula:
                sel_model = best_model
            else:
                sel_model = sm.ols(sel, data).fit()
                sel_model.save(s_file)
        print("testing base: {}".format(sel), file = file)
        print("testing base: {}".format(sel))
        for candidate in remain:
            s = remain[:]
            s.remove(candidate)
            if len(s) == 0 and previous.endswith('+'):
                previous = previous[:-1]
            formula = starting_formula.format(response=response, selected='+'.join(s), prev=previous)
            f_file = "b_models/" + formula.replace(response + " ~ ", "") + '.pickle'
            f_file = f_file.replace('+', '')
            if Path(f_file).exists():
                model = sma.load(f_file)
            else:
                model = sm.ols(formula, data).fit()
                model.save(f_file)
            print("testing removal: {}".format(formula), file = file)
            print("testing removal: {}".format(formula))
            prf = sma.stats.anova_lm(model,sel_model)['Pr(>F)'].loc[1]
            print("testing removal: {} result: {}".format(formula, prf), file = file)
            print("testing removal: {} result: {}".format(formula, prf))
            scores_with_candidates.append((prf, candidate, model, formula))
        scores_with_candidates.sort()
        best_new_score, best_candidate, best_model, best_formula = scores_with_candidates.pop()
        if current_score < best_new_score:
            remain.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    if previous[:1] != "+" and len(selected) == 0:
        previous = previous[:-1]
    for s in selected:
        remaining.remove(s)
    formula = starting_formula.format(response=response, selected='+'.join(remaining), prev=previous)
    model = sm.ols(formula, data).fit()
    model.save('best_model_backward2.pickle')
    return model, formula, remaining


d = pd.read_csv("cleaned.csv")
d['ReleaseMonth'] = d['ReleaseMonth'].astype('str')
d['ReleaseYear'] = d['ReleaseYear'].astype('str')

result, f, selected = backward_selected(d,'ReservesLevel',['Channel','Edition','Platform','ReleaseYear','ReleaseMonth','RelativeWeek'])

print(f, file = file)
print(selected, file = file)
print(result.summary(), file = file)

file.flush()
file.close()

print(f)
print(selected)
print(result.summary())