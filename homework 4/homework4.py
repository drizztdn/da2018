import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sma


def test_interactions(f, outcome, predictors,col_dt = None):
    d = pd.read_csv(f)
    if col_dt != None:
        for i in col_dt:
            d[i[0]] = d[i[0]].astype(i[1])
    f = outcome +" ~ " + "+".join(predictors)
    f2 = outcome +" ~ " + "*".join(predictors)

    no_inter = sm.ols(formula=f, data=d).fit()
    inter = sm.ols(formula=f2, data=d).fit()

    print("no interactions")
    print(no_inter.summary())

    print("interactions")
    print(inter.summary())
    return no_inter, inter

print('Part A')
no_interactions, interactions  = test_interactions("cats.csv",'Hwt',['Bwt', 'Sex'])

tc = pd.DataFrame(columns=['Bwt','Sex'])
tc.loc[0] = [3.4, 'F']
outcome = interactions.predict(tc)
outcome1 = no_interactions.predict(tc)
print("interactions - sigma's heart weight: " + str(outcome[0]))
print("no interactions - sigma's heart weight: " + str(outcome1[0]))
print()


print('Part B')
no_interactions, interactions  = test_interactions("trees.csv",'Volume',['Girth', 'Height'])

no_interactions, interactions  = test_interactions("trees.csv",'Volume',['np.log(Girth)', 'np.log(Height)'])

print()
print('Part C')
print('without weight')
no_interactions, part_int  = test_interactions("mtcars.csv",'mpg',['hp', 'cyl'])
print('with weight')
no_interactions, interactions  = test_interactions("mtcars.csv",'mpg',['hp', 'cyl','wt'])

no_interactions, interactions  = test_interactions("mtcars.csv",'mpg',['hp', 'cyl'],[['cyl','str']])

tc = pd.DataFrame(columns=['hp','cyl','wt'])
tc.loc[0] = [100,4,2.1]
tc.loc[1] = [210,8,3.9]
tc.loc[2] = [200,6,2.9]
print(tc)
outcome = part_int.predict(tc)
print(outcome)

print()
print('Part D')
d = pd.read_csv("diabetes.csv")
print(d)
d = d.dropna(subset=['chol'])
d = d.dropna(subset=['age'])
d = d.dropna(subset=['gender'])
d = d.dropna(subset=['height'])
d = d.dropna(subset=['weight'])
d = d.dropna(subset=['frame'])
d = d.dropna(subset=['waist'])
d = d.dropna(subset=['hip'])
d = d.dropna(subset=['location'])

print(d)

diaNull = sm.ols(formula="chol ~ 1",data=d).fit()
diaFull = sm.ols(formula="chol ~ age*gender*height*frame+waist*height*hip+location",data=d).fit()
print("null")
print(diaNull.summary())
print("full")
print(diaFull.summary())


def forward_selected(data, response, remaining, prev=[]):
    """
        based upon algorithm found at: https://planspace.org/20150423-forward_selection_with_statsmodels/
    """
    selected = []
    prv = []
    current_score, best_new_score = 0.05, 0.05
    starting_formula = "{response} ~ {prev}{selected}"

    for i in range(0,len(prev)):
        prv.append("*".join(prev[i]))
    if len(prv) > 0:
        previous = "+".join(prv)
    else:
        previous = '1'

    while remaining and current_score == best_new_score:
        current_score = 0.05
        scores_with_candidates = []
        sel = starting_formula.format(response=response, selected='*'.join(selected), prev=previous)
        sel_model = sm.ols(sel, data).fit()
        if previous == "1" or previous == "":
            previous = ""
        else:
            if previous[:1] != "+":
                previous = previous+"+"
        for candidate in remaining:
            formula = starting_formula.format(response=response, selected='*'.join(selected + [candidate]), prev=previous)
            model = sm.ols(formula, data).fit()
            prf = sma.stats.anova_lm(sel_model,model)['Pr(>F)'].loc[1]
            scores_with_candidates.append((prf, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop(0)
        if best_new_score < current_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    if previous[:1] != "+" and len(selected) == 0:
        previous = previous[:-1]
    formula = starting_formula.format(response=response, selected='*'.join(selected), prev=previous)
    model = sm.ols(formula, data).fit()
    return model, formula, selected


result, f, selected = forward_selected(d,'chol',['age','gender','frame','weight'])
prev = []
if selected:
    prev.append(selected)
result1, f1, selected1 = forward_selected(d,'chol',['waist','height','hip'],prev=prev)
if selected1:
    prev.append(selected1)
result2, f2, selected2 = forward_selected(d,'chol',['location'],prev=prev)
print("forward selected: " + f2)
print(result2.summary())


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
        for candidate in remain:
            s = remain[:]
            s.remove(candidate)
            if len(s) == 0 and previous.endswith('+'):
                previous = previous[:-1]
            formula = starting_formula.format(response=response, selected='*'.join(s), prev=previous)
            model = sm.ols(formula, data).fit()
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

prev = [['age','gender','frame','weight'],['waist','height','hip']]
result, f, selected = backward_selected(d,'chol',['location'],prev=prev)
prev.pop()
if selected:
    prev.append(selected)
result1, f1, selected1 = backward_selected(d,'chol',['waist','height','hip'],prev=prev)
prev.pop()
if len(prev) == 2:
    prev.pop()
if selected1:
    prev.append(selected1)
if selected:
    prev.append(selected)
result2, f2, selected2 = backward_selected(d,'chol',['age','gender','frame','weight'],prev=prev)

print("backward selected: " + f2)
print(result2.summary())