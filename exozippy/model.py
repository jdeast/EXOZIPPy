import pymc as pm

rv_model = pm.Model()

with rv_model:
    tc = pm.Normal
