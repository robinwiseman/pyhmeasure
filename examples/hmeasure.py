import pyhmeasure

c0scores = [0.0, 0.01, 0.02, 0.03, 0.2, 0.6, 0.66, 0.7]
c1scores = [0.3, 0.35, 0.36, 0.42, 0.5, 0.8, 0.82, 0.99]
cd_alpha = 2.0
cd_beta = 2.0
hm = pyhmeasure.PyHmeasure(c0scores, c1scores, cd_alpha, cd_beta)
print(f"H-Measure:{hm.h}")

