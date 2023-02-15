import numpy as np
import pingouin as pg
import pandas as pd

np.random.seed(123)
# mean, cov, n = [4, 5], [(1, .6), (.6, 1)], 30
# x, y = np.random.multivariate_normal(mean, cov, n).T

# z, w = np.random.multivariate_normal(mean, cov, n).T

# x = np.linspace(0, , 100) #+ np.random.normal(0, 1, 100) #get a list of 100 numbers from 0 to 10
# y = np.linspace(1, 11, 100) #get a list of 100 numbers from 0 to 10

z = np.random.randint(1, 10, 100000)

a = np.random.randint(2, 5, 100000)

x = z + np.random.randint(0, 20, 100000) 
y = 5*z + np.random.randint(0, 20, 100000) #np.roll(z, 0) #np.random.randint(0, 2, 10000)

x = z + np.linspace(0, 10, 100000) #+ np.random.normal(0, 1, 100000) 
y = z + np.linspace(10, 30, 100000) #+ np.random.normal(0, 1, 100000) 

df = pd.DataFrame({"x": x, "y": y, "z": z, "a": a})


corr = pg.corr(x, y)

print(corr)

print()

p_xy = pg.partial_corr(df, x = "x", y ="y", covar = "z")

#p_xz = pg.corr(x, z)

print(p_xy)

#print(p_xy['r'].values[0])   #this is the partial correlation coefficient as a number

print()

all_corrs = df.pcorr().round(3)
print(all_corrs)

#print(p_xz)

