import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic  # plotting debug package
from sklearn import linear_model
from anscombe_data import anscombe_dict

anscombe1_df = pd.DataFrame(anscombe_dict['1'])
anscombe2_df = pd.DataFrame(anscombe_dict['2'])
anscombe3_df = pd.DataFrame(anscombe_dict['3'])
anscombe4_df = pd.DataFrame(anscombe_dict['4'])
anscombe_DF = pd.concat([anscombe1_df, anscombe2_df, anscombe3_df, anscombe4_df], axis=1)

# calculate the first and second moments of the dataset. All have the same values for mean and variance.
Y_vars = [anscombe_DF.var(ddof=1)[['Y1', 'Y2', 'Y3', 'Y4']]]
X_vars = [anscombe_DF.var(ddof=1)[['X1', 'X2', 'X3', 'X4']]]
Y_mean = [anscombe_DF.mean()[['Y1', 'Y2', 'Y3', 'Y4']]]
X_mean = [anscombe_DF.mean()[['X1', 'X2', 'X3', 'X4']]]

# calculate the estimate parameters for linear regression line (theta0, theta1)
reg = linear_model.LinearRegression()
reg_dict = dict()
for i in range(4):
    reg_dict[i] = reg.fit(np.array(anscombe_DF['X'+str(i+1)]).reshape(-1, 1), np.array(anscombe_DF['Y'+str(i+1)]).reshape(-1, 1))
    print(f"The estimated parameters for dataset (X{i+1},Y{i+1}) are:"
          f"\n\t\t\t slope = {reg_dict[i].coef_}, intercept = {reg_dict[i].intercept_}")

#  Least Squares equation
N = len(anscombe1_df)  # the number of data samples.
K = 2  # the number of parameters
ONE_vec = np.ones(N)
H = np.array([ONE_vec, anscombe_dict['1']['X1']]).transpose()
theta = inv(H.T @ H) @ H.T @ np.array(anscombe_dict['1']['Y1']).reshape(-1, 1)
print(f"The estimated parameters are: f{theta}")

fig, axs = plt.subplots(2, 2)
for i, ax in enumerate(axs.flat):
    ax.set(xlabel=r"X"+str(i+1), ylabel=r"Y"+str(i+1))
    ax.set_title(r"Dataset"+str(i+1))
    ax.scatter(anscombe_dict[str(i+1)]['X'+str(i+1)], anscombe_dict[str(i+1)]['Y'+str(i+1)],
               label=f"Anscombe Data {i+1}")
    ax.plot(np.linspace(0, 20, 20), reg_dict[i].predict(np.linspace(0, 20, 20).reshape(-1, 1)),
            color=[240/255, 216/255, 83/255], linewidth=1.8, label="Regression Line Sklearn")
    ax.minorticks_on()
    ax.grid(visible=True, which='major', color=[0.6, 0.6, 0.6], linestyle='-.', linewidth=1.4)
    ax.plot(np.linspace(0, 20, 20), theta[0]+theta[1]*np.linspace(0, 20, 20),
            color=[215/255, 0, 107/255], linestyle='--', linewidth=0.8, label="Regression Line L.S")
    ax.legend(loc="upper left", fontsize="x-small")
plt.tight_layout()
plt.savefig("Regression Lines.jpg")
plt.show()
