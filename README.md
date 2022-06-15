# Understand-Your-Data
Data is crutial result in a system results. Implementing ML/DL methods straight forward without even observing the data may produce results that are far away from optimal. 
With Least Square method , Anscombe quarter data and Regression Line we'll notice the importance of "data analyze".

### Fit a linear model to 2D data set
In order to emphasize the significance of data analysis before using existing models one of the easiest ways is to show it on 2D dataset. 
Let assume we have data which arrange as set of points such as:
$(x_{i},y_{i})_{i=1}^{N}$

The subject is to fit a linear line that suits the "best" for the scatter of the data, in order for us to understand the function that represents the currently dataset and which will represents the future dataset the "best". But the question is __in the meaning of "what"__ we may say the linear line fit the "best" for the dataset? One approach try to minimize the squared error and I'll focus on that approach.

The statistical model is represents as 
$y_{i}=\alpha + \beta x_{i} + \epsilon_{i}$, were we need to estimate $\alpha , \beta$ 
and $\epsilon$ is random noise.

### Least Square
Least square approach is a generic one which can deals with linear models for larger number of unknown parameters and under several assumption can derive a close formula for a solution.

The model is set as: 
$$\mathbf{y} \approx h(\mathbf{\theta})$$
and for a linear model will describe it as: 
$$\mathbf{y} \approx H\mathbf{\theta}$$
The approximation defined instead of adding the random noise. The cost function is the objective function we need to derive in order to find the parameters which minimize it.
$$C_{LS}(\theta) = ||\mathbf{y}-H\mathbf{\theta} ||_{2}^{2}$$

The derivitive is the gradient for such objective:
$$\nabla(C_{LS}(\mathbf{\theta})) = 2H^{T}H\mathbf{\theta}-2H^{T}\mathbf{y}$$
To minimize let equal it to zero and we'll get:
$$\hat{\mathbf{\theta}} = (H^{T}H)^{-1}H^{T}\mathbf{y}$$

### Anscombe dataset
Anscombe created four datasets which has several common moments (mean, variance correlation and etc). At a glance, it is noticeable that the four scatter plot graph (attached below and in the script) are not identical at all and do not behave similarly, but their __statistical__ charectaristic are the same. Implementaion of least square formula or fitting the model by regression line (is the same) and both get the same values for the two estimated parameters. 



