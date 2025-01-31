# Linear Regression and Gradient Visualization
---
## Introduction
Linear regression is one of the fundamental tools in statistics and machine learning that allows modeling relationships between variables. It is widely used in data analysis, forecasting, and understanding the impact of one variable on another.

In this project, we will explain how linear regression works and how it can be trained using the **Gradient Descent algorithm**. We will also demonstrate a **visualization of the optimization process**, illustrating step by step how the algorithm updates the weights to minimize the cost function.

---
### Linear Regression Visualization
Linear regression can be applied to a single input variable (_univariate model_) or multiple input variables (_multivariate model_). Below are examples of fitting the model in both two and three dimensions:

- **Linear regression in 2D space** – the red line represents the model best fitted to the data:
  
<img src="attachments/Pasted image 20250128215155.png" alt="2D Linear Regression" width="800">

* **Linear regression in 3D space** – the red plane represents the _model best fitted to the data_:
  
<img src="attachments/Pasted image 20250128215402.png" alt="3D Linear Regression" width="800">

---
## 1. What is Linear Regression?

Linear regression aims to fit a line (_or a hyperplane in the case of multiple dimensions_) to the data in such a way as to minimize the error between the predicted and actual values.

For a single input variable xxx, the model is represented by:

$$y=w_1x+w_2$$​

where:
* $w_1 -$ the slope (direction coefficient),
* $w_2 -$ the bias (intercept).

For multiple variables, the model takes the form of a multivariate equation:

$$y = \sum_{i=1}^{n}(w_n^{(i)}*x_{features_n}^{(i)})$$

or equivalently:

$$y=w_1x_1 + w_2x_2 +... +w_nx_n + b$$

---
## 2. Prediction in Linear Regression

Once the model is trained, we can use it to make predictions. For example, given the following weights:

<center>

| w1 (Feature) | w2 (bias) |
| :----------: | :-------: |
|     0.97     |  -3.96   |

</center>


We want to make predictions for the following data:
<center>

| No. | x (Feature) |
| --- | ----------- |
| 1   | -50         |
| 2   | 0           |
| 3   | 50          |

</center>

**Adding the bias column**
Before making predictions, we need to add a column for bias, which ensures that the model can fit the data even when it does not pass through the origin (0, 0). The role of bias and its importance are explained in detail in **[Section 6](#6.the.role.of.bias.in.linear.regression**)**.

<center>

| No. | x_1 (Feature) | x_2 (bias column) |
| --- | ------------- | ----------------- |
| 1   | -50           | 1                 |
| 2   | 0             | 1                 |
| 3   | 50            | 1                 |

</center>
**Matrix multiplication for prediction:**

   $$\hat{y} = X \cdot W $$
   
**Sample calculations:**

   $$\hat{y}_1 = (-50 \cdot 0.97) + (1 \cdot -3.96) = -50.46 $$
   $$\hat{y}_2 = (0 \cdot 0.97) + (1 \cdot -3.96) = -3.96$$
   $$\hat{y}_3 = (50 \cdot 0.97) + (1 \cdot -3.96) = 44.54$$
   
  **Matrix form:**
  
   $$\hat{y} = \begin{bmatrix} -50 & 1 \\ 0 & 1 \\ 50 & 1 \end{bmatrix} \cdot \begin{bmatrix} 0.97 \\ -3.96 \end{bmatrix} = \begin{bmatrix} -50.46 \\ -3.96 \\ 44.54 \end{bmatrix}  $$

**Below is the plot representing the prediction:**

<img src="attachments/Pasted image 20250127175312.png" alt="Prediction Results" width="800">

---
## 3. Cost Function
The cost function measures how well the model fits the data. In linear regression, it is defined as the mean of the squared errors between the actual and predicted values.

The formula for the cost function is as follows:

$$
J(\Theta) = \frac{1}{2m} \sum_{i=1}^{m}(y^{(i)}-activation^{(i)})^2
$$

where:
* m - the number of samples,
* y - actual values,
* activation - predicted values.

As we can observe from the formula, the cost function in linear regression is a **quadratic function**, meaning its shape forms a **paraboloid**, as we will see in the **[section dedicated to gradient visualization](#5.gradient.visualization)**.

**Python implementation of the cost function:**
```python
error = np.square(y-activation).sum()/(2.0*len(y))
```

**Below is the visualization of the cost function:**

<img src="attachments/Pasted image 20250128221253.png" alt="Cost Function Visualization" width="800">

To summarize, the cost function calculates the model's error, and this error is iteratively optimized toward the minimum. The process of such optimization is described in the next section.

---
## 4. Gradient Descent
To update the model's weights, we use the **gradient descent algorithm**. The gradient tells us the direction in which the weights should be adjusted to minimize the cost function.

The cost function is optimized **iteratively**, meaning that with each iteration of the loop, the gradient descent algorithm updates the weights, bringing the model closer to the minimum of the cost function.

The gradient formula, derived as the partial derivative of the cost function:

$$
\frac{\partial J(w)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left( y^{(i)} - activation^{(i)} \right) x_j^{(i)}
$$

where:
* m - the number of samples,
* y - actual values,
* activation - predicted values,
* x - input data.

**Gradient update implementation in Python:**
```python
delta_w = self.eta * x_1.shape[0])np.dot((y - activation), x_1) / x_1.shape[0]
```
This expression tells us **how much the model’s weights should be adjusted** to minimize the cost function.

---
#### **The role of the learning rate**
As shown in the code, before updating the weights, they are multiplied by the **learning rate** `self.eta`. This is a crucial parameter that controls the speed of learning:

- **A learning rate that is too small** will result in slow convergence,
- **A learning rate that is too large** may cause the model to diverge and fail to converge to the minimum of the cost function.
---
##  **5. Gradient Visualization**

In this section, we visualize the **gradient path** in the cost function space. The visualization illustrates how the model’s weights evolved with each iteration and how they impacted the value of the cost function (representing the averaged prediction error).

By observing the **trajectory of the gradient**, we can see whether the model updates its weights in the correct direction, leading to convergence.

---
#### **1. Convergent Function**

A convergent function is a case where, with each iteration, the cost function successfully decreases until reaching a global minimum. In such cases, we can assume that the learning rate (`self.eta`) has been set appropriately.

Below are examples of **gradient descent paths** visualized in both 2D and 3D data:

<img src="attachments/Pasted image 20250124213901.png" alt="Convergent Gradient in 2D" width="800">


<img src="attachments/Pasted image 20250129162959.png" alt="Convergent Gradient in 3D" width="800">

---
#### **2. Divergent Function**

A **divergent function** occurs when the learning rate is set too high. Although the gradient correctly points in the direction to minimize the cost function, the learning rate causes the updates to overshoot the minimum, leading to divergence.

Below are examples of gradient descent paths in divergent cases for both 2D and 3D data:

<img src="attachments/Pasted image 20250124214806.png" alt="Divergent Gradient in 2D" width="800">

<img src="attachments/Pasted image 20250129163019.png" alt="Divergent Gradient in 3D" width="800">

---
#### **The Importance of the Learning Rate**

As seen in the illustrations, the learning rate is critical for optimal training.

- **If the learning rate is too large**, the gradient descent will overshoot the minimum, increasing the error instead of minimizing it.
- **If the learning rate is too small**, the cost function will be minimized too slowly, making the training inefficient.

---
#### **How to Choose the Learning Rate?**

The answer may seem amusing—it’s often determined by **trial and error**. Initially, a random learning rate is selected, and then adjustments are made based on performance:

- **If the learning rate is too large**, we reduce it.
- **If it’s too small**, we increase it.

This repository aims to **visualize this problem** and help you understand how the learning rate and the dataset influence the model’s learning process.

---
## **6. The Role of Bias in Linear Regression**

The **bias** in linear regression allows the model to shift relative to the **X-axis**. Without it, the regression line always passes through the point (0,0)(0, 0)(0,0), which can lead to inaccurate predictions.

**Linear regression without bias:**

<img src="attachments/Pasted image 20250129163050.png" alt="Linear Regression Without Bias" width="800">

**Linear regression with bias:**

<img src="attachments/Pasted image 20250129163056.png" alt="Linear Regression With Bias" width="800">

As shown in the above illustrations, the bias enables the model to vertically shift the regression line, allowing it to fit the data even when the points do not intersect at (0,0)(0, 0)(0,0).

In practice, to implement bias in linear regression, a column of ones is added to the input data, and weights for this column are optimized during training.

**Python implementation of bias:**
```python
def _get_ones(self, x):
	ones = np.ones((x.shape[0], 1))
	x_1 = np.append(x.copy(), ones, axis=1)
	return x_1
```

---
## **7. Project Structure**

📂 **Linear_Regression_Visualization**  
│── 📄 **README.md** — This file  
├── 📄 **Linear_Regression_Gradient_Example.ipynb** — Notebook with code and visualizations  
│── 📂 **attachments** — Screenshots and plots

---
## **Summary**

This project provides an **intuitive understanding of how linear regression works** and how the **cost function optimization process** takes place. The gradient visualization allows you to see how the model learns over time.

The notebook _Linear_Regression_Gradient_Example.ipynb_ includes a sample implementation of linear regression with gradient descent visualization. The function is designed for educational purposes, so have fun exploring it! 😉
