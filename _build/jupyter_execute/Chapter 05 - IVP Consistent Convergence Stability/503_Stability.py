#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/john-s-butler-dit/Numerical-Analysis-Python/blob/master/Chapter%2005%20-%20IVP%20Consistent%20Convergence%20Stability/503_Stability.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Stability of a Multistep method
# 
# #### John S Butler 
# john.s.butler@tudublin.ie 
# 
# [Course Notes](https://johnsbutler.netlify.com/files/Teaching/Numerical_Analysis_for_Differential_Equations.pdf)    [Github](https://github.com/john-s-butler-dit/Numerical-Analysis-Python)

# ## Overview
# A one-step or multistep method is used to approximate the solution of an initial value problem of the form
# \begin{equation} \frac{dy}{dt}=f(t,y),\end{equation}
# with the initial condition
# \begin{equation} y(a)=\alpha.\end{equation}
# The method should only be used if it satisfies the three criteria:
# 1. that difference equation is consistent with the differential equation;
# 2. that the numerical solution converges to the exact answer of the differential equation;
# 3. that the numerical solution is stable.
# 
# In the notebooks in this folder we will illustate examples of consistent and inconsistent, convergent and non-convergent, and stable and unstable methods. 
# 
# This notebook focuses on stable and unstable methods. The video below outlines the notebook.

# In[1]:


from IPython.display import HTML
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/c0Gr5mM3Np0" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')


# ## Introduction to unstable
# This notebook illustrates an unstable multistep method for numerically approximating an initial value problem
# \begin{equation} \frac{dy}{dt}=f(t,y), \end{equation}
# with the initial condition
# \begin{equation} y(a)=\alpha,\end{equation}
# using the Modified Abysmal Kramer-Butler method. The method is named after the great [Cosmo Kramer]( https://en.wikipedia.org/wiki/Cosmo_Kramer) and myself [John Butler](https://johnsbutler.netlify.com).
# 
# ## 2-step Abysmal Kramer-Butler Method
# 
# The 2-step Abysmal Kramer-Butler difference equation is given by
# \begin{equation}w_{i+1} = w_{i-1} + h(4f(t_i,w_i)-2f(t_{i-1},w_{i-1})) \end{equation}
# by changing $F$, the Modified Abysmal Butler Method (see convergent and consistent notebooks), the Abysmal Kramer-Butler method is consistent with the differential equation and convergent with the exact solution (see below for proof).
# But the most important thing is that method is weakly stable, it fluctuates widely around the exact answer, just like it's name sake Kramer (for examples see any Seinfeld episode).

# ## Definition of Stability
# The stability of a numerical method is not as tangable as consistency and convergence but when you see an unstable solution it is obvious.
# 
# 
# To determine the stabilty of a multistep method we need three definitions:
# 
# 
# ### Definition: Characteristic Equation
# Associated with the difference equation 
# \begin{equation} w_0=\alpha \ \ \ w_1=\alpha_1 \ \ \ ... \ \ \ w_{m-1}=\alpha_{m-1} \end{equation}
# \begin{equation}w_{i+1} = a_{m-1}w_{i}+a_{m-2}w_{i-1}+...+a_{0}w_{i+1-m} +hF(t_i,h,w_{i+1},...,w_{i+1-m}),\end{equation}
# is the __characteristic equation__ given by
# \begin{equation}\lambda^{m} - a_{m-1}\lambda^{m-1}-a_{m-2}\lambda^{m-2}-...-a_{0} =0. \end{equation}
# 
# ### Definition: Root Condition 
# 
# Let $\lambda_1,...,\lambda_m$ denote the roots of the that characteristic equation
# \begin{equation}\lambda^{m} - a_{m-1}\lambda^{m-1}-a_{m-2}\lambda^{m-2}-...-a_{0} =0 \end{equation}
# associated with the multi-step difference method
# \begin{equation} w_0=\alpha \ \ \ w_1=\alpha_1 \ \ \ ... \ \ \ w_{m-1}=\alpha_{m-1} \end{equation}
# \begin{equation} w_{i+1} = a_{m-1}w_{i}+a_{m-2}w_{i-1}+...+a_{0}w_{i+1-m} +hF(t_i,h,w_{i+1},...,w_{i+1-m}),\end{equation}
# If $|\lambda_{i}|\leq 1$ for each $i=1,...,m$ and all roots with absolute value 1
# are simple roots then the difference equation is said to satisfy the __root condition__.
# 
# ### Definition: Stability
# 1. Methods that satisfy the root condition and have $\lambda=1$ as the only root 
# of the characteristic equation of magnitude one and all other roots are 0 are called __strongly stable__;
# 2. Methods that satisfy the root condition and have more than one distinct root
# with magnitude one are called __weakly stable__;
# 3. Methods that do not satisfy the root condition are called __unstable__.
# 
# All one step methods, Adams-Bashforth and Adams-Moulton methods are all stongly stable.
# 
# 
# 
# ## Intial Value Problem
# To illustrate stability we will apply the method to a linear intial value problem given by
# \begin{equation}y^{'}=t-y, \ \ (0 \leq t \leq 2), \end{equation}
# with the initial condition
# \begin{equation}y(0)=1,\end{equation}
# with the exact solution
# \begin{equation}y(t)= 2e^{-t}+t-1.\end{equation}

# ## Python Libraries

# In[2]:


import numpy as np


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt # side-stepping mpl backend
import matplotlib.gridspec as gridspec # subplots
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


# ### Defining the function
# $$ f(t,y)=t-y.\end{equation}

# In[3]:


def myfun_ty(t,y):
    return t-y


# ## Discrete Interval
# Defining the step size  $h$  from the interval range  $a \leq t \leq b$  and number of steps  $N$ 
# \begin{equation}h=\frac{b-a}{N}.\end{equation}
#  
# This gives the discrete time steps,
# \begin{equation}t_i=t_0+ih,\end{equation}
# where  $t_0=a.$

# In[4]:


# Start and end of interval
b=2
a=0
# Step size
N=16
h=(b-a)/(N)
t=np.arange(a,b+h,h)
fig = plt.figure(figsize=(10,4))
plt.plot(t,0*t,'o:',color='red',label='Coarse Mesh')
plt.xlim((0,2))
plt.ylim((-0.1,.1))

plt.legend()
plt.title('Illustration of discrete time points')
plt.show()


# ## 2-step  Abysmal Kramer-Butler Method
# 
# For this initial value problem 2-step Abysmal Kramer-Butler difference equation is
# \begin{equation}w_{i+1} = w_{i-1} + h(4(t_i- w_i)-2(t_{i-1}-w_{i-1})) \end{equation}
# by changing $F$, the Modified Abysmal Butler Method, is consistent and convergent.
# 
# For $i=0$ the system of difference equation is:
# \begin{equation}w_{1} = w_{-1} + h(4(t_0-w_0)-2(t_{-1}-w_{-1})) \end{equation}
# this is not solvable as <font color='red'> $w_{-1}$ </font> is unknown.
# 
# For $i=1$ the difference equation is:
# \begin{equation}w_{2} = w_{0} + h(4(t_1-w_1)-2(t_{0}-w_{0})) \end{equation}
# this is not solvable as <font color='red'> $w_{1}$  </font> is unknown. $w_1$ can be  approximated using a one step method. Here, as the exact solution is known,
# \begin{equation}w_1=2e^{-t_1}+t_1-1.\end{equation}
# 

# In[5]:


### Initial conditions
IC=1
w=np.zeros(len(t))
y=(2)*np.exp(-t)+t-1
w[0]=IC
w[1]=y[1]


# ### Loop

# In[6]:


for i in range (1,N):
    w[i+1]=(w[i-1]+h*(4*myfun_ty(t[i],w[i])-2*myfun_ty(t[i-1],w[i-1])))   


# ### Plotting solution

# In[7]:


def plotting(t,w,y):
    
    fig = plt.figure(figsize=(10,4))
    plt.plot(t,w,'^:',color='red',label='Abysmal Kramer-Butler (N)')
    plt.plot(t,y, 'o-',color='black',label='Exact')
    plt.xlabel('time')
    plt.legend()
    plt.title(' Abysmal Kramer-Butler')
    plt.show 


# The plot below shows the Abysmal Kramer-Butler approximation $w_i$ (red) and the exact solution $y(t_i)$ (black) of the intial value problem. 
# 
# The numerically solution initially approximates the exact solution $(t<.5)$ reasonably but then the instability creeps in such that the numerical approximation starts to widely oscilate around the exact solution.

# In[8]:


plotting(t,w,y)


# The table below illustrates the absolute error and the signed error of the numerical method.

# In[9]:


n=10
d = {'time': t[0:n], 'Abysmal Kramer-Butler w': w[0:n],'Exact Error abs':np.abs(y[0:n]-w[0:n]),
     'Exact Error':(y[0:n]-w[0:n])}
df = pd.DataFrame(data=d)
df


# # Theory
# ## Consistent 
# The Abysmal Kramer-Butler method does satisfy the consistency condition
# \begin{equation}\tau_{i}(h)=\frac{y_{i+1}-y_{i-1}}{2h}-\frac{1}{2}[4(f(t_i,y_i)-2f(t_{i-1},y_{i-1})] \end{equation}
# As $h \rightarrow 0$  
# \begin{equation}\frac{1}{2}[4(f(t_i,y_i)-2f(t_{i-1},y_{i-1})] \rightarrow f(t_i,y_i).\end{equation}
# While as $h \rightarrow 0$  
# \begin{equation}\frac{y_{i+1}-y_{i-1}}{2h} \rightarrow \frac{y^{'}}{1}=\frac{f(t_i,y_i)}{1}.\end{equation}
# Hence as $h \rightarrow 0$ $$\frac{y_{i+1}-y_{i}}{h}-\frac{1}{2}[4(f(t_i,y_i)-2f(t_{i-1},y_{i-1})]\rightarrow f(t_i,y_i)-f(t_i,y_i)=0,\end{equation}
# which means it is consistent.
# 
# ## Convergent 
# The Abysmal Kramer-Butler method does satisfy the Lipschitz condition:
# \begin{equation}F(t,w:h)-F(t,\hat{w}:h)=\frac{4}{2}f(t,w_i)-\frac{2}{2}f(t-h,w_{i-1}))-(\frac{4}{2}f(t,\hat{w}_{i})-\frac{2}{2}f(t-h,\hat{w}_{i-1}))),\end{equation}
# \begin{equation}F(t,w:h)-F(t,\hat{w}:h)=\frac{4}{2}(f(t,w_i)-f(t,\hat{w}_i))-\frac{2}{2}(f(t-h,w_{i-1}))-f(t-h,\hat{w}_{i-1}))),\end{equation}
# \begin{equation}F(t,w:h)-F(t,\hat{w}:h)\leq\frac{4}{2}L|w_i-\hat{w_i}|+\frac{2}{2}L|w-\hat{w}|\leq \frac{6}{2} L|w_i-\hat{w_i}|,\end{equation}
# This means it is internally convergent,
# \begin{equation}|w_i-\hat{w_i}|\rightarrow 0\end{equation}
# as $h \rightarrow 0$.
# ## Stability
# The Abysmal Kramer-Butler method does __not__ satisfy the stability condition.
# The characteristic equation of the 
# \begin{equation}w_{i+1} = w_{i-1} + h(4f(t_i,w_i)-2f(t_{i-1},w_{i-1})) \end{equation}
# is
# \begin{equation}\lambda^2 = 1, \end{equation}
# This has two roots $\lambda=1$ and   $\lambda=-1$, hence the method is weakly stable.

# In[9]:




