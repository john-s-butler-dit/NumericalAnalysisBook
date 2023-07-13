#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/john-s-butler-dit/Numerical-Analysis-Python/blob/master/Chapter%2001%20-%20Euler%20Methods/101_Euler_method_with_Theorems_Growth_function.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # First Order Initial Value Problem
#  
# 
# The more general form of a first order Ordinary Differential Equation is: 
# \begin{equation}
# y^{'}=f(t,y).
# \end{equation}
# This can be solved analytically by integrating both sides but this is not straight forward for most problems.
# Numerical methods can be used to approximate the solution at discrete points.
# 

# ## Euler method
# 
# The simplest one step numerical method is the Euler Method named after the most prolific of mathematicians [Leonhard Euler](https://en.wikipedia.org/wiki/Leonhard_Euler) (15 April 1707 – 18 September 1783) .
# 
# The general Euler formula for the first order differential equation
# \begin{equation}
# y^{'} = f(t,y), 
# \end{equation}
# 
# approximates the derivative at time point $t_i$,
# 
# \begin{equation}
# y^{'}(t_i) \approx \frac{w_{i+1}-w_i}{t_{i+1}-t_{i}},
# \end{equation}
# 
# where $w_i$ is the approximate solution of $y$ at time $t_i$.
# 
# This substitution changes the differential equation  into a __difference__ equation of the form 
# 
# \begin{equation}
# \frac{w_{i+1}-w_i}{t_{i+1}-t_{i}}=f(t_i,w_i). 
# \end{equation}
# 
# Assuming uniform stepsize $t_{i+1}-t_{i}$ is replaced by $h$, re-arranging the equation gives
# \begin{equation} 
# w_{i+1}=w_i+hf(t_i,w_i).
# \end{equation}
#  This can be read as the future $w_{i+1}$ can be approximated by the present $w_i$ and the addition of the input to the system $f(t,y)$ times the time step.
# 

# In[1]:


## Library
import numpy as np
import math 
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt # side-stepping mpl backend
import matplotlib.gridspec as gridspec # subplots
import warnings

warnings.filterwarnings("ignore")


# ## Population growth
# 
# The general form of the population growth differential equation is: 
# \begin{equation} 
# y^{'}=\epsilon y 
# \end{equation}
# where $\epsilon$ is the growth rate. The initial population at time $a$ is 
# \begin{equation} 
# y(a)=A,
#  \end{equation}
# \begin{equation}
#  a\leq t \leq b. 
# \end{equation}
# Integrating gives  the general analytic (exact) solution: 
# \begin{equation}
#  y=Ae^{\epsilon x}. 
# \end{equation}
# We will use this equation to illustrate the application of the Euler method.
#       
# ## Discrete Interval
# The continuous time $a\leq t \leq b $ is discretised into $N$ points seperated by a constant stepsize
# \begin{equation} 
# h=\frac{b-a}{N}.
# \end{equation}
# Here the interval is $0\leq t \leq 2$ 
# \begin{equation}
#  h=\frac{2-0}{20}=0.1,
# \end{equation}
# this gives the 21 discrete points:
# \begin{equation}
#  t_0=0, \ t_1=0.1, \ ... t_{20}=2. 
# \end{equation}
# This is generalised to 
# \begin{equation}
# t_i=0+i0.1, \ \ \ i=0,1,...,20.
# \end{equation}
# The plot below shows the discrete time steps.

# In[2]:


### Setting up time
t_end=2.0
t_start=0
N=20
h=(t_end-t_start)/(N)
time=np.arange(t_start,t_end+0.01,h)
fig = plt.figure(figsize=(10,4))
plt.plot(time,0*time,'o:',color='red')
plt.xlim((0,2))
plt.title('Illustration of discrete time points for h=%s'%(h))
plt.plot();


# ## Initial Condition
# To get a specify solution to a first order initial value problem, an __initial condition__ is required.
# 
# For our population problem the intial condition is:
# \begin{equation}
# y(0)=10.
# \end{equation}
# This gives the analytic solution
# \begin{equation}
# y=10e^{\epsilon t}.
# \end{equation}
# ### Growth rate 
# Let the growth rate
# \begin{equation}
# \epsilon=0.5
# \end{equation}
# giving the analytic solution.
# \begin{equation}
# y=10e^{0.5 t}.
# \end{equation}
# The plot below shows the exact solution on the discrete time steps.

# In[3]:


## Analytic Solution y
y=10*np.exp(0.5*time)

fig = plt.figure(figsize=(10,4))
plt.plot(time,y,'o:',color='black')
plt.xlim((0,2))
plt.xlabel('time')
plt.ylabel('y')
plt.title('Analytic (Exact) solution')
plt.plot();


# ## Numerical approximation of Population growth
# The differential equation is transformed using the Euler method into a difference equation of the form
# \begin{equation} w_{i+1}=w_{i}+h \epsilon w_i. \end{equation}
# This approximates a series of of values $w_0, \ w_1, \ ..., w_{N}$.
# For the specific example of the population equation the difference equation is
#      \begin{equation} w_{i+1}=w_{i}+h 0.5 w_i. \end{equation}
# where $w_0=10$. From this initial condition the series is approximated.
# The plot below shows the exact solution $y$ in black circles and Euler approximation $w$ in blue squares. 

# In[4]:


w=np.zeros(N+1)
w[0]=10
for i in range (0,N):
    w[i+1]=w[i]+h*(0.5)*w[i]

fig = plt.figure(figsize=(10,4))
plt.plot(time,y,'o:',color='black',label='exact')
plt.plot(time,w,'s:',color='blue',label='Euler')
plt.xlim((0,2))
plt.xlabel('time')
plt.legend(loc='best')
plt.title('Analytic and Euler solution')
plt.plot();


# ## Numerical Error
# With a numerical solution there are two types of error: 
# * local truncation error at one time step; 
# * global error which is the propagation of local error. 
# 
# ### Derivation of  Euler Local truncation error
# The left hand side of a initial value problem $\frac{dy}{dt}$ is approximated by __Taylors theorem__ expand about a point $t_0$ giving:
# \begin{equation}
# y(t_1) = y(t_0)+(t_1-t_0)y^{'}(t_0) + \frac{(t_1-t_0)^2}{2!}y^{''}(\xi), \ \ \ \ \ \ \xi \in [t_0,t_1]. 
# \end{equation}
# Rearranging and letting $h=t_1-t_0$ the equation becomes
# \begin{equation}
# y^{'}(t_0)=\frac{y(t_1)-y(t_0)}{h}-\frac{h}{2}y^{''}(\xi). 
# \end{equation}
# From this the local truncation error is
# \begin{equation}
# \tau y^{'}(t_0)\leq \frac{h}{2}M  
# \end{equation}
# where $y^{''}(t) \leq M $.
# #### Derivation of  Euler Local truncation error for the Population Growth
# In most cases $y$ is unknown but in our example problem there is an exact solution which can be used to estimate the local truncation
# \begin{equation}
# y'(t)=5e^{0.5 t},
# \end{equation}
# \begin{equation}
# y''(t)=2.5e^{0.5 t}.
# \end{equation}
# From this a maximum upper limit can be calculated for $y^{''} $ on the interval $[t_0,t_1]=[0,0.1]$
# \begin{equation}
# y''(0.1)=2.5e^{0.1\times 0.5}=2.63=M,
# \end{equation}
# \begin{equation}
# \tau=\frac{h}{2}2.63=0.1315. 
# \end{equation}
# The plot below shows the exact local truncation error $|y-w|$ (red triangle) and the upper limit of the Truncation error (black v) for the first two time points $t_0$ and $t_1$.

# In[5]:


fig = plt.figure(figsize=(10,4))
plt.plot(time[0:2],np.abs(w[0:2]-y[0:2]),'^:'
         ,color='red',label='Error |y-w|')
plt.plot(time[0:2],0.1*2.63/2*np.ones(2),'v:'
         ,color='black',label='Upper Local Truncation')
plt.xlim((0,.15))
plt.xlabel('time')
plt.legend(loc='best')
plt.title('Local Truncation Error')
plt.plot();


# ## Global Error
# The error does not stay constant accross the time this is illustrated in the figure below for the population growth equation. The actual error (red triangles) increases over time while the local truncation error (black v) remains constant.

# In[6]:


fig = plt.figure(figsize=(10,4))
plt.plot(time,np.abs(w-y),'^:'
         ,color='red',label='Error |y-w|')
plt.plot(time,0.1*2.63/2*np.ones(N+1),'v:'
         ,color='black',label='Upper Local Truncation')
plt.xlim((0,2))
plt.xlabel('time')
plt.legend(loc='best')
plt.title('Why Local Truncation does not extend to global')
plt.plot();


# ## Theorems
# To theorem below proves an upper limit of the global truncation error.
# ### Euler Global Error
# __Theorem Global Error__
# 
# Suppose $f$ is continuous and satisfies a Lipschitz Condition with constant
# L on $D=\{(t,y)|a\leq t \leq b, -\infty < y < \infty \}$ and that a constant M
# exists with the property that 
# \begin{equation}
# |y^{''}(t)|\leq M. 
# \end{equation}
# Let $y(t)$ denote the unique solution of the Initial Value Problem
# 
# \begin{equation}
# y^{'}=f(t,y) \ \ \ a\leq t \leq b, \ \ \ y(a)=\alpha, 
# \end{equation}
# and $w_0,w_1,...,w_N$ be the approx generated by the Euler method for some
# positive integer N.  Then for $i=0,1,...,N$
# \begin{equation}
# |y(t_i)-w_i| \leq \frac{Mh}{2L}|e^{L(t_i-a)}-1|. 
# \end{equation}
# 
# ### Theorems about Ordinary Differential Equations
# __Definition__
# 
# A function $f(t,y)$ is said to satisfy a __Lipschitz Condition__ in the variable $y$ on 
# the set $D \subset R^2$ if a constant $L>0$ exist with the property that
# \begin{equation}
# |f(t,y_1)-f(t,y_2)| < L|y_1-y_2| 
# \end{equation}
# whenever $(t,y_1),(t,y_2) \in D$.  The constant L is call the Lipschitz Condition
# of $f$.
# 
# __Theorem__
# Suppose $f(t,y)$ is defined on a convex set $D \subset R^2$. If a constant
# $L>0$ exists with
# \begin{equation}
# \left|\frac{\partial f(t,y)}{\partial y}\right|\leq L,
# \end{equation}
# then $f$ satisfies a Lipschitz Condition an $D$ in the variable $y$ with
# Lipschitz constant L.
# 

# ### Global truncation error for the population equation
# For the population equation specific values $L$ and $M$ can be calculated.
# 
# In this case $f(t,y)=\epsilon y$ is continuous and satisfies a Lipschitz Condition with constant
# \begin{equation} \left|\frac{\partial f(t,y)}{\partial y}\right|\leq L, \end{equation}
# \begin{equation} \left|\frac{\partial \epsilon y}{\partial y}\right|\leq \epsilon=0.5=L, \end{equation}
# 
# on $D=\{(t,y)|0\leq t \leq 2, 10 < y < 30 \}$ and that a constant $M$
# exists with the property that 
# \begin{equation} |y^{''}(t)|\leq M. \end{equation}
# \begin{equation} |y^{''}(t)|=2.5e^{0.5\times 2} \leq 2.5 e=6.8. \end{equation}
# 
# __Specific Theorem Global Error__
# 
# Let $y(t)$ denote the unique solution of the Initial Value Problem
# \begin{equation} y^{'}=0.5 y \ \ \ 0\leq t \leq 10 \ \ \ y(0)=10 \end{equation}
# and $w_0,w_1,...,w_N$ be the approx generated by the Euler method for some
# positive integer N.  Then for $i=0,1,...,N$
# \begin{equation} |y(t_i)-w_i| \leq \frac{6.8 h}{2\times 0.5}|e^{0.5(t_i-0)}-1| \end{equation}
# 
# The figure below shows the exact error $y-w$ in red triangles and the upper global error in black x's.

# In[7]:


fig = plt.figure(figsize=(10,4))
plt.plot(time,np.abs(w-y),'^:'
         ,color='red',label='Error |y-w|')
plt.plot(time,0.1*6.8*(np.exp(0.5*time)-1),'x:'
         ,color='black',label='Upper Global Truncation')
plt.xlim((0,2))
plt.xlabel('time')
plt.legend(loc='best')
plt.title('Global Truncation Error')
plt.plot();


# ### Table
# The table below shows the iteration $i$, the discrete time point t[i], the Euler approximation w[i] of the solution $y$, the exact error $|y-w|$ and the upper limit of the global error for the linear population equation.

# In[8]:



d = {'time t_i': time[0:10],    'Euler (w_i) ':w[0:10],'Exact (y)':y[0:10],'Exact Error (|y_i-w_i|)':np.round(np.abs(w[0:10]-y[0:10]),10),r'Global Error ':np.round(0.1*6.8*(np.exp(0.5*time[0:10])-1),20)}
df = pd.DataFrame(data=d)
df


# In[ ]:




