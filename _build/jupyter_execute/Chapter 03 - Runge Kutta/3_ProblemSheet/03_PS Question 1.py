#!/usr/bin/env python
# coding: utf-8

# # Application of 2nd order Runge Kutta to Populations Equations
# 
# This notebook implements the 2nd Order Runge Kutta method for three different population intial value problems.
# 
# # 2nd Order Runge Kutta
# The general 2nd Order Runge Kutta method for to the first order differential equation
# $$ y^{'} = f(t,y) $$
# numerical approximates $y$ the at time point $t_i$ as $w_i$
# with the  formula:
# $$ w_{i+1}=w_i+\frac{h}{2}\big[k_1+k_2],$$
# for $i=0,...,N-1$, where 
# $$k_1=f(t_i,w_i)$$
# and
# $$k_2=f(t_i+h,w_i+hk_1)$$
# and $h$ is the stepsize.
# 
# To illustrate the method we will apply it to three intial value problems:
# ## 1. Linear 
# Consider the linear population Differential Equation
# $$ y^{'}=0.1y, \ \ (2000 \leq t \leq 2020), $$
# with the initial condition,
# $$y(2000)=6.$$
# 
# ## 2. Non-Linear Population Equation 
# Consider the non-linear population Differential Equation
# $$ y^{'}=0.2y-0.01y^2, \ \ (2000 \leq t \leq 2020), $$
# with the initial condition,
# $$y(2000)=6.$$
# 
# ## 3. Non-Linear Population Equation with an oscillation 
# Consider the non-linear population Differential Equation with an oscillation 
# $$ y^{'}=0.2y-0.01y^2+\sin(2\pi t), \ \ (2000 \leq t \leq 2020), $$
# with the initial condition,
# $$y(2000)=6.$$

# #### Setting up Libraries

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


# ## Discrete Interval
# The continuous time $a\leq t \leq b $ is discretised into $N$ points seperated by a constant stepsize
# $$ h=\frac{b-a}{N}.$$
# Here the interval is $2000\leq t \leq 2020,$ 
# $$ h=\frac{2020-2000}{200}=0.1.$$
# This gives the 201 discrete points:
# $$ t_0=2000, \ t_1=2000.1, \ ... t_{200}=2020. $$
# This is generalised to 
# $$ t_i=2000+i0.1, \ \ \ i=0,1,...,200.$$
# The plot below shows the discrete time steps:

# In[2]:


N=200
t_end=2020.0
t_start=2000.0
h=((t_end-t_start)/N)
t=np.arange(t_start,t_end+h/2,h)
fig = plt.figure(figsize=(10,4))
plt.plot(t,0*t,'o:',color='red')
plt.title('Illustration of discrete time points for h=%s'%(h))
plt.show()


# # 1. Linear Population Equation
# ## Exact Solution 
# The linear population equation
# $$ y^{'}=0.1y, \ \ (2000 \leq t \leq 2020), $$
# with the initial condition,
# $$y(2000)=6.$$
# has a known exact (analytic) solution
# $$ y=6e^{0.1(t-2000)}. $$
# 
# ## Specific 2nd Order Runge Kutta 
# To write the specific 2nd Order Runge Kutta method for the linear population equation we need 
# $$f(t,y)=0.1y$$

# In[3]:


def linfun(t,w):
    ftw=0.1*w
    return ftw


# this gives
# $$k_1=f(t_i,w_i)=0.lw_i,$$
# $$k_2=f(t_i+h,w_i+hk_1)=0.1(w_i+hk_1),$$
# and the difference equation
# $$w_{i+1}=w_{i}+\frac{h}{2}(k_1+k_2)$$
# for $i=0,...,199$, where $w_i$ is the numerical approximation of $y$ at time $t_i$, with step size $h$ and the initial condition
# $$w_0=6.$$

# In[4]:


w=np.zeros(N+1)
w[0]=6.0
## 2nd Order Runge Kutta
for k in range (0,N):
    k1=linfun(t[k],w[k])
    k2=linfun(t[k]+h,w[k]+h*k1)
    w[k+1]=w[k]+h/2*(k1+k2)


# ## Plotting Results

# In[5]:


y=6*np.exp(0.1*(t-2000))
fig = plt.figure(figsize=(8,4))
plt.plot(t,w,'o:',color='purple',label='Runge Kutta')
plt.plot(t,y,'s:',color='black',label='Exact')
plt.legend(loc='best')
plt.show()


# ## Table
# The table below shows the time, the Runge Kutta numerical approximation, $w$,  the exact solution, $y$, and the exact error $|y(t_i)-w_i|$ for the linear population equation:

# In[6]:



d = {'time t_i': t[0:10],    'Runge Kutta':w[0:10],'Exact (y)':y[0:10],'Exact Error':np.abs(np.round(y[0:10]-w[0:10],10))}
df = pd.DataFrame(data=d)
df


# ## 2. Non-Linear Population Equation 
# $$ y^{'}=0.2y-0.01y^2, \ \ (2000 \leq t \leq 2020), $$
# with the initial condition,
# $$y(2000)=6.$$
# ## Specific 2nd Order Runge Kutta for the Non-Linear Population Equation
# To write the specific 2nd Order Runge Kutta method we need
# $$f(t,y)=0.2y-0.01y^2,$$
# this gives
# $$k_1=f(t_i,w_i)=0.2w_i-0.01w_i^2,$$
# $$k_2=f(t_i+h,w_i+hk_1)=0.2(w_i+hk_1)-0.01(w_i+hk_1)^2,$$
# and the difference equation
# $$w_{i+1}=w_{i}+\frac{h}{2}(k_1+k_2)$$
# for $i=0,...,199$, where $w_i$ is the numerical approximation of $y$ at time $t_i$, with step size $h$ and the initial condition
# $$w_0=6.$$

# In[7]:


def nonlinfun(t,w):
    ftw=0.2*w-0.01*w*w
    return ftw


# In[8]:


w=np.zeros(N+1)
w[0]=6.0
## 2nd Order Runge Kutta
for k in range (0,N):
    k1=nonlinfun(t[k],w[k])
    k2=nonlinfun(t[k]+h,w[k]+h*k1)
    w[k+1]=w[k]+h/2*(k1+k2)


# ## Results
# The plot below shows the Runge Kutta numerical approximation, $w$ (circles) for the non-linear population equation:

# In[9]:


fig = plt.figure(figsize=(8,4))
plt.plot(t,w,'o:',color='purple',label='Runge Kutta')
plt.legend(loc='best')
plt.show()


# ## Table
# The table below shows the time and the Runge Kutta numerical approximation, $w$,  for the non-linear population equation:

# In[10]:


d = {'time t_i': t[0:10], 
     'Runge Kutta':w[0:10]}
df = pd.DataFrame(data=d)
df


# ## 3. Non-Linear Population Equation with an oscilation 
# $$ y^{'}=0.2y-0.01y^2+\sin(2\pi t), \ \ (2000 \leq t \leq 2020), $$
# with the initial condition,
# $$y(2000)=6.$$
# 
# ## Specific 2nd Order Runge Kutta for the Non-Linear Population Equation with an oscilation
# To write the specific 2nd Order Runge Kutta difference equation for the intial value problem we need 
# $$f(t,y)=0.2y-0.01y^2+\sin(2\pi t),$$
# which gives
# $$k_1=f(t_i,w_i)=0.2w_i-0.01w_i^2+\sin(2\pi t_i),$$
# $$k_2=f(t_i+h,w_i+hk_1)=0.2(w_i+hk_1)-0.01(w_i+hk_1)^2+\sin(2\pi (t_i+h)),$$
# and the difference equation
# $$w_{i+1}=w_{i}+\frac{h}{2}(k_1+k_2)$$
# for $i=0,...,199$, where $w_i$ is the numerical approximation of $y$ at time $t_i$, with step size $h$ and the initial condition
# $$w_0=6.$$

# In[11]:


def nonlin_oscfun(t,w):
    ftw=0.2*w-0.01*w*w+np.sin(2*np.math.pi*t)
    return ftw


# In[12]:


w=np.zeros(N+1)
w[0]=6.0
## 2nd Order Runge Kutta
for k in range (0,N):
    k1=nonlin_oscfun(t[k],w[k])
    k2=nonlin_oscfun(t[k]+h,w[k]+h*k1)
    w[k+1]=w[k]+h/2*(k1+k2)


# ## Results
# The plot below shows the 2nd order Runge Kutta numerical approximation, $w$ (circles) for the non-linear population equation:

# In[13]:


fig = plt.figure(figsize=(8,4))
plt.plot(t,w,'o:',color='purple',label='Taylor')
plt.legend(loc='best')
plt.show()


# ## Table
# The table below shows the time and the 2nd order Runge Kutta numerical approximation, $w$,  for the non-linear population equation:

# In[14]:


d = {'time t_i': t[0:10], 
     'Runge Kutta':w[0:10]}
df = pd.DataFrame(data=d)
df


# In[ ]:




