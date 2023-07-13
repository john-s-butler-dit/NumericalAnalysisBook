#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/john-s-butler-dit/Numerical-Analysis-Python/blob/master/Chapter%2002%20-%20Higher%20Order%20Methods/201_3rd%20Order%20Taylor_Population_growth.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Taylor Method
# 
# This notebook implements the 3rd order Taylor method for three different population intial value problems.
# 
# # 3rd Order Taylor
# The general 3rd Order Taylor method for to the first order differential equation
# \begin{equation} y^{'} = f(t,y), \end{equation}
# numerical approximates $y$ the at time point $t_i$ as $w_i$
# with the  formula:
# \begin{equation}
#  w_{i+1}=w_i+h\big[f(t_i,w_i)+\frac{h}{2}f'(t_i,w_i)+\frac{h^2}{6}f''(t_i,w_i)\big],\end{equation}
# where $h$ is the stepsize.
# for $i=0,...,N-1$.
# With the local truncation error of 
# \begin{equation}
# \tau=\frac{h^3}{24}y^{''''}(\xi_i),\end{equation}
# where $\xi \in [t_i,t_{i+1}]$.
# To illustrate the method we will apply it to three intial value problems:
# ## 1. Linear 
# Consider the linear population Differential Equation
# \begin{equation}
#  y^{'}=0.1y, \ \ (2000 \leq t \leq 2020), \end{equation}
# with the initial condition,
# \begin{equation}
# y(2000)=6.\end{equation}
# 
# ## 2. Non-Linear Population Equation 
# Consider the non-linear population Differential Equation
# \begin{equation}
#  y^{'}=0.2y-0.01y^2, \ \ (2000 \leq t \leq 2020), \end{equation}
# with the initial condition,
# \begin{equation}
# y(2000)=6.
# \end{equation}
# 
# ## 3. Non-Linear Population Equation with an oscillation 
# Consider the non-linear population Differential Equation with an oscillation 
# \begin{equation}
#  y^{'}=0.2y-0.01y^2+\sin(2\pi t), \ \ (2000 \leq t \leq 2020), 
#  \end{equation}
# with the initial condition,
# \begin{equation}
# y(2000)=6.\end{equation}

# ## Read in Libraries 

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
# \begin{equation} h=\frac{b-a}{N}.\end{equation}
# Here the interval is $2000\leq t \leq 2020,$ 
# \begin{equation} h=\frac{2020-2000}{200}=0.1.\end{equation}
# This gives the 201 discrete points:
# \begin{equation} t_0=2000, \ t_1=2000.1, \ ... t_{200}=2020. \end{equation}
# This is generalised to 
# \begin{equation} t_i=2000+i0.1, \ \ \ i=0,1,...,200.\end{equation}
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
# \begin{equation} y^{'}=0.1y, \ \ (2000 \leq t \leq 2020), \end{equation}
# with the initial condition,
# \begin{equation}y(2000)=6.\end{equation}
# has a known exact (analytic) solution
# \begin{equation} y=6e^{0.1(t-2000)}. \end{equation}
# 
# ## Specific 3rd Order Taylor for the Linear Population Equation
# To write the specific 3rd Order Taylor difference equation for the intial value problem we need to calculate the first derivative of 
# \begin{equation}f(t,y)=0.1y,\end{equation}

# In[3]:


def linfun(t,w):
    ftw=0.1*w
    return ftw


# with respect to $t$,
# \begin{equation} f'(t,y)=0.1y'=0.1(0.1y)=0.01y, \end{equation}

# In[4]:


def linfun_d(t,w):
    ftw=0.01*w
    return ftw


# and the second derivative of $f$ with respect to $t$,
# \begin{equation}f'(t,y)=0.01y'=0.01(0.1y)=0.001y.\end{equation}

# In[5]:


def linfun_dd(t,w):
    ftw=0.001*w
    return ftw


# ### Linear Population 3rd Order Taylor Difference equation
# Substituting the derivatives of the linear population equation into the 3rd order Taylor equation gives the difference equation 
# \begin{equation} w_{i+1}= w_{i} + h\big[0.1 w_i+\frac{h}{2}(0.01 w_i)+\frac{h^2}{6}(0.001w_i)\big],\end{equation}
# for $i=0,...,199$, where $w_i$ is the numerical approximation of $y$ at time $t_i$, with the initial condition
# \begin{equation}w_0=6.\end{equation}

# ## Method

# In[6]:


w=np.zeros(N+1)
w[0]=6.0
for i in range (0,N):
    w[i+1]=w[i]+h*(linfun(t[i],w[i])+h/2*linfun_d(t[i],w[i])+h*h/6*linfun_dd(t[i],w[i]))


# ## Results
# The plot below shows the exact solution, $y$ (squares) and the 3rd order numberical approximation, $w$ (circles) for the linear population equation:

# In[7]:


y=6*np.exp(0.1*(t-2000))
fig = plt.figure(figsize=(8,4))
plt.plot(t,w,'o:',color='purple',label='Taylor')
plt.plot(t,y,'s:',color='black',label='Exact')
plt.legend(loc='best')
plt.show()


# ## Table
# The table below shows the time, the 3rd order numerical approximation, $w$,  the exact solution, $y$, and the exact error $|y(t_i)-w_i|$ for the linear population equation:

# In[8]:



d = {'time t_i': t[0:10],    'Taylor ':w[0:10],'Exact (y)':y[0:10],'Exact Error':np.abs(np.round(y[0:10]-w[0:10],10))}
df = pd.DataFrame(data=d)
df


# ## 2. Non-Linear Population Equation 
# \begin{equation} y^{'}=0.2y-0.01y^2, \ \ (2000 \leq t \leq 2020), \end{equation}
# with the initial condition,
# \begin{equation}y(2000)=6.\end{equation}
# ## Specific 3rd Order Taylor for the Non-Linear Population Equation
# To write the specific 3rd Order Taylor difference equation for the intial value problem we need to calculate the first derivative of 
# \begin{equation}f(t,y)=0.2y-0.01y^2,\end{equation}

# In[9]:


def nonlinfun(t,w):
    ftw=0.2*w-0.01*w*w
    return ftw


# with respect to $t$
# \begin{equation} f'(t,y)=0.2y'-0.02y'y=0.2(0.2y-0.01y^2)-0.02(0.2y-0.01y^2)y,\end{equation}
# \begin{equation}=(0.2-0.02y)(0.2y-0.01y^2)=(0.2-0.02y)f(t,y), \end{equation}

# In[10]:


def nonlinfun_d(t,w):
    ftw=0.2*nonlinfun(t,w)-0.02*nonlinfun(t,w)*w
    return ftw


# and the second derivative with respect to $t$
# \begin{equation} f''(t,y)=-0.02y'(0.2y-0.01y^2)+(0.2-0.02y)(0.2y'-0.02y'y),\end{equation}
# \begin{equation}=-0.02(0.2y-0.01y^2)^2+(0.2-0.02y)^2(0.2y-0.01y^2), \end{equation}

# In[11]:


def nonlinfun_dd(t,w):
    ftw=-0.02*nonlinfun(t,w)*nonlinfun(t,w)+(0.2-0.02*w)*nonlinfun_d(t,w)
    return ftw


# ###  Non-Linear Population 3rd Order Taylor Difference equation
# Substituting the derivatives of the non-linear population equation into the 3rd order Taylor equation gives the difference equation 
# \begin{equation} w_{i+1}= w_{i} + h\big[\big(0.2w_i-0.01w_i^2\big)+\frac{h}{2}\big((0.2-0.02w_i)(0.2w_i-0.01w_i^2)\big)+\end{equation}
# \begin{equation}
# \frac{h^2}{6}\big(-0.02(0.2w_i-0.01w_i^2)^2+(0.2-0.02w_i)^2(0.2w_i-0.01w_i^2)\big)\big], \end{equation}
# for $i=0,...,199$, where $w_i$ is the numerical approximation of $y$ at time $t_i$, with the initial condition
# $$w_0=6.$$

# In[12]:


w=np.zeros(N+1)
w[0]=6.0
for i in range (0,N):
    w[i+1]=w[i]+h*(nonlinfun(t[i],w[i])+h/2*nonlinfun_d(t[i],w[i])+h*h/6*nonlinfun_dd(t[i],w[i]))


# ## Results
# The plot below shows the 3rd order numerical approximation, $w$ (circles) for the non-linear population equation:

# In[13]:


fig = plt.figure(figsize=(8,4))
plt.plot(t,w,'o:',color='purple',label='Taylor')
plt.legend(loc='best')
plt.show()


# ## Table
# The table below shows the time and the 3rd order numerical approximation, $w$,  for the non-linear population equation:

# In[14]:


d = {'time t_i': t[0:10],    'Taylor ':w[0:10]}
df = pd.DataFrame(data=d)
df


# ## 3. Non-Linear Population Equation with an oscilation 
# $$ y^{'}=0.2y-0.01y^2+\sin(2\pi t), \ \ (2000 \leq t \leq 2020), $$
# with the initial condition,
# $$y(2000)=6.$$
# 
# ## Specific 3rd Order Taylor for the Non-Linear Population Equation with an oscilation
# To write the specific 3rd Order Taylor difference equation for the intial value problem we need calculate the first derivative of 
# $$f(t,y)=0.2y-0.01y^2+\sin(2\pi t),$$

# In[15]:


def nonlin_oscfun(t,w):
    ftw=0.2*w-0.01*w*w+np.sin(2*np.math.pi*t)
    return ftw


# with respect to $t$
# $$ f'(t,y)=0.2y'-0.02y'y+2\pi\cos(2\pi t)$$
# $$=(0.2-0.02y)\big(0.2y-0.01y^2+\sin(2\pi t)\big)+2\pi\cos(2\pi t) $$

# In[16]:


def nonlin_oscfun_d(t,w):
    ftw=0.2*nonlinfun(t,w)-0.02*nonlinfun(t,w)*w+2*np.math.pi*np.cos(2*np.math.pi*t)
    return ftw


# and the second derivative with respect to $t$
# $$ f''(t,y)=-0.02y'(0.2y-0.01y^2+\sin(2\pi t))$$
# $$+(0.2-0.02y)\big(0.2y'-0.02y'y+2\pi\cos(2\pi t)\big)-(2\pi)^2\sin(2\pi t)$$
# $$=-0.02(0.2y-0.01y^2+2\pi\sin(2\pi t))^2$$
# $$+(0.2-0.02y)\big((0.2-0.02y)(0.2y-0.01y^2+\sin(2\pi t))+ 2\pi\cos(2\pi t) \big)-(2\pi)^2\sin(2\pi t) $$

# In[17]:


def nonlin_oscfun_dd(t,w):
    ftw=-0.02*nonlin_oscfun(t,w)*nonlin_oscfun(t,w)+(0.2-0.02*w)*((0.2-0.02*w)*nonlin_oscfun(t,w)+2*np.math.pi*np.cos(2*np.math.pi*t))#-2*np.math.pi*2*np.math.pi*np.sin(2*np.math.pi*t)
    return ftw


# ###  Non-Linear Population with oscilation 3rd Order Taylor Difference equation
# Substituting the derivatives of the non-linear population equation with oscilation into the 3rd order Taylor equation gives the difference equation 
# $$ w_{i+1}= w_{i} + h\big[\big(0.2w_i-0.01w_i^2+\sin(2\pi t_i)\big)$$
# $$+\frac{h}{2}\big((0.2-0.02w_i)\big(0.2w_i-0.01w_i^2+\sin(2\pi t_i)\big)+2\pi\cos(2\pi t_i)\big)+$$
# $$\frac{h^2}{6}\big(-0.02(0.2w_i-0.01w_i^2+2\pi\sin(2\pi t_i))^2$$ 
# $$+(0.2-0.02w_i)\big((0.2-0.02w_i)[0.2w_i-0.01w_i^2+\sin(2\pi t_i)]$$
# $$+ 2\pi\cos(2\pi t_i) \big)-(2\pi)^2\sin(2\pi t_i) \big)\big], $$
# for $i=0,...,199$, where $w_i$ is the numerical approximation of $y$ at time $t_i$, with the initial condition
# $$w_0=6.$$

# In[18]:


w=np.zeros(N+1)
w[0]=6.0
for i in range (0,N):
    w[i+1]=w[i]+h*(nonlin_oscfun(t[i],w[i])+h/2*nonlin_oscfun_d(t[i],w[i])+h*h/6*nonlin_oscfun_dd(t[i],w[i]))


# ## Results
# The plot below shows the 3rd order numerical approximation, $w$ (circles) for the non-linear population equation:

# In[19]:


fig = plt.figure(figsize=(8,4))
plt.plot(t,w,'o:',color='purple',label='Taylor')
plt.legend(loc='best')
plt.show()


# ## Table
# The table below shows the time and the 3rd order numerical approximation, $w$,  for the non-linear population equation:

# In[20]:


d = {'time t_i': t[0:10],    'Taylor ':w[0:10]}
df = pd.DataFrame(data=d)
df


# In[20]:




