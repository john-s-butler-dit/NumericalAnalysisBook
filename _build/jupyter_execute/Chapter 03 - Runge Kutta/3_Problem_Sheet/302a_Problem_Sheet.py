#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/john-s-butler-dit/Numerical-Analysis-Python/blob/master/Chapter%2003%20-%20Runge%20Kutta/Supplementary/01_RK%20Mid%20point%20Example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Problem Sheet 3 Question 1
# 
# The general form of the population growth differential equation
# \begin{equation} y^{'}=t-y, \ \ (0 \leq t \leq 4) \end{equation}
# with the initial condition
# \begin{equation}y(0)=1,\end{equation}
# Has the exact soulation. \begin{equation} y= 2e^{-t}+t-1\end{equation}
# 
# 

# #### Setting up Libraries

# In[1]:


import numpy as np
import math 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt # side-stepping mpl backend
import matplotlib.gridspec as gridspec # subplots
import warnings

warnings.filterwarnings("ignore")


# ## Defining the function
# \begin{equation}f(t,y)=t-y$$ 

# In[2]:


def myfun_ty(t,y):
    return t-y


# ## Initial Setup
# Defining the step size $h$ from the interval range $a\leq t \leq b$ and number of steps $N$
# \begin{equation}h=\frac{b-a}{h}.\end{equation}
# This gives the discrete time steps,
# \begin{equation}t_{i}=t_0+ih,\end{equation}
# where $t_0=a$.

# In[3]:


# Start and end of interval
b=4
a=0
# Step size
N=8
h=(b-a)/(N)
t=np.arange(a,b+h,h)


# ## Setting up the initial conditions of the equation
# \begin{equation}w_0=IC\end{equation}
# 

# In[4]:


# Initial Condition
IC=1
w=np.zeros(N+1)
y=(IC+1)*np.exp(-t)+t-1#np.zeros(N+1)
w[0]=IC


# ## 2nd Order Runge Kutta (Mid-method)
# \begin{equation}k_1=f(t,y),\end{equation}
# \begin{equation}k_2=f(t+\frac{h}{2},y+\frac{h}{2}k_2),\end{equation}
# \begin{equation}w_{i+1}=w_{i}+h(k_2).\end{equation}

# In[5]:


for k in range (0,N):
    k1=myfun_ty(t[k],w[k])
    k2=myfun_ty(t[k]+h/2,w[k]+h/2*k1)
    w[k+1]=w[k]+h*(k2)


# ## Plotting Results

# In[6]:


fig = plt.figure(figsize=(10,4))
# --- left hand plot
ax = fig.add_subplot(1,3,1)
plt.plot(t,w, '--',color='blue')
#ax.legend(loc='best')
plt.title('Numerical Solution h=%s'%(h))

ax = fig.add_subplot(1,3,2)
plt.plot(t,y,color='black')
plt.title('Exact Solution ')

ax = fig.add_subplot(1,3,3)
plt.plot(t,y-w, 'o',color='red')
plt.title('Error')
# --- title, explanatory text and save
fig.suptitle(r"$y'=t-y,   y(0)=%s$"%(IC), fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.85)    


# In[ ]:





# In[ ]:




