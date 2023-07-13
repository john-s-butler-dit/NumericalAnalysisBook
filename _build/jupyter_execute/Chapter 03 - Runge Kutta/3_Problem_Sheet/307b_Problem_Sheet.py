#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/john-s-butler-dit/Numerical-Analysis-Python/blob/master/Chapter%2003%20-%20Runge%20Kutta/Supplementary/02_RK%20Mid%20point%20Example%20-%20Review%20Question%207b.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ## Problem Sheet 3 Question 7b
# 
# The general form of the population growth differential equation
# \begin{equation} y^{'}-y+x=0, \ \ (0 \leq x \leq 1) \end{equation}
# with the initial condition
# \begin{equation}y(0)=0\end{equation}
# For h=0.2.
# # Midpoint method Solution
# \begin{equation}
# \frac{w_{i+1}-w_i}{h}=f(x_i+\frac{h}{2},w_i+\frac{h}{2}f(x_i,w_i))
# \end{equation}
# Rearranging 
# \begin{equation}
# w_{i+1}=w_i+hf(x_i+\frac{h}{2},w_i+\frac{h}{2}f(x_i,w_i))
# \end{equation}
# \begin{equation}
# w_{i+1}=w_i+h(k_2)
# \end{equation}
# \begin{equation}
# k_1=w_i-x_i+2
# \end{equation}
# \begin{equation}
# k_2=w_i+\frac{h}{2}k_1-(x_i+\frac{h}{2})+2)
# \end{equation}

# In[1]:


import numpy as np
import math 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt # side-stepping mpl backend
import matplotlib.gridspec as gridspec # subplots
import warnings
warnings.filterwarnings("ignore")


# In[2]:


def myfun_xy(x,y):
    return y-x+2

#PLOTS
def Midpoint_Question2(N,IC):

    x_start=0
    INTITIAL_CONDITION=IC
    h=0.2
    N=N+1
    x=np.zeros(N)
    w=np.zeros(N)
    k_mat=np.zeros((2,N))
    k=0
    w[0]=INTITIAL_CONDITION
    x[0]=x_start
    
    for k in range (0,N-1):
        k_mat[0,k]=myfun_xy(x[k],w[k])
        k_mat[1,k]=myfun_xy(x[k]+h/2,w[k]+h/2*k_mat[0,k])
        w[k+1]=w[k]+h*(k_mat[1,k])
        x[k+1]=x[k]+h


    fig = plt.figure(figsize=(10,4))
    plt.plot(x,w,'-.o',color='blue')
    plt.title('Numerical Solution h=%s'%(h))

    # --- title, explanatory text and save
    fig.suptitle(r"$y'=y-x+2$", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)    
    print('x')
    print(x)
    print('k1')
    print(k_mat[0,:])
    print('k2')
    print(k_mat[1,:])
    print('w')
    print(w)


# In[3]:


# Midpoint_Question2(N,IC)
Midpoint_Question2(5,1)


# In[3]:





# In[3]:




