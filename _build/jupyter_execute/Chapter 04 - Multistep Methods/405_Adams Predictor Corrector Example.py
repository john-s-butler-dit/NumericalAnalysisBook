#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/john-s-butler-dit/Numerical-Analysis-Python/blob/master/Chapter%2004%20-%20Multistep%20Methods/405_Adams%20Predictor%20Corrector%20Example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Adams Predictor Corrector
# The Adams Predictor corrector method is used to side step the issue of the Adams-Moulton method being implicit. It can also be used to estimate error when the solution is unknown.

# In[1]:


from IPython.display import HTML
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/RsceVcOLAC4" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')


# ## Intial Value Problem
# 
# The differential equation
# \begin{equation}y^{'}=t-y, \ \ (0 \leq t \leq 2) \end{equation}
# with the initial condition
# \begin{equation}y(0)=1\end{equation}
# will be used to illustrate the method.

# ## Python Libraries

# In[2]:


import numpy as np
import math 
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt # side-stepping mpl backend
import matplotlib.gridspec as gridspec # subplots
import warnings

warnings.filterwarnings("ignore")


# ### Defining the function
# \begin{equation} f(t,y)=t-y.$$

# In[3]:


def myfun_ty(t,y):
    return t-y


# ## Discrete Interval
# Defining the step size  $h$  from the interval range  $a\leq t \leq b$  and number of steps  $N$ 
# \begin{equation}h=\frac{b−a}{N}.\end{equation}
#  
# This gives the discrete time steps,
# \begin{equation}t_i=t_0+ih,\end{equation}
# where  $t0=a.$
# 
# Here the interval is  $0≤t≤2$  and number of step  4 
# \begin{equation}h=\frac{2−0}{4}=0.5.\end{equation}
#  
# This gives the discrete time steps,
# \begin{equation}t_i=0+i0.5,\end{equation}
# for  $i=0,1,⋯,4.$

# In[4]:


# Start and end of interval
b=2
a=0
# Step size
N=4
h=(b-a)/(N)
t=np.arange(a,b+h,h)
fig = plt.figure(figsize=(10,4))
plt.plot(t,0*t,'o:',color='red')
plt.xlim((0,2))
plt.title('Illustration of discrete time points for h=%s'%(h))


# ## Exact Solution
# The initial value problem has the exact solution
# \begin{equation}y(t)=2e^{-t}+t-1.\end{equation}
# The figure below plots the exact solution.

# In[5]:


IC=1 # Intial condtion
y=(IC+1)*np.exp(-t)+t-1
fig = plt.figure(figsize=(6,4))
plt.plot(t,y,'o-',color='black')
plt.title('Exact Solution ')
plt.xlabel('time')


# In[6]:


# Initial Condition
w=np.zeros(N+1)
#np.zeros(N+1)
w[0]=IC


# 
# 
# ## 2-step Adams Bashforth (Predictor)
# 
# The 2-step Adams Bashforth difference equation is
# \begin{equation}w^{0}_{i+1} = w_{i} + \frac{h}{2}(3f(t_i,w_i)-f(t_{i-1},w_{i-1})) \end{equation}
# \begin{equation}w^{0}_{i+1} = w_{i} + \frac{h}{2}(3(t_i-w_i)-(t_{i-1}-w_{i-1}))\end{equation}
# ## 1-step Adams Moulton (Corrector)
# \begin{equation}w^{1}_{i+1} = w_{i} + \frac{h}{2}(f(t_{i+1},w^{0}_{i+1})+f(t_{i},w_{i})) \end{equation}
# 
# \begin{equation} w^{1}_{i+1} = w_{i} + \frac{h}{2}((t_{i+1}-w^0_{i+1})+(t_{i}-w_{i})) \end{equation}
# 
# 
# For $i=0$ the system of difference equation is:
# \begin{equation}w^{0}_{1} = w_{0} + \frac{h}{2}(3(t_0-w_0)-(t_{-1}-w_{-1})) \end{equation}
# \begin{equation}w_{1} = w_{0} + \frac{h}{2}((t_{1}-w^{0}_{1})+(t_{0}-w_{0})). \end{equation}
# this is not solvable as <font color='red'> $w_{-1}$ </font> is unknown.
# 
# For $i=1$ the difference equation is:
# \begin{equation}w^{0}_{2} = w_{1} + \frac{h}{2}(3(t_1-w_1)-(t_{0}-w_{0})) \end{equation}
# \begin{equation}w_{2} = w_{1} + \frac{h}{2}((t_{2}-w^{0}_{2})+(t_{1}-w_{1})). \end{equation}
# this is not solvable as <font color='red'> $w_{1}$  </font> is unknown. $w_1$ can be  approximated using a one step method. Here, as the exact solution is known,
# \begin{equation}w_1=2e^{-t_1}+t_1-1.$$
# 

# In[7]:


### Initial conditions
w=np.zeros(len(t))
w0=np.zeros(len(t))
w[0]=IC
w[1]=y[1]
w0[0]=IC
w0[1]=y[1]


# ### Loop

# In[8]:


for k in range (1,N):
    w0[k+1]=(w[k]+h/2.0*(3*myfun_ty(t[k],w[k])-myfun_ty(t[k-1],w[k-1])))   
    w[k+1]=(w[k]+h/2.0*(myfun_ty(t[k+1],w0[k+1])+myfun_ty(t[k],w[k])))   


# ### Plotting solution

# In[9]:


def plotting(t,w,w0,y):
    fig = plt.figure(figsize=(10,4))
    plt.plot(t,y, 'o-',color='black',label='Exact')
    plt.plot(t,w0,'v:',color='blue',label='Adams-Bashforth Predictor')
    plt.plot(t,w,'^:',color='red',label='Adams-Moulton Corrector')
    plt.xlabel('time')
    plt.legend()
    plt.show 


# The plot below shows the exact solution (black) and the Adams Predictor Corrector approximation (red) of the intial value problem

# In[10]:


plotting(t,w,w0,y)


# ## Local Error 
# The error for the 1 step Adams Moulton is:
# \begin{equation}\frac{y_{n+1}-y_{n}}{h}=\frac{1}{2}[f(t_{n+1},w_{n+1})+f(t_{n-1},w_{n-1})] +\frac{-h^2}{12}y^{(3)}(\eta),\end{equation}
# where $\eta \in [t_{n-1},t_{n+1}]$.
# The error for the 2 step Adams Bashforth is:
# \begin{equation}\frac{y_{n+1}-y_{n}}{h}=\frac{1}{2}[3f(t_{n},w_{n})-f(t_{n},w_{n})] +\frac{5h^2}{12}y^{(3)}(\xi),\end{equation}
# where $\xi \in [t_{n-1},t_{n+1}]$.
# 
# Rearranging the equations gives 
# \begin{equation}\frac{w_{n+1}-w_{n+1}^0}{h}= h^2[\frac{y^{(3)}(\eta)}{12}+\frac{5y^{(3)}(\xi)}{12}]\approx h^2[\frac{6}{12}y^{(3)}(\xi)].\end{equation}
# Making the assumption that 
# \begin{equation}y^{(3)}(\xi)\approx \frac{12}{6h^2}\frac{w_{n+1}-w_{n+1}^0}{h},\end{equation}
# \begin{equation}y^{(3)}(\xi)\approx \frac{12}{6h^2}\frac{w_{n+1}-w_{n+1}^0}{h}.\end{equation}

# In[11]:



d = {'time t_i': t, 'Adams Predictor w0': w0,
     'Adams Corrector':w,'Exact (y)':y,'|w-y|':np.round(np.abs(y-w),5),'|w0-w|':np.round(np.abs(w0-w),5),'estimate LTE':np.round(1/(6*h)*abs(w0-w),5)}
df = pd.DataFrame(data=d)
df


# In[11]:




