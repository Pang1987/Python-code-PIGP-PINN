


##################################################################

### Python code, written by Guofei Pang (guofei_pang@brown.edu), 
### for the book chapter ''Physics-informed learning machines for PDEs: 
### Guassian processes versus neural networks'' 
###   authored by Guofei Pang and George Em Karniadakis

##################################################################

### Code for Section 4 of the book chapter 
### We solve the 1D diffusion equation
# \frac{\partial u(x,t)}{\partial t} = c \frac{\partial^2 u(x,t)}{\partial x^2} + s(x,t), x,t \in (0,1)
# with the boundary conditions u(0,t) = u(1,t) = 0 and 
# initial conditions u(x,0) = \sin(2\pi x).

### In the code, we solve the forward problem using the continuous time GP
### and continuous time NN.

#####################################################################



import numpy as np   
import matplotlib.pyplot as plt
from SALib.sample import sobol_sequence
import models   ### a collectio of machine learning models for GP and NN
import time

def u(X):   ### fabricated solution
   x = X[:,0:1]
   t = X[:,1:2]
   return np.sin(2.*np.pi*x)*np.exp(-t)


def U(X, c): ### source term computed using the fabricated solution
    x = X[:,0:1]
    t = X[:,1:2]
    return -np.sin(2*np.pi*x)*np.exp(-t)+c*4*np.pi**2*np.sin(2*np.pi*x)*np.exp(-t)

N_U = 30   # number of training inputs for \Omega_1 (see the book chapter for notations)
c = 0.1    # diffusivity
xU = sobol_sequence.sample(N_U+1,2)[1:,:]  # training inputs for \Omega 1


#noise = 0.0    # noise-free  
noise = 0.05   # 5% noise

np.random.seed(seed=1234)  # fix the realization of Guassian white noise for sake of reproducing
yU = U(xU, c)   # training outputs for \Omega_1 or observation y_1
yU = yU + noise * np.std(yU) * np.random.randn(N_U,1) # add the noise


N_u = 10     # number of points for \Omega_2 and \Omega_3
x_vec = np.linspace(0.0,1.0,N_u).reshape((-1,1))
init_pts = np.concatenate((x_vec,np.zeros((N_u,1))),axis=1) # training inputs for enforcing initial conditions
left_b = np.concatenate((np.zeros((N_u,1)),x_vec),axis=1) # training inputs for enfrocing left boundary condition
right_b = np.concatenate((np.ones((N_u,1)),x_vec),axis=1) # training inputs for enforcing right boundary condition
xu = np.concatenate((init_pts,left_b,right_b),axis=0) # concatenation of the training inputs
yu = u(xu) # training outputs for \Omega2 and \Omega 3 or observations y_2 and y_3
yu = yu + noise * np.std(yu) * np.random.randn(3*N_u,1) # add the noise


xx, tt = np.meshgrid(np.linspace(0,1,51),np.linspace(0,1,51)) 
xt = np.concatenate((xx.reshape((-1,1)),tt.reshape((-1,1))),axis=1) # test inputs on lattice
yt = u(xt)           # test outputs

dataset = {'xu_train': xu, 'yu_train': yu, \
           'xU_train': xU, 'yU_train': yU, \
           'x_test': xt, 'y_test': yt, \
           'diffusivity': c, 'noise': noise}  ## collect the training and test sets for feeding them to machine learnig models


###################### plot the fabricated solutionf field
fig = plt.figure()
plt.contourf(xx, tt, yt.reshape(xx.shape),50, cmap='jet')
plt.plot(xU[:,0], xU[:,1], 'wo', xu[:,0], xu[:,1], 'rs')
plt.xlabel('x')
plt.ylabel('t')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.colorbar()
plt.savefig('u-exact.png', dpi = 300)
plt.close(fig)


##################### Continuous time GP
GP_model = models.Continuous_time_GP_forward(dataset) # call the continuous time GP model from the modulus 'models.py' and build the model
t_start = time.time()

GP_model.training(num_iter = 30001, learning_rate = 1.0e-3) # training an testing

t_end = time.time()

print ('Computational time (secs): ', t_end-t_start)

fig = plt.figure()
plt.contourf(xx, tt, GP_model.mean.reshape(xx.shape),50, cmap='jet')
plt.colorbar()
plt.plot(xU[:,0], xU[:,1],'wo', xu[:,0], xu[:,1], 'rs')
plt.xlabel('x')
plt.ylabel('t')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.title('Mean -- Continuous time GP')
plt.savefig('C-GP_mean_noise_'+str(noise*100)+'.png', dpi = 300)
plt.close(fig)

fig = plt.figure()
plt.contourf(xx, tt, GP_model.std.reshape(xx.shape),50, cmap='jet')
plt.colorbar()
plt.plot(xU[:,0], xU[:,1],'wo', xu[:,0], xu[:,1], 'rs')
plt.xlabel('x')
plt.ylabel('t')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.title('Standard deviation -- Continuous time GP')
plt.savefig('C-GP_std_noise_'+str(noise*100)+'.png', dpi = 300)
plt.close(fig)


##################### Continuous time NN
NN_model = models.Continuous_time_NN_forward(dataset)

t_start = time.time()

NN_model.training(num_iter = 100001, learning_rate = 1.0e-3)

t_end = time.time()

print ('Computational time (secs): ', t_end-t_start)

fig = plt.figure()
plt.contourf(xx, tt, NN_model.u.reshape(xx.shape),50, cmap='jet')
plt.colorbar()
plt.plot(xU[:,0], xU[:,1],'wo', xu[:,0], xu[:,1], 'rs')
plt.xlabel('x')
plt.ylabel('t')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.title('Continuous time NN')
plt.savefig('C-NN_noise_'+str(noise*100)+'.png', dpi = 300)
plt.close(fig)










