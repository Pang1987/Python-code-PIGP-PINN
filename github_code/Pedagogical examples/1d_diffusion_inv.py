


import numpy as np
import matplotlib.pyplot as plt
import models
import time

def u(x,t):
   return np.sin(2.*np.pi*x)*np.exp(-t)


def U(x,t, c):
  return -np.sin(2*np.pi*x)*np.exp(-t)+c*4*np.pi**2*np.sin(2*np.pi*x)*np.exp(-t)

#noise = 0.0  # noise-free
noise = 0.05 # 1% noise

dt = 0.01
N_U = 20
c = 0.1
np.random.seed(seed=1234)
xU = np.linspace(0,1,N_U).reshape((-1,1))
yU = u(xU, 0.5) + dt * U(xU, 0.51, c)  
yU = yU + noise * np.std(yU) * np.random.randn(N_U,1) # add noise

N_u = 20
xu = np.linspace(0,1,N_u).reshape((-1,1))
yu = u(xu, 0.51)
yu = yu + noise * np.std(yu) * np.random.randn(N_u,1)




fig = plt.figure()

plt.plot(xU, yU, 'bo:',label='Snapshot t=0.5')
plt.plot(xu, yu, 'rs:',label='Snapshot t=0.51')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.savefig('snapshots_noise_'+str(noise*100)+'.png',dpi=300)
plt.close(fig)



dataset = {'xu_train': xu, 'yu_train': yu, \
           'xU_train': xU, 'yU_train': yU, 'noise': noise
           }



##################### Discrete time GP
GP_model = models.Discrete_time_GP_inverse(dataset)

t_start = time.time()

GP_model.training(num_iter = 10001, learning_rate = 1.0e-3)

t_end = time.time()

print ('D-GP-Computational time (secs): ', t_end-t_start)

#################### Discrete time NN

NN_model = models.Discrete_time_NN_inverse(dataset)

t_start = time.time()

NN_model.training(num_iter = 20001, learning_rate = 1.0e-3)

t_end = time.time()

print ('D-NN-Computational time (secs): ', t_end-t_start)









