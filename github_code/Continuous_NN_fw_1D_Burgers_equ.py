# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 20:33:28 2019

@author: gpang
"""





import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from SALib.sample import sobol_sequence
import scipy as sci
import scipy.io as sio


class one_NN:
    
    def __init__(self):
        pass
             
             
    def model(self, dataset):        
        self.xu_train = dataset['xu_train']
        self.yu_train = dataset['yu_train']
        self.xf_train = dataset['xf_train']
        self.yf_train = dataset['yf_train']
        self.xu_test = dataset['xu_test']
        self.yu_test = dataset['yu_test']
        self.xf_test = dataset['xf_test']
        self.yf_test = dataset['yf_test']      
        self.dim = self.xf_train.shape[1]
      
        
       

      
              
    def xavier_init(self,size): # weight intitailing 
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
        #variable creatuion inn tensor flow - intilatisation
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev,dtype=tf.float64,seed=None), dtype=tf.float64)

    
    def DNN(self, X, layers,weights,biases):
        L = len(layers)
        H = X 
        for l in range(0,L-2): # (X*w(X*w + b) + b)...b) Full conected neural network
            W = weights[l] 
            b = biases[l]
            H = tf.nn.tanh(tf.add(tf.matmul(H, W), b)) # H - activation function? 
            #H = tf.sin(tf.add(tf.matmul(H, W), b))
        #the loops are not in the same hirecachy as the loss functions
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b) # Y - output - final yayer
        return Y

                                               
    def training(self, optimizer = 'Adam', num_iter=10001, learning_rate = 5.0e-4):    

        
        print_skip = 200000
        tf.reset_default_graph()
        layers = [2]+[10]*8 +[1] #DNN layers
        
        L = len(layers)
        weights = [self.xavier_init([layers[l], layers[l+1]]) for l in range(0, L-1)]   
        biases = [tf.Variable( tf.zeros((1, layers[l+1]),dtype=tf.float64)) for l in range(0, L-1)]
        
        
        x_u = tf.placeholder(tf.float64, shape=(None,1))
        t_u = tf.placeholder(tf.float64, shape=(None,1))
        x_f = tf.placeholder(tf.float64, shape=(None,1))
        t_f = tf.placeholder(tf.float64, shape=(None,1))
      
        u_u = self.DNN(tf.concat((x_u,t_u),axis=1), layers, weights, biases) #fractional order - aplha
        u_f = self.DNN(tf.concat((x_f,t_f),axis=1), layers, weights, biases)
        u_f_x = tf.gradients(u_f,x_f)[0]
        u_f_xx = tf.gradients(u_f_x,x_f)[0]
        u_f_t = tf.gradients(u_f, t_f)[0]
           
    
        
        self.lambda1 = tf.exp(tf.Variable(np.log(1.0),dtype=np.float64, trainable=False))
        self.lambda2 = tf.exp(tf.Variable(np.log(0.1),dtype=np.float64, trainable=False))


        u_obs = self.yu_train
        f_obs = self.yf_train
        
        u_test = self.yu_test
        f_test = self.yf_test

       
        
        f_f = u_f_t + self.lambda1 * u_f * u_f_x - self.lambda2 * u_f_xx 
        

        
#        Nf = f_obs.shape[0]

        
        
        loss_u = tf.reduce_mean(tf.square(u_u-u_obs))/tf.reduce_mean(tf.square(u_obs))
                

        loss_f = tf.reduce_mean(tf.square(f_f-f_obs))
               
        
        loss = loss_f + loss_u
      
        
        feed_dict = {x_u: self.xu_train[:,0:1], t_u: self.xu_train[:,1:2],  \
                     x_f: self.xf_train[:,0:1], t_f: self.xf_train[:,1:2]
                    }
        
        
        loss_u_test = tf.reduce_mean(tf.square(u_u-u_test))/tf.reduce_mean(tf.square(u_test))
                    
                      
        loss_f_test = tf.reduce_mean(tf.square(f_f-f_test))
                     
        loss_test = loss_f_test + loss_u_test
      
        
        feed_dict_test = {x_u: self.xu_test[:,0:1], t_u: self.xu_test[:,1:2], \
                        x_f: self.xf_test[:,0:1] , t_f: self.xf_test[:,1:2]
                       }
        
           
   
        
        if optimizer == 'Adam':
           optimizer_Adam = tf.train.AdamOptimizer(learning_rate)
           train_op_Adam = optimizer_Adam.minimize(loss)   

      
        
           loss_train_history = []
           loss_test_history = []
           loss_f_train_history = []
           loss_f_test_history = []
           loss_u_train_history = []
           loss_u_test_history = []
#           err_u_train_history = []
           self.err_u_test_history = []
#           err_f_train_history = []
#           err_f_test_history = []
           x_index = []
           loss_max = 1.0e16
           is_shown = True
           with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                
                for i in range(num_iter+1):
                    sess.run(train_op_Adam, feed_dict = feed_dict)
                    if i % print_skip == 0:
                       loss_val = sess.run(loss, feed_dict=feed_dict)
                       if loss_val < loss_max:
                            loss_max = loss_val
                          
                            if is_shown == True:     
                                loss_train_val, loss_f_train_val, loss_u_train_val, self.u_train_val, self.f_train_val \
                                = sess.run([loss, loss_f, loss_u, u_u, f_f], feed_dict=feed_dict)
                                loss_test_val, loss_f_test_val, loss_u_test_val, self.u_test_val, self.f_test_val \
                                = sess.run([loss_test, loss_f_test, loss_u_test, u_u, f_f], feed_dict=feed_dict_test)
                                
#                                err_u_train = np.linalg.norm(self.u_train_val-u_obs)/np.linalg.norm(u_obs)
#                                err_f_train = np.linalg.norm(self.f_train_val-f_obs)#/np.linalg.norm(f_obs)
                                err_u_test = np.linalg.norm(self.u_test_val-u_test)/np.linalg.norm(u_test)
                                err_f_test = np.linalg.norm(self.f_test_val-f_test)
                                
                                loss_train_history.append(loss_train_val)
                                loss_f_train_history.append(loss_f_train_val)
                                loss_u_train_history.append(loss_u_train_val)
                                loss_test_history.append(loss_test_val)
                                loss_f_test_history.append(loss_f_test_val)
                                loss_u_test_history.append(loss_u_test_val)
                                
                                x_index.append(i)
                                
                                print ('***************Iteration:   ',  i, '************')
                                print ('error u= ', err_u_test)
                                print ('error f= ', err_f_test)                                
                                print( 'loss_u =', loss_u_train_val)
                                print( 'loss_f =', loss_f_train_val)

                                print ('lambda= ', sess.run([self.lambda1, self.lambda2]))

 

#
#                                plt.subplot(1,3,1)
#                                plt.semilogy(np.stack(x_index), np.stack(loss_train_history),'r.-', label='loss_train')
#                                plt.semilogy(np.stack(x_index), np.stack(loss_test_history), 'b.-', label='loss_test')
#                                plt.legend()
#                                plt.title('Loss history')
#                                plt.xlabel('Iter. No.')
#                                plt.subplot(1,3,2)
#                                plt.semilogy(np.stack(x_index),np.stack(loss_f_train_history),'r.-',label='loss_f_train')
#                                plt.semilogy(np.stack(x_index),np.stack(loss_f_test_history),'b.-',label='loss_f_test')
#                                plt.legend()
#                                plt.subplot(1,3,3)
#                                plt.semilogy(np.stack(x_index),np.stack(loss_u_train_history),'r.-',label='loss_u_train')
#                                plt.semilogy(np.stack(x_index),np.stack(loss_u_test_history),'b.-', label='loss_u_test')
#                                plt.legend()
#                                plt.savefig('fig/loss_history.png', dpi=1000)
#                                plt.show()
#                                
#                                err_u_train_history.append(err_u_train)
#                                err_f_train_history.append(err_f_train)
#                                
#                                self.err_u_test_history.append(err_u_test)
#                                err_f_test_history.append(err_f_test)
#                                
#                                
#                             
#                                plt.subplot(1,2,1)
#                                plt.semilogy(np.stack(x_index),np.stack(err_f_train_history),'r.-',label='err_f_train')
#                                plt.semilogy(np.stack(x_index),np.stack(err_f_test_history),'b.-',label='err_f_test')
#                                plt.legend()
#                                plt.title('Error history')
#                                plt.xlabel('Iter. No.')
#                                plt.subplot(1,2,2)
#                                plt.semilogy(np.stack(x_index),np.stack(err_u_train_history),'r.-',label='err_u_train')
#                                plt.semilogy(np.stack(x_index),np.stack(self.err_u_test_history),'b.-', label='err_u_test')
#                                plt.legend()
#                                plt.savefig('fig/err_history.png', dpi=1000)
#                                plt.show()                    
                                 
             
             
             
             
u_simulation = sio.loadmat('burgers.mat')
u_exa = np.real(u_simulation['usol'])
t_exa = u_simulation['t'].reshape((-1,1))
x_exa = u_simulation['x'].reshape((-1,1))
             
             
def u_exact(x,t, u_exa, t_exa, x_exa, dim):
    if dim == 1:
        tt = np.ndarray.flatten(t_exa)
        uu1 =  np.ndarray.flatten(u_exa[0,:])
        uu2 = np.ndarray.flatten(u_exa[-1,:])
        f1 = sci.interpolate.interp1d(tt,uu1,kind='cubic')
        f2 = sci.interpolate.interp1d(tt,uu2,kind='cubic')
        u1 = f1(t)
        u2 = f2(t)
        return np.array([[u1],[u2]],dtype=np.float64)
    elif dim == 2:
        t = t*np.ones((x.shape[0],1),dtype=np.float64)
        [tt, xx] = np.meshgrid(t_exa,x_exa)
        ttt = tt.reshape((-1,1))
        xxx = xx.reshape((-1,1))
        uuu = u_exa.reshape((-1,1))
        return sci.interpolate.griddata(np.concatenate((ttt,xxx),axis=1),uuu, np.concatenate((t,x),axis=1), fill_value = 0.0, method='cubic')



def f_exact(x,t):
#    return 4.0*np.ones((x.shape[0],1),dtype=np.float64)
#      return np.zeros((x.shape[0],1),dtype=np.float64)
    return np.zeros((x.shape[0],1),dtype=np.float64)

tt0 = time.time()

fig = plt.figure()
plt.contourf(np.ndarray.flatten(t_exa[:51]), np.ndarray.flatten(x_exa), u_exa[:,:51], 100, cmap='jet')
plt.colorbar()
plt.xlabel('t')
plt.ylabel('x')
plt.title('1D Burgers\' equation: Exact solution')
plt.tight_layout()
plt.savefig('C-NN-FW-FIG/Exact-Burgers.png',dpi=1000)
#plt.show()
plt.close(fig)

#Nu = 300
Nf = 2000


init_time = 0.0

tt, xx = np.meshgrid(np.ndarray.flatten(t_exa[:51]),np.ndarray.flatten(x_exa))



xf_train = sobol_sequence.sample(Nf+1,2)[1:,:]
xf_train[:,0] = -8.0+16.0*xf_train[:,0] # x
xf_train[:,1] = 5.0*xf_train[:,1]     # t


#xf_test = np.concatenate((x_exa, 8.2*np.ones((x_exa.shape[0],1))),axis=1)
xf_test = np.concatenate((xx.reshape((-1,1)),tt.reshape((-1,1))),axis=1)

xu_test = xf_test
yf_train = f_exact(xf_train[:,0:1],xf_train[:,1:2])
#yf_train = yf_train+np.linalg.cholesky(previous_cov_mat[:Nf,:Nf])@ np.random.randn(Nf,1)

xu_train = np.concatenate((x_exa,0.0*np.ones((x_exa.shape[0],1))),axis=1)
xu_train = np.concatenate((xu_train, np.concatenate((-8.0*np.ones((t_exa[:51].shape[0],1)),t_exa[:51]),axis=1)),axis=0)
xu_train = np.concatenate((xu_train, np.concatenate((8.0*np.ones((t_exa[:51].shape[0],1)),t_exa[:51]),axis=1)),axis=0)


#plt.plot(xu_train[:,0],xu_train[:,1],'ro',xf_train[:,0],xf_train[:,1],'bo',xf_test[:,0],xf_test[:,1],'go')
#plt.show()

Nt = xf_test.shape[0]

    
yu_train = u_exact(xu_train[:,0:1],xu_train[:,1:2], u_exa, t_exa, x_exa, 2)
yu_test = u_exact(xu_test[:,0:1],xu_test[:,1:2], u_exa, t_exa, x_exa, 2)

yf_test = f_exact(xf_test[:,0:1],xf_test[:,1:2])
    
  
    
dataset = {'xu_train': xu_train, 'yu_train': yu_train, \
           'xu_test':  xu_test,  'yu_test': yu_test, \
           'xf_train': xf_train, 'yf_train': yf_train,  \
           'xf_test': xf_test, 'yf_test': yf_test}
    
    
    
NN_instance = one_NN()
NN_instance.model(dataset)
NN_instance.training(num_iter=200001)
    
u_pred = NN_instance.u_test_val.reshape(tt.shape)    

del NN_instance

fig = plt.figure()
plt.contourf(tt, xx, u_pred, 100, cmap='jet')
plt.colorbar()
plt.xlabel('t')
plt.ylabel('x')
plt.title('1D Burgers\' equation: Continuous time NN (solution)')
plt.tight_layout()
plt.savefig('C-NN-FW-FIG/C-NN-Burgers-solution-'+str(Nf)+'.png',dpi=1000)
plt.close(fig)


fig = plt.figure()
plt.contourf(tt, xx, np.abs(u_exa[:,:51]-u_pred), 100, cmap='jet')
plt.colorbar()
plt.xlabel('t')
plt.ylabel('x')
plt.title('1D Burgers\' equation: Continuous time NN (absolute error)')
plt.tight_layout()
plt.savefig('C-NN-FW-FIG/C-NN-Burgers-Ab-err-'+str(Nf)+'.png',dpi=1000)
plt.close(fig)


np.savetxt('C-NN-FW-FIG/exact_u.txt', u_exa, fmt='%10.5e')    
np.savetxt('C-NN-FW-FIG/predicted_u.txt', u_pred, fmt='%10.5e')    

u_error = np.linalg.norm(u_exa[:,:51].reshape((-1,1))-u_pred.reshape((-1,1)))/np.linalg.norm(u_exa[:,:51].reshape((-1,1)))
print('u_error= ', u_error)
np.savetxt('C-NN-FW-FIG/u_error.txt', [u_error], fmt='%10.5e' )

tt1 = time.time()

print ('CPU time ', tt1-tt0)             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             