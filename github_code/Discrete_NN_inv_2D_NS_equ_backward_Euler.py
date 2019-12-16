# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:01:51 2019

@author: gpang
"""




import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
#from SALib.sample import sobol_sequence
#import scipy as sci
import scipy.io as sio


class one_NN:
    
    def __init__(self):
        pass
             
             
    def model(self, dataset, dt):        
        self.xu_train = dataset['xu_train']
        self.yu_train = dataset['yu_train']
        self.xf_train = dataset['xf_train']
        self.yf_train = dataset['yf_train']
        self.xu_test = dataset['xu_test']
        self.yu_test = dataset['yu_test']
        self.xf_test = dataset['xf_test']
        self.yf_test = dataset['yf_test']      
        self.dim = self.xf_train.shape[1]
        self.dt = dt

        
       

      
              
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

        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b) # Y - output - final yayer
        return Y

                                               
    def training(self, optimizer = 'Adam', num_iter=10001, learning_rate = 5.0e-4):    

        
        print_skip = 10000
        tf.reset_default_graph()
        layers = [2]+[10]*4 +[2] #DNN layers
        
        L = len(layers)
        weights = [self.xavier_init([layers[l], layers[l+1]]) for l in range(0, L-1)]   
        biases = [tf.Variable( tf.zeros((1, layers[l+1]),dtype=tf.float64)) for l in range(0, L-1)]
        
        
        x_u = tf.placeholder(tf.float64, shape=(None,1))
        y_u = tf.placeholder(tf.float64, shape=(None,1))  
        x_f = tf.placeholder(tf.float64, shape=(None,1))
        y_f = tf.placeholder(tf.float64, shape=(None,1))        
      
        U_u = self.DNN(tf.concat((x_u,y_u),axis=1), layers, weights, biases) #fractional order - aplha
        U_f = self.DNN(tf.concat((x_f,y_f),axis=1), layers, weights, biases)

        phi_u, p_u = U_u[:,0:1], U_u[:,1:2]
        phi_f, p_f = U_f[:,0:1], U_f[:,1:2] 
        
        u_u = tf.gradients(phi_u, y_u)[0]
        v_u = -tf.gradients(phi_u,x_u)[0]
        
        u_f = tf.gradients(phi_f, y_f)[0]
        v_f = -tf.gradients(phi_f,x_f)[0]

        u_f_x = tf.gradients(u_f, x_f)[0]
        u_f_y = tf.gradients(u_f, y_f)[0]
        u_f_xx = tf.gradients(u_f_x, x_f)[0]
        u_f_yy = tf.gradients(u_f_y, y_f)[0]
                  
        v_f_x = tf.gradients(v_f, x_f)[0]
        v_f_y = tf.gradients(v_f, y_f)[0]
        v_f_xx = tf.gradients(v_f_x, x_f)[0]
        v_f_yy = tf.gradients(v_f_y, y_f)[0]      
 
        p_f_x = tf.gradients(p_f, x_f)[0]
        p_f_y = tf.gradients(p_f, y_f)[0]
    
        
        self.lambda1 = tf.exp(tf.Variable(np.log(0.5),dtype=np.float64, trainable=True))
        self.lambda2 = tf.exp(tf.Variable(np.log(0.5),dtype=np.float64, trainable=True))


        u_obs = self.yu_train
        f_obs = self.yf_train
        
        u_test = self.yu_test
        f_test = self.yf_test

       
        
        fu  = u_f + self.lambda1*self.dt*(u_f*u_f_x+v_f*u_f_y) + self.dt*p_f_x - self.lambda2*self.dt*(u_f_xx+u_f_yy)
        fv  = v_f + self.lambda1*self.dt*(u_f*v_f_x+v_f*v_f_y) + self.dt*p_f_y - self.lambda2*self.dt*(v_f_xx+v_f_yy)
        
        

        
    
        

        

        
        
        loss_u = tf.reduce_mean(tf.square(u_u-u_obs[0]))/tf.reduce_mean(tf.square(u_obs[0]))\
                + tf.reduce_mean(tf.square(v_u-u_obs[1]))/tf.reduce_mean(tf.square(u_obs[1]))

        loss_f = tf.reduce_mean(tf.square(fu-f_obs[0]))/tf.reduce_mean(tf.square(f_obs[0]))\
                + tf.reduce_mean(tf.square(fv-f_obs[1]))/tf.reduce_mean(tf.square(f_obs[1]))
               
        
        loss = loss_f + loss_u
      
        
        feed_dict = {x_u: self.xu_train[:,0:1], y_u: self.xu_train[:,1:2],  \
                     x_f: self.xf_train[:,0:1], y_f: self.xf_train[:,1:2] }
        
        
        loss_u_test = tf.reduce_mean(tf.square(u_u-u_test[0]))/tf.reduce_mean(tf.square(u_test[0])) \
                      +tf.reduce_mean(tf.square(v_u-u_test[1]))/tf.reduce_mean(tf.square(u_test[1]))
                    
                      
        loss_f_test = tf.reduce_mean(tf.square(fu-f_test[0]))/tf.reduce_mean(tf.square(f_test[0]))\
                    + tf.reduce_mean(tf.square(fv-f_test[1]))/tf.reduce_mean(tf.square(f_test[1]))
                     
        loss_test = loss_f_test + loss_u_test
      
        
        feed_dict_test = {x_u: self.xu_test[:,0:1], y_u: self.xu_test[:,1:2], \
                         x_f: self.xf_test[:,0:1] , y_f: self.xf_test[:,1:2]
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
                                = sess.run([loss, loss_f, loss_u, u_u, fu], feed_dict=feed_dict)
                                loss_test_val, loss_f_test_val, loss_u_test_val, self.u_test_val, self.f_test_val \
                                = sess.run([loss_test, loss_f_test, loss_u_test, u_u, fu], feed_dict=feed_dict_test)
                                
#                                err_u_train = np.linalg.norm(self.u_train_val-u_obs[0])/np.linalg.norm(u_obs[0])
#                                err_f_train = np.linalg.norm(self.f_train_val-f_obs[0])/np.linalg.norm(f_obs[0])
                                err_u_test = np.linalg.norm(self.u_test_val-u_test[0])/np.linalg.norm(u_test[0])
                                err_f_test = np.linalg.norm(self.f_test_val-f_test[0])/np.linalg.norm(f_test[0])
                                
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
                                print('\n')
                                print ('Estimated lambda= ', sess.run([self.lambda1, self.lambda2]),'\n')
 



               
                                 
             
             
             
             
u_simulation = sio.loadmat('cylinder_fine.mat')

u_exa = np.real(u_simulation['U_star'])
t_exa = u_simulation['t_star'].reshape((-1,1))
x_exa = u_simulation['X_star']
 

tt0 = time.time()

Nf = 250
Nu = 250






dt = 0.02


init_time = 0.18
noise_rate = 0.0

np.random.seed(seed=1234)


index = np.random.permutation(np.arange(x_exa.shape[0]))
index_f = index[:Nf]


index = np.random.permutation(np.arange(x_exa.shape[0]))
index_u = index[:Nu]


xf_train = x_exa[index_f,:]
xf_test = xf_train

yf_train = []
yf_train.append(u_exa[index_f,0:1,9])
yf_train.append(u_exa[index_f,1:2,9])




xu_train = x_exa[index_u,:]

plt.contourf(np.linspace(1.0,7.5,66),np.linspace(-1.7,1.7,35), u_exa[:,1,9].reshape(66,35).T,100,cmap='jet')
plt.colorbar()
plt.plot(xf_train[:,0],xf_train[:,1],'bo',xu_train[:,0],xu_train[:,1],'ro')
plt.show()

xu_test = xf_test







noise_f = noise_rate*np.std(np.ndarray.flatten(yf_train[0]))*np.random.randn(Nf,1)


Nt = xf_test.shape[0]


for k in np.arange(1,2):
    yu_train = []
    yu_test = []
   
    yf_test = yf_train
    
    
    
    
    yu_train.append(u_exa[index_u,0:1,10])
    yu_train.append(u_exa[index_u,1:2,10])
    
    yu_train[0] = yu_train[0] +  noise_rate*np.std(np.ndarray.flatten(yu_train[0]))*np.random.randn(xu_train.shape[0],1)   
    yu_train[1] = yu_train[1] +  noise_rate*np.std(np.ndarray.flatten(yu_train[1]))*np.random.randn(xu_train.shape[0],1)   
    
    yu_test.append( u_exa[index_f,0:1,10])
    yu_test.append( u_exa[index_f,1:2,10])
    
    
    dataset = {'xu_train': xu_train,  'yu_train': yu_train, \
               'xu_test':  xu_test,  'yu_test': yu_test, \
               'xf_train': xf_train, 'yf_train': yf_train,  \
               'xf_test': xf_test, 'yf_test': yf_test}
    
    print ('\n      t = '+ str(dt*k)+ '  *********************')
   
    
    
    NN_instance = one_NN()
    NN_instance.model(dataset, dt)
    
    
    
    NN_instance.training(num_iter=200001)

    
    del NN_instance

tt1 = time.time()

print ('CPU time ', tt1-tt0)             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             