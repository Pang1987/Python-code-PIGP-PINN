# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:38:40 2019

@author: gpang
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



class Continuous_time_GP_forward:
    
    def __init__(self, dataset):
        self.xu_train = dataset['xu_train']
        self.yu_train = dataset['yu_train']
        self.xU_train = dataset['xU_train']
        self.yU_train = dataset['yU_train']
        self.x_test = dataset['x_test']
        self.y_test = dataset['y_test']
        self.diffusivity = dataset['diffusivity']

        
    def ku(self, x, xp, n_x, sig, lx, lt,  diag = False):  
        # xp : x';  hyp: hyper-parameters
        # n_x: number of x's
        # sig_f: signal standard deviation
        # lx:  characteristic length in x direction
        # lt:  characteristic legnth in t direction
        # 'diag = False' : output a matrix of size n_x \times n_xp
        # 'diag = True' : output a column vector of size n_x \times 1 (n_x=n_xp) 
       x1, t1 = x[:,0:1], x[:,1:2]
       x2, t2 = xp[:,0:1], xp[:,1:2]
       x2 = tf.reshape(x2, (1,-1))
       t2 = tf.reshape(t2, (1,-1))

       if diag == False:
           k = sig**2 * tf.exp(-(x1-x2)**2/2/lx**2 - (t1-t2)**2/2/lt**2)
       else:
           k = sig**2 * tf.ones((n_x,1),dtype=tf.float64)
           
       return k


    def kU(self, x, xp, n_x, sig, lx, lt, c,  diag = False):  
        # xp : x';  hyp: hyper-parameters
        # n_x: number of x's
        # sig: signal standard deviation
        # lx:  characteristic length in x direction
        # lt:  characteristic legnth in t direction
        # c: diffusivity
        # 'diag = False' : output a matrix of size n_x \times n_xp
        # 'diag = True' : output a column vector of size n_x \times 1 (n_x=n_xp) 
       x1, t1 = x[:,0:1], x[:,1:2]
       x2, t2 = xp[:,0:1], xp[:,1:2]
       x2 = tf.reshape(x2, (1,-1))
       t2 = tf.reshape(t2, (1,-1))
            
       # Use Maple to do symbol manipulation
       if diag == False:
           k = 3*tf.exp((-(x1 - x2)**2*lt**2 - (t1 - t2)**2*lx**2)/(2*lx**2*lt**2))*(c**2*(lx**4 - 2*(x1 - x2)**2*lx**2 + (x1 - x2)**4/3)*lt**4 + lx**8*lt**2/3 - lx**8*(t1 - t2)**2/3)/(lt**4*lx**8)
           k = sig**2 * k
       else:
           k = 3*(c**2*lt**4*lx**4 + 1/3*lx**8*lt**2)/(lt**4*lx**8)
           k = sig**2 * k
           
       return k
   
    
    def kuU(self, x, xp, n_x, sig, lx, lt, c):  
        # xp : x';  hyp: hyper-parameters
        # n_x: number of x's
        # sig: signal standard deviation
        # lx:  characteristic length in x direction
        # lt:  characteristic legnth in t direction
        # c: diffusivity
      
       x1, t1 = x[:,0:1], x[:,1:2]
       x2, t2 = xp[:,0:1], xp[:,1:2]
       x2 = tf.reshape(x2, (1,-1))
       t2 = tf.reshape(t2, (1,-1))
            
       # Use Maple to do symbol manipulation
       k = tf.exp((-(x1 - x2)**2*lt**2 - (t1 - t2)**2*lx**2)/(2*lx**2*lt**2))*(c*(lx + x1 - x2)*(lx - x1 + x2)*lt**2 + (t1 - t2)*lx**4)/(lt**2*lx**4)
       
           
       return k * sig**2    
    
    def kUu(self, x, xp, n_x, sig, lx, lt, c):  
        # xp : x';  hyp: hyper-parameters
        # n_x: number of x's
        # sig: signal standard deviation
        # lx:  characteristic length in x direction
        # lt:  characteristic legnth in t direction
        # c: diffusivity
      
       x1, t1 = x[:,0:1], x[:,1:2]
       x2, t2 = xp[:,0:1], xp[:,1:2]
       x2 = tf.reshape(x2, (1,-1))
       t2 = tf.reshape(t2, (1,-1))
            
       # Use Maple to do symbol manipulation
       k = tf.exp((-(x1 - x2)**2*lt**2 - (t1 - t2)**2*lx**2)/(2*lx**2*lt**2))*(c*(lx + x1 - x2)*(lx - x1 + x2)*lt**2 - (t1 - t2)*lx**4)/(lt**2*lx**4)

           
       return k  * sig**2     
   
        
    def K_train(self, xu, xU, n_u, n_U, sig, lx, lt, c):  ## assemble the convariance matrix for training
        KU = self.kU(xU, xU, n_U, sig, lx, lt, c)
        Ku = self.ku(xu, xu, n_u, sig, lx, lt)
        KuU = self.kuU(xu, xU, n_u, sig, lx, lt, c)
        KUu = self.kUu(xU, xu, n_U, sig, lx, lt, c)
        K1 = tf.concat((KU, KUu),axis=1)
        K2 = tf.concat((KuU, Ku),axis=1)
        K = tf.concat((K1,K2),axis=0)
        return K

    def K_test(self, xt, xu, xU, n_t, sig, lx, lt, c): ## assemble the covariance matrix for testing or predicting
        Ku = self.ku(xt, xu, n_t, sig, lx, lt)
        KuU = self.kuU(xt, xU, n_t, sig, lx, lt, c)
        K = tf.concat((KuU, Ku),axis=1)
        K_diag = self.ku(xt, xu, n_t, sig, lx, lt, diag = True)
        return K, K_diag
    
  
    def nlml(self, xu, yu, n_u, xU, yU, n_U, sig, lx, lt, sig_n, c): ## negative log-marginal likeliood
        N = n_u + n_U
        self.Kn =  self.K_train(xu, xU, n_u, n_U, sig, lx, lt, c)+ (sig_n**2+1.0e-10) * tf.eye(N, dtype=tf.float64) 
        self.L = tf.cholesky(self.Kn)
        r = tf.concat((yU,yu),axis=0)
        self.alpha = tf.cholesky_solve(self.L, r)
        temp = tf.matmul(r, self.alpha, transpose_a=True)
        return temp /2.0  +tf.reduce_sum(tf.log(tf.diag_part(self.L))) \
                  + 0.5 * N * np.log(2.0*np.pi) 
                  
                  
    def training(self, num_iter=10001, learning_rate = 5.0e-4):
        
        tf.reset_default_graph()

        ## initialize hyperparameters of GP; 
        ## 'tf.exp' preserves the positivity of hyperparameters
        sig = tf.exp(tf.Variable(0.0,dtype=np.float64))  # signal standard deviation 
        lx  = tf.exp(tf.Variable(0.0,dtype=np.float64))  # charactersitic length in space
        lt  = tf.exp(tf.Variable(0.0,dtype=np.float64))  # characteristic length in time 
        sig_n = tf.exp(tf.Variable(0.0,dtype=np.float64)) # noise standard deviation
        
        
        
        
        c = self.diffusivity
        
        
        n_u = self.xu_train.shape[0]
        n_U = self.xU_train.shape[0]
        n_t = self.x_test.shape[0]

        xu = tf.placeholder(tf.float64, shape=(None,2))
        yu = tf.placeholder(tf.float64, shape=(None,1))
        xU = tf.placeholder(tf.float64, shape=(None,2))
        yU = tf.placeholder(tf.float64, shape=(None,1))        
        xt = tf.placeholder(tf.float64, shape=(None,2))
        

                
        nlml_tf = self.nlml(xu, yu, n_u, xU, yU, n_U, sig, lx, lt, sig_n, c) 
        k_test, k_diag = self.K_test(xt, xu, xU, n_t, sig, lx, lt, c)
        mean_u = tf.matmul(k_test,self.alpha)     ## posterior mean   
        V = tf.linalg.triangular_solve(self.L,tf.transpose(k_test))
        var_u = k_diag - tf.reshape(tf.reduce_sum(V*V,axis=0),(-1,1)) + sig_n**2 ## posterior variance                    
        std_u = tf.sqrt(tf.maximum(var_u, tf.zeros((n_t,1),dtype=tf.float64)))  ## keep the variance non-negative   
    
    
        optimizer_Adam = tf.train.AdamOptimizer(learning_rate) ### Employ Adam stochastic gradient descent
        train_op_Adam = optimizer_Adam.minimize(nlml_tf)   ## try to miniminze the 'nlml'
        
        nlml_min = 1.0e16
        
        feed_dict = {xu: self.xu_train, yu: self.yu_train, \
                     xU: self.xU_train, yU: self.yU_train, \
                     xt: self.x_test}
        
        
        
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(num_iter):    
                sess.run(train_op_Adam, feed_dict=feed_dict) # training for one iteration
                if i % 10000 == 0:  # print results every 10000 iterations or epochs 
                    nlml_temp = sess.run(nlml_tf, feed_dict = feed_dict)
                    if nlml_temp < nlml_min:
                        nlml_min = nlml_temp   # keep the results corresponding to lowest loss
                        self.mean, self.std, sig0, lx0, lt0, sig_n0 =  \
                           sess.run([mean_u, std_u, sig, lx, lt, sig_n], feed_dict=feed_dict)
                           
                        print ('*****************Iter:  ',i, '   *********** \n')
                        print ('nlml:   ', nlml_min)
                        print ('signal std:   ', sig0)
                        print ('noise std:   ',sig_n0)
                        print ('lx:   ', lx0)
                        print ('lt:   ', lt0)
                        
                        print ('L2_error:    ', np.linalg.norm(self.mean-self.y_test,2)/np.linalg.norm(self.y_test,2))           
                        print ('\n')
        
       
            
    
    
class Continuous_time_NN_forward:   
  
    def __init__(self, dataset):
        self.xu_train = dataset['xu_train']
        self.yu_train = dataset['yu_train']
        self.xU_train = dataset['xU_train']
        self.yU_train = dataset['yU_train']
        self.x_test = dataset['x_test']
        self.y_test = dataset['y_test']
        self.diffusivity = dataset['diffusivity']
        self.noise = dataset['noise']
        
        
    def xavier_init(self,size): # Initialize weight matrices of NN and fix the initialization of weights for sake of reproducing
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev,dtype=tf.float64,seed=1234), dtype=tf.float64)

    
    def DNN(self, X, layers,weights,biases): # forward propagatio of a fully-connected NN
        L = len(layers)
        H = X 
        for l in range(0,L-2): 
            W = weights[l] 
            b = biases[l]
            H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))        
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b) 
        return Y  




    def training(self, num_iter=10001, learning_rate = 5.0e-4):    

        
        tf.reset_default_graph()
        
        layers = [2]+[20]*3 +[1] #DNN layers: two neurons in input layer, three hidden layers with 20 neurons in each, and one neuron in output layer
        
        L = len(layers)
        weights = [self.xavier_init([layers[l], layers[l+1]]) for l in range(0, L-1)]   
        biases = [tf.Variable( tf.zeros((1, layers[l+1]),dtype=tf.float64)) for l in range(0, L-1)]
        
        
        x_u = tf.placeholder(tf.float64, shape=(None,1))
        t_u = tf.placeholder(tf.float64, shape=(None,1))
        x_U = tf.placeholder(tf.float64, shape=(None,1))
        t_U = tf.placeholder(tf.float64, shape=(None,1))
      
        u_u = self.DNN(tf.concat((x_u,t_u),axis=1), layers, weights, biases) ## NN for the soltion u, with inputs (x_u,t_u)
        u_U = self.DNN(tf.concat((x_U,t_U),axis=1), layers, weights, biases) ## NN for u, but with different inputs (x_U, t_U)
        u_U_x = tf.gradients(u_U,x_U)[0]  ## automatic differentiation to compute the gradients of output with respect to input 
        u_U_xx = tf.gradients(u_U_x,x_U)[0] ## second derivative with respect to spatial coordinate
        u_U_t = tf.gradients(u_U, t_U)[0]  ## first derivtive in time
           
    
        
        c = self.diffusivity


        u_obs = self.yu_train  ### observations for y_2 and y_3 (see the book chapter for notations y_2 and y_3)
        U_obs = self.yU_train  ### observation for y_1
        
 
        
        U_U = u_U_t - c * u_U_xx 
        

        loss_u = tf.reduce_mean(tf.square(u_u-u_obs))/tf.reduce_mean(tf.square(u_obs)) ## mean squared error in the sense of relative error
                

        loss_U = tf.reduce_mean(tf.square(U_U-U_obs))/tf.reduce_mean(tf.square(U_obs))
        
        if self.noise == 0.0:
            strength = 0.0     
        else:  
            strength = 1.0e-4   ### strength of regularization
            
        reg = 0.0
        for i in range(len(weights)):
            reg = reg + strength * tf.nn.l2_loss(weights[i]) ### add l_2 regularization for reducing the overfitting when noise comes out.
        
        loss = loss_U + loss_u + reg
      
        
        feed_dict = {x_u: self.xu_train[:,0:1], t_u: self.xu_train[:,1:2],  \
                     x_U: self.xU_train[:,0:1], t_U: self.xU_train[:,1:2]
                    }
        

        feed_dict_test = {x_u: self.x_test[:,0:1], t_u: self.x_test[:,1:2], \
                          x_U: self.x_test[:,0:1], t_U: self.x_test[:,1:2]
                       }    
        
        optimizer_Adam = tf.train.AdamOptimizer(learning_rate)
        train_op_Adam = optimizer_Adam.minimize(loss)   

      
        
        x_index = []
        loss_max = 1.0e16
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for i in range(num_iter+1):
                sess.run(train_op_Adam, feed_dict = feed_dict)
                if i % 10000 == 0:
                   loss_val, loss_u_val, loss_U_val = sess.run([loss,loss_u, loss_U], feed_dict=feed_dict)
                   self.u, self.f = sess.run([u_u, U_U], feed_dict=feed_dict_test)
                                                   
                   if loss_val < loss_max:
                        loss_max = loss_val
                        error= np.linalg.norm(self.u-self.y_test)/np.linalg.norm(self.y_test)

                        
                        x_index.append(i)
                        
                        print ('***************Iteration:   ',  i, '************\n')
                        print ('Loss:  ' , loss_max, 'Loss_u: ', loss_u_val, 'Loss_U: ', loss_U_val)
                        print ('L2_error:  ', error)




class Discrete_time_GP_inverse:

    def __init__(self, dataset):
        self.xu_train = dataset['xu_train']
        self.yu_train = dataset['yu_train']
        self.xU_train = dataset['xU_train']
        self.yU_train = dataset['yU_train']
        self.noise = dataset['noise']

    
    def ku(self, x, xp, n_x, sig, lx, lt,  diag = False):  
        # xp : x';  hyp: hyper-parameters
        # n_x: number of x's
        # sig_f: signal standard deviation
        # lx:  characteristic length in x direction
        # lt:  characteristic legnth in t direction
        # 'diag = False' : output a matrix of size n_x \times n_xp
        # 'diag = True' : output a column vector of size n_x \times 1 (n_x=n_xp) 
       x1 = x[:,0:1]
       x2 = xp[:,0:1]
       x2 = tf.reshape(x2, (1,-1))
     

       if diag == False:
           k = sig**2 * tf.exp(-(x1-x2)**2/2/lx**2 )
       else:
           k = sig**2 * tf.ones((n_x,1),dtype=tf.float64)
           
       return k


    def kU(self, x, xp, n_x, sig, lx, lt, c, dt, diag = False):  
        # xp : x';  hyp: hyper-parameters
        # n_x: number of x's
        # sig: signal standard deviation
        # lx:  characteristic length in x direction
        # lt:  characteristic legnth in t direction
        # c: diffusivity
        # 'diag = False' : output a matrix of size n_x \times n_xp
        # 'diag = True' : output a column vector of size n_x \times 1 (n_x=n_xp) 
       x1 = x[:,0:1]
       x2 = xp[:,0:1]
       x2 = tf.reshape(x2, (1,-1))
            
       # Use Maple to do symbol manipulation
       if diag == False:
           k = 3*tf.exp(-(x1 - x2)**2/(2*lx**2))*(c**2*(lx**4 - 2*(x1 - x2)**2*lx**2 + (x1 - x2)**4/3)*dt**2 + (2*c*lx**4*(lx + x1 - x2)*(lx - x1 + x2)*dt)/3 + lx**8/3)/lx**8
           k = sig**2 * k
       else:
           k =  3*(c**2*dt**2*lx**4 + 2/3*c*dt*lx**6 + 1/3*lx**8)/lx**8
           k = sig**2 * k
           
       return k
   
    
    def kuU(self, x, xp, n_x, sig, lx, lt, c, dt):  
        # xp : x';  hyp: hyper-parameters
        # n_x: number of x's
        # sig: signal standard deviation
        # lx:  characteristic length in x direction
        # lt:  characteristic legnth in t direction
        # c: diffusivity
      
       x1 = x[:,0:1]
       x2 = xp[:,0:1]
       x2 = tf.reshape(x2, (1,-1))
            
       # Use Maple to do symbol manipulation
       k = (c*(lx + x1 - x2)*(lx - x1 + x2)*dt + lx**4)*tf.exp(-(x1 - x2)**2/(2*lx**2))/lx**4
       
           
       return k * sig**2    
    
    def kUu(self, x, xp, n_x, sig, lx, lt, c,dt):  
        # xp : x';  hyp: hyper-parameters
        # n_x: number of x's
        # sig: signal standard deviation
        # lx:  characteristic length in x direction
        # lt:  characteristic legnth in t direction
        # c: diffusivity
      
       x1 = x[:,0:1]
       x2 = xp[:,0:1]
       x2 = tf.reshape(x2, (1,-1))
            
       # Use Maple to do symbol manipulation
       k = (c*(lx + x1 - x2)*(lx - x1 + x2)*dt + lx**4)*tf.exp(-(x1 - x2)**2/(2*lx**2))/lx**4

           
       return k  * sig**2     
   
        
    def K_train(self, xu, xU, n_u, n_U, sig, lx, lt, c, dt):
        KU = self.kU(xU, xU, n_U, sig, lx, lt, c, dt)
        Ku = self.ku(xu, xu, n_u, sig, lx, lt)
        KuU = self.kuU(xu, xU, n_u, sig, lx, lt, c, dt)
        KUu = self.kUu(xU, xu, n_U, sig, lx, lt, c, dt)
        K1 = tf.concat((KU, KUu),axis=1)
        K2 = tf.concat((KuU, Ku),axis=1)
        K = tf.concat((K1,K2),axis=0)
        return K

    def K_test(self, xt, xu, xU, n_t, sig, lx, lt, c, dt):
        Ku = self.ku(xt, xu, n_t, sig, lx, lt)
        KuU = self.kuU(xt, xU, n_t, sig, lx, lt, c, dt)
        K = tf.concat((KuU, Ku),axis=1)
        K_diag = self.ku(xt, xu, n_t, sig, lx, lt, diag = True)
        return K, K_diag
    
  
    def nlml(self, xu, yu, n_u, xU, yU, n_U, sig, lx, lt, sig_n, c, dt):
        N = n_u + n_U
        self.Kn =  self.K_train(xu, xU, n_u, n_U, sig, lx, lt, c, dt)+ (sig_n**2+1.0e-10) * tf.eye(N, dtype=tf.float64)


       
        
        self.L = tf.cholesky(self.Kn)
        r = tf.concat((yU,yu),axis=0)
        self.alpha = tf.cholesky_solve(self.L, r)
        temp = tf.matmul(r, self.alpha, transpose_a=True)
        return temp /2.0  +tf.reduce_sum(tf.log(tf.diag_part(self.L))) \
                  + 0.5 * N * np.log(2.0*np.pi) 
                  
                  
    def training(self, num_iter=10001, learning_rate = 5.0e-4):
        
        tf.reset_default_graph()

        sig = tf.exp(tf.Variable(0.0,dtype=np.float64))   
        lx  = tf.exp(tf.Variable(0.0,dtype=np.float64))
        lt  = tf.exp(tf.Variable(0.0,dtype=np.float64))   
        sig_n = tf.exp(tf.Variable(0.0,dtype=np.float64))
        c = tf.exp(tf.Variable(0.0,dtype=np.float64, trainable=True)) 
        
        dt = 0.01
        
        n_u = self.xu_train.shape[0]
        n_U = self.xU_train.shape[0]

        xu = tf.placeholder(tf.float64, shape=(None,1))
        yu = tf.placeholder(tf.float64, shape=(None,1))
        xU = tf.placeholder(tf.float64, shape=(None,1))
        yU = tf.placeholder(tf.float64, shape=(None,1))        
        

                
        nlml_tf = self.nlml(xu, yu, n_u, xU, yU, n_U, sig, lx, lt, sig_n, c, dt)
  
    
    
        optimizer_Adam = tf.train.AdamOptimizer(learning_rate)
        train_op_Adam = optimizer_Adam.minimize(nlml_tf)   
        
        nlml_min = 1.0e16
        
        feed_dict = {xu: self.xu_train, yu: self.yu_train, \
                     xU: self.xU_train, yU: self.yU_train
                     }
        
        
        index = []
        c_record = []
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(num_iter):    
                sess.run(train_op_Adam, feed_dict=feed_dict)
                if i % 100 == 0:
                    nlml_temp = sess.run(nlml_tf, feed_dict = feed_dict)
                    if nlml_temp < nlml_min:
                        index.append(i)
                  
                        nlml_min = nlml_temp
                        sig0, lx0, lt0, sig_n0, self.c0 =  \
                           sess.run([sig, lx, lt, sig_n, c], feed_dict=feed_dict)
                        c_record.append(self.c0)   
            fig=plt.figure()            
            plt.plot(np.stack(index), np.stack(c_record),'r.-',label='Optimization history')
            plt.plot(np.stack(index), 0.1*np.ones((len(index),1)),'b-.',label='True parameter')
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('Estimated c')
            plt.title('D-GP-estimated c: '+str(self.c0)+' (True c: 0.1)')
            plt.savefig('D-GP_noise_'+str(self.noise*100)+'.png',dpi=300)
            
            plt.close(fig)
                        
    


class Discrete_time_NN_inverse:
    
    def __init__(self, dataset):
        self.xu_train = dataset['xu_train']
        self.yu_train = dataset['yu_train']
        self.xU_train = dataset['xU_train']
        self.yU_train = dataset['yU_train']
        self.noise = dataset['noise']
        
        
    def xavier_init(self,size): # Initializing the weight matrices of NN
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev,dtype=tf.float64,seed=1234), dtype=tf.float64)

    
    def DNN(self, X, layers,weights,biases):
        L = len(layers)
        H = X 
        for l in range(0,L-2): 
            W = weights[l] 
            b = biases[l]
            H = tf.nn.tanh(tf.add(tf.matmul(H, W), b)) 
          
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b) 
        return Y  




    def training(self, num_iter=10001, learning_rate = 5.0e-4):    

        
        tf.reset_default_graph()
        
        layers = [1]+[20]*3 +[1] #DNN layers
        
        L = len(layers)
        weights = [self.xavier_init([layers[l], layers[l+1]]) for l in range(0, L-1)]   
        biases = [tf.Variable( tf.zeros((1, layers[l+1]),dtype=tf.float64)) for l in range(0, L-1)]
        
        dt = 0.01
        x_u = tf.placeholder(tf.float64, shape=(None,1))
        x_U = tf.placeholder(tf.float64, shape=(None,1))
      
        u_u = self.DNN(x_u, layers, weights, biases) #fractional order - aplha
        u_U = self.DNN(x_U, layers, weights, biases)
        u_U_x = tf.gradients(u_U,x_U)[0]
        u_U_xx = tf.gradients(u_U_x,x_U)[0]
           
    
        
        c = tf.exp(tf.Variable(0.0,dtype=np.float64,trainable=True))


        u_obs = self.yu_train
        U_obs = self.yU_train
        
 
        
        U_U = u_U - dt * c * u_U_xx 
        

        loss_u = tf.reduce_mean(tf.square(u_u-u_obs))/tf.reduce_mean(tf.square(u_obs))
                

        loss_U = tf.reduce_mean(tf.square(U_U-U_obs))/tf.reduce_mean(tf.square(U_obs))
              
        reg = 0.0
        if self.noise == 0.0:
            strength=0.0
        else:
            strength = 1.0e-4
           
        for i in range(len(weights)):
            reg = reg + tf.nn.l2_loss(weights[i])
        
        loss = loss_U + loss_u + strength * reg
      
        
        feed_dict = {x_u: self.xu_train[:,0:1],  \
                     x_U: self.xU_train[:,0:1]
                    }
        

        optimizer_Adam = tf.train.AdamOptimizer(learning_rate)
        train_op_Adam = optimizer_Adam.minimize(loss)   

      
        c_record = []  
        index = []
        loss_max = 1.0e16
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for i in range(num_iter+1):
                sess.run(train_op_Adam, feed_dict = feed_dict)
                if i % 100 == 0:
                   loss_val, loss_u_val, loss_U_val, self.c_val = sess.run([loss,loss_u, loss_U, c], feed_dict=feed_dict)
                                                   
                   if loss_val < loss_max:
                        loss_max = loss_val
                        c_record.append(self.c_val)
                        index.append(i)
                        
            fig=plt.figure()            
            plt.plot(np.stack(index), np.stack(c_record),'r.-',label='Optimization history')
            plt.plot(np.stack(index), 0.1*np.ones((len(index),1)),'b-.',label='True parameter')
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('Estimated c')
            plt.title('D-NN-estimated c: '+str(self.c_val)+' (True c: 0.1)')            
            plt.savefig('D-NN_noise_'+str(self.noise*100)+'.png',dpi=300)
            plt.close(fig)