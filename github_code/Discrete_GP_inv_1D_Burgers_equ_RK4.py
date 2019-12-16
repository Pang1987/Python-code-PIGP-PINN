# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 11:34:00 2019

@author: gpang
"""




import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
#from SALib.sample import sobol_sequence
import scipy as sci
import scipy.io as sio


class one_GP:
    
    def __init__(self):
        pass
             
             
    def model(self, dataset, dt, prior_mean_train, prior_mean_test, previous_cov_mat, a, b, c, un_u, un_f, un_t, kernel_type = 'SE', is_noise = True):        
        self.xu_train = dataset['xu_train']
        self.yu_train = dataset['yu_train']
        self.xf_train = dataset['xf_train']
        self.yf_train = dataset['yf_train']
        self.xu_test = dataset['xu_test']
        self.yu_test = dataset['yu_test']
        self.xf_test = dataset['xf_test']
        self.yf_test = dataset['yf_test'] 
        self.un_u = un_u
        self.un_f = un_f
        self.un_t = un_t
        self.kernel_type = kernel_type
        self.dt = dt
        self.prior_mean_train = prior_mean_train
        self.prior_mean_test = prior_mean_test
        self.previous_cov_mat=previous_cov_mat
        self.dim = self.xf_train.shape[1]
        self.is_noise = is_noise
        self.a = a
        self.b = b
        self.c = c
       

      
              
               
    
    def kernel(self, X, Y, t1, equal=False, diag=False):
        if self.kernel_type == 'SE':
            if diag == False:
                return tf.exp(-0.5* (X-Y.T)**2/t1**2)
            else:
                return tf.ones((X.shape[0],1),dtype=tf.float64)
        elif self.kernel_type == 'Matern1':
            dist = tf.sqrt(self.square_dist(X,Y,t1,equal))
            return (1.0+3.0**0.5*dist)*tf.exp(-3.0**0.5*dist)
        elif self.kernel_type == 'Matern2':
            dist = tf.sqrt(self.square_dist(X,Y,t1,equal))
            return (1.0+5.0**0.5*dist+5.0/3.0*dist**2)*tf.exp(-5.0**0.5*dist)            


    def kx(self, X, Y, t1, diag=False):
        Y = Y.T
        if diag == False:
            return (Y-X)/t1**2*tf.exp(-0.5*(X-Y)**2/t1**2)
        else:
            return tf.zeros((X.shape[0],1),dtype=tf.float64)
   
    def ky(self, X, Y, t1, diag=False):
        Y = Y.T
        if diag == False:
            return (X-Y)/t1**2*tf.exp(-0.5*(X-Y)**2/t1**2)
        else:
            return tf.zeros((X.shape[0],1),dtype=tf.float64)
        
    def kxx(self, X, Y, t1, diag=False):
        Y = Y.T
        if diag==False:
            return (-1.0/t1**2+(X-Y)**2/t1**4)*tf.exp(-0.5*(X-Y)**2/t1**2)
        else:
            return -1.0/t1**2 * tf.ones((X.shape[0],1),dtype=tf.float64)
        
        
    def kyy(self, X, Y, t1, diag=False):
        Y = Y.T
        if diag==False:
            return (-1.0/t1**2+(X-Y)**2/t1**4)*tf.exp(-0.5*(X-Y)**2/t1**2)
        else:
            return -1.0/t1**2 * tf.ones((X.shape[0],1),dtype=tf.float64)
        
        
    def kxy(self, X, Y, t1, diag=False):
        Y = Y.T
        if diag==False:
            return (1.0/t1**2-(X-Y)**2/t1**4)*tf.exp(-0.5*(X-Y)**2/t1**2)
        else:
            return 1.0/t1**2*tf.ones((X.shape[0],1),dtype=tf.float64)
        
    def kyyx(self, X, Y, t1, diag=False):
        Y = Y.T
        if diag==False:
            return (3*(X-Y)/t1**4-(X-Y)**3/t1**6)*tf.exp(-0.5*(X-Y)**2/t1**2)
        else:
            return tf.zeros((X.shape[0],1),dtype=tf.float64)
        
    def kyxx(self, X, Y, t1, diag=False):
        Y = Y.T
        if diag==False:
            return (3*(Y-X)/t1**4+(X-Y)**3/t1**6)*tf.exp(-0.5*(X-Y)**2/t1**2)
        else:
            return tf.zeros((X.shape[0],1),dtype=tf.float64)        
        
    def kxxyy(self, X, Y, t1, diag=False):
        Y = Y.T
        if diag==False:
            return (3.0/t1**4-6*(X-Y)**2/t1**6+(X-Y)**4/t1**8)*tf.exp(-0.5*(X-Y)**2/t1**2)
        else:
            return 3.0/t1**4*tf.ones((X.shape[0],1),dtype=tf.float64)
        
        

    def Lap2_kernel(self, X, Y, t1, lambda1, lambda2, un_x, un_y, equal=False, diag=False):
        unx = np.ndarray.flatten(un_x)
        uny = np.ndarray.flatten(un_y)
        unx = tf.diag(unx)
        uny = tf.diag(uny)
        if self.kernel_type == 'SE':
            if diag == False:
                k = lambda1**2*tf.matmul(tf.matmul(unx,self.kxy(X,Y,t1,diag)),uny)-lambda1*lambda2*tf.matmul(unx,self.kyyx(X,Y,t1,diag))\
                  -lambda1*lambda2*tf.matmul(self.kyxx(X,Y,t1,diag),uny)+lambda2**2*self.kxxyy(X,Y,t1,diag)
            else:
                k  = lambda1**2* un_x**2*self.kxy(X,Y,t1,diag)-lambda1*lambda2*un_x*self.kyyx(X,Y,t1,diag)\
                  -lambda1*lambda2*un_y*self.kyxx(X,Y,t1,diag)+lambda2**2*self.kxxyy(X,Y,t1,diag)
            return k

    def Lap1_kernel(self, X, Y, t1, lambda1, lambda2, un_x, un_y, equal=False, diag=False): ## -\Delta rather than \Delta


        if self.kernel_type == 'SE':
            unx = np.ndarray.flatten(un_x)
            uny = np.ndarray.flatten(un_y)
            unx = tf.diag(unx)
            uny = tf.diag(uny)
            if diag == False:
                 k = lambda1*tf.matmul(unx,self.kx(X,Y,t1,diag))-lambda2*self.kxx(X,Y,t1,diag)
               
            else:
                 k = lambda1*un_x*self.kx(X,Y,t1,diag)-lambda2*self.kxx(X,Y,t1,diag)
            return k 

    def Lap1_kernel_prime(self, X, Y, t1, lambda1, lambda2, un_x, un_y, equal=False, diag=False): ## -\Delta rather than \Delta


        if self.kernel_type == 'SE':
            unx = np.ndarray.flatten(un_x)
            uny = np.ndarray.flatten(un_y)
            unx = tf.diag(unx)
            uny = tf.diag(uny)  
            if diag == False:
                  k = lambda1*tf.matmul(self.ky(X,Y,t1,diag),uny)-lambda2*self.kyy(X,Y,t1,diag)
                                
            else:
                  k = lambda1*un_y*self.ky(X,Y,t1,diag)-lambda2*self.kyy(X,Y,t1,diag)

            return k



    def kernel_uf_train(self, Xu, Xf, t1, t3, t5, a, b, c, lambda1, lambda2, un_u, un_f, dt, diag=False):


        
        if self.kernel_type == 'SE':
            if diag == False:
                ku3u3 = self.kernel(Xu[2], Xu[2], t1, equal=True)
                ku2u2 = self.kernel(Xu[1], Xu[1], t3, equal=True)
                ku1u1 = self.kernel(Xu[0], Xu[0], t5, equal=True)
                kf3f3 = self.kernel(Xf, Xf, t1, equal=True)  \
                        + dt**2*b[0]**2*self.Lap2_kernel(Xf, Xf, t5, lambda1, lambda2, un_f, un_f, equal=True) \
                        + dt**2*b[1]**2*self.Lap2_kernel(Xf, Xf, t3, lambda1, lambda2, un_f, un_f, equal=True)
                kf2f2 = self.kernel(Xf, Xf, t3, equal=True) \
                        + dt*a[1,1]*self.Lap1_kernel(Xf, Xf, t3, lambda1, lambda2, un_f, un_f, equal=True) \
                        + dt*a[1,1]*self.Lap1_kernel_prime(Xf, Xf, t3, lambda1, lambda2, un_f, un_f, equal=True)\
                        +dt**2*a[1,0]**2*self.Lap2_kernel(Xf, Xf, t5, lambda1, lambda2, un_f, un_f, equal=True) \
                        +dt**2*a[1,1]**2*self.Lap2_kernel(Xf, Xf, t3, lambda1, lambda2, un_f, un_f, equal=True)
                kf1f1 = self.kernel(Xf, Xf, t5, equal=True) \
                        +dt*a[0,0]*self.Lap1_kernel(Xf, Xf, t5, lambda1, lambda2, un_f, un_f, equal=True)\
                        +dt*a[0,0]*self.Lap1_kernel_prime(Xf, Xf, t5, lambda1, lambda2, un_f, un_f, equal=True)\
                        +dt**2*a[0,1]**2*self.Lap2_kernel(Xf, Xf, t3, lambda1, lambda2, un_f, un_f, equal=True) \
                        +dt**2*a[0,0]**2*self.Lap2_kernel(Xf, Xf, t5, lambda1, lambda2, un_f, un_f, equal=True)
                        
                        
                kf3u3 = self.kernel(Xf, Xu[2], t1)
                kf3u2 = dt*b[1]*self.Lap1_kernel(Xf, Xu[1], t3, lambda1, lambda2, un_f, un_u[1])
                kf2u2 = self.kernel(Xf, Xu[1], t3) + dt*a[1,1]*self.Lap1_kernel(Xf,Xu[1],t3,lambda1, lambda2, un_f, un_u[1])
                kf1u2 = dt*a[0,1]*self.Lap1_kernel(Xf, Xu[1], t3, lambda1, lambda2, un_f, un_u[1])
                kf3u1 = dt*b[0]*self.Lap1_kernel(Xf, Xu[0], t5, lambda1, lambda2, un_f, un_u[0])
                kf2u1 = dt*a[1,0]*self.Lap1_kernel(Xf, Xu[0], t5, lambda1, lambda2, un_f, un_u[0])
                kf1u1 = self.kernel(Xf, Xu[0], t5) + dt*a[0,0]*self.Lap1_kernel(Xf, Xu[0], t5, lambda1, lambda2, un_f, un_u[0])
                kf2f3 = dt*b[1]*self.Lap1_kernel_prime(Xf, Xf, t3, lambda1, lambda2, un_f, un_f) \
                       +dt**2*b[0]*a[1,0]*self.Lap2_kernel(Xf, Xf, t5, lambda1, lambda2, un_f, un_f) \
                       +dt**2*b[1]*a[1,1]*self.Lap2_kernel(Xf, Xf, t3, lambda1, lambda2, un_f, un_f)
                kf1f3 = dt*b[0]*self.Lap1_kernel_prime(Xf, Xf, t5, lambda1, lambda2, un_f, un_f) \
                        + dt**2*b[0]*a[0,0]*self.Lap2_kernel(Xf, Xf, t5, lambda1, lambda2, un_f, un_f) \
                        + dt**2*b[1]*a[0,1]*self.Lap2_kernel(Xf, Xf, t3, lambda1, lambda2, un_f, un_f)
                kf1f2 = dt*a[0,1]*self.Lap1_kernel(Xf, Xf, t3, lambda1, lambda2, un_f, un_f) \
                        +dt*a[1,0]*self.Lap1_kernel_prime(Xf, Xf, t5, lambda1, lambda2, un_f, un_f) \
                        + dt**2*a[1,0]*a[0,0]*self.Lap2_kernel(Xf, Xf, t5, lambda1, lambda2, un_f, un_f)\
                        + dt**2*a[1,1]*a[0,1]*self.Lap2_kernel(Xf, Xf, t3, lambda1, lambda2, un_f, un_f)
                        
                zu3u2 = tf.zeros((Xu[2].shape[0],Xu[1].shape[0]),dtype=tf.float64)        
                zu3u1 = tf.zeros((Xu[2].shape[0],Xu[0].shape[0]),dtype=tf.float64)
                zu2u1 = tf.zeros((Xu[1].shape[0],Xu[0].shape[0]),dtype=tf.float64)        
                
                zu3f = tf.zeros((Xu[2].shape[0],Xf.shape[0]),dtype=tf.float64)        
                zfu3 = tf.zeros((Xf.shape[0],Xu[2].shape[0]),dtype=tf.float64)        

                k1 = tf.concat( (ku3u3, zu3u2, zu3u1, tf.transpose(kf3u3), zu3f, zu3f),axis=1)         
                k2 = tf.concat( (tf.transpose(zu3u2), ku2u2, zu2u1, tf.transpose(kf3u2), tf.transpose(kf2u2), tf.transpose(kf1u2)),axis=1)         
                k3 = tf.concat( (tf.transpose(zu3u1), tf.transpose(zu2u1), ku1u1, tf.transpose(kf3u1), tf.transpose(kf2u1), tf.transpose(kf1u1)),axis=1)         
                k4 = tf.concat( (kf3u3, kf3u2, kf3u1, kf3f3, tf.transpose(kf2f3), tf.transpose(kf1f3)),axis=1)         
                k5 = tf.concat( (zfu3, kf2u2, kf2u1, kf2f3, kf2f2, tf.transpose(kf1f2)),axis=1)         
                k6 = tf.concat( (zfu3, kf1u2, kf1u1, kf1f3, kf1f2, kf1f1),axis=1)         
                
              
                k = tf.concat((k1,k2,k3,k4,k5,k6),axis=0)

                                
                return k
            else:
                ku3u3 = self.kernel(Xu[2], Xu[2], t1, diag=True)
                ku2u2 = self.kernel(Xu[1], Xu[1], t3, diag=True)
                ku1u1 = self.kernel(Xu[0], Xu[0], t5, diag=True)
                kf3f3 = self.kernel(Xf, Xf, t1, diag=True)  \
                        + dt**2*b[0]**2*self.Lap2_kernel(Xf, Xf, t5, lambda1, lambda2, un_f, un_f, diag=True) \
                        + dt**2*b[1]**2*self.Lap2_kernel(Xf, Xf, t3, lambda1, lambda2, un_f, un_f, diag=True)
                kf2f2 = self.kernel(Xf, Xf, t3,  diag=True) \
                        + 2.0*dt*a[1,1]*self.Lap1_kernel(Xf, Xf, t3, lambda1, lambda2, un_f, un_f, diag=True) \
                        +dt**2*a[1,0]**2*self.Lap2_kernel(Xf, Xf, t5, lambda1, lambda2, un_f, un_f, diag=True) \
                        +dt**2*a[1,1]**2*self.Lap2_kernel(Xf, Xf, t3, lambda1, lambda2, un_f, un_f, diag=True)
                kf1f1 = self.kernel(Xf, Xf, t5, diag=True) \
                        +2.0*dt*a[0,0]*self.Lap1_kernel(Xf, Xf, t5, lambda1, lambda2, un_f, un_f, diag=True)\
                        +dt**2*a[0,1]**2*self.Lap2_kernel(Xf, Xf, t3, lambda1, lambda2, un_f, un_f, diag=True) \
                        +dt**2*a[0,0]**2*self.Lap2_kernel(Xf, Xf, t5, lambda1, lambda2, un_f, un_f, diag=True)
                        
                        
                        
                        
                return tf.concat((ku3u3,ku2u2,ku1u1,kf3f3, kf2f2, kf1f1),axis=0)




       
    def kernel_u_test(self, Xt, Xu, Xf, t1, t3, t5, a, b, c, lambda1, lambda2, un_u, un_f, un_t, dt):
        if self.kernel_type == 'SE':
            ku3u3 = self.kernel(Xt, Xu[2], t1)
            ku2u2 = self.kernel(Xt, Xu[1], t3)
            ku1u1 = self.kernel(Xt, Xu[0], t5)   

             

            ku3f3 = self.kernel(Xt, Xf, t1)
            
            ku2f3 = dt*b[1]*self.Lap1_kernel_prime(Xt, Xf, t3,lambda1, lambda2, un_t, un_f )
            ku2f2 = self.kernel(Xt, Xf, t3) + dt*a[1,1]*self.Lap1_kernel_prime(Xt,Xf,t3,lambda1, lambda2, un_t, un_f)
            ku2f1 = dt*a[0,1]*self.Lap1_kernel_prime(Xt, Xf, t3, lambda1, lambda2, un_t, un_f)
            ku1f3 = dt*b[0]*self.Lap1_kernel_prime(Xt, Xf, t5, lambda1, lambda2, un_t, un_f)
            ku1f2 = dt*a[1,0]*self.Lap1_kernel_prime(Xt, Xf, t5, lambda1, lambda2, un_t, un_f)
            ku1f1 = self.kernel(Xt, Xf, t5) + dt*a[0,0]*self.Lap1_kernel_prime(Xt, Xf, t5, lambda1, lambda2, un_t, un_f)
  


            zuu3 = tf.zeros((Xt.shape[0],Xu[2].shape[0]),dtype=tf.float64)                            
            zuu2 = tf.zeros((Xt.shape[0],Xu[1].shape[0]),dtype=tf.float64)        
            zuu1 = tf.zeros((Xt.shape[0],Xu[0].shape[0]),dtype=tf.float64)        

            zuf = tf.zeros((Xt.shape[0],Xf.shape[0]),dtype=tf.float64)        

            k1 = tf.concat( (ku3u3, zuu2, zuu1, ku3f3, zuf, zuf),axis=1)         
            k2 = tf.concat( (zuu3, ku2u2, zuu1, ku2f3, ku2f2, ku2f1),axis=1)         
            k3 = tf.concat( (zuu3, zuu2, ku1u1, ku1f3, ku1f2, ku1f1),axis=1)         
           
            k = tf.concat((k1,k2,k3),axis=0)                                
            return k


    def kernel_f_test(self, Xt, Xu, Xf, t1, t3, t5, a, b, c, lambda1, lambda2, un_u, un_f, un_t, dt):
 

        
        if self.kernel_type == 'SE':
            kf3f3 = self.kernel(Xt, Xf, t1)  \
                    + dt**2*b[0]**2*self.Lap2_kernel(Xt, Xf, t5, lambda1, lambda2, un_t, un_f) \
                    + dt**2*b[1]**2*self.Lap2_kernel(Xt, Xf, t3, lambda1, lambda2, un_t, un_f)


            kf2f2 = self.kernel(Xt, Xf, t3) \
                     + dt*a[1,1]*self.Lap1_kernel(Xt, Xf, t3, lambda1, lambda2, un_t, un_f) \
                    + dt*a[1,1]*self.Lap1_kernel_prime(Xt, Xf, t3, lambda1, lambda2, un_t, un_f)\
                    +dt**2*a[1,0]**2*self.Lap2_kernel(Xt, Xf, t5, lambda1, lambda2, un_t, un_f) \
                    +dt**2*a[1,1]**2*self.Lap2_kernel(Xt, Xf, t3, lambda1, lambda2, un_t, un_f)
            kf1f1 = self.kernel(Xt, Xf, t5) \
                    +dt*a[0,0]*self.Lap1_kernel(Xt, Xf, t5, lambda1, lambda2, un_t, un_f)\
                    +dt*a[0,0]*self.Lap1_kernel_prime(Xt, Xf, t5, lambda1, lambda2, un_t, un_f)\
                    +dt**2*a[0,1]**2*self.Lap2_kernel(Xt, Xf, t3, lambda1, lambda2, un_t, un_f) \
                    +dt**2*a[0,0]**2*self.Lap2_kernel(Xt, Xf, t5, lambda1, lambda2, un_t, un_f)
                        



                    
                    
            kf3u3 = self.kernel(Xt, Xu[2], t1)
            kf3u2 = dt*b[1]*self.Lap1_kernel(Xt, Xu[1], t3, lambda1, lambda2, un_t, un_u[1])
            kf2u2 = self.kernel(Xt, Xu[1], t3) + dt*a[1,1]*self.Lap1_kernel(Xt,Xu[1],t3,lambda1, lambda2, un_t, un_u[1])
            kf1u2 = dt*a[0,1]*self.Lap1_kernel(Xt, Xu[1], t3, lambda1, lambda2, un_t, un_u[1])
            kf3u1 = dt*b[0]*self.Lap1_kernel(Xt, Xu[0], t5, lambda1, lambda2, un_t, un_u[0])
            kf2u1 = dt*a[1,0]*self.Lap1_kernel(Xt, Xu[0], t5, lambda1, lambda2, un_t, un_u[0])
            kf1u1 = self.kernel(Xt, Xu[0], t5) + dt*a[0,0]*self.Lap1_kernel(Xt, Xu[0], t5, lambda1, lambda2, un_t, un_u[0])


            
            kf2f3 = dt*b[1]*self.Lap1_kernel_prime(Xt, Xf, t3,lambda1, lambda2, un_t, un_f) \
                   +dt**2*b[0]*a[1,0]*self.Lap2_kernel(Xt, Xf, t5, lambda1, lambda2, un_t, un_f) \
                   +dt**2*b[1]*a[1,1]*self.Lap2_kernel(Xt, Xf, t3, lambda1, lambda2, un_t, un_f)
            kf1f3 = dt*b[0]*self.Lap1_kernel_prime(Xt, Xf, t5, lambda1, lambda2, un_t, un_f) \
                    + dt**2*b[0]*a[0,0]*self.Lap2_kernel(Xt, Xf, t5, lambda1, lambda2, un_t, un_f) \
                    + dt**2*b[1]*a[0,1]*self.Lap2_kernel(Xt, Xf, t3, lambda1, lambda2, un_t, un_f)
            kf1f2 = dt*a[0,1]*self.Lap1_kernel(Xt, Xf, t3, lambda1, lambda2, un_t, un_f) \
                        +dt*a[1,0]*self.Lap1_kernel_prime(Xt, Xf, t5, lambda1, lambda2, un_t, un_f) \
                        + dt**2*a[1,0]*a[0,0]*self.Lap2_kernel(Xt, Xf, t5, lambda1, lambda2, un_t, un_f)\
                        + dt**2*a[1,1]*a[0,1]*self.Lap2_kernel(Xt, Xf, t3, lambda1, lambda2, un_t, un_f)

            kf3f2 = dt*b[1]*self.Lap1_kernel(Xt, Xf, t3,lambda1, lambda2, un_t, un_f) \
                   +dt**2*b[0]*a[1,0]*self.Lap2_kernel(Xt, Xf, t5, lambda1, lambda2, un_t, un_f) \
                   +dt**2*b[1]*a[1,1]*self.Lap2_kernel(Xt, Xf, t3, lambda1, lambda2, un_t, un_f)
            kf3f1 = dt*b[0]*self.Lap1_kernel(Xt, Xf, t5, lambda1, lambda2, un_t, un_f) \
                    + dt**2*b[0]*a[0,0]*self.Lap2_kernel(Xt, Xf, t5, lambda1, lambda2, un_t, un_f) \
                    + dt**2*b[1]*a[0,1]*self.Lap2_kernel(Xt, Xf, t3, lambda1, lambda2, un_t, un_f)
            kf2f1 = dt*a[0,1]*self.Lap1_kernel_prime(Xt, Xf, t3, lambda1, lambda2, un_t, un_f) \
                        +dt*a[1,0]*self.Lap1_kernel(Xt, Xf, t5, lambda1, lambda2, un_t, un_f) \
                        + dt**2*a[1,0]*a[0,0]*self.Lap2_kernel(Xt, Xf, t5, lambda1, lambda2, un_t, un_f)\
                        + dt**2*a[1,1]*a[0,1]*self.Lap2_kernel(Xt, Xf, t3, lambda1, lambda2, un_t, un_f)

              
            zfu = tf.zeros((Xt.shape[0],Xu[2].shape[0]),dtype=tf.float64)        

          
            k4 = tf.concat( (kf3u3, kf3u2, kf3u1, kf3f3, kf3f2,kf3f1),axis=1)         
            k5 = tf.concat( (zfu, kf2u2, kf2u1, kf2f3, kf2f2, kf2f1),axis=1)         
            k6 = tf.concat( (zfu, kf1u2, kf1u1, kf1f3, kf1f2, kf1f1),axis=1)         
            
          
            k = tf.concat((k4,k5,k6),axis=0)    


            return k
     
   

    def nlml(self,Xu,Xf,Yu,Yf,dt, hyp1, hyp3, hyp5, sig_n, lambda1, lambda2, un_u, un_f, kernel_type, jitter=1.0e-10): # negative logarithm marginal-likelihood


        Nu = Xu[0].shape[0]+Xu[1].shape[0]+Xu[2].shape[0]

        N = Nu +3 * Xf.shape[0]
        self.K0 = self.kernel_uf_train(Xu,Xf,hyp1,hyp3,hyp5,self.a, self.b, self.c, lambda1, lambda2, un_u, un_f, dt)
        K = self.K0 + (sig_n**2+jitter)*tf.eye(N,dtype=tf.float64)
         
        self.L = tf.cholesky(K)
        r = np.concatenate((Yu[0],Yu[1],Yu[2],Yf,Yf,Yf),axis=0)\
          - np.concatenate((np.zeros((Nu,1),dtype=np.float64), self.prior_mean_train[0], self.prior_mean_train[1], self.prior_mean_train[2]),axis=0)
        self.alpha = tf.cholesky_solve(self.L, r)
        self.sig2_tf = tf.matmul(r, self.alpha, transpose_a=True)/N
        return 0.5 * N * tf.log(2.0*np.pi*self.sig2_tf)\
                +tf.reduce_sum(tf.log(tf.diag_part(self.L))) \
                + N/2.0    



                                               
    def training(self, optimizer = 'Adam', num_iter=10001, learning_rate = 5.0e-4, jitter = 1.0e-10):    

        tf.reset_default_graph()

        self.hyp1 = tf.exp(tf.Variable(0.0,dtype=np.float64))   
        self.hyp3 = tf.exp(tf.Variable(0.0,dtype=np.float64))
        self.hyp5 = tf.exp(tf.Variable(0.0,dtype=np.float64))

        self.lambda1 = tf.exp(tf.Variable(np.log(0.5),dtype=np.float64))
        self.lambda2 = tf.exp(tf.Variable(np.log(0.5),dtype=np.float64))

        

        if self.is_noise:
            self.sig_n = tf.exp(tf.Variable(np.log(1.0e-4),dtype=tf.float64))
        else:
            self.sig_n = tf.Variable(0.0,dtype=tf.float64, trainable=False)

   
        self.num_iter = num_iter
        self.jitter = jitter
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        
        
        Nu1 = self.xu_train[0].shape[0]
        Nu2 = self.xu_train[1].shape[0]
        Nu3 = self.xu_train[2].shape[0]

        Nu = Nu1+Nu2+Nu3
        Nf = self.xf_train.shape[0]        
        Nt = self.xf_test.shape[0]

        nlml_tf \
             = self.nlml(self.xu_train,self.xf_train, self.yu_train, self.yf_train, self.dt, self.hyp1, self.hyp3, self.hyp5, self.sig_n, self.lambda1, self.lambda2, self.un_u, self.un_f, self.kernel_type, self.jitter)
        self.sign_var = self.sig2_tf * self.sig_n**2


        self.K_train = self.kernel_uf_train(self.xu_train, self.xf_train, self.hyp1, self.hyp3, self.hyp5, self.a, self.b, self.c, self.lambda1, self.lambda2, self.un_u, self.un_f, self.dt)
        self.m_train = np.concatenate((np.zeros((Nu,1),dtype=np.float64),self.prior_mean_train[0],self.prior_mean_train[1],self.prior_mean_train[2]),axis=0) \
                                       + tf.matmul(self.K_train,self.alpha)
        L1 =  tf.concat((tf.zeros((3*Nf,Nu),dtype=tf.float64),self.previous_cov_mat),axis=1) 
        L1 = tf.concat((tf.zeros((Nu,Nu+3*Nf),dtype=tf.float64),L1),axis=0) 
        V1 = tf.linalg.triangular_solve(self.L,tf.transpose(self.K_train))
        V2 = tf.cholesky_solve(self.L, tf.transpose(self.K_train))
        self.var_train = self.sig2_tf*(self.kernel_uf_train(self.xu_train, self.xf_train, self.hyp1, self.hyp3, self.hyp5, self.a, self.b, self.c, self.lambda1, self.lambda2, self.un_u, self.un_f, self.dt, diag=True)\
                                  - tf.reshape(tf.reduce_sum(V1*V1,axis=0),(-1,1)))
        self.var_train = self.var_train + tf.reshape( tf.diag_part(tf.matmul(tf.matmul(tf.transpose(V2),L1),V2)),(-1,1))

        self.var_train = tf.maximum(self.var_train,tf.zeros((Nu+3*Nf,1),dtype=tf.float64) )    

     

        k_test_u = self.kernel_u_test(self.xu_test[Nf:,:], self.xu_train, self.xf_train, self.hyp1, self.hyp3, self.hyp5, self.a, self.b, self.c, self.lambda1, self.lambda2, self.un_u, self.un_f, self.un_t, self.dt)
        self.m_test_u = tf.matmul(k_test_u,self.alpha)        
        V1_test_u = tf.linalg.triangular_solve(self.L,tf.transpose(k_test_u))
        V2_test_u = tf.cholesky_solve(self.L, tf.transpose(k_test_u))
        self.var_test_u = self.sig2_tf * (1.0 - tf.reshape(tf.reduce_sum(V1_test_u*V1_test_u,axis=0),(-1,1))) +self.sign_var                        
        self.var_test_u = self.var_test_u + tf.reshape( tf.diag_part(tf.matmul(tf.matmul(tf.transpose(V2_test_u),L1),V2_test_u)),(-1,1))

        self.var_test_u = tf.maximum(self.var_test_u,tf.zeros((3*Nt,1),dtype=tf.float64) )    
    

   

        k_test_u0 = self.kernel_u_test(self.xu_test[:Nf,:], self.xu_train, self.xf_train, self.hyp1, self.hyp3, self.hyp5,self.a, self.b, self.c, self.lambda1, self.lambda2, self.un_u, self.un_f, self.un_f,  self.dt)
        self.m_test_u0 = tf.matmul(k_test_u0[:Nf,:],self.alpha)        
        V1_test_u0 = tf.linalg.triangular_solve(self.L,tf.transpose(k_test_u0[:Nf,:]))
        V2_test_u0 = tf.cholesky_solve(self.L, tf.transpose(k_test_u0[:Nf,:]))
        self.var_test_u0 = self.sig2_tf * (self.kernel(self.xu_test[:Nf,:],self.xu_test[:Nf,:],self.hyp1,equal=True)\
                    - tf.matmul(tf.transpose(V1_test_u0),V1_test_u0)) + self.sign_var* tf.eye(Nf,dtype=tf.float64)
        self.var_test_u0 = self.var_test_u0 + tf.reshape( tf.diag_part(tf.matmul(tf.matmul(tf.transpose(V2_test_u0),L1),V2_test_u0)),(-1,1))
        self.var_test_u0 = tf.maximum(self.var_test_u0,tf.zeros((Nf,Nf),dtype=tf.float64) )    


        k_test_f = self.kernel_f_test(self.xf_test, self.xu_train, self.xf_train, self.hyp1,  self.hyp3,  self.hyp5,  self.a, self.b, self.c, self.lambda1, self.lambda2, self.un_u, self.un_f, self.un_t, self.dt)
        self.m_test_f = tf.concat((self.prior_mean_test[0],self.prior_mean_test[1],self.prior_mean_test[2]),axis=0)+tf.matmul(k_test_f,self.alpha)   
        V1_test_f = tf.linalg.triangular_solve(self.L,tf.transpose(k_test_f))
        V2_test_f = tf.cholesky_solve(self.L, tf.transpose(k_test_f))
        self.var_test_f = self.sig2_tf * (self.kernel_uf_train([self.xf_test,self.xf_test,self.xf_test], self.xf_test, self.hyp1,  self.hyp3, self.hyp5,self.a, self.b, self.c, self.lambda1, self.lambda2, [self.un_t,self.un_t,self.un_t], self.un_t,  self.dt,diag=True)[3*self.xf_test.shape[0]:,0:1] \
                   - tf.reshape(tf.reduce_sum(V1_test_f*V1_test_f,axis=0),(-1,1)))  + self.sign_var
        self.var_test_f = self.var_test_f + tf.reshape( tf.diag_part(tf.matmul(tf.matmul(tf.transpose(V2_test_f),L1),V2_test_f)),(-1,1))

        self.var_test_f = tf.maximum(self.var_test_f,tf.zeros((3*Nt,1),dtype=tf.float64) )            
        
       


        if optimizer == 'Adam':
            optimizer_Adam = tf.train.AdamOptimizer(learning_rate)
            train_op_Adam = optimizer_Adam.minimize(nlml_tf)   

            grad1 = tf.gradients(nlml_tf,self.hyp1)[0]
            grad2 = tf.gradients(nlml_tf,self.hyp3)[0]
            grad3 = tf.gradients(nlml_tf,self.hyp5)[0]
            gradn = tf.gradients(nlml_tf,self.sig_n)[0]      
            gradl1 = tf.gradients(nlml_tf,self.lambda1)[0]
            gradl2 = tf.gradients(nlml_tf,self.lambda2)[0]        
            std_train = tf.sqrt(self.var_train)
            std_test_u = tf.sqrt(self.var_test_u)
            std_test_f = tf.sqrt(self.var_test_f)
            std_signal = tf.sqrt(self.sig2_tf)
            std_noise = tf.sqrt(self.sign_var)    
            
        nlml_min = 1.0e16
        
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.num_iter):    
                sess.run(train_op_Adam)
                if i % 5000 == 0:
                    nlml_temp = sess.run(nlml_tf)
                    if nlml_temp < nlml_min:
                        nlml_min = nlml_temp
                        self.mm_train = sess.run(self.m_train)
                        self.ss_train = sess.run(std_train)
                        self.mm_test_u = sess.run(self.m_test_u)
                        self.ss_test_u = sess.run(std_test_u)
                        self.mm_test_f = sess.run(self.m_test_f)
                        self.ss_test_f = sess.run(std_test_f)
                        self.mm_test_u0 = sess.run(self.m_test_u0)
                        self.posterior_cov_mat = np.tile(sess.run(self.var_test_u0),(3,3))
                        lambda1_val, lambda2_val, nlml_val, hyp1_val, hyp3_val, hyp5_val, sig_f, sig_n, grad_f1, grad_f3, grad_f5, grad_n, grad_l1, grad_l2= \
                        sess.run([self.lambda1, self.lambda2, nlml_tf, self.hyp1, self.hyp3,  self.hyp5,  std_signal, \
                                    std_noise,grad1,grad2,grad3,gradn,\
                                    gradl1,gradl2])                    
                        
                        
                        print ('*************************\n')
                        print ('Iter: ', i, '  nlml =', nlml_min, '\n')
                        print ('nlml:   '   , nlml_val)
                        print ('hyp:   ' , [hyp1_val,hyp3_val, hyp5_val])
                        print ('signal std:   ', sig_f)
                        print ('noise std:   ',sig_n)
                        print('grads of nlml over hyp ', [grad_f1, grad_f3, grad_f5]) 
                        print('grads of nlml over lambda ', [grad_l1, grad_l2],'\n') 

                        print ('Estimated lambda: ', [lambda1_val, lambda2_val],'\n')
                        print ('grad of nlml over sig_n', grad_n)                        
                        
                        print ('Training_err_u3:', np.linalg.norm(self.mm_train[:Nu3,0:1]-self.yu_train[0],2)/np.linalg.norm(self.yu_train[0],2))
                        print ('Training_err_f3:', np.linalg.norm(self.mm_train[Nu:(Nu+Nf),0:1]-self.yf_train,2)/np.linalg.norm(self.yf_train,2))


                        print ('Training_err_u2:', np.linalg.norm(self.mm_train[Nu3:(Nu3+Nu2),0:1]-self.yu_train[1],2)/np.linalg.norm(self.yu_train[1],2))
                        print ('Training_err_f2:', np.linalg.norm(self.mm_train[(Nu+Nf):(Nu+2*Nf),0:1]-self.yf_train,2)/np.linalg.norm(self.yf_train,2))

                        print ('Training_err_u1:', np.linalg.norm(self.mm_train[(Nu3+Nu2):Nu,0:1]-self.yu_train[2],2)/np.linalg.norm(self.yu_train[2],2))
                        print ('Training_err_f1:', np.linalg.norm(self.mm_train[(Nu+2*Nf):(Nu+3*Nf),0:1]-self.yf_train,2)/np.linalg.norm(self.yf_train,2))


                        print ('Test_err_u0:', np.linalg.norm(self.mm_test_u0-self.yu_test[0],2)/np.linalg.norm(self.yu_test[0],2))           

                        print ('Test_err_u3:', np.linalg.norm(self.mm_test_u[:Nt,0:1]-self.yu_test[1],2)/np.linalg.norm(self.yu_test[1],2))           
                        print ('Test_err_f3:', np.linalg.norm(self.mm_test_f[:Nt,0:1]-self.yf_test,2)/np.linalg.norm(self.yf_test,2))           



                        print ('Test_err_u2:', np.linalg.norm(self.mm_test_u[Nt:(2*Nt),0:1]-self.yu_test[2],2)/np.linalg.norm(self.yu_test[2],2))           
                        print ('Test_err_f2:', np.linalg.norm(self.mm_test_f[Nt:(2*Nt),0:1]-self.yf_test,2)/np.linalg.norm(self.yf_test,2))           


                        print ('Test_err_u1:', np.linalg.norm(self.mm_test_u[(2*Nt):(3*Nt),0:1]-self.yu_test[3],2)/np.linalg.norm(self.yu_test[3],2))           
                        print ('Test_err_f1:', np.linalg.norm(self.mm_test_f[(2*Nt):(3*Nt),0:1]-self.yf_test,2)/np.linalg.norm(self.yf_test,2))           





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

    return np.zeros((x.shape[0],1),dtype=np.float64)



tt0 = time.time()

Nu = 70
Nf = 70
dt = 1.0e-1


init_time = 1.5
noise_rate = 0.0

xf_train = np.linspace(-8.0,8.0,Nf).reshape((-1,1))
xf_test = x_exa#np.linspace(-8.0,8.0,1000).reshape((-1,1))


xu_test = np.concatenate((xf_train,xf_test),axis=0)
yf_train = u_exact(xf_train,init_time, u_exa, t_exa, x_exa, 2)
previous_cov_mat = noise_rate*np.std(np.ndarray.flatten(yf_train))*np.eye(3*Nf,dtype=np.float64)
#yf_train = yf_train+np.linalg.cholesky(previous_cov_mat[:Nf,:Nf])@ np.random.randn(Nf,1)



#noise_f = np.linalg.cholesky(np.diag(noise_rate**2*np.ndarray.flatten(yf_train**2))+1.e-10*np.eye(Nf)) @ np.random.randn(Nf,1)
noise_f = noise_rate*np.std(np.ndarray.flatten(yf_train))*np.random.randn(Nf,1)
plt.plot(yf_train,'ro:')
yf_train = yf_train + noise_f
plt.plot(yf_train,'b*:')
plt.show()
xu_train = []
xu_train.append(np.array([[-8.0],[8.0]],dtype=np.float64))
xu_train.append(np.array([[-8.0],[8.0]],dtype=np.float64))
xu_train.append(np.linspace(-8.0,8.0,Nu).reshape((-1,1)))

Nt = xf_test.shape[0]
a = np.array([[0.25, 0.25-np.sqrt(3.0)/6.0], [0.25+np.sqrt(3.0)/6.0, 0.25]],dtype=np.float64)
b = np.array([0.5,0.5],dtype=np.float64)
c = np.array([0.5-np.sqrt(3.0)/6.0, 0.5+np.sqrt(3.0)/6.0],dtype=np.float64)


un_u = []
un_u.append(u_exact(xu_train[0],init_time,u_exa, t_exa, x_exa, 1))
un_u.append(u_exact(xu_train[1],init_time,u_exa, t_exa, x_exa, 1))
un_u.append(u_exact(xu_train[2],init_time,u_exa, t_exa, x_exa, 2))

un_f = yf_train
un_t = u_exact(xf_test,init_time,u_exa,t_exa,x_exa,2)


for k in np.arange(16,17):
    yu_train = []
    yu_test = []
    prior_mean_train = []
    prior_mean_test = []    
    yf_test = u_exact(xf_test,dt*(k-1),u_exa,t_exa,x_exa,2)
    
    
#    np.random.seed(seed=1234)
    
    
    yu_train.append(u_exact(xu_train[2],dt*k,u_exa, t_exa, x_exa, 2))
    yu_train.append(u_exact(xu_train[1],dt*(k-1)+c[1]*dt,u_exa, t_exa, x_exa, 1))
    yu_train.append(u_exact(xu_train[0],dt*(k-1)+c[0]*dt, u_exa, t_exa, x_exa, 1))     
    
    yu_train[0] = yu_train[0] +  noise_rate*np.std(np.ndarray.flatten(yu_train[0]))*np.random.randn(xu_train[2].shape[0],1)   
    yu_train[1] = yu_train[1] +  noise_rate*np.std(np.ndarray.flatten(yu_train[1]))*np.random.randn(xu_train[1].shape[0],1)   
    yu_train[2] = yu_train[2] +  noise_rate*np.std(np.ndarray.flatten(yu_train[2]))*np.random.randn(xu_train[0].shape[0],1)   
    
    yu_test.append(u_exact(xf_train,dt*k, u_exa, t_exa, x_exa, 2))
    yu_test.append(u_exact(xf_test,dt*k, u_exa, t_exa, x_exa, 2))
    yu_test.append(u_exact(xf_test,dt*(k-1)+c[1]*dt, u_exa, t_exa, x_exa, 2))
    yu_test.append(u_exact(xf_test,dt*(k-1)+c[0]*dt, u_exa, t_exa, x_exa, 2))
    
    
    
    dataset = {'xu_train': xu_train, 'yu_train': yu_train, \
               'xu_test':  xu_test,  'yu_test': yu_test, \
               'xf_train': xf_train, 'yf_train': yf_train,  \
               'xf_test': xf_test, 'yf_test': yf_test}
    
    
    print ('\n      t = '+ str(dt*k)+ '  *********************')
    
    
    prior_mean_train.append(-f_exact(xf_train, dt*(k-1)+c[0]*dt)*b[0]*dt-f_exact(xf_train,dt*(k-1)+c[1]*dt)*b[1]*dt)
    prior_mean_train.append(-f_exact(xf_train, dt*(k-1)+c[0]*dt)*a[1,0]*dt-f_exact(xf_train,dt*(k-1)+c[1]*dt)*a[1,1]*dt)
    prior_mean_train.append(-f_exact(xf_train, dt*(k-1)+c[0]*dt)*a[0,0]*dt-f_exact(xf_train,dt*(k-1)+c[1]*dt)*a[0,1]*dt)


    prior_mean_test.append(-f_exact(xf_test, dt*(k-1)+c[0]*dt)*b[0]*dt-f_exact(xf_test,dt*(k-1)+c[1]*dt)*b[1]*dt)
    prior_mean_test.append(-f_exact(xf_test, dt*(k-1)+c[0]*dt)*a[1,0]*dt-f_exact(xf_test,dt*(k-1)+c[1]*dt)*a[1,1]*dt)
    prior_mean_test.append(-f_exact(xf_test, dt*(k-1)+c[0]*dt)*a[0,0]*dt-f_exact(xf_test,dt*(k-1)+c[1]*dt)*a[0,1]*dt)

    
    
    
    GP_instance = one_GP()
    GP_instance.model(dataset, dt, prior_mean_train, prior_mean_test, previous_cov_mat, a, b, c, un_u, un_f, un_t,  is_noise=True)
    GP_instance.training(num_iter=5001,jitter=0.0)
    previous_cov_mat = GP_instance.posterior_cov_mat
    yf_train = GP_instance.mm_test_u0 

    un_u = yu_train[0]
    un_f = GP_instance.mm_test_u0
    un_test = GP_instance.mm_test_u[:Nt,0:1]
    


    del GP_instance

tt1 = time.time()

print ('CPU time ', tt1-tt0)             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             