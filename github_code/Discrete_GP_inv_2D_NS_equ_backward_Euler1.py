# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:28:19 2019

@author: gpang
"""





import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from SALib.sample import sobol_sequence
import scipy as sci
import scipy.io as sio


class one_GP:
    
    def __init__(self):
        pass
             
             
    def model(self, dataset, dt,previous_cov_mat, un_u, un_f, un_t, kernel_type = 'SE', is_noise = True):        
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

        self.previous_cov_mat=previous_cov_mat
        self.dim = self.xf_train.shape[1]
        self.is_noise = is_noise

      
        
    def ku1u1(self, X, Y, t1, t2, sigf1, diag=False):
        x1 = X[:,0:1]
        y1 = X[:,1:2]
        x2 = Y[:,0:1].T
        y2 = Y[:,1:2].T
        if diag==False:
           k = tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t2 + y1 - y2)*(t2 - y1 + y2)/t2**4 
           return sigf1**2*k
        else:
           k = tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t2 + y1 - y2)*(t2 - y1 + y2)/t2**4 
           return sigf1**2*tf.reshape(tf.diag_part(k),(-1,1))
       
        
    def ku1v1(self, X, Y, t1, t2, sigf1, diag=False):
        x1 = X[:,0:1]
        y1 = X[:,1:2]
        x2 = Y[:,0:1].T
        y2 = Y[:,1:2].T
        if diag==False:
           k = -(x1 - x2)*(y1 - y2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))/(t1**2*t2**2)
           return -sigf1**2*k
        else:
           k = -(x1 - x2)*(y1 - y2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))/(t1**2*t2**2)
           return -sigf1**2*tf.reshape(tf.diag_part(k),(-1,1))        


    def ku1u0(self, X, Y, t1, t2, sigf1,un_x, un_y, diag=False):
        
        x1 = X[:,0:1]
        y1 = X[:,1:2]
        x2 = Y[:,0:1].T
        y2 = Y[:,1:2].T     
        
        unx = np.ndarray.flatten(un_x[0])
        vnx = np.ndarray.flatten(un_x[1])

        uny = np.ndarray.flatten(un_y[0])
        vny = np.ndarray.flatten(un_y[1])

        unx = tf.diag(unx)
        vnx = tf.diag(vnx)
        
        uny = tf.diag(uny)     
        vny = tf.diag(vny)     
        
        k1  = tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t2 + y1 - y2)*(t2 - y1 + y2)/t2**4
        
        k2  = (x1 - x2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t2 + y1 - y2)*(t2 - y1 + y2)/(t1**2*t2**4)
        
        k3  = (y1 - y2)*(3*t2**2 - y1**2 + 2*y1*y2 - y2**2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))/t2**6
        
        k4  = -3*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(((t1 + x1 - x2)*(t1 - x1 + x2)*t2**6)/3 + (t1**4 - (y1 - y2)**2*t1**2/3 + (y1 - y2)**2*(x1 - x2)**2/3)*t2**4 - 2*t1**4*(y1 - y2)**2*t2**2 + t1**4*(y1 - y2)**4/3)/(t1**4*t2**8)
           
        if diag==False:
            k = k1 + self.lambda1 * self.dt * tf.matmul(k2, uny) \
              + self.lambda1*self.dt*tf.matmul(k3, vny) - self.lambda2*self.dt * k4
            return sigf1**2*k
        else:
            k = k1 + self.lambda1 * self.dt * tf.matmul(k2, uny) \
              + self.lambda1*self.dt*tf.matmul(k3, vny) - self.lambda2*self.dt * k4
            return sigf1**2*tf.reshape(tf.diag_part(k),(-1,1))             
 

    def ku1v0(self, X, Y, t1, t2, sigf1,un_x, un_y, diag=False):
        
        x1 = X[:,0:1]
        y1 = X[:,1:2]
        x2 = Y[:,0:1].T
        y2 = Y[:,1:2].T     
        
        unx = np.ndarray.flatten(un_x[0])
        vnx = np.ndarray.flatten(un_x[1])

        uny = np.ndarray.flatten(un_y[0])
        vny = np.ndarray.flatten(un_y[1])

        unx = tf.diag(unx)
        vnx = tf.diag(vnx)
        
        uny = tf.diag(uny)     
        vny = tf.diag(vny)        
        
        k1  = -(x1 - x2)*(y1 - y2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))/(t1**2*t2**2)
        
        k2  = (y1 - y2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t1 + x1 - x2)*(t1 - x1 + x2)/(t1**4*t2**2)
        
        k3  = (x1 - x2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t2 + y1 - y2)*(t2 - y1 + y2)/(t1**2*t2**4)
        
        k4  = 3*(y1 - y2)*((t2**2 - (y1 - y2)**2/3)*t1**4 + t1**2*t2**4 - (x1 - x2)**2*t2**4/3)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(x1 - x2)/(t1**6*t2**6)
       
        if diag==False:
            k = k1 + self.lambda1 * self.dt * tf.matmul(k2, uny) \
              + self.lambda1*self.dt*tf.matmul(k3, vny) - self.lambda2*self.dt * k4
            return -sigf1**2*k
        else:
            k = k1 + self.lambda1 * self.dt * tf.matmul(k2, uny) \
              + self.lambda1*self.dt*tf.matmul(k3, vny) - self.lambda2*self.dt * k4
            return -sigf1**2*tf.reshape(tf.diag_part(k),(-1,1))                

    def kv1v1(self, X, Y, t1, t2, sigf1,diag=False):
        x1 = X[:,0:1]
        y1 = X[:,1:2]
        x2 = Y[:,0:1].T
        y2 = Y[:,1:2].T
        if diag==False:
           k = tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t1 + x1 - x2)*(t1 - x1 + x2)/t1**4
           return sigf1**2*k
        else:
           k = tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t1 + x1 - x2)*(t1 - x1 + x2)/t1**4
           return sigf1**2*tf.reshape(tf.diag_part(k),(-1,1))    


    def kv1u0(self, X, Y, t1, t2, sigf1,un_x, un_y, diag=False):
        
        x1 = X[:,0:1]
        y1 = X[:,1:2]
        x2 = Y[:,0:1].T
        y2 = Y[:,1:2].T     
        
        unx = np.ndarray.flatten(un_x[0])
        vnx = np.ndarray.flatten(un_x[1])

        uny = np.ndarray.flatten(un_y[0])
        vny = np.ndarray.flatten(un_y[1])

        unx = tf.diag(unx)
        vnx = tf.diag(vnx)
        
        uny = tf.diag(uny)     
        vny = tf.diag(vny)    
        
        k1  = -(x1 - x2)*(y1 - y2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))/(t1**2*t2**2)
        
        k2  = (y1 - y2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t1 + x1 - x2)*(t1 - x1 + x2)/(t1**4*t2**2)
        
        k3  = (x1 - x2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t2 + y1 - y2)*(t2 - y1 + y2)/(t1**2*t2**4)
        
        k4  = 3*(y1 - y2)*((t2**2 - (y1 - y2)**2/3)*t1**4 + t1**2*t2**4 - (x1 - x2)**2*t2**4/3)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(x1 - x2)/(t1**6*t2**6)
        if diag==False:
            k = k1 + self.lambda1 * self.dt * tf.matmul(k2, uny) \
              + self.lambda1*self.dt*tf.matmul(k3, vny) - self.lambda2*self.dt * k4
            return -sigf1**2*k
        else:
            k = k1 + self.lambda1 * self.dt * tf.matmul(k2,uny) \
              + self.lambda1*self.dt*tf.matmul(k3, vny) - self.lambda2*self.dt * k4
            return -sigf1**2*tf.reshape(tf.diag_part(k),(-1,1))                


    def kv1v0(self, X, Y, t1, t2, sigf1,un_x, un_y, diag=False):
        
        x1 = X[:,0:1]
        y1 = X[:,1:2]
        x2 = Y[:,0:1].T
        y2 = Y[:,1:2].T     
        
        unx = np.ndarray.flatten(un_x[0])
        vnx = np.ndarray.flatten(un_x[1])

        uny = np.ndarray.flatten(un_y[0])
        vny = np.ndarray.flatten(un_y[1])

        unx = tf.diag(unx)
        vnx = tf.diag(vnx)
        
        uny = tf.diag(uny)     
        vny = tf.diag(vny)    
        
        
        k1  = tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t1 + x1 - x2)*(t1 - x1 + x2)/t1**4
         
        k2  = (3*t1**2 - x1**2 + 2*x1*x2 - x2**2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(x1 - x2)/t1**6
         
        k3  = (y1 - y2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t1 + x1 - x2)*(t1 - x1 + x2)/(t1**4*t2**2)
         
        k4  = -((t2 + y1 - y2)*(t2 - y1 + y2)*t1**6 + (3*t2**4 - (x1 - x2)**2*t2**2 + (y1 - y2)**2*(x1 - x2)**2)*t1**4 - 6*t2**4*(x1 - x2)**2*t1**2 + t2**4*(x1 - x2)**4)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))/(t1**8*t2**4)
 

        if diag==False:
            k = k1 + self.lambda1 * self.dt * tf.matmul(k2, uny) \
              + self.lambda1*self.dt*tf.matmul(k3, vny) - self.lambda2*self.dt * k4
            return sigf1**2*k
        else:
            k = k1 + self.lambda1 * self.dt * tf.matmul(k2, uny) \
              + self.lambda1*self.dt*tf.matmul(k3, vny) - self.lambda2*self.dt * k4
            return sigf1**2*tf.reshape(tf.diag_part(k),(-1,1))                




    def ku0u0(self, X, Y, t1, t2, s1, s2, sigf1,sigf2,un_x, un_y, diag=False):

#         sess1 = tf.Session()
#         sess1.run(tf.global_variables_initializer())

         x1 = X[:,0:1]
         y1 = X[:,1:2]
         x2 = Y[:,0:1].T
         y2 = Y[:,1:2].T     
        
         unx = np.ndarray.flatten(un_x[0])
         vnx = np.ndarray.flatten(un_x[1])

         uny = np.ndarray.flatten(un_y[0])
         vny = np.ndarray.flatten(un_y[1])

         unx = tf.diag(unx)
         vnx = tf.diag(vnx)
        
         uny = tf.diag(uny)     
         vny = tf.diag(vny)    

         k1 = tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t2 + y1 - y2)*(t2 - y1 + y2)/t2**4
         k2 = -(x1 - x2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t2 + y1 - y2)*(t2 - y1 + y2)/(t1**2*t2**4)
         k3 = -(y1 - y2)*(3*t2**2 - y1**2 + 2*y1*y2 - y2**2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))/t2**6
         k4 = -3*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(((t1 + x1 - x2)*(t1 - x1 + x2)*t2**6)/3 + (t1**4 - (y1 - y2)**2*t1**2/3 + (y1 - y2)**2*(x1 - x2)**2/3)*t2**4 - 2*t1**4*(y1 - y2)**2*t2**2 + t1**4*(y1 - y2)**4/3)/(t1**4*t2**8)
         k5 = (x1 - x2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t2 + y1 - y2)*(t2 - y1 + y2)/(t1**2*t2**4)
         k6 = tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t2 + y1 - y2)*(t2 - y1 + y2)*(t1 + x1 - x2)*(t1 - x1 + x2)/(t1**4*t2**4)
         k7 = -(y1 - y2)*(3*t2**2 - y1**2 + 2*y1*y2 - y2**2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(x1 - x2)/(t1**2*t2**6)
         k8 = -3*((t1**2 - (x1 - x2)**2/3)*t2**6 + (t1**4 - (y1 - y2)**2*t1**2 + (y1 - y2)**2*(x1 - x2)**2/3)*t2**4 - 2*t1**4*(y1 - y2)**2*t2**2 + t1**4*(y1 - y2)**4/3)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(x1 - x2)/(t1**6*t2**8)
         k9 = (y1 - y2)*(3*t2**2 - y1**2 + 2*y1*y2 - y2**2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))/t2**6
         k10 = -(y1 - y2)*(3*t2**2 - y1**2 + 2*y1*y2 - y2**2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(x1 - x2)/(t1**2*t2**6)
         k11 = tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(3*t2**4 - 6*t2**2*y1**2 + 12*t2**2*y1*y2 - 6*t2**2*y2**2 + y1**4 - 4*y1**3*y2 + 6*y1**2*y2**2 - 4*y1*y2**3 + y2**4)/t2**8
         k12 = -15*(y1 - y2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(((t1 + x1 - x2)*(t1 - x1 + x2)*t2**6)/5 + (-((t1 + x1 - x2)*(t1 - x1 + x2)*y1**2)/15 + (2*y2*(t1 + x1 - x2)*(t1 - x1 + x2)*y1)/15 - ((t1 + x1 - x2)*(t1 - x1 + x2)*y2**2)/15 + t1**4)*t2**4 - (2*t1**4*(y1 - y2)**2*t2**2)/3 + t1**4*(y1 - y2)**4/15)/(t1**4*t2**10)
         k13 = -3*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(((t1 + x1 - x2)*(t1 - x1 + x2)*t2**6)/3 + (t1**4 - (y1 - y2)**2*t1**2/3 + (y1 - y2)**2*(x1 - x2)**2/3)*t2**4 - 2*t1**4*(y1 - y2)**2*t2**2 + t1**4*(y1 - y2)**4/3)/(t1**4*t2**8)
         k14 = 3*((t1**2 - (x1 - x2)**2/3)*t2**6 + (t1**4 - (y1 - y2)**2*t1**2 + (y1 - y2)**2*(x1 - x2)**2/3)*t2**4 - 2*t1**4*(y1 - y2)**2*t2**2 + t1**4*(y1 - y2)**4/3)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(x1 - x2)/(t1**6*t2**8)
         k15 = 15*(y1 - y2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(((t1 + x1 - x2)*(t1 - x1 + x2)*t2**6)/5 + (-((t1 + x1 - x2)*(t1 - x1 + x2)*y1**2)/15 + (2*y2*(t1 + x1 - x2)*(t1 - x1 + x2)*y1)/15 - ((t1 + x1 - x2)*(t1 - x1 + x2)*y2**2)/15 + t1**4)*t2**4 - (2*t1**4*(y1 - y2)**2*t2**2)/3 + t1**4*(y1 - y2)**4/15)/(t1**4*t2**10)
         k16 = 15*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*((t1**4/5 - (2*(x1 - x2)**2*t1**2)/5 + (x1 - x2)**4/15)*t2**10 + ((2*t1**6)/5 + (-y1**2/5 + (2*y1*y2)/5 - y2**2/5 - (2*(x1 - x2)**2)/5)*t1**4 + (2*(y1 - y2)**2*(x1 - x2)**2*t1**2)/5 - (y1 - y2)**2*(x1 - x2)**4/15)*t2**8 + (t1**4 - (4*(y1 - y2)**2*t1**2)/5 + (4*(y1 - y2)**2*(x1 - x2)**2)/5)*t1**4*t2**6 - 3*(y1 - y2)**2*(t1**4 - (2*(y1 - y2)**2*t1**2)/45 + (2*(y1 - y2)**2*(x1 - x2)**2)/45)*t1**4*t2**4 + t1**8*(y1 - y2)**4*t2**2 - t1**8*(y1 - y2)**6/15)/(t1**8*t2**12)
      
         kpx2x1 = tf.exp((-(y1 - y2)**2*s1**2 - (x1 - x2)**2*s2**2)/(2*s1**2*s2**2))*(s1 + x1 - x2)*(s1 - x1 + x2)/s1**4          

         if diag==False:
            k = k1 + self.lambda1 * self.dt * tf.matmul(unx, k2) \
              + self.lambda1*self.dt*tf.matmul(vnx, k3) - self.lambda2*self.dt * k4 \
              +self.lambda1*self.dt*tf.matmul(k5, uny) + self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(unx, k6), uny) \
              +self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(vnx,k7),uny)-self.lambda2*self.dt**2*self.lambda1*tf.matmul(k8, uny) \
              +self.lambda1*self.dt*tf.matmul(k9, vny) + self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(unx, k10), vny) \
              +self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(vnx, k11),vny) - self.dt**2*self.lambda1*self.lambda2*tf.matmul(k12, vny)\
              -self.dt*self.lambda2*k13 - self.lambda1*self.dt**2*self.lambda2*tf.matmul(unx, k14) \
              -self.lambda1*self.lambda2*self.dt**2*tf.matmul(vnx, k15) + self.dt**2 * self.lambda2**2*k16
            k = sigf1**2 * k  + sigf2**2*self.dt**2 * kpx2x1

#            aaa = sess1.run(k)
#            print (np.max(np.abs(aaa-aaa.T)))
#            ssf=3

            return k
         else:
            k = k1 + self.lambda1 * self.dt * tf.matmul(unx, k2) \
              + self.lambda1*self.dt*tf.matmul(vnx, k3) - self.lambda2*self.dt * k4 \
              +self.lambda1*self.dt*tf.matmul(k5, uny) + self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(unx, k6), uny) \
              +self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(vnx,k7),uny)-self.lambda2*self.dt**2*self.lambda1*tf.matmul(k8, uny) \
              +self.lambda1*self.dt*tf.matmul(k9, vny) + self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(unx, k10), vny) \
              +self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(vnx, k11),vny) - self.dt**2*self.lambda1*self.lambda2*tf.matmul(k12, vny)\
              -self.dt*self.lambda2*k13 - self.lambda1*self.dt**2*self.lambda2*tf.matmul(unx, k14) \
              -self.lambda1*self.lambda2*self.dt**2*tf.matmul(vnx, k15) + self.dt**2 * self.lambda2**2*k16 
            k = sigf1**2*k+sigf2**2*self.dt**2 * kpx2x1



            return tf.reshape(tf.diag_part(k),(-1,1))           



    def ku0v0(self, X, Y, t1, t2, s1, s2, sigf1,sigf2,un_x, un_y, diag=False):
         x1 = X[:,0:1]
         y1 = X[:,1:2]
         x2 = Y[:,0:1].T
         y2 = Y[:,1:2].T     
        
         unx = np.ndarray.flatten(un_x[0])
         vnx = np.ndarray.flatten(un_x[1])

         uny = np.ndarray.flatten(un_y[0])
         vny = np.ndarray.flatten(un_y[1])

         unx = tf.diag(unx)
         vnx = tf.diag(vnx)
        
         uny = tf.diag(uny)     
         vny = tf.diag(vny)    

         k1 = -(x1 - x2)*(y1 - y2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))/(t1**2*t2**2)
         k2 = -(y1 - y2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t1 + x1 - x2)*(t1 - x1 + x2)/(t1**4*t2**2)
         k3 = -(x1 - x2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t2 + y1 - y2)*(t2 - y1 + y2)/(t1**2*t2**4)
         k4 = 3*(y1 - y2)*((t2**2 - (y1 - y2)**2/3)*t1**4 + t1**2*t2**4 - (x1 - x2)**2*t2**4/3)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(x1 - x2)/(t1**6*t2**6)
         k5 = (y1 - y2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t1 + x1 - x2)*(t1 - x1 + x2)/(t1**4*t2**2)
         k6 = -(y1 - y2)*(x1 - x2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(3*t1**2 - x1**2 + 2*x1*x2 - x2**2)/(t1**6*t2**2)
         k7 = tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t2 + y1 - y2)*(t2 - y1 + y2)*(t1 + x1 - x2)*(t1 - x1 + x2)/(t1**4*t2**4)
         k8 = -3*(y1 - y2)*((t2**2 - (y1 - y2)**2/3)*t1**6 + (t2**4 - (x1 - x2)**2*t2**2 + (y1 - y2)**2*(x1 - x2)**2/3)*t1**4 - 2*t2**4*(x1 - x2)**2*t1**2 + t2**4*(x1 - x2)**4/3)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))/(t1**8*t2**6)
         k9 = (x1 - x2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t2 + y1 - y2)*(t2 - y1 + y2)/(t1**2*t2**4)
         k10 = tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t2 + y1 - y2)*(t2 - y1 + y2)*(t1 + x1 - x2)*(t1 - x1 + x2)/(t1**4*t2**4)
         k11 = -(y1 - y2)*(3*t2**2 - y1**2 + 2*y1*y2 - y2**2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(x1 - x2)/(t1**2*t2**6)
         k12 = -3*((t1**2 - (x1 - x2)**2/3)*t2**6 + (t1**4 - (y1 - y2)**2*t1**2 + (y1 - y2)**2*(x1 - x2)**2/3)*t2**4 - 2*t1**4*(y1 - y2)**2*t2**2 + t1**4*(y1 - y2)**4/3)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(x1 - x2)/(t1**6*t2**8)
         k13 = 3*(y1 - y2)*((t2**2 - (y1 - y2)**2/3)*t1**4 + t1**2*t2**4 - (x1 - x2)**2*t2**4/3)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(x1 - x2)/(t1**6*t2**6)
         k14 = 3*(y1 - y2)*((t2**2 - (y1 - y2)**2/3)*t1**6 + (t2**4 - (x1 - x2)**2*t2**2 + (y1 - y2)**2*(x1 - x2)**2/3)*t1**4 - 2*t2**4*(x1 - x2)**2*t1**2 + t2**4*(x1 - x2)**4/3)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))/(t1**8*t2**6)
         k15 = 3*((t1**2 - (x1 - x2)**2/3)*t2**6 + (t1**4 - (y1 - y2)**2*t1**2 + (y1 - y2)**2*(x1 - x2)**2/3)*t2**4 - 2*t1**4*(y1 - y2)**2*t2**2 + t1**4*(y1 - y2)**4/3)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(x1 - x2)/(t1**6*t2**8)
         k16 = -15*(y1 - y2)*((t2**4 - (2*(y1 - y2)**2*t2**2)/3 + (y1 - y2)**4/15)*t1**8 + (6*t2**4*(t2**2 - (y1 - y2)**2/3)*t1**6)/5 + t2**4*(t2**4 - (2*(x1 - x2)**2*t2**2)/5 + (2*(y1 - y2)**2*(x1 - x2)**2)/15)*t1**4 - (2*(x1 - x2)**2*t1**2*t2**8)/3 + (x1 - x2)**4*t2**8/15)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(x1 - x2)/(t1**10*t2**10)
         
         kpy2x1 = -(y1 - y2)*(x1 - x2)*tf.exp((-(y1 - y2)**2*s1**2 - (x1 - x2)**2*s2**2)/(2*s1**2*s2**2))/(s1**2*s2**2)
         if diag==False:
            k = k1 + self.lambda1 * self.dt * tf.matmul(unx, k2) \
              + self.lambda1*self.dt*tf.matmul(vnx, k3) - self.lambda2*self.dt * k4 \
              +self.lambda1*self.dt*tf.matmul(k5, uny) + self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(unx, k6), uny) \
              +self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(vnx,k7),uny)-self.lambda2*self.dt**2*self.lambda1*tf.matmul(k8, uny) \
              +self.lambda1*self.dt*tf.matmul(k9, vny) + self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(unx, k10), vny) \
              +self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(vnx, k11),vny) - self.dt**2*self.lambda1*self.lambda2*tf.matmul(k12, vny)\
              -self.dt*self.lambda2*k13 - self.lambda1*self.dt**2*self.lambda2*tf.matmul(unx, k14) \
              -self.lambda1*self.lambda2*self.dt**2*tf.matmul(vnx, k15) + self.dt**2 * self.lambda2**2*k16
            k = -sigf1**2*k  +sigf2**2*self.dt**2 * kpy2x1
            return k
         else:
            k = k1 + self.lambda1 * self.dt * tf.matmul(unx, k2) \
              + self.lambda1*self.dt*tf.matmul(vnx, k3) - self.lambda2*self.dt * k4 \
              +self.lambda1*self.dt*tf.matmul(k5, uny) + self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(unx, k6), uny) \
              +self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(vnx,k7),uny)-self.lambda2*self.dt**2*self.lambda1*tf.matmul(k8, uny) \
              +self.lambda1*self.dt*tf.matmul(k9, vny) + self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(unx, k10), vny) \
              +self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(vnx, k11),vny) - self.dt**2*self.lambda1*self.lambda2*tf.matmul(k12, vny)\
              -self.dt*self.lambda2*k13 - self.lambda1*self.dt**2*self.lambda2*tf.matmul(unx, k14) \
              -self.lambda1*self.lambda2*self.dt**2*tf.matmul(vnx, k15) + self.dt**2 * self.lambda2**2*k16 
            k = -sigf1**2*k + sigf2**2*self.dt**2 * kpy2x1
            return tf.reshape(tf.diag_part(k),(-1,1))   



    def kv0v0(self, X, Y, t1, t2, s1, s2, sigf1,sigf2,un_x, un_y, diag=False):



         x1 = X[:,0:1]
         y1 = X[:,1:2]
         x2 = Y[:,0:1].T
         y2 = Y[:,1:2].T     

        
        
         unx = np.ndarray.flatten(un_x[0])
         vnx = np.ndarray.flatten(un_x[1])

         uny = np.ndarray.flatten(un_y[0])
         vny = np.ndarray.flatten(un_y[1])

         unx = tf.diag(unx)
         vnx = tf.diag(vnx)
        
         uny = tf.diag(uny)     
         vny = tf.diag(vny)    

         k1 = tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t1 + x1 - x2)*(t1 - x1 + x2)/t1**4
         k2 = -(3.0*t1**2 - x1**2 + 2*x1*x2 - x2**2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(x1 - x2)/t1**6
         k3 = -(y1 - y2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t1 + x1 - x2)*(t1 - x1 + x2)/(t1**4*t2**2)
         k4 = -((t2 + y1 - y2)*(t2 - y1 + y2)*t1**6 + (3*t2**4 - (x1 - x2)**2*t2**2 + (y1 - y2)**2*(x1 - x2)**2)*t1**4 - 6*t2**4*(x1 - x2)**2*t1**2 + t2**4*(x1 - x2)**4)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))/(t1**8*t2**4)
         k5 = (3.0*t1**2 - x1**2 + 2*x1*x2 - x2**2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(x1 - x2)/t1**6
         k6 = tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(3*t1**4 - 6*t1**2*x1**2 + 12*t1**2*x1*x2 - 6*t1**2*x2**2 + x1**4 - 4*x1**3*x2 + 6*x1**2*x2**2 - 4*x1*x2**3 + x2**4)/t1**8
         k7 = -(y1 - y2)*(x1 - x2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(3*t1**2 - x1**2 + 2*x1*x2 - x2**2)/(t1**6*t2**2)
         k8 = -3.0*((t2 + y1 - y2)*(t2 - y1 + y2)*t1**6 + (-((t2 + y1 - y2)*(t2 - y1 + y2)*x1**2)/3 + (2*x2*(t2 + y1 - y2)*(t2 - y1 + y2)*x1)/3 - ((t2 + y1 - y2)*(t2 - y1 + y2)*x2**2)/3 + 5*t2**4)*t1**4 - (10*t2**4*(x1 - x2)**2*t1**2)/3 + t2**4*(x1 - x2)**4/3)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(x1 - x2)/(t1**10*t2**4)
         k9 = (y1 - y2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t1 + x1 - x2)*(t1 - x1 + x2)/(t1**4*t2**2)
         k10 =-(y1 - y2)*(x1 - x2)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(3*t1**2 - x1**2 + 2*x1*x2 - x2**2)/(t1**6*t2**2)
         k11 = tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(t2 + y1 - y2)*(t2 - y1 + y2)*(t1 + x1 - x2)*(t1 - x1 + x2)/(t1**4*t2**4)
         k12 = -3.0*(y1 - y2)*((t2**2 - (y1 - y2)**2/3)*t1**6 + (t2**4 - (x1 - x2)**2*t2**2 + (y1 - y2)**2*(x1 - x2)**2/3)*t1**4 - 2*t2**4*(x1 - x2)**2*t1**2 + t2**4*(x1 - x2)**4/3.0)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))/(t1**8*t2**6)
         k13 = -((t2 + y1 - y2)*(t2 - y1 + y2)*t1**6 + (3*t2**4 - (x1 - x2)**2*t2**2 + (y1 - y2)**2*(x1 - x2)**2)*t1**4 - 6*t2**4*(x1 - x2)**2*t1**2 + t2**4*(x1 - x2)**4)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))/(t1**8*t2**4)
         k14 =3.0*((t2 + y1 - y2)*(t2 - y1 + y2)*t1**6 + (-((t2 + y1 - y2)*(t2 - y1 + y2)*x1**2)/3 + (2*x2*(t2 + y1 - y2)*(t2 - y1 + y2)*x1)/3 - ((t2 + y1 - y2)*(t2 - y1 + y2)*x2**2)/3 + 5*t2**4)*t1**4 - (10*t2**4*(x1 - x2)**2*t1**2)/3 + t2**4*(x1 - x2)**4/3)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))*(x1 - x2)/(t1**10*t2**4)
         k15 =3.0*(y1 - y2)*((t2**2 - (y1 - y2)**2/3)*t1**6 + (t2**4 - (x1 - x2)**2*t2**2 + (y1 - y2)**2*(x1 - x2)**2/3)*t1**4 - 2*t2**4*(x1 - x2)**2*t1**2 + t2**4*(x1 - x2)**4/3.0)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))/(t1**8*t2**6)
         k16 =3.0*((t2**4 - 2*(y1 - y2)**2*t2**2 + (y1 - y2)**4/3)*t1**10 + (2*t2**6 + (-x1**2 + 2*x1*x2 - x2**2 - 2*(y1 - y2)**2)*t2**4 + 2*(y1 - y2)**2*(x1 - x2)**2*t2**2 - (y1 - y2)**4*(x1 - x2)**2/3)*t1**8 + 5*t2**4*(t2**4 - (4*(x1 - x2)**2*t2**2)/5 + (4*(y1 - y2)**2*(x1 - x2)**2)/5)*t1**6 - 15*t2**4*(x1 - x2)**2*(t2**4 - (2*(x1 - x2)**2*t2**2)/45.0 + (2*(y1 - y2)**2*(x1 - x2)**2)/45)*t1**4 + 5*t2**8*(x1 - x2)**4*t1**2 - t2**8*(x1 - x2)**6/3)*tf.exp((-(y1 - y2)**2*t1**2 - (x1 - x2)**2*t2**2)/(2*t1**2*t2**2))/(t1**12*t2**8)

         
         kpy2y1 = tf.exp((-(y1 - y2)**2*s1**2 - (x1 - x2)**2*s2**2)/(2*s1**2*s2**2))*(s2 + y1 - y2)*(s2 - y1 + y2)/s2**4


         if diag==False:
            k = k1 + self.lambda1 * self.dt * tf.matmul(unx, k2) \
              + self.lambda1*self.dt*tf.matmul(vnx, k3) - self.lambda2*self.dt * k4 \
              +self.lambda1*self.dt*tf.matmul(k5, uny) + self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(unx, k6), uny) \
              +self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(vnx,k7),uny)-self.lambda2*self.dt**2*self.lambda1*tf.matmul(k8, uny) \
              +self.lambda1*self.dt*tf.matmul(k9, vny) + self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(unx, k10), vny) \
              +self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(vnx, k11),vny) - self.dt**2*self.lambda1*self.lambda2*tf.matmul(k12, vny)\
              -self.dt*self.lambda2*k13 - self.lambda1*self.dt**2*self.lambda2*tf.matmul(unx, k14) \
              -self.lambda1*self.lambda2*self.dt**2*tf.matmul(vnx, k15) + self.dt**2 * self.lambda2**2*k16 
            k=sigf1**2*k  + sigf2**2*self.dt**2 * kpy2y1

        
        

            return k        
         else:
            k = k1 + self.lambda1 * self.dt * tf.matmul (unx, k2) \
              + self.lambda1*self.dt*tf.matmul(vnx, k3) - self.lambda2*self.dt * k4 \
              +self.lambda1*self.dt*tf.matmul(k5, uny) + self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(unx, k6), uny) \
              +self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(vnx,k7),uny)-self.lambda2*self.dt**2*self.lambda1*tf.matmul(k8, uny) \
              +self.lambda1*self.dt*tf.matmul(k9, vny) + self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(unx, k10), vny) \
              +self.lambda1**2*self.dt**2*tf.matmul(tf.matmul(vnx, k11),vny) - self.dt**2*self.lambda1*self.lambda2*tf.matmul(k12, vny)\
              -self.dt*self.lambda2*k13 - self.lambda1*self.dt**2*self.lambda2*tf.matmul(unx, k14) \
              -self.lambda1*self.lambda2*self.dt**2*tf.matmul(vnx, k15) + self.dt**2 * self.lambda2**2*k16 
            k= sigf1**2*k+sigf2**2*self.dt**2 * kpy2y1
            return tf.reshape(tf.diag_part(k),(-1,1)) 



    def kernel_uf_train(self, Xu, Xf, t1, t2, s1, s2, lambda1, lambda2, sigf1, sigf2, un_u, un_f, dt, diag=False):

        if self.kernel_type == 'SE':
            if diag == False:
                ku1u1 = self.ku1u1(Xu, Xu, t1, t2,sigf1)
                ku1v1 = self.ku1v1(Xu,Xu,t1,t2,sigf1)
                ku1u0 = self.ku1u0(Xu,Xf,t1,t2,sigf1,un_u,un_f)                
                ku1v0 = self.ku1v0(Xu,Xf,t1,t2,sigf1,un_u,un_f)
                
                kv1v1 = self.kv1v1(Xu,Xu,t1,t2,sigf1)
                kv1u0 = self.kv1u0(Xu,Xf,t1,t2,sigf1,un_u,un_f)
                kv1v0 = self.kv1v0(Xu,Xf,t1,t2,sigf1,un_u,un_f)
                
                ku0u0 = self.ku0u0(Xf,Xf,t1,t2,s1,s2,sigf1,sigf2,un_f,un_f)
                ku0v0 = self.ku0v0(Xf,Xf,t1,t2,s1,s2,sigf1,sigf2,un_f,un_f)
                
                kv0v0 = self.kv0v0(Xf,Xf,t1,t2,s1,s2,sigf1,sigf2,un_f,un_f)
             
                
                k1 = tf.concat((ku1u1, ku1v1, ku1u0, ku1v0),axis=1)
                k2 = tf.concat((tf.transpose(ku1v1),kv1v1,kv1u0,kv1v0),axis=1)
                k3 = tf.concat((tf.transpose(ku1u0),tf.transpose(kv1u0), ku0u0, ku0v0),axis=1)
                k4 = tf.concat((tf.transpose(ku1v0),tf.transpose(kv1v0), tf.transpose(ku0v0),kv0v0),axis=1)
                
                k = tf.concat((k1,k2,k3,k4),axis=0)
 
                return k
            else:
                ku1u1 = self.ku1u1(Xu, Xu, t1, t2,sigf1)
                ku1v1 = self.ku1v1(Xu,Xu,t1,t2,sigf1)
                ku1u0 = self.ku1u0(Xu,Xf,t1,t2,sigf1,un_u,un_f)                
                ku1v0 = self.ku1v0(Xu,Xf,t1,t2,sigf1,un_u,un_f)
                
                kv1v1 = self.kv1v1(Xu,Xu,t1,t2,sigf1)
                kv1u0 = self.kv1u0(Xu,Xf,t1,t2,sigf1,un_u,un_f)
                kv1v0 = self.kv1v0(Xu,Xf,t1,t2,sigf1,un_u,un_f)
                
                ku0u0 = self.ku0u0(Xf,Xf,t1,t2,s1,s2,sigf1,sigf2,un_f,un_f)
                ku0v0 = self.ku0v0(Xf,Xf,t1,t2,s1,s2,sigf1,sigf2,un_f,un_f)
                
                kv0v0 = self.kv0v0(Xf,Xf,t1,t2,s1,s2,sigf1,sigf2,un_f,un_f)
             
                
                k1 = tf.concat((ku1u1, ku1v1, ku1u0, ku1v0),axis=1)
                k2 = tf.concat((tf.transpose(ku1v1),kv1v1,kv1u0,kv1v0),axis=1)
                k3 = tf.concat((tf.transpose(ku1u0),tf.transpose(kv1u0), ku0u0, ku0v0),axis=1)
                k4 = tf.concat((tf.transpose(ku1v0),tf.transpose(kv1v0), tf.transpose(ku0v0),kv0v0),axis=1)
                
                k = tf.concat((k1,k2,k3,k4),axis=0)
                        
                        
                        
                return tf.reshape(tf.diag_part(k),(-1,1))




       
    def kernel_u_test(self, Xt, Xu, Xf, t1, t2, s1, s2, lambda1, lambda2, sigf1, sigf2, un_f, un_t, dt):
        if self.kernel_type == 'SE':
            ku1u1 = self.ku1u1(Xt, Xu, t1, t2,sigf1)
            ku1v1 = self.ku1v1(Xt,Xu,t1,t2,sigf1)
            ku1u0 = self.ku1u0(Xt,Xf,t1,t2,sigf1,un_t,un_f)                
            ku1v0 = self.ku1v0(Xt,Xf,t1,t2,sigf1,un_t,un_f)
            
            kv1v1 = self.kv1v1(Xt,Xu,t1,t2,sigf1)
            kv1u0 = self.kv1u0(Xt,Xf,t1,t2,sigf1,un_t,un_f)
            kv1v0 = self.kv1v0(Xt,Xf,t1,t2,sigf1,un_t,un_f)
            
        
            ku1v1_T = self.ku1v1(Xu, Xt, t1, t2, sigf1)
            
            
            k1 = tf.concat((ku1u1, ku1v1, ku1u0, ku1v0),axis=1)
            k2 = tf.concat((tf.transpose(ku1v1_T),kv1v1,kv1u0,kv1v0),axis=1)
     
            k = tf.concat((k1,k2),axis=0)
          
  
            return k


    def kernel_f_test(self, Xt, Xu, Xf, t1, t2, s1, s2, lambda1, lambda2, sigf1, sigf2, un_f, un_t, dt):
 
    
        
        if self.kernel_type == 'SE':
         
           
            
            ku0u0 = self.ku0u0(Xt,Xf,t1,t2,s1,s2,sigf1,sigf2,un_t,un_f)
            ku0v0 = self.ku0v0(Xt,Xf,t1,t2,s1,s2,sigf1,sigf2,un_t,un_f)
            
            kv0v0 = self.kv0v0(Xt,Xf,t1,t2,s1,s2,sigf1,sigf2,un_t,un_f)
         
            ku1u0_T = self.ku1u0(Xu, Xt, t1, t2, sigf1,un_f, un_t)
            kv1u0_T = self.kv1u0(Xu, Xt, t1, t2, sigf1,un_f, un_t)
            ku1v0_T = self.ku1v0(Xu, Xt, t1, t2, sigf1,un_f, un_t)
            kv1v0_T = self.kv1v0(Xu, Xt, t1, t2, sigf1,un_f, un_t)
            ku0v0_T = self.ku0v0(Xf, Xt, t1, t2, s1, s2, sigf1,sigf2,un_f, un_t)
           
            k3 = tf.concat((tf.transpose(ku1u0_T),tf.transpose(kv1u0_T), ku0u0, ku0v0),axis=1)
            k4 = tf.concat((tf.transpose(ku1v0_T),tf.transpose(kv1v0_T), tf.transpose(ku0v0_T),kv0v0),axis=1)
            
            k = tf.concat((k3,k4),axis=0)
          
  
            return k            
            
            

    def nlml(self,Xu,Xf,Yu,Yf,dt, hyp1, hyp2, hyp3, hyp4, sig_n, lambda1, lambda2, sigf1, sigf2, un_u, un_f, kernel_type, jitter=1.0e-10): # negative logarithm marginal-likelihood


      

        N = 2 * Xu.shape[0] + 2 * Xf.shape[0]
        self.K0 = self.kernel_uf_train(Xu,Xf,hyp1,hyp2,hyp3, hyp4, lambda1, lambda2, sigf1, sigf2,  un_u, un_f, dt)
        K = self.K0 + (sig_n**2+jitter)*tf.eye(N,dtype=tf.float64)

        self.L = tf.cholesky(K)
        r = np.concatenate((Yu[0],Yu[1],Yf[0],Yf[1]),axis=0)
        self.alpha = tf.cholesky_solve(self.L, r)
        temp = tf.matmul(r, self.alpha, transpose_a=True)

        return 0.5 * N * np.log(2.0*np.pi) + temp /2.0 \
                +tf.reduce_sum(tf.log(tf.diag_part(self.L))) 
                


                                               
    def training(self, optimizer = 'Adam', num_iter=10001, learning_rate = 1.0e-3, jitter = 1.0e-15):    

        tf.reset_default_graph()

        self.hyp1 = tf.exp(tf.Variable(0.0,dtype=np.float64))   
        self.hyp2 = tf.exp(tf.Variable(0.0,dtype=np.float64))
        self.hyp3 = tf.exp(tf.Variable(0.0,dtype=np.float64))
        self.hyp4 = tf.exp(tf.Variable(0.0,dtype=np.float64))
        self.sigf1 = tf.exp(tf.Variable(0.0,dtype=np.float64))   
        self.sigf2 = tf.exp(tf.Variable(0.0,dtype=np.float64))   
        self.lambda1 = tf.exp(tf.Variable(0.0,dtype=np.float64))
        self.lambda2 = tf.exp(tf.Variable(0.0,dtype=np.float64))

        

        if self.is_noise:
            self.sig_n = tf.exp(tf.Variable(np.log(1.0e-4),dtype=tf.float64))
        else:
            self.sig_n = tf.Variable(0.0,dtype=tf.float64, trainable=False)


        self.num_iter = num_iter
        self.jitter = jitter
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        
        
       

        Nu = self.xu_train.shape[0]
        Nf = self.xf_train.shape[0]        
        Nt = self.xf_test.shape[0]

        nlml_tf \
             = self.nlml(self.xu_train,self.xf_train, self.yu_train, self.yf_train, self.dt, self.hyp1, self.hyp2, self.hyp3, self.hyp4, self.sig_n, self.lambda1, self.lambda2, self.sigf1, self.sigf2, self.un_u, self.un_f, self.kernel_type, self.jitter)
        


        self.K_train = self.kernel_uf_train(self.xu_train, self.xf_train, self.hyp1, self.hyp2, self.hyp3, self.hyp4,  self.lambda1, self.lambda2, self.sigf1, self.sigf2, self.un_u, self.un_f, self.dt)
        self.m_train = tf.matmul(self.K_train,self.alpha)
        L1 =  tf.concat((tf.zeros((2*Nf,2*Nu),dtype=tf.float64),self.previous_cov_mat),axis=1) 
        L1 = tf.concat((tf.zeros((2*Nu,2*Nu+2*Nf),dtype=tf.float64),L1),axis=0) 
        V1 = tf.linalg.triangular_solve(self.L,tf.transpose(self.K_train))
        V2 = tf.cholesky_solve(self.L, tf.transpose(self.K_train))

        self.var_train = self.kernel_uf_train(self.xu_train, self.xf_train, self.hyp1, self.hyp2, self.hyp3, self.hyp4, self.lambda1, self.lambda2, self.sigf1, self.sigf2, self.un_u, self.un_f, self.dt, diag=True)\
                                  - tf.reshape(tf.reduce_sum(V1*V1,axis=0),(-1,1))
        self.var_train = self.var_train + tf.reshape( tf.diag_part(tf.matmul(tf.matmul(tf.transpose(V2),L1),V2)),(-1,1))

        self.var_train = tf.maximum(self.var_train,tf.zeros((2*Nu+2*Nf,1),dtype=tf.float64) )    

     

        k_test_u = self.kernel_u_test(self.xu_test, self.xu_train, self.xf_train, self.hyp1, self.hyp2, self.hyp3, self.hyp4, self.lambda1, self.lambda2, self.sigf1, self.sigf2, self.un_f, self.un_t, self.dt)
        self.m_test_u = tf.matmul(k_test_u,self.alpha)        
        V1_test_u = tf.linalg.triangular_solve(self.L,tf.transpose(k_test_u))
        V2_test_u = tf.cholesky_solve(self.L, tf.transpose(k_test_u))
        self.var_test_u = self.kernel_uf_train(self.xu_test, self.xu_test, self.hyp1,  self.hyp2, self.hyp3, self.hyp4, self.lambda1, self.lambda2, self.sigf1, self.sigf2, self.un_u,  self.un_u,  self.dt,diag=True)[:2*Nt,0:1] - tf.reshape(tf.reduce_sum(V1_test_u*V1_test_u,axis=0),(-1,1)) +self.sig_n**2                       
        self.var_test_u = self.var_test_u + tf.reshape( tf.diag_part(tf.matmul(tf.matmul(tf.transpose(V2_test_u),L1),V2_test_u)),(-1,1))

        self.var_test_u = tf.maximum(self.var_test_u,tf.zeros((2*Nt,1),dtype=tf.float64) )    
    

   
        k_test_f = self.kernel_f_test(self.xf_test, self.xu_train, self.xf_train, self.hyp1,  self.hyp2,  self.hyp3, self.hyp4, self.lambda1, self.lambda2, self.sigf1, self.sigf2, self.un_f, self.un_t, self.dt)
        self.m_test_f = tf.matmul(k_test_f,self.alpha)   
        V1_test_f = tf.linalg.triangular_solve(self.L,tf.transpose(k_test_f))
        V2_test_f = tf.cholesky_solve(self.L, tf.transpose(k_test_f))
        self.var_test_f = self.kernel_uf_train(self.xf_test, self.xf_test, self.hyp1,  self.hyp2, self.hyp3, self.hyp4, self.lambda1, self.lambda2, self.sigf1, self.sigf2, self.un_t,  self.un_t,  self.dt,diag=True)[2*Nt:,0:1] \
                   - tf.reshape(tf.reduce_sum(V1_test_f*V1_test_f,axis=0),(-1,1))  + self.sig_n**2
        self.var_test_f = self.var_test_f + tf.reshape( tf.diag_part(tf.matmul(tf.matmul(tf.transpose(V2_test_f),L1),V2_test_f)),(-1,1))

        self.var_test_f = tf.maximum(self.var_test_f,tf.zeros((2*Nt,1),dtype=tf.float64) )            
        
       


        if optimizer == 'Adam':
            optimizer_Adam = tf.train.AdamOptimizer(learning_rate)
            train_op_Adam = optimizer_Adam.minimize(nlml_tf)   

            grad1 = tf.gradients(nlml_tf,self.hyp1)[0]
            grad2 = tf.gradients(nlml_tf,self.hyp2)[0]
            grad3 = tf.gradients(nlml_tf,self.hyp3)[0]
            grad4 = tf.gradients(nlml_tf,self.hyp4)[0]
            grads1 = tf.gradients(nlml_tf,self.sigf1)[0]
            grads2 = tf.gradients(nlml_tf,self.sigf2)[0]
               
            gradn = tf.gradients(nlml_tf,self.sig_n)[0]
            std_train = tf.sqrt(self.var_train)
            std_test_u = tf.sqrt(self.var_test_u)
            std_test_f = tf.sqrt(self.var_test_f)
            
            gradl1 = tf.gradients(nlml_tf,self.lambda1)[0]
            gradl2 = tf.gradients(nlml_tf,self.lambda2)[0]          
        
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
                      

                        lambda1_val, lambda2_val, nlml_val, hyp1_val, hyp2_val, hyp3_val, hyp4_val, sigf1_val, sigf2_val, sig_n, grad_f1, grad_f2, grad_f3, grad_f4, grad_s1, grad_s2, grad_n, grad_l1, grad_l2= \
                        sess.run([self.lambda1, self.lambda2, nlml_tf, self.hyp1, self.hyp2, self.hyp3, self.hyp4, self.sigf1, self.sigf2, \
                                    self.sig_n ,grad1,\
                                    grad2, grad3,\
                                    grad4,grads1, grads2,gradn,\
                                    gradl1,gradl2])                    
                        
                        
                        print ('*************************\n')
                        print ('Iter: ', i, '  nlml =', nlml_min, '\n')
                        print ('nlml:   '   , nlml_val)
                        print ('############### lambda: ', [lambda1_val, lambda2_val],'\n')
                        print ('hyp:   ' , [hyp1_val,hyp2_val, hyp3_val, hyp4_val])
                        print ('signal std:   ', [sigf1_val, sigf2_val])
                        print ('noise std:   ',sig_n)
                        print('grads of nlml over hyp ', [grad_f1, grad_f2, grad_f3, grad_f4]) 
                        print('grads of nlml over sigf ', [grad_s1, grad_s2]) 
                        
                        print('grads of nlml over lambda ', [grad_l1, grad_l2]) 
                        print ('grad of nlml over sig_n', grad_n)                        

                        
                        print ('Training_err_u1:', np.linalg.norm(self.mm_train[:Nu,0:1]-self.yu_train[0],2)/np.linalg.norm(self.yu_train[0],2))
                        print ('Training_err_u0:', np.linalg.norm(self.mm_train[(2*Nu):(2*Nu+Nf),0:1]-self.yf_train[0],2)/np.linalg.norm(self.yf_train[0],2))


                        print ('Training_err_v1:', np.linalg.norm(self.mm_train[Nu:(2*Nu),0:1]-self.yu_train[1],2)/np.linalg.norm(self.yu_train[1],2))
                        print ('Training_err_v0:', np.linalg.norm(self.mm_train[(2*Nu+Nf):(2*Nu+2*Nf),0:1]-self.yf_train[1],2)/np.linalg.norm(self.yf_train[1],2))

                        print ('Test_err_u1:', np.linalg.norm(self.mm_test_u[:Nt,0:1]-self.yu_test[0],2)/np.linalg.norm(self.yu_test[0],2))           
                        print ('Test_err_u0:', np.linalg.norm(self.mm_test_f[:Nt,0:1]-self.yf_test[0],2)/np.linalg.norm(self.yf_test[0],2))           



                        print ('Test_err_v1:', np.linalg.norm(self.mm_test_u[Nt:(2*Nt),0:1]-self.yu_test[1],2)/np.linalg.norm(self.yu_test[1],2))           
                        print ('Test_err_v0:', np.linalg.norm(self.mm_test_f[Nt:(2*Nt),0:1]-self.yf_test[1],2)/np.linalg.norm(self.yf_test[1],2))           




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


yu_train0 = [] 
yu_train0.append(u_exa[index_f,0:1,9])
yu_train0.append(u_exa[index_f,1:2,9])

xu_train = x_exa[index_u,:]

plt.contourf(np.linspace(1.0,7.5,66),np.linspace(-1.7,1.7,35), u_exa[:,0,9].reshape(66,35).T,100,cmap='jet')
plt.colorbar()
plt.plot(xf_train[:,0],xf_train[:,1],'bo',xu_train[:,0],xu_train[:,1],'ro')
plt.show()

xu_test = xf_test






previous_cov_mat = noise_rate*np.std(np.ndarray.flatten(yf_train[0]))*np.eye(2*Nf,dtype=np.float64)



noise_f = noise_rate*np.std(np.ndarray.flatten(yf_train[0]))*np.random.randn(Nf,1)


Nt = xf_test.shape[0]

un_u = [np.ones((yu_train0[0].shape[0],1))]*2
un_f = yf_train
un_t = yf_train


for k in np.arange(0,1):
    yu_train = []
    yu_test = []
   
    yf_test = yf_train
    
    
    np.random.seed(seed=1234)
    
    
    yu_train.append(u_exa[index_u,0:1,10])
    yu_train.append(u_exa[index_u,1:2,10])
    
   
    yu_test.append( u_exa[index_f,0:1,10])
    yu_test.append( u_exa[index_f,1:2,10])
    
    

    
    dataset = {'xu_train': xu_train, 'yu_train': yu_train, \
               'xu_test':  xu_test,  'yu_test': yu_test, \
               'xf_train': xf_train, 'yf_train': yf_train,  \
               'xf_test': xf_test, 'yf_test': yf_test}
    
    
    print ('\n      t = 0.18  *********************')
    

    
    GP_instance = one_GP()
    GP_instance.model(dataset, dt, previous_cov_mat, un_u, un_f, un_t,  is_noise=True)
    GP_instance.training(num_iter=20001,jitter=0.0)

    del GP_instance

tt1 = time.time()

print ('CPU time ', tt1-tt0)             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             