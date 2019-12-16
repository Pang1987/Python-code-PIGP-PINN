# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:32:11 2019

@author: gpang
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
#from SALib.sample import sobol_sequence
import scipy as sci
import scipy.io as sio
#

class one_GP:
    
    def __init__(self):
        pass
             
             
    def model(self, dt, a, b, c, Nu, Nf, kernel_type = 'SE', is_noise = True):        
        self.dt = dt
        self.a = a
        self.b = b
        self.c = c
        self.Nu = Nu
        self.Nf = Nf
        self.kernel_type = kernel_type
        
       
        self.is_noise = is_noise
       

      
    def u_exact(self, x,t, u_exa, t_exa, x_exa, dim):
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
    
    
    
    def f_exact(self, x,t):

        return np.zeros((x.shape[0],1),dtype=np.float64)             

               
    
    def kernel(self, X, Y, NX, NY, t1, equal=False, diag=False):
        if self.kernel_type == 'SE':
            Y = tf.reshape(Y,(1,-1))
            if diag == False:
                return tf.exp(-0.5* (X-Y)**2/t1**2)
            else:
                return tf.ones((NX,1),dtype=tf.float64)
        elif self.kernel_type == 'Matern1':
            dist = tf.sqrt(self.square_dist(X,Y,t1,equal))
            return (1.0+3.0**0.5*dist)*tf.exp(-3.0**0.5*dist)
        elif self.kernel_type == 'Matern2':
            dist = tf.sqrt(self.square_dist(X,Y,t1,equal))
            return (1.0+5.0**0.5*dist+5.0/3.0*dist**2)*tf.exp(-5.0**0.5*dist)            


    def kx(self, X, Y, NX, NY, t1, diag=False):
        Y = tf.reshape(Y,(1,-1))
        if diag == False:
            return (Y-X)/t1**2*tf.exp(-0.5*(X-Y)**2/t1**2)
        else:
            return tf.zeros((NX,1),dtype=tf.float64)
   
    def ky(self, X, Y, NX, NY, t1, diag=False):
        Y = tf.reshape(Y,(1,-1))
        if diag == False:
            return (X-Y)/t1**2*tf.exp(-0.5*(X-Y)**2/t1**2)
        else:
            return tf.zeros((NX,1),dtype=tf.float64)
        
    def kxx(self, X, Y, NX, NY, t1, diag=False):
        Y = tf.reshape(Y,(1,-1))
        if diag==False:
            return (-1.0/t1**2+(X-Y)**2/t1**4)*tf.exp(-0.5*(X-Y)**2/t1**2)
        else:
            return -1.0/t1**2 * tf.ones((NX,1),dtype=tf.float64)
        
        
    def kyy(self, X, Y, NX, NY, t1, diag=False):
        Y = tf.reshape(Y,(1,-1))
        if diag==False:
            return (-1.0/t1**2+(X-Y)**2/t1**4)*tf.exp(-0.5*(X-Y)**2/t1**2)
        else:
            return -1.0/t1**2 * tf.ones((NX,1),dtype=tf.float64)
        
        
    def kxy(self, X, Y, NX, NY, t1, diag=False):
        Y = tf.reshape(Y,(1,-1))
        if diag==False:
            return (1.0/t1**2-(X-Y)**2/t1**4)*tf.exp(-0.5*(X-Y)**2/t1**2)
        else:
            return 1.0/t1**2*tf.ones((NX,1),dtype=tf.float64)
        
    def kyyx(self, X, Y, NX, NY, t1, diag=False):
        Y = tf.reshape(Y,(1,-1))
        if diag==False:
            return (3*(X-Y)/t1**4-(X-Y)**3/t1**6)*tf.exp(-0.5*(X-Y)**2/t1**2)
        else:
            return tf.zeros((NX,1),dtype=tf.float64)
        
    def kyxx(self, X, Y, NX, NY, t1, diag=False):
        Y = tf.reshape(Y,(1,-1))
        if diag==False:
            return (3*(Y-X)/t1**4+(X-Y)**3/t1**6)*tf.exp(-0.5*(X-Y)**2/t1**2)
        else:
            return tf.zeros((NX,1),dtype=tf.float64)        
        
    def kxxyy(self, X, Y, NX, NY, t1, diag=False):
        Y = tf.reshape(Y,(1,-1))
        if diag==False:
            return (3.0/t1**4-6*(X-Y)**2/t1**6+(X-Y)**4/t1**8)*tf.exp(-0.5*(X-Y)**2/t1**2)
        else:
            return 3.0/t1**4*tf.ones((NX,1),dtype=tf.float64)
        
        

    def Lap2_kernel(self, X, Y, NX, NY, t1, lambda1, lambda2, un_x, un_y, equal=False, diag=False):
        unx = tf.reshape(un_x,[-1])
        uny = tf.reshape(un_y,[-1])
        unx = tf.diag(unx)
        uny = tf.diag(uny)
        if self.kernel_type == 'SE':
            if diag == False:
                k = lambda1**2*tf.matmul(tf.matmul(unx,self.kxy(X,Y,NX, NY, t1,diag)),uny)-lambda1*lambda2*tf.matmul(unx,self.kyyx(X,Y,NX, NY, t1,diag))\
                  -lambda1*lambda2*tf.matmul(self.kyxx(X,Y,NX, NY, t1,diag),uny)+lambda2**2*self.kxxyy(X,Y,NX, NY, t1,diag)
            else:
                k  = lambda1**2* un_x**2*self.kxy(X,Y,NX, NY, t1,diag)-lambda1*lambda2*un_x*self.kyyx(X,Y,NX, NY, t1,diag)\
                  -lambda1*lambda2*un_y*self.kyxx(X,Y,NX, NY, t1,diag)+lambda2**2*self.kxxyy(X,Y,NX, NY, t1,diag)
            return k

    def Lap1_kernel(self, X, Y, NX, NY, t1, lambda1, lambda2, un_x, un_y, equal=False, diag=False): ## -\Delta rather than \Delta

      if self.kernel_type == 'SE':
            unx = tf.reshape(un_x,[-1])
            uny = tf.reshape(un_y,[-1])
            unx = tf.diag(unx)
            uny = tf.diag(uny)
            if diag == False:
                 k = lambda1*tf.matmul(unx,self.kx(X,Y, NX, NY, t1,diag))-lambda2*self.kxx(X,Y,NX, NY, t1,diag)
               
            else:
                 k = lambda1*un_x*self.kx(X,Y,NX, NY, t1,diag)-lambda2*self.kxx(X,Y,NX, NY, t1,diag)
            return k 

    def Lap1_kernel_prime(self, X, Y, NX, NY, t1, lambda1, lambda2, un_x, un_y, equal=False, diag=False): ## -\Delta rather than \Delta

        if self.kernel_type == 'SE':
            unx = tf.reshape(un_x,[-1])
            uny = tf.reshape(un_y,[-1])
            unx = tf.diag(unx)
            uny = tf.diag(uny)  
            if diag == False:
                  k = lambda1*tf.matmul(self.ky(X,Y,NX, NY, t1,diag),uny)-lambda2*self.kyy(X,Y,NX, NY, t1,diag)
                                
            else:
                  k = lambda1*un_y*self.ky(X,Y,NX, NY, t1,diag)-lambda2*self.kyy(X,Y,NX, NY, t1,diag)

            return k



    def kernel_uf_train(self, Xu, Xf, Nu, Nf, t1, t3, t5, a, b, c, lambda1, lambda2, un_u, un_f, dt, diag=False):

        
        if self.kernel_type == 'SE':
            if diag == False:
                ku3u3 = self.kernel(Xu, Xu, Nu, Nu, t1, equal=True)
                ku2u2 = self.kernel(Xu, Xu, Nu, Nu, t3, equal=True)
                ku1u1 = self.kernel(Xu, Xu, Nu, Nu, t5, equal=True)
                kf3f3 = self.kernel(Xf, Xf, Nf, Nf, t1, equal=True)  \
                        + dt**2*b[0]**2*self.Lap2_kernel(Xf, Xf, Nf, Nf, t5, lambda1, lambda2, un_f, un_f, equal=True) \
                        + dt**2*b[1]**2*self.Lap2_kernel(Xf, Xf, Nf, Nf, t3, lambda1, lambda2, un_f, un_f, equal=True)
                kf2f2 = self.kernel(Xf, Xf, Nf, Nf, t3, equal=True) \
                        + dt*a[1,1]*self.Lap1_kernel(Xf, Xf, Nf, Nf, t3, lambda1, lambda2, un_f, un_f, equal=True) \
                        + dt*a[1,1]*self.Lap1_kernel_prime(Xf, Xf, Nf, Nf, t3, lambda1, lambda2, un_f, un_f, equal=True)\
                        +dt**2*a[1,0]**2*self.Lap2_kernel(Xf, Xf, Nf, Nf, t5, lambda1, lambda2, un_f, un_f, equal=True) \
                        +dt**2*a[1,1]**2*self.Lap2_kernel(Xf, Xf, Nf, Nf, t3, lambda1, lambda2, un_f, un_f, equal=True)
                kf1f1 = self.kernel(Xf, Xf, Nf, Nf, t5, equal=True) \
                        +dt*a[0,0]*self.Lap1_kernel(Xf, Xf, Nf, Nf, t5, lambda1, lambda2, un_f, un_f, equal=True)\
                        +dt*a[0,0]*self.Lap1_kernel_prime(Xf, Xf, Nf, Nf, t5, lambda1, lambda2, un_f, un_f, equal=True)\
                        +dt**2*a[0,1]**2*self.Lap2_kernel(Xf, Xf, Nf, Nf, t3, lambda1, lambda2, un_f, un_f, equal=True) \
                        +dt**2*a[0,0]**2*self.Lap2_kernel(Xf, Xf, Nf, Nf, t5, lambda1, lambda2, un_f, un_f, equal=True)
                        
                        
                kf3u3 = self.kernel(Xf, Xu, Nf, Nu, t1)
                kf3u2 = dt*b[1]*self.Lap1_kernel(Xf, Xu, Nf, Nu, t3, lambda1, lambda2, un_f, un_u)
                kf2u2 = self.kernel(Xf, Xu, Nf, Nu, t3) + dt*a[1,1]*self.Lap1_kernel(Xf,Xu,Nf, Nu, t3,lambda1, lambda2, un_f, un_u)
                kf1u2 = dt*a[0,1]*self.Lap1_kernel(Xf, Xu, Nf, Nu, t3, lambda1, lambda2, un_f, un_u)
                kf3u1 = dt*b[0]*self.Lap1_kernel(Xf, Xu,Nf,Nu, t5, lambda1, lambda2, un_f, un_u)
                kf2u1 = dt*a[1,0]*self.Lap1_kernel(Xf, Xu, Nf,Nu,t5, lambda1, lambda2, un_f, un_u)
                kf1u1 = self.kernel(Xf, Xu, Nf,Nu,t5) + dt*a[0,0]*self.Lap1_kernel(Xf, Xu, Nf,Nu,t5, lambda1, lambda2, un_f, un_u)
                kf2f3 = dt*b[1]*self.Lap1_kernel_prime(Xf, Xf, Nf,Nf,t3, lambda1, lambda2, un_f, un_f) \
                       +dt**2*b[0]*a[1,0]*self.Lap2_kernel(Xf, Xf, Nf,Nf,t5, lambda1, lambda2, un_f, un_f) \
                       +dt**2*b[1]*a[1,1]*self.Lap2_kernel(Xf, Xf, Nf,Nf,t3, lambda1, lambda2, un_f, un_f)
                kf1f3 = dt*b[0]*self.Lap1_kernel_prime(Xf, Xf, Nf,Nf,t5, lambda1, lambda2, un_f, un_f) \
                        + dt**2*b[0]*a[0,0]*self.Lap2_kernel(Xf, Xf,Nf,Nf, t5, lambda1, lambda2, un_f, un_f) \
                        + dt**2*b[1]*a[0,1]*self.Lap2_kernel(Xf, Xf, Nf,Nf,t3, lambda1, lambda2, un_f, un_f)
                kf1f2 = dt*a[0,1]*self.Lap1_kernel(Xf, Xf, Nf,Nf,t3, lambda1, lambda2, un_f, un_f) \
                        +dt*a[1,0]*self.Lap1_kernel_prime(Xf, Xf, Nf,Nf,t5, lambda1, lambda2, un_f, un_f) \
                        + dt**2*a[1,0]*a[0,0]*self.Lap2_kernel(Xf, Xf, Nf,Nf,t5, lambda1, lambda2, un_f, un_f)\
                        + dt**2*a[1,1]*a[0,1]*self.Lap2_kernel(Xf, Xf, Nf, Nf, t3, lambda1, lambda2, un_f, un_f)
                        
                zuu = tf.zeros((Nu,Nu),dtype=tf.float64)        
                zuf = tf.zeros((Nu,Nf),dtype=tf.float64)        
                zfu = tf.zeros((Nf,Nu),dtype=tf.float64)        

                k1 = tf.concat( (ku3u3, zuu, zuu, tf.transpose(kf3u3), zuf, zuf),axis=1)         
                k2 = tf.concat( (zuu, ku2u2, zuu, tf.transpose(kf3u2), tf.transpose(kf2u2), tf.transpose(kf1u2)),axis=1)         
                k3 = tf.concat( (zuu, zuu, ku1u1, tf.transpose(kf3u1), tf.transpose(kf2u1), tf.transpose(kf1u1)),axis=1)         
                k4 = tf.concat( (kf3u3, kf3u2, kf3u1, kf3f3, tf.transpose(kf2f3), tf.transpose(kf1f3)),axis=1)         
                k5 = tf.concat( (zfu, kf2u2, kf2u1, kf2f3, kf2f2, tf.transpose(kf1f2)),axis=1)         
                k6 = tf.concat( (zfu, kf1u2, kf1u1, kf1f3, kf1f2, kf1f1),axis=1)         
                
              
                k = tf.concat((k1,k2,k3,k4,k5,k6),axis=0)


                             
                return k
            else:
                ku3u3 = self.kernel(Xu, Xu, Nu,Nu,t1, diag=True)
                ku2u2 = self.kernel(Xu, Xu, Nu,Nu,t3, diag=True)
                ku1u1 = self.kernel(Xu, Xu, Nu,Nu,t5, diag=True)
                kf3f3 = self.kernel(Xf, Xf, Nf,Nf,t1, diag=True)  \
                        + dt**2*b[0]**2*self.Lap2_kernel(Xf, Xf, Nf,Nf,t5, lambda1, lambda2, un_f, un_f, diag=True) \
                        + dt**2*b[1]**2*self.Lap2_kernel(Xf, Xf, Nf,Nf,t3, lambda1, lambda2, un_f, un_f, diag=True)
                kf2f2 = self.kernel(Xf, Xf, Nf,Nf,t3,  diag=True) \
                        + 2.0*dt*a[1,1]*self.Lap1_kernel(Xf, Xf, Nf,Nf,t3, lambda1, lambda2, un_f, un_f, diag=True) \
                        +dt**2*a[1,0]**2*self.Lap2_kernel(Xf, Xf, Nf,Nf,t5, lambda1, lambda2, un_f, un_f, diag=True) \
                        +dt**2*a[1,1]**2*self.Lap2_kernel(Xf, Xf, Nf,Nf,t3, lambda1, lambda2, un_f, un_f, diag=True)
                kf1f1 = self.kernel(Xf, Xf, Nf,Nf,t5, diag=True) \
                        +2.0*dt*a[0,0]*self.Lap1_kernel(Xf, Xf, Nf,Nf,t5, lambda1, lambda2, un_f, un_f, diag=True)\
                        +dt**2*a[0,1]**2*self.Lap2_kernel(Xf, Xf, Nf,Nf,t3, lambda1, lambda2, un_f, un_f, diag=True) \
                        +dt**2*a[0,0]**2*self.Lap2_kernel(Xf, Xf, Nf,Nf,t5, lambda1, lambda2, un_f, un_f, diag=True)
                        
                        
                        
                        
                return tf.concat((ku3u3,ku2u2,ku1u1,kf3f3, kf2f2, kf1f1),axis=0)




       
    def kernel_u_test(self,  Xt, Xu, Xf, Nt, Nu,Nf,t1, t3, t5, a, b, c, lambda1, lambda2, un_u, un_f, un_t, dt):



        if self.kernel_type == 'SE':
            ku3u3 = self.kernel(Xt, Xu, Nt,Nu,t1)
            ku2u2 = self.kernel(Xt, Xu, Nt,Nu,t3)
            ku1u1 = self.kernel(Xt, Xu, Nt,Nu,t5)   

             
            ku3f3 = self.kernel(Xt, Xf, Nt,Nf,t1)
            
            ku2f3 = dt*b[1]*self.Lap1_kernel_prime(Xt, Xf, Nt,Nf,t3,lambda1, lambda2, un_t, un_f )
            ku2f2 = self.kernel(Xt, Xf, Nt,Nf,t3) + dt*a[1,1]*self.Lap1_kernel_prime(Xt,Xf,Nt,Nf,t3,lambda1, lambda2, un_t, un_f)
            ku2f1 = dt*a[0,1]*self.Lap1_kernel_prime(Xt, Xf, Nt,Nf,t3, lambda1, lambda2, un_t, un_f)
            ku1f3 = dt*b[0]*self.Lap1_kernel_prime(Xt, Xf, Nt,Nf,t5, lambda1, lambda2, un_t, un_f)
            ku1f2 = dt*a[1,0]*self.Lap1_kernel_prime(Xt, Xf, Nt,Nf,t5, lambda1, lambda2, un_t, un_f)
            ku1f1 = self.kernel(Xt, Xf, Nt,Nf, t5) + dt*a[0,0]*self.Lap1_kernel_prime(Xt, Xf, Nt,Nf,t5, lambda1, lambda2, un_t, un_f)
  


                    
            zuu = tf.zeros((Nt,Nu),dtype=tf.float64)        
            zuf = tf.zeros((Nt,Nf),dtype=tf.float64)        
#            zfu = tf.zeros((Xf.shape[0],Xt.shape[0]),dtype=tf.float64)        

            k1 = tf.concat( (ku3u3, zuu, zuu, ku3f3, zuf, zuf),axis=1)         
            k2 = tf.concat( (zuu, ku2u2, zuu, ku2f3, ku2f2, ku2f1),axis=1)         
            k3 = tf.concat( (zuu, zuu, ku1u1, ku1f3, ku1f2, ku1f1),axis=1)         
           
            k = tf.concat((k1,k2,k3),axis=0)                                
            return k


    def kernel_f_test(self, Xt, Xu, Xf, Nt,Nu,Nf,t1, t3, t5, a, b, c, lambda1, lambda2, un_u, un_f, un_t, dt):
 
#        sess1 = tf.Session()
#        sess1.run(tf.global_variables_initializer())   
#        
        
        if self.kernel_type == 'SE':
            kf3f3 = self.kernel(Xt, Xf, Nt,Nf,t1)  \
                    + dt**2*b[0]**2*self.Lap2_kernel(Xt, Xf, Nt,Nf,t5, lambda1, lambda2, un_t, un_f) \
                    + dt**2*b[1]**2*self.Lap2_kernel(Xt, Xf, Nt,Nf,t3, lambda1, lambda2, un_t, un_f)

            kf2f2 = self.kernel(Xt, Xf, Nt,Nf,t3) \
                     + dt*a[1,1]*self.Lap1_kernel(Xt, Xf, Nt,Nf,t3, lambda1, lambda2, un_t, un_f) \
                    + dt*a[1,1]*self.Lap1_kernel_prime(Xt, Xf, Nt,Nf, t3, lambda1, lambda2, un_t, un_f)\
                    +dt**2*a[1,0]**2*self.Lap2_kernel(Xt, Xf, Nt,Nf,t5, lambda1, lambda2, un_t, un_f) \
                    +dt**2*a[1,1]**2*self.Lap2_kernel(Xt, Xf, Nt,Nf,t3, lambda1, lambda2, un_t, un_f)
            kf1f1 = self.kernel(Xt, Xf, Nt, Nf, t5) \
                    +dt*a[0,0]*self.Lap1_kernel(Xt, Xf, Nt,Nf,t5, lambda1, lambda2, un_t, un_f)\
                    +dt*a[0,0]*self.Lap1_kernel_prime(Xt, Xf, Nt,Nf,t5, lambda1, lambda2, un_t, un_f)\
                    +dt**2*a[0,1]**2*self.Lap2_kernel(Xt, Xf, Nt,Nf,t3, lambda1, lambda2, un_t, un_f) \
                    +dt**2*a[0,0]**2*self.Lap2_kernel(Xt, Xf, Nt,Nf,t5, lambda1, lambda2, un_t, un_f)
                        



                    
                    
            kf3u3 = self.kernel(Xt, Xu, Nt,Nu,t1)
            kf3u2 = dt*b[1]*self.Lap1_kernel(Xt, Xu, Nt,Nu,t3, lambda1, lambda2, un_t, un_u)
            kf2u2 = self.kernel(Xt, Xu, Nt,Nu, t3) + dt*a[1,1]*self.Lap1_kernel(Xt,Xu,Nt,Nu,t3,lambda1, lambda2, un_t, un_u)
            kf1u2 = dt*a[0,1]*self.Lap1_kernel(Xt, Xu, Nt,Nu, t3, lambda1, lambda2, un_t, un_u)
            kf3u1 = dt*b[0]*self.Lap1_kernel(Xt, Xu, Nt,Nu,t5, lambda1, lambda2, un_t, un_u)
            kf2u1 = dt*a[1,0]*self.Lap1_kernel(Xt, Xu, Nt,Nu,t5, lambda1, lambda2, un_t, un_u)
            kf1u1 = self.kernel(Xt, Xu, Nt, Nu, t5) + dt*a[0,0]*self.Lap1_kernel(Xt, Xu, Nt, Nu, t5, lambda1, lambda2, un_t, un_u)


            
            kf2f3 = dt*b[1]*self.Lap1_kernel_prime(Xt, Xf, Nt, Nf, t3,lambda1, lambda2, un_t, un_f) \
                   +dt**2*b[0]*a[1,0]*self.Lap2_kernel(Xt, Xf, Nt,Nf,t5, lambda1, lambda2, un_t, un_f) \
                   +dt**2*b[1]*a[1,1]*self.Lap2_kernel(Xt, Xf, Nt,Nf,t3, lambda1, lambda2, un_t, un_f)
            kf1f3 = dt*b[0]*self.Lap1_kernel_prime(Xt, Xf, Nt,Nf,t5, lambda1, lambda2, un_t, un_f) \
                    + dt**2*b[0]*a[0,0]*self.Lap2_kernel(Xt, Xf, Nt,Nf,t5, lambda1, lambda2, un_t, un_f) \
                    + dt**2*b[1]*a[0,1]*self.Lap2_kernel(Xt, Xf, Nt,Nf,t3, lambda1, lambda2, un_t, un_f)
            kf1f2 = dt*a[0,1]*self.Lap1_kernel(Xt, Xf, Nt,Nf,t3, lambda1, lambda2, un_t, un_f) \
                        +dt*a[1,0]*self.Lap1_kernel_prime(Xt, Xf, Nt,Nf,t5, lambda1, lambda2, un_t, un_f) \
                        + dt**2*a[1,0]*a[0,0]*self.Lap2_kernel(Xt, Xf, Nt,Nf,t5, lambda1, lambda2, un_t, un_f)\
                        + dt**2*a[1,1]*a[0,1]*self.Lap2_kernel(Xt, Xf, Nt,Nf,t3, lambda1, lambda2, un_t, un_f)

            kf3f2 = dt*b[1]*self.Lap1_kernel(Xt, Xf, Nt,Nf,t3,lambda1, lambda2, un_t, un_f) \
                   +dt**2*b[0]*a[1,0]*self.Lap2_kernel(Xt, Xf, Nt,Nf,t5, lambda1, lambda2, un_t, un_f) \
                   +dt**2*b[1]*a[1,1]*self.Lap2_kernel(Xt, Xf, Nt,Nf,t3, lambda1, lambda2, un_t, un_f)
            kf3f1 = dt*b[0]*self.Lap1_kernel(Xt, Xf, Nt,Nf,t5, lambda1, lambda2, un_t, un_f) \
                    + dt**2*b[0]*a[0,0]*self.Lap2_kernel(Xt, Xf, Nt,Nf,t5, lambda1, lambda2, un_t, un_f) \
                    + dt**2*b[1]*a[0,1]*self.Lap2_kernel(Xt, Xf, Nt,Nf,t3, lambda1, lambda2, un_t, un_f)
            kf2f1 = dt*a[0,1]*self.Lap1_kernel_prime(Xt, Xf, Nt,Nf, t3, lambda1, lambda2, un_t, un_f) \
                        +dt*a[1,0]*self.Lap1_kernel(Xt, Xf, Nt,Nf,t5, lambda1, lambda2, un_t, un_f) \
                        + dt**2*a[1,0]*a[0,0]*self.Lap2_kernel(Xt, Xf, Nt,Nf,t5, lambda1, lambda2, un_t, un_f)\
                        + dt**2*a[1,1]*a[0,1]*self.Lap2_kernel(Xt, Xf, Nt,Nf,t3, lambda1, lambda2, un_t, un_f)

              
            zfu = tf.zeros((Nt,Nu),dtype=tf.float64)        

          
            k4 = tf.concat( (kf3u3, kf3u2, kf3u1, kf3f3, kf3f2,kf3f1),axis=1)         
            k5 = tf.concat( (zfu, kf2u2, kf2u1, kf2f3, kf2f2, kf2f1),axis=1)         
            k6 = tf.concat( (zfu, kf1u2, kf1u1, kf1f3, kf1f2, kf1f1),axis=1)         
            
          
            k = tf.concat((k4,k5,k6),axis=0)    


            return k
     
#    def nlml(self,Xu,Xf,Yu,Yf,dt, hyp1, hyp2, sig_n, kernel_type, jitter=1.0e-10): # negative logarithm marginal-likelihood
#
##        sess1 = tf.Session()
##        sess1.run(tf.global_variables_initializer())
#
#        N = Xu.shape[0] + Xf.shape[0]
#        Nu = Xu.shape[0]
#        K = self.kernel_uf_train(Xu,Xf,hyp1,hyp2,dt) + (sig_n**2+jitter)*tf.eye(N,dtype=tf.float64)              
#        self.L = tf.cholesky(K)
#        r = np.concatenate((Yu,Yf),axis=0) - np.concatenate((np.zeros((Nu,1),dtype=np.float64), self.prior_mean_train),axis=0)
#        self.alpha = tf.cholesky_solve(self.L, r)
#        self.sig2_tf = tf.matmul(r, self.alpha, transpose_a=True)/N
#        return 0.5 * N * tf.log(2.0*np.pi*self.sig2_tf)\
#                +tf.reduce_sum(tf.log(tf.diag_part(self.L))) \
#                + N/2.0    

    def nlml(self,Xu,Xf,Yu1, Yu2, Yu3, Yf,dt, hyp1, hyp3, hyp5, sig_n, lambda1, lambda2, un_u, un_f, kernel_type, jitter=1.0e-10): # negative logarithm marginal-likelihood

#        sess1 = tf.Session()
#        sess1.run(tf.global_variables_initializer())


#        xf_train = np.linspace(-8.0,8.0,self.Nf+2)[1:-1].reshape((-1,1))
#        
#        
#        yf_train = self.u_exact(xf_train,0.0, u_exa, t_exa, x_exa, 2)
#        #yf_train = yf_train+np.linalg.cholesky(previous_cov_mat[:Nf,:Nf])@ np.random.randn(Nf,1)
#        
#        xu_train = np.array([[-8.0], [8.0]],dtype=np.float64)
#        
#        Nu = xu_train.shape[0]
#        Nf = xf_train.shape[0]        
#
#
#        
#        un_u = self.u_exact(xu_train,init_time,u_exa, t_exa, x_exa, 1)
#        un_f = yf_train

        N = 3*(self.Nu + self.Nf)
        self.K0 = self.kernel_uf_train(Xu,Xf,self.Nu, self.Nf, hyp1,hyp3,hyp5,self.a, self.b, self.c, lambda1, lambda2, un_u, un_f, dt)
#        self.K0 = self.kernel_uf_train(xu_train,xf_train,self.Nu, self.Nf, hyp1,hyp3,hyp5,self.a, self.b, self.c, lambda1, lambda2, un_u, un_f, dt)

        K = self.K0 + (sig_n**2+jitter)*tf.eye(N,dtype=tf.float64)
         
        self.L = tf.cholesky(K)
        r = tf.concat((Yu1,Yu2,Yu3,Yf,Yf,Yf),axis=0)
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

        self.lambda1 = tf.Variable(1.0,dtype=np.float64, trainable=False)
        self.lambda2 = tf.Variable(0.1,dtype=np.float64, trainable=False)

        
        
        self.xu_train = tf.placeholder(tf.float64, shape=(None,1))
        self.xu_test = tf.placeholder(tf.float64, shape=(None,1))
        self.xf_train = tf.placeholder(tf.float64, shape=(None,1))
        self.xf_test = tf.placeholder(tf.float64, shape=(None,1))
        self.yu_train1 = tf.placeholder(tf.float64, shape=(None,1))
        self.yu_train2 = tf.placeholder(tf.float64, shape=(None,1))
        self.yu_train3 = tf.placeholder(tf.float64, shape=(None,1))
        self.yf_train = tf.placeholder(tf.float64, shape=(None,1))
        self.un_u = tf.placeholder(tf.float64, shape=(None,1))
        self.un_f = tf.placeholder(tf.float64, shape=(None,1))
        self.un_t = tf.placeholder(tf.float64, shape=(None,1))

#        self.hyp = tf.Variable(self.hyp)

        if self.is_noise:
            self.sig_n = tf.exp(tf.Variable(np.log(1.0e-4),dtype=tf.float64))
        else:
            self.sig_n = tf.Variable(0.0,dtype=tf.float64, trainable=False)

#        sess1 = tf.Session()
#        sess1.run(tf.global_variables_initializer()) 
#        k1,k4 = self.kernel_uf_train(self.xu_train, self.xf_train, self.hyp1, self.hyp3, self.hyp5, self.a, self.b, self.c, self.lambda1, self.lambda2, self.un_u, self.un_f, self.dt)
#        kk1, kk4 = self.kernel_f_test(self.xf_test, self.xu_train, self.xf_train, self.hyp1,  self.hyp3,  self.hyp5,  self.a, self.b, self.c, self.lambda1, self.lambda2, self.un_u, self.un_f, self.un_t, self.dt)   
#        aaa = sess1.run(k4-kk4)    
        self.num_iter = num_iter
        self.jitter = jitter
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        


        init_time = 0.0

        u_simulation = sio.loadmat('burgers.mat')
        u_exa = np.real(u_simulation['usol'])
        t_exa = u_simulation['t'].reshape((-1,1))
        x_exa = u_simulation['x'].reshape((-1,1))


      
        tt0 = time.time()
        
#    Nu = 2
#        Nf = 70
#        self.dt = 1.0e-1
        
        
        
        
        xf_train = np.linspace(-8.0,8.0,self.Nf+2)[1:-1].reshape((-1,1))
        xf_test = x_exa # np.linspace(-8.0,8.0,1000).reshape((-1,1))
        
        
        xu_test = np.concatenate((xf_train,xf_test),axis=0)
        yf_train = self.u_exact(xf_train,init_time, u_exa, t_exa, x_exa, 2)
        #yf_train = yf_train+np.linalg.cholesky(previous_cov_mat[:Nf,:Nf])@ np.random.randn(Nf,1)
        
        xu_train = np.array([[-8.0], [8.0]],dtype=np.float64)
        
        Nu = xu_train.shape[0]
        Nf = xf_train.shape[0]        
        Nt = xf_test.shape[0]


        self.previous_cov_mat = 1e-4*np.eye(3*Nf,dtype=np.float64)
        
        un_u = self.u_exact(xu_train,init_time,u_exa, t_exa, x_exa, 1)
        un_f = yf_train
        un_t = self.u_exact(xf_test,init_time,u_exa,t_exa,x_exa,2)

        yu_train = []
        yu_train.append(self.u_exact(xu_train,dt*1,u_exa, t_exa, x_exa, 1))
        yu_train.append(self.u_exact(xu_train,dt*(1-1)+self.c[1]*dt,u_exa, t_exa, x_exa, 1))
        yu_train.append(self.u_exact(xu_train,dt*(1-1)+self.c[0]*dt, u_exa, t_exa, x_exa, 1))     
   
        
                 
        
        nlml_tf \
             = self.nlml(self.xu_train,self.xf_train, self.yu_train1,self.yu_train2,self.yu_train3, self.yf_train, self.dt, self.hyp1, self.hyp3, self.hyp5, self.sig_n, self.lambda1, self.lambda2, self.un_u, self.un_f, self.kernel_type, self.jitter)
        self.sign_var = self.sig2_tf * self.sig_n**2

        self.K_train = self.kernel_uf_train(self.xu_train, self.xf_train, Nu, Nf, self.hyp1, self.hyp3, self.hyp5, self.a, self.b, self.c, self.lambda1, self.lambda2, self.un_u, self.un_f, self.dt)
        self.m_train = tf.matmul(self.K_train,self.alpha)
        L1 =  tf.concat((tf.zeros((3*Nf,3*Nu),dtype=tf.float64),self.previous_cov_mat),axis=1) 
        L1 = tf.concat((tf.zeros((3*Nu,3*Nu+3*Nf),dtype=tf.float64),L1),axis=0) 
        V1 = tf.linalg.triangular_solve(self.L,tf.transpose(self.K_train))
        V2 = tf.cholesky_solve(self.L, tf.transpose(self.K_train))
        self.var_train = self.sig2_tf*(self.kernel_uf_train(self.xu_train, self.xf_train, Nu, Nf, self.hyp1, self.hyp3, self.hyp5, self.a, self.b, self.c, self.lambda1, self.lambda2, self.un_u, self.un_f, self.dt, diag=True)\
                                  - tf.reshape(tf.reduce_sum(V1*V1,axis=0),(-1,1)))
        self.var_train = self.var_train + tf.reshape( tf.diag_part(tf.matmul(tf.matmul(tf.transpose(V2),L1),V2)),(-1,1))

        self.var_train = tf.maximum(self.var_train,tf.zeros((3*Nu+3*Nf,1),dtype=tf.float64) )    

     
#

        k_test_u = self.kernel_u_test(self.xu_test[Nf:,:], self.xu_train, self.xf_train, Nt, Nu, Nf, self.hyp1, self.hyp3, self.hyp5, self.a, self.b, self.c, self.lambda1, self.lambda2, self.un_u, self.un_f, self.un_t, self.dt)
        self.m_test_u = tf.matmul(k_test_u,self.alpha)        
        V1_test_u = tf.linalg.triangular_solve(self.L,tf.transpose(k_test_u))
        V2_test_u = tf.cholesky_solve(self.L, tf.transpose(k_test_u))
        self.var_test_u = self.sig2_tf * (1.0 - tf.reshape(tf.reduce_sum(V1_test_u*V1_test_u,axis=0),(-1,1))) +self.sign_var                        
        self.var_test_u = self.var_test_u + tf.reshape( tf.diag_part(tf.matmul(tf.matmul(tf.transpose(V2_test_u),L1),V2_test_u)),(-1,1))

        self.var_test_u = tf.maximum(self.var_test_u,tf.zeros((3*Nt,1),dtype=tf.float64) )    
    

   

        k_test_u0 = self.kernel_u_test(self.xu_test[:Nf,:], self.xu_train, self.xf_train, Nf, Nu, Nf, self.hyp1, self.hyp3, self.hyp5,self.a, self.b, self.c, self.lambda1, self.lambda2, self.un_u, self.un_f, self.un_f,  self.dt)
        self.m_test_u0 = tf.matmul(k_test_u0[:Nf,:],self.alpha)        
        V1_test_u0 = tf.linalg.triangular_solve(self.L,tf.transpose(k_test_u0[:Nf,:]))
        V2_test_u0 = tf.cholesky_solve(self.L, tf.transpose(k_test_u0[:Nf,:]))
        self.var_test_u0 = self.sig2_tf * (self.kernel(self.xu_test[:Nf,:],self.xu_test[:Nf,:],Nf, Nf, self.hyp1,equal=True)\
                    - tf.matmul(tf.transpose(V1_test_u0),V1_test_u0)) + self.sign_var* tf.eye(Nf,dtype=tf.float64)
        self.var_test_u0 = self.var_test_u0 + tf.reshape( tf.diag_part(tf.matmul(tf.matmul(tf.transpose(V2_test_u0),L1),V2_test_u0)),(-1,1))
        self.var_test_u0 = tf.maximum(self.var_test_u0,tf.zeros((Nf,Nf),dtype=tf.float64) )    


        k_test_f = self.kernel_f_test(self.xf_test, self.xu_train, self.xf_train, Nt, Nu, Nf, self.hyp1,  self.hyp3,  self.hyp5,  self.a, self.b, self.c, self.lambda1, self.lambda2, self.un_u, self.un_f, self.un_t, self.dt)
        self.m_test_f = tf.matmul(k_test_f,self.alpha)   
        V1_test_f = tf.linalg.triangular_solve(self.L,tf.transpose(k_test_f))
        V2_test_f = tf.cholesky_solve(self.L, tf.transpose(k_test_f))
        self.var_test_f = self.sig2_tf * (self.kernel_uf_train(self.xf_test, self.xf_test, Nt, Nt, self.hyp1,  self.hyp3, self.hyp5,self.a, self.b, self.c, self.lambda1, self.lambda2, self.un_t, self.un_t,  self.dt,diag=True)[3*Nt:,0:1] \
                   - tf.reshape(tf.reduce_sum(V1_test_f*V1_test_f,axis=0),(-1,1)))  + self.sign_var
        self.var_test_f = self.var_test_f + tf.reshape( tf.diag_part(tf.matmul(tf.matmul(tf.transpose(V2_test_f),L1),V2_test_f)),(-1,1))

        self.var_test_f = tf.maximum(self.var_test_f,tf.zeros((3*Nt,1),dtype=tf.float64) )            
        
 
        
        
        
        
        num_t = int(5.0/self.dt)

        
        
        
        u_pred = np.zeros((x_exa.shape[0],num_t+1),dtype=np.float64)
        u_pred[:,0:1] = self.u_exact(x_exa,0.0,u_exa,t_exa,x_exa,2)

        u_interp = np.zeros((x_exa.shape[0],num_t+1),dtype=np.float64)
        u_interp[:,0:1] = self.u_exact(x_exa,0.0,u_exa,t_exa,x_exa,2)
        
        
        
        sig_pred = np.zeros((x_exa.shape[0],num_t+1),dtype=np.float64)
        
        sig_pred[:,0] = 1.0e-2
        
        if optimizer == 'Adam':
            optimizer_Adam = tf.train.AdamOptimizer(learning_rate)
            train_op_Adam = optimizer_Adam.minimize(nlml_tf)   

            grad1 = tf.gradients(nlml_tf,self.hyp1)[0]
            grad2 = tf.gradients(nlml_tf,self.hyp3)[0]
            grad3 = tf.gradients(nlml_tf,self.hyp5)[0]
            gradn = tf.gradients(nlml_tf,self.sig_n)[0]
            std_train = tf.sqrt(self.var_train)
            std_test_u = tf.sqrt(self.var_test_u)
            std_test_f = tf.sqrt(self.var_test_f)
            std_signal = tf.sqrt(self.sig2_tf)
            std_noise = tf.sqrt(self.sign_var) 
            
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            
            for k in np.arange(1,num_t+1):
                print ('\n      t = '+ str(dt*k)+ '  *********************')
                nlml_min = 1.0e16

                yu_train = []
                yu_test = []
                 
                u_interp[:,k:(k+1)] = self.u_exact(xf_test,dt*k, u_exa, t_exa, x_exa, 2)
                yf_test = self.u_exact(xf_test,dt*(k-1),u_exa,t_exa,x_exa,2)
                
                yu_train.append(self.u_exact(xu_train,dt*k,u_exa, t_exa, x_exa, 1))
                yu_train.append(self.u_exact(xu_train,dt*(k-1)+self.c[1]*dt,u_exa, t_exa, x_exa, 1))
                yu_train.append(self.u_exact(xu_train,dt*(k-1)+self.c[0]*dt, u_exa, t_exa, x_exa, 1))     
                
                
                yu_test.append(self.u_exact(xf_train,dt*k, u_exa, t_exa, x_exa, 2))
                yu_test.append(u_interp[:,k:(k+1)])
                yu_test.append(self.u_exact(xf_test,dt*(k-1)+self.c[1]*dt, u_exa, t_exa, x_exa, 2))
                yu_test.append(self.u_exact(xf_test,dt*(k-1)+self.c[0]*dt, u_exa, t_exa, x_exa, 2))
                
                
                
                feed_dict = {self.xu_train: xu_train, self.yu_train1: yu_train[0], \
                             self.yu_train2: yu_train[1], self.yu_train3: yu_train[2],\
                            self.xu_test:  xu_test,   \
                            self.xf_train: xf_train, self.yf_train: yf_train,  \
                           self.xf_test: xf_test, self.un_u: un_u, self.un_f: un_f, self.un_t: un_t}            
            
                if k == 1:
                    self.num_iter = 5001
                    print_skip = 5000
                else:
                    self.num_iter = np.maximum(int(5000*self.dt)+1,50)
                    print_skip = self.num_iter-1
                    
                    
                for i in range(self.num_iter):    
                    sess.run(train_op_Adam, feed_dict)
                    if i % print_skip == 0:
                        nlml_temp = sess.run(nlml_tf, feed_dict)
                        if nlml_temp < nlml_min:
                            nlml_min = nlml_temp
                            self.mm_train = sess.run(self.m_train, feed_dict)
                            self.ss_train = sess.run(std_train,feed_dict)
                            self.mm_test_u = sess.run(self.m_test_u, feed_dict)
                            self.ss_test_u = sess.run(std_test_u, feed_dict)
                            self.mm_test_f = sess.run(self.m_test_f, feed_dict)
                            self.ss_test_f = sess.run(std_test_f, feed_dict)
                            self.mm_test_u0 = sess.run(self.m_test_u0, feed_dict)
                            self.previous_cov_mat = np.tile(sess.run(self.var_test_u0, feed_dict),(3,3))
                            lambda1_val, lambda2_val, nlml_val, hyp1_val, hyp3_val, hyp5_val, sig_f, sig_n, grad_f1, grad_f3, grad_f5, grad_n= \
                            sess.run([self.lambda1, self.lambda2, nlml_tf, self.hyp1, self.hyp3,  self.hyp5,  std_signal, \
                                        std_noise,grad1,grad2,grad3,gradn],feed_dict)                    
                            
                            
                            print ('*************************\n')
                            print ('Iter: ', i, '  nlml =', nlml_min, '\n')
                            print ('nlml:   '   , nlml_val)
                            print ('hyp:   ' , [hyp1_val,hyp3_val, hyp5_val])
                            print ('signal std:   ', sig_f)
                            print ('noise std:   ',sig_n)
                            print('grads of nlml over hyp ', [grad_f1, grad_f3, grad_f5]) 
                            print ('lambda: ', [lambda1_val, lambda2_val])
                            print ('grad of nlml over sig_n', grad_n)                        
                            
                            print ('Training_err_u3:', np.linalg.norm(self.mm_train[:Nu,0:1]-yu_train[0],2)/np.linalg.norm(yu_train[0],2))
                            print ('Training_err_f3:', np.linalg.norm(self.mm_train[(3*Nu):(3*Nu+Nf),0:1]-yf_train,2)/np.linalg.norm(yf_train,2))
    
    
                            print ('Training_err_u2:', np.linalg.norm(self.mm_train[Nu:(2*Nu),0:1]-yu_train[1],2)/np.linalg.norm(yu_train[1],2))
                            print ('Training_err_f2:', np.linalg.norm(self.mm_train[(3*Nu+Nf):(3*Nu+2*Nf),0:1]-yf_train,2)/np.linalg.norm(yf_train,2))
    
                            print ('Training_err_u1:', np.linalg.norm(self.mm_train[(2*Nu):(3*Nu),0:1]-yu_train[2],2)/np.linalg.norm(yu_train[2],2))
                            print ('Training_err_f1:', np.linalg.norm(self.mm_train[(3*Nu+2*Nf):(3*Nu+3*Nf),0:1]-yf_train,2)/np.linalg.norm(yf_train,2))
    
    
                            print ('Test_err_u0:', np.linalg.norm(self.mm_test_u0-yu_test[0],2)/np.linalg.norm(yu_test[0],2))           
    
                            print ('Test_err_u3:', np.linalg.norm(self.mm_test_u[:Nt,0:1]-yu_test[1],2)/np.linalg.norm(yu_test[1],2))           
                            print ('Test_err_f3:', np.linalg.norm(self.mm_test_f[:Nt,0:1]-yf_test,2)/np.linalg.norm(yf_test,2))           
    
    
    
                            print ('Test_err_u2:', np.linalg.norm(self.mm_test_u[Nt:(2*Nt),0:1]-yu_test[2],2)/np.linalg.norm(yu_test[2],2))           
                            print ('Test_err_f2:', np.linalg.norm(self.mm_test_f[Nt:(2*Nt),0:1]-yf_test,2)/np.linalg.norm(yf_test,2))           
    
    
                            print ('Test_err_u1:', np.linalg.norm(self.mm_test_u[(2*Nt):(3*Nt),0:1]-yu_test[3],2)/np.linalg.norm(yu_test[3],2))           
                            print ('Test_err_f1:', np.linalg.norm(self.mm_test_f[(2*Nt):(3*Nt),0:1]-yf_test,2)/np.linalg.norm(yf_test,2))           


                yf_train = self.mm_test_u0 #+ np.linalg.cholesky(previous_cov_mat)@np.random.randn(Nf,1)
                u_pred[:,k:(k+1)] = self.mm_test_u[:xf_test.shape[0]]
                sig_pred[:,k:(k+1)] = self.ss_test_u[:xf_test.shape[0]]
                
                un_u = yu_train[0]
                un_f = self.mm_test_u0
                un_t = self.mm_test_u[:Nt,0:1]



            fig = plt.figure()
            plt.contourf(np.linspace(0.0,5.0,num_t+1), np.ndarray.flatten(x_exa), u_interp, 100, cmap='jet')
            plt.colorbar()
            plt.xlabel('t')
            plt.ylabel('x')
            plt.title('1D Burgers\' equation: Exact solution')
            plt.tight_layout()
            plt.savefig('D-GP-FW-FIG/Exact-Burgers-dt-'+str(100*self.dt)+'.png',dpi=1000)
        #plt.show()
            plt.close(fig)

            fig = plt.figure()
            plt.contourf(np.linspace(0.0,5.0,num_t+1), np.ndarray.flatten(x_exa), u_pred, 100, cmap='jet')
            plt.colorbar()
            plt.xlabel('t')
            plt.ylabel('x')
            plt.title('1D Burgers\' equation: Discrete time GP (solution)')
            plt.tight_layout()
            plt.savefig('D-GP-FW-FIG/D-GP-Burgers-solution-dt-'+str(100*self.dt)+'.png',dpi=1000)
            #plt.show()
            plt.close(fig)
            
            
            fig = plt.figure()
            plt.contourf(np.linspace(0.0,5.0,num_t+1), np.ndarray.flatten(x_exa), sig_pred, 100, cmap='jet')
            plt.colorbar()
            plt.xlabel('t')
            plt.ylabel('x')
            plt.title('1D Burgers\' equation: Discrete time GP (std)')
            plt.tight_layout()
            plt.savefig('D-GP-FW-FIG/D-GP-Burgers-std-dt-'+str(100*self.dt)+'.png',dpi=1000)
            #plt.show()
            plt.close(fig)
            
            
            
            fig = plt.figure()
            plt.contourf(np.linspace(0.0,5.0,num_t+1), np.ndarray.flatten(x_exa), np.abs(u_interp-u_pred), 100, cmap='jet')
            plt.colorbar()
            plt.xlabel('t')
            plt.ylabel('x')
            plt.title('1D Burgers\' equation: Discrete time GP (absolute error)')
            plt.tight_layout()
            plt.savefig('D-GP-FW-FIG/D-GP-Burgers-Ab-err-dt-'+str(100*self.dt)+'.png',dpi=1000)
            #plt.show()
            plt.close(fig)
            
            np.savetxt('D-GP-FW-FIG/exact_u.txt', u_exa, fmt='%10.5e')    
            np.savetxt('D-GP-FW-FIG/interp_u.txt', u_interp, fmt='%10.5e')    
            
            np.savetxt('D-GP-FW-FIG/predicted_u_dt_'+str(100*self.dt)+'.txt', u_pred, fmt='%10.5e')    
            np.savetxt('D-GP-FW-FIG/predicted_std_dt_'+str(100*self.dt)+'.txt', sig_pred, fmt='%10.5e')    
            
            u_error = np.linalg.norm(u_interp.reshape((-1,1))-u_pred.reshape((-1,1)))/np.linalg.norm(u_interp.reshape((-1,1)))
            print('u_error= ', u_error)
            np.savetxt('D-GP-FW-FIG/u_error_dt_'+str(100*self.dt)+'.txt', [u_error], fmt='%10.5e' )
            tt1 = time.time()
            
            print ('CPU time ', tt1-tt0)                  



             
             

a = np.array([[0.25, 0.25-np.sqrt(3.0)/6.0], [0.25+np.sqrt(3.0)/6.0, 0.25]],dtype=np.float64)
b = np.array([0.5,0.5],dtype=np.float64)
c = np.array([0.5-np.sqrt(3.0)/6.0, 0.5+np.sqrt(3.0)/6.0],dtype=np.float64)
Nu = 2
Nf = 70
dt = 5.0e-3        


GP_instance = one_GP()
GP_instance.model(dt,a,b,c,Nu,Nf,is_noise=True)
GP_instance.training(num_iter=5001,jitter=0.0)

del GP_instance


        
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
