#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:51:33 2019

@author: michaelcornish
"""

import numpy as np
import os
from math import pi
import matplotlib.pyplot as plt
import pickle

# math functions
def sin(x):
    from math import sin
    output = np.array([sin(i) for i in x])
    return output

def cos(x):
    from math import cos
    output = np.array([cos(i) for i in x])
    return output

def ddx(y,k,N):
    # differentiates in fourier space
    output = np.real( np.fft.ifft( N*1j*k*np.fft.fft(y) ) )
    return output

def d2dx2(y,k,N):
    # differentiates in fourier space
    output = np.real( np.fft.ifft( -(N*k)**2*np.fft.fft(y) ) )
    return output


def f1(s, w, k,N): 
    # The inputs s & w are in fourier space
    # output is the RHS of S evolution equation in fourier space
    
    K = N*k;
    
    Sx = np.real(np.fft.ifft(1j*K*s));                  
    Wx = np.real(np.fft.ifft(1j*K*w));                  

    #Runge-Kutta Step
    temp = np.fft.fft(-Sx*np.real(np.fft.ifft(w)) -0.5*Wx*(1 + np.real(np.fft.ifft(s))));

    # Preserve Symmetries
    temp  = np.real(temp);            
    # Remove Machine Error    
    temp[np.abs(temp) <1e-1] = 0;
    
    return temp

def f2(s, w,k,B_0, epsilon,N): 
    # The inputs s & w are in fourier space
    # output is the RHS of W evolution equation in fourier space    
    
    K = N*k;
    
    # Runge-Kutta Step
    first = 0.5*np.fft.ifft(w)**2  - (0.5*B_0/((1+np.fft.ifft(s))**2));
    second = (1/(1+np.fft.ifft(s)) - (epsilon*np.fft.ifft(-K**2*s))/(1 + (epsilon*(1+np.fft.ifft(1j*K*s))**2) ) )/np.sqrt(1 + (epsilon*(1+np.fft.ifft(1j*K*s))**2));
    third = -1j*K*np.fft.fft(first + second);
    fourth = np.fft.fft( np.fft.ifft(1j*K*np.fft.fft( (1+np.fft.ifft(s))**2*np.fft.ifft(1j*K*w)))*3/(1+np.fft.ifft(s))**2);
    temp = third + fourth;
    
    # Preserve Symmetries       
    temp = 1j*np.imag(temp);     
    # Remove Machine error
    temp[np.abs(temp) <1e-1] = 0;

    return temp


def single_fluid_plots(S,W,B_0,k,x,dt,save_prefix):

    factor = dt;
    p = min(S.shape[0], W.shape[0]);
    M = 10; # Number of curves
    
    plt.figure()
    for j in np.linspace(p//M, p,M).astype(int)[0:-1]:
        if j==p//M:
            plt.plot(x,1+S[0,:], label = 'Initial State');
        elif j == M-1:
            plt.plot(x,1+S[j,:],label = 't = {}'.format(factor*p))
        else:
            plt.plot(x,1+S[j,:]);

    
    plt.xlabel('z',fontsize = 18)
    plt.ylabel('S',fontsize = 18)
    plt.savefig(os.path.join(save_prefix, 'S.eps')) 
    plt.close()
    
    plt.figure()
    for j in np.linspace(p//M, p,M).astype(int)[0:-1]:
        if j==p//M:
            plt.plot(x,1+W[0,:], label = 'Initial State');
        elif j == M-1:
            plt.plot(x,1+W[j,:],label = 't = {}'.format(factor*p))
        else:
            plt.plot(x,1+W[j,:]);

    
    plt.xlabel('z',fontsize = 18)
    plt.ylabel('W',fontsize = 18)
    plt.savefig(os.path.join(save_prefix, 'W.eps'))
    plt.close()



def Diss_Full_Curv_main(epsilon, B_0):
    file_path =  'Dissipation\B_0__{}\epsilon__{}'.format(B_0,epsilon)
    try:
        os.mkdir(file_path)
    except:
        pass 
    
    # Initialization
    
    N = 2**10;                                     # fft works fastest with 2^n points. 
    x = (2*pi/N)*np.array([i for i in range(N)]);  # spatial domain
    k = np.fft.fftfreq(x.shape[-1])                # frequency domain 
    dt = 1e-4;                                     # Time step
    
    Q = 1;                            # Number of solution recordings
    S = np.zeros([Q,N]);                    # Solution matrix in real space
    W = np.zeros([Q,N]);                    # Solution matrix in real space
    
    S[0,:] = 0.1*cos(x);                 # Initial real space wave profile
    W[0,:] = -0.1*sin(x);              # Initial real space velocity profile
    Sf_old = np.fft.fft(S[0,:]);              # Initial Fourier space wave profile
    Wf_old = np.fft.fft(W[0,:]);              # Initial Fourier space velocity profile

    Sf_old = np.real(Sf_old)
    Wf_old = 1j*np.imag(Wf_old)
    Sf_old[Sf_old <1] = 0;
    Wf_old[abs(Wf_old) < 1] = 0;
    
    ## Main Loop
    
    #Diagnostics
    t = 1; p = 0; 
    S_min = min(1+S[0,:]); 
    S_threshold = 0.01;
    real_time = 0;
    nontrivial = True; 
    non_converged = True; 
    
    ##
    while S_min > S_threshold and real_time < 200 and nontrivial and non_converged:
    
        # Runge-Kutta Method
        k0 = f1(Sf_old, Wf_old,k,N);                        
        l0 = f2(Sf_old, Wf_old,k,B_0, epsilon,N);
        k1 = f1(Sf_old + 0.5*dt*k0,Wf_old + 0.5*dt*l0,k,N); 
        l1 = f2(Sf_old + 0.5*dt*k0,Wf_old + 0.5*dt*l0,k,B_0, epsilon,N);
    
        Sf_new = Sf_old + dt*k1;
        Wf_new = Wf_old + dt*l1;
    
        # Diagnostics
        S_min = min(1+np.real(np.fft.ifft(Sf_new)));
        if S_min < S_threshold or np.isnan(S_min):
            print('Minimum S triggered stop')
    
        S_temp = np.real(np.fft.ifft(Sf_new))
        Sx  = np.max(np.abs( ddx(   S_temp,k,N)));
        Wx  = np.max(np.abs( ddx(   np.real(np.fft.ifft(Wf_new)),k,N)));
        Sxx = np.max(np.abs( d2dx2( S_temp,k,N)));
    
        if Sx >100 or Wx >100 or Sxx > 100: 
            print('Slope Singlurity')
            break;
        
        # convergence test
        S_old = np.real(np.fft.ifft(Sf_old))
        S_new = np.real(np.fft.ifft(Sf_new))
        L2 = np.dot(S_old-S_new, S_old - S_new)
        if L2 < 1e-10 and t > 100:
            print('Convergence!')
            non_converged = False;
    
        # Update Solution and Scheme
        t = t+1;
        real_time =  dt*t;
        Sf_old = Sf_new; Wf_old = Wf_new;
        if real_time > 0.1*(p+1):
            S = np.vstack([S, np.real(np.fft.ifft(Sf_new))]);
            W = np.vstack([W, np.real(np.fft.ifft(Wf_new))]);
            p = p+1;
            
            if np.min(S[p,:]) > np.min(S[p-1,:]):
                nontrivial = False;
                print('')
                print('Trivial Solution Expected, Computations Stopped')
                print('')
            
    

    single_fluid_plots(S,W,B_0,k,x,dt,file_path)

    # Saving the objects:
    with open(os.path.join(file_path,'objs.pkl')) as f:  # Python 3: open(..., 'wb')
        pickle.dump([S,W], f)
    

    return S, W

'''
Main File 
'''


B_0 = 0.5;                         # Magnetic Bond Number
linear_epsilon = 0.5;
'''
epsilon_list = np.linspace(linear_epsilon - 0.7*linear_epsilon, linear_epsilon + 0.3*linear_epsilon, 7);
epsilon_list = epsilon_list[-1:-1:0];
conservation_error_list = [];

ep_list = cat(2,epsilon_list2, epsilon_list3);
conservation_error_list2 = [];
'''
conservation_error_list = [];
N = 2**10;                                     # fft works fastest with 2^n points. 
x = (2*pi/N)*np.array([i for i in range(N)]);  # spatial domain
k = np.fft.fftfreq(x.shape[-1])                # frequency domain 
dt = 1e-4;                                     # Time step
    
dz = x[1]-x[0];

for epsilon in [0.1]:
    print('Working on epsilon = {}'.format(epsilon))
    S,W = Diss_Full_Curv_main(epsilon, B_0);
    conservation_error_list.append(np.trapz((1+S[0,:])**2,dx = dz) - np.trapz((1+S[-1,:])**2,dx = dz))