# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import math as m
import numpy as np
import timeit
import random
from scipy.integrate import odeint
from odeintw import odeintw

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

plt.rcParams.update({'font.size': 16})

"Simple Euler ODE Solver"
def EulerForward(f,y,t,h,del_0L,gamma,phi,tp,Omega0,w0,wl): # Vectorized forward Euler (so no need to loop)
# asarray converts to np array - so you can pass lists or numpy arrays
    k1 = h*np.asarray(f(t,y,del_0L,gamma,phi,tp,Omega0,w0,wl))
    y=y+k1
    return y

def rk4(f,y,t,h,del_0L,gamma_d,phi,tp,Omega0,w0,wl):
    # Runge Kutta 4 iterative method
    k_1 = h * np.asarray(f(t,y,del_0L,gamma_d,phi,tp,Omega0,w0,wl))
    k_2 = h * np.asarray(f((t+(h/2)),(y+(k_1/2)),del_0L,gamma_d,phi,tp,Omega0,w0,wl))
    k_3 = h * np.asarray(f((t + (h/2)),(y+(k_2/2)),del_0L,gamma_d,phi,tp,Omega0,w0,wl))
    k_4 = h * np.asarray(f(t + h, y + k_3,del_0L,gamma_d,phi,tp,Omega0,w0,wl))
    y_next = y + (1/6)*(k_1 + 2 * k_2 + 2 * k_3 + k_4)
    return y_next

"OBEs - with simple CW excitation"
def derivs(t,y,del_0L,gamma,phi,tp,Omega0,w0,wl): # derivatives function
    #dy=np.zeros((len(y)))
    dy = [0] * len(y) # could also use lists here which can be faster if
                       # using non-vectorized ODE "
    dy[0] = 0.
    dy[1] = Omega0/2*(2.*y[2]-1.)
    dy[2] = -Omega0*y[1]
    return dy

def OBE(t,y,del_0L,gamma_d,phi,tp,Omega0,w0,wl):
    # RWA Approx
    dy = [0] * len(y) # could also use lists here which can be faster if
    Omega =  Omega0 * m.exp(-((t-5)**2)/(tp**2))              # using non-vectorized ODE "
    dy[0] = -gamma_d * (y[0]) + del_0L * y[1]
    dy[1] =  Omega/2*(2.*y[2]-1.) - del_0L * (y[0]) - gamma_d * y[1]
    dy[2] = -Omega*y[1]
    return dy

def FullBloch(t,y,del_0L,gamma_d,phi,tp,Omega0,w0,wl):
    # OBE for Full Rabi wave with no RWA
    dy = [0] * len(y) # could also use lists here which can be faster if
    Omega =  Omega0 * m.exp(-((t-5)**2)/(tp**2)) * m.sin(wl*t + phi)             # using non-vectorized ODE "
    dy[0] = -gamma_d * (y[0]) + w0 * y[1]
    dy[1] =  Omega*(2.*y[2]-1.) - w0 * (y[0]) - gamma_d * y[1]
    dy[2] = -2*Omega*y[1]
    return dy

def reset(npts):
    y = np.zeros((npts,3)) # or can introduce arrays to append to
    yinit = np.array([0.0,0.0,0.0]) # initial conditions (TLS in ground state)
    y1 = yinit # just a temp array to pass into solver
    y[0,:]= y1
    return y , yinit , y1

dt = 0.001
tmax = 10.
def call(func,OBE,
            del_0L=0,
            gamma_d=0,
            phi=0,
            tlist=np.arange(0.0, tmax, dt),
            tp = 1/(m.sqrt(np.pi)),
            Omega0 = 2*np.pi,
            w0 = 0,
            wl = 0):


    # Calls specific step method with selectable parameters
    print("Call Function activated, Parameters:")
    print("del_0L=" +str(del_0L))
    print("gamma_d=" +str(gamma_d))
    print("phi=" +str(phi))
    print("tp=" +str(tp))
    print("Omega0=" +str(Omega0))
    print("w0=" +str(w0))
    print("wl=" +str(wl)+"\n")

    npts = len(tlist)
    y , yinit , y1 = reset(npts)
    npts = len(tlist)
    # Iterates over resolution of step size
    for i in range(1,npts):
        y1=func(OBE,y1,tlist[i-1],dt,del_0L,gamma_d,phi,tp,Omega0,w0,wl)
        y[i,:]= y1
    return y

def triplePlot(tlist,y,ti,legend=["","",""],legOn=False):
    # Creates a standard 3 plot for Real, imaginary and Population components
    fig, axs = plt.subplots(3,1,sharex=True)
    fig.subplots_adjust(hspace=0.2)
    axs[0].set(title=ti)
    axs[0].plot(tlist, y[:,0],label=legend[0])
    axs[1].plot(tlist, y[:,1],label=legend[1])
    axs[2].plot(tlist, y[:,2],label=legend[2])
    axs[0].set(ylabel='Re$[U]$')
    axs[1].set(ylabel='Im$[U]$')
    axs[2].set(xlabel='$t (s)$', ylabel='$n_e$')
    axs[0].tick_params(axis='y', which='major', pad=10)
    axs[1].tick_params(axis='y', which='major', pad=10)
    axs[2].tick_params(axis='y', which='major', pad=10)
    if legOn:
        axs[0].legend(loc='top right')
        axs[1].legend(loc='top right')
        axs[2].legend(loc='top right')

    plt.savefig('./'+ti+'.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()

def doublePlot(tlist,y,ti):
    # Creates 2 subplot figure for imaginary and population components
    fig, axs = plt.subplots(2,1,sharex=True)
    fig.subplots_adjust(hspace=0)
    axs[0].set(title=ti)
    axs[0].plot(tlist, y[:,1])
    axs[1].plot(tlist, y[:,2])
    axs[0].set( ylabel='Im$[U]$')
    axs[1].set(xlabel='$t (s)$', ylabel='$n_e$')
    plt.savefig('./'+ti+'.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()

def Q1():

    Omega0 = m.pi*2
    tmax = 5.
    tlist=np.arange(0.0, tmax, dt)
    npts = len(tlist)
    yEuler = call(EulerForward,derivs,tlist=tlist) # euler solution
    yrk4 = call(rk4,derivs,tlist=tlist) # Rk4 solution
    yexact = [m.sin(Omega0*tlist[i]/2)**2 for i in range(npts)] # Exact solution
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(tlist, yexact, 'r', label="Exact Solution")
    ax.plot(tlist, yEuler[:,2], 'b', label="Euler")
    ax.plot(tlist, yrk4[:,2]-0.015, 'g', label="RK4")
    box = ax.get_position()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set(xlabel="t ($s$)",ylabel="$n_e$",title="h="+str(dt))
    plt.savefig('./Q1b.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()

def Q2():
    tlist=np.arange(0, tmax, dt)
    y = call(rk4,OBE) # Calls new OBE with the Gaussian Pulse the default
    ti = "Time dependent Gaussian Pulse"
    doublePlot(tlist,y,ti)

def findMax(tuning,phasing,res):
    peak_ne = []
    # Finds the maximum population for a given solution
    tlist=np.arange(0, tmax, dt)
    for j in range(res):
        y_rk = call(rk4,OBE,tuning[j],phasing[j],tlist=tlist)
        peak_ne.append(np.max(y_rk[:,2]))
    return peak_ne

def Q3():
    res = 100
    Omega0 = 2*m.pi
    a = np.arange(0,Omega0,(Omega0/res))
    b= np.zeros(res)
    # For (a) Detuning, dephasing =0
    peak_ne_detune = findMax(a,b,res)
    # For (b) Dephasing, detuning = 0
    peak_ne_dephase = findMax(b,a,res)
    plt.plot(a,peak_ne_detune)
    plt.xlabel("$Δ_{0L}$")
    plt.ylabel("$n_e$")
    plt.title("Detuning")
    #plt.savefig('./Q3aDetuning.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()

    plt.plot(a,peak_ne_dephase)
    plt.xlabel("$γ_d$")
    plt.ylabel("$n_e$")
    plt.title("Dephasing")
    #plt.savefig('./Q3bDephasing.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()

def Q4():
    w_l = np.asarray([20,10,5,2])
    tlist=np.arange(0, tmax, dt)
    tp = 1
    Omega0 = 2*m.sqrt(m.pi)

    #4a
    for x in range(len(w_l)):
        ti = "$ω_L$ =" + str(w_l[x])+"$Ω_0$, φ=0"#  π$"
        wl_item = Omega0 * w_l[x]
        y = call(rk4,FullBloch,0,0,0,tlist,tp,Omega0,wl_item,wl_item)
        triplePlot(tlist,y,ti)

    Omega0 = 2*m.sqrt(m.pi)
    y = call(rk4,FullBloch,0,0,m.pi/2,tlist,tp,Omega0,2*Omega0,2*Omega0)
    ti = "$ω_L$ = 2$Ω_0$, φ = 0.5π"
    triplePlot(tlist,y,ti)


    # 4b
    Omegas = [2,10,20]
    wl =4*m.sqrt(m.pi)
    for x in range(len(Omegas)):
        ti = "$ω_L$ = 4√π, Area = "+str(Omegas[x])+"π"
        tlist=np.arange(0, tmax, dt)
        y = call(rk4,FullBloch,0,0,0,tlist,tp,Omegas[x]*m.sqrt(m.pi),wl,wl)
        triplePlot(tlist,y,ti)


def Q4c():

    """ODE integrate function"""
    tlist=np.arange(0, tmax, dt)
    npts = len(tlist)
    y , yinit , y1 = reset(npts)
    del_0L = 0
    gamma_d = 0.4
    phi = 0
    tp = 1
    Omega0 = 2*m.sqrt(m.pi)
    wl =4*m.sqrt(m.pi)
    y = np.asarray(y)
    arg = [del_0L,gamma_d,phi,tp,Omega0,wl,wl]
    start = timeit.default_timer()

    # Library Package Odeint solution
    odesolved = odeintw(FullBloch,y[0],tlist,args=(del_0L,gamma_d,phi,tp,Omega0,wl,wl),tfirst=True)
    finish = timeit.default_timer()
    print("Time to complete Library Package: "+ str(finish-start))
    #Fourier transform of real component
    odesolved1 = (np.abs(np.fft.fftshift(np.fft.fft((odesolved[:,0]))/len(y[0]))))
    w_ode = ((2*m.pi*(np.fft.fftshift(np.fft.fftfreq(len(tlist),dt))))/wl)
    max_ode = np.max(odesolved1)

    FT = []
    w = []
    max_t = []
    tp = 1
    wl =4*m.sqrt(m.pi)
    Omegas = [2*m.sqrt(m.pi),10*m.sqrt(m.pi),20*m.sqrt(m.pi)]
    for x in range(len(Omegas)):
        tlist=np.arange(0, 50, dt)
        gamma_d = 0.4/tp
        start = timeit.default_timer()
        y = call(rk4,FullBloch,0,gamma_d,0,tlist,tp,Omegas[x],wl,wl)
        finish = timeit.default_timer()
        ft_y = np.abs(np.fft.fft((y[:,0]))/len(y[0]))
        ft_y = np.fft.fftshift(ft_y)
        FT.append(ft_y)

        w.append((2*m.pi*(np.fft.fftshift(np.fft.fftfreq(len(tlist),dt))))/wl)

        print("Time to complete: "+ str(finish-start))
        max_t.append(np.max(FT))
    fig, axs = plt.subplots(4,1, sharex=True)
    fig.subplots_adjust(hspace=0.2)
    # Log graph
    axs[0].semilogy(w[0], FT[0],label="Area=2π")
    axs[0].set_xlim(0,6)
    axs[0].set_ylim(1,max_t[0])
    axs[0].legend(loc='top right')
    axs[1].semilogy(w[1], FT[1],label="Area=10π")
    axs[1].set_xlim(0,6)
    axs[1].set_ylim(1,max_t[1])
    axs[1].legend(loc='top right')
    axs[2].semilogy(w[2], FT[2],label="Area=20π")
    axs[2].set_xlim(0,6)
    axs[2].set_ylim(1,max_t[2])
    axs[2].legend(loc='top right')
    axs[3].semilogy(w_ode,odesolved1,label="Area=2π")
    axs[3].set_xlim(0,6)
    axs[3].set_ylim(1,max_ode)
    axs[3].legend(loc='top right')
    axs[0].set(ylabel="|$P_2$(ω)|",title="Power Spectrum")
    axs[1].set(ylabel="|$P_{10}$(ω)|")
    axs[2].set(ylabel="|$P_{20}$(ω)|")
    axs[3].set(xlabel="$ω/ω_L$",ylabel="|$P_{ode}$(ω)|")
    plt.savefig('./PowerSpectrum.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()
    #4c






Q1()
Q2()
Q3()
Q4()
Q4c()
