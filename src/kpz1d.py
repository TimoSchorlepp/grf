import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import sys
import sampleRandomForcing1d as rdf1
from scipy import stats
import matplotlib as mpl

mpl.rcParams.update({'font.size': 16})

class kpz1d(object):
	# integrate dh/dt = nu d^2h/dx^2 + lambda/2 (dh/dx)^2 + sqrt(D) xi with Heun method
	# h could be the height of a surface upon which particles are deposited
	# xi is centered Gaussian, delta-correlated in time and spatially correlated on a scale l, with intensity D
	# boundaries are periodic
	def __init__(self, nu, lbda, D, l, N, L):
		self.nu = nu # diffusion strength
		self.lbda = lbda # deposition rate
		self.D = D # intensity of noise / particle arrival on surface
		self.N = N # number of gridpoints in space
		self.l = l # correlation length of xi in space
		self.L = L # interval length [0,L) which is discretized
		self.dx = L/N
		self.dt = 0.25 * self.dx**2/self.nu # CFL (deterministic)
		self.steps = 0 # count number of steps..
		self.time = 0 # count elapsed time
		self.xi = rdf1.RandField1d(l,1.,L,N) # intensity of xi is D/dx
		#initial condition for surface height
		W1 = np.cumsum(np.sqrt(self.dx) * np.random.randn(N/2))
		W2 = np.cumsum(np.sqrt(self.dx) * np.random.randn(N/2))
		self.h = np.concatenate([W1,W2[::-1]]) # two sided Wiener process
	
	def step(self):
		h = self.h
		Xi = self.xi.getFieldRealizationRealSpaceSpectral()
		#~ Xi = np.random.randn(self.N)/np.sqrt(self.dx)
		dt = self.dt
		dx = self.dx
		nu = self.nu
		lbda = self.lbda
		D = self.D
		
		while True:
			h1 = h \
				+ (nu*dt)/dx**2 * (np.roll(h,-1) - 2 * h + np.roll(h,1)) \
				+ (lbda*dt)/(8*dx**2) * (np.roll(h,-1) - np.roll(h,1))**2 \
				+ np.sqrt(D*dt) * Xi
			
			hnew = h \
				+ (nu*dt)/(2*dx**2) * (np.roll(h,-1) - 2 * h + np.roll(h,1) + np.roll(h1,-1) - 2 * h1 + np.roll(h1,1)) \
				+ (lbda*dt)/(16*dx**2) * ((np.roll(h,-1) - np.roll(h,+1))**2 + (np.roll(h1,-1) - np.roll(h1,1))**2) \
				+ np.sqrt(D*dt) * Xi
			
			maxhnew = np.max(abs(hnew))
			ind = abs(hnew).argmax()
			
			if (hnew-h-np.sqrt(dt)*Xi)[ind] > maxhnew:
				dt *= 0.95
			else:
				break
		self.dt = dt
		self.time += dt
		self.h = hnew
		self.steps += 1
	
	def multistep(self, n):
		for i in range(n):
			self.step()
	
	def stepUntilT(self,T):
		while self.time < T:
			self.step()
	
	def getdhdt(self,n):
		h_old = self.h
		time_old = self.time
		self.multistep(n)
		return (np.mean(self.h)-np.mean(h_old))/(self.time-time_old)
		
	def stepUntilTArray(self,T):
		h00 = self.h[0]
		ret = np.zeros(len(T))
		for i in range(len(T)):
			self.stepUntilT(T[i])
			ret[i] = self.h[0]
		
		return ret - self.h[0]
		

def init():
    line.set_data([], [])
    text.set_text(r"$t = $")
    return line,text

def animate(i):
    simulation.multistep(100)
    h = simulation.h
    hmean = sum(h)/simulation.N
    #~ hmean = 0
    line.set_data(x_values,h-hmean)
    text.set_text(r'$t = ${0:.5f}'.format(simulation.time))
    sys.stdout.write("\rNumber of steps: {0}".format(simulation.steps))
    sys.stdout.flush()  
    return line,text

nu = 1.
lbda = 1.
D = 1.
l = 0.1
N = 128
L = 2 * np.pi
dx = L/N

###################################################
# uncomment for animation of surface growth 
#~ simulation = kpz1d(nu,lbda,D,l,N,L)
#~ x_values = np.linspace(0, (N - 1) * dx, N)
#~ plot_xmax = N * dx
#~ plot_hmin, plot_hmax = -1.,5.
#~ plot_height = plot_hmax - plot_hmin
#~ fig = plt.figure()
#~ plt.axes(xlim=(0, plot_xmax), ylim=(plot_hmin, plot_hmax))

#~ line = plt.plot([], [])[0]
#~ text = plt.text(0.2*plot_xmax, plot_hmin + 0.9*plot_height, r"$t = 0$", fontsize=20)
            
#~ plt.xlabel('$x$', fontsize=16)
#~ plt.ylabel(r'$h$', fontsize=16)

#~ animationObject = anim.FuncAnimation(fig, animate, init_func=init, blit=True)
#~ plt.show()

###################################################
# determine mean time derivative of h
#~ N_iter = 300 #number of times the simulation os repeated
#~ dhdtArr = np.zeros(N_iter) # will hold the average time derivative of h per simulation
#~ for i in range(N_iter):
	#~ sys.stdout.write("\rNumber of steps: {0}".format(i))
    #~ sys.stdout.flush()
	#~ simulation = kpz1d(nu,lbda,D,l,N,L)
	#~ N_multisteps = 500 # number of multisteps per simulation
	#~ N_onemult = 50 # number of steps per multistep
	#~ dhdt_one_iter = np.zeros(N_multisteps)
	#~ for j in range(N_multisteps):
		#~ dhdt_one_iter[j] = simulation.getdhdt(N_onemult)
	#~ dhdtArr[i] = np.sum(dhdt_one_iter)/N_multisteps

#~ print " "
#~ print np.mean(dhdtArr)
#~ print np.std(dhdtArr)
#~ print " "

# l = 0.0000001: 2.6606056387 +- 0.239304534343
# l = 0.1: 0.919280856789 +- 0.139945416789
# l = 0.5: 0.174165294171 +- 0.11745536202
###################################################
# simulate to a given time in order to determine two-point correlation
N_iter = 5000
T = np.array([0.1,0.2,0.4,0.8,1.0,1.5,2.0,3.0,4.0,5.0,10.0,15.0,18.0,20.0,30.0,40.0,50.0])
Trep = np.repeat(T[:,np.newaxis],N_iter,axis=1).shape
hT = np.zeros((len(T),N_iter))
for i in range(N_iter):
	sys.stdout.write("\rNumber of steps: {0}".format(i))
	sys.stdout.flush()
	simulation = kpz1d(nu,lbda,D,l,N,L)
	hT[:,i] = simulation.stepUntilTArray(T)

print " "
print T
C = np.sum((hT - Tarr * 2.66061)**2,axis=1)/N_iter
print C
print " "

###################################################
#~ #results and plot
#~ # no correlation (l=0.0000001)
#~ T0 = [0.1,0.2,0.4,0.8,1.0,1.5,2.0,3.0,4.0,5.0,10.0,15.0,18.0,20.0,30.0,40.0,50.0]
#~ C0 = [0.364698746382,0.369589335121,0.503156568733,0.701948358368,0.80019422148,1.00439449743,1.13760972451,1.32461128624,1.56560673803,1.71280019755,2.6958902609,3.73011266158,4.28104761891,5.3979972634,8.85050157658,11.6814020968,15.8339615499]

#~ # l = 0.5
#~ T05 = [0.1,0.2,0.4,0.8,1.0,1.5,2.0,3.0,4.0,5.0,10.0,15.0]
#~ C05 = [0.30788159774,0.281025908555,0.418584461225,0.839875198543,0.78025475347,0.875447946796,0.98699055876,1.42225038693,1.4569261589,1.6382943847,2.422920858,3.25392026254]

#~ # l = 0.1
#~ T01 = [0.1,0.2,0.4,0.8,1.0,1.5,2.0,3.0,4.0,5.0,10.0,15.0,18.0,20.0,30.0,40.0]
#~ C01 = [0.307634037564,0.34297547455,0.475592578528,0.82241902433,0.948830789165,0.971833220291,1.05436653359,1.35022703835,1.6576021401,1.75569758417,2.52190382193,3.46302914284,3.77045218754,4.17399354072,5.91496482688,7.53595963263]


# comparison T^(2/3)
#~ Tline = np.logspace(np.log10(0.1),np.log10(50.),100)
#~ Cline = 0.4 * Tline**(2./3.)

#~ # linear regression for slope
#~ alpha0,_,_,_,_ = stats.linregress(np.log10(T0),np.log10(C0))
#~ alpha01,_,_,_,_ = stats.linregress(np.log10(T01),np.log10(C01))
#~ alpha05,_,_,_,_ = stats.linregress(np.log10(T05),np.log10(C05))

#~ plt.loglog(T0,C0,'*',label = r'$\lambda = 0.0000001$, exponent: ' + str(round(alpha0,3)),markersize=8)
#~ plt.loglog(T01,C01,'*',label = r'$\lambda = 0.1$, exponent: ' + str(round(alpha01,3)),markersize=8)
#~ plt.loglog(T05,C05,'*',label = r'$\lambda = 0.5$, exponent: ' + str(round(alpha05,3)),markersize=8)
#~ plt.loglog(Tline,Cline,label = r'$C(t) = \alpha t^{2/3} $',linewidth=2)
#~ plt.xlabel(r'$t$')
#~ plt.ylabel(r'$C$')
#~ plt.legend(loc='best')
#~ plt.show()


		
