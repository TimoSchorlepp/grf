import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import sys
import sampleRandomForcing1d as rdf1

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
		self.h = np.zeros(N)
		#~ self.h[0] = 5.
		X = np.linspace(0,(N-1)*self.dx,N)
		for i in range(N):
			self.h[i] = np.min([abs(X[0]-X[i]),abs(X[0]-X[i]+L),abs(X[0]-X[i]-L)])
	
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
		return (np.sum(self.h)-np.sum(h_old))/(self.time-time_old)
		

def init():
    line.set_data([], [])
    return line,

def animate(i):
    simulation.multistep(100)
    h = simulation.h
    hmean = sum(h)/simulation.N
    hmean = 0
    line.set_data(x_values,h-hmean)
    sys.stdout.write("\rNumber of steps: {0}".format(simulation.steps))
    sys.stdout.flush()  
    return line,

nu = 1.
lbda = 1.
D = 1.
l = 0.0000001
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
            
#~ plt.xlabel('$x$', fontsize=16)
#~ plt.ylabel(r'$h$', fontsize=16)

#~ animationObject = anim.FuncAnimation(fig, animate, init_func=init, blit=True)
#~ plt.show()

###################################################
# determine mean time derivative of h
#~ N_iter = 300 #number of times the simulation os repeated
#~ dhdtArr = np.zeros(N_iter) # will hold the average time derivative of h per simulation
#~ for i in range(N_iter):
	#~ print i
	#~ simulation = kpz1d(nu,lbda,D,l,N,L)
	#~ N_multisteps = 100 # number of multisteps per simulation
	#~ N_onemult = 50 # number of steps per multistep
	#~ dhdt_one_iter = np.zeros(N_multisteps)
	#~ for j in range(N_multisteps):
		#~ dhdt_one_iter[j] = simulation.getdhdt(N_onemult)
	#~ dhdtArr[i] = np.sum(dhdt_one_iter)/N_multisteps

#~ print np.mean(dhdtArr)
#~ print np.std(dhdtArr)

# no correlation: 337.73779737587535 +-29.992229916804153

###################################################
# simulate to a given time in order to determine two-point correlation
N_iter = 1000
T = 1.
hT = np.zeros(N_iter)
for i in range(N_iter):
	print i
	simulation = kpz1d(nu,lbda,D,l,N,L)
	simulation.stepUntilT(T)
	hT[i] = simulation.h[0]

simulation = kpz1d(nu,lbda,D,l,N,L)
C = np.sum((hT - simulation.h[0] - T * 340)**2)/N_iter

# no correlation
# t = 0.1: C = 
# t = 0.25: C = 
# t = 0.5: C =
# t = 1: C =
# t = 2: C = 
# t = 3: C =
# t = 4: C =
# t = 5: C = 




		
