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
        self.dt = 0.25 * self.dx**2/self.D # CFL (deterministic)
        self.steps = 0 # count number of steps..
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
		self.h = hnew
		self.steps += 1
	
    def multistep(self, n):
        for i in range(n):
            self.step()

def init():
    line.set_data([], [])
    return line,

def animate(i):
    simulation.multistep(1)
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
l = 0.1
N = 128
L = 2 * np.pi
dx = L/N

simulation = kpz1d(nu,lbda,D,l,N,L)

x_values = np.linspace(0, (N - 1) * dx, N)
plot_xmax = N * dx
plot_hmin, plot_hmax = -1.,5.
plot_height = plot_hmax - plot_hmin
fig = plt.figure()
plt.axes(xlim=(0, plot_xmax), ylim=(plot_hmin, plot_hmax))

line = plt.plot([], [])[0]
            
plt.xlabel('$x$', fontsize=16)
plt.ylabel(r'$h$', fontsize=16)

animationObject = anim.FuncAnimation(fig, animate, init_func=init, blit=True)

plt.show()
