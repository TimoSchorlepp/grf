import numpy as np
import matplotlib.pyplot as plt
import sys

class RandField1d(object):
	
	def __init__(self,l,chi0,xSz,nx):
		
		self.l = float(l) #correlation length, see getChi() for definition of covariance
		self.chi0 = float(chi0) #correlation strength
		self.xSz = float(xSz) # simulation on interval [0,xSz]
		self.nx = nx #number of gridpoints
		self.dx = xSz/nx #grid spacing
		self.X = np.linspace(0.,xSz-self.dx,nx)
		self.KX = 2*np.pi*np.fft.fftfreq(self.nx, self.dx) #2*pi needed due to different convention in np.fft
		self.Mmax = 33 # number of neigbour intervals considered for periodically wrapped correlation
		self.M = np.linspace(-(self.Mmax-1)/2,(self.Mmax-1)/2,self.Mmax)
		self.lbda_spec = np.sqrt(self.getChiHat(self.KX)) #decomposition in Fourier space
		
		self.chi = np.zeros((self.nx,self.nx)) #real space non-periodic correlation for direct method
		for i in range(self.nx):
			for j in range(self.nx):
				self.chi[i,j] = self.getChi(self.X[i]-self.X[j])
		
		D,V = np.linalg.eigh(self.chi) # decomposition chi = V diag(D) V.T
		print "Maximum eigenvalue of grid covariance matrix ", np.amax(D)
		print "Minimum eigenvalue of grid covariance matrix ", np.amin(D)
		V[:,D<1e-14] = 0
		D[D<1e-14] = 0.
		self.lbda_direct = np.dot(V,np.sqrt(np.diag(D)))
		
##################################################################
	def getFieldRealizationKSpace(self):	
		return 1/np.sqrt(2) * self.lbda_spec * (np.random.randn(self.nx) + 1j * np.random.randn(self.nx))
	
	def getFieldRealizationRealSpaceSpectral(self):
		xi = self.getFieldRealizationKSpace()
		return np.fft.ifft(xi).real/self.dx*np.sqrt(2*self.xSz)
	
	def getFieldRealizationRealSpaceDirect(self):
		return np.dot(self.lbda_direct,np.random.randn(self.nx))
##################################################################
	def testErrorConvergenceRealSpaceSpectral(self,n1,n2,n):
		# compute correlation between all points
		# n1 and n2 are the start and end numbers of realizations which are sampled
		# n is the number of intermediate steps were the error is computed
		
		print "--------------------------------------------------"
		print "Testing convergence of correlation function in real space generated with spectral method"
		print "Total number of samples that will be generated: ", n2
		print "--------------------------------------------------"
		
		num = np.logspace(np.log10(n1),np.log10(n2),n,dtype = int)
		err = np.zeros(n)
		
		correlRealizations = np.zeros((self.nx,self.nx))
		correlExact = np.zeros((self.nx,self.nx))
		
		for j in range(self.nx):
			for k in range(self.nx):
				correlExact[j,k] = np.sum(self.getChi(self.X[j] - self.X[k] + self.M * self.xSz))
		
		#~ plt.plot(correlExact[0,:])
		#~ plt.show()
		
		#~ plt.imshow(correlExact)
		#~ plt.colorbar()
		#~ plt.show()
		
		for i in range(n):
			
			sys.stdout.write("\rStep {0} of {1}".format(i+1,n))
			sys.stdout.flush()
			
			if i == 0:
				
				for l in range(num[i]):
					xi = self.getFieldRealizationRealSpaceSpectral()
					correlRealizations += np.outer(xi,xi)
			else:
				
				for l in range(num[i]-num[i-1]):
					xi = self.getFieldRealizationRealSpaceSpectral()
					correlRealizations += np.outer(xi,xi)
							
			errMatrix = abs(correlRealizations/num[i] - correlExact)
			
			#~ for j in range(self.nx):
				#~ for k in range(self.nx):
					#~ if correlExact[j,k] > 1e-8:
						#~ errMatrix[j,k] /= correlExact[j,k]
						
			err[i] = np.amax(errMatrix)
		
		#~ plt.plot(correlRealizations[0,:]/n2)
		#~ plt.show()
		
		#~ plt.imshow(correlRealizations/n2)
		#~ plt.colorbar()
		#~ plt.show()
		
		plt.figure()
		plt.loglog(num,err,c='blue')
		plt.loglog(num,err,'.',c='blue')
		plt.xlabel(r'$n$')
		plt.ylabel(r'$e$')
		#plt.title(r'Maximum relative error of covariance $e$ between any two points' + '\n' + r'in real space, 1d, for different sample numbers $n$')
		plt.savefig('RealSpaceError1dSpectral.pdf')
		plt.close()
		
		print " "
		return
##################################################################
	def testErrorConvergenceKSpace(self,n1,n2,n):
		
		print "--------------------------------------------------"
		print "Testing convergence of correlation function in Fourier space"
		print "Total number of samples that will be generated: ", n2
		print "--------------------------------------------------"
		
		num = np.logspace(np.log10(n1),np.log10(n2),n,dtype = int)
		err = np.zeros(n)
		
		correlRealizations = np.zeros((self.nx,self.nx),dtype=complex)
		correlExact = np.zeros((self.nx,self.nx))
		
		for j in range(self.nx):
			correlExact[j,j] = self.getChiHat(self.KX[j])
		
		for i in range(n):
			
			sys.stdout.write("\rStep {0} of {1}".format(i+1,n))
			sys.stdout.flush()
			
			if i == 0:
				
				for l in range(num[i]):
					xihat = self.getFieldRealizationKSpace()
					correlRealizations += np.outer(xihat,np.conjugate(xihat))
				
			else:
				
				for l in range(num[i]-num[i-1]):
					xihat = self.getFieldRealizationKSpace()
					correlRealizations += np.outer(xihat,np.conjugate(xihat))
			
			errMatrix = abs(correlRealizations/num[i] - correlExact)	
			
			#~ for j in range(self.nx):
				#~ for k in range(self.nx):
					#~ if abs(correlExact[j,k]) > 1e-8:
						#~ errMatrix[j,k] /= correlExact[j,k]
			
			err[i] = np.amax(errMatrix)
		
		plt.figure()
		plt.loglog(num,err,c='blue')
		plt.loglog(num,err,'.',c='blue')
		plt.xlabel(r'$n$')
		plt.ylabel(r'$e$')
		#plt.title(r'Maximum relative error of covariance $e$ between any two points' + '\n' + r'in Fourier space, 1d, for different sample numbers $n$')
		plt.savefig('KSpaceError1d.pdf')
		plt.close()
		
		print " "
		return	
##################################################################
	def testErrorConvergenceRealSpaceDirect(self,n1,n2,n):
		# compute correlation between all points
		# n1 and n2 are the start and end numbers of realizations which are sampled
		# n is the number of intermediate steps were the error is computed
		
		print "--------------------------------------------------"
		print "Testing convergence of correlation function in real space generated with direct method"
		print "Total number of samples that will be generated: ", n2
		print "--------------------------------------------------"
		
		num = np.logspace(np.log10(n1),np.log10(n2),n,dtype = int)
		err = np.zeros(n)
		
		correlRealizations = np.zeros((self.nx,self.nx))
		correlExact = np.zeros((self.nx,self.nx))
		
		for j in range(self.nx):
			for k in range(self.nx):
				correlExact[j,k] = self.getChi(self.X[j] - self.X[k])
		
		plt.plot(correlExact[0,:])
		plt.show()
		
		plt.imshow(correlExact)
		plt.colorbar()
		plt.show()
		
		for i in range(n):
			
			sys.stdout.write("\rStep {0} of {1}".format(i+1,n))
			sys.stdout.flush()
			
			if i == 0:
				
				for l in range(num[i]):
					xi = self.getFieldRealizationRealSpaceDirect()
					correlRealizations += np.outer(xi,xi)
			else:
				
				for l in range(num[i]-num[i-1]):
					xi = self.getFieldRealizationRealSpaceDirect()
					correlRealizations += np.outer(xi,xi)
							
			errMatrix = abs(correlRealizations/num[i] - correlExact)
			
			#~ for j in range(self.nx):
				#~ for k in range(self.nx):
					#~ if correlExact[j,k] > 1e-8:
						#~ errMatrix[j,k] /= correlExact[j,k]
						
			err[i] = np.amax(errMatrix)
		
		plt.plot(correlRealizations[0,:]/n2)
		plt.show()
		
		plt.imshow(correlRealizations/n2)
		plt.colorbar()
		plt.show()
		
		plt.figure()
		plt.loglog(num,err,c='blue')
		plt.loglog(num,err,'.',c='blue')
		plt.xlabel(r'$n$')
		plt.ylabel(r'$e$')
		#plt.title(r'Maximum relative error of covariance $e$ between any two points' + '\n' + r'in real space, 1d, for different sample numbers $n$')
		plt.savefig('RealSpaceError1dDirect.pdf')
		plt.close()
		
		print " "
		return
##################################################################	
	def plotFieldRealizationRealSpace(self):
		xi = self.getFieldRealizationRealSpaceDirect()
		plt.figure()
		plt.plot(self.X,xi)
		plt.xlabel(r"$x$")
		plt.ylabel(r"$\xi$")
		plt.show()
		plt.close()
		return
##################################################################	
	def getChiHat(self,kx):
		return np.sqrt(2*np.pi*self.l**2)*self.chi0*np.exp(-0.5*self.l**2*kx**2)
	
	def getChi(self,x):
		return self.chi0*np.exp(-x**2/(2.*self.l**2))
##################################################################
if __name__ == '__main__':
	
	chi0 = 1.
	l = 0.5
	xSz = 2*np.pi
	nx = 32

	rdf = RandField1d(l,chi0,xSz,nx)
	rdf.plotFieldRealizationRealSpace()
	#~ rdf.testErrorConvergenceKSpace(10,1000000,60)
	#~ rdf.testErrorConvergenceRealSpaceSpectral(10,100000,60)
	rdf.testErrorConvergenceRealSpaceDirect(10,10000,60)
