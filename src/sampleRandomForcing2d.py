import numpy as np
import matplotlib.pyplot as plt
import sys
import time

class RandField2d(object):
	
	def __init__(self,l,chi0,xSz,ySz,nx,ny):
		
		self.l = l
		self.chi0 = chi0
		self.xSz = xSz
		self.ySz = ySz
		self.nx = nx
		self.ny = ny 
		self.dx = xSz/nx
		self.dy = ySz/ny
		self.X = np.linspace(0,xSz-self.dx,nx)
		self.Y = np.linspace(0,ySz-self.dy,ny)
		self.KX = 2*np.pi*np.fft.fftfreq(self.nx, self.dx)
		self.KY = 2*np.pi*np.fft.fftfreq(self.ny, self.dy)
		self.Mmax = 7
		self.M = np.linspace(-(self.Mmax-1)/2,(self.Mmax-1)/2,self.Mmax)
		
		self.init_lbda_spec()
		#~ self.init_lbda_direct()
		#~ self.init_lbda_embedded()
		
##################################################################
	def init_lbda_spec(self):
		self.lbda_spec = np.zeros((self.nx,self.ny,2,2))
		
		for i in range(nx):
			for j in range(ny):
				self.lbda_spec[i,j,:,:] = self.getLambda(self.KX[i],self.KY[j])
				
		return
	
	def init_lbda_direct(self):
		Sigma = np.zeros((self.nx*self.ny*2,self.nx*self.ny*2))
		for i in range(self.nx):
			for j in range(self.ny):
				for k in range(self.nx):
					for m in range(self.ny):
						h = self.getChi(self.X[i]-self.X[k], self.Y[j]-self.Y[m])
						Sigma[2 * i * self.ny + 2 * j, 2 * k * self.ny + 2 * m] = h[0,0]
						Sigma[2 * i * self.ny + 2* j + 1, 2 * k * self.ny + 2* m] = h[1,0]
						Sigma[2 * i * self.ny + 2 * j, 2 * k * self.ny + 2* m + 1] = h[0,1]
						Sigma[2 * i * self.ny + 2 * j + 1, 2 * k * self.ny + 2 * m + 1] = h[1,1]
		
		#~ plt.imshow(Sigma) #BT matrix
		#~ plt.gca().xaxis.set_major_locator(plt.NullLocator())
		#~ plt.gca().yaxis.set_major_locator(plt.NullLocator())
		#~ plt.gca().set_axis_off()
		#~ plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
		#~ plt.margins(0,0)
		#~ plt.savefig("bt.pdf", bbox_inches = 'tight', pad_inches = 0)
		#~ plt.show()
		
		D,V = np.linalg.eigh(Sigma)
		print "Maximum eigenvalue of grid covariance matrix ", np.amax(D)
		print "Minimum eigenvalue of grid covariance matrix ", np.amin(D)
		V[:,D<1e-14] = 0
		D[D<1e-14] = 0.
		self.lbda_direct = np.dot(V,np.sqrt(np.diag(D)))
		return
	
	def init_lbda_embedded(self):
		# search for positive definite embedding, need to optimize the search step size, value given in paper is unnecessarily large !!!
		# also, the implementation is not using any properties of block circulant matrices, so it becomes very slow for large embedding matrices...
		
		mxmin_exp = int(np.ceil(np.log(2 * self.nx - 2)/np.log(3))) # minimum grid size (3**mxmin_exp) in x direction to embed into
		mymin_exp = int(np.ceil(np.log(2 * self.ny - 2)/np.log(3)))
		mxmax_exp = mxmin_exp + 10 # maximum grid size which will be tried in order to find a positive definite embedding
		mymax_exp = mymin_exp + 10
		
		for mx_exp in range(mxmin_exp, mxmax_exp):
			for my_exp in range(mymin_exp, mymax_exp): # loop until we find a positive definite embedding
				mx = int(3**mx_exp)
				my = int(3**my_exp)
				Xperio = np.linspace(0., (mx -1)* self.dx, mx)
				Yperio = np.linspace(0., (my -1)* self.dy, my)
				
				Sigma = np.zeros((mx*my*2,mx*my*2))
				print Sigma.shape
				for i in range(mx):
					for j in range(my):
						for k in range(mx):
							for m in range(my):
								# need to find position differences modulo periodic boundaries!
								xdist = np.array([Xperio[i]-Xperio[k],Xperio[i]-Xperio[k] - mx * self.dx , Xperio[i]-Xperio[k] + mx * self.dx])
								x = xdist[np.argmin(abs(xdist))]
								ydist = np.array([Yperio[j]-Yperio[m],Yperio[j]-Yperio[m] - mx * self.dx , Yperio[j]-Yperio[m] + mx * self.dx])
								y = ydist[np.argmin(abs(ydist))]
								
								h = self.getChi(x, y)
								Sigma[2 * i * my + 2 * j, 2 * k * my + 2 * m] = h[0,0]
								Sigma[2 * i * my + 2* j + 1, 2 * k * my + 2* m] = h[1,0]
								Sigma[2 * i * my + 2 * j, 2 * k * my + 2* m + 1] = h[0,1]
								Sigma[2 * i * my + 2 * j + 1, 2 * k * my + 2 * m + 1] = h[1,1]
				
				
				#~ plt.imshow(Sigma) #BC matrix
				#~ plt.gca().xaxis.set_major_locator(plt.NullLocator())
				#~ plt.gca().yaxis.set_major_locator(plt.NullLocator())
				#~ plt.gca().set_axis_off()
				#~ plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
				#~ plt.margins(0,0)
				#~ plt.savefig("bc.pdf", bbox_inches = 'tight', pad_inches = 0)
				#~ plt.show()
				
				D,V = np.linalg.eigh(Sigma)
				print "Maximum eigenvalue of periodic grid covariance matrix ", np.amax(D)
				print "Minimum eigenvalue of periodic grid covariance matrix ", np.amin(D)
				
				if np.amin(D) > 0 or -np.amax(D)/np.amin(D)< 1e-8:
					# stopping condition is arbitrarily chosen so far
					V[:,D<1e-14] = 0
					D[D<1e-14] = 0.
					self.lbda_embedded = np.dot(V,np.sqrt(np.diag(D)))
					return			
##################################################################
	def getFieldRealizationKSpace(self):
		# start in real space with discretized white noise, apply FFT
		xi = 1./np.sqrt(self.dx * self.dy) * (np.random.randn(self.nx,self.ny,2))
		xihat = np.fft.fftn(xi,axes=(0,1),norm='ortho')
		return (2 * np.pi) * np.multiply(self.lbda_spec,xihat[:,:,np.newaxis,:]).sum(axis=3)
	
	def getFieldRealizationKSpaceFast(self):
		# Start directly in Fourier space fpr sampling, thus omitting one FFT!
		# If the pseudo covariance in fourier space is of importance for the application, 
		# one needs to make sure that the symmetry condition following from the noise being real is fulfilled.
		# However, this results in the function being slower than the FFT application in many cases...
		
		#~ what = 2*np.pi/np.sqrt(2* self.dx * self.dy) * (np.random.randn(self.nx,self.ny,2) + 1j * np.random.randn(self.nx,self.ny,2))
		#~ for i in range(self.nx):
			#~ for j in range(self.ny):
				#~ if what[-i,-j,0] == what[i,j,0]:
					#~ what[-i,-j,:].imag = [0,0]
				#~ else:
					#~ what[-i,-j,:] = np.conjugate(what[i,j,:])
		
		# If you don't care about what's happening in Fourier space and only need the final realizations in real space, use the lines below instead
		# Here, one may either take the real or imaginary part of the resulting realization in real space; the correct covariance will be obtained either way
		# Note the missing factor of 1/sqrt(2) in the normalization in this case!
		
		what = 2*np.pi/np.sqrt( self.dx * self.dy) * (np.random.randn(self.nx,self.ny,2) + 1j * np.random.randn(self.nx,self.ny,2))
		return np.multiply(self.lbda_spec,what[:,:,np.newaxis,:]).sum(axis=3)
	
	def getFieldRealizationRealSpaceSpectral(self):
		xihat = self.getFieldRealizationKSpaceFast()
		return 1./(2*np.pi) * np.fft.ifftn(xihat,axes=(0,1),norm='ortho').imag
	
	def getFieldRealizationRealSpaceDirect(self):
		xi = np.random.randn(self.nx*self.ny*2)
		xi = np.dot(self.lbda_direct, xi)
		return np.reshape(xi,(self.nx,self.ny,2))
##################################################################
	def testErrorConvergenceRealSpaceDifferentXSpectral(self,n1,n2,n):
		
		print "--------------------------------------------------"
		print "Testing convergence of correlation function between one fixed point and all others in real space using spectral method"
		print "Total number of samples that will be generated: ", n2
		print "--------------------------------------------------"
		
		num = np.logspace(np.log10(n1),np.log10(n2),n,dtype = int)
		err = np.zeros(n)
		idxX = 0
		idxY = 0
		
		correlRealizations = np.zeros((self.nx,self.ny,4))
		correlExact = np.zeros((self.nx,self.ny,4))
		
		
		for j in range(self.nx):
			for k in range(self.ny):
				
				xarr = self.X[j]-self.X[idxX] + self.M * self.xSz
				yarr = self.Y[k]-self.Y[idxY] + self.M * self.ySz
				correlarr = np.zeros((self.Mmax,self.Mmax,4))
				
				for l in range(self.Mmax):
					for m in range(self.Mmax):
						correlarr[l,m,:] = self.getChi(xarr[l],yarr[m]).reshape(4)
				
				correlExact[j,k,:] = np.sum(correlarr,axis=(0,1))
		
		correlNorm = np.sqrt(np.sum(np.square(abs(correlExact)),axis=2))
		
		plt.imshow(correlNorm)
		plt.colorbar()
		plt.show()
		plt.close()
		
		for i in range(n):
			
			sys.stdout.write("\rStep {0} of {1}".format(i+1,n))
			sys.stdout.flush()
			
			if i == 0:
				
				for l in range(num[i]):
					xi = self.getFieldRealizationRealSpaceSpectral()
					for j in range(self.nx):
						for k in range(self.ny):
							correlRealizations[j,k,:] += np.outer(xi[j,k,:],xi[idxX,idxY,:]).reshape(4)
			else:
				
				for l in range(num[i]-num[i-1]):
					xi = self.getFieldRealizationRealSpaceSpectral()
					for j in range(self.nx):
						for k in range(self.ny):
							correlRealizations[j,k,:] += np.outer(xi[j,k,:],xi[idxX,idxY,:]).reshape(4)
							
			errMatrix = np.square(abs(correlRealizations/num[i] - correlExact))
			errMatrix = np.sqrt(np.sum(errMatrix,axis=2))
			
			#~ for j in range(self.nx):
				#~ for k in range(self.ny):
					#~ if abs(correlNorm[j,k]) > 1e-8:
						#~ errMatrix[j,k] /= correlNorm[j,k]
			
			err[i] = np.amax(errMatrix)
		
		
		plt.imshow(np.sqrt(np.sum(np.square(abs(correlRealizations/num[i])),axis=2)))
		plt.colorbar()
		plt.show()
		plt.close()
		
		plt.figure()
		plt.loglog(num,err,c='blue')
		plt.loglog(num,err,'.',c='blue')
		plt.xlabel(r'$n$')
		plt.ylabel(r'$e$')
		plt.show()
		plt.close()
		
		
		print " "
		return
##################################################################
	def testErrorConvergenceKSpaceSameK(self,n1,n2,n):
		
		print "--------------------------------------------------"
		print "Testing convergence of correlation function on each point in Fourier space"
		print "Total number of samples that will be generated: ", n2
		print "--------------------------------------------------"
		
		num = np.logspace(np.log10(n1),np.log10(n2),n,dtype = int)
		err = np.zeros(n)
		
		correlRealizations = np.zeros((self.nx,self.ny,4),dtype=complex)
		correlExact = np.zeros((self.nx,self.ny,4))
		
		for j in range(self.nx):
			for k in range(self.ny):
				xihat = (2 * np.pi)**2 /(self.dx * self.dy) * self.getChiHat(self.KX[j],self.KY[k])
				correlExact[j,k,:] = xihat.reshape(4)
				
		correlNorm = np.sqrt(np.sum(np.square(abs(correlExact)),axis=2))
		
		plt.imshow(correlNorm)
		plt.colorbar()
		plt.show()
		
		for i in range(n):
			
			sys.stdout.write("\rStep {0} of {1}".format(i+1,n))
			sys.stdout.flush()
			
			if i == 0:
				
				for l in range(num[i]):
					xihat = self.getFieldRealizationKSpace()
					for j in range(self.nx):
						for k in range(self.ny):
							correlRealizations[j,k,:] += np.outer(xihat[j,k,:],np.conjugate(xihat[j,k,:])).reshape(4)
				
			else:
				
				for l in range(num[i]-num[i-1]):
					xihat = self.getFieldRealizationKSpace()
					for j in range(self.nx):
						for k in range(self.ny):
							correlRealizations[j,k,:] += np.outer(xihat[j,k,:],np.conjugate(xihat[j,k,:])).reshape(4)
			
			errMatrix = np.square(abs(correlRealizations/num[i] - correlExact))	
			errMatrix = np.sqrt(np.sum(errMatrix,axis=2))
			
			#~ for j in range(self.nx):
				#~ for k in range(self.ny):
					#~ if abs(correlNorm[j,k]) > 1e-10:
						#~ errMatrix[j,k] /= correlNorm[j,k]
			
			err[i] = np.amax(errMatrix)
		
		print errMatrix
		plt.imshow(np.sqrt(np.sum(np.square(abs(correlRealizations/num[i])),axis=2)))
		plt.colorbar()
		plt.show()
		
		plt.figure()
		plt.loglog(num,err,c='blue')
		plt.loglog(num,err,'.',c='blue')
		plt.xlabel(r'$n$')
		plt.ylabel(r'$e$')
		plt.show()
		plt.close()
		
		print " "
		return
##################################################################
	def testErrorConvergenceKSpaceDifferentK(self,n1,n2,n):
		#numerically compute correlation between different k's and one fixed (kx,ky)
		
		print "--------------------------------------------------"
		print "Testing convergence of correlation function between one fixed point and the other points in Fourier space"
		print "Total number of samples that will be generated: ", n2
		print "--------------------------------------------------"
		
		num = np.logspace(np.log10(n1),np.log10(n2),n,dtype = int)
		err = np.zeros(n)
		
		correlRealizations = np.zeros((self.nx,self.ny,4),dtype=complex)
		correlExact = np.zeros((self.nx,self.ny,4))
		
		xihat = (2 * np.pi)**2 /(self.dx * self.dy) * self.getChiHat(self.KX[-1],self.KY[-1])
		correlExact[-1,-1,:] = xihat.reshape(4)
		
		correlNorm = np.sqrt(np.sum(np.square(abs(correlExact)),axis=2))
		
		for i in range(n):
			
			sys.stdout.write("\rStep {0} of {1}".format(i+1,n))
			sys.stdout.flush()
			
			if i == 0:
				
				for l in range(num[i]):
					xihat = self.getFieldRealizationKSpace()
					for j in range(self.nx):
						for k in range(self.ny):
							correlRealizations[j,k,:] += np.outer(xihat[j,k,:],np.conjugate(xihat[-1,-1,:])).reshape(4)
				
			else:
				
				for l in range(num[i]-num[i-1]):
					xihat = self.getFieldRealizationKSpace()
					for j in range(self.nx):
						for k in range(self.ny):
							correlRealizations[j,k,:] += np.outer(xihat[j,k,:],np.conjugate(xihat[-1,-1,:])).reshape(4)
			
			errMatrix = np.square(abs(correlRealizations/num[i] - correlExact))	
			errMatrix = np.sqrt(np.sum(errMatrix,axis=2))
			
			#~ for j in range(self.nx):
				#~ for k in range(self.ny):
					#~ if abs(correlNorm[j,k]) > 1e-10:
						#~ errMatrix[j,k] /= correlNorm[j,k]
			
			err[i] = np.amax(errMatrix)
		
		plt.figure()
		plt.loglog(num,err,c='blue')
		plt.loglog(num,err,'.',c='blue')
		plt.xlabel(r'$n$')
		plt.ylabel(r'$e$')
		plt.show()
		plt.close()
		
		print " "
		return
##################################################################
	def testErrorConvergenceRealSpaceDifferentXDirect(self,n1,n2,n):
		
		print "--------------------------------------------------"
		print "Testing convergence of correlation function between one fixed point and all others in real space using direct method"
		print "Total number of samples that will be generated: ", n2
		print "--------------------------------------------------"
		
		num = np.logspace(np.log10(n1),np.log10(n2),n,dtype = int)
		err = np.zeros(n)
		idxX = 0
		idxY = 0
		
		correlRealizations = np.zeros((self.nx,self.ny,4))
		correlExact = np.zeros((self.nx,self.ny,4))
		
		
		for j in range(self.nx):
			for k in range(self.ny):
				correlExact[j,k,:] = self.getChi(self.X[j]-self.X[idxX],self.Y[k]-self.Y[idxY]).reshape(4)
		
		correlNorm = np.sqrt(np.sum(np.square(abs(correlExact)),axis=2))
		
		plt.imshow(correlNorm,origin='lower')
		plt.colorbar()
		plt.show()
		plt.close()
		
		for i in range(n):
			
			sys.stdout.write("\rStep {0} of {1}".format(i+1,n))
			sys.stdout.flush()
			
			if i == 0:
				
				for l in range(num[i]):
					xi = self.getFieldRealizationRealSpaceDirect()
					for j in range(self.nx):
						for k in range(self.ny):
							correlRealizations[j,k,:] += np.outer(xi[j,k,:],xi[idxX,idxY,:]).reshape(4)
			else:
				
				for l in range(num[i]-num[i-1]):
					xi = self.getFieldRealizationRealSpaceDirect()
					for j in range(self.nx):
						for k in range(self.ny):
							correlRealizations[j,k,:] += np.outer(xi[j,k,:],xi[idxX,idxY,:]).reshape(4)
							
			errMatrix = np.square(abs(correlRealizations/num[i] - correlExact))
			errMatrix = np.sqrt(np.sum(errMatrix,axis=2))
			
			#~ for j in range(self.nx):
				#~ for k in range(self.ny):
					#~ if abs(correlNorm[j,k]) > 1e-8:
						#~ errMatrix[j,k] /= correlNorm[j,k]
			
			err[i] = np.amax(errMatrix)
		
		plt.imshow(np.sqrt(np.sum(np.square(abs(correlRealizations/num[i])),axis=2)),origin='lower')
		plt.colorbar()
		plt.show()
		plt.close()
		
		plt.figure()
		plt.loglog(num,err,c='blue')
		plt.loglog(num,err,'.',c='blue')
		plt.xlabel(r'$n$')
		plt.ylabel(r'$e$')
		plt.show()
		plt.close()
		
		
		print " "
		return
##################################################################
	def testExectionTime(self):
		N = 10000
		start = time.time()
		for i in range(N):
			self.getFieldRealizationKSpace()
		print "Exection time using FFT: ", (time.time()-start)/N
		
		start = time.time()
		for i in range(N):
			self.getFieldRealizationKSpaceFast()
		print "Exection time without FFT: ", (time.time()-start)/N
		
		return
##################################################################
	def plotFieldRealizationRealSpace(self):
		xi = self.getFieldRealizationRealSpaceSpectral()
		Xmgrd,Ymgrd = np.meshgrid(self.X,self.Y)
		plt.figure()
		plt.pcolormesh(Xmgrd,Ymgrd,np.sqrt(xi[:,:,0]**2+xi[:,:,1]**2),shading='gouraud')
		plt.colorbar()
		plt.quiver(Xmgrd,Ymgrd,xi[:,:,0],xi[:,:,1])
		plt.show()
		plt.close()
		return
##################################################################
	def getLambda(self,kx,ky):
		ret = np.zeros((2,2))
		ret[0,1] = -ky
		ret[1,1] = kx
		ret = ret * np.sqrt(self.chi0) * self.l * np.exp(-0.25 * (kx**2+ky**2) * self.l**2)
		return ret
	
	def getChiHat(self,kx,ky):
		ret = np.zeros((2,2))
		ret[0,0] = ky**2
		ret[0,1] = -kx*ky
		ret[1,0] = -kx*ky
		ret[1,1] = kx**2
		ret = ret * self.chi0*self.l**2*np.exp(-0.5*(kx**2+ky**2)*self.l**2)
		return ret
	
	def getChi(self,x,y):
		ret = np.zeros((2,2))
		ret[0,0] = self.l**2 - y**2
		ret[0,1] = x*y
		ret[1,0] = x*y
		ret[1,1] = self.l**2 - x**2
		ret = ret * self.chi0 / (2 * np.pi * self.l**4) * np.exp(-(x**2+y**2)/2./self.l**2)
		return ret
##################################################################

if __name__ == '__main__':
	
	chi0 = 1.
	l = 1.5
	xSz = 2*np.pi
	ySz = 2*np.pi
	nx = 16
	ny = 16
	
	rdf = RandField2d(l,chi0,xSz,ySz,nx,ny)
	rdf.plotFieldRealizationRealSpace()
	#~ rdf.testExectionTime()
	#~ rdf.testErrorConvergenceKSpaceSameK(10,100000,50)
	#~ rdf.testErrorConvergenceKSpaceDifferentK(10,10000,50)
	rdf.testErrorConvergenceRealSpaceDifferentXSpectral(10,1000,50)
	#~ rdf.testErrorConvergenceRealSpaceDifferentXDirect(10,4000,50)
