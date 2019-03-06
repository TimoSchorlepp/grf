import numpy as np
import matplotlib.pyplot as plt
import sys

class RandField3d(object):
	
	def __init__(self,l,chi0,xSz,ySz,zSz,nx,ny,nz):
		
		self.l = float(l)
		self.chi0 = float(chi0)
		self.xSz = float(xSz)
		self.ySz = float(ySz)
		self.zSz = float(zSz)
		self.nx = nx
		self.ny = ny
		self.nz = nz
		self.dx = xSz/nx
		self.dy = ySz/ny
		self.dz = zSz/nz
		self.X = np.linspace(0,xSz-self.dx,nx)
		self.Y = np.linspace(0,ySz-self.dy,ny)
		self.Z = np.linspace(0,zSz-self.dz,nz)
		self.KX = 2*np.pi*np.fft.fftfreq(self.nx, self.dx)
		self.KY = 2*np.pi*np.fft.fftfreq(self.ny, self.dy)
		self.KZ = 2*np.pi*np.fft.fftfreq(self.nz, self.dz)
		self.Mmax = 7
		self.M = np.linspace(-(self.Mmax-1)/2,(self.Mmax-1)/2,self.Mmax)
		
		self.lbda = np.zeros((self.nx,self.ny,self.nz,3,3))
		for i in range(nx):
			for j in range(ny):
				for k in range(nz):
					self.lbda[i,j,k,:,:] = self.getLambda(self.KX[i],self.KY[j],self.KZ[k])
##################################################################
	def getFieldRealizationKSpace(self):
		xi = 1./np.sqrt(self.dx * self.dy * self.dz) * np.random.randn(self.nx,self.ny,self.nz,3)
		xihat = np.fft.fftn(xi,axes=(0,1,2),norm='ortho')
		return (2*np.pi)**(3./2.) * np.multiply(self.lbda,xihat[:,:,:,np.newaxis,:]).sum(axis=4)
	
	def getFieldRealizationRealSpace(self):
		xihat = self.getFieldRealizationKSpace()
		return np.fft.ifftn(xihat,axes=(0,1,2),norm='ortho').real/(2*np.pi)**(3./2.)
##################################################################
	def testErrorConvergenceRealSpaceDifferentX(self,n1,n2,n):
		
		print "--------------------------------------------------"
		print "Testing convergence of correlation function between one fixed point and all others in real space"
		print "Total number of samples that will be generated: ", n2
		print "--------------------------------------------------"
		
		num = np.logspace(np.log10(n1),np.log10(n2),n,dtype = int)
		err = np.zeros(n)
		idxX = 0
		idxY = 0
		idxZ = 0
		
		correlRealizations = np.zeros((self.nx,self.ny,self.nz,9))
		correlExact = np.zeros((self.nx,self.ny,self.nz,9))
		
		for j in range(self.nx):
			for k in range(self.ny):
				for l in range(self.nz):
					xarr = self.X[j]-self.X[idxX] + self.M * self.xSz
					yarr = self.Y[k]-self.Y[idxY] + self.M * self.ySz
					zarr = self.Z[l]-self.Z[idxZ] + self.M * self.zSz
					correlarr = np.zeros((self.Mmax,self.Mmax,self.Mmax,9))
				
					for u in range(self.Mmax):
						for v in range(self.Mmax):
							for w in range(self.Mmax):
								correlarr[u,v,w,:] = self.getChi(xarr[u],yarr[v],zarr[w]).reshape(9)
					
					correlExact[j,k,l,:] = np.sum(correlarr,axis=(0,1,2))
		
		correlNorm = np.sqrt(np.sum(np.square(abs(correlExact)),axis=3))
		
		for i in range(n):
			
			sys.stdout.write("\rStep {0} of {1}".format(i+1,n))
			sys.stdout.flush()
			
			if i == 0:
				
				for l in range(num[i]):
					xi = self.getFieldRealizationRealSpace()
					for j in range(self.nx):
						for k in range(self.ny):
							for m in range(self.nz):
								correlRealizations[j,k,m,:] += np.outer(xi[j,k,m,:],xi[idxX,idxY,idxZ,:]).reshape(9)
			else:
				
				for l in range(num[i]-num[i-1]):
					xi = self.getFieldRealizationRealSpace()
					for j in range(self.nx):
						for k in range(self.ny):
							for m in range(self.nz):
								correlRealizations[j,k,m,:] += np.outer(xi[j,k,m,:],xi[idxX,idxY,idxZ,:]).reshape(9)
							
			errMatrix = np.square(abs(correlRealizations/num[i] - correlExact))	
			errMatrix = np.sqrt(np.sum(errMatrix,axis=3))
			
			#~ for j in range(self.nx):
				#~ for k in range(self.ny):
					#~ for l in range(self.nz):
						#~ if correlNorm[j,k,l] != 0:
							#~ errMatrix[j,k,l] /= correlNorm[j,k,l]
						
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
	def testErrorConvergenceKSpaceSameK(self,n1,n2,n):
		
		print "--------------------------------------------------"
		print "Testing convergence of correlation function on each point in Fourier space"
		print "Total number of samples that will be generated: ", n2
		print "--------------------------------------------------"
		
		num = np.logspace(np.log10(n1),np.log10(n2),n,dtype = int)
		err = np.zeros(n)
		
		correlRealizations = np.zeros((self.nx,self.ny,self.nz,9),dtype=complex)
		correlExact = np.zeros((self.nx,self.ny,self.nz,9),dtype=complex)
		
		for j in range(self.nx):
			for k in range(self.ny):
				for m in range(self.nz):
					xihat = (2*np.pi)**3/(self.dx*self.dy*self.dz) * self.getChiHat(self.KX[j],self.KY[k],self.KZ[m])
					correlExact[j,k,m,:] = xihat.reshape(9)
				
		correlNorm = np.sqrt(np.sum(np.square(abs(correlExact)),axis=3))
		
		for i in range(n):
			
			sys.stdout.write("\rStep {0} of {1}".format(i+1,n))
			sys.stdout.flush()
			
			if i == 0:
				
				for l in range(num[i]):
					xihat = self.getFieldRealizationKSpace()
					for j in range(self.nx):
						for k in range(self.ny):
							for m in range(self.nz):
								correlRealizations[j,k,m,:] += np.outer(xihat[j,k,m,:],np.conjugate(xihat[j,k,m,:])).reshape(9)
				
			else:
				
				for l in range(num[i]-num[i-1]):
					xihat = self.getFieldRealizationKSpace()
					for j in range(self.nx):
						for k in range(self.ny):
							for m in range(self.nz):
								correlRealizations[j,k,m,:] += np.outer(xihat[j,k,m,:],np.conjugate(xihat[j,k,m,:])).reshape(9)
			
			errMatrix = np.square(abs(correlRealizations/num[i] - correlExact))	
			errMatrix = np.sqrt(np.sum(errMatrix,axis=3))
			
			#~ for j in range(self.nx):
				#~ for k in range(self.ny):
					#~ for l in range(self.nz):
					   #~ if correlNorm[j,k,l] != 0:
						  #~ errMatrix[j,k,l] /= correlNorm[j,k,l]
			
			err[i] = np.amax(errMatrix)
		
		#~ print np.sqrt(np.sum(np.square(abs(correlRealizations/num[i])),axis=3))
		#~ print correlNorm
		
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
		
		print "--------------------------------------------------"
		print "Testing convergence of correlation function between one fixed point and the other points in Fourier space"
		print "Total number of samples that will be generated: ", n2
		print "--------------------------------------------------"
		
		num = np.logspace(np.log10(n1),np.log10(n2),n,dtype = int)
		err = np.zeros(n)
		
		correlRealizations = np.zeros((self.nx,self.ny,self.nz,9),dtype=complex)
		correlExact = np.zeros((self.nx,self.ny,self.nz,9))
		
		xihat = (2*np.pi)**3/(self.dx*self.dy*self.dz) * self.getChiHat(self.KX[-1],self.KY[-1],self.KZ[-1])
		correlExact[-1,-1,-1,:] = xihat.reshape(9)
		
		correlNorm = np.sqrt(np.sum(np.square(abs(correlExact)),axis=3))
		
		for i in range(n):
			
			sys.stdout.write("\rStep {0} of {1}".format(i+1,n))
			sys.stdout.flush()
			
			if i == 0:
				
				for l in range(num[i]):
					xihat = self.getFieldRealizationKSpace()
					for j in range(self.nx):
						for k in range(self.ny):
							for m in range(self.nz):
								correlRealizations[j,k,m,:] += np.outer(xihat[j,k,m,:],np.conjugate(xihat[-1,-1,-1,:])).reshape(9)
				
			else:
				
				for l in range(num[i]-num[i-1]):
					xihat = self.getFieldRealizationKSpace()
					for j in range(self.nx):
						for k in range(self.ny):
							for m in range(self.nz):
								correlRealizations[j,k,m,:] += np.outer(xihat[j,k,m,:],np.conjugate(xihat[-1,-1,-1,:])).reshape(9)
			
			errMatrix = np.square(abs(correlRealizations/num[i] - correlExact))	
			errMatrix = np.sqrt(np.sum(errMatrix,axis=3))
			
			#~ for j in range(self.nx):
				#~ for k in range(self.ny):
					#~ for l in range(self.nz):
						#~ if correlNorm[j,k,l] != 0:
							#~ errMatrix[j,k,l] /= correlNorm[j,k,l]
			
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
	def plotFieldRealizationRealSpace(self):
		xi = self.getFieldRealizationRealSpace()
		idx = 3
		Ymgrd,Xmgrd = np.meshgrid(self.Y,self.X)
		plt.figure()
		plt.pcolormesh(Xmgrd,Ymgrd,np.sqrt(xi[:,:,idx,0]**2+xi[:,:,idx,1]**2+xi[:,:,idx,2]**2),shading='gouraud')
		plt.colorbar()
		plt.quiver(Xmgrd,Ymgrd,xi[:,:,idx,0],xi[:,:,idx,1])
		plt.show()
		plt.close()
		return
##################################################################	
	def getLambda(self,kx,ky,kz):
		ret = np.zeros((3,3))
		k = np.sqrt(kx**2+ky**2+kz**2)
		k1 = np.sqrt(kx**2+ky**2)
		k2 = np.sqrt(kx**2*kz**2+ky**2*kz**2+(kx**2+ky**2)**2)
		if k1 != 0:
			ret[0,1] = -ky/k1
			ret[1,1] = kx/k1
			ret[0,2] = -kx*kz/k2
			ret[1,2] = -ky*kz/k2
			ret[2,2] = k1**2/k2
		elif kz != 0:
			# choose different orthonormal basis of eigenvectors in this case
			ret[1,1] = -1.
			ret[0,2] = -1
		ret = ret * 2**0.25 * np.pi**0.75 * np.sqrt(self.chi0) * self.l**2.5 * k * np.exp(-0.25 * k**2 * self.l**2)
		return ret
	
	def getChiHat(self,kx,ky,kz):
		ret = np.zeros((3,3))
		ret[0,0] = ky**2+kz**2
		ret[0,1] = -kx*ky
		ret[0,2] = -kx*kz
		ret[1,0] = -kx*ky
		ret[1,1] = kx**2+kz**2
		ret[1,2] = -ky*kz
		ret[2,0] = -kx*kz
		ret[2,1] = -ky*kz
		ret[2,2] = kx**2+ky**2
		ret = ret * np.sqrt(2*np.pi**3)*self.chi0*self.l**5*np.exp(-0.5*(kx**2+ky**2+kz**2)*self.l**2)
		return ret
	
	def getChi(self,x,y,z):
		ret = np.zeros((3,3))
		ret[0,0] = 2*self.l**2 - y**2 - z**2
		ret[0,1] = x*y
		ret[0,2] = x*z
		ret[1,0] = x*y
		ret[1,1] = 2*self.l**2 - x**2 - z**2
		ret[1,2] = y*z
		ret[2,0] = x*z
		ret[2,1] = y*z
		ret[2,2] = 2*self.l**2 - x**2 - y**2
		ret = ret * self.chi0 / (2. * self.l**2) * np.exp(-(x**2+y**2+z**2)/2./self.l**2)
		return ret
##################################################################
if __name__ == '__main__':
	
	chi0 = 1.
	l = 1.
	xSz = 2*np.pi
	ySz = 2*np.pi
	zSz = 2*np.pi
	nx = 16
	ny = 16
	nz = 16
	
	rdf = RandField3d(l,chi0,xSz,ySz,zSz,nx,ny,nz)
	rdf.plotFieldRealizationRealSpace()
	#~ rdf.testErrorConvergenceKSpaceSameK(10,1000,40)
	#~ rdf.testErrorConvergenceKSpaceDifferentK(10,10000,40)
	#~ rdf.testErrorConvergenceRealSpaceDifferentX(10,100000,40)
