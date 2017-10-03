import numpy as np;
from numpy.linalg import inv
from numpy.linalg import svd
from numpy.linalg import eig
from numpy.linalg import det
from scipy.optimize import leastsq,least_squares,fmin
import pandas as pd
import numpy as np
import time
import random
from stl import mesh
from scipy.spatial import Delaunay


class RecModel:
	'''
	Class to hold all the result of a reconstruction
	'''
	def __init__(self):
		self._P = 0			#projection matrices for each view
		self._R = 0			#rotation matrices for each view
		self._t = 0			#translation vectors for each view
		self._points3D=0	#3D points in metric space
		self._Tm=0			#Transformation matrix from projective to metric


	@property
	def P(self):
		return self._P

	@P.setter
	def P(self,val):
		self._R,self._t=extract_rotation_translation(P)
		self._P=val

	@property
	def points3D(self):
		return self._points3D

	@points3D.setter
	def points3D(self,val):
		self._points3D=val


	def extract_rotation_translation(self, P):  
		'''
		Method to extract the rotation matrices and translation vectors from the projection matrices
		'''
		sequence_length=P.shape[2]
		R=np.zeros((3,3,self._sequence_length))
		t=np.zeros((3,self._sequence_length))

		for i in range(1,self._sequence_length):   
			PP=inv(K[:,:,i]).dot(P[:,:,i]);
			R[:,:,i]=np.transpose(PP[0:3,0:3])  
			t[:,i]=-inv(np.transpose(R[:,:,i])).dot(PP[0:3,3])
		return R,t

	def export_stl_file(self,filename):
		'''
		Method that generates an .stl file with the generated model
		'''
		vertices=self.points3D[:,0:3];

		faces=Delaunay(vertices[:,0:2],incremental=1);
		faces= faces.simplices;
		
		wireframe = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
		for i, f in enumerate(faces):
		    for j in range(3):
		        wireframe.vectors[i][j] = vertices[f[j],:]

		wireframe.save(filename)

