import numpy as np
from numpy.linalg import inv
from numpy.linalg import svd
from numpy.linalg import eig
from scipy.optimize import leastsq,least_squares,fmin



class EpipolarGeometry: 
	'''
	Class that implements basic epipolar geometry operations 
	'''


	def fundamental_matrix(self,view1_feat2D,view2_feat2D,optimize):
		'''
		Method to computes the fundamental matrix betwen two images. 
		Args: 
			view1_feat2D: 2D feature coordinates in view 1
			view2_feat2D: 2D feature coordinates in view 2
			optimize:  optimize the F estimation using non-linear least squares 
		Returns:
			the fundamendal matrix F
		'''
		number_of_features=view1_feat2D.shape[1]
		
		#convert to homogeneous coordinates
		view1_feat2D=np.vstack((view1_feat2D,np.ones(number_of_features)))
		view2_feat2D=np.vstack((view2_feat2D,np.ones(number_of_features)))

		# create a homogeneous system Af=0 (f : elements of fundamental matrix), using mFm'=0
		A=np.zeros((number_of_features,9))
		A[:,0]=np.transpose(view2_feat2D[0,:])*(view1_feat2D[0,:])
		A[:,1]=np.transpose(view2_feat2D[0,:])*(view1_feat2D[1,:])
		A[:,2]=np.transpose(view2_feat2D[0,:])
		A[:,3]=np.transpose(view1_feat2D[0,:])*(view1_feat2D[1,:])
		A[:,4]=np.transpose(view2_feat2D[1,:])*(view1_feat2D[1,:])
		A[:,5]=np.transpose(view2_feat2D[1,:])
		A[:,6]=np.transpose(view1_feat2D[0,:])
		A[:,7]=np.transpose(view1_feat2D[1,:])
		A[:,8]=np.ones(number_of_features)


		# use the eigenvector that corresponds to the smallest eigenvalue as initial estimate for F.
		U, s, Vh = svd(A)
		V=np.transpose(Vh)

		# fundamental matrix is now the eigenvector corresponding to the smallest eigen value.
		F=np.reshape(V[:,8],(3,3))

		# make sure that fundamental matrix is of rank 2
		U,s,V = svd(F);
		s[2]=0
		S = np.diag(s)
		F=np.dot(U, np.dot(S, V))

		if optimize==1:
			#Optimize initial estimate using the algebraic error
			f=np.append(np.concatenate((F[0,:],F[1,:]),axis=0),F[2,0])/F[2,2]
	
			result=least_squares(self.fundamental_matrix_error,f, args=(view1_feat2D[0:2,:],view2_feat2D[0:2,:]))
			f=result.x

			F=np.asarray([np.transpose(f[0:3]), np.transpose(f[3:6]), [f[6],-(-f[0]*f[4]+f[6]*f[2]*f[4]+f[3]*f[1]-f[6]*f[1]*f[5])/(-f[3]*f[2]+f[0]*f[5]),1]]);
			F=F/sum(sum(F))*9;

		return F	


	def fundamental_matrix_error(self,f,view1_feat2D,view2_feat2D):
		'''
		Method to compute the fundamental matrix error based on the epipolar constraint (mFm'=0)
		Args: 
			f : a vector with the elements of the fundamental matrix
			view1_feat2D: 2D feature coordinates in view 1
 			view2_feat2D: 2D feature coordinates in view 2
 		Returns: 
 			the error based on mFm'=0

		'''
		F=np.asarray([np.transpose(f[0:3]), np.transpose(f[3:6]), [f[6],-(-f[0]*f[4]+f[6]*f[2]*f[4]+f[3]*f[1]-f[6]*f[1]*f[5])/(-f[3]*f[2]+f[0]*f[5]),1]]);

		number_of_features=view1_feat2D.shape[1]
		#convert to homogeneous coordinates
		view1_feat2D=np.vstack((view1_feat2D,np.ones(number_of_features)))
		view2_feat2D=np.vstack((view2_feat2D,np.ones(number_of_features)))
		#compute error
		error=np.zeros((number_of_features,1))
		for i in range(0,number_of_features):
			error[i]=np.dot(view2_feat2D[:,i],np.dot(F,np.transpose(view1_feat2D[:,i])));

		return error.flatten()

	def get_epipole(self,F):
		'''
		Method to return the epipole from the fundamental matrix
		Args: 
			F: the fundamental matrix
		Retruns: 
			the epipole
		'''
		u,v=eig(F)
		dd=np.square(u);
		min_index=np.argmin(np.abs(dd))	# SOS: These are complex numbers, so compare their modulus
		epipole=v[:,min_index]
		epipole=epipole/epipole[2]
		return epipole.real

	def compute_homography(self,epipole,F):
		'''
		Method that computes the homograhpy [epipole]x[F]
		Args: 
			epipole: the epipole
			F : the fundamental matrix
		Returns: 
			the homography H
		'''
		e_12=np.asarray([[0,-epipole[2], epipole[1]],[epipole[2],0,-epipole[0]],[-epipole[1],epipole[0],0]]);
		H=np.dot(e_12,F)
		H=H*np.sign(np.trace(H))
		return H


	def triangulate_points(self,view1_feat2D,view2_feat2D,P1,P2):
		'''
		Method that triangulates using two projection matrices, and the
		normalized 2D features coordinates.
		Args: 
			view1_feat2D: 2D feature coordinates in view 1
			view2_feat2D: 2D feature coordinates in view 2
			P1: projection matrix for view 1
			P2: projection matrix for view 2
		Returns: 
			the 3D points
		'''
		A=np.zeros((4,4));
		A[0,:]=P1[2,:]*view1_feat2D[0]-P1[0,:]
		A[1,:]=P1[2,:]*view1_feat2D[1]-P1[1,:]
		A[2,:]=P2[2,:]*view2_feat2D[0]-P2[0,:]
		A[3,:]=P2[2,:]*view2_feat2D[1]-P2[1,:]

		U, s, Vh = svd(A)
		V=np.transpose(Vh)
		feat3D=V[:,V.shape[0]-1]
		feat3D=feat3D/feat3D[3]
		return feat3D	


	def compute_L_R(self,view_feat2D,epipole):
		'''
		Utility method used in polynomial triangulation
		'''
		L=np.eye(3);
		L[0:2,2]=-view_feat2D; 

		th = np.arctan(-(epipole[1]-epipole[2]*view_feat2D[1])/(epipole[0]-epipole[2]*view_feat2D[0]));
		R=np.eye(3)
		R[0,0]=np.cos(th)
		R[1,1]=np.cos(th)
		R[0,1]=-np.sin(th)
		R[1,0]=np.sin(th)
		return L,R

	def polynomial_triangulation(self,feat1,feat2,epipole_1,epipole_2,F,P1,P2):
		'''
		Method to perform 'polynomial' triangulation, as suggested in 
		R. Hartley and P. Sturm, "Triangulation", Computer Vision and Image Understanding, 68(2):146-157, 1997
		The method searches for the epipolar line which fits best to both feature points, and then project both points on tis line. Triangulation
		is then performed on these new points. (This now happens through SVD since the error will be zero)
		Args: 
			feat1: 2D feature coordinates in view 1
			feat2: 2D feature coordinates in view 2
			epipole_1: the epipole of view 1
			epipole_2: the epipole of view 2
			F: the fundamental matrix 
			P1: projection matrix for view 1
			P2: projection matrix for view 2
		Returns: 
			the 3D points
		'''
		L1,R1=self.compute_L_R(feat1,epipole_1)
		L2,R2=self.compute_L_R(feat2,epipole_2)

		F1=R2.dot(np.transpose(inv(L2))).dot(F).dot(inv(L1)).dot(np.transpose(R1))
		a=F1[1,1];
		b=F1[1,2];
		c=F1[2,1];
		d=F1[2,2];
		f1=-F1[1,0]/b;
		f2=-F1[0,1]/c;

		p=np.zeros(7);
		p[0]=-2*np.power(f1,4)*np.power(a,2)*c*d+2*np.power(f1,4)*a*np.power(c,2)*b;
		p[1]=2*np.power(a,4)+2*np.power(f2,4)*np.power(c,4)-2*np.power(f1,4)*np.power(a,2)*np.power(d,2)+2*np.power(f1,4)*np.power(b,2)*np.power(c,2)+4*np.power(a,2)*np.power(f2,2)*np.power(c,2);
		p[2]=8*np.power(f2,4)*d*np.power(c,3)+8*b*np.power(a,3)+8*b*a*np.power(f2,2)*np.power(c,2)+8*np.power(f2,2)*d*c*np.power(a,2)-4*np.power(f1,2)*np.power(a,2)*c*d+4*np.power(f1,2)*a*np.power(c,2)*b-2*np.power(f1,4)*b*np.power(d,2)*a+2*np.power(f1,4)*np.power(b,2)*d*c;
		p[3]=-4*np.power(f1,2)*np.power(a,2)*np.power(d,2)+4*np.power(f1,2)*np.power(b,2)*np.power(c,2)+4*np.power(b,2)*np.power(f2,2)*np.power(c,2)+16*b*a*np.power(f2,2)*d*c+12*np.power(f2,4)*np.power(d,2)*np.power(c,2)+12*np.power(b,2)*np.power(a,2)+4*np.power(f2,2)*np.power(d,2)*np.power(a,2);
		p[4]=8*np.power(f2,4)*np.power(d,3)*c+8*np.power(b,2)*np.power(f2,2)*d*c+8*np.power(f2,2)*np.power(d,2)*b*a-4*np.power(f1,2)*b*np.power(d,2)*a+4*np.power(f1,2)*np.power(b,2)*d*c+2*a*np.power(c,2)*b+8*np.power(b,3)*a-2*np.power(a,2)*c*d;
		p[5]=4*np.power(b,2)*np.power(f2,2)*np.power(d,2)+2*np.power(f2,4)*np.power(d,4)+2*np.power(b,2)*np.power(c,2)-2*np.power(a,2)*np.power(d,2)+2*np.power(b,4);
		p[6]=-2*b*d*(a*d-b*c);
		
		r=np.roots(p)
		y=np.polyval(p,r)
		
		aa=max(y)
		i=0;
		ii=0;
		
		for i in range(0,6):
			if ((np.imag(r[i])==0) and (abs(y[i])<aa)):
				aa=abs(y[i]);
				ii=i;
			i=i+1;

		t=r[ii].real;

		#get the refined feature coordinates
		u=np.asarray([f1,1/t,1]);
		v=np.asarray([-1/f1,t,1]);
		s=inv([[u[0],-v[0]],[u[1],-v[1]]]).dot([1/f1,0])
		refined_feat1=np.asarray([u[0]*s[0],u[1]*s[0],1])

		u=F1.dot([0,t,1]);
		u=u/u[2];
		v=[1/u[0],-1/u[1],1];
		s=inv([[u[0],-v[0]],[u[1],-v[1]]]).dot([1/f2,0]);
		refined_feat2=np.asarray([u[0]*s[0],u[1]*s[0],1]);

		refined_feat1=inv(L1).dot(np.transpose(R1)).dot(refined_feat1);
		refined_feat2=inv(L2).dot(np.transpose(R2)).dot(refined_feat2);

		# normalize to get homogeneous coordinates
		refined_feat1=refined_feat1/refined_feat1[2]
		refined_feat2=refined_feat2/refined_feat2[2]

		return self.triangulate_points(refined_feat1,refined_feat2,P1,P2);


	def compute_reprojection_error_point(self,P,point3D,view_feat2D):
		'''
		Method to compute the reprojection error of a 3D point on a particular view
		Args: 
			P: the projection matrix of the view
			point3D: the 3D point coordinates
			view_feat2D: 2D feature coordinates for that point in that view
		Returns: 
			the reprojection error
		'''
		return np.power((P[0:2,:].dot(point3D)/P[2,:].dot(point3D)-view_feat2D),2)

	def compute_reprojection_error_points(self,P,feat3D,view_feat2D):
		'''
		Method that returns the reprojection error for a cloud of 3D points.
		Args: 
			P: the projection matrix of the view
			feat3D: the 3D point cloud coordinates
			view_feat2D: 2D feature coordinates for all the 3D points in that view
		Returns: 
			the reprojection error
		'''
		reproj_feature=P.dot(np.transpose(feat3D))
		reproj_feature=reproj_feature/reproj_feature[2,:]
		error=sum(sum(np.power((view_feat2D-reproj_feature[0:2,:]),2)))
		return error

	def refine_projection_matrix(self,VV,feat3D,view_feat2D):
		'''
		Utility method, used to optimize the projection matrix estimation
		Args: 
			VV: vector containing elements of the projection matrix of the i-th view
			feat3D: 3D point cloud coordinates
			view_feat2D: 2D feature coordinate for all the feature points in the i-th view 
		'''
		#construct the projection matrix
		P=np.zeros(shape=[3,4]);
		P[0,:]=VV[0:4]
		P[1,:]=VV[4:8]
		P[2,:]=np.append(np.append(VV[8:10],1),VV[10])
		return self.compute_reprojection_error_points(P,feat3D,view_feat2D)
	

	def overall_reprojection_error(self,X,feat2d):
		'''
		Method to compute the overall reprojection error by reprojecting the 3D model back to each images. This is 
		used in bundle adustment to optimize the structure (i.e. 3D points), and the motion (i.e. projection matrices)
		The 
		Args: 
			X: vector containing the projection matrices and the 3D coordinates (all concatenated in one vector)
			feat2D: 2D feature coordinates for all the features for all views
		Returns: 
			the overall reprojection error
		'''
		number_of_features=feat2d.shape[2]
		sequence_length=feat2d.shape[0]
		error_vector=[];
		for img_number in range(0,sequence_length):
			for feat_id in range(0,number_of_features):
				img_feat=feat2d[img_number,:,feat_id]
				
				a=np.transpose(X[(0+img_number*11):(4+img_number*11)])
				a1=np.transpose(X[(4+img_number*11):(8+img_number*11)])
				b=np.append(X[(0+sequence_length*11+feat_id*3):(3+sequence_length*11+feat_id*3)],1)
				c= np.append(X[8+img_number*11:10+img_number*11],1)
				c= np.append(c,X[10+img_number*11]);
				c=np.transpose(c);
				
				error_x=img_feat[0]-a.dot(b)/c.dot(b)
				error_y=img_feat[1]-a1.dot(b)/c.dot(b)
			
				error_vector.append(error_x)
				error_vector.append(error_y)

		return np.asarray(error_vector)	