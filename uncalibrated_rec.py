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
import os.path
import argparse

from epipolar_geometry import EpipolarGeometry
from model import RecModel 
from image_sequence import ImageSequence

class UncalibratedReconstruction: 
	'''
	Class that contains high level methods to perform 3D reconstruction from a sequence of uncalibrated images.
	'''
	def __init__(self,sequence_length,width,height,triang_method=0,opt_triang=0,opt_f=1,self_foc=0):
		'''
		Constructor
		Args: 
			sequence_length: number of images (views)
			width: width of the images
			height: height of the images
			triang_method: triangulation method (0: standard, 1: polynomial)
			opt_triang: optimize initial 3D point estimate
			opt_f: optimize fundamental matrix estimation
			self_foc: for self-calibration, type of focal length expected across views (0: fixed, 1: varying )

		'''
		self._eg_utils=EpipolarGeometry()

		#things that are needed throught the class
		self._sequence_length=sequence_length
		self._width=width;
		self._height=height;
		self._mm=(width+height)/2;	

		self._triangulation_method=triang_method	
		self._optimize_triangulation=opt_triang		 
		self._optimize_f=opt_f						
		self._self_foc=self_foc						



	def two_view_geometry_computation(self,view1_feat2D,view2_feat2D):
		'''
		Method to compute the fundamental matrix and epipoles for two views
		Args: 
			view1_feat2D: 2D feature coordinates in view 1
			view2_feat2D: 2D feature coordinates in view 2
		Returns: 
			F: the fundamental matrix 
			epipole_1: view1 epipole
			epipole_2: view2 epipoles

		'''
		F=self._eg_utils.fundamental_matrix(view1_feat2D,view2_feat2D,self._optimize_f)
		epipole_1=self._eg_utils.get_epipole(F)
		epipole_2=self._eg_utils.get_epipole(np.transpose(F))
		return F,epipole_1,epipole_2

	def compute_reference_frame(self,epipole,F):
		'''
		Method to compute the reference frame of the reconstruction (i.e. plane at infinity in an affine or metric space).
		Args: 
			epipole: the epipole
			F: the fundamental matrix
		Returns:
			p: the reference plane
			h: the homography [e]xF

		'''
		H=self._eg_utils.compute_homography(epipole,F)	#compute the homography [e]xF 
		# get the reference plane 
		p = np.sum(np.divide(np.eye(3)-H, np.transpose(np.asarray([epipole, epipole, epipole]))),axis=0)/3
		# adjust reference plane to make the first two projection matrices as equal as possible
		p=fmin(self.init_plane,np.append(p,1),xtol=1e-25,ftol=1e-25,args=(H.real,epipole.real));
		p=p[0:3]
		return p, H

	def init_plane(self,p,H,epi):
		'''
		Error function to make the difference between the first two projection matrices as small as possible
		Note: assuming that the two views used for the initial reconstruction are not too far apart (this their projection matrices are almost equal), has proven to give good results
		Args: 
			p: the reference plane (i.e. plane at infinity)
			H: homography [e]x[F]
			epi: epipola
		Returns: 
			error: difference between two projection matrices
		'''
		epi=np.reshape(epi,(3,1));
		p=np.reshape(p,(1,4));

		t=p[0,0:3]
		t=np.reshape(t,(1,3))
		
		error =sum(sum(abs(H+epi.dot(t)-p[0,3]*np.eye(3))));
		
		return error

	def estimate_initial_projection_matrices(self,H,epipole_2,p):
		'''
		Method to estimate the projection matrices for the two views (i.e. P1=[I | 0], P2=[H+epi1|e])
		Args: 
			H: homography [e]x[F]
			epipole_2: epipole in the 2nd view
			p: the reference plane of the reconstruction (i.e. plane at infinity)
		Returns: 
			P: projection matrices for these two views 

		'''
		P=np.zeros((3,4,self._sequence_length))
		P[:,:,0] = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]; # P1=[I | 0], i.e. frist frame aligned with world frame
		epi_tmp=np.reshape(epipole_2,(3,1));	# P2=[H+epi1|e]
		P[:,:3,1]=H+epi_tmp.dot(np.reshape(p,(1,3)))
		P[:,3,1]= epipole_2
		P[:,:,1]= P[:,:,1]/P[2,2,1]
		return P


	def get_initial_structure(self,feat_2D,P,epipole_1,epipole_2,F):
		'''
		Method to get an initial 3D structure (i.e. 3D point cloud), from the first two projection matrices through triangulation.
		Args: 
			feat_2D: 2D feature coordinates for all images
			P: projection matrices for all views (only the first two views are used)
			epipole_1: view 1 epipole
			epipole_2: view 2 epipole
			F: fundamental matrix
		Returns: 
			points3D: 3D point cloud
		'''
		number_of_features=feat_2D.shape[2]
		points3D=np.zeros(shape=[number_of_features,4]);
		for i in range(0,number_of_features):
			
			if (self._triangulation_method==0):	
				x=self._eg_utils.triangulate_points(feat_2D[0,:,i],feat_2D[1,:,i],P[:,:,0],P[:,:,1]);
				x=x[0:3]
			elif (self._triangulation_method==1):
				x=self._eg_utils.polynomial_triangulation(feat_2D[1,:,i],feat_2D[1,:,i],epipole_1,epipole_2,F,P[:,:,0],P[:,:,1])
				x=x[0:3]/x[3] # normalize 

			if (self._optimize_triangulation==1):
			#refine 3D point estimation (due to noise, lines of sight may not intersect perfectly). Minimizations should be carried out in the images
			# and not in the projective 3D space, thus the reprojection error is used.
				x=fmin(self.refine_3d_point,x, xtol=1e-25,ftol=1e-25, full_output=0, args=(P[:,:,0],P[:,:,1],feat_2D[0,:,i],feat_2D[1,:,i]))

			points3D[i,:]=np.append(x,1)

		return points3D


	def refine_3d_point(self,point3D,P1,P2,view1_feat2D,view2_feat2D):
		'''
		Method to compute the reprojection error of a 3D point in two views
		Args: 
			point3D: 3D point cloud
			P1: projection matrix of view 1
			P2: projection matrix of view 2
			view1_feat2D: 2D feature coordinates in view 1
			view2_feat2D: 2D feature coordinates in view 1
		Returns: 
			error: the reprojection error

		'''
		point3D=np.append(point3D,1);
		error=sum(self.compute_reprojection_error_point(P1,point3D,view1_feat2D)+self.compute_reprojection_error_point(P2,point3D,view2_feat2D))
		# sdfds
		return error

	def projective_pose_estimation(self,feat_2D,P,points3D):
		'''
		Method to add views using an initial 3D structure, i.e. compute the projection matrices for all the additional views (the first two are already
		estimated in previous steps)
		Args: 
			feat_2D: 2D feature coordinates for all images
			P: projection matrices
			points3d: 3D point cloud
		Returns: 
			P: projection matrices for all views
		'''
		number_of_features=feat_2D.shape[2]

		AA=np.zeros(shape=[2*number_of_features,12]);

		for i in range(2,self._sequence_length): 
			for j in range(0,number_of_features):
				AA[2*j,0:4]=points3D[j];
				AA[2*j,8:12]=-feat_2D[i,0,j]*points3D[j]
				AA[2*j+1,4:8]=points3D[j];
				AA[2*j+1,8:12]=-feat_2D[i,1,j]*points3D[j]

			U, s, Vh = svd(AA)
			V=np.transpose(Vh)

			VV=V[0:12,11]
			VV=VV/VV[10]
			VV=np.delete(VV,10)

			#refine the estimate for the i-th projection matrix
			result=least_squares(self._eg_utils.refine_projection_matrix,VV, args=(points3D,feat_2D[i,:,:]))
			VV=result.x

			Pr=np.zeros(shape=[3,4]);
			Pr[0,:]=VV[0:4]
			Pr[1,:]=VV[4:8]
			Pr[2,:]=np.append(np.append(VV[8:10],1),VV[10])
			P[:,:,i]=Pr

		return P

	def bundle_adjustment(self,feat_2D,P,feat3D): 
		'''
		Method to refine structure and motion, i.e. refine the projection matrices and 3D points using the reprojection error
		Args: 
			feat_2D: 2D feature coordinates for all images
			P: projection matrices
			points3d: 3D point cloud
		Returns: 
			P: the refined projection matrices
			feat3D: the refined 3D point cloud
			error: the reprojection error
		'''
		number_of_features=feat_2D.shape[2]

		#The vector to be optimized 
		X=np.reshape(P[:,:,0],(1,12));

		# Append the projection matrices...
		for i in range(1,self._sequence_length):
			X=np.append(X,np.reshape(P[:,:,i],(1,12)))
		X=np.delete(X,[10,22,(self._sequence_length-1)*12+10])

		# ...and then append the 3D points
		X=np.append(X,np.reshape(feat3D[:,0:3],number_of_features*self._sequence_length))

		# Optimize using Levenberg-Marquardt 
		result=least_squares(self._eg_utils.overall_reprojection_error,X, max_nfev=1000,method='lm',args=([feat_2D]))
		X=result.x
		error=np.power(sum(self._eg_utils.overall_reprojection_error(X,feat_2D)),2)

		#get the refined projection matrices from the optimal vector
		for i in range(0,self._sequence_length):
			P[:,:,i]=np.reshape(X[0+i*11:12+i*11],(3,4));
			P[2,3,i]=P[2,2,i]
			P[2,2,i]=1

		#get the refined 3D coordinates from the optimal vector
		feat3D[:,0:3]=np.reshape(X[self._sequence_length*11:self._sequence_length*11+self._sequence_length*number_of_features*3],(number_of_features,3))

		Tp1= np.vstack([P[:,:,0],[0,0,0,1]]);
		for i in range(0,self._sequence_length):
			P[:,:,i]=P[:,:,i].dot(inv(Tp1))

		feat3D=Tp1.dot(np.transpose(feat3D))
		feat3D=np.transpose(feat3D/feat3D[3,:]);

		return P,feat3D,error

	def self_calibration(self,P):
		'''
		Self calibration using the procedure described in 
		M. Pollefeys, R. Koch and L. Van Gool, "Self-Calibration and Metric Reconstruction in spite of Varying and Unknown Internal Camera Parameters", Proc. International Conference on Computer Vision, Narosa Publishing House, pp.90-95, 1998.
		Args: 
			P: projection matrices
		Returns: 
			Tm: transformation matrix that will transform from the projective space to metric space
			K: camera intrisic parameters for each view
			error: the reprojection error

		'''
		# setup the system of equations
		AAA=np.zeros(shape=[4*self._sequence_length-4,6]);
		for i in range(0,self._sequence_length-1):
			P_tmp=P[:,:,i+1]
			AAA[0+4*i,:]=[(-np.power(P_tmp[1, 1],2)+np.power(P_tmp[0, 1],2)-np.power(P_tmp[1, 0],2)+np.power(P_tmp[0, 0],2)) ,(-2*P_tmp[1, 0]*P_tmp[1, 3]+2*P_tmp[0, 0]*P_tmp[0, 3]),(-2*P_tmp[1, 1]*P_tmp[1, 3]+2*P_tmp[0, 1]*P_tmp[0, 3]),(2*P_tmp[0, 2]*P_tmp[0, 3]-2*P_tmp[1, 2]*P_tmp[1, 3]),(-np.power(P_tmp[1, 3],2)+np.power(P_tmp[0, 3],2)),(-np.power(P_tmp[1, 2],2)+np.power(P_tmp[0, 2],2))];
			AAA[1+4*i,:]=[(P_tmp[1, 0]*P_tmp[0, 0]+P_tmp[1, 1]*P_tmp[0, 1]),(P_tmp[1, 0]*P_tmp[0, 3]+P_tmp[1, 3]*P_tmp[0, 0]),(P_tmp[1, 1]*P_tmp[0, 3]+P_tmp[1, 3]*P_tmp[0, 1]),(P_tmp[1, 2]*P_tmp[0, 3]+P_tmp[1, 3]*P_tmp[0, 2]),P_tmp[1, 3]*P_tmp[0, 3],P_tmp[1, 2]*P_tmp[0, 2]];    
			AAA[2+4*i,:]=[(P_tmp[2, 0]*P_tmp[0, 0]+P_tmp[2, 1]*P_tmp[0, 1]),(P_tmp[2, 0]*P_tmp[0, 3]+P_tmp[2, 3]*P_tmp[0, 0]),(P_tmp[2, 1]*P_tmp[0, 3]+P_tmp[2, 3]*P_tmp[0, 1]),(P_tmp[2, 2]*P_tmp[0, 3]+P_tmp[2, 3]*P_tmp[0, 2]),P_tmp[2, 3]*P_tmp[0, 3],P_tmp[2, 2]*P_tmp[0, 2]];
			AAA[3+4*i,:]=[(P_tmp[2, 0]*P_tmp[1, 0]+P_tmp[2, 1]*P_tmp[1, 1]),(P_tmp[2, 0]*P_tmp[1, 3]+P_tmp[2, 3]*P_tmp[1, 0]),(P_tmp[2, 1]*P_tmp[1, 3]+P_tmp[2, 3]*P_tmp[1, 1]),(P_tmp[2, 2]*P_tmp[1, 3]+P_tmp[2, 3]*P_tmp[1, 2]),P_tmp[2, 3]*P_tmp[1, 3],P_tmp[2, 2]*P_tmp[1, 2]];

		U, s, Vh = svd(AAA)
		V=np.transpose(Vh)
		x=V[0:5,5]/V[5,5];
		jj=np.sign(x[0])
		b=x*np.sign(x[0]);

		# initial estimate of the absolute conic
		W=np.asarray([[b[0],0,0,b[1]],[0,b[0],0,b[2]],[0,0,1,b[3]],[b[1],b[2],b[3],b[4]]]);

		#initial estimate of the focal lengths
		y=np.ones(shape=[self._sequence_length,1]);
		for i in range(0,self._sequence_length):
			y[i]=np.sqrt(np.abs(P[0,:,i].dot(W).dot(np.transpose(P[0,:,i])))/(P[2,:,i].dot(W).dot(np.transpose(P[2,:,i]))))

		if (self._self_foc==0):
			#optimize for fixed focal lengths
			pp2=np.asarray([-b[1]/b[0],-b[2]/b[0],-b[3]]);  
			x=np.hstack((sum(y)/y.shape[0],pp2))
			x=fmin(self.fixed_f_error,x,args=(P,self._sequence_length));
			error=self.fixed_f_error(x,P,self._sequence_length)

			# fill out the camera instrisic parameters.
			K=np.zeros((3,3,self._sequence_length))
			for i in range(0,self._sequence_length):
				K[:,:,i]=np.eye(3);
				K[0,0,i]=x[0]*self._mm;
				K[1,1,i]=x[0]*self._mm;
				K[0,2,i]=self._width;
				K[1,2,i]=self._height;

			inf_plane=x[1:4];

			#construct the transformation matrix that will take us from the projective space to to metric
			a=inv([[x[0],0,0],[0,x[0],0],[0,0,1]])
			a=np.asarray(a)*jj;
			tmp=np.asarray([0,0,0])
			tmp=np.reshape(tmp,(3,1))
			Tm=np.append(a,tmp,1);
			Tm=np.vstack((Tm,np.append(inf_plane,1)))

		else: 
			#optimize for varying focal lenghts (to do)
			print "not yet supported"

		return Tm,K,error


	def fixed_f_error(self,x,P,n):
		'''
		Error function for the self-calibration error when the focal lengths are fixed (i.e. we 
		assume the same focal lengths accross the image sequence)
		'''
		K1=np.eye(3);
		K1[0,0]=x[0];
		K1[1,1]=x[0];
		pp=np.asarray(x[1:4]);
		pp=np.reshape(pp,(3,1));
		 
		W=np.asarray(np.append(K1.dot(np.transpose(K1)),-K1.dot(np.transpose(K1)).dot(pp),1))
		tmp=np.append((np.transpose(-pp).dot(K1).dot(np.transpose(K1))),np.transpose(pp).dot(K1).dot(np.transpose(K1)).dot(pp))
		W=np.vstack((W,tmp)) 

		a=K1.dot(np.transpose(K1));
		b=a/np.sqrt(np.trace(np.transpose(a).dot(a)));
		error=0;
		for i in range(0,n-1):
			P_tmp=P[:,:,i+1]
			c=P_tmp.dot(W).dot(np.transpose(P_tmp))
			d=c/np.sqrt(np.trace(np.transpose(c)*c))
			error=error+np.trace(np.transpose((b-d)).dot(b-d))

		return error	

	def convert_to_metric_space(self,Tm,feat3D,P,K):
		'''
		Transform the 3D points and projective matrices to the metric space
		Args: 
			Tm: transformation matrix for transformic from projective to metric space
			feat3D: 3D point cloud 
			P: projection matrices (for all views)
			K: camera intrisic parameters (for all views)
		Returns: 
			feat3D: 3D point cloud in metric space
			P: projectio matrices in metric space
		'''
		# transform the projective 3d coordinates to metric
		InvT=Tm.dot(np.eye(4));
		a=inv(InvT);
		InvT=InvT*a[3,3];

		feat3D=InvT.dot(np.transpose(feat3D))
		feat3D=np.transpose(feat3D/feat3D[3,:])	 

		#Rescale the projection matrices to width,height (no -1, 1)
		tmp=np.eye(3);
		tmp[0,0]=self._mm;
		tmp[1,1]=self._mm;
		tmp[0,2]=self._width;
		tmp[1,2]=self._height;
		for i in range(0,self._sequence_length):
		    P[:,:,i]=P[:,:,i].dot(inv(InvT))
		    P[:,:,i]=tmp.dot(P[:,:,i])
		    a=det(inv(K[:,:,i]).dot(P[0:3,0:3,i]))
		    P[:,:,i]=P[:,:,i]*np.sign(a)/np.power(abs(a),0.333)

		return feat3D,P;



def main(input_file,show):
	print "--------------------------------"
	print " Uncalibrated 3D Reconstruction "
	print ""
	print "--------------------------------"

	sequence=ImageSequence(input_file);

	start_time = time.time()
	rec_engine=UncalibratedReconstruction(sequence.length,sequence.width,sequence.height)

	#normalize coordinates
	norm_feat_2d=sequence.get_normalized_coordinates()

	print "> Estimating fundamental matrix...."
	F,epipole_1,epipole_2=rec_engine.two_view_geometry_computation(norm_feat_2d[0],norm_feat_2d[1])


	print "> Computing reference plane...." 
	# Step 2: compute the reconstruction reference plane using the epipole in the second image
	p, H = rec_engine.compute_reference_frame(epipole_2,F)


	print "> Estimating projection matrices for first two views...." 
	# Step 3: Estimate projection matrices for the first two views
	P = rec_engine.estimate_initial_projection_matrices(H,epipole_2,p)

	print "> 3D point estimate triangulation...." 
	# Step 4: triangulate points to get an initial estimate of the 3D point cloud
	feat3D=rec_engine.get_initial_structure(norm_feat_2d,P,epipole_1,epipole_2,F)

	print "> Estimating projection matrices for additional views...." 
	# Step 5: Use the 3D point estimates to estimate the projection matrices of the remaining views 
	P=rec_engine.projective_pose_estimation(norm_feat_2d,P,feat3D)


	print("> Bundle Adjustment....")
	# Step 5: Optimize 3D points and projection matrices using the reprojection error
	P,feat3D,error=rec_engine.bundle_adjustment(norm_feat_2d,P,feat3D)
	print "  - Bundle adjustment error: ",error


	print("> Self-calibration")
	# Step 6: Self-calibration
	Tm,K,error=rec_engine.self_calibration(P)
	print "  - Self-calibration error: ",error
	print "  - Tranformation Matrix (Projective -> Metric): "

	print "> Converting to metric space" 
	metric_feat3D,metric_P=rec_engine.convert_to_metric_space(Tm,feat3D,P,K)

	print "> Saving model..." 

	recModel=RecModel();
	recModel.P=P
	recModel.points3D=metric_feat3D
	recModel.Tm=Tm

	np.savetxt('rec_model_cloud.txt',metric_feat3D,delimiter=',')
	print "	 - 3D point cloud saved in rec_model_cloud.txt "

	recModel.export_stl_file('reconstructed_model.stl')
	print "  - STL model saved in reconstructed_model.stl"

	if (show==True):
		sequence.show()

	print "> 3D reconstruction completed in ",round(time.time() - start_time, 1)," sec!" 


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='3D Reconstruction from uncalibrated images')
	parser.add_argument('--input_file', metavar='path', required=True, help='Input file containing image point correspondences')
	parser.add_argument('--show', required=False, action="store_true",help="Display the image sequence with the 2D features")
	#parser.add_argument('--no-show', dest='show', action='store_false')
	args = parser.parse_args()
	main(input_file=args.input_file,show=args.show)


