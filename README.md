# py3DRec
A 3D modeling from uncalibrated images algorithm implementation. The algorithm takes a series of features (i.e. 2D coordinates), tracked through a sequence of images (i.e. taken with a common handheld camera), and returns the 3D coordinates of these features in the metric space. To make this happen, the algorithm consists of the following steps: 
* Selecting two views(images), e.g. i-th and j-th view, the fundamental matrix and the epipoles are computed from the corresponding 2D features of the i-th and j-th view. For best results, the fartherst views in the sequence are selected.
* Estimating the projection matrices for the i-th and j-th view. To do so, the i-th view is assumed to be aligned with the world frame, and the projection matrix for the j-th view can be deduced using the fundamental matrix, the epipole and the reference frame of the reconstruction.
* Triangulation of the 2D features of the i-th and j-th views to get an initial estimate of the 3D structure. 
* Estimating the projection matrices for all the remaining views, using the 3D points we got from triangulation.
* Bundle adjustment, i.e. an overall optimization to refine the 3D points and the projection matrices by minimizing the reprojection error of the 3D points back to each view. 
* Self-calibration to estimate the camera intrisic parameters for each view, and transform the 3D structure from the projective to metric space. 

This code is a revised version of the core algorithm used in [3], where it was coupled with a generic face model to transform the 3D structure into the Euclidean space for face modeling purposes.


## Prerequisites 
The goal was to make the code as light as possible, so only the following packages are required
* numpy
* pandas
* scipy
* stl

## Usage

## Example


## References
[1] M. Pollefeys, R. Koch and L. Van Gool, “Self-Calibration and Metric Reconstruction in spite of Varying and Unknown Internal Camera Parameters”, Proc. International Conference on Computer Vision, Narosa Publishing House, pp.90-95, 1998
[2] A. Zisserman R. Hartley. Multiple View Geometry in Computer Vision. Cambridge University Press, 2003.
[3] N.Barbalios, N.Nikolaidis and I.Pitas, "3D Human Face Modelling >From Uncalibrated Images Using Spline Based Deformation" in 3rd International Conference on Computer Vision Theory and Applications (VISAPP 2008), Funchal Madeira, Portugal, January, 2008.
