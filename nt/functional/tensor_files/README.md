These are header files and build files that are made that rely on the Tensor class
This seperation has been done in order to speed up compilation/test/bug fix times, and make it easier for different device functions


Tensor unfold; -> colim_transform.h
Tensor& unfold_backward; -> colim_transform.h 
Tensor fold; -> colim_transform.h 
Tensor fold_backward; -> colim_transform.h 
Tensor& fold_backward; -> colim_transform.h 


//3d transforms:
Tensor unfold3d; -> colim_transform.h 
Tensor unfold3d_backward; -> colim_transform.h 
Tensor& unfold3d_backward; -> colim_transform.h 


//1d transforms:
Tensor unfold1d; -> colim_transform.h 
Tensor unfold1d_backward; -> colim_transform.h 
Tensor& unfold1d_backward; -> colim_transform.h

Tensor one_hot; -> mesh.h

