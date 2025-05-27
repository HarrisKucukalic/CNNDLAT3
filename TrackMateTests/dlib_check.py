import dlib
print(dlib.DLIB_USE_CUDA)
dlib.DLIB_USE_CUDA = True
print(dlib.DLIB_USE_CUDA)