#/bin/bash
/usr/local/cuda-8.0/bin/nvcc tf_grouping_g.cu \
-o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.4 Python 3.5 CUDA 8.0
g++ -std=c++11 tf_grouping.cpp \
tf_grouping_g.cu.o \
-o tf_grouping_so.so \
-shared -fPIC \
-I$HOME/anaconda3/envs/tensorflow_1.4/lib/python3.5/site-packages/tensorflow/include \
-I/usr/local/cuda-8.0/include \
-I$HOME/anaconda3/envs/tensorflow_1.4/lib/python3.5/site-packages/tensorflow/include/external/nsync/public \
-lcudart -L/usr/local/cuda-8.0/lib64/ \
-L$HOME/anaconda3/envs/tensorflow_1.4/lib/python3.5/site-packages/tensorflow \
-ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
