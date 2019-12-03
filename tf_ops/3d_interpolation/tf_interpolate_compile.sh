# TF1.4 Python 3.5 CUDA 8.0

#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 tf_interpolate.cpp \
-o tf_interpolate_so.so \
-shared \
-fPIC \
-I$HOME/anaconda3/envs/tensorflow_1.4/lib/python3.5/site-packages/tensorflow/include \
-I/usr/local/cuda-8.0/include \
-I$HOME/anaconda3/envs/tensorflow_1.4/lib/python3.5/site-packages/tensorflow/include/external/nsync/public \
-lcudart \
-L/usr/local/cuda-8.0/lib64/ \
-L$HOME/anaconda3/envs/tensorflow_1.4/lib/python3.5/site-packages/tensorflow \
-ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0