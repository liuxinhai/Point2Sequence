TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

#/bin/bash
/usr/local/cuda-8.0/bin/nvcc -std=c++11  tf_sampling_g.cu -c -o tf_sampling_g.cu.o -I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CXXFLAGS

# TF1.2
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I $TF_INC -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I $TF_INC -I $IF_INC/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L $TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
#g++ -std=c++11 -shared -o tf_sampling_so.so tf_sampling.cpp tf_sampling_g.cu.o -I $TF_INC -I $IF_INC/external/nsync/public -D GOOGLE_CUDA=1 -fPIC $CXXFLAGS -D_GLIBCXX_USE_CXX11_ABI=0 -lcudart -L /usr/local/cuda/lib64 -L $TF_LIB -ltensorflow_framework
