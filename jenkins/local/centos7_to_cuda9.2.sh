# please run with '. this-file-name.sh' or 'source this-file-name.sh'

export JAVA_HOME=/opt/conda
export NCCL_ROOT=/usr/local/nccl
rm -f /usr/local/cuda /usr/local/nccl
ln -s cuda-9.2 /usr/local/cuda
ln -s nccl_2.4.7-1+cuda9.2_x86_64 /usr/local/nccl
