#!/usr/bin/env python
import os
import re
import subprocess
import sys

# version -> classifier
# '' means default classifier
cuda_vers = {
  '10.1': ['cuda10-1', ''],
  '10.2': ['cuda10-2']
}

def check_classifier(classifier):
    '''
    Check the mapping from cuda version to jar classifier.
    Used by maven build.
    '''
    cu_ver = detect_cuda_ver()
    classifier_list = cuda_vers[cu_ver]
    if classifier not in classifier_list:
        raise Exception("Jar classifier '{}' mismatches the 'nvcc' version {} !".format(classifier, cu_ver))


def get_classifier():
    cu_ver = detect_cuda_ver()
    classifier_list = cuda_vers[cu_ver]
    return classifier_list[0]


def get_supported_vers():
    '''
    Get the supported cuda versions.
    '''
    return cuda_vers.keys()


def get_supported_vers_str():
    '''
    Get the supported cuda versions and join them as a string.
    Used by shell script.
    '''
    return ' '.join(cuda_vers.keys())


def detect_cuda_ver():
    '''
    Detect the cuda version from current nvcc tool.
    '''
    nvcc_ver_bin = subprocess.check_output('nvcc --version', shell=True)
    nvcc_ver = re.search('release ([.0-9]+), V([.0-9]+)', str(nvcc_ver_bin)).group(1)
    if nvcc_ver in get_supported_vers():
        return nvcc_ver
    else:
        raise Exception("Unsupported cuda version: {}, Please check your 'nvcc' version.".format(nvcc_ver))


def cudaver():
    return 'cuda{}'.format(detect_cuda_ver())


if __name__ == "__main__":
    num_args = len(sys.argv)
    action = sys.argv[1].lower() if num_args > 1 else 'l'
    if action =='c':
        classifier = sys.argv[2].lower() if num_args > 2 else ''
        check_classifier(classifier)
    elif action == 'd':
        print(detect_cuda_ver())
    elif action == 'g':
        print(get_classifier())
    elif action == 'l':
        print(get_supported_vers_str())
    else:
        print("Unsupported action: " + action)
