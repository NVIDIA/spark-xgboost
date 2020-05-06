#!/usr/bin/env python
import errno
import glob
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager


# Monkey-patch the API inconsistency between Python2.X and 3.X.
if sys.platform.startswith("linux"):
    sys.platform = "linux"


CONFIG = {
    "USE_OPENMP": "ON",
    "USE_HDFS": "OFF",
    "USE_AZURE": "OFF",
    "USE_S3": "OFF",

    "USE_CUDA": "ON",
    "USE_NCCL": "ON",
    "USE_CUDF": "ON",
    "JVM_BINDINGS": "ON"
}


@contextmanager
def cd(path):
    path = normpath(path)
    cwd = os.getcwd()
    os.chdir(path)
    print("cd " + path)
    try:
        yield path
    finally:
        os.chdir(cwd)


def maybe_makedirs(path):
    path = normpath(path)
    print("mkdir -p " + path)
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def run(command, **kwargs):
    print(command)
    subprocess.check_call(command, shell=True, **kwargs)


def cp(source, target):
    source = normpath(source)
    target = normpath(target)
    print("cp {0} {1}".format(source, target))
    shutil.copy(source, target)


def normpath(path):
    """Normalize UNIX path to a native path."""
    normalized = os.path.join(*path.split("/"))
    if os.path.isabs(path):
        return os.path.abspath("/") + normalized
    else:
        return normalized


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Usage: create_jni.py <cuda version>")

    cuda_ver = sys.argv[1].lower()
    if cuda_ver == "cuda10.1":
        cuda = "cuda10.1"
    elif cuda_ver == "cuda10.2":
        cuda = "cuda10.2"
    else:
        raise Exception("Unsupported cuda version: " + cuda_ver)

    if sys.platform == "darwin":
        # Enable of your compiler supports OpenMP.
        CONFIG["USE_OPENMP"] = "OFF"
        os.environ["JAVA_HOME"] = subprocess.check_output(
            "/usr/libexec/java_home").strip().decode()

    print("building Java wrapper on " + cuda)
    with cd(".."):
        maybe_makedirs("build")
        with cd("build"):
            if sys.platform == "win32":
                # Force x64 build on Windows.
                maybe_generator = ' -G"Visual Studio 14 Win64"'
            else:
                maybe_generator = ""
            if sys.platform == "linux":
                maybe_parallel_build = " -- -j $(nproc)"
            else:
                maybe_parallel_build = ""

            args = ["-D{0}:BOOL={1}".format(k, v) for k, v in CONFIG.items()]
            run("cmake .. " + " ".join(args) + maybe_generator)
            run("cmake --build . --config Release" + maybe_parallel_build)

        with cd("demo/regression"):
            run(sys.executable + " mapfeat.py")
            run(sys.executable + " mknfold.py machine.txt 1")

    print("copying native library")
    library_name = {
        "win32": "xgboost4j.dll",
        "darwin": "libxgboost4j.dylib",
        "linux": "libxgboost4j.so"
    }[sys.platform]
    lib_path = "xgboost4j/src/main/resources/lib/" + cuda
    maybe_makedirs(lib_path)
    cp("../lib/" + library_name, lib_path)

    print("copying pure-Python tracker")
    cp("../dmlc-core/tracker/dmlc_tracker/tracker.py",
       "xgboost4j/src/main/resources")

    print("copying train/test files")
    maybe_makedirs("xgboost4j-spark/src/test/resources")
    with cd("../demo/regression"):
        run("{} mapfeat.py".format(sys.executable))
        run("{} mknfold.py machine.txt 1".format(sys.executable))

    for file in glob.glob("../demo/regression/machine.txt.t*"):
        cp(file, "xgboost4j-spark/src/test/resources")
    for file in glob.glob("../demo/data/agaricus.*"):
        cp(file, "xgboost4j-spark/src/test/resources")
