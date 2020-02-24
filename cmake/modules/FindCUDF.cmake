# Tries to find cuDF headers and libraries.
#
# Usage of this module as follows:
#
#  find_package(CUDF)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  CUDF_ROOT - When set, this path is inspected instead of standard library
#              locations as the root of the CUDF installation.
#              The environment variable CUDF_ROOT overrides this variable.
#
# This module defines
#  CUDF_FOUND, whether cuDF has been found
#  CUDF_INCLUDE_DIR, directory containing header
#
# This module assumes that the user has already called find_package(CUDA)


find_path(CUDF_INCLUDE_DIR "cudf"
  HINTS "$ENV{CUDF_ROOT}/include"
        "$ENV{CONDA_PREFIX}/include/cudf"
        "$ENV{CONDA_PREFIX}/include")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDF_INCLUDE DEFAULT_MSG
                                  CUDF_INCLUDE_DIR)

find_library(CUDF_LIBRARY
  NAMES cudf
  HINTS $ENV{CUDF_ROOT}/lib/
        ${CUDF_ROOT}/lib
        $ENV{CONDA_PREFIX}/lib)


find_package_handle_standard_args(CUDF_LIBRARY DEFAULT_MSG
                                  CUDF_LIBRARY)

mark_as_advanced(
  CUDF_INCLUDE_DIR
  CUDF_LIBRARY
)
