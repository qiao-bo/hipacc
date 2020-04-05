# Find the FFTW includes and library
#
# Once done this will define
#  FFTW_INCLUDE_DIRS    - where to find FFTW include files
#  FFTW_LIBRARIES       - where to find FFTW libs
#  FFTW_FOUND           - True if FFTW is found

# TODO: fix paths
find_path(FFTW_INCLUDE_DIR fftw3.h PATHS ${FFTW_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../../fftw3/include)
find_library(FFTW_LIBRARY fftw3 ${CMAKE_CURRENT_SOURCE_DIR}/../../fftw3/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW DEFAULT_MSG FFTW_INCLUDE_DIR FFTW_LIBRARY)

set(FFTW_INCLUDE_DIRS ${FFTW_INCLUDE_DIR})
set(FFTW_LIBRARIES ${FFTW_LIBRARY})
if(NOT FFTW_LIBRARIES)
    set(FFTW_LIBRARIES "")
endif()

mark_as_advanced(FFTW_INCLUDE_DIR FFTW_LIBRARY)
