# (C) Copyright 2017-2020 UCAR.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

################################################################################
# PYFV3LM
################################################################################

cmake_minimum_required( VERSION 3.3.2 FATAL_ERROR )

project( pyfv3lm VERSION 1.4.0 LANGUAGES Fortran C )

set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH} )

set( CMAKE_DIRECTORY_LABELS "pyfv3lm" )

set( ECBUILD_DEFAULT_BUILD_TYPE Release )
set( ENABLE_OS_TESTS           OFF CACHE BOOL "Disable OS tests" FORCE )
set( ENABLE_LARGE_FILE_SUPPORT OFF CACHE BOOL "Disable testing of large file support" FORCE )
set( ENABLE_MPI ON CACHE BOOL "Compile with MPI" )

include( ecbuild_system NO_POLICY_SCOPE )

ecbuild_requires_macro_version( 2.7 )

################################################################################
# Project
################################################################################

ecbuild_declare_project()

ecbuild_enable_fortran( REQUIRED )

################################################################################
# Dependencies
################################################################################

# MKL / LAPACK
# Default: use MKL as LAPACK implementation.
# Set -DENABLE_MKL=Off to disable use of MKL and prefer LAPACK
option( ENABLE_MKL "Use MKL for LAPACK implementation (if available)" ON )
if( ENABLE_MKL )
    find_package( MKL )
endif()
if( MKL_FOUND )
    set( LAPACK_LIBRARIES ${MKL_LIBRARIES} )
else()
    find_package( LAPACK REQUIRED )
endif()

#openMP
ecbuild_enable_omp()

# MPI
ecbuild_add_option( FEATURE MPI DEFAULT ON
                    DESCRIPTION "Support for MPI distributed parallelism"
                    REQUIRED_PACKAGES "MPI COMPONENTS Fortran" )
ecbuild_find_mpi( COMPONENTS Fortran REQUIRED )

# NetCDF
find_package( jedicmake QUIET )
find_package( NetCDF REQUIRED COMPONENTS Fortran )

# Set default model
if (NOT FV3_FORECAST_MODEL MATCHES "GEOS" AND NOT FV3_FORECAST_MODEL MATCHES "UFS")
  set(FV3_FORECAST_MODEL "FV3CORE")
endif()

# fms
if (FV3_FORECAST_MODEL MATCHES "FV3CORE")
  find_package( fms REQUIRED )
  include_directories( ${FMS_INCLUDE_DIRS} )
  list( APPEND PYFV3LM_TPLS fms )
endif()

# geos
if(FV3_FORECAST_MODEL MATCHES "GEOS")
  find_package(MAPL REQUIRED PATHS ${FV3_FORECAST_MODEL_ROOT}/lib/cmake/MAPL )
  find_package(GEOSgcm REQUIRED PATHS ${FV3_FORECAST_MODEL_ROOT}/lib/cmake/GEOSgcm )
  if (FV3_PRECISION MATCHES DOUBLE OR NOT FV3_PRECISION)
    link_libraries(fms_r8)
  else()
    link_libraries(fms_r4)
  endif()
endif()

# ufs
if(FV3_FORECAST_MODEL MATCHES "UFS")
  # Set CMAKE_PREFIX_PATH to include the install location of ufs
  find_package(fms REQUIRED )
  find_package(stochastic_physics REQUIRED )
  find_package(ccpp REQUIRED )
  find_package(ccppphys REQUIRED )
  find_package(fv3atm REQUIRED )
  find_package(ufs REQUIRED )
  link_libraries(FMS::fms_r8)
  link_libraries(NetCDF::NetCDF_Fortran)
  # Install includes
  get_target_property(fmsinc FMS::fms_r8 INTERFACE_INCLUDE_DIRECTORIES)
  include_directories( ${fmsinc} )
endif()

################################################################################
# Export package info
################################################################################

list( APPEND PYFV3LM_TPLS MPI )

set( PYFV3LM_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
                            ${CMAKE_CURRENT_BINARY_DIR}/src
                            ${CMAKE_Fortran_MODULE_DIRECTORY} )
set( PYFV3LM_LIBRARIES pyfv3lm )

get_directory_property( PYFV3LM_DEFINITIONS COMPILE_DEFINITIONS )

foreach( _tpl ${PYFV3LM_TPLS} )
  string( TOUPPER ${_tpl} TPL )
  list( APPEND PYFV3LM_EXTRA_DEFINITIONS   ${${TPL}_DEFINITIONS}  ${${TPL}_TPL_DEFINITIONS}  )
  list( APPEND PYFV3LM_EXTRA_INCLUDE_DIRS  ${${TPL}_INCLUDE_DIRS} ${${TPL}_TPL_INCLUDE_DIRS} )
  list( APPEND PYFV3LM_EXTRA_LIBRARIES     ${${TPL}_LIBRARIES}    ${${TPL}_TPL_LIBRARIES}    )
endforeach()

################################################################################
# Sources
################################################################################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_FORTRAN_STANDARD 08)
set(CMAKE_FORTRAN_STANDARD_REQUIRED ON)
set(CMAKE_FORTRAN_EXTENSIONS OFF)

include( pyfv3lm_compiler_flags )
include_directories( ${PYFV3LM_INCLUDE_DIRS} ${PYFV3LM_EXTRA_INCLUDE_DIRS} )

add_subdirectory( src )

################################################################################
# Finalise configuration
################################################################################

# prepares a tar.gz of the sources and/or binaries
ecbuild_install_project( NAME pyfv3lm )

# print the summary of the configuration
ecbuild_print_summary()
