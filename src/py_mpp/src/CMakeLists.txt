

###################################################################################

find_package( fms REQUIRED )
add_library(mpp_ops "mpp_ops/mpp_ops.f90" "mpp_ops/interface_utils.f90")
target_link_libraries(mpp_ops PRIVATE fms)
set_target_properties(mpp_ops PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/"
)

###################################################################################

# Grab Python, 3.10 or newer
find_package( Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
set(pyver python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR})

# Grab the variables from a local Python installation F2PY headers
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Print out the discovered paths
include(CMakePrintHelpers)
cmake_print_variables(Python_INCLUDE_DIRS)
cmake_print_variables(F2PY_INCLUDE_DIR)
cmake_print_variables(Python_NumPy_INCLUDE_DIRS)

set(package_name "numpy_mpp_ops")
set(f2py_input_files "${CMAKE_CURRENT_SOURCE_DIR}/mpp_ops/mpp_ops.f90")
set(f2py_output_files "${package_name}module.c" "${package_name}-f2pywrappers2.f90")
add_custom_command(
  OUTPUT ${f2py_output_files}
  COMMAND ${Python_EXECUTABLE} -m "numpy.f2py" ${f2py_input_files} -m "${package_name}" --lower  --f2cmap "${CMAKE_CURRENT_SOURCE_DIR}/py_mpp_numpy/.f2py_f2cmap_default_real_is_double"
  DEPENDS ${f2py_input_files}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_target(
  "${package_name}_generator"
  DEPENDS ${f2py_output_files}
)

# Define the extension module
python_add_library("${package_name}" MODULE WITH_SOABI
  ${f2py_output_files}                    # <-- Generated source
  "${F2PY_INCLUDE_DIR}/fortranobject.c"   # <-- From NumPy
)
add_dependencies("${package_name}" "${package_name}_generator")
target_link_libraries("${package_name}" PUBLIC mpp_ops)
target_link_libraries("${package_name}" PRIVATE Python::NumPy)
target_include_directories("${package_name}" PRIVATE "${F2PY_INCLUDE_DIR}")
target_include_directories("${package_name}" PRIVATE ${CMAKE_Fortran_MODULE_DIRECTORY})
set_target_properties("${package_name}" PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/${pyver}/py_mpp_numpy"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/${pyver}/py_mpp_numpy"
)
add_custom_command(
  TARGET "${package_name}" POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/py_mpp_numpy/*.py" "${CMAKE_BINARY_DIR}/lib/${pyver}/py_mpp_numpy"
)
add_custom_command(
  TARGET "${package_name}" POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/py_mpp_common/*.py" "${CMAKE_BINARY_DIR}/lib/${pyver}/py_mpp_numpy"
)

# install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${package_name}.py
# 	      DESTINATION ${CMAKE_INSTALL_LIBDIR}/${pyver}/${package_name}/
#         COMPONENT Python)

# install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/"_${package_name}".py
# 	      DESTINATION ${CMAKE_INSTALL_LIBDIR}/${pyver}/${package_name}/
#         COMPONENT Python
#         FILES_MATCHING PATTERN "_${package_name}.cpython*.so")


# #################################################################################

# set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
python_add_library(jax_mpp_ops MODULE WITH_SOABI
  "${CMAKE_CURRENT_SOURCE_DIR}/py_mpp_jax/jax_mpp_ops.cpp"
)
target_link_libraries(jax_mpp_ops PRIVATE pybind11::headers)
target_link_libraries(jax_mpp_ops PRIVATE mpp_ops)
# set_target_properties(jax_mpp_ops PROPERTIES
#                       INTERPROCEDURAL_OPTIMIZATION ON
#                       CXX_VISIBILITY_PRESET ON
#                       VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(jax_mpp_ops PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/${pyver}/py_mpp_jax"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/${pyver}/py_mpp_jax"
)
add_custom_command(
  TARGET jax_mpp_ops POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/py_mpp_jax/*.py" "${CMAKE_BINARY_DIR}/lib/${pyver}/py_mpp_jax"
)
add_custom_command(
  TARGET jax_mpp_ops POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/py_mpp_common/*.py" "${CMAKE_BINARY_DIR}/lib/${pyver}/py_mpp_jax"
)