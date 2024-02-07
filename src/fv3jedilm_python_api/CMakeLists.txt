
find_package( fv3jedilm 1.0.0 REQUIRED )

# Grab Python, 3.10 or newer
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
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


#### Arguments for generating wrappers and python extension module

set(package_name "fv3jedilm_python_api")
set(generator_script "${CMAKE_CURRENT_SOURCE_DIR}/generate_wrappers.py")
set(src_files 
    "${CMAKE_CURRENT_SOURCE_DIR}/fv3jedilm_copy/fv_arrays_nlm.F90"
    "${CMAKE_CURRENT_SOURCE_DIR}/fv3jedilm_copy/fv_grid_utils_nlm.F90"
    "${CMAKE_CURRENT_SOURCE_DIR}/fv3jedilm_copy/mpp_domains.F90"
)
set(f2cmap_option "-k" "${CMAKE_CURRENT_SOURCE_DIR}/.f2py_f2cmap_default_real_is_double")
set(package_name_option "-p${package_name}")
set(include_options "-I${CMAKE_CURRENT_SOURCE_DIR}/fv3jedilm_copy/include")
set(definition_options
"-Duse_libMPI"
"-Duse_netCDF"
"-DSPMD"
"-DUSE_LOG_DIAG_FIELD_INFO"
"-Duse_LARGEFILE"
"-DOLDMPP"
"-DGFS_PHYS"
"-DINTERNAL_FILE_NML")
set(only_option 
"--only"
"allocate_fv_atmos_type"
"deallocate_fv_atmos_type"
"cubed_to_latlon"
"mpp_domains_init"
"mpp_domains_exit"
)

##### Generation target and associated custom command
set(f90wrap_module_py "${package_name}.py")
set(f2py_module_c "_${package_name}module.c")

set(generated_python_files
    "${CMAKE_CURRENT_BINARY_DIR}/${f90wrap_module_py}"
)
set(generated_src_files 
    "${CMAKE_CURRENT_BINARY_DIR}/f90wrap_preprocessed_fv_arrays_nlm.f90"
    "${CMAKE_CURRENT_BINARY_DIR}/f90wrap_preprocessed_fv_grid_utils_nlm.f90"
    "${CMAKE_CURRENT_BINARY_DIR}/f90wrap_preprocessed_mpp_domains.f90"
    "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_c}"
)

add_custom_target(
  "_${package_name}_generator"
  DEPENDS ${generated_src_files} ${generated_python_files}
)
add_custom_command(
  OUTPUT ${generated_src_files} ${generated_python_files}
  COMMAND ${Python_EXECUTABLE} ${generator_script} ${src_files} ${f2cmap_option} ${package_name_option} ${definition_options} ${include_options} ${only_option}
  DEPENDS ${src_files}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Define the extension module
Python_add_library("_${package_name}" MODULE WITH_SOABI
  ${generated_src_files}                  # <-- Generated source
  "${F2PY_INCLUDE_DIR}/fortranobject.c"   # <-- From NumPy
)
target_link_libraries("_${package_name}" PRIVATE fv3jedilm)
target_link_libraries("_${package_name}" PRIVATE fms)
target_link_libraries("_${package_name}" PRIVATE Python::NumPy)
target_include_directories("_${package_name}" PRIVATE "${F2PY_INCLUDE_DIR}")
target_include_directories("_${package_name}" PRIVATE ${CMAKE_Fortran_MODULE_DIRECTORY})
set_target_properties("_${package_name}" PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/${pyver}/${package_name}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/${pyver}/${package_name}"
)

# Add dependency from extension module to generation target
add_dependencies("_${package_name}" "_${package_name}_generator")

# Copy Python wrapper interface to library directory once the extension module has been built
add_custom_command(TARGET "_${package_name}" POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy ${generated_python_files} "${CMAKE_BINARY_DIR}/lib/${pyver}/${package_name}/")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${package_name}.py
	      DESTINATION ${CMAKE_INSTALL_LIBDIR}/${pyver}/${package_name}/
        COMPONENT Python)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/"_${package_name}".py
	      DESTINATION ${CMAKE_INSTALL_LIBDIR}/${pyver}/${package_name}/
        COMPONENT Python
        FILES_MATCHING PATTERN "_${package_name}.cpython*.so")