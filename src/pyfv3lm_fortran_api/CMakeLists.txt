set(SRC_FILES
    pyfv3lm_mod.F90
    pyfv3lm_fv3jedilm_mock_mod.F90
)

# if( NOT FV3_FORECAST_MODEL MATCHES "GEOS" AND NOT FV3_FORECAST_MODEL MATCHES "UFS" )
#   list( APPEND FV3JEDILM_LIB_DEP fms )
# endif()

ecbuild_add_library( TARGET pyfv3lm_fortran_api
                     SOURCES  ${SRC_FILES}
                     PRIVATE_LIBS fms fv3jedilm pyfv3lm_c_api
                     INSTALL_HEADERS LISTED )

add_dependencies(pyfv3lm_fortran_api pyfv3lm_c_api_generator)

target_include_directories( pyfv3lm_fortran_api INTERFACE $<BUILD_INTERFACE:${CMAKE_Fortran_MODULE_DIRECTORY}>
                                                $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}>)
# target_link_libraries(pyfv3lm_fortran_api PRIVATE pyfv3lm_c_api)
target_link_directories(pyfv3lm_fortran_api PRIVATE "${CMAKE_BINARY_DIR}/lib")