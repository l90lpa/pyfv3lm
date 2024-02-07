set(generator_script "${CMAKE_CURRENT_SOURCE_DIR}/build_pyfv3lm_c_api.py")
set(plugin_name "pyfv3lm_c_api")
set(plugin_build_dir ${CMAKE_CURRENT_BINARY_DIR})
set(plugin_header_file "${CMAKE_CURRENT_SOURCE_DIR}/c_api.h")
set(plugin_module_file "${CMAKE_CURRENT_SOURCE_DIR}/module.py")

add_custom_command(
    OUTPUT "lib${plugin_name}.so" "${plugin_name}.c"
    COMMAND python ${generator_script} ${plugin_name} ${plugin_build_dir} ${plugin_header_file} ${plugin_module_file}
    COMMAND ${CMAKE_COMMAND} -E copy "lib${plugin_name}.so" "${CMAKE_BINARY_DIR}/lib"
    COMMENT "Generating python plugin"
    DEPENDS ${plugin_header_file} ${plugin_module_file} ${generator_script}
)

# Create a custom target that depends on the custom command
add_custom_target("${plugin_name}_generator" DEPENDS "lib${plugin_name}.so" "${plugin_name}.c")
add_dependencies("${plugin_name}_generator" _fv3jedilm_python_api)
add_dependencies("${plugin_name}_generator" numpy_mpp_ops)
add_dependencies("${plugin_name}_generator" jax_mpp_ops)