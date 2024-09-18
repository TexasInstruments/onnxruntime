onnxruntime_add_shared_library(onnxruntime_plugin ${ONNXRUNTIME_ROOT}/remote/onnx_messages.cc ${ONNXRUNTIME_ROOT}/remote/onnx_registry.cc)

onnxruntime_add_include_to_target(onnxruntime_plugin cpuinfo)
onnxruntime_add_include_to_target(onnxruntime_plugin clog)

list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES cpuinfo clog)
list(APPEND onnxruntime_EXTERNAL_LIBRARIES cpuinfo clog)

message("onnxruntime_plugin::onnxruntime_EXTERNAL_DEPENDENCIES -1 : ${onnxruntime_EXTERNAL_DEPENDENCIES}")
add_dependencies(onnxruntime_plugin ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnxruntime_plugin PRIVATE ${ONNXRUNTIME_ROOT})
onnxruntime_add_include_to_target(onnxruntime_plugin)

message("onnxruntime_plugin::onnxruntime_EXTERNAL_DEPENDENCIES - 2: ${onnxruntime_EXTERNAL_DEPENDENCIES}")

# strip binary on Android, or for a minimal build on Unix
if(CMAKE_SYSTEM_NAME STREQUAL "Android" OR (onnxruntime_MINIMAL_BUILD AND UNIX))
  if (onnxruntime_MINIMAL_BUILD AND ADD_DEBUG_INFO_TO_MINIMAL_BUILD)
    # don't strip
  else()
    set_target_properties(onnxruntime_plugin PROPERTIES LINK_FLAGS_RELEASE -s)
    set_target_properties(onnxruntime_plugin PROPERTIES LINK_FLAGS_MINSIZEREL -s)
  endif()
endif()

target_include_directories(onnxruntime_plugin PRIVATE ${onnxruntime_TIIE_HOME})

if(onnxruntime_USE_TIIE)
    add_definitions(-DUSE_TIIE=1)
    add_custom_target(remote_header
        COMMAND ${onnxruntime_TIIE_HOME}/parse_messages.py ${ONNXRUNTIME_ROOT}/remote/onnx_messages.yaml > ${ONNXRUNTIME_ROOT}/remote/onnx_messages.h
        VERBATIM
    )
    
    add_dependencies(onnxruntime_plugin remote_header)
endif(onnxruntime_USE_TIIE)

message("onnxruntime_plugin::onnxruntime_EXTERNAL_DEPENDENCIES - 3: ${onnxruntime_EXTERNAL_DEPENDENCIES}")


set(onnxruntime_plugin_INTERNAL_LIBRARIES
    onnxruntime_session
    ${onnxruntime_libs}
    ${PROVIDERS_ACL}
    ${PROVIDERS_ARMNN}
    ${PROVIDERS_COREML}
    ${PROVIDERS_DML}
    ${PROVIDERS_TIDL}
    ${PROVIDERS_NNAPI}
    ${PROVIDERS_SNPE}
    ${PROVIDERS_TVM}
    ${PROVIDERS_RKNPU}
    ${PROVIDERS_ROCM}
    ${PROVIDERS_VITISAI}
    ${PROVIDERS_XNNPACK}
    ${PROVIDERS_AZURE}
    ${PROVIDERS_INTERNAL_TESTING}
    ${onnxruntime_winml}
    onnxruntime_optimizer
    onnxruntime_providers
    ${onnxruntime_tvm_libs}
    onnxruntime_framework
    onnxruntime_graph
    onnxruntime_util
    onnxruntime_mlas	# probably getting cpuinfo dependency from here -> check mlasi.h 
    ${ONNXRUNTIME_MLAS_LIBS}
    onnxruntime_common
    onnxruntime_flatbuffers
)

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  list(APPEND onnxruntime_plugin_INTERNAL_LIBRARIES 
  onnxruntime_language_interop 
  onnxruntime_pyop
  )
endif()

message("onnxruntime_plugin_INTERNAL_LIBRARIES: ${onnxruntime_plugin_INTERNAL_LIBRARIES}")
message("onnxruntime_EXTERNAL_LIBRARIES: ${onnxruntime_EXTERNAL_LIBRARIES}")

target_link_libraries(onnxruntime_plugin PRIVATE
    ${onnxruntime_plugin_INTERNAL_LIBRARIES}
    ${onnxruntime_EXTERNAL_LIBRARIES}
)

install(TARGETS onnxruntime_plugin
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
