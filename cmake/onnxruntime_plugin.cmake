onnxruntime_add_shared_library(onnxruntime_plugin ${ONNXRUNTIME_ROOT}/remote/onnx_messages.cc ${ONNXRUNTIME_ROOT}/remote/onnx_registry.cc)

add_dependencies(onnxruntime_plugin ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnxruntime_plugin PRIVATE ${ONNXRUNTIME_ROOT})
onnxruntime_add_include_to_target(onnxruntime_plugin)

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

target_link_libraries(onnxruntime_plugin PRIVATE
    onnxruntime_session
    ${onnxruntime_libs}
    ${PROVIDERS_CUDA}
    ${PROVIDERS_NNAPI}
    ${PROVIDERS_RKNPU}
    ${PROVIDERS_MIGRAPHX}
    ${PROVIDERS_NUPHAR}
    ${PROVIDERS_VITISAI}
    ${PROVIDERS_DML}
    ${PROVIDERS_TIDL}
    ${PROVIDERS_ACL}
    ${PROVIDERS_ARMNN}
    ${PROVIDERS_INTERNAL_TESTING}
    ${onnxruntime_winml}
    ${PROVIDERS_ROCM}
    ${PROVIDERS_COREML}
    onnxruntime_optimizer
    onnxruntime_providers
    onnxruntime_util
    ${onnxruntime_tvm_libs}
    onnxruntime_framework
    onnxruntime_graph
    onnxruntime_common
    onnxruntime_mlas
    onnxruntime_flatbuffers
    ${onnxruntime_EXTERNAL_LIBRARIES})

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  target_link_libraries(onnxruntime_plugin PRIVATE onnxruntime_language_interop onnxruntime_pyop)
endif()

install(TARGETS onnxruntime_plugin
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
