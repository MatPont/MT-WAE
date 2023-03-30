ttk_add_base_library(mergeTreeAutoencoder
  SOURCES
    MergeTreeAutoencoder.cpp
    MergeTreeAutoencoderUtils.cpp
    TorchUtils.cpp
  HEADERS
    MergeTreeAutoencoder.h
    MergeTreeAutoencoderUtils.h
    TorchUtils.h
  DEPENDS
    triangulation
    mergeTreePrincipalGeodesics
    geometry
)

if(TTK_ENABLE_TORCH)
  target_include_directories(mergeTreeAutoencoder PUBLIC ${TORCH_INCLUDE_DIRS})
  target_compile_options(mergeTreeAutoencoder PUBLIC "${TORCH_CXX_FLAGS}")
  target_link_libraries(mergeTreeAutoencoder PUBLIC "${TORCH_LIBRARIES}")
  target_compile_definitions(mergeTreeAutoencoder PUBLIC TTK_ENABLE_TORCH)
  # target_include_directories(mergeTreeAutoencoder PRIVATE ${TORCH_INCLUDE_DIRS})
  # set_property(TARGET mergeTreeAutoencoder  PROPERTY CXX_STANDARD 14)
endif()

#target_compile_options(mergeTreeAutoencoder PRIVATE -ggdb -O0 -fsanitize=address)
#target_link_options(mergeTreeAutoencoder PRIVATE -fsanitize=address)
