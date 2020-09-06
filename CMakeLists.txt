cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ts_ssd)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

aux_source_directory(. DIR_SRCS)

add_executable(ts_ssd ${DIR_SRCS})
target_link_libraries(ts_ssd "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET ts_ssd PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ts_ssd
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:ts_ssd>)
endif (MSVC)