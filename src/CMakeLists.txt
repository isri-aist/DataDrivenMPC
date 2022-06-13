add_library(DDMPC
  Dataset.cpp
  )
target_include_directories(DDMPC PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${catkin_INCLUDE_DIRS}
  )
target_link_libraries(DDMPC PUBLIC
  ${TORCH_LIBRARIES} ${catkin_LIBRARIES}
  )
