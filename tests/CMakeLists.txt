find_package(rostest REQUIRED)

set(DDMPC_gtest_list
  TestMathUtils
  TestTorchUtils
  TestStateEq
  TestDataset
  TestTraining
  TestMpcOscillator
  TestMpcPushWalk
  )

set(DDMPC_rostest_list
  TestMpcCart
  TestMpcCartWalk
  )

foreach(NAME IN LISTS DDMPC_gtest_list)
  catkin_add_gtest(${NAME} src/${NAME}.cpp)
  target_link_libraries(${NAME} DDMPC)
endforeach()

foreach(NAME IN LISTS DDMPC_rostest_list)
  add_rostest_gtest(${NAME} test/${NAME}.test src/${NAME}.cpp)
  target_link_libraries(${NAME} DDMPC)
endforeach()
