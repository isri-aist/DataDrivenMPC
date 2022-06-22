set(DDMPC_gtest_list
  TestMathUtils
  TestTorchUtils
  TestStateEq
  TestDataset
  TestTraining
  TestMpcOscillator
  TestMpcPushWalk
  TestMpcCart
  )

foreach(NAME IN LISTS DDMPC_gtest_list)
  catkin_add_gtest(${NAME} src/${NAME}.cpp)
  target_link_libraries(${NAME} DDMPC)
endforeach()
