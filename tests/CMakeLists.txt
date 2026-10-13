# SPDX-License-Identifier: MIT
add_executable(microtensor_tests microtensor_test.cpp)
target_link_libraries(microtensor_tests PRIVATE mt::microtensor)
mt_set_warnings(microtensor_tests)

add_test(NAME microtensor_tests COMMAND microtensor_tests)
