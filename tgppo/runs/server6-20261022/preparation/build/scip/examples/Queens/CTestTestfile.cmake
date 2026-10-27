# CMake generated Testfile for 
# Source directory: /home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/Queens
# Build directory: /home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip/examples/Queens
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(examples-queens-build "/usr/bin/cmake" "--build" "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip" "--target" "queens")
set_tests_properties(examples-queens-build PROPERTIES  RESOURCE_LOCK "libscip" _BACKTRACE_TRIPLES "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/Queens/CMakeLists.txt;22;add_test;/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/Queens/CMakeLists.txt;0;")
add_test(examples-queens-1 "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip/bin/examples/queens" "1")
set_tests_properties(examples-queens-1 PROPERTIES  DEPENDS "examples-queens-build" _BACKTRACE_TRIPLES "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/Queens/CMakeLists.txt;46;add_test;/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/Queens/CMakeLists.txt;0;")
add_test(examples-queens-2 "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip/bin/examples/queens" "2")
set_tests_properties(examples-queens-2 PROPERTIES  DEPENDS "examples-queens-build" _BACKTRACE_TRIPLES "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/Queens/CMakeLists.txt;46;add_test;/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/Queens/CMakeLists.txt;0;")
add_test(examples-queens-4 "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip/bin/examples/queens" "4")
set_tests_properties(examples-queens-4 PROPERTIES  DEPENDS "examples-queens-build" _BACKTRACE_TRIPLES "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/Queens/CMakeLists.txt;46;add_test;/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/Queens/CMakeLists.txt;0;")
add_test(examples-queens-8 "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip/bin/examples/queens" "8")
set_tests_properties(examples-queens-8 PROPERTIES  DEPENDS "examples-queens-build" _BACKTRACE_TRIPLES "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/Queens/CMakeLists.txt;46;add_test;/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/Queens/CMakeLists.txt;0;")
add_test(examples-queens-16 "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip/bin/examples/queens" "16")
set_tests_properties(examples-queens-16 PROPERTIES  DEPENDS "examples-queens-build" _BACKTRACE_TRIPLES "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/Queens/CMakeLists.txt;46;add_test;/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/Queens/CMakeLists.txt;0;")
