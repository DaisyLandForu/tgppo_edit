# CMake generated Testfile for 
# Source directory: /home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/VRP/check
# Build directory: /home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip/examples/VRP/check
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(examples-vrp-build "/usr/bin/cmake" "--build" "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip" "--target" "vrp")
set_tests_properties(examples-vrp-build PROPERTIES  RESOURCE_LOCK "libscip" _BACKTRACE_TRIPLES "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/VRP/check/CMakeLists.txt;18;add_test;/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/VRP/check/CMakeLists.txt;0;")
add_test(examples-vrp-eil13 "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip/bin/examples/vrp" "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/VRP/check/../data/eil13.vrp")
set_tests_properties(examples-vrp-eil13 PROPERTIES  DEPENDS "examples-vrp-build" _BACKTRACE_TRIPLES "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/VRP/check/CMakeLists.txt;38;add_test;/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/VRP/check/CMakeLists.txt;0;")
add_test(examples-vrp-eil7 "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip/bin/examples/vrp" "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/VRP/check/../data/eil7.vrp")
set_tests_properties(examples-vrp-eil7 PROPERTIES  DEPENDS "examples-vrp-build" _BACKTRACE_TRIPLES "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/VRP/check/CMakeLists.txt;38;add_test;/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/examples/VRP/check/CMakeLists.txt;0;")
