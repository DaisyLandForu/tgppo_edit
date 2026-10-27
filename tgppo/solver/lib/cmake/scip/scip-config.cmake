if(NOT TARGET libscip)
  include("${CMAKE_CURRENT_LIST_DIR}/scip-targets.cmake")
endif()

set(ZIMPL_NEEDED "")
set(SOPLEX_NEEDED "1")

if(ZIMPL_NEEDED)
   set(ZIMPL_DIR "")
   find_package(ZIMPL QUIET CONFIG)
endif()

if(SOPLEX_NEEDED)
   set(SOPLEX_DIR "/home/duweiyue25/tgppo-transfer/research/tgppo/solver/lib/cmake/soplex")
   find_package(SOPLEX QUIET CONFIG)
endif()

set(SCIP_LIBRARIES libscip)
set(SCIP_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../../include")
set(SCIP_FOUND TRUE)
