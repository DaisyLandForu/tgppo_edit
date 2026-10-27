if(NOT TARGET libsoplex)
  include("${CMAKE_CURRENT_LIST_DIR}/soplex-targets.cmake")
endif()

set(SOPLEX_LIBRARIES libsoplex)
set(SOPLEX_PIC_LIBRARIES libsoplex-pic)
set(SOPLEX_INCLUDE_DIRS "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src")
set(SOPLEX_FOUND TRUE)
