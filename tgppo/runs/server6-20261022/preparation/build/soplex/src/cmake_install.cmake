# Install script for directory: /home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/duweiyue25/tgppo-transfer/research/tgppo/solver")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/soplex" TYPE FILE FILES
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/array.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/basevectors.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/classarray.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/clufactor.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/clufactor_rational.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/cring.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/dataarray.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/datahashtable.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/datakey.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/dataset.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/didxset.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/dsvectorbase.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/dsvector.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/dvectorbase.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/dvector.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/exceptions.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/gzstream.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/idlist.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/idxset.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/islist.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/lpcolbase.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/lpcol.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/lpcolsetbase.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/lpcolset.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/lprowbase.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/lprow.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/lprowsetbase.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/lprowset.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/mpsinput.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/nameset.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/notimer.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/random.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/rational.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/ratrecon.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/slinsolver.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/slinsolver_rational.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/slufactor.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/slufactor_rational.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/solbase.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/sol.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/sorter.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxalloc.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxautopr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxbasis.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxboundflippingrt.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxdantzigpr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxdefaultrt.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxdefines.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxdevexpr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxequilisc.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxfastrt.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxfileio.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxgeometsc.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxgithash.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxharrisrt.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxhybridpr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxid.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxleastsqsc.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxlpbase.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxlp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxmainsm.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxout.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxparmultpr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxpricer.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxratiotester.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxscaler.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxsimplifier.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxsolver.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxstarter.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxsteepexpr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxsteeppr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxsumst.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxvectorst.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxweightpr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/spxweightst.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/ssvectorbase.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/ssvector.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/stablesum.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/statistics.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/svectorbase.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/svector.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/svsetbase.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/svset.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/timerfactory.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/timer.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/unitvectorbase.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/unitvector.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/updatevector.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/usertimer.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/validation.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/vectorbase.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/vector.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex/wallclocktimer.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/soplex-4.0.1/src/soplex.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/soplex" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/soplex")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/soplex"
         RPATH "/home/duweiyue25/tgppo-transfer/research/tgppo/solver/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/soplex/bin/soplex")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/soplex" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/soplex")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/soplex"
         OLD_RPATH ":::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
         NEW_RPATH "/home/duweiyue25/tgppo-transfer/research/tgppo/solver/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/soplex")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/soplex/lib/libsoplex.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/soplex/lib/libsoplex-pic.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/soplex/soplex-targets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/soplex/soplex-targets.cmake"
         "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/soplex/src/CMakeFiles/Export/lib/cmake/soplex/soplex-targets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/soplex/soplex-targets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/soplex/soplex-targets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/soplex" TYPE FILE FILES "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/soplex/src/CMakeFiles/Export/lib/cmake/soplex/soplex-targets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/soplex" TYPE FILE FILES "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/soplex/src/CMakeFiles/Export/lib/cmake/soplex/soplex-targets-release.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/soplex" TYPE FILE FILES "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/soplex/CMakeFiles/soplex-config.cmake")
endif()

