# Install script for directory: /home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/lpi" TYPE FILE FILES
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/lpi/lpi.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/lpi/type_lpi.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/dijkstra" TYPE FILE FILES "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/dijkstra/dijkstra.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/nlpi" TYPE FILE FILES
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/nlpi/exprinterpret.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/nlpi/intervalarithext.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/nlpi/nlpi_all.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/nlpi/nlpi.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/nlpi/nlpi_filtersqp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/nlpi/nlpi_ipopt.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/nlpi/nlpi_worhp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/nlpi/nlpioracle.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/nlpi/pub_expr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/nlpi/struct_expr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/nlpi/struct_nlpi.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/nlpi/type_expr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/nlpi/type_exprinterpret.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/nlpi/type_nlpi.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/objscip" TYPE FILE FILES
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objbenders.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objbenderscut.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objbranchrule.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objcloneable.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objconshdlr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objdialog.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objdisp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objeventhdlr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objheur.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objmessagehdlr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objnodesel.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objpresol.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objpricer.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objprobcloneable.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objprobdata.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objprop.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objreader.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objrelax.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objscipdefplugins.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objscip.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objsepa.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objtable.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/objvardata.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/type_objcloneable.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/objscip/type_objprobcloneable.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/scip" TYPE FILE FILES
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/bandit.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/bandit_epsgreedy.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/bandit_exp3.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/bandit_ucb.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/benders.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/benders_default.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/benderscut.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/benderscut_feas.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/benderscut_int.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/benderscut_nogood.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/benderscut_opt.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/bendersdefcuts.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/bitencode.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/boundstore.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/branch_allfullstrong.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/branch_cloud.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/branch_distribution.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/branch_fullstrong.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/branch.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/branch_inference.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/branch_leastinf.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/branch_lookahead.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/branch_mostinf.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/branch_multaggr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/branch_nodereopt.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/branch_pscost.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/branch_random.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/branch_relpscost.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/clock.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/compr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/compr_largestrepr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/compr_weakcompr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/concsolver.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/concsolver_scip.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/concurrent.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/conflict.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/conflictstore.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_abspower.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_and.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_benders.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_benderslp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_bivariate.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_bounddisjunction.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_cardinality.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_components.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_conjunction.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_countsols.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_cumulative.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_disjunction.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_indicator.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_integral.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_knapsack.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_linear.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_linking.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_logicor.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_nonlinear.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_orbisack.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_orbitope.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_or.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_pseudoboolean.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_quadratic.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_setppc.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_soc.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_sos1.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_sos2.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_superindicator.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_symresack.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_varbound.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cons_xor.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cutpool.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/cuts.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/dbldblarith.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/debug.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/def.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/dialog_default.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/dialog.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/disp_default.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/disp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/event_globalbnd.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/event.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/event_softtimelimit.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/event_solvingphase.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_actconsdiving.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_bound.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_clique.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_coefdiving.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_completesol.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_conflictdiving.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_crossover.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_dins.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_distributiondiving.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_dualval.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_farkasdiving.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_feaspump.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_fixandinfer.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_fracdiving.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_gins.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_guideddiving.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_indicator.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_intdiving.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_intshifting.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heuristics.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_linesearchdiving.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_localbranching.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_locks.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_alns.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_lpface.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_multistart.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_mutation.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_mpec.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_nlpdiving.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_objpscostdiving.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_octane.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_ofins.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_oneopt.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_proximity.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_pscostdiving.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_randrounding.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_rens.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_reoptsols.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_repair.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_rins.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_rootsoldiving.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_rounding.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_shiftandpropagate.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_shifting.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_simplerounding.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_subnlp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_sync.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_trivial.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_trivialnegation.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_trysol.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_twoopt.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_undercover.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_vbounds.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_veclendiving.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_zeroobj.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/heur_zirounding.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/history.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/implics.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/interrupt.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/intervalarith.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/lp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/mem.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/message_default.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/message.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/misc.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/nlp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/nodesel_bfs.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/nodesel_breadthfirst.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/nodesel_dfs.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/nodesel_estimate.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/nodesel.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/nodesel_hybridestim.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/nodesel_restartdfs.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/nodesel_uct.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/paramset.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/presol_boundshift.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/presol_convertinttobin.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/presol_domcol.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/presol_dualagg.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/presol_dualcomp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/presol_dualinfer.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/presol_gateextraction.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/presol.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/presol_implics.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/presol_inttobinary.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/presol_qpkktref.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/presol_redvub.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/presol_sparsify.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/presol_stuffing.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/presol_symbreak.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/presol_symmetry.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/presol_trivial.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/presol_tworowbnd.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/presolve.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pricer.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pricestore.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/primal.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/prob.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/prop_dualfix.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/prop_genvbounds.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/prop.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/prop_nlobbt.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/prop_obbt.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/prop_orbitalfixing.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/prop_probing.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/prop_pseudoobj.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/prop_redcost.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/prop_rootredcost.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/prop_sync.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/prop_vbounds.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_branch.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_bandit.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_bandit_epsgreedy.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_bandit_exp3.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_bandit_ucb.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_benders.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_benderscut.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_compr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_conflict.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_cons.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_cutpool.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_dialog.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_disp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_event.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_fileio.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_heur.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_history.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_implics.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_lp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_matrix.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_message.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_misc.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_misc_linear.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_misc_select.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_misc_sort.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_nlp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_nodesel.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_paramset.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_presol.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_pricer.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_prop.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_reader.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_relax.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_reopt.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_sepa.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_sol.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_table.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_tree.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/pub_var.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/rbtree.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_bnd.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_ccg.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_cip.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_cnf.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_cor.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_diff.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_fix.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_fzn.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_gms.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_lp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_mps.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_mst.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_opb.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_osil.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_pbm.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_pip.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_ppm.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_rlp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_sol.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_smps.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_sto.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_tim.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_wbo.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reader_zpl.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/relax.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/reopt.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/retcode.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scipbuildflags.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scipcoreplugins.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scipdefplugins.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scipgithash.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_bandit.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_benders.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_branch.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_compr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_concurrent.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_conflict.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_cons.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_copy.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_cut.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_datastructures.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_debug.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_dialog.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_disp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_event.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_expr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_general.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_heur.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_lp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_mem.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_message.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_nlp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_nodesel.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_nonlinear.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_numerics.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_param.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_presol.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_pricer.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_prob.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_probing.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_prop.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_randnumgen.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_reader.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_relax.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_reopt.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_sepa.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_sol.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_solve.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_solvingstats.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_table.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_timing.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_tree.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_validation.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scip_var.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/scipshell.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/sepa_cgmip.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/sepa_clique.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/sepa_closecuts.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/sepa_aggregation.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/sepa_convexproj.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/sepa_disjunctive.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/sepa_eccuts.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/sepa_gauge.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/sepa_gomory.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/sepa.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/sepa_impliedbounds.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/sepa_intobj.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/sepa_mcf.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/sepa_oddcycle.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/sepa_rapidlearning.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/sepastore.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/sepa_strongcg.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/sepa_zerohalf.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/set.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/sol.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/solve.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/stat.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_bandit.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_benders.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_benderscut.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_branch.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_clock.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_compr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_concsolver.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_concurrent.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_conflict.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_conflictstore.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_cons.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_cutpool.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_cuts.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_dialog.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_disp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_event.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_heur.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_history.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_implics.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_lp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_matrix.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_mem.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_message.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_misc.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_nlp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_nodesel.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_paramset.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_presol.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_pricer.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_pricestore.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_primal.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_prob.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_prop.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_reader.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_relax.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_reopt.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_scip.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_sepa.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_sepastore.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_set.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_sol.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_stat.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_syncstore.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_table.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_tree.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_var.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/struct_visual.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/syncstore.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/table_default.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/table.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/tree.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_bandit.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_benders.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_benderscut.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_branch.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_clock.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_compr.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_concsolver.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_concurrent.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_conflict.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_conflictstore.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_cons.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_cutpool.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_cuts.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_dialog.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_disp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_event.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_heur.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_history.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_implics.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_interrupt.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_lp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_matrix.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_mem.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_message.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_misc.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_nlp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_nodesel.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_paramset.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_presol.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_pricer.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_pricestore.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_primal.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_prob.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_prop.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_reader.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_relax.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_reopt.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_result.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_retcode.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_scip.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_sepa.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_sepastore.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_set.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_sol.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_stat.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_syncstore.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_table.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_timing.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_tree.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_var.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/type_visual.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/var.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/scip/visual.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip/scip/config.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tclique" TYPE FILE FILES
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/tclique/tclique_coloring.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/tclique/tclique_def.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/tclique/tclique.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tinycthread" TYPE FILE FILES "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/tinycthread/tinycthread.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tpi" TYPE FILE FILES
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/tpi/def_openmp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/tpi/tpi.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/tpi/tpi_none.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/tpi/tpi_openmp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/tpi/tpi_tnycthrd.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/tpi/type_tpi.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/tpi/type_tpi_none.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/tpi/type_tpi_openmp.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/tpi/type_tpi_tnycthrd.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/xml" TYPE FILE FILES
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/xml/xmldef.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/xml/xml.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symmetry" TYPE FILE FILES
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/symmetry/compute_symmetry.h"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/symmetry/type_symmetry.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/blockmemshell" TYPE FILE FILES "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1/src/blockmemshell/memory.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/scip" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/scip")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/scip"
         RPATH "/home/duweiyue25/tgppo-transfer/research/tgppo/solver/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip/bin/scip")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/scip" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/scip")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/scip"
         OLD_RPATH "/home/duweiyue25/tgppo-transfer/research/tgppo/solver/lib:"
         NEW_RPATH "/home/duweiyue25/tgppo-transfer/research/tgppo/solver/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/scip")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libscip.so.6.0.1.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libscip.so.6.0"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "/home/duweiyue25/tgppo-transfer/research/tgppo/solver/lib")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip/lib/libscip.so.6.0.1.0"
    "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip/lib/libscip.so.6.0"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libscip.so.6.0.1.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libscip.so.6.0"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/home/duweiyue25/tgppo-transfer/research/tgppo/solver/lib:"
           NEW_RPATH "/home/duweiyue25/tgppo-transfer/research/tgppo/solver/lib")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libscip.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libscip.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libscip.so"
         RPATH "/home/duweiyue25/tgppo-transfer/research/tgppo/solver/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip/lib/libscip.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libscip.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libscip.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libscip.so"
         OLD_RPATH "/home/duweiyue25/tgppo-transfer/research/tgppo/solver/lib:"
         NEW_RPATH "/home/duweiyue25/tgppo-transfer/research/tgppo/solver/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libscip.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/scip/scip-targets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/scip/scip-targets.cmake"
         "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip/src/CMakeFiles/Export/lib/cmake/scip/scip-targets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/scip/scip-targets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/scip/scip-targets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/scip" TYPE FILE FILES "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip/src/CMakeFiles/Export/lib/cmake/scip/scip-targets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/scip" TYPE FILE FILES "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip/src/CMakeFiles/Export/lib/cmake/scip/scip-targets-release.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/scip" TYPE FILE FILES "/home/duweiyue25/tgppo-transfer/research/tgppo/runs/server6-20261022/preparation/build/scip/CMakeFiles/scip-config.cmake")
endif()

