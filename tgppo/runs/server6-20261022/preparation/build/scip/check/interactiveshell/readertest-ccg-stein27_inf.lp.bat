read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/stein27_inf.lp"
write problem temp/stein27_inf.lp.ccg
presolve
write transproblem temp/stein27_inf.lp_trans.ccg
read temp/stein27_inf.lp_trans.ccg
optimize
read temp/stein27_inf.lp.ccg
optimize
validatesolve "+infinity" "+infinity"
quit
