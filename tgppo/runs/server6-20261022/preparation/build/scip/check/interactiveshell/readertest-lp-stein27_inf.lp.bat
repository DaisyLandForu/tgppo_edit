read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/stein27_inf.lp"
write problem temp/stein27_inf.lp.lp
presolve
write transproblem temp/stein27_inf.lp_trans.lp
read temp/stein27_inf.lp_trans.lp
optimize
read temp/stein27_inf.lp.lp
optimize
validatesolve "+infinity" "+infinity"
quit
