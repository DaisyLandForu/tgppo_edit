read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/stein27_inf.lp"
write problem temp/stein27_inf.lp.pip
presolve
write transproblem temp/stein27_inf.lp_trans.pip
read temp/stein27_inf.lp_trans.pip
optimize
read temp/stein27_inf.lp.pip
optimize
validatesolve "+infinity" "+infinity"
quit
