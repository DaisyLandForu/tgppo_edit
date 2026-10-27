read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/stein27_inf.lp"
write problem temp/stein27_inf.lp.opb
presolve
write transproblem temp/stein27_inf.lp_trans.opb
read temp/stein27_inf.lp_trans.opb
optimize
read temp/stein27_inf.lp.opb
optimize
validatesolve "+infinity" "+infinity"
quit
