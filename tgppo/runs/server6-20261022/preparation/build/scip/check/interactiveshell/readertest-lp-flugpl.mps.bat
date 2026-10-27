read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/flugpl.mps"
write problem temp/flugpl.mps.lp
presolve
write transproblem temp/flugpl.mps_trans.lp
read temp/flugpl.mps_trans.lp
optimize
read temp/flugpl.mps.lp
optimize
validatesolve "1201500" "1201500"
quit
