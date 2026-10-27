read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/flugpl.mps"
write problem temp/flugpl.mps.ccg
presolve
write transproblem temp/flugpl.mps_trans.ccg
read temp/flugpl.mps_trans.ccg
optimize
read temp/flugpl.mps.ccg
optimize
validatesolve "1201500" "1201500"
quit
