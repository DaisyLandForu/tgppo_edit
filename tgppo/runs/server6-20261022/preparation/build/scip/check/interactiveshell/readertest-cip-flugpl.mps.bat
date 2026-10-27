read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/flugpl.mps"
write problem temp/flugpl.mps.cip
presolve
write transproblem temp/flugpl.mps_trans.cip
read temp/flugpl.mps_trans.cip
optimize
read temp/flugpl.mps.cip
optimize
validatesolve "1201500" "1201500"
quit
