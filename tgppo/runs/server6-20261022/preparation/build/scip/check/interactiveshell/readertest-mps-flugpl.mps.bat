read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/flugpl.mps"
write problem temp/flugpl.mps.mps
presolve
write transproblem temp/flugpl.mps_trans.mps
read temp/flugpl.mps_trans.mps
optimize
read temp/flugpl.mps.mps
optimize
validatesolve "1201500" "1201500"
quit
