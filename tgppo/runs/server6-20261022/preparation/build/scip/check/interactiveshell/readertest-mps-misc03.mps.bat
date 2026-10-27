read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/misc03.mps"
write problem temp/misc03.mps.mps
presolve
write transproblem temp/misc03.mps_trans.mps
read temp/misc03.mps_trans.mps
optimize
read temp/misc03.mps.mps
optimize
validatesolve "3360" "3360"
quit
