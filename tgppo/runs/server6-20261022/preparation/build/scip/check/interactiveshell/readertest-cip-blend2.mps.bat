read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/blend2.mps"
write problem temp/blend2.mps.cip
presolve
write transproblem temp/blend2.mps_trans.cip
read temp/blend2.mps_trans.cip
optimize
read temp/blend2.mps.cip
optimize
validatesolve "7.598985" "7.598985"
quit
