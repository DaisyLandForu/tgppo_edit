read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/blend2.mps"
write problem temp/blend2.mps.pip
presolve
write transproblem temp/blend2.mps_trans.pip
read temp/blend2.mps_trans.pip
optimize
read temp/blend2.mps.pip
optimize
validatesolve "7.598985" "7.598985"
quit
