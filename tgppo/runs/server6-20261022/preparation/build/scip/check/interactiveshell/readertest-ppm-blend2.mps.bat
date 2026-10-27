read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/blend2.mps"
write problem temp/blend2.mps.ppm
presolve
write transproblem temp/blend2.mps_trans.ppm
read temp/blend2.mps_trans.ppm
optimize
read temp/blend2.mps.ppm
optimize
validatesolve "7.598985" "7.598985"
quit
