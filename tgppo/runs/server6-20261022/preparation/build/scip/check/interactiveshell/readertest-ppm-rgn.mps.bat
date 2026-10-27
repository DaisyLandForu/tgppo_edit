read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/rgn.mps"
write problem temp/rgn.mps.ppm
presolve
write transproblem temp/rgn.mps_trans.ppm
read temp/rgn.mps_trans.ppm
optimize
read temp/rgn.mps.ppm
optimize
validatesolve "82.1999974" "82.1999974"
quit
