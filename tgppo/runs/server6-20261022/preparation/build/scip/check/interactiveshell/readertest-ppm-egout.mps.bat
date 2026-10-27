read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/egout.mps"
write problem temp/egout.mps.ppm
presolve
write transproblem temp/egout.mps_trans.ppm
read temp/egout.mps_trans.ppm
optimize
read temp/egout.mps.ppm
optimize
validatesolve "568.1007" "568.1007"
quit
