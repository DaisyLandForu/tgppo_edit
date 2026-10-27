read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/egout.mps"
write problem temp/egout.mps.pip
presolve
write transproblem temp/egout.mps_trans.pip
read temp/egout.mps_trans.pip
optimize
read temp/egout.mps.pip
optimize
validatesolve "568.1007" "568.1007"
quit
