read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/egout.mps"
write problem temp/egout.mps.mps
presolve
write transproblem temp/egout.mps_trans.mps
read temp/egout.mps_trans.mps
optimize
read temp/egout.mps.mps
optimize
validatesolve "568.1007" "568.1007"
quit
