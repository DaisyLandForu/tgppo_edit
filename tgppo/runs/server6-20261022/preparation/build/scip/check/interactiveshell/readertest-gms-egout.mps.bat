read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/egout.mps"
write problem temp/egout.mps.gms
presolve
write transproblem temp/egout.mps_trans.gms
read temp/egout.mps_trans.gms
optimize
read temp/egout.mps.gms
optimize
validatesolve "568.1007" "568.1007"
quit
