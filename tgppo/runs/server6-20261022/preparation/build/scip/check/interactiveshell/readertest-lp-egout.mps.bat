read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/egout.mps"
write problem temp/egout.mps.lp
presolve
write transproblem temp/egout.mps_trans.lp
read temp/egout.mps_trans.lp
optimize
read temp/egout.mps.lp
optimize
validatesolve "568.1007" "568.1007"
quit
