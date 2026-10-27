read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/egout.mps"
write problem temp/egout.mps.ccg
presolve
write transproblem temp/egout.mps_trans.ccg
read temp/egout.mps_trans.ccg
optimize
read temp/egout.mps.ccg
optimize
validatesolve "568.1007" "568.1007"
quit
