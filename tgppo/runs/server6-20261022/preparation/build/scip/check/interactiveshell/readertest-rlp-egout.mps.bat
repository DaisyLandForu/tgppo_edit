read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/egout.mps"
write problem temp/egout.mps.rlp
presolve
write transproblem temp/egout.mps_trans.rlp
read temp/egout.mps_trans.rlp
optimize
read temp/egout.mps.rlp
optimize
validatesolve "568.1007" "568.1007"
quit
