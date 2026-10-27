read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/p0548.mps"
write problem temp/p0548.mps.rlp
presolve
write transproblem temp/p0548.mps_trans.rlp
read temp/p0548.mps_trans.rlp
optimize
read temp/p0548.mps.rlp
optimize
validatesolve "8691" "8691"
quit
