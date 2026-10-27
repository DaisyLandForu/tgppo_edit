read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/bell5.mps"
write problem temp/bell5.mps.rlp
presolve
write transproblem temp/bell5.mps_trans.rlp
read temp/bell5.mps_trans.rlp
optimize
read temp/bell5.mps.rlp
optimize
validatesolve "8966406.49" "8966406.49"
quit
