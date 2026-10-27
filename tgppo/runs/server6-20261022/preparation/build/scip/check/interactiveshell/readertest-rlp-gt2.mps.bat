read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/gt2.mps"
write problem temp/gt2.mps.rlp
presolve
write transproblem temp/gt2.mps_trans.rlp
read temp/gt2.mps_trans.rlp
optimize
read temp/gt2.mps.rlp
optimize
validatesolve "21166" "21166"
quit
