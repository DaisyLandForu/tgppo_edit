read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/bell5.mps"
write problem temp/bell5.mps.cip
presolve
write transproblem temp/bell5.mps_trans.cip
read temp/bell5.mps_trans.cip
optimize
read temp/bell5.mps.cip
optimize
validatesolve "8966406.49" "8966406.49"
quit
