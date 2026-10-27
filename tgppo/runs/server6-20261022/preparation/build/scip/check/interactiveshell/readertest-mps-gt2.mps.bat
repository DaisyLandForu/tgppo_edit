read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/gt2.mps"
write problem temp/gt2.mps.mps
presolve
write transproblem temp/gt2.mps_trans.mps
read temp/gt2.mps_trans.mps
optimize
read temp/gt2.mps.mps
optimize
validatesolve "21166" "21166"
quit
