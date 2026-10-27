read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/rgn.mps"
write problem temp/rgn.mps.pbm
presolve
write transproblem temp/rgn.mps_trans.pbm
read temp/rgn.mps_trans.pbm
optimize
read temp/rgn.mps.pbm
optimize
validatesolve "82.1999974" "82.1999974"
quit
