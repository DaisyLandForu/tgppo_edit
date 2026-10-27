read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/flugpl.mps"
write problem temp/flugpl.mps.pbm
presolve
write transproblem temp/flugpl.mps_trans.pbm
read temp/flugpl.mps_trans.pbm
optimize
read temp/flugpl.mps.pbm
optimize
validatesolve "1201500" "1201500"
quit
