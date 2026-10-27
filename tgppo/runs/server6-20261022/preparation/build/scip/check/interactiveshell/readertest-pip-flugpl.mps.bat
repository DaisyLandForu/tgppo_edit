read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/flugpl.mps"
write problem temp/flugpl.mps.pip
presolve
write transproblem temp/flugpl.mps_trans.pip
read temp/flugpl.mps_trans.pip
optimize
read temp/flugpl.mps.pip
optimize
validatesolve "1201500" "1201500"
quit
