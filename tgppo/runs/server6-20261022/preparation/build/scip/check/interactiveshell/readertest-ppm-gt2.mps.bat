read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/gt2.mps"
write problem temp/gt2.mps.ppm
presolve
write transproblem temp/gt2.mps_trans.ppm
read temp/gt2.mps_trans.ppm
optimize
read temp/gt2.mps.ppm
optimize
validatesolve "21166" "21166"
quit
