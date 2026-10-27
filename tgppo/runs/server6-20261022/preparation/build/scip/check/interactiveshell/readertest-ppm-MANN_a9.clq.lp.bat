read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/MANN_a9.clq.lp"
write problem temp/MANN_a9.clq.lp.ppm
presolve
write transproblem temp/MANN_a9.clq.lp_trans.ppm
read temp/MANN_a9.clq.lp_trans.ppm
optimize
read temp/MANN_a9.clq.lp.ppm
optimize
validatesolve "16" "16"
quit
