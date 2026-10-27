read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/MANN_a9.clq.lp"
write problem temp/MANN_a9.clq.lp.gms
presolve
write transproblem temp/MANN_a9.clq.lp_trans.gms
read temp/MANN_a9.clq.lp_trans.gms
optimize
read temp/MANN_a9.clq.lp.gms
optimize
validatesolve "16" "16"
quit
