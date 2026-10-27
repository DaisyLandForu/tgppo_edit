read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/MANN_a9.clq.lp"
write problem temp/MANN_a9.clq.lp.mps
presolve
write transproblem temp/MANN_a9.clq.lp_trans.mps
read temp/MANN_a9.clq.lp_trans.mps
optimize
read temp/MANN_a9.clq.lp.mps
optimize
validatesolve "16" "16"
quit
