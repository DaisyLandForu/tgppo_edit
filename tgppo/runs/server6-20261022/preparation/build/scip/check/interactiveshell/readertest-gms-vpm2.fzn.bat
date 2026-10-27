read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/vpm2.fzn"
write problem temp/vpm2.fzn.gms
presolve
write transproblem temp/vpm2.fzn_trans.gms
read temp/vpm2.fzn_trans.gms
optimize
read temp/vpm2.fzn.gms
optimize
validatesolve "13.75" "13.75"
quit
