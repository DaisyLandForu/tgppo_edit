read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/stein27.fzn"
write problem temp/stein27.fzn.cip
presolve
write transproblem temp/stein27.fzn_trans.cip
read temp/stein27.fzn_trans.cip
optimize
read temp/stein27.fzn.cip
optimize
validatesolve "18" "18"
quit
