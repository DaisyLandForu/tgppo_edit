read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/stein27.fzn"
write problem temp/stein27.fzn.ccg
presolve
write transproblem temp/stein27.fzn_trans.ccg
read temp/stein27.fzn_trans.ccg
optimize
read temp/stein27.fzn.ccg
optimize
validatesolve "18" "18"
quit
