read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/stein27.fzn"
write problem temp/stein27.fzn.pip
presolve
write transproblem temp/stein27.fzn_trans.pip
read temp/stein27.fzn_trans.pip
optimize
read temp/stein27.fzn.pip
optimize
validatesolve "18" "18"
quit
