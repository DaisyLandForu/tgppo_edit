"""Exact gated-8 tail replacement already tested in the offline head probe."""
import torch


def replace_tail(actor):
    head = actor.output_layer
    assert len(head.BranchingNet) == 8 and head.n_units_dict[5] == 8
    last = torch.nn.Linear(8, 1)
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
    head.BranchingNet = torch.nn.ModuleList([*list(head.BranchingNet[:5]), torch.nn.Sequential(last)])
    head.n_units_dict = {i: head.n_units_dict[i] for i in range(6)}
    head.n_layers = 6
    assert head.n_attentional_units == 510
    return actor
