#!/usr/bin/env python

"""Validate MS-SMC' solutions for waiting distances to tree/topo changes.

The main function (simulate) takes a species tree and other parameters
as input and simulates tree sequences under the specified species tree.
For each tree sequence the first genealogy is sampled (tree0). We then
sample until the first topology-change event occurs to find tree1. This
is treated as our true starting tree. (This tweak was made after finding
that the distance from the init tree in a tskit tree-sequence until the
next recomb event is anomalous compared to all other waiting distances,
so we instead start fresh from the next topology change.)
From here, we calculate the probabilities of events and record the 
observed recombination events to the next tree or topology.

The final results is saved as .npy (ndarray) and written to the current
directory where this script is run.

# results array shape = (nloci, n_neff_values_examined, 8):
# 0 = smc_tree_probs (MS-SMC calculated probs)
# 1 = smc_topo_probs (MS-SMC calculated probs)
# 2 = smc_tree_dists (MS-SMC predicted distances)
# 3 = smc_topo_dists (MS-SMC predicted distances)
# 4 = sim_tree_dists (observed distances)
# 5 = sim_topo_dists (observed distances)
# 6 = sum_edge_lengths of genealogy 0 (observed data)
# 7 = event type (observed data)
"""

from typing import Dict, List
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy import stats
from numba import set_num_threads
import toytree
import ipcoal
import msprime


def get_sptree(ntips: int, root_height: float=1e6) -> toytree.ToyTree:
    """Return a balanced species tree with the specified number of tips and even branch lengths."""
    if ntips == 1:
        sptree = toytree.tree("(a);")
    else:
        sptree = toytree.rtree.baltree(ntips)
    sptree = sptree.mod.edges_scale_to_root_height(root_height, include_stem=True)
    return sptree


def get_bias(
    sptree: toytree.ToyTree,
    nsamples: int,
    neff: int,
    nsites: int,
    nloci: int,
    recomb: float,
    seed: int,
    first_tree: bool,
) -> np.ndarray:
    """Calculate waiting distances from simulations and MS-SMC'.

    Paramters
    ----------
    sptree: ToyTree
        A species tree topology w/ edge lengths in generations.
    nsamples: int
        Number of samples to simulate in each population.
    neff: int
        Constant effective pop size applied to all sptree intervals.
    nsites: int
        The length of simulated tree sequences. Must be long enough to
        contain a topology event type or an exception will be raised.
    nloci: int
        Replicate tree sequences from diff seeds for which results
        are returned in the results array.
    recomb: float
        The per site per generation recombination rate.
    seed: int
        Seed for the random number generator.
    first_tree: bool
        True returns values for first two trees separated by tree-change
        if False returns first and last trees in an interval between
        a topo-change event.
    """
    set_num_threads(1)

    # ipcoal Model
    model = ipcoal.Model(
        tree=sptree,
        Ne=neff,
        seed_trees=seed,
        nsamples=nsamples,
        recomb=recomb,
        record_full_arg=True,
        discrete_genome=False,
        ancestry_model="hudson",
    )

    # mapping species tree names to gene tree samples.
    imap: Dict[str, List[str]] = model.get_imap_dict()

    # results array (to be filled)
    results: np.ndarray = np.zeros(shape=(nloci, 3))

    # simulate nloci independent tree sequences
    lidx = 0
    while 1:
        # get FULL tree sequence generator and simplified tree generator
        tseq = next(model._get_tree_sequence_generator(nsites=nsites))
        stseq = tseq.simplify(filter_sites=False)

        # get the starting tree in each tree sequence.
        tree = tseq.first(sample_lists=True)
        simple_tree0 = stseq.first(sample_lists=True).copy()

        # iterate over subsequent intervals until the first topology
        # change event is observed. Start from that new fresh tree.
        while 1:
            tree.next()
            next_simple_tree = stseq.at(tree.interval.left, sample_lists=True)

            # if the topology changed then break and save tree as tree1
            # this will be our new starting tree.
            if next_simple_tree.kc_distance(simple_tree0, lambda_=0):
                tree1 = next_simple_tree.copy()
                start = tree1.interval.left
                break

        # get sum edge lengths of tree at starting position
        tsumlen1 = tree1.get_total_branch_length()

        # compute analytical probabilities of change given tree1
        toy1 = toytree.tree(tree1.as_newick(node_labels=model.tipdict))
        T = ipcoal.smc.TreeEmbedding(model.tree, toy1, imap, nproc=1)
        p_topo1 = 1 - ipcoal.smc.src.get_prob_topo_unchanged_from_arrays(
            T.emb[0], T.enc[0], T.barr[0], T.sarr[0], T.rarr[0])

        # iterate until condition is met
        tree_changed = False
        while 1:

            # get simplified tree in next interval
            tree.next()
            next_simple_tree = stseq.at(tree.interval.left, sample_lists=True)

            # if conditioning on FIRST tree change
            if first_tree:
                if tree1.kc_distance(next_simple_tree, lambda_=1):
                    if tree1.kc_distance(next_simple_tree, lambda_=0):
                        break  # no tree-change occurred before topo-change
                    else:
                        tree_changed = True
                        break

            # if conditioning on topo-change (last tree sampled)
            else:
                if tree1.kc_distance(next_simple_tree, lambda_=0):
                    break

        # if tree changed store the result, else re-loop
        if first_tree:
            if tree_changed:
                tsumlen2 = next_simple_tree.get_total_branch_length()
                toy2 = toytree.tree(next_simple_tree.as_newick(node_labels=model.tipdict))
                p_topo2 = 1 - ipcoal.smc.get_prob_topo_unchanged(model.tree, toy2, imap)
                results[lidx] = (
                        p_topo1 / p_topo2,
                        tsumlen1 / tsumlen2,
                        (p_topo1 * tsumlen1) / (p_topo2 * tsumlen2)
                    )
                lidx += 1

        # conditioned on topo-change
        else:
            next_simple_tree = stseq.at(tree.interval.left - 1, sample_lists=True)
            tsumlen2 = next_simple_tree.get_total_branch_length()
            toy2 = toytree.tree(next_simple_tree.as_newick(node_labels=model.tipdict))
            p_topo2 = 1 - ipcoal.smc.get_prob_topo_unchanged(model.tree, toy2, imap)
            results[lidx] = (
                    p_topo1 / p_topo2,
                    tsumlen1 / tsumlen2,
                    (p_topo1 * tsumlen1) / (p_topo2 * tsumlen2)
                )
            lidx += 1

        # finished
        if lidx == NLOCI:
            break        
    return results                    


def distribute_jobs(
    sptree: toytree.ToyTree,
    nsamples: int,
    first_tree: bool,
    neffs: List[int],
    nloci: int,
    nreps: int,
    nsites: int,
    recomb: float,
    seed: int,
    ncores: int,
    outname: str,
) -> np.ndarray:
    """Parallelize simulator function across Ne values, reps, and SMC/Full
    """
    # compare smc and full
    results = np.zeros(shape=(nloci * nreps, 3, 3, 3))

    # run jobs in parallel to fill array
    rasyncs = {}
    with ProcessPoolExecutor(max_workers=ncores) as pool:

        # apply a different seed to each rep
        rng = np.random.default_rng(seed)
        for rep in range(nreps):
            seed = rng.integers(1e12)

            # apply same seed for each diff value of Ne
            for nidx, neff in enumerate(neffs):

                for nsa, mult in enumerate([1, 2, 4]):

                    # submit smc' jobs
                    kwargs = {
                        "sptree": sptree,
                        "nsamples": nsamples,
                        "neff": neff,
                        "nsites": nsites,
                        "nloci": nloci,
                        "recomb": recomb,
                        "seed": seed,
                        "first_tree": first_tree,
                    }
                    rasyncs[(nidx, nsa, rep)] = pool.submit(get_bias, **kwargs)

    # collect results into large res array
    for key, future in rasyncs.items():
        nidx, nsa, rep = key
        ival = slice(nloci * rep, nloci * (rep + 1))
        iresults = future.result()
        results[ival, :, nidx, nsa] = iresults

        # ...
        # tmpname = Path(".") / f"tmp-{outname}-{nidx}-{rep}"
        # np.save(tmpname.with_suffix(".npy"), iresults)

    # save results to file
    outname = Path(".") / f"{outname}"
    np.save(outname.with_suffix(".npy"), results)


if __name__ == "__main__":

    # GLOBALS
    RECOMB = 2e-9
    SPECIES_TREE_HEIGHT = 1e6
    NEFFS = [50_000, 100_000, 500_000]
    SEED = 123
    NLOCI = 10
    NREPS = 100  
    # thus we will get 1000 results from diff random seeds. The params
    # nloci and nreps only affect how the jobs are distributed.
    NCORES = 8

    SETUPS = [(1, 8), (2, 4), (8, 1)]
    #SETUPS = [(8, 1)]
    for (pops, samps) in SETUPS:
        first_tree = False
        sptree = get_sptree(pops)

        distribute_jobs(
            sptree,
            nsamples=samps,
            first_tree=first_tree,
            neffs=NEFFS,
            nloci=NLOCI,
            nreps=NREPS,
            nsites=500_000,
            recomb=RECOMB,
            seed=SEED,
            ncores=NCORES,
            outname=f"bias-fold-full-p{pops}-s{samps}-f{int(first_tree)}.npy",
        )

    # # TEST PARAMS
    # # NCORES = 8
    # # NLOCI = 2
    # # NREPS = 6
    # # OUTNAME = "TESTX2"

    # # THE TEST PARAMS TAKE <10 minutes TO RUN ON AN 8-CORE LAPTOP.
    # # THE FULL PARAMS TAKE 100X longer and should be run on a cluster
    # # or workstation with the NCORES params cranked up.
    # SETUPS = [(8, 3), (8, 1), (1, 8), (2, 4), (2, 8)]
    # for (npops, nsamples) in SETUPS:
    #     for smc in [False, True]:

    #         kwargs = dict(
    #             sptree=get_sptree(ntips=npops, height=SPECIES_TREE_HEIGHT),
    #             nsamples=nsamples,
    #             neff_min=NEFF_MIN,
    #             neff_max=NEFF_MAX,
    #             neff_nvalues=NEFF_NVALUES,
    #             # nsites=500_000,  # automatically set in distribute_jobs().
    #             nloci=NLOCI,
    #             nreps=NREPS,
    #             recomb=RECOMB,
    #             seed=SEED,
    #             smc=smc,
    #             ncores=NCORES,
    #             outname=f"{OUTNAME}-npops{npops}-nsamps{nsamples}-{'smc' if smc else 'full'}",
    #         )
    #         distribute_jobs(**kwargs)
