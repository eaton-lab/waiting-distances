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


def simulate(
    sptree: toytree.ToyTree,
    nsamples: int,
    neff: int,
    nsites: int,
    nloci: int,
    recomb: float,
    seed: int,
    smc: bool,
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
    smc: bool
        If True then the ancestry_model in msprime is "smc_prime" else
        it is "hudson". In 'hudson' recomb events can re-attach to diff
        ancestors, whereas in 'smc_prime' they can only re-attach to
        ancestors of the samples. Thus 'smc_prime' leads to fewer
        observed recombination events.
    """
    set_num_threads(1)

    # ipcoal Model
    model = ipcoal.Model(
        tree=sptree,
        Ne=int(neff),
        seed_trees=seed,
        nsamples=nsamples,
        recomb=recomb,
        record_full_arg=True,
        discrete_genome=False,
        ancestry_model="smc_prime" if smc else "hudson",
    )

    # mapping species tree names to gene tree samples.
    imap: Dict[str, List[str]] = model.get_imap_dict()

    # results array (to be filled)
    results: np.ndarray = np.zeros(shape=(nloci, 8))

    # simulate nloci independent tree sequences
    for lidx in range(nloci):

        # init a new tree sequence generator (new seed)
        while 1:
            # sample a full tree sequence for this chromosome
            try:
                tseq = next(model._get_tree_sequence_generator(nsites=nsites))

            # catch the rare error that can occur when doing many many sims
            # at very low Ne setting: "msprime._msprime.LibraryError: The simulation
            # model supplied resulted in a parent node having a time value <= to
            # its child. This can occur either as a result of multiple bottlenecks
            # happening at the same time or because of numerical imprecision with
            # very small population sizes.".
            except msprime._msprime.LibraryError:
                continue
            break

        # get a copy of the tree sequence that has been simplified.
        # Because it was simulated with record_full_arg=True there are
        # many records of recombination that cause no-change that add
        # extra nodes to the trees. We need to simplify these to more
        # easily find the intervals at which changes occur, and to have
        # simpler trees that can be compared to detect event types.
        stseq = tseq.simplify(filter_sites=False)

        # get the starting tree in each tree sequence.
        tree = tseq.first(sample_lists=True)
        simple_tree0 = stseq.first(sample_lists=True).copy()

        # iterate over subsequent intervals until the first topology
        # change event is observed. Start from that new fresh tree.
        while 1:
            tree.next()
            next_simple_tree = stseq.at(tree.interval.left, sample_lists=True)

            # if the topology changed then break and save tree as
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
        prob_tree_unchanged = ipcoal.smc.src.get_prob_tree_unchanged_from_arrays(
            T.emb[0], T.enc[0], T.barr[0], T.sarr[0])
        prob_topo_unchanged = ipcoal.smc.src.get_prob_topo_unchanged_from_arrays(
            T.emb[0], T.enc[0], T.barr[0], T.sarr[0], T.rarr[0])

        # compute lambda_ (rate) of tree/topo change given sptree and tree1
        tree_rate = tsumlen1 * (1 - prob_tree_unchanged) * recomb
        topo_rate = tsumlen1 * (1 - prob_topo_unchanged) * recomb

        # RECORD FIRST EVENT TYPE ------------------------------------
        # iterate over subsequent intervals of non-simplified tree seq
        # until each change event type is observed.
        observed_topo_dist = 0.
        observed_tree_dist = 0.
        event_type = None

        # advance to next tree in non-simple treeseq and get simplified tree
        tree.next()
        next_simple_tree = stseq.at(tree.interval.left, sample_lists=True)

        # record event type between this and next interval and record
        # the distance if there was a change.
        if tree1.kc_distance(next_simple_tree, lambda_=1):      # diff in blens
            if tree1.kc_distance(next_simple_tree, lambda_=0):  # diff in topology
                event_type = 2
                observed_tree_dist = tree.interval.left
                observed_topo_dist = tree.interval.left
            else:
                event_type = 1
                observed_tree_dist = tree.interval.left
        else:
            event_type = 0

        # RECORD DISTANCE TO REMAINING EVENTS NOT YET OBSERVED --------
        # iterate over subsequent trees until a topo change is observed.
        while 1:

            # end when a topology-change event has been observed.
            if observed_topo_dist:
                break

            # advance the sampled tree in the full tree seq
            if not tree.next():
                raise ValueError(
                    "End of tree sequence reached with no observed "
                    "topology change events. Try increasing the nsites "
                    "parameter, or changing the species tree parameters."
                )

            # get simplified tree in this new interval
            next_simple_tree = stseq.at(tree.interval.left, sample_lists=True)

            # if no difference in branch lengths then go to next
            if tree1.kc_distance(next_simple_tree, lambda_=1):
                # diff in blens only
                if not observed_tree_dist:
                    observed_tree_dist = tree.interval.left

                # diff in topology only (done)
                if tree1.kc_distance(next_simple_tree, lambda_=0):
                    observed_topo_dist = tree.interval.left
                    break

        # store final results
        results[lidx, 0] = (1 - prob_tree_unchanged)
        results[lidx, 1] = (1 - prob_topo_unchanged)
        results[lidx, 2] = stats.expon.mean(scale=1 / tree_rate)
        results[lidx, 3] = stats.expon.mean(scale=1 / topo_rate)
        results[lidx, 4] = observed_tree_dist - start
        results[lidx, 5] = observed_topo_dist - start
        results[lidx, 6] = tsumlen1
        results[lidx, 7] = event_type
    return results


def get_sptree(ntips: int, height: float) -> toytree.ToyTree:
    """Return balanced tree with ntips and even internal branch lengths.
    """
    if ntips == 1:
        sptree = toytree.tree("(a);")
    else:
        sptree = toytree.rtree.baltree(ntips)
    sptree = sptree.mod.edges_scale_to_root_height(height, include_stem=True)
    return sptree


def distribute_jobs(
    sptree: toytree.ToyTree,
    nsamples: int,
    neff_min: int,
    neff_max: int,
    neff_nvalues: int,
    nloci: int,
    nreps: int,
    recomb: float,
    seed: int,
    smc: bool,
    ncores: int,
    outname: str,
) -> np.ndarray:
    """Parallelize simulator function across Ne values, reps, and SMC/Full
    """
    # compare smc and full
    results = np.zeros((nloci * nreps, neff_nvalues, 8))

    # Ne values to test over
    nes = np.linspace(neff_min, neff_max, neff_nvalues).astype(int)

    # run jobs in parallel to fill array
    rasyncs = {}
    with ProcessPoolExecutor(max_workers=ncores) as pool:

        # apply a different seed to each rep
        rng = np.random.default_rng(seed)
        for rep in range(nreps):
            seed = rng.integers(1e12)

            # apply same seed for each diff value of Ne
            for nidx, neff in enumerate(nes):

                # smaller nsites simulates faster. We just need to make
                # tree sequences long enough to ensure that each contains
                # at least one topology change event. To keep sims fast
                # we thus inversely scale nsites by neff since large nsites
                # is only needed when neff is very small. This may need to
                # be increased if other parameters are changed.
                nsites = int(550000 - neff)

                # submit smc' jobs
                kwargs = {
                    "sptree": sptree,
                    "nsamples": nsamples,
                    "neff": neff,
                    "nsites": nsites,
                    "seed": seed,
                    "nloci": nloci,
                    "recomb": recomb,
                    "smc": bool(smc),
                }
                rasyncs[(nidx, rep)] = pool.submit(simulate, **kwargs)

    # collect results into large res array
    for key, future in rasyncs.items():
        nidx, rep = key
        ival = slice(nloci * rep, nloci * (rep + 1))
        iresults = future.result()
        results[ival, nidx, :] = iresults

        # ...
        # tmpname = Path(".") / f"tmp-{outname}-{nidx}-{rep}"
        # np.save(tmpname.with_suffix(".npy"), iresults)

    # save results to file
    outname = Path(f"{outname}")
    np.save(outname.with_suffix(".npy"), results)


if __name__ == "__main__":

    # parse some optional args if provided, else use defaults
    import sys
    args = sys.argv[1:]
    NCORES = int(args[0]) if args else 55
    OUTNAME = str(args[1]) if len(args) > 1 else "validate-100K"
    NLOCI = int(args[2]) if len(args) > 2 else 100
    NREPS = int(args[3]) if len(args) > 3 else 1000

    # other GLOBALS
    RECOMB = 2e-9
    SPECIES_TREE_HEIGHT = 1e6
    NEFF_MIN = 50_000
    NEFF_MAX = 500_000
    NEFF_NVALUES = 10
    SEED = 123
    # NLOCI = 100
    # NREPS = 1000
    # OUTNAME = "validate-100K"
    # NCORES = 55

    # TEST PARAMS
    # NCORES = 8
    # NLOCI = 2
    # NREPS = 6
    # OUTNAME = "TESTX2"

    # THE TEST PARAMS TAKE <10 minutes TO RUN ON AN 8-CORE LAPTOP.
    # THE FULL PARAMS TAKE 100X longer and should be run on a cluster
    # or workstation with the NCORES params cranked up.
    # SETUPS = [(8, 3), (8, 1), (1, 8), (2, 4), (2, 8)]
    SETUPS = [(8, 1), (2, 4), (1, 8)]    
    for (npops, nsamples) in SETUPS:
        for smc in [False, True]:

            kwargs = dict(
                sptree=get_sptree(ntips=npops, height=SPECIES_TREE_HEIGHT),
                nsamples=nsamples,
                neff_min=NEFF_MIN,
                neff_max=NEFF_MAX,
                neff_nvalues=NEFF_NVALUES,
                # nsites=500_000,  # automatically set in distribute_jobs().
                nloci=NLOCI,
                nreps=NREPS,
                recomb=RECOMB,
                seed=SEED,
                smc=smc,
                ncores=NCORES,
                outname=f"{OUTNAME}-npops{npops}-nsamps{nsamples}-{'smc' if smc else 'full'}",
            )
            distribute_jobs(**kwargs)
