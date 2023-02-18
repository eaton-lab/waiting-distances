#!/usr/bin/env python

"""Validate MS-SMC' solutions for waiting distances to tree/topo changes.

The main function (simulate) takes a species tree and other parameters
as input and simulates tree sequences under the specified species tree.
For each tree sequence the first genealogy is sampled (tree0) and then
the observed waiting distance until each of the possible event types
is recorded (tree-change, or topo-change). In addition, the probabilities
and expected distances until each event type is observed is also calculated
for each tree sequence based on the species tree and first genealogy.

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
import toytree
import ipcoal


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
    # ipcoal Model
    model: ipcoal.Model = ipcoal.Model(
        tree=sptree,
        Ne=neff,
        seed_trees=seed,
        nsamples=nsamples,
        recomb=recomb,
        record_full_arg=True,
        ancestry_model="smc_prime" if smc else "hudson",
    )

    # mapping species tree names to gene tree samples.
    imap: Dict[str, List[str]] = model.get_imap_dict()

    # results array (to be filled)
    results: np.ndarray = np.zeros(shape=(nloci, 8))

    # simulate nloci independent tree sequences
    for lidx in range(nloci):

        # init a new tree sequence generator (new seed)
        tgen = model._get_tree_sequence_generator(nsites=nsites)

        # sample the tree sequence
        tseq = next(tgen)

        # get a copy of the tree sequence that has been simplified.
        # Because it was simulated with record_full_arg=True there are
        # many records of recombination that cause no-change that add
        # extra nodes to the trees. We need to simplify these to more
        # easily find the intervals at which changes occur, and to have
        # simpler trees that can be compared to detect event types.
        stseq = tseq.simplify(filter_sites=False)

        # get the starting tree and its sum branch lengths
        tree0 = stseq.at(0, sample_lists=True)
        tsumlen0 = tree0.get_total_branch_length()

        # compute analytical probabilities of change given tree0
        toy0 = toytree.tree(tree0.as_newick(node_labels=model.tipdict))
        prob_tree = ipcoal.smc.get_probability_tree_change(
            model.tree, toy0, imap)
        prob_topo = ipcoal.smc.get_probability_topology_change(
            model.tree, toy0, imap)

        # compute lambda_ (rate) of tree/topo change given sptree and tree0
        tree_rate = tsumlen0 * prob_tree * recomb
        topo_rate = tsumlen0 * prob_topo * recomb

        # get the gene tree at the next *event*. Here we first fetch 
        # from `tseq` since only this tree sequence contains events that
        # may not change the tree. At this position we then fetch the
        # tree from the simplified tree sequence.
        first_event_position = tseq.at(0).interval[1]

        # get simplified tree at this position so we can compared to
        # tree0 to identify which type of event occurred.
        tree1 = stseq.at(first_event_position, sample_lists=True)

        # 1st event -----------------------------------------------
        # store type of 1st event: 0=no-change, 1=tree-change, 2=topo-change
        # and the observed distance until that event.
        observed_tree_dist = 0
        observed_topo_dist = 0
        if tree0.kc_distance(tree1, lambda_=1):      # diff in blens
            if tree0.kc_distance(tree1, lambda_=0):  # diff in topology
                event_type = 2
                observed_tree_dist = tree1.interval[0]
                observed_topo_dist = tree1.interval[0]
            else:
                event_type = 1
                observed_tree_dist = tree1.interval[0]
        else:
            event_type = 0

        # until all event types observed --------------------------
        # iterate over trees from index 2 -> End to get waiting
        # distances until the first observed tree and topo-change
        # events in this tree sequence, then break, or an Iteration
        # error will raise if you reach end of the tree sequence and
        # and need to increase nsites.
        for idx in range(2, 999999):

            # end when a topology-change event has been observed.
            if observed_topo_dist:
                break

            # get position at end of this interval
            try:
                interval = tseq.at_index(idx).interval
            except IndexError as inst:
                raise ValueError(
                    "End of tree sequence reached with no observed "
                    "topology change events. Try increasing the nsites "
                    "parameter, or changing the species tree parameters."
                ) from inst

            # get new tree at end of interval position
            tree1 = stseq.at(interval[0], sample_lists=True)

            # if no difference in branch lengths then go to next idx
            if tree0.kc_distance(tree1, lambda_=1):
                # diff in blens only
                if not observed_tree_dist:
                    observed_tree_dist = tree1.interval[0]

                # diff in topology only (done)
                if tree0.kc_distance(tree1, lambda_=0):
                    observed_topo_dist = tree1.interval[0]
                    break

        # store final results
        results[lidx, 0] = prob_tree
        results[lidx, 1] = prob_topo
        results[lidx, 2] = stats.expon.mean(scale=1 / tree_rate)
        results[lidx, 3] = stats.expon.mean(scale=1 / topo_rate)
        results[lidx, 4] = observed_tree_dist
        results[lidx, 5] = observed_topo_dist
        results[lidx, 6] = tsumlen0
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
            seed = rng.integers(1e9)

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

    # save results to file
    outname = Path(".") / f"{outname}"
    np.save(outname.with_suffix(".npy"), results)


if __name__ == "__main__":

    # GLOBALS
    RECOMB = 2e-9
    SPECIES_TREE_HEIGHT = 1e6
    NEFF_MIN = 50_000
    NEFF_MAX = 500_000
    NEFF_NVALUES = 10
    SEED = 123
    NLOCI = 100
    NREPS = 100
    OUTNAME = "validate"
    NCORES = 70
    # NLOCI = 10
    # NREPS = 10
    # OUTNAME = "test"

    # THIS TAKES ABOUT xxx TO RUN ON AN 8-CORE LAPTOP. (6:13 START)
    # TRY LOWERING NLOCI AND NREPS WHEN TESTING.
    for npops, nsamples in [(1, 8), (2, 4), (8, 1)]:
        for smc in [True, False]:

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
                outname=f"{OUTNAME}-npops{npops}-{'smc' if smc else 'full'}",
            )
            distribute_jobs(**kwargs)
