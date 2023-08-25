#!/usr/bin/env python

"""Posterior sampling of demographic model parameters under the MS-SMC'
by Metropolis-Hastings MCMC.

TODO:
Instead of making a separate embedding table for the topology-changes
we should instead make the same table as for tree-changes but also
return a mask or index of the trees representing topology-changes to
subset from the same embedding. This saves time for re-embeddings.

"""

from typing import Dict, Sequence
# from abc import ABC, abstractmethod
import argparse
import time
import sys
import itertools
from copy import deepcopy
from pathlib import Path
from datetime import timedelta
import numpy as np
import ipcoal
# import toyplot, toyplot.svg, toyplot.png
import toytree
from numba import set_num_threads
from scipy import stats
from loguru import logger
from ipcoal.msc import get_msc_loglik_from_embedding
from ipcoal.smc import get_genealogy_embedding_arrays
from ipcoal.smc import get_ms_smc_loglik_from_embedding
# from ipcoal.smc.src import iter_spans_and_topos_from_model
# from ipcoal.smc.src import iter_spans_and_trees_from_model
from ipcoal.smc.src.utils import get_waiting_distance_data_from_model
# get_ms_smc_data_from_model,
from ipcoal.smc.src.embedding import _jit_update_neff

# optional. If installed the ESS will be printed.
try:
    import arviz as az
except ImportError:
    pass


logger = logger.bind(name="ipcoal")


class Mcmc2:
    """A custom Metropolis-Hastings sampler."""
    def __init__(
        self,
        species_tree: toytree.ToyTree,
        genealogies: Sequence[toytree.ToyTree],
        imap: Dict[str, Sequence[str]],
        embedding: ipcoal.smc.TreeEmbedding,
        tree_spans: np.ndarray,
        topo_spans: np.ndarray,
        topo_idxs: np.ndarray,
        init_params: np.ndarray,
        prior_dists: Sequence[stats.rv_continuous],
        seed: int,
        msc: bool,
        smc: bool,
        outpath: Path,
        fixed_params: Sequence[float],
        # jumpsize: int,
    ):
        # store the inputs
        self.species_tree = species_tree
        self.genealogies = genealogies
        self.imap = imap
        self.tree_spans = tree_spans
        self.topo_spans = topo_spans
        self.topo_idxs = topo_idxs
        """: Lengths of tree and/or topo waiting distances."""
        self.embedding = embedding
        """: Embedding with current accepted param values."""
        self.prior_dists = prior_dists
        """: Frozen prob densities for priors."""
        self.params = np.array(init_params)
        """: The current params."""
        self._params = np.array(init_params)
        """: The proposed params."""
        self.rng = np.random.default_rng(seed)
        """: Seeded random number generator"""
        self.jumpsize = self.params * 0.10  # 0.125
        """: MCMC proposal size. Larger values decrease acceptance rate."""
        self.outpath = outpath
        """: Path to npy saved output."""
        self.smc = smc
        """: calculate smc likelihood."""
        self.msc = msc
        """: calculate msc likelihood."""

        # ...
        self.fixed_params = [] if fixed_params is None else fixed_params
        pidxs = list(range(len(self.params)))
        for i in self.fixed_params:
            pidxs.remove(i)
        self._proposal_idx = itertools.cycle(pidxs)

        # The embedding updated to the proposed params
        self._embedding = deepcopy(self.embedding)

    @classmethod
    def get_accept_ratio(cls, old: float, new: float) -> bool:
        """Return acceptance ratio for proposal"""
        if np.isnan(new):
            ratio = 0
        else:
            ratio = 1 if new < old else np.exp(old - new)
        return ratio

    def log_likelihood(self) -> float:
        """Return log-likelihood of the data (lengths, edicts) given params.

        This uses the current embedding arrays data which stores info
        such as the population Ne and diverence times.
        """
        loglik = 0
        # logger.warning(self.smc)
        # logger.warning(self._embedding)
        # logger.warning(self._params)
        if self.smc == 1:
            loglik += get_ms_smc_loglik_from_embedding(
                embedding=self._embedding,
                recombination_rate=self._params[-1],
                lengths=self.tree_spans,
                event_type=1,
            )
        elif self.smc == 2:
            loglik += get_ms_smc_loglik_from_embedding(
                embedding=self._embedding,
                recombination_rate=self._params[-1],
                lengths=self.topo_spans,
                event_type=2,
                idxs=self.topo_idxs,
            )
        elif self.smc == 3:
            loglik += get_ms_smc_loglik_from_embedding(
                embedding=self._embedding,
                recombination_rate=self._params[-1],
                lengths=self.tree_spans,
                event_type=1,
            )
            loglik += get_ms_smc_loglik_from_embedding(
                embedding=self._embedding,
                recombination_rate=self._params[-1],
                lengths=self.topo_spans,
                event_type=2,
                idxs=self.topo_idxs,
            )
        if self.msc:
            loglik += get_msc_loglik_from_embedding(embedding=self._embedding.emb)
        return loglik

    def get_proposal(self, index: int) -> np.ndarray:
        """Return a proposal for one parameter at a time."""
        return self.rng.normal(loc=self.params[index], scale=self.jumpsize[index])

    def prior_log_likelihood(self) -> float:
        """Return prior loglikelihood"""
        pdens = [i.logpdf(j) for (i, j) in zip(self.prior_dists, self._params)]
        return -sum(pdens)

    def update_embedding(self, pidx: int = None) -> None:
        """Update the embedding given the proposed parameter change.

        If Ne value changed then this needs to be updated in each
        layer of the embedding array. If Tau changed then the embedding
        array needs to be completely reconstructed. If recomb changed
        then no change is needed.
        """
        # select the next parameter to change
        if pidx is None:
            pidx = next(self._proposal_idx)

        # set neff value on current embedding tables
        if pidx < self.species_tree.nnodes:
            self._params = self.params.copy()
            self._params[pidx] = self.get_proposal(pidx)
            self._embedding.emb = _jit_update_neff(self.embedding.emb, pidx, self._params[pidx])
            logger.debug(f"CURRENT EMBEDDING {self.params}\n{self.embedding.get_table(1)}")
            logger.debug(f"PROPOSE EMBEDDING {self._params}\n{self._embedding.get_table(1)}")

        # set a tau value to new setting and get new embeddings. This is
        # only currently tested for 2-tip tree. Need a more advanced
        # node slider for more tips.
        elif pidx < len(self.params) - 1:
            self._params = self.params.copy()
            while 1:
                tau = self.get_proposal(pidx)
                self._params[pidx] = tau

                # try/except to sample an allowed node height
                try:
                    # raise an exception to sample new value if out of bounds
                    t = self.species_tree
                    node = t[pidx - t.ntips + 1]
                    for child in node.children:
                        if child.height >= tau:
                            raise ValueError("tau too low")
                    if node.up and (node.up.height <= tau):
                        raise ValueError("tau too high")

                    # tau cannot be higher than parent, or lower than child
                    taus = dict(zip(range(t.ntips, t.nnodes), self._params[t.nnodes:-1]))
                    neffs = dict(zip(range(t.nnodes), self._params[:t.nnodes]))
                    tmptree = t.set_node_data("height", taus).set_node_data("Ne", neffs)

                    # re-embed the gene trees in the species tree
                    # logger.debug(f"trying tau = {self.params[pidx]:.3e} -> {self._params[pidx]:.3e} ")
                    emb, enc = get_genealogy_embedding_arrays(tmptree, self.genealogies, self.imap)
                    break
                except ValueError:
                    pass
            self._embedding.emb = emb
            self._embedding.enc = enc
            # logger.info(f"\n{self.embedding.get_table(0)}")
            # logger.info(f"\n{self._embedding.get_table(0)}")

        # update recombination rate
        else:
            self._params = self.params.copy()
            self._params[pidx] = self.get_proposal(pidx)
            # self._embedding = deepcopy(self.embedding)
            # self._embedding.emb = self.embedding.emb.copy()
            # self._embedding.enc = embedding.enc.copy()
        return pidx

    def run(
        self,
        nsamples: int = 1000,
        burnin: int = 2000,
        sample_interval: int = 5,
        print_interval: int = 25,
        init_values: Sequence[float] = None,
    ) -> np.ndarray:
        """Run to sample from the posterior distribution.

        Parameters
        ----------
        nsamples: int
            Number of accepted samples to save in the posterior distribution.
        sample_interval: int
            Number of accepted proposals between saving a sample to the posterior.
        print_interval: int
            Number of accepted proposals between printing progress to stdout.
        burnin: int
            Number of accepted samples to skip before starting sampling.

        Returns
        -------
        np.ndarray
        """
        # get starting values user or sample from priors
        if init_values in ([], None):
            init_values = np.array([
                self.params[i] if i in self.fixed_params
                else self.prior_dists[i].rvs()
                for i in range(len(self.params))
            ])

        # set init values as the current param settings
        self.params = init_values
        self._params = self.params.copy()

        # update embedding to match current params
        jumpsize = self.jumpsize.copy()
        self.jumpsize[:] = 0
        for i in range(len(self.params)):
            self.update_embedding()
            self.embedding.emb = self._embedding.emb.copy()
            self.embedding.enc = self._embedding.enc.copy()
        self.jumpsize = jumpsize

        # self.update_embedding(0)
        # self.embedding._emb = self._embedding.emb.copy()
        # self.embedding._enc = self._embedding.enc.copy()

        # format initial values (sampled or user supplied)
        values = '\t'.join([f'{i:.5g}' for i in self._params])

        # report likelihood of starting params
        prior_loglik = self.prior_log_likelihood()
        data_loglik = self.log_likelihood()
        curr_loglik = prior_loglik + data_loglik
        logger.info(f"log-likelihood of start params ({values}): {prior_loglik:.3f} {data_loglik:.3f} {curr_loglik:.3f}")

        # start sampler
        logger.info(f"starting MCMC sampler for {nsamples} samples.")
        pnames = "\t".join([f"   param{i}" for i in range(len(self.params))])
        logger.info(f"iter\tsample\tprior-loglik\tdata-loglik\tposterior\t{pnames}\taccept\ttime\tcat")
        start = time.time()

        # array to store posterior samples
        posterior = np.zeros(shape=(nsamples, len(self.params) + 1))

        # counters
        idx = 0      # iteration index
        aidx = 0     # accepted proposal index
        sidx = 0     # sample stored index
        its = 0
        acc = 0
        pidx = 0
        aratios = np.ma.array([0] * len(self.params), mask=np.bincount(self.fixed_params, minlength=len(self.params)))
        apropos = np.zeros(len(self.params))
        while 1:

            # update ._params and ._embedding with new proposal. In some
            # proposals the value cannot be set because it does not
            # allow embedding of genealogies. In this case loglik=0.
            uidx = self.update_embedding()
            prior_loglik = self.prior_log_likelihood()
            data_loglik = self.log_likelihood()
            new_loglik = prior_loglik + data_loglik
            aratio = self.get_accept_ratio(curr_loglik, new_loglik)

            # accept or reject
            acc += aratio
            its += 1
            accept = int(aratio > self.rng.random())
            _cparams = "  ".join([f"{i:.4e}" for i in self.params])
            _nparams = "  ".join([f"{i:.4e}" for i in self._params])
            logger.debug(f"{curr_loglik:.2f} old=[{_cparams}] ")
            logger.debug(f"{new_loglik:.2f} new=[{_nparams}] aratio={aratio:.3f}\n")

            aratios[uidx] += aratio
            apropos[uidx] += 1

            # tuning during the burnin
            if (idx < burnin) and (not idx % 30):
                oldjump = self.jumpsize.copy()
                accept_rates = aratios / apropos
                for i, j in enumerate(accept_rates.mask):
                    if not j:
                        rate = accept_rates[i]
                        if rate < 0.3:
                            self.jumpsize[i] *= 0.5
                        elif rate > 0.6:
                            self.jumpsize[i] *= 1.5
                logger.debug(f"\n{accept_rates}\n{oldjump}\n{self.jumpsize}")
                aratios = np.ma.array([0] * len(self.params), mask=np.bincount(self.fixed_params, minlength=len(self.params)))
                apropos = np.zeros(len(self.params))

            # proposal rejected
            if not accept:
                # revert to last accepted embedding
                self._embedding.emb = self.embedding.emb.copy()
                if (uidx > self.species_tree.nnodes) and (uidx < len(self.params)):
                    self._embedding.enc = self.embedding.enc.copy()

            # proposal accepted
            else:
                self.params = self._params.copy()
                # store accepted embedding
                if uidx < len(self.params):
                    self.embedding.emb = self._embedding.emb.copy()
                if (uidx > self.species_tree.nnodes) and (uidx < len(self.params)):
                    self.embedding.enc = self._embedding.enc.copy()
                curr_loglik = new_loglik
                aidx += 1

                # only store every Nth accepted result
                if idx > burnin:
                    if (aidx % sample_interval) == 0:
                        posterior[sidx] = list(self.params) + [new_loglik]
                        sidx += 1

                # print on interval
                if not aidx % print_interval:
                    elapsed = timedelta(seconds=int(time.time() - start))
                    stype = "sample" if idx > burnin else "burnin"
                    params = "\t".join([f"{i:.3e}" for i in self.params])
                    logger.info(
                        f"{aidx:>4}\t"
                        f"{sidx:>6}\t"
                        f"{prior_loglik:>12.3f}\t"
                        f"{data_loglik:>8.3f}\t"
                        f"{new_loglik:>8.3f}\t"
                        f"{params}\t"
                        f"{acc/its:.2f}\t"
                        f"{elapsed}\t{stype}"
                    )

                # save to disk and print summary every 1K sidx
                if sidx and (not sidx % 100) and sidx != pidx:
                    np.save(self.outpath, posterior[:sidx])
                    logger.info("checkpoint saved.")
                    means = posterior[:sidx].mean(axis=0)
                    means = "\t".join([f"{i:.5e}" for i in means])
                    stds = posterior[:sidx].std(axis=0)
                    stds = "\t".join([f"{i:.5e}" for i in stds])
                    logger.info(f"MCMC current posterior mean={means}")
                    logger.info(f"MCMC current posterior std ={stds}")

                    # print mcmc if optional pkg arviz is installed.
                    if sys.modules.get("arviz"):
                        ess_vals = []
                        for col in range(posterior.shape[1]):
                            azdata = az.convert_to_dataset(posterior[:sidx, col])
                            ess = az.ess(azdata).x.values
                            ess_vals.append(int(ess))
                        logger.info(f"MCMC current posterior ESS ={ess_vals}\n")
                    pidx = sidx

                # # adjust tuning of the jumpsize during burnin
                # if idx < burnin:
                #    self.jumpsize = self.params * 0.2
                #     if not idx % 100:
                #         if acc/its < 44:
                #             self.jumpsize += 1000
                #         if acc/its > 44:
                #             self.jumpsize -= 1000

            # advance counter and break when nsamples reached
            idx += 1
            if sidx == nsamples:
                break
        logger.info(f"MCMC sampling complete. Means={posterior.mean(axis=0)}")
        return posterior


def main(
    tree: str,
    params: Sequence[float],
    nsamples: int,
    nloci: int,
    nsites: int,
    priors: Sequence[str],
    seed: int,
    name: str,
    mcmc_nsamples: int,
    mcmc_sample_interval: int,
    mcmc_print_interval: int,
    mcmc_burnin: int,
    msc: bool,
    smc: bool,
    smc_data_type: str,
    force: bool,
    fixed_params: Sequence[int],
    init_values: Sequence[float],
    *args,
    **kwargs,
) -> None:
    """Run the main function of the script.

    This simulates a tree sequence under a given demographic model
    and generates a genealogy embedding table representing info on all
    genealogies across the chromosome, and their lengths (the ARG
    embedded in the MSC model, as a table).

    An MCMC algorithm is then run to search over the prior parameter
    space of the demographic model parameters to find the best fitting
    Ne values to explain the observed tree sequence waiting distances
    under tree changes.

    The posterior is saved to file as a numpy array.
    """
    outpath = Path(name).expanduser().absolute().with_suffix(".npy")
    outpath.parent.mkdir(exist_ok=True)
    if force and outpath.exists():
        outpath.unlink()

    # get species tree topology
    sptree = toytree.tree(tree)

    # require correct number params for tree size
    assert len(params) == sptree.nnodes + (sptree.nnodes - sptree.ntips) + 1

    # set Ne and Tau values on species tree
    true_params = params.copy()
    true_neff = dict(zip(range(sptree.nnodes), params[:sptree.nnodes]))
    true_tau = dict(zip(range(sptree.ntips, sptree.nnodes), params[sptree.nnodes:-1]))
    true_recomb = params[-1]
    sptree.set_node_data("Ne", data=true_neff, inplace=True)
    sptree.set_node_data("height", data=true_tau, inplace=True)

    # simulate genealogies under True parameter settings to get ARGs
    model = ipcoal.Model(
        sptree,
        nsamples=nsamples,
        recomb=true_recomb,
        seed_trees=seed,
        discrete_genome=False,
        ancestry_model="smc_prime",
    )

    # get mapping of sample names to lineages
    imap = model.get_imap_dict()

    # generate tree sequences given true params
    logger.info(f"simulating {nloci} tree sequences of len={nsites:.2g} w/ recomb={true_recomb:.2g}")
    model.sim_trees(nloci=nloci, nsites=nsites, nproc=5)

    # get MS-SMC ARG data from model
    logger.info("extracting interval lengths from tree sequences")
    # tree_spans, topo_spans, topo_idxs, trees = get_ms_smc_data_from_model(model)
    tree_spans, topo_spans, topo_idxs, trees = get_waiting_distance_data_from_model(model)

    # get initial embedding
    logger.info(f"embedding includes {len(trees)} tree-changes, {len(topo_spans)} topo-changes")
    edata = ipcoal.smc.TreeEmbedding(model.tree, trees, imap)

    # convert priors into probability distributions
    prior_dists = []
    for tup in priors:
        tup = tup.split(",")
        # (U, 2, 3) = uniform min max
        if tup[0].upper() == "U":
            dist = stats.uniform.freeze(loc=float(tup[1]), scale=float(tup[2]) - float(tup[1]))
        # (I, 2, 300) = invgamma alpha beta
        elif tup[0].upper() == "I":
            dist = stats.invgamma.freeze(a=float(tup[1]), scale=float(tup[2]))
        else:
            raise ValueError("prior not supported.")
        prior_dists.append(dist)

    # smc
    if not smc:
        smc = 0
    else:
        if smc_data_type == "combined":
            smc = 3
        elif smc_data_type == "topology":
            smc = 2
        else:
            smc = 1

    # ...
    mcmc = Mcmc2(
        species_tree=model.tree,
        genealogies=trees,
        imap=imap,
        embedding=edata,
        tree_spans=tree_spans,
        topo_spans=topo_spans,
        topo_idxs=topo_idxs,
        init_params=true_params,
        prior_dists=prior_dists,
        seed=seed,
        msc=msc,
        smc=smc,
        outpath=outpath,
        fixed_params=fixed_params,
    )

    # report loglik at true params and then set back to start params
    prior_loglik = mcmc.prior_log_likelihood()
    data_loglik = mcmc.log_likelihood()
    loglik = data_loglik + prior_loglik
    logger.info(f"log-likelihood of true params ({params}): {prior_loglik:.3f} {data_loglik:.3f} {loglik:.3f}")

    # run MCMC chain
    posterior = mcmc.run(
        nsamples=mcmc_nsamples,
        print_interval=mcmc_print_interval,
        sample_interval=mcmc_sample_interval,
        burnin=mcmc_burnin,
        init_values=init_values,
    )

    # if adding to existing data then concatenate first.
    # if outpath.exists():
    #     posterior = np.concatenate([sampled, posterior])
    np.save(outpath, posterior)
    logger.info(f"saved posterior w/ {posterior.shape[0]} samples to {outpath}.")


def command_line():
    """Parse command line arguments and return.

    python mcmc2.py \
        --tree (a,b); \
        --params 2e5 2e5 2e5 1e6 2e-9 \
        --nloci 10 \
        --nsites 1e6 \
        --nsamples 8 \
        --name TEST \
        --priors i,3,5e5 i,3,5e5 i,3,5e5 i,3,2e6 i,3,5e-9 \
        --mcmc-sample-interval 3 \
        --threads 4 \
        --force True \
        --log-level DEBUG \
        --data-type combined
    """
    parser = argparse.ArgumentParser(description="MCMC model fit for MS-SMC")
    # parser.add_argument(
    #     '--ntips', type=int, default=2, help='Number of species tree tips')
    # parser.add_argument(
    #     '--root-height', type=float, default=1e6, help='Root height of species tree.')
    # parser.add_argument(
    #     '--recomb', type=float, default=2e-9, help='Recombination rate.')
    parser.add_argument(
        '--tree', type=str, default="(A,B)C;", help='A newick tree topology.')
    parser.add_argument(
        '--params', type=float, default=[2e5, 2e5, 2e5, 1e6, 2e-9], nargs="*", help='Ne[x nnodes] Tau[x nnodes-2] and recomb values.')
    # i,3,5e5 i,3,5e5 i,3,5e5 i,3,2e6 i,3,4e-9
    parser.add_argument(
        '--priors', type=str, nargs="*", default=["i,3,4e5", "i,3,4e5", "i,3,4e5", "i,3,2e6", "i,3,4e-9"], help='Priors: U,2e5,5e5 I,3,1000')
    parser.add_argument(
        '--nloci', type=int, default=6, help='number of independent loci to simulate')
    parser.add_argument(
        '--nsites', type=float, default=5e5, help='length of simulated tree sequences (loci)')
    parser.add_argument(
        '--nsamples', type=int, default=4, help='Number of samples per species lineage')
    parser.add_argument(
        '--seed', type=int, default=666, help='Random number generator seed')
    parser.add_argument(
        '--name', type=str, default='smc', help='Prefix path for output files')
    parser.add_argument(
        '--mcmc-nsamples', type=int, default=1000, help='Number of samples in posterior')
    parser.add_argument(
        '--mcmc-sample-interval', type=int, default=6, help='N accepted iterations between samples')
    parser.add_argument(
        '--mcmc-print-interval', type=int, default=6, help='N accepted iterations between printing progress')
    parser.add_argument(
        '--mcmc-burnin', type=int, default=100, help='N accepted iterations before starting sampling')
    parser.add_argument(
        '--threads', type=int, default=7, help='Max number of threads (0=all detected)')
    parser.add_argument(
        '--force', type=bool, default=True, help='Overwrite existing file w/ same name.')
    parser.add_argument(
        '--msc', action="store_true", help='Calculate MSC likelihood.')
    parser.add_argument(
        '--smc', action="store_true", help='Calculate SMC likelihood.')
    parser.add_argument(
        '--smc-data-type', type=str, default="combined", help='tree, topology, or combined')
    # parser.add_argument(
    #     '--mcmc-jumpsize', type=float, default=[10_000, 20_000, 30_000], nargs="*", help='MCMC jump size.')
    parser.add_argument(
        '--log-level', type=str, default="INFO", help='logger level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument(
        '--fixed-params', type=int, default=None, nargs="*", help='Fixed params at true values')
    parser.add_argument(
        '--init-values', type=float, default=None, nargs="*", help='Fixed params at true values')
    return parser.parse_args()


# TODO: RESUME?


if __name__ == "__main__":

    # get command line args
    cli_args = command_line()

    # set logger
    ipcoal.set_log_level(cli_args.log_level)

    # limit n threads
    if cli_args.threads:
        set_num_threads(cli_args.threads)

    # run main
    main(**vars(cli_args))
