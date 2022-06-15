#!/usr/bin/env python

"""Posterior sampling of demographic model parameters under the MS-SMC' by Metropolis-Hastings MCMC

"""

from typing import Tuple, Dict
import argparse
import time
from pathlib import Path
from datetime import timedelta
import numpy as np
import ipcoal
# import toyplot, toyplot.svg, toyplot.png
import toytree
from numba import set_num_threads
from loguru import logger

from ipcoal.smc import get_genealogy_embedding_table
from ipcoal.smc.likelihood.likelihood2 import (
    get_data,
    get_tree_distance_loglik
)

logger = logger.bind(name="ipcoal")


class Mcmc:
    """A custom Metropolis-Hastings sampler."""
    def __init__(
        self,
        recomb: float,
        lengths: np.ndarray,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        priors: Tuple[int, int], 
        init_params: np.ndarray,
        seed: int,
        jumpsize: int
        ):
        # store the inputs
        self.recomb = recomb
        self.lengths = lengths
        self.data = data
        self.priors = np.array([priors[0]] * len(init_params)), np.array([priors[1]] * len(init_params))
        self.params = init_params
        self.rng = np.random.default_rng(seed)
        self.jumpsize = jumpsize
        assert self.prior_uniform(self.params), "starting values are outside priors."
        
    def transition(self) -> np.ndarray:
        """Jitters the current params to new proposal values."""
        return self.rng.normal(self.params, scale=self.jumpsize)
    
    def acceptance(self, old: float, new: float) -> bool:
        """Return boolean for whether to accept new proposal params."""
        if np.isnan(new):
            accepted = 0
        else:
            accepted = 1 if new < old else np.exp(old - new)
        logger.debug(f"old={old:.2f}, new={new:.2f}, accepted={accepted:.2f}")
        return accepted
    
    def log_likelihood(self, params) -> float:
        """Return log-likelihood of the data (lengths, edicts) given the params."""
        return get_tree_distance_loglik(
            params, 
            recomb=self.recomb, 
            lengths=self.lengths, 
            embedding_arr=self.data[0],
            blen_arr=self.data[1], 
            sumlen_arr=self.data[2])

    def prior_uniform(self, params: np.ndarray) -> float:
        """Return prior loglikelihood. Uniform priors return 1 if inside bounds, else inf."""
        low = np.all(params >= self.priors[0])
        high = np.all(params <= self.priors[1])
        return 1 if float(low & high) else 0
        
    def run(self, nsamples: int=1000, burnin=2000, sample_interval=5, print_interval=25):
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
        logger.info(f"starting MCMC sampler for {nsamples} samples.")
        space = " " * (5 * len(self.params))
        logger.info(f"sample\tloglik   \tparams{space}\taccept\truntime\tcat")
        start = time.time()
        posterior = np.zeros(shape=(nsamples, len(self.params) + 1))
        loglik = self.log_likelihood(self.params) * self.prior_uniform(self.params)
        
        its = 0
        idx = 0
        acc = 0
        while 1:
                        
            # propose new params
            new_params = self.transition()

            # get likelihood
            prior_lik = self.prior_uniform(new_params)
            if prior_lik:
                new_loglik = self.log_likelihood(new_params) * prior_lik
            else:
                new_loglik = np.inf
            
            # accept or reject
            aratio = self.acceptance(loglik, new_loglik)
            acc += aratio
            its += 1
            if aratio > self.rng.random():
                
                # proposal accepted
                self.params = new_params
                loglik = new_loglik
                
                # only store every Nth accepted result
                if (idx % sample_interval) == 0:
                    sidx = int(idx / sample_interval)

                    # print/store depending on if in burnin or not
                    if idx > burnin:
                        posterior[int(idx / sample_interval)] = list(self.params) + [new_loglik]
                        if not idx % print_interval:
                            elapsed = timedelta(seconds=int(time.time() - start))
                            logger.info(
                                f"{sidx}\t"
                                f"{new_loglik:.3f}\t"
                                f"{self.params.astype(int)}\t"
                                f"{acc/its:.2f}\t"
                                f"{elapsed}\t"
                                f"sample\t"
                            )
                    else:
                        if not idx % print_interval:
                            elapsed = timedelta(seconds=int(time.time() - start))
                            logger.info(
                                f"{sidx}\t"
                                f"{new_loglik:.3f}\t"
                                f"{self.params.astype(int)}\t"
                                f"{acc/its:.2f}\t"
                                f"{elapsed}\t"
                                f"burnin\t"
                            )
                
                # advance counter of accepted proposals
                idx += 1
                
                # break when requested number of samples are saved
                if idx >= nsamples * sample_interval:
                    break
        logger.info(f"MCMC sampling complete. Means={posterior.mean(axis=0).round(2)}")
        return posterior


def simulate_and_get_embeddings(
    sptree: toytree.ToyTree, 
    params: Dict[str, int], 
    nsamples: int, 
    nsites: int,
    recomb: float,
    seed: int,
    ) -> Tuple:
    """Simulate a tree sequence, get embedding info, and return.
    """
    # set Ne values on species tree
    sptree.set_node_data("Ne", mapping=params, inplace=True)
    
    # setup a coalescent simulation model
    model = ipcoal.Model(sptree, nsamples=nsamples, recomb=recomb, seed_trees=seed)

    # get mapping of sample names to lineages
    imap = model.get_imap_dict()

    # print some details
    logger.info(f"simulating tree sequence for {nsites:.2g} sites w/ recomb={recomb:.2g}.")
    
    # generate a tree sequence and store to a table
    model.sim_trees(nloci=1, nsites=nsites)

    # get lengths for every genealogy
    lengths = model.df.nbps.values

    # load genealogies
    genealogies = toytree.mtree(model.df.genealogy)

    # print some details
    logger.info(f"loading genealogy embedding table for {len(genealogies)} genealogies.")
    
    # get cached embedding tables
    etables = [get_genealogy_embedding_table(model.tree, i, imap) for i in genealogies]
    
    # get combined arrays of embedding data
    earr, barr, sarr = get_data(etables)
    
    # return all data
    return lengths, earr, barr, sarr


def get_species_tree(
    ntips: int, 
    root_height: float,
    ) -> toytree.ToyTree:
    """Return a species tree with same height given ntips."""
    if ntips == 1:
        sptree = toytree.tree("(r);")
    else:
        sptree = toytree.rtree.baltree(ntips)
    sptree = sptree.mod.edges_scale_to_root_height(treeheight=root_height, include_stem=True)
    return sptree


def main(
    ntips: int,
    root_height: float,
    params: Tuple[int],
    nsamples: int,
    nsites: int,
    recomb: float,
    seed: int,
    name: str,
    mcmc_nsamples: int,
    mcmc_sample_interval: int,
    mcmc_print_interval: int,
    mcmc_burnin: int,
    mcmc_jumpsize: int,    
    force: bool,
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
    sptree = get_species_tree(ntips, root_height)

    # convert params to dict
    params = {i: params[i] for i in range(sptree.nnodes)}

    # simulate genealogies under MSC topology and parameters
    # and get the ARG and embedding data.
    lengths, earr, barr, sarr = simulate_and_get_embeddings(
        sptree, params, nsamples, nsites, recomb, seed)

    # initial random params 
    init_params = np.repeat(5e5, len(params))

    # does a checkpoint file already exist for this run?
    if outpath.exists():
        # get its length, subtract from nsamples, and set burnin to 0
        sampled = np.load(outpath)
        nsampled = sampled.shape[0]
        mcmc_burnin = 0
        mcmc_nsamples = mcmc_nsamples - nsampled
        init_params = sampled[-1][:-1]
        seed = sampled[-1][-1]
        logger.info(f"restarting from checkpoint (samples={nsampled})")

    # init MCMC object
    mcmc = Mcmc(
        recomb=recomb,
        lengths=lengths,
        data=(earr, barr, sarr),
        priors=(1e3, 5e6),
        init_params=init_params,
        jumpsize=mcmc_jumpsize,
        seed=seed,
    )

    # run MCMC chain
    posterior = mcmc.run(
        nsamples=mcmc_nsamples, 
        print_interval=mcmc_print_interval, 
        sample_interval=mcmc_sample_interval, 
        burnin=mcmc_burnin,
    )

    # if adding to existing data then concatenate first.
    if outpath.exists():
        posterior = np.concatenate([sampled, posterior])
    np.save(outpath, posterior)
    logger.info(f"saved posterior w/ {posterior.shape[0]} samples to {outpath}.")


def command_line():
    """Parse command line arguments and return."""
    parser = argparse.ArgumentParser(
        description="MCMC model fit for MS-SMC'")
    parser.add_argument(
        '--ntips', type=int, default=2, help='Number of species tree tips')
    parser.add_argument(
        '--root-height', type=float, default=1e6, help='Root height of species tree.')
    parser.add_argument(
        '--params', type=float, default=[100_000, 200_000, 300_000], nargs="*", help='True Ne values used for simulated data.')
    parser.add_argument(
        '--recomb', type=float, default=2e-9, help='Recombination rate.')
    parser.add_argument(
        '--nsites', type=float, default=5e6, help='length of simulated tree sequence')
    parser.add_argument(
        '--nsamples', type=int, default=4, help='Number of samples per species lineage')
    parser.add_argument(
        '--seed', type=int, default=123, help='Random number generator seed')
    parser.add_argument(
        '--name', type=str, default='test', help='Prefix path for output files')
    parser.add_argument(
        '--mcmc-nsamples', type=int, default=300, help='Number of samples in posterior')
    parser.add_argument(
        '--mcmc-sample-interval', type=int, default=2, help='N accepted iterations between samples')
    parser.add_argument(
        '--mcmc-print-interval', type=int, default=5, help='N accepted iterations between printing progress')
    parser.add_argument(
        '--mcmc-burnin', type=int, default=50, help='N accepted iterations before starting sampling')
    parser.add_argument(
        '--threads', type=int, default=7, help='Max number of threads (0=all detected)')
    parser.add_argument(
        '--force', type=bool, default=True, help='Overwrite existing file w/ same name.')
    parser.add_argument(
        '--mcmc-jumpsize', type=int, default=30_000, help='MCMC jump size.')
    parser.add_argument(
        '--log-level', type=str, default="INFO", help='logger level (DEBUG, INFO, WARNING, ERROR)')
    return parser.parse_args()


if __name__ == "__main__":

    # get command line args
    args = command_line()

    # set logger
    ipcoal.set_log_level(args.log_level)

    # limit n threads
    if args.threads:
        set_num_threads(args.threads)
    

    # run main
    main(**vars(args))
