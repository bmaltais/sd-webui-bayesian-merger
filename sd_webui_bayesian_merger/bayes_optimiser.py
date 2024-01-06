from typing import Dict, List

from bayes_opt import BayesianOptimization, Events
from bayes_opt.domain_reduction import SequentialDomainReductionTransformer
from scipy.stats import qmc

from sd_webui_bayesian_merger.optimiser import Optimiser

class BoundsLogger:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def log_bounds(self, event, instance):
        current_bounds = instance.space.bounds
        print(f"Bounds after optimization step: {current_bounds}")
        
class BoundsLoggerSubscriber:
    def __init__(self, bounds_logger):
        self.bounds_logger = bounds_logger

    def update(self, event, instance):
        self.bounds_logger.log_bounds(event, instance)

class BayesOptimiser(Optimiser):
    bounds_transformer = SequentialDomainReductionTransformer()

    def optimise(self) -> None:
        pbounds = self.init_params()
        print(f"Initial Parameter Bounds: {pbounds}")

        # TODO: fork bayesian-optimisation and add LHS
        self.optimizer = BayesianOptimization(
            f=self.sd_target_function,
            pbounds=pbounds,
            random_state=1,
            bounds_transformer=self.bounds_transformer
            if self.cfg.bounds_transformer
            else None,
            
        )
        
        bounds_logger = BoundsLogger(self.optimizer)
        bounds_logger_subscriber = BoundsLoggerSubscriber(bounds_logger)

        # Log the initial bounds
        print("Initial bounds for optimization:", self.optimizer.space.bounds)
        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, self.logger)
        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, bounds_logger_subscriber)

        init_points = self.cfg.init_points
        if self.cfg.latin_hypercube_sampling:
            sampler = qmc.LatinHypercube(d=len(pbounds))
            samples = sampler.random(self.cfg.init_points)
            l_bounds = [b[0] for b in pbounds.values()]
            u_bounds = [b[1] for b in pbounds.values()]
            scaled_samples = qmc.scale(samples, l_bounds, u_bounds)
            # After sampling, log the sampled points
            print("Sampled points:", scaled_samples)

            for sample in scaled_samples.tolist():
                print("Sampled point:", sample)
                params = dict(zip(pbounds, sample))
                print("Probing point:", params)
                self.optimizer.probe(params=params, lazy=True)

            init_points = 0

        self.optimizer.maximize(
            init_points=init_points,
            n_iter=self.cfg.n_iters,
        )
        print("Final bounds after optimization:", self.optimizer.space.bounds)

    def postprocess(self) -> None:
        print("\nRecap!")
        for i, res in enumerate(self.optimizer.res):
            print(f"Iteration {i}: \n\t{res}")

        scores = parse_scores(self.optimizer.res)
        best_weights, best_bases = self.bounds_initialiser.assemble_params(
            self.optimizer.max["params"],
            self.merger.greek_letters,
            self.cfg.optimisation_guide.frozen_params
            if self.cfg.guided_optimisation
            else None,
            self.cfg.optimisation_guide.groups
            if self.cfg.guided_optimisation
            else None,
            sdxl=self.cfg.sdxl
        )

        self.plot_and_save(
            scores,
            best_bases,
            best_weights,
            minimise=False,
        )


def parse_scores(iterations: List[Dict]) -> List[float]:
    return [r["target"] for r in iterations]
