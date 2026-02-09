"""Muitl-objective Bayesian optimization for optimizing PCSELs (using OpenAI Gym).
#Renjie Li, Dec 2025
"""
#2025.5.15: this version increases N_batch and beta in UCB.
#2025.7.6: this version moved Q and lam from objectives to constraints
#2025.9.12: this version adds a upper constraint for power
#2025.11.8: this version uses additive Gaussian RBF kernel 

#import sys
import gym
from gym.envs.registration import register
#import math
#import random
import numpy as np
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
#from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
#from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.priors.torch_priors import GammaPrior
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement as qNEHVI
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
#import matplotlib
#import matplotlib.pyplot as plt
import torch
#import torch.nn as nn
#import torch.optim as optim 
#import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
#import torchvision.transforms as T
import logging
from itertools import count
from datetime import datetime, timezone
from PyQt5.QtCore import QProcess

torch.set_printoptions(precision=10)

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# register the env with gym
register(
    id='Fdtd_NB-v0',
    entry_point='envs:FdtdEnv',
    max_episode_steps=150,
    reward_threshold=250.0,
)

writer = SummaryWriter()  # log the training process

# instantiate the fdtd env
env = gym.make('Fdtd_NB-v0').unwrapped

state = env.reset()


def generate_initial_data():
    # generate training data
    #input variable: state
    #train_X = torch.zeros(1, 8, device=device, dtype=torch.float64)  #fixed zero initialization
    
    # x1 = -200*torch.rand(1, 1, device=device, dtype=torch.float64) + 100   #random initialization
    # x2 = -0.3*torch.rand(1, 1, device=device, dtype=torch.float64) + 0.15
    # x3 = -0.6*torch.rand(1, 1, device=device, dtype=torch.float64) + 0.3
    # x4 = -2000*torch.rand(1, 1, device=device, dtype=torch.float64) + 1000
    # train_X = torch.hstack((x4, x1, x1, x1, x2, x2, x3, x1))
    
    #pre-defined x inputs
    train_X = torch.tensor([[509.629416717757, -44.34259004220376, 29.745338344535647, -89.50799516144542, 0.031145917518602545, 0.08155523610525323, 
                            0.17836821962097613, 55.67268093244802], 
                            [-818.2751157235862, -2.524478436450169, -4.409965972741907, -88.68226383082023, 0.041451476103349975, 
                            0.093439809304917, 0.23143725787110003, 27.566375078796028],
                            [128.6916347095833, -79.44291452192296, -46.30132503356166, 49.20402400737032, -0.14993586781222507, 
                            0.006494112252437392, 0.19799047637149653, -54.949572999498834],
                            [-1000.0, 75.02319392506163, 132.42626834447768, 198.99793604560793, -0.15, -0.15, -0.3, -100.0],
                            [213.26807886306602, 78.26355993113582, 161.6658668772068, -49.46291819177854, -0.09531999216645468, 
                             -0.05917190127899333, -0.18046501271303797, -74.2740660890565]], device=device, dtype=torch.float64)  
    
    #pre-defined y values
    obj = torch.tensor([[0.15109595472962178, 0.9480188453208492, 0.278330487635008, 0.5123229764525776, 0.8384717088950511], 
                        [0.06294926708063109, 0.9436142559493659, 0.2533773212951561, 0.5052365679530397, 0.8384691867547098], 
                        [0.020621634380936782, 0.9589971764858201, 0.288685813314407, 0.4117515467510151, 0.8384724241710111], 
                        [0.010857210155417518, 0.9781537418931955, 0.24129178206868274, 2.018895668364936, 1.4241604982873188], 
                        [0.020743736627467202, 0.9734937260668722, 0.23630675073491858, 2.8411610287488633, 0.8483708140344541]], device=device, dtype=torch.float64)
    
    train_Y = obj[:, 2:5]

    #constraints
    con1 = obj[:, 0].unsqueeze(1) - 0        # Q factor > 0
    con2 = obj[:, 1].unsqueeze(1) - 0.923664    #abs(self.lam_goal - lam) < 100 nm
    con3 = 1.98 - obj[:, 3].unsqueeze(1)     #power < 0.99, conservation of energy.
    train_con = -1*torch.cat([con1, con2, con3], 1)
    
    #use simulator to generate y values
    # train_Y = torch.zeros(5,5, device=device, dtype=torch.float64)
    # for i in range(0, 5):
    #     obj, _ = env.step(train_X[i, :].tolist())
    #     # output variable: score
    #     train_Y[i, :] = torch.tensor(obj, device=device)

    return train_X, train_Y, train_con

def initialize_model(train_X, train_Y, train_con):
    # define models for objective and constraint

    d = 8  #No. of input parameters
    m = 5  #No. of objectives
    theta = 3   #lengthscale
    #define the matern kernel with custom lengthscale. Default lengthscale = 1.
    # covariance = ScaleKernel(
    #     base_kernel=MaternKernel(
    #     nu=2.5,
    #     ard_num_dims=d,
    #     #lengthscale_prior=GammaPrior(6.0, 9.0),
    #     )
    # )

    #define an additive GP with RBF kernel
    add_kernel  = RBFKernel(active_dims=torch.tensor([0])) + \
        RBFKernel(active_dims=torch.tensor([1]))+  \
        RBFKernel(active_dims=torch.tensor([2]))+ \
        RBFKernel(active_dims=torch.tensor([3])) + \
        RBFKernel(active_dims=torch.tensor([4]))+  \
        RBFKernel(active_dims=torch.tensor([5]))+ \
        RBFKernel(active_dims=torch.tensor([6]))+  \
        RBFKernel(active_dims=torch.tensor([7]))
                  
    #use single task GP model for now; switch to multi task/model list GP later.
    # gp = SingleTaskGP(train_X, train_Y, covar_module= covariance,
    #                   outcome_transform=Standardize(m=m), input_transform=Normalize(d=d))
    # mll = ExactMarginalLogLikelihood(gp.likelihood, gp)  #marginal likelihood

    #use model list for improved performance
    train_y = torch.cat([train_Y, train_con], dim=-1)
    models = []
    for i in range(train_y.shape[-1]):
        models.append(
            SingleTaskGP(train_X, train_y[..., i : i + 1], covar_module= add_kernel,
                      outcome_transform=Standardize(m=1), input_transform=Normalize(d=d)
            )
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    
    return mll, model


def optimize_acqf_and_get_observation(model, train_x):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""

    #bounds for the x input
    high = [1000.0, 200., 200., 200., 0.15, 0.15, 0.3, 100.]
    low = [-1000.0, -200., -200., -200., -0.15, -0.15, -0.3, -100.]
    bounds = torch.tensor([low, high], dtype=torch.float64, device=device)

    train_x = normalize(train_x, bounds)

    #acquisition function (qNEHVI for multi-objective)
    acq_func = qNEHVI(
            model=model,
            ref_point= torch.zeros(3, dtype=torch.float64, device=device),  #lower bound for objectives
            X_baseline=train_x,
            objective = IdentityMCMultiOutputObjective(outcomes = [0,1,2]),
            constraints=[lambda Z: Z[..., -3], lambda Z: Z[..., -2], lambda Z: Z[..., -1]],
            prune_baseline=True,
            alpha=0.0,
    )

    #optimize acq. Higher raw_samples is better but more expensive.
    candidate, acq_value = optimize_acqf(
        acq_function = acq_func, bounds=bounds, q=1, num_restarts=5, raw_samples=128,
    )

    # observe new values
    new_x = candidate.detach()
    obj, score = env.step(new_x.tolist()[0])
    obj = torch.tensor(obj, device=device).unsqueeze(0)
    new_y = obj[:, 2:5]
    #constraints
    con1 = obj[:, 0].unsqueeze(1) - 0   # Q factor > 0
    con2 = obj[:, 1].unsqueeze(1) - 0.923664    #abs(self.lam_goal - lam) < 100 nm
    con3 = 1.98 - obj[:, 3].unsqueeze(1)     #power < 0.99, conservation of energy.
    new_con = -1*torch.cat([con1, con2, con3], 1)

    return new_x, new_y, new_con, acq_value, score


N_TRIALS = 60
N_BATCH = 300

verbose = True

#save overall best solutions
best_observed_all_ei = []

#save data
input = torch.tensor([], device=device)
objective = torch.tensor([], device=device)

steps_done = 0

print('\nRunning MOBO for PCSEL with 3 obj and 3 cons...')

# average over multiple trials
for trial in range(1, N_TRIALS + 1):
    utc_dt = datetime.now(timezone.utc)

    print(f"\nTrial {trial:>2} of {N_TRIALS} \n", end="")

    print("\nLocal time {}".format(utc_dt.astimezone().isoformat()))
    
    best_observed_ei = []

    #define the hypervolume
    hv = Hypervolume(ref_point=torch.zeros(3, dtype=torch.float64, device=device))
    hvs_qnehvi = []

    # call helper functions to generate initial training data and initialize model
    (
        train_x_ei,
        train_obj_ei,
        train_con
    ) = generate_initial_data()

    #print('\nx_initial: {}, f_initial: {:.5f}\n'.format(train_x_ei.tolist()[0], train_obj_ei.tolist()[0]))
    print('\nf_initial:')
    print(train_obj_ei.tolist())
    print('\nx_initial:')
    print(train_x_ei.tolist())

    mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei, train_con)

    # compute pareto front
    is_feas = (train_con < 0).all(dim=-1)
    feas_train_obj = train_obj_ei[is_feas]
    if feas_train_obj.shape[0] > 0:
        pareto_mask = is_non_dominated(feas_train_obj)
        pareto_y = feas_train_obj[pareto_mask]
        # compute hypervolume
        volume = hv.compute(pareto_y)
    else:
        volume = 0.0

    hvs_qnehvi.append(volume)

    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_BATCH + 1):
        print('\nStarting step No.{}'.format(iteration))

        steps_done += 1

        # fit the models
        fit_gpytorch_mll(mll_ei)

        # print('\n lengthscale')
        # print(model_ei.covar_module.base_kernel.lengthscale)

        # optimize and get new observation
        new_x_ei, new_obj_ei, new_con, acq, score = optimize_acqf_and_get_observation(model_ei, train_x_ei)

        writer.add_scalar('BO/f score', score, steps_done)
        new_obj = new_obj_ei.tolist()[0]
        con = new_con.tolist()[0]
        writer.add_scalars('BO/objectives', {'Q': -1*con[0],
                                             'lam': -1*con[1],
                                             'area': new_obj[0],
                                             'power': new_obj[1],
                                             'div': new_obj[2]}, steps_done)
        writer.add_scalar('BO/acq value', acq.item(), steps_done)

        print('f_obj: area, power, div; \t f_score:')
        print(new_obj_ei.tolist(), score)
        print('\ncon: Q, lam, power.')
        print((-new_con).tolist())
        print('\nx_candidate:')
        print(new_x_ei.tolist())
        print('\nacquisition:')
        print(acq.tolist())

        posterior = model_ei.posterior(new_x_ei)
        mean = posterior.mean  #posterior mean
        # Get upper and lower confidence bounds (2 std dev from the mean)
        lower, upper = posterior.mvn.confidence_region()

        print('\nPosterior mean and uncertainty:')
        print(mean.tolist(), upper.tolist())

        # update training points
        train_x_ei = torch.cat([train_x_ei, new_x_ei])
        train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])
        train_con = torch.cat([train_con, new_con])

        # update progress
        for hvs_list, train_obj, train_con in zip(
            (hvs_qnehvi,),
            (train_obj_ei,),
            (train_con,),
        ):
            # compute pareto front
            is_feas = (train_con < 0).all(dim=-1)
            feas_train_obj = train_obj[is_feas].unsqueeze(0)
            if feas_train_obj.shape[0] > 0:
                pareto_mask = is_non_dominated(feas_train_obj)
                pareto_y = feas_train_obj[pareto_mask]
                # compute feasible hypervolume
                volume = hv.compute(pareto_y)
            else:
                volume = 0.0
            hvs_list.append(volume)

        print('\nhypervolume:')
        print(volume)
        writer.add_scalar('BO/hypervolume', volume, steps_done)

        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        mll_ei, model_ei = initialize_model(
            train_x_ei,
            train_obj_ei,
            train_con
        )


        if iteration % 25 == 0:
            #save data
            input = torch.cat([input, train_x_ei])
            objective = torch.cat([objective, train_obj_ei])
            torch.save(input, 'input.pt')
            torch.save(objective, 'objective.pt')

        if verbose:
            print(
                f"Batch {iteration:>2}: Hypervolume (qNEHVI) = "
                f"({hvs_qnehvi[-1]:>4.5f}), ",
                end="\n",
            )
        else:
            print(".", end="\n")


writer.close()



