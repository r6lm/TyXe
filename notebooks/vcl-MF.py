#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# NOTE: currently I only perform validations loops at the end of each epoch if early stopping is enabled.
# NOTE: when transformed to python script magics and parse args 


# In[ ]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
import argparse

# parameters to tune on Eddie
parser = argparse.ArgumentParser()
parser.add_argument(
    "--init-scale", default="1e-2", help="guide factory initial parameter scale")
parser.add_argument(
    "--seed", default="6202", help="random seed for reproducibility")
# parser.add_argument("--inference", choices=["mean-field", "ml"], required=True)

# parsed_args = parser.parse_args(["--init-scale", "1e-3", "--seed", "3"])
# parsed_args = parser.parse_args([])
parsed_args = parser.parse_args()
parsed_args


# In[ ]:


validation_config = False
test_offline = False
test_online = False
plot_perf = False
fast_dev_run = False


# In[ ]:


import copy
import functools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc

from tqdm import tqdm
from collections import defaultdict
from time import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score

# import torchvision.transforms as tf
# from torchvision.datasets import MNIST, CIFAR10, CIFAR100

import pyro
import pyro.distributions as dist

import tyxe

from MF.model import get_model
from dataset.ASMGMovieLens import ASMGMovieLens
from utils.save import (get_version, save_as_json, append_json_array, 
    load_json_array)
from pytorchtools import EarlyStopping


# # Parameters

# In[ ]:


# tyxe global parameters
ROOT = os.environ.get("DATASETS_PATH", "./data")
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda") if USE_CUDA else torch.device("cpu")
C10_MEAN = (0.49139968, 0.48215841, 0.44653091)
C10_SD = (0.24703223, 0.24348513, 0.26158784)
inference = "mean-field"
DEVICE


# In[ ]:


# control flow parameters
train_params = dict(
    input_path="data/movielens/processed/ml_processed.csv",
    val_start_period=11,
    val_end_period= 24,  #12
    test_start_period=25, # change to None if running as validation
    test_end_period=31,  # 25,
    train_window=10,
    seed=int(parsed_args.seed),
    model_filename='first_mf',
    offline_path=None,  #"../model/MF/mean-field/version_29/offline_state_dict.pt",
    online_end_of_validation_path=None,  # "../model/MF/mean-field/version_14/online_state_dict.pt",
    # save_model=False, \todo
    save_result=True,
    save_preds = True,
)

model_params = dict(
    alias="MF",
    n_users=43183,
    n_items=51149,
    n_latents=8,
    l2_regularization_constant=1e-6,
    learning_rate=1e-3,  # 1e-2 is the ASMG MF implementation
    batch_size=1024,
    n_epochs_offline=30, # 11,
    n_epochs_online=40,  # 19,
    early_stopping_offline=True,
    early_stopping_online=True, # train_params["test_start_period"] is None,
    update_prior=True,
    random_init=False,
    test_samples=40,
    guide_init_scale=float(parsed_args.init_scale)
)


train_params["model_checkpoint_dir"] = f'./../model/{model_params["alias"]}'

if fast_dev_run:
    model_params["n_epochs_offline"] = 1
    model_params["n_epochs_online"] = 1
    model_params["save_result"] = False

if validation_config:
    train_params.update(dict(
        val_start_period=11,
        val_end_period=20,
        test_start_period=21, 
        test_end_period=24, 
        # offline_path="../model/MF/mean-field/version_29/offline_state_dict.pt",
        # online_end_of_validation_path="../model/MF/mean-field/version_29/online_state_dict.pt",
    ))
    # model_params.update(dict(
    #     n_epochs_online=18
    # ))


# adapt function to TyXe experiment 
get_version = functools.partial(get_version, logdir=inference)
experiment_params = {**train_params, **model_params}
params = argparse.Namespace(**experiment_params)
params.guide_init_scale, params.seed


# In[ ]:


# get version
version = get_version(train_params["model_checkpoint_dir"])

# make checkpoint dir
model_checkpoint_subdir = train_params["model_checkpoint_dir"] + (
    f'/{inference}/{version}')
    
if not os.path.exists(model_checkpoint_subdir):
    os.makedirs(model_checkpoint_subdir)
    
    # save json 
    json_path = f"{model_checkpoint_subdir}/params"
    save_as_json(experiment_params, json_path)


# # Custom functions

# In[ ]:


# script functions
def validation_loop(model, dataloader, model_samples):
    """Returns error and likelihood as cpu floats.

    Parameters
    ----------
    model : BNN
    dataloader 
    model_samples : int
        number of samples of model parameters for the evaluation.
    
    """    
    err, log_likelihood = torch.tensor([
        model.evaluate(x.to(DEVICE), y.to(DEVICE), 
        num_predictions=model_samples
            ) for x, y in dataloader]
        ).sum(dim = 0)
    mean_nll = - log_likelihood.item() / len(dataloader.sampler)
    mean_error =  err.item() / len(dataloader.sampler)

    return mean_nll, mean_error

def auc(bnn, params, y_true, test_loader, prediction_dst=None):
    """
    Uses scikit-learn roc_auc_score.

    Parameters
    ----------
    bnn 
    params 
    y_true : torch.Tensor
    test_loader : Dataloader
    prediction_dst : str or os.PathLike
        where predictions are saved.

    Returns
    -------
    float
    """    
    preds = torch.ones_like(y_true) * -1

    for i, (x, _) in enumerate(test_loader):
        preds[
            i * params.batch_size:min((i + 1) * params.batch_size, len(y_true))
        ] = bnn.predict(x.to(DEVICE), num_predictions=params.test_samples)

    assert torch.all(preds != -1), "Not all values replaced for predictions."

    if prediction_dst is not None:
        torch.save(preds, prediction_dst)

    return roc_auc_score(y_true, preds)


def fit_aux(train_loader, n_epochs, early_stopper, **kwargs):

    elbos = []
    postfix_dict = {"Epoch Loss": np.Inf}
    pbar = tqdm(total=n_epochs, 
    unit="Epochs", postfix=f"Epoch Loss: {postfix_dict['Epoch Loss']}")

    # non-enclosed function
    def reporting_callback(model, _ii, e):
        """
        Used at the end of each epoch on the TyXe `fit` method.
        
        model: BNN
        _ii: epoch number
        e: epoch loss
        """
        mean_loss = e / len(train_loader.sampler) / train_loader.batch_size
        elbos.append(mean_loss)
        pbar.update()
        postfix_dict['Epoch Loss'] = f"{mean_loss:.5f}"
        pbar.set_postfix(postfix_dict)

    if early_stopper is not None:
        test_errors = []
        test_nll = []

        def early_stopping_callback(model, _ii, e):
            """Calls `reporting_callback` and adds validation loop for 
            early-stopping.

            Parameters
            ----------
            model: BNN
            _ii: epoch number
            e: epoch loss
            """           

            reporting_callback(model, _ii, e)

            # include this snippet in callback
            mean_nll, mean_error =  validation_loop(
                model, kwargs["test_dataloader"], kwargs["test_samples"])
            test_errors.append(mean_error)
            test_nll.append(mean_nll)
            
            # update pbar with validation results
            postfix_dict["Val Loss"] = f"{mean_nll:.5f}"
            pbar.set_postfix(postfix_dict)

            early_stopper(mean_nll, model)

            if early_stopper.early_stop:
                print(f"early-stopped: {early_stopper.early_stop}")
                return True

        return (elbos, postfix_dict, pbar, early_stopping_callback, 
            test_errors, test_nll)
            
    else:
        return elbos,postfix_dict,pbar,reporting_callback

def get_bnn(model_params, inference, device=DEVICE, prior=None):
    """Builds a BNN.

    Parameters
    ----------
    model_params : dict
    inference : str
    device : str, optional
        by default DEVICE
    prior : tyxe.Prior, optional
        If not given, the IID. Gaussian(0,1) is used.

    Returns
    -------
    tyxe.VariationalBNN
    """    
    net = get_model(model_params).to(device)
    obs = tyxe.likelihoods.Bernoulli(-1, event_dim=1, logit_predictions=False)

    if inference == "mean-field":
        
        if prior is None:
            prior_ = tyxe.priors.IIDPrior(dist.Normal(torch.tensor(
                0., device=device), torch.tensor(1., device=device)),
                expose_all=True)
        else:
            prior_ = prior
        
        guide_factory = functools.partial(
            tyxe.guides.AutoNormal, init_scale=model_params["guide_init_scale"],
            init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(net))
    elif inference == "ml":
        prior_ = tyxe.priors.IIDPrior(dist.Normal(0, 1), expose_all=False, hide_all=True)
        guide_factory = None

    return tyxe.VariationalBNN(net, prior_, obs, guide_factory)


def load_bnn(model_params, inference, train_params, path):
    """Loads a BNN. Needs to call fit to instantiate a guide to then 
    load the weights.

    Parameters
    ----------
    model_params : dict   
    inference : str  
    train_params : dict  
    path : str or Path
        for the state dict.

    Returns
    -------
    tyxe.VariationalBNN
    """    
    
    # get bnn
    new_bnn = get_bnn(model_params, inference)

    # load data and create a single mini-batch dataloader
    train_data = ASMGMovieLens(train_params["input_path"], 1)
    single_mbatch_data = random_split(train_data, [model_params[
        "batch_size"], len(train_data) - params.batch_size])[0]
    singleton_loader = DataLoader(
        single_mbatch_data, batch_size=params.batch_size)
    new_bnn.likelihood.dataset_size = params.batch_size

    # define optimizer
    optim = pyro.optim.Adam({"lr": model_params["learning_rate"], 
    "weight_decay": model_params["l2_regularization_constant"]})

    # instance guide
    with tyxe.poutine.local_reparameterization():
        new_bnn.fit(singleton_loader, optim, 1, 
        device=DEVICE)
    
    # load BNN
    new_bnn.load_state_dict(torch.load(path))

    # prior is assumed to be the approximate posterior in case we
    # are using mean field inference and prior update
    if inference == "mean-field" and model_params["update_prior"]:
        new_bnn.update_prior(tyxe.priors.DictPrior(
            new_bnn.net_guide.get_detached_distributions(
            tyxe.util.pyro_sample_sites(new_bnn.net))))
    return new_bnn


def dataloader(
    params, start_period, end_period=None, fast_dev_run=False, shuffle=True, 
    return_y=False):
    """
    Returns a dataloader and allows for fast development runs (testing).

    Parameters
    ----------
    params : Namespace
    start_period : int
    end_period : int, by default None
    fast_dev_run : bool, optional
        by default False
    return_y: bool
        return y_true for prediction.

    """
    # restrict to just one minibatch for development testing
    dataset = ASMGMovieLens(
        params.input_path, start_period, end_period)
    y_true = dataset.y

    if fast_dev_run:

        if return_y:
            y_true = dataset.y[:params.batch_size]

        dataset = random_split(dataset, [
            params.batch_size, len(dataset) - params.batch_size])[0]

    dataloader_ = DataLoader(
        dataset, batch_size=params.batch_size, shuffle=shuffle,
        num_workers=os.cpu_count(), pin_memory=USE_CUDA)
    
    if return_y:
        return dataloader_, y_true
    else:
        return dataloader_


# # Offline training

# In[ ]:


# if there is no saved premodel
if train_params["offline_path"] is None:
    offline_checkpoint_path = f"{model_checkpoint_subdir}/offline_state_dict.pt"
    early_stopping = EarlyStopping(
        delta=1e-4, path=offline_checkpoint_path, trace_func=lambda x: None) if \
            model_params["early_stopping_offline"] else None
    bnn = get_bnn(model_params, inference)
    # ensure reproducibility
    torch.manual_seed(train_params["seed"])

    # define periods
    train_start_period = train_params["val_start_period"] - train_params["train_window"]
    train_end_period = train_params["val_start_period"] - 1
    val_period = train_params["val_start_period"]
    print(
        "OFFLINE TRAINING",
        f"train period: {train_start_period}-{train_end_period}", 
        f"validation period: {val_period}", sep="\n")

    # get dataloaders
    train_loader = dataloader(params, train_start_period, train_end_period,
        fast_dev_run=fast_dev_run)
    test_loader = dataloader(params, val_period, fast_dev_run=fast_dev_run)

    # initialize fit auxiliary variables
    n_epochs_offline = model_params["n_epochs_offline"] 

    if early_stopping is None:
        elbos,postfix_dict, pbar, callback = fit_aux(
            train_loader, n_epochs_offline, None)
    else:
        elbos,postfix_dict, pbar, callback, test_errors, test_nlls = fit_aux(
            train_loader, n_epochs_offline, early_stopping, 
            test_dataloader=test_loader, test_samples=model_params["test_samples"])

    bnn.likelihood.dataset_size = len(train_loader.sampler)

    optim = pyro.optim.Adam({"lr": model_params["learning_rate"], 
        "weight_decay": model_params["l2_regularization_constant"]})

    with tyxe.poutine.local_reparameterization():
        bnn.fit(train_loader, optim, n_epochs_offline, 
            device=DEVICE, callback=callback)


    # if early stopping is None get validation performance, otherwise it is
    # calculated on callback
    if early_stopping is None:
        mean_nll = validation_loop(bnn, test_loader, model_params["test_samples"])[0]
        postfix_dict["Val Loss"] = f"{mean_nll:.5f}"
        pbar.set_postfix(postfix_dict)

    pbar.close()

    # counter starts from 1
    print(f"finished after {pbar.last_print_n} epochs")

    # update prior
    if (inference == "mean-field") and params.update_prior:
        bnn.update_prior(tyxe.priors.DictPrior(bnn.net_guide.get_detached_distributions(
            tyxe.util.pyro_sample_sites(bnn.net))))


    if early_stopping is None:
        # save model 
        torch.save(bnn.state_dict(), offline_checkpoint_path)

    else:
        # load best model
        bnn.load_state_dict(torch.load(early_stopping.path))

else:
    offline_checkpoint_path = params.offline_path
    bnn = load_bnn(model_params, inference, train_params, params.offline_path)


# In[ ]:


if test_offline:
    # unit test for reproducibility of base model
    torch.manual_seed(train_params["seed"])
    _, test_log_likelihood = torch.tensor([
        bnn.evaluate(x.to(DEVICE), y.to(DEVICE), 
        num_predictions=model_params["test_samples"]
            ) for x, y in test_loader]
        ).sum(dim = 0)
    test_mean_nll = - test_log_likelihood.item() / len(test_loader.sampler)
    print(f"{test_mean_nll:.5f}")


# In[ ]:


if test_offline:

    # build new BNN
    new_bnn = load_bnn(model_params, inference, train_params, offline_checkpoint_path)

    torch.manual_seed(train_params["seed"]) # if not set breaks
    new_err, new_log_likelihood = torch.tensor([
        new_bnn.evaluate(x.to(DEVICE), y.to(DEVICE), 
        num_predictions=model_params["test_samples"]
            ) for x, y in test_loader]
        ).sum(dim = 0)
    new_mean_nll = - new_log_likelihood.item() / len(test_loader.sampler)
    print(f"{new_mean_nll:.5f}")
    assert new_mean_nll == test_mean_nll


# In[ ]:


if params.save_result and (params.offline_path is None):

    # save train performance statistics  
    train_perf_path = f"{model_checkpoint_subdir}/train_perf.csv"
    train_perf_dict = {"elbo": elbos}
    if params.early_stopping_offline and (params.offline_path is None):
        train_perf_dict.update({
            "test_error": test_errors,
            "test_nlls": test_nlls
        })

    train_perf_df = pd.DataFrame(train_perf_dict)
    train_perf_df.index.set_names("epoch", inplace=True)
    train_perf_df.to_csv(train_perf_path)
    print(f"saving train performance statistics at: {os.path.abspath(train_perf_path)}")


# In[ ]:


if plot_perf and (params.offline_path is None):
    
    # train losses
    plt.figure(figsize=(9, 6))
    plt.plot(elbos)
    plt.xlabel("Epoch")
    plt.ylabel("ELBO loss")
    plt.title("Raw ELBO loss")
    plt.show()

    if early_stopping is not None:
        
        # test errors
        plt.figure(figsize=(9, 6))
        plt.plot(test_errors)
        plt.xlabel("epoch")
        plt.ylabel("test_error")
        plt.show()

        # test losses
        plt.figure(figsize=(9, 6))
        plt.plot(test_nlls)
        plt.xlabel("epoch")
        plt.ylabel("NLL")

        print(test_nlls[-1])
        print(np.array(test_nlls).min(), np.array(test_nlls).argmin())


# # Online training

# In[ ]:


if params.online_end_of_validation_path is None:

    # ensure reproducibility
    torch.manual_seed(train_params["seed"])

    val_periods = range(
        # starts at `val_start_period + 1` because the first validation is 
        # used for offline training
        train_params["val_start_period"] + 1, train_params["val_end_period"] + 1)


    n_epochs_online = model_params["n_epochs_online"] 

    # initialize performance containers
    val_losses = []
    val_epochs = [] if params.early_stopping_online is not None else None
    val_dict = defaultdict(lambda: [])

    # same checkpoint path used along the online training
    online_checkpoint_path = f"{model_checkpoint_subdir}/online_state_dict.pt"

    for i, val_period in enumerate(val_periods, 1):

        # initialize variational distribution randomly
        if params.random_init:
            new_bnn = get_bnn(model_params,inference)

            # use last model approx. posterior as prior 
            if params.update_prior:
                new_bnn.update_prior(tyxe.priors.DictPrior(
                    bnn.net_guide.get_detached_distributions(
                        tyxe.util.pyro_sample_sites(bnn.net))))
                        
            bnn = new_bnn
            gc.collect()

        # find the good number of epochs
        early_stopping = EarlyStopping(
            delta=1e-4, path=online_checkpoint_path, trace_func=lambda x: None) if \
                model_params["early_stopping_online"] else None

        # update periods
        train_period = val_period - 1 
        print(
            f"train period: {train_period}", 
            f"test period: {val_period}", sep="\n")
        

        train_loader = dataloader(params, train_period,
            fast_dev_run=fast_dev_run)
        test_loader = dataloader(params, val_period, fast_dev_run=fast_dev_run, 
            shuffle=False)   

        # initialize fit auxiliary variables
        bnn.likelihood.dataset_size = len(train_loader.sampler)
        


        if early_stopping is None:
            elbos,postfix_dict, pbar, callback = fit_aux(
                train_loader, n_epochs_online, None)
        else:
            # find the optimal amount of epochs if early stopping is enabled
            elbos,postfix_dict, pbar, callback, val_errors, val_nlls = fit_aux(
                train_loader, n_epochs_online, early_stopping, 
                test_dataloader=test_loader, test_samples=model_params["test_samples"])

        bnn.likelihood.dataset_size = len(train_loader.sampler)

        optim = pyro.optim.Adam({"lr": model_params["learning_rate"], 
            "weight_decay": model_params["l2_regularization_constant"]})

        with tyxe.poutine.local_reparameterization():
            bnn.fit(train_loader, optim, n_epochs_online, 
                device=DEVICE, callback=callback)

        # if early stopping is None get validation performance, otherwise it is
        # calculated on callback
        if early_stopping is None:
            mean_nll = validation_loop(bnn, test_loader, model_params["test_samples"])[0]
            postfix_dict["Val Loss"] = f"{mean_nll:.5f}"
            pbar.set_postfix(postfix_dict)
            val_losses.append(mean_nll)
        
        else:
            val_losses.append(early_stopping.best_score)
            val_epochs.append(len(val_nlls) - (
                early_stopping.patience * early_stopping.early_stop))
            val_dict[f"{val_period}-test_err"] = val_errors
            val_dict[f"{val_period}-test_nll"] = val_nlls

        pbar.close()
        print(f"finished after {pbar.last_print_n} epochs")

        # val_losses.append(early_stopping.best_score)

        if (inference == "mean-field") and params.update_prior:
            bnn.update_prior(tyxe.priors.DictPrior(bnn.net_guide.get_detached_distributions(
                tyxe.util.pyro_sample_sites(bnn.net))))
        
        # if early stopping is used, fallback to best model for next validation 
        # period
        if early_stopping is not None:
            # load best model
            bnn.load_state_dict(torch.load(early_stopping.path))


    # at the end of all validation periods, save the latest model in case 
    # not done by early stopping
    if early_stopping is None:
        # save model 
        torch.save(bnn.state_dict(), online_checkpoint_path)
    else:
        # save validation results which are deemed interesting only if early
        # stopping is enabled
        val_dict_path = f"{model_checkpoint_subdir}/val_dict"
        save_as_json(val_dict, val_dict_path)

else: 
    bnn = load_bnn(model_params, inference, train_params,
        params.online_end_of_validation_path)


# In[ ]:


if test_online:

    new_bnn = load_bnn(model_params, inference, train_params, offline_checkpoint_path)

    # ensure reproducibility
    torch.manual_seed(train_params["seed"])

    val_periods = range(
    # starts at `val_start_period + 1` because the first validation is 
    # used for offline training
    train_params["val_start_period"] + 1, train_params["val_end_period"] + 1)


    n_epochs_online = model_params["n_epochs_online"] 

    # initialize performance containers
    val_losses = []
    val_dict = defaultdict(lambda: [])

    # same checkpoint path used along the online training
    online_checkpoint_path = f"{model_checkpoint_subdir}/test-reproducibility_online_state_dict.pt"

    for i, val_period in enumerate(val_periods, 1):

        # find the good number of epochs
        early_stopping = EarlyStopping(
            delta=1e-4, path=online_checkpoint_path, trace_func=lambda x: None) if \
                model_params["early_stopping_online"] else None

        # update periods
        train_period = val_period - 1 
        print(
            f"train period: {train_period}", 
            f"test period: {val_period}", sep="\n")
        
        train_loader = dataloader(params, train_period,
            fast_dev_run=fast_dev_run)
        test_loader = dataloader(params, val_period, fast_dev_run=fast_dev_run, 
            shuffle=False)   

        # initialize fit auxiliary variables
        new_bnn.likelihood.dataset_size = len(train_loader.sampler)


        if early_stopping is None:
            elbos,postfix_dict, pbar, callback = fit_aux(
                train_loader, n_epochs_online, None)
        else:
            elbos,postfix_dict, pbar, callback, val_errors, val_nlls = fit_aux(
                train_loader, n_epochs_online, early_stopping, 
                test_dataloader=test_loader, test_samples=model_params["test_samples"])

        new_bnn.likelihood.dataset_size = len(train_loader.sampler)

        optim = pyro.optim.Adam({"lr": model_params["learning_rate"], 
            "weight_decay": model_params["l2_regularization_constant"]})

        with tyxe.poutine.local_reparameterization():
            new_bnn.fit(train_loader, optim, n_epochs_online, 
                device=DEVICE, callback=callback)


        # if early stopping is None get validation performance, otherwise it is
        # calculated on callback
        if early_stopping is None:
            mean_nll = validation_loop(new_bnn, test_loader, model_params["test_samples"])[0]
            postfix_dict["Val Loss"] = f"{mean_nll:.5f}"
            pbar.set_postfix(postfix_dict)
            val_losses.append(mean_nll)
        
        else:
            val_losses.append(early_stopping.best_score)
            val_dict[f"{val_period}-test_err"] = val_errors
            val_dict[f"{val_period}-test_nll"] = val_nlls

        pbar.close()
        print(f"finished after {pbar.last_print_n} epochs")

        if (inference == "mean-field") and params.update_prior:
            new_bnn.update_prior(tyxe.priors.DictPrior(new_bnn.net_guide.get_detached_distributions(
                tyxe.util.pyro_sample_sites(new_bnn.net))))
        
        # if early stopping is used, fallback to best model for next validation 
        # period
        if early_stopping is not None:
            # load best model
            new_bnn.load_state_dict(torch.load(early_stopping.path))


    # at the end of all validation periods, save the latest model in case 
    # not done by early stopping
    if early_stopping is None:
        # save model 
        torch.save(new_bnn.state_dict(), online_checkpoint_path)


# # Online test

# In[ ]:


if params.early_stopping_online:
    # select the number of epochs for online test from validation
    if params.online_end_of_validation_path is None:
        n_epochs_online = int(pd.Series(val_epochs).median())
        print(f"optimal online epochs: {n_epochs_online}")
    
    # if validation model was provided, then use the original
    else:
        n_epochs_online = params.n_epochs_online


# In[ ]:



if train_params["test_start_period"] is not None:
    test_periods = range(
        train_params["test_start_period"], train_params["test_end_period"] + 1)
else:
    test_periods = []
    print("skipped test cycle")



test_dict = defaultdict(lambda: [])

for i, test_period in enumerate(test_periods, 1):

    # initialize variational distribution randomly
    if params.random_init:
        
        new_bnn = get_bnn(model_params,inference)

        # use last model approx. posterior as prior 
        if params.update_prior:
            new_bnn.update_prior(tyxe.priors.DictPrior(
                bnn.net_guide.get_detached_distributions(
                    tyxe.util.pyro_sample_sites(bnn.net))))
                    
        bnn = new_bnn
        gc.collect()


    # update periods
    train_period = test_period - 1 
    print(
        f"train period: {train_period}", 
        f"test period: {test_period}", sep="\n")

    train_loader = dataloader(params, train_period, 
        fast_dev_run=fast_dev_run)
    test_loader, y_true_test = dataloader(params, test_period, 
        fast_dev_run=fast_dev_run, shuffle=False, return_y=True)   

    elbos, postfix_dict, pbar, callback = fit_aux(
                train_loader, n_epochs_online, None)

    bnn.likelihood.dataset_size = len(train_loader.sampler)
    optim = pyro.optim.Adam({"lr": model_params["learning_rate"], 
    "weight_decay": model_params["l2_regularization_constant"]})


    # for epoch in range(n_epochs_online):
    start_time = time()
    with tyxe.poutine.local_reparameterization():
        bnn.fit(train_loader, optim, n_epochs_online, 
            device=DEVICE, callback=callback)
    trainig_time = time() - start_time

    # get validation loss
    mean_nll, mean_error = validation_loop(
        bnn, test_loader, model_params["test_samples"])
    postfix_dict["Test Loss"] = f"{mean_nll:.5f}"
    pbar.set_postfix(postfix_dict)

    # get test AUC and save predictions
    predictions_dir = f"{model_checkpoint_subdir}/T{test_period}"
    if params.save_preds and (not os.path.exists(predictions_dir)):
        os.makedirs(predictions_dir)
    predictions_path = f'{predictions_dir}/preds-s{params.seed}.pt'
    auc_test = auc(bnn, params, y_true_test, test_loader, 
        prediction_dst=(predictions_path if params.save_preds else None))
    postfix_dict["Test AUC"] = f"{auc_test:.5f}"
    pbar.set_postfix(postfix_dict)

    pbar.close()

    test_dict["period"].append(test_period)
    test_dict["loss"].append(mean_nll)
    test_dict["error"].append(mean_error)
    test_dict["auc"].append(auc_test)
    test_dict["train_time"].append(trainig_time)

    if (inference == "mean-field") and params.update_prior:
        bnn.update_prior(tyxe.priors.DictPrior(bnn.net_guide.get_detached_distributions(
            tyxe.util.pyro_sample_sites(bnn.net))))


# In[ ]:


df_path = f"{model_checkpoint_subdir}/first_biu.csv"
res_df = pd.DataFrame(test_dict)
average_srs = res_df.mean()
average_srs.at["period"] = "mean"

if train_params["save_result"]: 
    pd.concat((res_df, average_srs.to_frame().T), axis=0, ignore_index=True
    ).to_csv(df_path, index=False)
    print(f"saved results csv at: {os.path.abspath(df_path)}")


# In[ ]:


# save summary results at two levels above the checkpoint path
results_master_path = model_checkpoint_subdir[
    : model_checkpoint_subdir.rfind("/", 0, model_checkpoint_subdir.rfind("/"))
    ] + "/results.json"


# In[ ]:


# concatenate summary results and script args
res_dict = average_srs.to_dict()
res_dict.update(**vars(parsed_args), **{
    "n_epochs_on_test": n_epochs_online,
    "version": version
    })
print(res_dict)
append_json_array(res_dict, results_master_path)


# In[ ]:


load_json_array(results_master_path)

