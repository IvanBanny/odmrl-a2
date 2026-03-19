"""Shared RL utilities across Tasks 1-4."""

import numpy as np
import matplotlib.pyplot as plt
from mdp import (
    gamma, DEPOT, AT_1, AT_2, REP_1, REP_2, xi1, xi2,
    states, state_index, n_states,
    ACTIONS, action_index, n_actions, feasible_actions,
    cost, transitions, simulate_step,
)


def compute_reference_policy(tol=1e-10):
    """Run A1 policy iteration. Returns (policy, V) dicts."""
    raise NotImplementedError


def epsilon_greedy(Q, state, epsilon):
    """Epsilon-greedy over feasible actions. Returns action string.

    Q: array (n_states, n_actions). Minimization.
    """
    raise NotImplementedError


def extract_policy_tabular(Q):
    """Greedy policy from Q-table. Returns dict state -> action string."""
    raise NotImplementedError


def print_policy_table(policy, location, name):
    """Print policy table for a given engineer location."""
    raise NotImplementedError


def print_all_policy_tables(policy):
    """Print policy tables for DEPOT, AT_1, AT_2."""
    raise NotImplementedError


def compare_policies(policy, ref_policy, label=""):
    """Compare policy to reference. Returns number of differing states."""
    raise NotImplementedError


def plot_convergence(values, ylabel, title, xlabel="Episode", savefig=None):
    """Plot a convergence curve."""
    raise NotImplementedError
