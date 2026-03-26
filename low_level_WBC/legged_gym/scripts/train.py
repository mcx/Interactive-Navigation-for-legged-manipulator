# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
import sys
import importlib
import importlib.util
from datetime import datetime
import isaacgym

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

def _prepend_sys_path(path):
    real_path = os.path.realpath(path)
    existing = []
    for entry in sys.path:
        candidate = entry if entry else os.getcwd()
        existing.append(os.path.realpath(candidate))
    sys.path = [entry for i, entry in enumerate(sys.path) if existing[i] != real_path]
    sys.path.insert(0, real_path)

_prepend_sys_path(REPO_ROOT)

def _import_local_legged_gym(repo_root):
    expected_pkg_dir = os.path.join(repo_root, "legged_gym")
    expected_init = os.path.join(expected_pkg_dir, "__init__.py")
    module = importlib.import_module("legged_gym")
    module_path = os.path.realpath(getattr(module, "__file__", ""))
    if module_path.startswith(expected_pkg_dir + os.sep):
        return module

    stale_modules = [m for m in list(sys.modules.keys()) if m == "legged_gym" or m.startswith("legged_gym.")]
    for module_name in stale_modules:
        sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(
        "legged_gym",
        expected_init,
        submodule_search_locations=[expected_pkg_dir],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load local legged_gym spec from: {expected_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["legged_gym"] = module
    spec.loader.exec_module(module)

    module_path = os.path.realpath(getattr(module, "__file__", ""))
    if not module_path.startswith(expected_pkg_dir + os.sep):
        raise RuntimeError(
            f"Imported legged_gym from unexpected path: {module_path}. "
            f"Expected under: {expected_pkg_dir}"
        )
    return module

_import_local_legged_gym(REPO_ROOT)

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import wandb

def train(args):
    proj_name = getattr(args, "proj_name", None) or getattr(args, "experiment_name", "default_project")
    exptid = getattr(args, "exptid", None) or getattr(args, "run_name", "default_run")
    debug = bool(getattr(args, "debug", False))

    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(proj_name) + exptid
    try:
        os.makedirs(log_pth)
    except:
        pass
    if debug:
        mode = "disabled"
        args.rows = 6
        args.cols = 2
        args.num_envs = 4096
    else:
        mode = "online"
    wandb.init(project=proj_name, name=exptid, mode=mode, dir=LEGGED_GYM_ENVS_DIR +"/logs")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/manip_loco/b2z1_config.py", policy="now")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/manip_loco/manip_loco.py", policy="now")

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
