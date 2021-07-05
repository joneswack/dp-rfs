
#  Copyright (c) 2019
#
#  This program is free software: you can redistribute it and/or modify it
#  under the terms of the GNU General Public License as published by the
#  Free Software Foundation, either version 3 of the License, or (at your
#  option) any later version.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#  Authors:
#      Simone Rossi <simone.rossi@eurecom.fr>
#      Maurizio Filippone <maurizio.filippone@eurecom.fr>
#
#  Original code and updates by:
#      Karl Krauth, Kurt Cutajar, Edwin V. Bonilla, Pietro Michiardi

import numpy as np
import torch
import torch.nn as nn

from . import BaseLikelihood

import logging
logger = logging.getLogger(__name__)


class Gaussian(BaseLikelihood):

    def __init__(self):
        super(Gaussian, self).__init__()
        self.log_noise_var = nn.Parameter(torch.ones(1) * -1.5, requires_grad=True)
        return

    def log_cond_prob(self, output: torch.Tensor,
                      latent_val: torch.Tensor) -> torch.Tensor:
        result = - 0.5 * (self.log_noise_var + np.log(2.*np.pi) +
                        torch.pow(output - latent_val, 2) * torch.exp(-self.log_noise_var))

        return result

    def get_params(self):
        return self.log_theta_noise_var

    def predict(self, latent_val: torch.Tensor, percentile: float = None):
        mean = torch.mean(latent_val, dim=0).cpu().detach().numpy()  # type: np.ndarray
        var = torch.var(latent_val, dim=0).cpu().detach().numpy() #+ self.log_noise_var.exp().detach().numpy()  # type: np.ndarray
        if percentile is None:
            return mean
        pred_lower = mean - 2 * np.sqrt(var) #np.percentile(latent_val.detach(), 100. - percentile * 100, axis=0)  # type: np.ndarray
        pred_upper = mean + 2 * np.sqrt(var) #np.percentile(latent_val.detach(), percentile * 100, axis=0)  # type: np.ndarray
        return mean, pred_lower, pred_upper