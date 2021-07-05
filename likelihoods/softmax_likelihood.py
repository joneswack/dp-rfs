
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

import torch

from . import BaseLikelihood


class Softmax(BaseLikelihood):
    """
    Implements softmax (categorical) likelihood for multi-class classification.
    """

    def __init__(self):
        super(Softmax, self).__init__()

    def log_cond_prob(self, output: torch.Tensor,
                      latent_val: torch.Tensor) -> torch.Tensor:
        """
        log p(y_i|x_i), i.e. for each datapoint separately
        output: labels (BS x out_dim)
        latent_val: predictions (NMC x BS x out_dim)

        We assume p(x_1, x_2, ..., x_n) = p(x_1) * p(x_2) * ... * p(x_n)
        with p(x_i) = p_1^(x_i=1) * p_2^(x_i=2) * ... * p_k(x_i=k) <=> p(x_i) ~ Cat(k)

        log( p(x_i) ) = (x_i=1) * log(p_1) + ... + (x_i=k) * log(p_k)
        = sum_j t_j * latent_j - logsumexp(latent)
        """

        return torch.sum(output * latent_val, 2) - torch.logsumexp(latent_val, 2)
        # logits = torch.nn.functional.log_softmax(latent_val, dim=-1)
        # return (logits * output).sum(-1)


    def predict(self, latent_val):
        """
        This function simply returns the softmax probabilities.
        log p(y=j | x) = latent_j - logsumexp(latent)
        """
        logprob = latent_val - torch.unsqueeze(torch.logsumexp(latent_val, 2), 2)
        return logprob.exp()