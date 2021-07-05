
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
import abc
import torch.nn as nn

class BaseLikelihood(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self):
        super(BaseLikelihood, self).__init__()
        # The basic likelihood doesn't have any parameter

    @abc.abstractmethod
    def log_cond_prob(self, output: torch.Tensor,
                      latent_val: torch.Tensor) -> torch.Tensor:
        """
        Subclass should implement log p(Y | F)
        :param output:  (batch_size x Dout) matrix containing true outputs
        :param latent_val: (MC x batch_size x Q) matrix of latent function values,
                            usually Q=F
        :return:
        """
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def predict(self, *args, **kargs):
        raise NotImplementedError("Subclass should implement this.")

    def forward(self, *input):
        return self.forward(input)