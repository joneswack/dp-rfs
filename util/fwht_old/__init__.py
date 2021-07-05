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

import torch
import warnings
import os

from torch.utils.cpp_extension import load
filedir = os.path.dirname(os.path.realpath(__file__)) + '/'

warnings.warn('Including and compiling a custom C++ and CUDA (if available) extension might take a while...', )

if os.environ.get('CXX', '-1') == '-1':
    warnings.warn('CXX variable not set. Setting CXX=g++...',)
    os.environ['CXX'] = 'g++'

dotlocalbin = os.environ['HOME'] + '/.local/bin'
if not(dotlocalbin in os.environ['PATH'].split(':')):
    warnings.warn('PATH variable does not include ~/.local/bin. Updating PATH=$HOME/.local/bin:$PATH')
    os.environ['PATH'] += (':%s' % dotlocalbin)

sources = [filedir + 'fwht_host.cc']
flags = ['-O3']
if torch.cuda.is_available():
    sources.extend([filedir + 'fwht_kernel.cu'])
    flags.extend(['-DIS_CUDA_AVAILABLE'])
    if os.environ.get('CUDA_HOME', '-1') == '-1':
        warnings.warn('CUDA_HOME variable not set. Setting CUDA_HOME=/usr/local/cuda-9.0...',)
        os.environ['CUDA_HOME'] = '/usr/local/cuda-9.0'
 
fwht_py = load(name='fwht_py', sources=sources, verbose=False, extra_cflags=flags)

from fwht_py import forward

from .fwht import FastWalshHadamardTransform
