import numpy as np

from scipy.special import binom
from scipy.special import factorial

from scipy.stats import rv_discrete

class Fixed_Measure():
    """
    This is a dummy measure to be used when we do not sample but provide a fixed distribution.
    """

    def __init__(self, h01, distribution, has_constant):
        super(Fixed_Measure, self).__init__()
        self.h01 = h01
        # array of dimensions EXCLUDING bias term
        self.distribution = distribution
        self.has_constant = has_constant
        self.max_val = len(distribution)
        if h01:
            self.max_val += 1

    def rvs(self, size):
        offset = 1
        if self.h01:
            offset += 1

        samples = [x * [i+offset] for i, x in enumerate(self.distribution)]
        samples = [item for sublist in samples for item in sublist]

        return np.array(samples).astype(np.int)

class Exponential_Measure(rv_discrete):
    """
    Maclaurin Sampling distribution for exponential kernel:
    exp(x.T y)
    """

    def __new__(cls, *args, **kwargs):
        return rv_discrete.__new__(cls, kwargs)

    def __init__(self, h01, **kwargs):
        self.h01 = h01
        self.has_constant = True
        super(Exponential_Measure, self).__init__(**kwargs)

    def _pmf(self, k):
        pmf_vals = self.coefs(k)
        norm_const = np.exp(1.)

        if self.has_constant:
            pmf_vals[k==0] = 0
            norm_const = norm_const - 1.

        if self.h01:
            pmf_vals[k==1] = 0
            norm_const = norm_const - 1.

        return pmf_vals / norm_const

    @staticmethod
    def coefs(k):
        return 1. / factorial(k)

class Polynomial_Measure(rv_discrete):
    """
    Maclaurin Sampling distribution for polynomial kernel.
    """

    def __new__(cls, *args, **kwargs):
        # __new__ is called before __init__
        return rv_discrete.__new__(cls, kwargs)

    def __init__(self, p, c, h01, **kwargs):
        """
        p: degree
        c: bias
        h01: Whether to set p(0)=p(1)=0
        """
        if p<0 or c<0:
            raise RuntimeError('We need p>=0 and c>=0!')
        if h01 and (c==0 or p<2):
            raise RuntimeError('H01 only works for p>=2 and c>0!')

        self.p = p
        self.c = float(c)
        self.h01 = h01
        self.has_constant = (c > 0)
        super(Polynomial_Measure, self).__init__(**kwargs)

    def _pmf(self, k):
        norm_const = (self.c+1)**self.p

        # automatically returns 0 if not 0<=k<=p
        pmf_vals = self.coefs(k, self.p, self.c)

        if self.has_constant:
            norm_const = norm_const - self.c**self.p
            pmf_vals[k==0] = 0

        if self.h01:
            norm_const = norm_const - self.p*self.c**(self.p-1)
            pmf_vals[k==1] = 0

        return pmf_vals / norm_const

    @staticmethod
    def coefs(k, p, c):
        if isinstance(k, np.ndarray):
            coefs = np.zeros_like(k).astype('float64')
            coefs[k<=p] = binom(p, k[k<=p]) * c**(p-k[k<=p])
        else:
            coefs = binom(p, k) * c**(p-k) if k <= p else 0
        return coefs

class P_Measure(rv_discrete):
    """
    The "external measure" proposed in Kar & Karnick 2012.
    """

    def __new__(cls, *args, **kwargs):
        # __new__ is called before __init__
        return rv_discrete.__new__(cls, kwargs)

    def __init__(self, p, h01, max_val, **kwargs):
        """
        p: Parameter for sampling distribution 1./p**(k+1)
        We need p>1, p=2 leads to normalized pmf.
        h01: Whether to set p(0)=p(1)=0
        """
        if p <= 1:
            raise RuntimeError('p needs to be greater than 1!')

        self.p = p
        self.h01 = h01
        self.max_val = max_val
        self.has_constant = True
        super(P_Measure, self).__init__(**kwargs)
        
    def _pmf(self, k):
        if self.max_val == np.inf:
            norm_const = 1./(self.p-1.)
        else:
            norm_const = np.sum(np.array(
                [1./self.p**(n+1.) for n in range(self.max_val+1)]
            ))
        
        pmf_vals = 1./(self.p**(k+1.))
        pmf_vals[k > self.max_val] = 0

        # there is no point in wasting features on the constant
        pmf_vals[k==0] = 0
        norm_const = norm_const - 1./self.p
        
        if self.h01:
            norm_const = norm_const - 1./(self.p**2)
            pmf_vals[k==1] = 0

        return pmf_vals / norm_const