import numpy as np

from toolbox import flatten, rand, tensor, zeros
from gum import GUM

class UnidimensionalGUM(GUM):
    def __init__(self, a, b, c, eta_, requires_grad=False):
        super().__init__(a, b, c, eta_, tensor([]), tensor([]), requires_grad=requires_grad)

    @property
    def alpha(self):
        return (1 - self.a**2 - 2 * self.a * self.b * self.c) * self.eta - self.c**2

    @property
    def alpha_(self):
        return flatten(self.alpha)

    @property
    def beta(self):
        return 1 - self.b**2 * self.eta

    @property
    def beta_(self):
        return flatten(self.beta)

    @property
    def A(self):
        return (self.a + self.b * self.c).detach()

    @property
    def B(self):
        return (self.b * (self.a  * self.b * self.eta + self.c)).detach()

    @property
    def theta(self):
        return self.a, self.b, self.c, self.eta_

    def __str__(self):
        s = super().__str__()
        s += "A={0}, B={1}".format(self.A, self.B)
        return s


class UnidimensionalHMM(UnidimensionalGUM):
    def __init__(self, a, b, eta_, requires_grad=False):
        super().__init__(a, b, tensor([]), eta_, requires_grad=requires_grad)
        self.m_c = zeros((1, 1))

    @property
    def theta(self):
        return self.a, self.b, self.eta_

class UnidimensionalRNN(UnidimensionalGUM):
    def __init__(self, a, b, c, requires_grad=False):
        super().__init__(tensor([]), tensor([]), tensor([]), tensor([]), requires_grad=requires_grad)
        if c == 0:
            self.m_a = a.requires_grad_(requires_grad)
            self.m_b = b.requires_grad_(requires_grad)
            self.m_c = zeros((1, 1))
        elif a == 0:
            self.m_a = zeros((1, 1))
            self.m_b = b.requires_grad_(requires_grad)
            self.m_c = c.requires_grad_(requires_grad)
        else:
            self.m_a = None
            self.m_b = b.requires_grad_(requires_grad)
            self.m_c = c.requires_grad_(requires_grad)

            if a != -2*b*c:
                print("WARNING: Parameters a={0}, b={1}, and c={2} violate constraint (*).".format(a, b, c))
                print("WARNING: Parameter a is now bounded to -2*b*c={0}.\n".format(-2*self.b*self.c))

    @property
    def a(self):
        if self.m_a is None:
            return -2*self.b*self.c
        else:
            return self.m_a

    @property
    def eta(self):
        return self.c**2

    @property
    def eta_(self):
        return flatten(self.eta)

    @property
    def theta(self):
        if self.c == 0:
            return self.a, self.b
        else:
            return self.b, self.c

    def __str__(self):
        s = super().__str__()
        s += " ; Note: c**2={0} (=eta)\n".format(self.c**2)
        return s

def sample_AB(model=None):
    A = rand(-1, 1)
    if   model == "hmm":
        B = np.sign(A) * rand(0, np.abs(A))
    elif model == "rnn":
        choice = np.random.choice(3)
        if   choice == 0:
            B = 0
        elif choice == 1:
            B = A
        else:
            B = A * (2 * A**2 - 1)
    elif model == "hmm+rnn":
        # Let's jump directly to the interesting case
        A = rand(1 / np.sqrt(2), 1)
        B = A * (2 * A ** 2 - 1)
        # if np.abs(A) <= 1 / np.sqrt(2):
        #     choice = np.random.choice(2)
        #     if choice == 0:
        #         B = 0
        #     else:
        #         B = A
        # else:
        #     choice = np.random.choice(3)
        #     if choice == 0:
        #         B = 0
        #     elif choice == 1:
        #         B = A
        #     else:
        #         B = A * (2 * A ** 2 - 1)
    else:
        B = rand((A - 1) / 2, (A + 1) / 2)
    return A, B

def get_GUM_from_AB(A, B, eta=None, requires_grad=False, submodel=None):
    eta_ = np.abs(eta) if eta is not None else .5

    if submodel == "hmm":
        assert 0 <= np.sign(A) * B
        assert np.sign(A) * B <= np.abs(A)

        if B == 0:
            if np.random.choice(2) == 0:
                b = 0
                a = rand(-1, 1)
            else:
                a = 0
                b = 1 / np.sqrt(eta_)
                b = rand(-b, b)
        else:
            if A == B:
                a = A
                b = -1**np.random.choice(2) / np.sqrt(eta_)
            else:
                b = -1**np.random.choice(2) * np.sqrt(B / (A * eta_))
                a = (A - B) / (1 - b**2 * eta_)
        return UnidimensionalHMM(tensor([[a]]), tensor([[b]]), tensor([[eta_]]), requires_grad=requires_grad)
    elif submodel == "rnn":
        assert B == 0 or A == B or B == A * (2*A**2 - 1)

        c = -1**np.random.choice(2) * np.sqrt(eta_)

        if B == 0:
            if np.random.choice(2):
                b = 0
                a = 0
            else:
                b = 1 / np.sqrt(eta_)
                b = rand(-b, b)
                if np.random.choice(2):
                    a = 0
                else:
                    a = -2*b*c
        else:
            if A == B:
                if np.random.choice(2):
                    a = 0
                    b = A / c
                else:
                    a = A + -1**np.random.choice(2)
                    b = (A - a) / c
            else:
                b = -1**np.random.choice(2) * A / c
                a = (A - B) / (1 - b**2 * eta_)
        return UnidimensionalRNN(tensor([[a]]), tensor([[b]]), tensor([[c]]), requires_grad=requires_grad)
    else:
        if B == 0:
            a = rand(-1, 1)
            if np.random.choice(2) == 0:
                c = np.sqrt((1 - a ** 2) * eta_)
                c = rand(-c, c)
                b = 0
            else:
                b = 1 / np.sqrt(eta_)
                b = rand(-b, b)
                c = - a * b * eta_
        else:
            if A == B:
                branch = np.random.choice(2)
                if branch == 0:
                    b = 1 / np.sqrt(eta_)
                    b = -1 ** np.random.choice(2) * rand(np.abs(A) * b, b)
                    a = 0
                    c = A / b
                else:
                    a = np.random.exponential()
                    b = -1 ** np.random.choice(2) / np.sqrt(eta_)
                    c = (A - a) * eta_ * b
            else:
                x1 = np.sqrt(
                    max(0., 1 + 2 * A * B - A ** 2 - np.sqrt((A ** 2 - 1) * ((2 * B - A) ** 2 - 1))) / (2 * eta_)
                )
                x2 = np.sqrt(
                    max(0., 1 + 2 * A * B - A ** 2 + np.sqrt((A ** 2 - 1) * ((2 * B - A) ** 2 - 1))) / (2 * eta_)
                )
                b = -1 ** np.random.choice(2) * rand(x1, x2)
                a = (A - B) / (1 - b ** 2 * eta_)
                c = (B - A * b ** 2 * eta_) / (b * (1 - b ** 2 * eta_))

        return UnidimensionalGUM(
            a=tensor([[a]]),
            b=tensor([[b]]),
            c=tensor([[c]]),
            eta_=tensor([[eta_]]),
            requires_grad=requires_grad
        )
