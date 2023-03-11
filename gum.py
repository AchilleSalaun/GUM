from toolbox import flatten, inv, log_gaussian, randn, tensor, transpose, unflatten, zeros

class GUM:
    def __init__(self, a, b, c, eta_, alpha_, beta_, requires_grad=False):
        self.m_a = a.requires_grad_(requires_grad)
        self.m_b = b.requires_grad_(requires_grad)
        self.m_c = c.requires_grad_(requires_grad)

        # flattened versions of the covariance matrices
        self.m_eta_   = eta_.requires_grad_(requires_grad)
        self.m_alpha_ = alpha_.requires_grad_(requires_grad)
        self.m_beta_  = beta_.requires_grad_(requires_grad)

    @property
    def a(self):
        return self.m_a

    @property
    def b(self):
        return self.m_b

    @property
    def c(self):
        return self.m_c

    @property
    def alpha_(self):
        return self.m_alpha_

    @property
    def alpha(self):
        return unflatten(self.alpha_)

    @property
    def beta_(self):
        return self.m_beta_

    @property
    def beta(self):
        return unflatten(self.beta_)

    @property
    def eta_(self):
        return self.m_eta_

    @property
    def eta(self):
        return unflatten(self.eta_)

    @property
    def theta(self):
        return self.a, self.b, self.c, self.eta_, self.alpha_, self.beta_

    def __str__(self):
        s = "Parameters:\n"

        s += "\t> a     : {0}\n".format(self.a)
        s += "\t> b     : {0}\n".format(self.b)
        s += "\t> c     : {0}\n".format(self.c)
        s += "\t> eta   : {0}\n".format(self.eta)
        s += "\t> alpha : {0}\n".format(self.alpha)
        s += "\t> beta  : {0}\n".format(self.beta)

        return s

    def predict(self, h_0=None, x_0=None):
        if h_0 is None and x_0 is None:
            zero = zeros(self.c.size())
            h_1 = randn(zero, self.eta)
        else:
            h_1 = randn(self.a @ h_0 + self.c @ x_0, self.alpha)

        x_1 = randn(self.b @ h_1, self.beta)

        return h_1, x_1

    def sample(self, n):
        h_0, x_0 = self.predict()

        h = [h_0]
        x = [x_0]

        for i in range(1, n):
            h_t, x_t = self.predict(h[-1], x[-1])

            h.append(h_t)
            x.append(x_t)

        return h, x

    def negative_log_likelihood(self, x):
        a     = self.a
        b     = self.b
        c     = self.c
        eta   = self.eta
        alpha = self.alpha
        beta  = self.beta

        # Initialisation
        mu_s = zeros((len(beta), 1))
        sigma_s = beta + b @ eta @ transpose(b)
        mu_ss = zeros((len(alpha), 1))
        sigma_ss = eta

        L = log_gaussian(x[0], mu_s, sigma_s)

        # Recursion
        for s in range(1, len(x)):
            # Eq. 86
            mu_s = b @ (a @ mu_ss + c @ x[s - 1])
            sigma_s = beta + b @ (alpha + a @ sigma_ss @ transpose(a)) @ transpose(b)

            # Eq. 87
            mu_s1 = a @ mu_ss + c @ x[s - 1]
            sigma_s1 = alpha + a @ sigma_ss @ transpose(a)

            # Eq. 88
            mu_ss = mu_s1 + sigma_s1 @ transpose(b) @ inv(sigma_s) @ (x[s] - mu_s)
            sigma_ss = sigma_s1 - sigma_s1 @ transpose(b) @ inv(sigma_s) @ b @ sigma_s1

            L += log_gaussian(x[s], mu_s, sigma_s)

        return -L / len(x)

class HMM(GUM):
    def __init__(self, a, b, eta_, alpha_, beta_, requires_grad=False):
        super().__init__(a, b, tensor([]), eta_, alpha_, beta_, requires_grad=requires_grad)
        self.m_c = zeros((len(alpha_), 1))

    @property
    def theta(self):
        return self.a, self.b, self.eta_, self.alpha_, self.beta_

class DGUM(GUM):
    def __init__(self, a, b, c, eta_, beta_, requires_grad=False):
        super().__init__(a, b, c, eta_, tensor([]), beta_, requires_grad=requires_grad)
        self.m_alpha_ = zeros(eta_.size())

    @property
    def theta(self):
        return self.a, self.b, self.c, self.eta_, self.beta_

class RNN(DGUM):
    def __init__(self, a, b, c, beta_, requires_grad=False):
        super().__init__(a, b, c, tensor([]), beta_, requires_grad=requires_grad)

    @property
    def eta(self):
        return self.c @ transpose(self.c)

    @property
    def eta_(self):
        return flatten(self.eta)

    @property
    def theta(self):
        return self.a, self.b, self.c, self.beta_

    def __str__(self):
        s = super().__str__()
        s += " ; Note: cc^T={0} (=eta)\n".format(self.c @ transpose(self.c))
        return s
