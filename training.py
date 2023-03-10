from toolbox import detach

class DefaultTrainingVisitor:
    def __init__(self):
        pass

    def start(self, gum, n_epochs, size_batch):
        pass

    def update(self, epoch, i, gum, loss):
        pass

    def schedule(self, scheduler):
        pass

    def end(self, gum, unknown_gum, n_observations):
        pass


class AggregateTrainingVisitor(DefaultTrainingVisitor):
    def __init__(self, visitors):
        super().__init__()
        self.m_visitors = visitors

    def start(self, gum, n_epochs, size_batch):
        for visitor in self.m_visitors:
            visitor.start(gum, n_epochs, size_batch)

    def update(self, epoch, i, gum, loss):
        for visitor in self.m_visitors:
            visitor.update(epoch, i, gum, loss)

    def schedule(self, scheduler):
        for visitor in self.m_visitors:
            visitor.schedule(scheduler)

    def end(self, gum, unknown_gum, n_observations):
        for visitor in self.m_visitors:
            visitor.end(gum, unknown_gum, n_observations)


class TalkativeTrainingVisitor(DefaultTrainingVisitor):
    def __init__(self):
        super().__init__()
        self.m_n_epochs   = None
        self.m_size_batch = None

    def start(self, gum, n_epochs, size_batch):
        self.m_n_epochs   = n_epochs
        self.m_size_batch = size_batch

    def update(self, epoch, i, gum, loss):
        print("> Iteration {0}/{1}".format(
            epoch * self.m_size_batch + i + 1,
            self.m_n_epochs * self.m_size_batch
        ))
        print(gum)
        print("Loss: {0}\n".format(detach(loss)))

    def schedule(self, scheduler):
        print("--- Update learning rate: {0} ---\n".format(scheduler.get_last_lr()))


class LossTrainingVisitor(DefaultTrainingVisitor):
    def __init__(self):
        super().__init__()
        self.m_losses = []

    @property
    def losses(self):
        return self.m_losses

    def update(self, epoch, i, gum, loss):
        self.m_losses.append(detach(loss))

    def end(self, gum, unknown_gum, n_observations):
        _, x = unknown_gum.sample(n_observations)
        self.m_losses.append(detach(gum.negative_log_likelihood(x)))

class ABTrainingVisitor(DefaultTrainingVisitor):
    def __init__(self):
        super().__init__()
        self.m_As   = []
        self.m_Bs   = []

    @property
    def As(self):
        return self.m_As

    @property
    def Bs(self):
        return self.m_Bs

    def start(self, gum, n_epochs, size_batch):
        self.m_As.append(detach(gum.A))
        self.m_Bs.append(detach(gum.B))

    def update(self, epoch, i, gum, loss):
        self.m_As.append(detach(gum.A))
        self.m_Bs.append(detach(gum.B))


def parameters_estimation(
        gum,
        unknown_gum,
        n_observations,
        n_epochs,
        size_batch,
        optimizer,
        scheduler,
        visitor=DefaultTrainingVisitor()
):
    visitor.start(gum, n_epochs, size_batch)

    for epoch in range(n_epochs):

        for i in range(size_batch):
            # Generate sample from the unknown generator
            h, x = unknown_gum.sample(n_observations)

            optimizer.zero_grad()

            # Compute gradients (w.r.t. our model)
            loss = gum.negative_log_likelihood(x)
            loss.backward()

            # Update model's parameters
            optimizer.step()

            visitor.update(epoch, i, gum, loss)

        # Update learning rate
        scheduler.step()
        visitor.schedule(scheduler)

    visitor.end(gum, unknown_gum, n_observations)
    return gum
