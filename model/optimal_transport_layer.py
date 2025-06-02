import torch
import torch.nn as nn

class Optimal_Transport_Layer(nn.Module):
    def __init__(self, config):
        super(Optimal_Transport_Layer, self).__init__()
        self.iters = config['sinkhorn_iterations']
        self.feature_dim = config['feature_dim']
        self.matched_threshold = config['matched_threshold']
        self.epsilon = config.get('epsilon', 0.8)  # Entropy regularization
        self.tau = config.get('tau', 1.02)  # Unbalanced regularization
        self.bin_score = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.register_parameter('bin_score', self.bin_score)

    @property
    def loss(self):
        return self.matching_loss, self.hard_pair_loss

    def forward(self, mdesc0, mdesc1, match_gt=None, ignore=False):
        # Compute matching descriptor distance
        sim_matrix = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = sim_matrix / self.feature_dim ** .5

        # Run the optimal transport with regularization
        scores = log_optimal_transport_extended(
            scores, self.bin_score,
            iters=self.iters,
            epsilon=self.epsilon,
            tau=self.tau)

        # Get the matches with score above "match_threshold"
        max0 = scores[:, :-1, :-1].max(2)
        max1 = scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices

        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)

        valid0 = mutual0 & (mscores0 > self.matched_threshold)
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        scores = scores.squeeze(0).exp()

        if match_gt is not None:
            matched_mask = torch.zeros(scores.size()).long().to(scores)
            matched_mask[match_gt['a2b'][:, 0], match_gt['a2b'][:, 1]] = 1
            if not ignore: matched_mask[match_gt['un_a'], -1] = 1
            if not ignore: matched_mask[-1, match_gt['un_b']] = 1

            self.matching_loss = -torch.log(scores[matched_mask == 1])
            
            # Enhanced hard negative mining
            top2_mask = matched_mask[:-1, :-1]
            scores_ = scores[:-1, :-1] * (1 - top2_mask)
            hard_negatives = torch.cat([scores_.max(1)[0], scores_.max(0)[0]])
            self.hard_pair_loss = -(torch.log(1 - hard_negatives) * (hard_negatives > 0.2).float())

        return scores, indices0.squeeze(0), indices1.squeeze(0), mscores0.squeeze(0), mscores1.squeeze(0)

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int, epsilon: float, tau: float):
    """ Perform Sinkhorn Normalization in Log-space with regularization """
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    
    for _ in range(iters):
        # u-update with unbalanced regularization
        u_new = (log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)) / tau
        u = u + epsilon * (u_new - u)
        
        # v-update with unbalanced regularization
        v_new = (log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)) / tau
        v = v + epsilon * (v_new - v)

    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores, alpha, iters: int, epsilon: float = 0.1, tau: float = 1.0):
    """ Perform Differentiable Optimal Transport in Log-space with regularization """
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    # Add dustbin channels
    couplings = torch.cat([torch.cat([scores, bins0], -1),
                          torch.cat([bins1, alpha], -1)], 1)

    # Compute marginal vectors
    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    # Perform Sinkhorn iterations with regularization
    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters, epsilon, tau)
    Z = Z - norm  # Multiply probabilities by M+N
    
    return Z

def log_optimal_transport_extended(scores, alpha, iters: int, epsilon: float = 0.1, tau: float = 1.0):
    """ Extended UOT method with optional support for additional regularization or structures """
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    # Add dustbin channels
    couplings = torch.cat([torch.cat([scores, bins0], -1),
                          torch.cat([bins1, alpha], -1)], 1)

    # Compute marginal vectors
    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    # Perform Sinkhorn iterations with regularization
    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters, epsilon, tau)

    # Optional post-processing or regularization extension goes here

    Z = Z - norm  # Multiply probabilities by M+N
    return Z

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1