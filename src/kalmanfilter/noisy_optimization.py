import numpy as np

def solve_noise_fixed_point(f_vals, g_vals, delta_x_dict, weights, max_iter=100, tol=1e-6):
    # Copied minimal, trusted implementation from workspace root noisy_optimization.py
    m = len(f_vals)
    n_grad = g_vals.shape[1] if g_vals.ndim > 1 else 1
    e = np.zeros(m)
    e_bar = np.zeros((m, n_grad))
    w_xi = weights.get('w_xi', 1.0)
    w_g_xi = weights.get('w_g_xi', 1.0)
    w_e = weights.get('w_e', 1.0)
    w_g_e = weights.get('w_g_e', 1.0)
    for iteration in range(max_iter):
        e_prev = e.copy()
        e_bar_prev = e_bar.copy()
        S_bar = {}
        alpha = {}
        for k, zone_data in delta_x_dict.items():
            I_k = zone_data['I_k']
            dim_y = zone_data['dim_y']
            S_k_sum = np.zeros((dim_y, dim_y))
            alpha_k_sum = np.zeros(dim_y)
            for i in I_k:
                a_ik = zone_data['a_i'][i]
                A_ik = zone_data['A_i'][i]
                S_k_sum += w_xi * np.outer(a_ik, a_ik) + w_g_xi * np.dot(A_ik.T, A_ik)
                alpha_k_sum += w_xi * (f_vals[i] - e[i]) * a_ik + w_g_xi * np.dot(A_ik.T, g_vals[i] - e_bar[i])
            S_bar[k] = S_k_sum
            alpha[k] = alpha_k_sum
        new_e = np.zeros(m)
        new_e_bar = np.zeros((m, n_grad))
        for i in range(m):
            sum_term_e = 0.0
            sum_term_ebar = np.zeros(n_grad)
            active_zones = [k for k, zone_data in delta_x_dict.items() if i in zone_data['I_k']]
            for k in active_zones:
                zone_data = delta_x_dict[k]
                a_ik = zone_data['a_i'][i]
                A_ik = zone_data['A_i'][i]
                S_inv_alpha = np.linalg.solve(S_bar[k], alpha[k])
                sum_term_e += w_xi * f_vals[i] + w_xi * np.dot(a_ik, S_inv_alpha)
                sum_term_ebar += w_g_xi * g_vals[i] + w_g_xi * np.dot(A_ik, S_inv_alpha)
            total_w_e = w_e + sum(w_xi for k in active_zones)
            total_w_ebar = w_g_e + sum(w_g_xi for k in active_zones)
            if total_w_e > 0:
                new_e[i] = -sum_term_e / total_w_e
            if total_w_ebar > 0:
                new_e_bar[i] = -sum_term_ebar / total_w_ebar
        e = new_e
        e_bar = new_e_bar
        diff = np.max(np.abs(e - e_prev)) + np.max(np.abs(e_bar - e_bar_prev))
        if diff < tol:
            break
    return e, e_bar


def create_test_cloud(x_k, m_points=20, scale=0.01):
    x_k = np.asarray(x_k, dtype=float)
    n = len(x_k)
    perturbations = np.random.normal(loc=0.0, scale=scale, size=(m_points, n))
    x_points = x_k + perturbations
    delta_x = x_points - x_k
    # include the center as the first point for convenience
    x_with_center = np.vstack((x_k[None, :], x_points))
    delta_with_center = np.vstack((np.zeros((1, n)), delta_x))
    return x_with_center, delta_with_center
