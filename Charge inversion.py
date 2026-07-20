import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import solve
import matplotlib.pyplot as plt

class ChargeInversionIRLS:
    def __init__(self, n_meas_x=20, n_meas_y=20, n_recon_x=60, n_recon_y=60, delta=0.03, alpha_ema=0.8):
        self.k_e = 8.9875517923e9
        self.delta = delta
        self.alpha_ema = alpha_ema
        self.n_meas_x = n_meas_x
        self.n_meas_y = n_meas_y
        self.n_recon_x = n_recon_x
        self.n_recon_y = n_recon_y
        self.N = n_meas_x * n_meas_y
        self.M = n_recon_x * n_recon_y

    def build_forward_matrix(self, pos_meas, pos_recon):
        A = np.zeros((self.N, self.M))
        for i in range(self.N):
            dist_sq = np.sum((pos_recon - pos_meas[i]) ** 2, axis=1)
            A[i, :] = self.k_e / np.sqrt(dist_sq + self.delta ** 2)
        return A

    def _awgcv_fast_objective(self, lambd, S, c, omega):
        D_AM = S / (S + lambd)
        num = self.N * np.sum(((lambd / (S + lambd)) * c) ** 2)
        den = (self.N - omega * np.sum(D_AM)) ** 2
        return num / (den + 1e-12)

    def solve(self, A, Phi_meas_matrix, max_iter=50, tol=1e-4, epsilon=1e-2):
        Phi_meas = Phi_meas_matrix.ravel()

        scale_A = np.max(np.abs(A)) + 1e-12
        A_norm = A / scale_A

        scale_Phi = np.max(np.abs(Phi_meas)) + 1e-12
        Phi_norm = Phi_meas / scale_Phi

        A_T_A = A_norm.T @ A_norm
        sigma_k = np.linalg.solve(A_T_A + 1e-2 * np.eye(self.M), A_norm.T @ Phi_norm)
        omega_k = 1.0
        G_history = []

        Phi_norm_norm = np.linalg.norm(Phi_norm)

        for k in range(max_iter):
            print(f"Performing iteration {k + 1}/{max_iter} ", end="")

            W_k_diag = 1.0 / np.sqrt(np.abs(sigma_k) + epsilon)
            W_inv = 1.0 / W_k_diag

            current_residual = np.linalg.norm(Phi_norm - A_norm @ sigma_k)
            omega_current = min(1.0, current_residual / (Phi_norm_norm + 1e-12))

            if k > 0:
                omega_k = self.alpha_ema * omega_current + (1 - self.alpha_ema) * omega_k

            H = (A_norm * W_inv) @ A_norm.T
            S, U = np.linalg.eigh(H)

            S = np.maximum(S, 0)
            c = U.T @ Phi_norm

            res = minimize_scalar(
                lambda log_lam: self._awgcv_fast_objective(10 ** log_lam, S, c, omega_k),
                bounds=(-2.5, 1), method='bounded'
            )
            best_lambda = 10 ** res.x
            G_val = res.fun
            G_history.append(G_val)

            print(f" [Optimal lambda: {best_lambda:.2e}, GCV : {G_val:.4e}]")

            y = U @ (c / (S + best_lambda))
            sigma_next = W_inv * (A_norm.T @ y)

            if k > 2:
                rel_change = np.abs(G_history[-1] - G_history[-2]) / G_history[0]
                semi_convergence = (G_history[-1] > G_history[-2]) and (G_history[-2] > G_history[-3])

                if rel_change < tol or semi_convergence:
                    print(f"\n The algorithm completed normally in the {k + 1}-th iteration")
                    break

            sigma_k = sigma_next

        sigma_real = sigma_k * (scale_Phi / scale_A)
        return sigma_real.reshape((self.n_recon_x, self.n_recon_y)), G_history


if __name__ == "__main__":
    meas_x, meas_y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
    pos_meas = np.column_stack([meas_x.ravel(), meas_y.ravel()])

    recon_x, recon_y = np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 60))
    pos_recon = np.column_stack([recon_x.ravel(), recon_y.ravel()])

    Phi_meas_20x20 = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            heel = 5.0 * np.exp(-((i - 9) ** 2 / 8.0 + (j - 4) ** 2 / 10.0))
            forefoot = 3.5 * np.exp(-((i - 11) ** 2 / 12.0 + (j - 14) ** 2 / 18.0))
            midfoot = 1.5 * np.exp(-((i - 10) ** 2 / 6.0 + (j - 9) ** 2 / 15.0))
            Phi_meas_20x20[i, j] = heel + forefoot + midfoot

    inversion_algo = ChargeInversionIRLS(
        n_meas_x=20, n_meas_y=20,
        n_recon_x=60, n_recon_y=60,
        delta=0.03, alpha_ema=0.8
    )

    A_matrix = inversion_algo.build_forward_matrix(pos_meas, pos_recon)
    sigma_60x60, G_curve = inversion_algo.solve(A_matrix, Phi_meas_20x20)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    im_input = plt.imshow(Phi_meas_20x20.T, cmap='jet', origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar(im_input, label='Measured Potential (V)')
    plt.title('Simulated Input Array Voltage Amplitude', fontweight='bold')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')

    plt.subplot(1, 2, 2)
    im_output = plt.imshow(sigma_60x60.T, cmap='jet', origin='lower', extent=[0, 1, 0, 1], interpolation='bicubic')
    cbar = plt.colorbar(im_output)
    cbar.set_label('Charge Density (C/m²)')
    plt.title('Inversion Result', fontweight='bold')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')

    plt.tight_layout()
    plt.show()