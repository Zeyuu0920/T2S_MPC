import casadi as cs
import numpy as np


class Quadrotor2DDynamics:
    def __init__(
        self,
        gym_env,
        residual_model=None,
        use_residual=False,
        use_time_embedding=False,
        time_feat_dim=32,
        time_scale=1.0,
    ):
        self.gym_env = gym_env
        self.residual_model = residual_model
        self.use_residual = use_residual
        self.use_time_embedding = use_time_embedding
        self.time_feat_dim = time_feat_dim
        self.time_scale = time_scale

    def model(self):
        # nominal parameters
        m_nominal_ratio = 1
        Iyy_nominal_ratio = 1
        m = self.gym_env.MASS * m_nominal_ratio
        Iyy = self.gym_env.J[1, 1] * Iyy_nominal_ratio
        g, length = self.gym_env.GRAVITY_ACC, self.gym_env.L

        # states
        z = cs.MX.sym('z')
        z_dot = cs.MX.sym('z_dot')
        x = cs.MX.sym('x')
        x_dot = cs.MX.sym('x_dot')
        theta = cs.MX.sym('theta')
        theta_dot = cs.MX.sym('theta_dot')
        X = cs.vertcat(x, x_dot, z, z_dot, theta, theta_dot)

        # controls
        T1 = cs.MX.sym('T1')
        T2 = cs.MX.sym('T2')
        U = cs.vertcat(T1, T2)

        nx, nu = 6, 2

        # input bounds
        n_mot = 4 / nu
        a_low = self.gym_env.KF * n_mot * (
            self.gym_env.PWM2RPM_SCALE * self.gym_env.MIN_PWM + self.gym_env.PWM2RPM_CONST
        ) ** 2
        a_high = self.gym_env.KF * n_mot * (
            self.gym_env.PWM2RPM_SCALE * self.gym_env.MAX_PWM + self.gym_env.PWM2RPM_CONST
        ) ** 2
        u_min = a_low * np.ones(nu)
        u_max = a_high * np.ones(nu)

        # nominal dynamics
        X_dot_nominal = cs.vertcat(
            x_dot,
            cs.sin(theta) * (T1 + T2) / m,
            z_dot,
            cs.cos(theta) * (T1 + T2) / m - g,
            theta_dot,
            length * (T2 - T1) / Iyy / np.sqrt(2)
        )

        # parameter for time embedding
        if self.use_time_embedding:
            tau = cs.MX.sym('tau')
            P = cs.vertcat(tau)

            d = self.time_feat_dim
            half = d // 2
            omegas = [np.pi / (j + 1) for j in range(half)]
            sin_feats = [cs.sin(w * tau) for w in omegas]
            cos_feats = [cs.cos(w * tau) for w in omegas]
            phi_tau = cs.vertcat(*(sin_feats + cos_feats))
        else:
            P = cs.vertcat([])

        # residual dynamics
        if self.use_residual:
            if self.residual_model is None:
                raise ValueError("use_residual=True but residual_model is None.")

            if self.use_time_embedding:
                mlp_input = cs.vertcat(X, U, phi_tau)
            else:
                mlp_input = cs.vertcat(X, U)

            residual = self.residual_model(mlp_input.T).T
            X_dot_residual = cs.vertcat(0, residual[0], 0, residual[1], 0, residual[2])
            f_expl = X_dot_nominal + X_dot_residual
            model_name = "quadrotor2D_learned"
        else:
            X_dot_residual = cs.vertcat(0, 0, 0, 0, 0, 0)
            f_expl = X_dot_nominal
            model_name = "quadrotor2D_nominal"

        x_start = np.array([0, 0, 0.75, 0, 0, 0])

        model = cs.types.SimpleNamespace()
        model.x = X
        model.xdot = cs.MX.sym('xdot', 6)
        model.u = U
        model.u_min = u_min
        model.u_max = u_max
        model.z = cs.vertcat([])
        model.p = P
        model.f_expl = f_expl
        model.f_nominal = X_dot_nominal
        model.f_residual = X_dot_residual
        model.x_start = x_start
        model.constraints = cs.vertcat([])
        model.name = model_name

        return model