import numpy as np

from reward_model import RewardModel


class RewardModel_Task_Dynamic(RewardModel):

    def __init__(self, ds, da, device,
                 task_dim=None,
                 ensemble_size=3, lr=3e-4, mb_size=128, size_segment=1,
                 env_maker=None, max_size=100, activation='tanh', capacity=5e5,
                 large_batch=1, label_margin=0.0,
                 teacher_beta=-1, teacher_gamma=1,
                 teacher_eps_mistake=0,
                 teacher_eps_skip=0,
                 teacher_eps_equal=0,

                 data_aug_ratio=20,
                 data_aug_window=5,
                 threshold_u=0.99,
                 lambda_u=1,

                 coef_disagree=0.1,
                 coef_tdminus=2.0,
                 coef_tdist=0.1,
                 ):

        super().__init__(ds=ds, da=da, device=device,
                         ensemble_size=ensemble_size, lr=lr, mb_size=mb_size, size_segment=size_segment,
                         env_maker=env_maker, max_size=max_size, activation=activation, capacity=capacity,
                         large_batch=large_batch, label_margin=label_margin,
                         teacher_beta=teacher_beta, teacher_gamma=teacher_gamma,
                         teacher_eps_mistake=teacher_eps_mistake,
                         teacher_eps_skip=teacher_eps_skip,
                         teacher_eps_equal=teacher_eps_equal,
                         data_aug_ratio=data_aug_ratio,
                         data_aug_window=data_aug_window,
                         threshold_u=threshold_u,
                         lambda_u=lambda_u,)
        self.coef_disagree = coef_disagree
        self.coef_tdminus = coef_tdminus
        self.coef_tdist = coef_tdist

    def get_task_distance(self, sa_t_1, sa_t_2, distance_metric_func):

        d = distance_metric_func
        s10, s11 = sa_t_1[:, 0, :self.ds], sa_t_1[:, -1, :self.ds]
        s20, s21 = sa_t_2[:, 0, :self.ds], sa_t_2[:, -1, :self.ds]
        d_arr = np.array([d(s11, s21), d(s11, s20),
                         d(s10, s21), d(s10, s20),])
        score_d = np.mean(np.abs(d_arr), axis=0)
        return score_d

    def dynamic_task_consistent_sampling(self, distance_metric_func):

        sa_t_1, sa_t_2, r_t_1, r_t_2, ep_num_1, ep_num_2, ep_start_1, ep_start_2 = \
            self.get_queries(mb_size=self.mb_size * self.large_batch)

        score_d = self.get_task_distance(sa_t_1, sa_t_2,
                                         distance_metric_func=distance_metric_func)

        top_k_index = score_d.argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        ep_num_1, ep_num_2 = ep_num_1[top_k_index], ep_num_2[top_k_index]
        ep_start_1, ep_start_2 = ep_start_1[top_k_index], ep_start_2[top_k_index]

        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, ep_num_1, ep_num_2, ep_start_1, ep_start_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        if self.data_aug_ratio:
            self.surf_data_aug_process()

        return len(labels)

    def dynamic_task_consistent_disagreement_filter_sampling(self, distance_metric_func, disagree_filter_ratio=0.5):

        sa_t_1, sa_t_2, r_t_1, r_t_2, ep_num_1, ep_num_2, ep_start_1, ep_start_2 = \
            self.get_queries(mb_size=self.mb_size * self.large_batch)

        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        disagree_top_index = (-disagree).argsort()[:int(self.mb_size * self.large_batch * disagree_filter_ratio)]
        sa_t_1, sa_t_2 = sa_t_1[disagree_top_index], sa_t_2[disagree_top_index]
        r_t_1, r_t_2 = r_t_1[disagree_top_index], r_t_2[disagree_top_index]

        score_d = self.get_task_distance(sa_t_1, sa_t_2,
                                         distance_metric_func=distance_metric_func)

        top_k_index = score_d.argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        ep_num_1, ep_num_2 = ep_num_1[top_k_index], ep_num_2[top_k_index]
        ep_start_1, ep_start_2 = ep_start_1[top_k_index], ep_start_2[top_k_index]

        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, ep_num_1, ep_num_2, ep_start_1, ep_start_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        if self.data_aug_ratio:
            self.surf_data_aug_process()

        return len(labels)

    def dynamic_task_consistent_disagreement_real_multiple_sampling(self, distance_metric_func):

        sa_t_1, sa_t_2, r_t_1, r_t_2, ep_num_1, ep_num_2, ep_start_1, ep_start_2 = \
            self.get_queries(mb_size=self.mb_size * self.large_batch)

        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        score_d = self.get_task_distance(sa_t_1, sa_t_2,
                                         distance_metric_func=distance_metric_func)
        score_d_norm = (score_d - score_d.min()) / (score_d.max() - score_d.min())
        disagree_norm = (disagree - disagree.min()) / (disagree.max() - disagree.min())

        rank_score = (self.coef_disagree + disagree_norm) * (self.coef_tdminus - score_d_norm)

        top_k_index = (-rank_score).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        ep_num_1, ep_num_2 = ep_num_1[top_k_index], ep_num_2[top_k_index]
        ep_start_1, ep_start_2 = ep_start_1[top_k_index], ep_start_2[top_k_index]

        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, ep_num_1, ep_num_2, ep_start_1, ep_start_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        if self.data_aug_ratio:
            self.surf_data_aug_process()

        return len(labels)
