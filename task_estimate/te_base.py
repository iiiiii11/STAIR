class TE_Base:
    task_type = None

    def __init__(self, n_tasks=None) -> None:
        self.n_tasks = n_tasks

    def predict_task(self, state, info: dict):
        '''
        return a probability vector indicating the subtask the state belongs
        '''
        return None

    def update(self, replay_buffer, logger, step):
        pass


class TE_Dynamic_Base:
    task_type = None

    def __init__(self, n_tasks=None) -> None:
        self.n_tasks = n_tasks

    def get_task_distance(self, state1, state2):
        '''
        return a probability vector indicating the subtask the state belongs
        '''
        return None

    def update(self, replay_buffer, logger, step):
        pass
