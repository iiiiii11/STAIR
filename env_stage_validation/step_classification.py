import torch
import torch.nn as nn
import numpy as np

import utils
from .replay_buffer_step import ReplayBufferWithStep


class StepClassficationNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim, hidden_depth, step_range):
        super().__init__()
        self.step_range = step_range
        self.min_step = step_range[0]
        self.max_step = step_range[1]
        self.layers = utils.mlp(obs_dim, hidden_dim, self.max_step - self.min_step + 1, hidden_depth)

    def forward(self, state):
        logits = self.layers(state)
        return logits

    def estimateStep(self, state):
        logits = self.forward(state)
        return torch.argmax(logits) + self.min_step


class StepClassfication:
    def __init__(self, cfg, hidden_dim, hidden_depth, step_range, eval_data_rate, lr, epochs, batch_size):
        self.device = cfg.device
        self.model_classifier = StepClassficationNet(obs_dim=cfg.agent.params.obs_dim, hidden_dim=hidden_dim,
                                                     hidden_depth=hidden_depth, step_range=step_range).to(self.device)
        self.eval_data_rate = eval_data_rate
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, replay_buffer: ReplayBufferWithStep):
        print("train Classfication")
        state = torch.from_numpy(replay_buffer.obses).to(self.device)
        steps = torch.from_numpy(replay_buffer.timestep).to(self.device)

        dataset_size = state.size(0)
        indices = torch.randperm(dataset_size)
        train_size = int((1 - self.eval_data_rate) * dataset_size)
        train_indices, eval_indices = indices[:train_size], indices[train_size:]
        train_state, train_steps = state[train_indices], steps[train_indices]
        eval_state, eval_steps = state[eval_indices], steps[eval_indices]

        optimizer = torch.optim.Adam(self.model_classifier.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        train_log = {
            "train_loss": [],
            "train_acc": [],
            "eval_loss": [],
            "eval_acc": [],
        }

        for epoch in range(self.epochs):
            self.model_classifier.train()
            total_train_loss = 0.0
            total_train_acc = 0.0
            train_order = torch.randperm(train_state.size(0))

            for batch in range(0, train_state.size(0), self.batch_size):
                batch_indices = train_order[batch:batch + self.batch_size]
                batch_state = train_state[batch_indices]
                batch_steps = train_steps[batch_indices]

                optimizer.zero_grad()
                logits = self.model_classifier(batch_state)
                loss = criterion(logits, batch_steps.long().squeeze(-1))

                with torch.no_grad():
                    total_train_loss += loss.item() * batch_state.size(0)
                    preds = torch.argmax(logits, dim=1)
                    batch_acc = (preds == batch_steps).float().mean().item()
                    total_train_acc += batch_acc * batch_state.size(0)
                loss.backward()
                optimizer.step()

            avg_train_loss = total_train_loss / train_state.size(0)
            avg_train_acc = total_train_acc / train_state.size(0)
            print(f"[{epoch+1}/{self.epochs}] Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}")
            train_log["train_loss"].append(avg_train_loss)
            train_log["train_acc"].append(avg_train_acc)

            self.model_classifier.eval()
            total_eval_loss = 0.0
            total_eval_acc = 0.0

            with torch.no_grad():
                for batch in range(0, eval_state.size(0), self.batch_size):
                    batch_eval_state = eval_state[batch:batch + self.batch_size]
                    batch_eval_steps = eval_steps[batch:batch + self.batch_size]
                    eval_logits = self.model_classifier(batch_eval_state)
                    total_eval_loss += criterion(eval_logits, batch_eval_steps.long().squeeze(-1)
                                                 ).item() * batch_eval_state.size(0)
                    eval_preds = torch.argmax(eval_logits, dim=1)
                    total_eval_acc += (eval_preds == batch_eval_steps).float().mean().item() * batch_eval_state.size(0)

            avg_eval_loss = total_eval_loss / eval_state.size(0)
            avg_eval_acc = total_eval_acc / eval_state.size(0)
            print(f"[{epoch+1}/{self.epochs}] Eval Loss: {avg_eval_loss:.4f}, Eval Acc: {avg_eval_acc:.4f}")
            train_log["eval_loss"].append(avg_eval_loss)
            train_log["eval_acc"].append(avg_eval_acc)
        return eval_indices, train_log

    def batch_predict(self, state_list: np.ndarray, batch_size: int):
        state_tensor = torch.from_numpy(state_list).float().to(self.device)

        self.model_classifier.eval()
        all_preds = []

        with torch.no_grad():
            for start_idx in range(0, len(state_tensor), batch_size):
                end_idx = start_idx + batch_size
                batch_data = state_tensor[start_idx:end_idx]
                batch_logits = self.model_classifier(batch_data)
                batch_pred = torch.argmax(batch_logits, dim=1)
                preds = batch_pred.cpu().numpy().flatten()
                all_preds.append(preds)

        est_step_list = np.concatenate(all_preds, axis=0)
        return est_step_list
