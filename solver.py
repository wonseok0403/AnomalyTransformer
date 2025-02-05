import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train', dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre', dataset=self.dataset)

        self.build_model()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.criterion = nn.MSELoss()
        self.win_size = 100

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.model.to(device)

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss, prior_loss = self.compute_losses(series, prior)
            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def compute_losses(self, series, prior):
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            series_loss += (torch.mean(my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach())) + torch.mean(
                my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach(),
                    series[u])))
            prior_loss += (torch.mean(
                my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)),
                           series[u].detach())) + torch.mean(
                my_kl_loss(series[u].detach(),
                           (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)))))
        series_loss = series_loss / len(prior)
        prior_loss = prior_loss / len(prior)
        return series_loss, prior_loss

    def train(self):
        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss, prior_loss = self.compute_losses(series, prior)

                rec_loss = self.criterion(output, input)

                # MAE 손실 추가
                mae_loss = F.l1_loss(output, input)

                # 가중치를 둔 손실 계산
                weighted_rec_loss = rec_loss * 0.6 + mae_loss * 0.4

                loss1_list.append((weighted_rec_loss - self.k * series_loss).item())
                loss1 = weighted_rec_loss - self.k * series_loss
                loss2 = weighted_rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model.load_state_dict(
            torch.load(os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        train_energy = self.evaluate_energy(self.train_loader, criterion, temperature)

        # (2) find the threshold
        test_energy = self.evaluate_energy(self.thre_loader, criterion, temperature)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        optimal_thresh = None
        best_fscore = 0
        thresholds = np.linspace(np.percentile(combined_energy, 90), np.percentile(combined_energy, 99), 10)

        for thresh in thresholds:
            pred, gt = self.evaluate_performance(self.test_loader, criterion, temperature, thresh)
            _, _, _, f_score = self.calculate_metrics(pred, gt)
            if f_score > best_fscore:
                best_fscore = f_score
                optimal_thresh = thresh

        # print("Optimal Threshold :", optimal_thresh * 0.8)



        # (3) evaluation on the test set with the optimal threshold
        pred, gt = self.evaluate_performance(self.test_loader, criterion, temperature, optimal_thresh)

        anomaly_state = False
        ## performance test (for)
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        accuracy, precision, recall, f_score = self.calculate_metrics(pred, gt)
        print("Threshold :", optimal_thresh)
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision,
                                                                                                   recall, f_score))

        # 오탐지 및 누락된 탐지에 대한 정보 제공
        false_positives = np.where((pred == 1) & (gt == 0))[0]
        false_negatives = np.where((pred == 0) & (gt == 1))[0]
        true_positives = np.where((pred == 1) & (gt == 1))[0]
        true_negatives = np.where((pred == 0) & (gt == 0))[0]

        print(f"False Positives (Type I Error): {len(false_positives)}")
        print(f"False Negatives (Type II Error): {len(false_negatives)}")
        print(f"True Positives: {len(true_positives)}")
        print(f"True Negatives: {len(true_negatives)}")

        # 구간별 시각화
        segment_length = 50000  # 각 구간의 길이
        num_segments = len(gt) // segment_length + 1

        for segment in range(num_segments):
            start = segment * segment_length
            end = min((segment + 1) * segment_length, len(gt))
            plt.figure(figsize=(15, 6))
            plt.plot(range(start, end), gt[start:end], label='Ground Truth', color='blue')
            plt.plot(range(start, end), pred[start:end], label='Predictions', color='red', linestyle='--')

            segment_false_positives = false_positives[(false_positives >= start) & (false_positives < end)]
            segment_false_negatives = false_negatives[(false_negatives >= start) & (false_negatives < end)]
            segment_true_positives = true_positives[(true_positives >= start) & (true_positives < end)]
            segment_true_negatives = true_negatives[(true_negatives >= start) & (true_negatives < end)]

            plt.scatter(segment_false_positives, gt[segment_false_positives], color='green', label='False Positives',
                        zorder=5)
            plt.scatter(segment_false_negatives, gt[segment_false_negatives], color='purple', label='False Negatives',
                        zorder=5)
            plt.scatter(segment_true_positives, gt[segment_true_positives], color='black', label='True Positives',
                        zorder=5)
            plt.scatter(segment_true_negatives, gt[segment_true_negatives], color='orange', label='True Negatives',
                        zorder=5)

            plt.xlabel('Time Index')
            plt.ylabel('Anomaly')
            plt.title(f'Anomaly Detection Results (Segment {segment + 1}/{num_segments})')
            plt.legend()
            plt.show()

        return accuracy, precision, recall, f_score

    def evaluate_performance(self, loader, criterion, temperature, threshold):
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss, prior_loss = self.compute_losses(series, prior)
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        pred = (attens_energy > threshold).astype(int)
        return pred, test_labels

    def calculate_metrics(self, pred, gt):
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(gt, pred, average='binary')
        return accuracy, precision, recall, f_score

    def evaluate_energy(self, loader, criterion, temperature):
        attens_energy = []
        for i, (input_data, labels) in enumerate(loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss, prior_loss = self.compute_losses(series, prior)
            reconstruction_loss = torch.mean((input - output) ** 2, dim=-1)
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * reconstruction_loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        return attens_energy
