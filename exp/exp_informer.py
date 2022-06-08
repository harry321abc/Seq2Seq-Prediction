from data.data_loader import Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models import informer, informer_stack, transformer, TCN, informerT1, informerT2, lstm, ann, rnn

from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'transformer': transformer,
            'informer': informer,
            'Informer_stack': informer_stack,
            'TCN':TCN,
            'lstm':lstm,
            'ann':ann,
            'rnn':rnn,
            'informerT1':informerT1,
            'informerT2':informerT2,
        }
        model = model_dict[self.args.model].Informer(
            self.args.enc_in,
            self.args.dec_in,
            self.args.c_out,
            self.args.seq_len,
            self.args.label_len,
            self.args.pred_len,
            self.args.factor,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.dropout,
            self.args.attn,
            self.args.activation,
            self.args.output_attention,
            self.args.distil,
            self.device
        ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'custom': Dataset_Custom,
            'ba102': Dataset_Custom,
            'ba105': Dataset_Custom,
            'ba_all': Dataset_Custom,
            'ba_no_period':Dataset_Custom,
            'ba_pc': Dataset_Custom,
            'sinter_8000': Dataset_Custom,
            'sinter_27000': Dataset_Custom,
            'sinter_27000_order': Dataset_Custom,
            'sinter_sv1': Dataset_Custom,
            'sinter_sv2': Dataset_Custom,
            'sinter_sv3': Dataset_Custom,
            'sinter_sv4': Dataset_Custom,
            'sinter_sv5': Dataset_Custom,
            'sinter_sv6': Dataset_Custom,
        }

        Data = data_dict[self.args.data]

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device) #shape: B*L*D
            batch_y = batch_y.float()

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, dec_inp)[0]
                    else:
                        outputs = self.model(batch_x, dec_inp)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, dec_inp)[0]
                else:
                    outputs = self.model(batch_x, dec_inp)
            f_dim = -6 if self.args.features == 'MS' else 0
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true)

            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                # print(batch_y[0,:,:])

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, dec_inp)[0]
                        else:
                            outputs = self.model(batch_x, dec_inp)

                        f_dim = -6 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # loss = 10*criterion(outputs[:,:,-3], batch_y[:,:,-3])+10*criterion(outputs[:,:,-4], batch_y[:,:,-4])+3*criterion(outputs[:,:,-1], batch_y[:,:,-1])+criterion(outputs[:,:,-2], batch_y[:,:,-2])+criterion(outputs[:,:,-5], batch_y[:,:,-5])+criterion(outputs[:,:,-6], batch_y[:,:,-6])
                        loss = criterion(outputs,batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, dec_inp)[0]
                    else:
                        outputs = self.model(batch_x, dec_inp)

                    f_dim = -6 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        mean = np.array([-14.20067736,-14.36835474,98.7131974,419.94757219,85.03548647,406.51655405])
        var = np.array([1.23702907e+00,1.36719260e+00,3.65652871e+02,9.54663824e+02,3.83617617e+00,5.11087647e+02])
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()

        preds = []
        trues = []

        folder_path = './test_results/' + setting + '/'    
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, dec_inp)[0]
                    else:
                        outputs = self.model(batch_x, dec_inp)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, dec_inp)[0]
                else:
                    outputs = self.model(batch_x, dec_inp)
            f_dim = -6 if self.args.features == 'MS' else 0
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            

            pred = outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y.detach().cpu().numpy()  # .squeeze()

            preds.append(pred)
            trues.append(true)
            if i % 20 == 0:
                input = batch_x.detach().cpu().numpy()
                gt = np.concatenate((input[0, :, -1]*(5.11087647e+02**0.5)+406.51655405, true[0, :, -1]*(5.11087647e+02**0.5)+406.51655405), axis=0)
                pd = np.concatenate((input[0, :, -1]*(5.11087647e+02**0.5)+406.51655405, pred[0, :, -1]*(5.11087647e+02**0.5)+406.51655405), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        preds = preds * np.sqrt(var) + mean #反标准化
        trues = trues * np.sqrt(var) + mean #反标准化

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds[:,:,0], trues[:,:,0])
        print('mse:{}, mae:{}'.format(mse, mae))
        np.save(folder_path + 'metrics_0.npy', np.array([mae, mse, rmse, mape, mspe]))
        mae, mse, rmse, mape, mspe = metric(preds[:,:,1], trues[:,:,1])
        print('mse:{}, mae:{}'.format(mse, mae))
        np.save(folder_path + 'metrics_1.npy', np.array([mae, mse, rmse, mape, mspe]))
        mae, mse, rmse, mape, mspe = metric(preds[:,:,2], trues[:,:,2])
        print('mse:{}, mae:{}'.format(mse, mae))
        np.save(folder_path + 'metrics_2.npy', np.array([mae, mse, rmse, mape, mspe]))
        mae, mse, rmse, mape, mspe = metric(preds[:,:,3], trues[:,:,3])
        print('mse:{}, mae:{}'.format(mse, mae))
        np.save(folder_path + 'metrics_3.npy', np.array([mae, mse, rmse, mape, mspe]))
        mae, mse, rmse, mape, mspe = metric(preds[:,:,4], trues[:,:,4])
        print('mse:{}, mae:{}'.format(mse, mae))
        np.save(folder_path + 'metrics_4.npy', np.array([mae, mse, rmse, mape, mspe]))
        mae, mse, rmse, mape, mspe = metric(preds[:,:,5], trues[:,:,5])
        print('mse:{}, mae:{}'.format(mse, mae))
        np.save(folder_path + 'metrics_5.npy', np.array([mae, mse, rmse, mape, mspe]))

        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if self.args.features == 'MS' else 0
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

            pred = outputs.detach().cpu().numpy()  # .squeeze()

            preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
