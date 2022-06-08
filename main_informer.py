import argparse
import os
import torch

from exp.exp_informer import Exp_Informer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')

    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='informer',
                        help='model of experiment, options: [informer, informerstack, informerlight(TBD), ')

    parser.add_argument('--data', type=str, required=True, default='ba102', help='data')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ba102.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='C2H4', help='target feature in S or MS task')
   
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
    parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')

    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ba102': {'data': 'ba102.csv', 'T': ['H2','CH4','C2H4','C2H6','C3H6','C3H8'], 'M': [26, 26, 26], 'S': [1, 1, 1], 'MS': [26, 26, 6]},
        'ba105': {'data': 'ba105.csv', 'T': ['H2','CH4','C2H4','C2H6','C3H6','C3H8'], 'M': [26, 26, 26], 'S': [1, 1, 1], 'MS': [26, 26, 6]},
        'ba_all': {'data': 'ba_all.csv', 'T': ['H2','CH4','C2H4','C2H6','C3H6','C3H8'], 'M': [25, 25, 25], 'S': [1, 1, 1], 'MS': [25, 25, 6]},
        'ba_no_period': {'data': 'ba_no_period.csv', 'T': ['H2','CH4','C2H4','C2H6','C3H6','C3H8'], 'M': [24, 24, 24], 'S': [1, 1, 1], 'MS': [24, 24, 6]},
        'ba_pc':{'data': 'ba_pc.csv', 'T': ['H2','CH4','C2H4','C2H6','C3H6','C3H8'], 'M': [19, 19, 19], 'S': [1, 1, 1], 'MS': [19, 19, 6]},
        'sinter_8000':{'data': 'sinter_8000.csv', 'T': ['sv1','sv2','sv3','sv4','sv5','sv6'], 'M': [15, 15, 15], 'S': [1, 1, 1], 'MS': [15, 15, 6]},
        'sinter_27000':{'data': 'sinter_27000.csv', 'T': ['sv1','sv2','sv3','sv4','sv5','sv6'], 'M': [15, 15, 15], 'S': [1, 1, 1], 'MS': [15, 15, 6]},
        'sinter_27000_order':{'data': 'sinter_27000_order.csv', 'T': ['sv1','sv2','sv3','sv4','sv5','sv6'], 'M': [15, 15, 15], 'S': [1, 1, 1], 'MS': [15, 15, 6]},
        'sinter_sv1':{'data': 'sinter_sv1.csv', 'T': ['sv1'], 'M': [10, 10, 10], 'S': [1, 1, 1], 'MS': [10, 10, 1]},
        'sinter_sv2':{'data': 'sinter_sv2.csv', 'T': ['sv2'], 'M': [10, 10, 10], 'S': [1, 1, 1], 'MS': [10, 10, 1]},
        'sinter_sv3':{'data': 'sinter_sv3.csv', 'T': ['sv3'], 'M': [10, 10, 10], 'S': [1, 1, 1], 'MS': [10, 10, 1]},
        'sinter_sv4':{'data': 'sinter_sv4.csv', 'T': ['sv4'], 'M': [10, 10, 10], 'S': [1, 1, 1], 'MS': [10, 10, 1]},
        'sinter_sv5':{'data': 'sinter_sv5.csv', 'T': ['sv5'], 'M': [10, 10, 10], 'S': [1, 1, 1], 'MS': [10, 10, 1]},
        'sinter_sv6':{'data': 'sinter_sv6.csv', 'T': ['sv6'], 'M': [10, 10, 10], 'S': [1, 1, 1], 'MS': [10, 10, 1]},    
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]


    print('Args in experiment:')
    print(args)

    Exp = Exp_Informer

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                            args.model,
                                                                                                            args.data,
                                                                                                            args.features,
                                                                                                            args.seq_len,
                                                                                                            args.label_len,
                                                                                                            args.pred_len,
                                                                                                            args.d_model,
                                                                                                            args.n_heads,
                                                                                                            args.e_layers,
                                                                                                            args.d_layers,
                                                                                                            args.d_ff,
                                                                                                            args.attn,
                                                                                                            args.factor,
                                                                                                            args.distil,
                                                                                                            args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                        args.model,
                                                                                                        args.data,
                                                                                                        args.features,
                                                                                                        args.seq_len,
                                                                                                        args.label_len,
                                                                                                        args.pred_len,
                                                                                                        args.d_model,
                                                                                                        args.n_heads,
                                                                                                        args.e_layers,
                                                                                                        args.d_layers,
                                                                                                        args.d_ff,
                                                                                                        args.attn,
                                                                                                        args.factor,
                                                                                                        args.distil,
                                                                                                        args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()