import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import backend as K
from keras.applications import vgg16

from model import create_model
from grating import create_gratings
from utils import prepare_train_test

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['font.size'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True

def get_response_function(layer_name):
    # input -> hidden activation
    layer_output = layer_dict[layer_name].output
    if layer_name in ['fc1', 'fc2']:
        activation = layer_output
    else:
        activation = K.mean(K.mean(layer_output, axis=1), axis=1)
    func = K.function([model.input], [activation])
    return func

def get_oristats(n_cells, responses_all, n_plot=0, n_ori=12):
    pref_oris = np.zeros(n_cells)
    orth_oris = np.zeros(n_cells)
    osis = np.zeros(n_cells)
    gosis = np.zeros(n_cells)
    ori_ranges = np.zeros(n_cells)

    for c in range(n_cells):
        # get max along the phase axis
        responses_max = np.max(responses_all[:,:,:,c], axis=2) # (n_spf, n_ori)
        pref_spf = np.argmax(np.max(responses_max, axis=1))
        responses_pref_spf = responses_max[pref_spf, :] # (n_ori)

        # get oristats
        if np.sum(responses_pref_spf ** 2) == 0:
            pref_oris[c] = np.nan
            orth_oris[c] = np.nan
            osis[c] = 0
            gosis[c] = 0
            ori_ranges[c] = np.nan
        else:
            # preferred orientation
            pref_ori = np.argmax(responses_pref_spf)
            pref_oris[c] = pref_ori

            # orthogonal orientation
            orth_ori = pref_ori + n_ori // 2
            if orth_ori >= n_ori:
                orth_ori -= n_ori
            orth_oris[c] = orth_ori

            # OSI
            osi = (responses_pref_spf[pref_ori] - responses_pref_spf[orth_ori]) / \
                    (responses_pref_spf[pref_ori] + responses_pref_spf[orth_ori])
            osis[c] = osi

            # gOSI
            gosi = np.sqrt((np.sum(responses_pref_spf * np.sin(2 * np.pi / n_ori * np.arange(n_ori)))) ** 2 +
                        (np.sum(responses_pref_spf * np.cos(2 * np.pi / n_ori * np.arange(n_ori)))) ** 2) / \
                        np.sum(responses_pref_spf)
            gosis[c] = gosi

            # pref. oris wrt. different SPFs
            responses_vs_spf = np.max(responses_max, axis=1) # (n_spf)
            th = (np.max(responses_vs_spf) + np.min(responses_vs_spf)) / 2
            _responses_max = responses_max[responses_vs_spf > th, :]
            pref_degs = np.argmax(_responses_max, axis=1) * 180 / n_ori # unit: deg

            # bandwidth of pref_oris
            pref_oris_aug = np.sort(np.hstack((pref_degs, pref_degs+180)))
            pref_oris_aug_diff = np.diff(pref_oris_aug)
            pref_oris_diff_idx = np.argmax(pref_oris_aug_diff)
            if pref_oris_aug[pref_oris_diff_idx + 1] < 180:
                ori_ranges[c] = 180 - np.max(pref_oris_aug_diff)
            elif pref_oris_aug[pref_oris_diff_idx + 1] >= 180:
                ori_ranges[c] = np.max(pref_degs) - np.min(pref_degs)

        # plot the tuning curve [if necessary]
        if c < n_plot:
            plt.plot(responses_pref_spf)
            plt.show()
            print(osis[c], gosis[c])

    return pref_oris, osis, gosis, ori_ranges

if __name__ == '__main__':
    # parameter
    images = 'cifar10' # 'cifar10' or 'imagenet'
    K.set_learning_phase(0)
    np.random.seed(seed=0)

    # path
    model_path = 'saved_models/model1/cifar10_cnn.hdf5'
    output_path = 'results/model1/'

    # grating parameters
    n_spf = 9 # 9 or 57
    n_ori = 12
    n_phase = 4

    # create directories to save the results
    if not os.path.exists(output_path + 'stim'):
        os.makedirs(output_path + 'stims')
        os.makedirs(output_path + 'ori_tuning')

    if images == 'cifar10':
        input_shape = (32, 32, 1)

        # load dataset
        x_train, x_test, y_train_cat, y_test_cat, x_train_mean = prepare_train_test()

        # load pretrained model
        model = create_model(x_train.shape[1:], num_classes=10)
        model.load_weights(model_path)
        analyzed_layers = ['layer' + str(i) for i in np.arange(1, 7, 1)] + ['fc1']
    elif images == 'imagenet':
        input_shape = (224, 224, 3)

        # dataset stats
        x_train_mean = np.zeros(input_shape)
        for c, m in zip(range(3), [103.939, 116.779, 123.68]): # 'BGR' order
            x_train_mean[:, :, c] = m / 255.0 # temporarily divide by 255

        # load pretrained model
        model = vgg16.VGG16(weights='imagenet')
        analyzed_layers = ['block1_conv1', 'block1_conv2', 'block1_pool',
                           'block2_conv1', 'block2_conv2', 'block2_pool',
                           'block3_conv1', 'block3_conv2',
                           'block3_conv3', 'block3_pool',
                           'block4_conv1', 'block4_conv2',
                           'block4_conv3', 'block4_pool',
                           'block5_conv1', 'block5_conv2',
                           'block5_conv3', 'block5_pool',
                           'fc1', 'fc2']

    # set model attributes
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    n_cells_dict = {}
    for layer in model.layers:
        n_cells_dict[layer.name] = layer.output_shape[-1]

    # create grating set
    print('create grating set')
    grating_all, ctrl_grating_all = create_gratings(n_spf, n_ori, n_phase,
                                                    input_shape, x_train_mean,
                                                    plot=True,
                                                    output_path=output_path)

    if images == 'imagenet':
        grating_all *= 255
        ctrl_grating_all *= 255

    # analyze tuning
    print('analyze tuning')
    pref_ori_all = []
    osi_all = []
    gosi_all = []
    pref_spf_all = []
    ori_range_all = []
    for layer_name in analyzed_layers:
        # basics
        n_cells = n_cells_dict[layer_name]
        func = get_response_function(layer_name)

        # get response wrt. grating and ctrl grating
        if images == 'cifar10':
            gt_responses = func([grating_all])[0]
            ctrl_responses = func([ctrl_grating_all])[0]
        elif images == 'imagenet':
            # stack by 100-binned batches to avoid memory error
            gt_responses = np.zeros((len(grating_all), n_cells))
            for n in range(int(np.ceil(len(grating_all) / 100))):
                idx_min = n * 100
                idx_max = min((n+1) * 100, len(grating_all))
                gt_responses[idx_min: idx_max] = func([grating_all[idx_min: idx_max]])[0]
            ctrl_responses = np.zeros((len(ctrl_grating_all), n_cells))
            for n in range(int(np.ceil(len(ctrl_grating_all) / 100))):
                idx_min = n * 100
                idx_max = min((n+1) * 100, len(ctrl_grating_all))
                ctrl_responses[idx_min: idx_max] = func([ctrl_grating_all[idx_min: idx_max]])[0]

        gt_responses_all = np.zeros((n_spf, n_ori, n_phase, n_cells))
        for s in range(n_spf):
            for o in range(n_ori):
                gt_responses_all[s, o, :, :] = gt_responses[(s * n_ori + o) * n_phase:(s * n_ori + o + 1) * n_phase, :]
        ctrl_responses_all = np.zeros((n_spf, n_ori, n_phase, n_cells))
        for s in range(n_spf):
            for o in range(n_ori):
                ctrl_responses_all[s, o, :, :] = ctrl_responses[(s * n_ori + o) * n_phase:(s * n_ori + o + 1) * n_phase, :]

        # get oristats
        pref_oris, osis, gosis, ori_ranges = \
                            get_oristats(n_cells, gt_responses_all, n_plot=0)
        ctrl_pref_oris, ctrl_osis, ctrl_gosis, ctrl_ori_ranges = \
                            get_oristats(n_cells, ctrl_responses_all, n_plot=0)
        pref_ori_all.append(pref_oris)
        osi_all.append(osis)
        gosi_all.append(gosis)
        ori_range_all.append(ori_ranges)

        # visualize the OSI distribution
        fig = plt.figure(figsize=(10, 4))
        ax = plt.subplot(1, 2, 1)
        ax.hist(osis, alpha=0.5, color='r', range=(0, 1), bins=10)
        ax.hist(ctrl_osis, alpha=0.5, color='b', range=(0, 1), bins=10)
        ax.legend(['grating', 'permutated'])
        ax.set_xlabel('OSI')
        ax.set_title(layer_name)
        ax = plt.subplot(1, 2, 2)
        ax.hist(gosis, alpha=0.5, color='r', range=(0, 1), bins=10)
        ax.hist(ctrl_gosis, alpha=0.5, color='b', range=(0, 1), bins=10)
        ax.legend(['grating', 'permutated'])
        ax.set_xlabel('gOSI')
        ax.set_title(layer_name)
        plt.tight_layout()
        plt.savefig(output_path + 'ori_tuning/dist_OSI_' + str(layer_name) + '.png')
        plt.savefig(output_path + 'ori_tuning/dist_OSI_' + str(layer_name) + '.pdf')
        plt.close()

        # visualize the ori_ranges distribution
        plt.hist(ori_ranges[gosis > 0.33], range=(0, 180), bins=n_ori, align='left')
        plt.xticks(np.arange(0, 180, 30), ha='center')
        plt.xlabel('Ori range')
        plt.ylabel('Count')
        plt.title(layer_name)
        plt.savefig(output_path + 'ori_tuning/ori_ranges_' + str(layer_name) + '.png')
        plt.savefig(output_path + 'ori_tuning/ori_ranges_' + str(layer_name) + '.pdf')
        plt.close()

        # plot 1D tuning
        n_plot = int(np.ceil(n_cells / 100))
        for n in range(n_plot):
            fig = plt.figure(figsize=(20, 20))
            for i in range(min(100, n_cells - n * 100)):
                idx = 100 * n + i

                # get max along the phase axis
                responses_max = np.max(gt_responses_all[:,:,:,idx], axis=2)
                pref_spf = np.argmax(np.max(responses_max, axis=1))
                responses_pref_spf = responses_max[pref_spf, :] # (n_ori)

                # plot tuning curve for the cell
                ax = plt.subplot(10, 10, i+1)
                ax.plot(responses_pref_spf)
                ax.set_xticks(np.arange(0, 13, 6))
                ax.set_xticklabels(np.arange(0, 210, 90))
                plt.title('cell {}'.format(idx))

            plt.tight_layout()
            plt.savefig(output_path + 'ori_tuning/tuning_curve_' + layer_name + \
                        '_' + str(n) + '.png')
            plt.savefig(output_path + 'ori_tuning/tuning_curve_' + layer_name + \
                        '_' + str(n) + '.pdf')
            plt.close()

    pickle.dump(pref_ori_all, open(output_path + 'pref_ori_all.pkl', 'wb'), 2)
    pickle.dump(osi_all, open(output_path + 'osi_all.pkl', 'wb'), 2)
    pickle.dump(gosi_all, open(output_path + 'gosi_all.pkl', 'wb'), 2)
    pickle.dump(ori_range_all, open(output_path + 'ori_range_all.pkl', 'wb'), 2)
