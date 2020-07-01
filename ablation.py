import os
from tqdm import tqdm
import pickle

import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from keras.layers import Input
from keras import backend as K
from keras.models import Model

from model import create_model
from utils import prepare_train_test

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['font.size'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True

def ablation(x, layer_name, unit_idxs, print_summary=False):
    """
    x: stimulus images fed into the network (x_test)
    layer_name: layer name to perform ablation
    unit_idxs (list): unit indices to perform ablation
    """

    model_ori = create_model(x_test.shape[1:], num_classes=10)
    model_ori.load_weights(model_path)
    layer_dict = dict([(layer.name, layer) for layer in model_ori.layers])

    layer_idx = np.where(np.array(model_ori.layers) == model_ori.get_layer(layer_name))[0][0]

    # model1: input -> target layer
    layer_output = layer_dict[layer_name].output
    model1 = Model(outputs=layer_output, inputs=model_ori.input)
    if print_summary:
        print(model1.summary())

    # feed the images
    responses = model1.predict(x)

    # ablation on the target layer
    responses_ablated = np.copy(responses)
    if unit_idxs != []:
        if layer_name == 'bn7':
            responses_ablated[:, unit_idxs] = 0
        else:
            responses_ablated[:, :, :, unit_idxs] = 0

    # create a copy of the model
    model_copy = keras.models.clone_model(model_ori)
    model_copy.set_weights(model_ori.get_weights())

    # model2: target layer -> output layer
    model2_layers = model_copy.layers[layer_idx+1:]
    inputs2 = Input(shape=responses.shape[1:])
    for l, layer in enumerate(model2_layers):
        if l == 0:
            x = layer(inputs2)
        else:
            x = layer(x)
    model2 = Model(outputs=x, inputs=inputs2)
    if print_summary:
        print(model2.summary())

    # feed the ablated activations of the target layer
    outputs = model2.predict(responses_ablated)

    K.clear_session()
    return outputs

def cross_entropy(target, output):
    output /= np.sum(output, axis=len(output.shape) - 1, keepdims=True)
    eps = K.epsilon()
    output = np.clip(output, eps, 1 - eps)
    ce = -np.sum(target * np.log(output), axis=len(output.shape) - 1)
    return ce

def get_original_loss(x):
    model_ori = create_model(x_test.shape[1:], num_classes=10)
    model_ori.load_weights(model_path)

    # calculate the loss
    loss_original = cross_entropy(y_test_cat, model_ori.predict(x))

    K.clear_session()
    return loss_original

def ablation_selective(x, selectivity_all, percentile):
    """
    ablate cells with top-PERCENTILE% selectivity
    ctrl: ablate cells with bottom-PERCENTILE% selectivity
    """
    # get original loss
    loss_original = get_original_loss(x)

    impact_all = np.zeros((len(analyzed_layers), len(x)))
    impact_ctrl_all = np.zeros((len(analyzed_layers), len(x)))
    for l in tqdm(range(len(analyzed_layers))):
        layer_name = analyzed_layers[l]
        n_cells = n_cells_dict[layer_name]

        # selectivity
        selectivity = selectivity_all[l]

        # perform ablation on highly ori-tuned neurons
        th = np.percentile(selectivity, 100 - percentile)
        unit_idxs = np.where(selectivity > th)[0]
        outputs_ablation = ablation(x, layer_name, unit_idxs=unit_idxs)

        # calculate the loss
        loss_ablation = cross_entropy(y_test_cat, outputs_ablation)
        impact_all[l] = loss_ablation - loss_original

        # perform ctrl ablation
        th = np.percentile(selectivity, percentile)
        unit_idxs = np.where(selectivity < th)[0]
        outputs_ablation_ctrl = ablation(x, layer_name, unit_idxs=unit_idxs)

        # calculate the ctrl loss
        loss_ablation_ctrl = cross_entropy(y_test_cat, outputs_ablation_ctrl)
        impact_ctrl_all[l] = loss_ablation_ctrl - loss_original

    return impact_all, impact_ctrl_all

def plot_impact_tuned(impact_all, impact_ctrl_all, savename):
    fig = plt.figure(figsize=(10, 4))
    ax = plt.subplot(1, 2, 1)
    yerr = np.std(impact_all, axis=-1) / np.sqrt(impact_all.shape[1])
    ax.errorbar(np.arange(len(analyzed_layers)),
                np.mean(impact_all, axis=-1), yerr=yerr, c='r')
    yerr = np.std(impact_ctrl_all, axis=-1) / np.sqrt(impact_ctrl_all.shape[1])
    ax.errorbar(np.arange(len(analyzed_layers)),
                np.mean(impact_ctrl_all, axis=-1), yerr=yerr, c='b')
    ax.set_xlabel('layer')
    ax.set_ylabel('impact on test loss')
    ax.set_xticks(np.arange(len(analyzed_layers)))
    ax.set_xticklabels(analyzed_layers)
    plt.tight_layout()
    plt.savefig(output_path + 'ablation/' + savename + '.png')
    plt.savefig(output_path + 'ablation/' + savename + '.pdf')
    plt.close()

if __name__ == '__main__':
    # parameter
    input_shape = (32, 32, 1)
    K.set_learning_phase(0)
    np.random.seed(seed=0)

    # path
    model_path = 'saved_models/model1/cifar10_cnn.hdf5'
    output_path = 'results/model1/'

    # grating parameters
    n_spf = 9
    n_ori = 12
    n_phase = 4

    # create directories to save the results
    if not os.path.exists(output_path + 'ablation'):
        os.makedirs(output_path + 'ablation')

    # load dataset
    x_train, x_test, y_train_cat, y_test_cat, x_train_mean = prepare_train_test()

    # load the trained model
    model = create_model(x_test.shape[1:], num_classes=10)
    model.load_weights(model_path)

    analyzed_layers = ['bn' + str(i) for i in np.arange(1, 8, 1)]

    # load selectivity
    gosi_all = pickle.load(open(output_path + 'gosi_all.pkl', 'rb'))

    # set model attributes
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    n_cells_dict = {}
    for layer in model.layers:
        n_cells_dict[layer.name] = layer.output_shape[-1]

    # ablate units with high or low orientation selectivity
    print('perform ablation')
    impact_all, impact_ctrl_all = ablation_selective(x_test, gosi_all, 50)
    plot_impact_tuned(impact_all, impact_ctrl_all, 'gOSI_50')
