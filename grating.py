import numpy as np
import matplotlib.pyplot as plt

def create_gratings(n_spf, n_ori, n_phase, input_shape, x_train_mean, plot=False,
                    output_path=''):
    # create various grating stimuli
    grating_all = np.zeros((n_spf * n_ori * n_phase, ) + input_shape[:-1] + (3, ))
    for s in range(n_spf):
        spf = np.pi / (input_shape[0] / 2 + 4) * (s + 1)
        for o in range(n_ori):
            ori = np.pi / n_ori * o
            for p in range(n_phase):
                phase = 2 * np.pi / n_phase * p

                # create a gray-scale grating
                gray_grating = create_grating(ori, phase, spf, input_shape[0], input_shape[1])

                # standardize the grating into [0, 1]
                gray_grating = (gray_grating - np.min(gray_grating)) / \
                               (np.max(gray_grating) - np.min(gray_grating))
                grating = np.zeros(input_shape[:-1] + (3,))
                for ch in range(3):
                    grating[:, :, ch] = gray_grating

                grating_all[(s * n_ori + o) * n_phase + p, :, :, :] = grating - x_train_mean

    # shuffled gratings
    ctrl_grating_all = grating_all[np.random.permutation(len(grating_all))]

    # plot
    if plot:
        for p in range(n_phase):
            fig = plt.figure(figsize=(10, 10))
            for s in range(n_spf):
                for o in range(n_ori):
                    ax = plt.subplot(n_spf, n_ori, s * n_ori + o + 1)
                    img = grating_all[(s * n_ori + o) * n_phase + p]
                    ax.imshow((img - np.min(img)) / (np.max(img) - np.min(img)))
                    plt.axis('off')
            plt.savefig(output_path + 'stims/grating_phase' + str(p) + '.png')
            plt.savefig(output_path + 'stims/grating_phase' + str(p) + '.pdf')
            plt.close()

        for p in range(n_phase):
            fig = plt.figure(figsize=(10, 10))
            for s in range(n_spf):
                for o in range(n_ori):
                    ax = plt.subplot(n_spf, n_ori, s * n_ori + o + 1)
                    img = ctrl_grating_all[(s * n_ori + o) * n_phase + p]
                    ax.imshow((img - np.min(img)) / (np.max(img) - np.min(img)))
                    plt.axis('off')
            plt.savefig(output_path + 'stims/grating_shuffled_phase' + str(p) + '.png')
            plt.savefig(output_path + 'stims/grating_shuffled_phase' + str(p) + '.pdf')
            plt.close()

    return grating_all, ctrl_grating_all

def create_grating(phi, tau, k, h, w):
    """
    phi, tau, k: Gabor parameters (ori, phase, SPF)
    h, w: shape parameters
    """

    gx, gy = np.ogrid[0:h, 0:w]
    gx -= h // 2
    gy -= w // 2

    rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    k_rot = np.dot(np.transpose(rot), np.array([k, 0]))
    grating = np.cos((k_rot[1] * gx + k_rot[0] * gy) + tau)
    return grating
