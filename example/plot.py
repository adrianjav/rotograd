import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


@torch.no_grad()
def plot_logistic_regression(rep, tgt, task_model, above, xmin=None, xmax=None, alpha=1, ax=None, sigmoid=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    tgt = tgt.clone()
    tgt += .1 * (tgt == 1.0)
    tgt -= .1 * (tgt == 0.0)

    ax.scatter(rep, tgt, color='black', zorder=20, alpha=alpha)

    xmin = min(rep) if xmin is None else min(xmin, min(rep))
    xmax = max(rep) if xmax is None else max(xmax, max(rep))
    xmin, xmax = -50, 50
    # print('xmin', xmin, 'xmax', xmax)

    X_test = np.linspace(xmin, xmax, 300)
    X_test = X_test[..., np.newaxis]

    vy = np.array([[2, 2.]]) - np.array([[-2., -2]])
    # X_test =

    X_test_p = torch.from_numpy((X_test * vy).astype(np.float32)).view((300, 2))
    Y_test_p = task_model(X_test_p)
    loss = Y_test_p.flatten().numpy()

    # print('loss', loss[:10])

    # if above:
    #     ax.xaxis.set_ticks_position('none')
    #     ax.xaxis.set_ticklabels([])

    if sigmoid:
        ax.plot(X_test, torch.sigmoid(X_test_p).numpy(), linewidth=3)
        ax.axvline(0, color='blue')

    ax.plot(X_test, loss, linewidth=3) #, linestyle='dashed')
    x_mean = Y_test_p[(np.abs(loss - 0.5)).argmin()].item()
    ax.axvline(x_mean, color='orange' if sigmoid else 'blue')

    # ax.set_xlabel('Input')
    # ax.set_ylabel('Probability')
    ax.set_ylabel('Cluster 1' if not above else 'Cluster 2', fontsize=14)
    ax.set_xlim(xmin, xmax)
    sns.despine()

    return fig, ax


@torch.no_grad()
def plot2d(X, y, title=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if title:
        ax.set_title(title)

    num_to_draw = 200 # we will only draw a small number of points to avoid clutter
    x_draw = X[:num_to_draw]
    y_draw = y[:num_to_draw]

    X_s_0 = x_draw
    y_s_0 = y_draw
    ax.scatter(X_s_0[y_s_0==1.0][:, 0], X_s_0[y_s_0==1.0][:, 1], color='green', marker='x', s=30, linewidth=1.5, label= "Prot. +ve")
    ax.scatter(X_s_0[y_s_0==0.0][:, 0], X_s_0[y_s_0==0.0][:, 1], color='red', marker='x', s=30, linewidth=1.5, label = "Prot. -ve")

    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off') # dont need the ticks to see the data distribution
    ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
#     plt.legend(loc=2, fontsize=15)
    ax.set_xlim((-15,10))
    ax.set_ylim((-10,15))
#     plt.savefig("data.png")
#     plt.show()


def setup_grid(range_lim, n_pts):
    x = torch.linspace(range_lim[0][0], range_lim[0][1], n_pts)
    y = torch.linspace(range_lim[1][0], range_lim[1][1], n_pts)

    xx, yy = torch.meshgrid((x, y))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    return xx, yy, zz


def format_ax(ax, lims):
    # if lims is not None:
    #     ax.set_xlim(*lims[0])
    #     ax.set_ylim(*lims[1])
    ax.grid()


def plot(ax, x, y, z, optimum, title, levels=20):
    ax.contour(x, y, z.reshape((x.shape[0], y.shape[0])), levels=levels)
    for optimal in optimum:
        ax.scatter(*optimal, marker='*', s=100)
    ax.set_title(title, size=18)
    return ax


@torch.no_grad()
def plot_toy(grid, model, tasks, zs, step, lims, show=(1, 2, 3), levels=range(0, 100, 5)):
    optimum = [[1.5, 1.5], [-1.5, -1.5]]
    # optimum = [[-1., 0], [1., 0]]

    fig, axs = plt.subplots(len(show), 2, figsize=(12, 6 * len(show)), subplot_kw={'aspect': 'equal'})

    values = (
        tasks[0].loss(grid[2], 0),
        tasks[1].loss(grid[2], 0)
    )

    rotated_values = (
        tasks[0].loss(model.heads[0][0].rotate(grid[2]), 0),
        tasks[1].loss(model.heads[1][0].rotate(grid[2]), 0)
    )

    rotated_optimum = (
        model.heads[0][0].rotate_back(torch.tensor(optimum[0:1])).squeeze(),
        model.heads[1][0].rotate_back(torch.tensor(optimum[1:2])).squeeze(),
    )

    # g1t, g2t = -grads[0], -grads[1]
    # g1t, g2t = -2e1*model[0].task_grad, -2e1*model[1].task_grad

    zt, z1t, z2t = zs

    # zt = torch.cat(zt, dim=0)
    # z1t = torch.cat(z1t, dim=0)
    # z2t = torch.cat(z2t, dim=0)
    ztt = torch.stack(zt, dim=1)
    z1tt = torch.stack(z1t, dim=1)
    z2tt = torch.stack(z2t, dim=1)

    zt = ztt.mean(dim=0)
    z1t = z1tt.mean(dim=0)
    z2t = z2tt.mean(dim=0)

    format_ax(axs[0, 0], lims)
    plot(axs[0, 0], grid[0], grid[1], values[0], optimum[0:1], title=r'$f_1(Z_1)$ in $Z_1$', levels=levels)
    axs[0, 0].plot(z1t[:, 0], z1t[:, 1], linewidth=3, c='b')
    axs[0, 0].scatter(z1t[:1, 0], z1t[:1, 1], s=100, marker='s', c='b')
    axs[0, 0].scatter(z1t[-2:, 0], z1t[-2:, 1], s=100, marker='o', c='b')

    # axs[0, 0].scatter(*z1tt[:10, -1].unbind(dim=-1))
    # for i in range(10):
    #     axs[0, 0].arrow(*z1tt[i, -1].unbind(dim=-1), *g1t[i], width=0.002)

    format_ax(axs[0, 1], lims)
    plot(axs[0, 1], grid[0], grid[1], values[1], optimum[1:2], title=r'$f_2(Z_2)$ in $Z_2$', levels=levels)
    axs[0, 1].plot(z2t[:, 0], z2t[:, 1], linewidth=3, c='g')
    axs[0, 1].scatter(z2t[:1, 0], z2t[:1, 1], s=100, marker='s', c='g')
    axs[0, 1].scatter(z2t[-2:, 0], z2t[-2:, 1], s=100, marker='o', c='g')

    # axs[0, 1].scatter(*z2tt[:10, - 1].unbind(dim=-1))
    # for i in range(10):
    #     axs[0, 1].arrow(*z2tt[i, -1].unbind(dim=-1), *g2t[i], width=0.002)

    format_ax(axs[1, 0], lims)
    plot(axs[1, 0], grid[0], grid[1], rotated_values[0], rotated_optimum[0:1],
         title=f'$f_1(R_1^{{{step + 1}}}Z + d_1^{{{step + 1}}})$ in $Z$', levels=levels)
    axs[1, 0].plot(zt[:, 0], zt[:, 1], linewidth=3, c='b')
    axs[1, 0].scatter(zt[:1, 0], zt[:1, 1], s=100, marker='s', c='b')
    axs[1, 0].scatter(zt[-2:, 0], zt[-2:, 1], s=100, marker='o', c='b')

    format_ax(axs[1, 1], lims)
    plot(axs[1, 1], grid[0], grid[1], rotated_values[1], rotated_optimum[1:2],
         title=f'$f_2(R_2^{{{step + 1}}}Z + d_2^{{{step + 1}}})$ in $Z$', levels=levels)
    axs[1, 1].plot(zt[:, 0], zt[:, 1], linewidth=3, c='g')
    axs[1, 1].scatter(zt[:1, 0], zt[:1, 1], s=100, marker='s', c='g')
    axs[1, 1].scatter(zt[-2:, 0], zt[-2:, 1], s=100, marker='o', c='g')

    format_ax(axs[2, 0], lims)
    plot(axs[2, 0], grid[0], grid[1], tasks[0].weight * values[0] + tasks[1].weight * values[1], optimum,
         title=f'${tasks[0].weight:.2f} f_1(Z_1) + {tasks[1].weight:.2f} f_2(Z_2)$ in $Z_1\\cong Z_2$', levels=levels)
    axs[2, 0].plot(z1t[:, 0], z1t[:, 1], linewidth=3, c='b')
    axs[2, 0].plot(z2t[:, 0], z2t[:, 1], linewidth=3, c='g')
    axs[2, 0].scatter(*z1t[0], marker='s', s=100, c='b')
    axs[2, 0].scatter(*z2t[0], marker='s', s=100, c='g')
    axs[2, 0].scatter(*z1t[step], marker='o', s=100, c='b')
    axs[2, 0].scatter(*z2t[step], marker='o', s=100, c='g')

    format_ax(axs[2, 1], lims)
    plot(axs[2, 1], grid[0], grid[1],
         tasks[0].weight * rotated_values[0] + tasks[1].weight * rotated_values[1], rotated_optimum,
         title=f'${tasks[0].weight:.2f} f_1(R_1^{{{step + 1}}}Z + d_1^{{{step + 1}}}) + {tasks[1].weight:.2f} f_2(R_2^{{{step + 1}}}Z + d_2^{{{step + 1}}})$ in $Z$',
         levels=levels)
    #     a = (rotated_optimum[0] + rotated_optimum[1])/2
    #     plot(axs[2,1], grid[0], grid[1], (rotated_values[0] + rotated_values[1])/2., a.unsqueeze(0), title='Weighted sum (Alt. view)')
    axs[2, 1].plot(zt[:, 0], zt[:, 1], linewidth=3, c='orange')
    axs[2, 1].scatter(*zt[0], marker='D', s=100, c='orange')

    fig.suptitle(f'Step {step}', size=20)

    return fig
