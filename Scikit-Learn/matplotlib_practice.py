# encoding=utf-8

import numpy as np
import matplotlib.pyplot as plt


def main():
    # line

    # x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    # c, s = np.cos(x), np.sin(x)
    # plt.figure(1)
    # # plt.plot(x, c)
    # plt.plot(x, c, color="blue", linewidth=1.0, linestyle="-", label="COS", alpha=0.5)
    # plt.plot(x, s, "r*", label="SIN")
    # plt.title("COS & SN")
    # # 轴的编辑器
    # ax = plt.gca()
    # ax.spines["right"].set_color("none")
    # ax.spines["top"].set_color("none")
    # ax.spines["left"].set_position(("data", 0))
    # ax.spines["bottom"].set_position(("data", 0))
    # ax.xaxis.set_ticks_position("bottom")
    # ax.yaxis.set_ticks_position("left")
    # plt.xticks([-np.pi, -np.pi / 2.0, np.pi / 2, np.pi],
    #            [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
    # plt.yticks(np.linspace(-1, 1, 5, endpoint=True))
    # for label in ax.get_xticklabels() + ax.get_yticklabels():
    #     label.set_fontsize(16)
    #     label.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.2))
    # plt.legend(loc="upper left")
    # plt.grid()
    # # plt.axis([-1, 1, -0.5, 1])
    # plt.fill_between(x, np.abs(x) < 0.5, c, c > 0.5, color="green", alpha=0.5)
    # t = 1
    # plt.plot([t, t], [0, np.cos(t)], "y", linewidth=3, linestyle="--")
    # # 加注释 +是相对位置
    # plt.annotate("cos(1)", xy=(t, np.cos(1)), xycoords="data", xytext=(+10, +30),
    #              textcoords="offset points", arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    # plt.show()

    # scatter 散点图
    fig = plt.figure()

    ax = fig.add_subplot(3, 3, 1)
    n = 128
    X = np.random.normal(0, 1, n)
    Y = np.random.normal(0, 1, n)
    T = np.arctan2(Y, X)
    # plt.axes([0.025, 0.025, 0.95, 0.95])

    ax.scatter(X, Y, s=75, c=T, alpha=.5)
    plt.xlim(-1.5, 1.5), plt.xticks([])
    plt.ylim(-1.5, 1.5), plt.yticks([])
    plt.axis()
    plt.title("scatter")
    plt.xlabel("x")
    plt.ylabel("y")

    # bar 柱状图
    ax = fig.add_subplot(3, 3, 2)
    n = 10
    X = np.arange(n)
    Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
    Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

    ax.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
    ax.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
    for x, y in zip(X, Y1):
        plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')
    for x,y in zip(X, Y2):
        plt.text(x + 0.4, -y - 0.05, '%.2f' % y, ha='center', va='top')
    # Pie 饼图
    fig.add_subplot(333)
    n = 20
    Z = np.ones(n)
    Z[-1] *= 2
    plt.pie(Z, explode=Z * .05, colors=['%f' % (i / float(n)) for i in range(n)],
            labels=['%.2f' % (i / float(n)) for i in range(n)])
    # 成为正的圆形
    plt.gca().set_aspect('equal')
    plt.xticks([]), plt.yticks([])
    # polar 极坐标
    fig.add_subplot(334, polar=True)
    n = 20
    theta = np.arange(0.0, 2 * np.pi, 2 * np.pi / n)
    radii = 10 * np.random.rand(n)
    # plot 是画折线的
    # plt.plot(theta, radii)
    plt.polar(theta, radii)
    # heatmap 热图
    fig.add_subplot(335)
    from matplotlib import cm
    data = np.random.rand(3, 3)
    cmap = cm.Blues
    # imshow 插值方法画出来
    map = plt.imshow(data, interpolation='nearest', cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # 3D图
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(336, projection='3d')
    ax.scatter(1, 1, 3, s=100)

    # hot map 热力图
    fig.add_subplot(313)
    def f(x, y):
        return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
    n = 256
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, y)
    plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)
    plt.savefig("./fig.png")
    plt.show()

if __name__ == "__main__":
    main()
