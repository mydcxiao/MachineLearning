import numpy as np
import matplotlib.pyplot as plt


def main():
    x1 = -1
    y1 = 1
    x2 = 1
    y2 = 1
    x3 = 0
    y3 = 0
    lx = np.linspace(-5, 5, 10)
    ly = 0.5 * np.ones((10, 1))
    ## first 
    fig, ax = plt.subplots(1)
    ax.plot(x1, y1, 'o', color='blue')
    ax.plot(x2, y2, 'o', color='blue')
    ax.plot(x3, y3, 'o', color='red')
    ax.plot(lx, ly)

    xmin, xmax, ymin, ymax = -3, 3, -1, 3
    ticks_frequency = 1
    ax.set(xlim=(xmin-1, xmax+1), ylim=(ymin-1, ymax+1), aspect='equal')

    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create 'x' and 'y' labels placed at the end of the axes
    ax.set_xlabel('x1', size=14, labelpad=-24, x=1.03)
    ax.set_ylabel('x2', size=14, labelpad=-21, y=1.02, rotation=0)

    # Create custom major ticks to determine position of tick labels
    x_ticks = np.arange(xmin, xmax+1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax+1, ticks_frequency)
    ax.set_xticks(x_ticks[x_ticks != 0])
    ax.set_yticks(y_ticks[y_ticks != 0])

    # Draw arrows
    arrow_fmt = dict(markersize=4, color='black', clip_on=False)
    ax.plot((1), (0), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
    ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)
    plt.savefig("P1-2.png")
    plt.close()

    #second
    fig, ax = plt.subplots(1)
    ax.plot(x1, y1, 'o', color='blue')
    ax.plot(x2, y2, 'o', color='blue')
    ax.plot(x3, y3, 'o', color='red')
    ax.plot(x1, y1, marker='o', ms=20, mfc='None', mec='orange', label='support vector', linestyle='none')
    ax.plot(x2, y2, 'o', ms=20, mfc='None', mec='orange')
    ax.plot(x3, y3, 'o', ms=20, mfc='None', mec='orange')
    ax.plot(lx, ly, label="decision boundary")

    xmin, xmax, ymin, ymax = -3, 3, -1, 3
    ticks_frequency = 1
    ax.set(xlim=(xmin-1, xmax+1), ylim=(ymin-1, ymax+1), aspect='equal')

    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create 'x' and 'y' labels placed at the end of the axes
    ax.set_xlabel('x1', size=14, labelpad=-24, x=1.03)
    ax.set_ylabel('x2', size=14, labelpad=-21, y=1.02, rotation=0)

    # Create custom major ticks to determine position of tick labels
    x_ticks = np.arange(xmin, xmax+1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax+1, ticks_frequency)
    ax.set_xticks(x_ticks[x_ticks != 0])
    ax.set_yticks(y_ticks[y_ticks != 0])

    # Draw arrows
    arrow_fmt = dict(markersize=4, color='black', clip_on=False)
    ax.plot((1), (0), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
    ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)

    ax.legend(borderpad=0.8)
    # plt.show()
    plt.savefig("P1-6-1.png", bbox_inches="tight")
    plt.close()

    #third
    y1, y2, y3 = 0, 0, 0
    lx = np.linspace(-5, 5, 100)
    ly = -2 * lx**2 + 1
    dx1 = -np.sqrt(1/2)
    dx2 = np.sqrt(1/2) 
    fig, ax = plt.subplots(1)
    ax.plot(x1, y1, 'o', color='blue')
    ax.plot(x2, y2, 'o', color='blue')
    ax.plot(x3, y3, 'o', color='red')
    ax.plot(lx, ly, '--', label="non-linear decision boundary")
    ax.plot(dx1, 0, 'o', label="1-d decision boundary", color='green')
    ax.plot(dx2, 0, 'o', color='green')
    

    xmin, xmax, ymin, ymax = -3, 3, -1, 2
    ticks_frequency = 1
    ax.set(xlim=(xmin-1, xmax+1), ylim=(ymin-1, ymax+1), aspect='equal')

    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Create 'x' and 'y' labels placed at the end of the axes
    ax.set_xlabel('x', size=14, labelpad=-24, x=1.03)
    # ax.set_ylabel('x2', size=14, labelpad=-21, y=1.02, rotation=0)

    # Create custom major ticks to determine position of tick labels
    x_ticks = np.arange(xmin, xmax+1, ticks_frequency)
    # y_ticks = np.arange(ymin, ymax+1, ticks_frequency)
    ax.set_xticks(x_ticks[x_ticks != 0])
    # ax.set_yticks(y_ticks[y_ticks != 0])
    ax.get_yaxis().set_visible(False)

    # Draw arrows
    arrow_fmt = dict(markersize=4, color='black', clip_on=False)
    ax.plot((1), (0), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
    # ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)

    ax.legend(borderpad=0.8)
    # plt.show()
    plt.savefig("P1-6-2.png", bbox_inches="tight")

if __name__ == '__main__':
    main()
