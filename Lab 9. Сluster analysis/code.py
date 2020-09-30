import matplotlib.pyplot as plt

prev = ''
updated = 0


def manhetten(t1, t2):
    return abs(t2[0] - t1[0]) + abs(t2[1] - t1[1])


def chebishev(t1, t2):
    return max(abs(t2[0] - t1[0]), abs(t2[1] - t1[1]))


def my_input():
    x_arr = [
        [143, 213]
    ]

    y_arr = [
    ]

    return x_arr, y_arr


def onclick(event):
    global prev, y_arr, x_arr, updated
    print('X{}: ({}, {})'.format(len(x_arr) + 1, round(event.xdata, 2), round(event.ydata, 2)))
    updated = 1
    if event.dblclick:
        i = x_arr.index(prev)
        del x_arr[i]
        print('Y{}: ({}, {})'.format(len(y_arr) + 1, round(event.xdata, 2), round(event.ydata, 2)))
        y_arr.append([event.xdata, event.ydata])
    else:
        prev = [event.xdata, event.ydata]
        x_arr.append([event.xdata, event.ydata])


def distance(x_arr, y_arr):
    ans = {}
    for x in x_arr:
        for y in y_arr:
            ans[str([x, y])] = chebishev(x, y)
    return ans


def mapping(x_arr, y_arr):
    global updated
    ans = distance(x_arr, y_arr)
    x_arr1 = {}
    y_arr1 = {}
    for x in x_arr:
        m = -1
        r = 11111111
        for y in y_arr:
            if updated:
                print('X{}-Y{}: {}'.format(x_arr.index(x), y_arr.index(y), round(ans[str([x, y])], 3)))
            if ans[str([x, y])] < r:
                r = ans[str([x, y])]
                m = y
        x_arr1[str(x)] = m
        try:
            y_arr1[str(m)].append(x)
        except KeyError:
            y_arr1[str(m)] = [x]
    updated = 0
    return x_arr1, y_arr1


def epoch(x_arr, y_arr):
    x_arr1, y_arr1 = mapping(x_arr, y_arr)
    new_y_arr = []
    for x_of_y in y_arr1:
        x_coords_x = [x[0] for x in y_arr1[x_of_y]]
        sred_x = sum(x_coords_x) / len(x_coords_x)
        x_coords_y = [x[1] for x in y_arr1[x_of_y]]
        sred_y = sum(x_coords_y) / len(x_coords_y)
        new_y_arr.append([sred_x, sred_y])
    return new_y_arr


x_arr, y_arr = my_input()


if __name__ == "__main__":
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    while 1:
        new_y_arr = epoch(x_arr, y_arr)
        if new_y_arr != y_arr:
            updated=1
            print('Old Y array:', [[round(k, 3) for k in y] for y in y_arr])
            print('New Y array:', 'Y = ('+'),Y = ('.join([','.join([str(round(k, 3)) for k in y]) for y in new_y_arr]))
            print('X array: ', ';\n'.join(
                ['X{}=('.format(x_arr.index(x)) + ','.join([str(round(x1, 2)) for x1 in x]) + '); ' for x in
                 x_arr]))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        for y, templ in zip(new_y_arr, colors):
            plt.plot(y[0], y[1], templ + 'v')
        x_arr1, y_arr1 = mapping(x_arr, new_y_arr)
        for dot_y, templ in zip(y_arr1, colors):
            [plt.plot(x[0], x[1], templ + 'o') for x in y_arr1[dot_y]]
        fig.canvas.draw()
        y_arr = new_y_arr
        plt.pause(0.01)
        fig.clf()
