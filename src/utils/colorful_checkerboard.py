import numpy as np
from matplotlib import cm


def get_colorful(n_grid, img_size, cmap='tab20'):
    grid_size = img_size // n_grid

    num_colors = n_grid * n_grid

    hsv = cm.get_cmap(cmap, num_colors)
    panels = [np.ones((grid_size, grid_size, 3)) * np.array(hsv(i)[:3]) for i in range(num_colors)]

    np.random.shuffle(panels)

    img = np.ones((img_size, img_size, 3))

    for i in range(num_colors):    
        row = i // n_grid
        col = i % n_grid
        img[row*grid_size:(row+1)*grid_size, col*grid_size:(col+1)*grid_size] = panels[i]

    return img

if __name__ == '__main__':
    n_grid = 5
    img_size = 256

    img = get_colorful(n_grid, img_size)

    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()

    plt.imsave('colormap.png', img)
