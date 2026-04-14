import matplotlib.pyplot as plt

def show(img, title=""):
    plt.imshow(img, cmap='hot')
    plt.title(title)
    plt.show()