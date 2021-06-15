import matplotlib.pyplot as plt


def plot_augmentations(images, titles, sup_title):
    fig, axes = plt.subplots(figsize=(20, 16), nrows=3, ncols=4, squeeze=False)
    
    for indx, (img, title) in enumerate(zip(images, titles)):
        axes[indx // 4][indx % 4].imshow(img)
        axes[indx // 4][indx % 4].set_title(title, fontsize=15)
        
    plt.tight_layout()
    fig.suptitle(sup_title, fontsize = 20)
    fig.subplots_adjust(wspace=0.2, hspace=0.2, top=0.93)
    plt.show()