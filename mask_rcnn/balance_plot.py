import numpy as np
from matplotlib import pyplot as plt


def balance_plot(*labels_and_dataloaders, filename: str):
    dataloader_labels = labels_and_dataloaders[::2]
    assert all(isinstance(lbl, str) for lbl in dataloader_labels)
    dataloaders = labels_and_dataloaders[1::2]
    if any(dl is None for dl in dataloaders):
        return

    plt.figure()
    labels = list(range(1, 11))
    off = len(dataloaders)
    for i, (dl, dl_lbl) in enumerate(zip(dataloaders, dataloader_labels)):
        all_dl_labels = []
        n_images = sum(len(b['target']) for b in dl)
        for batch in dl:
            tgt = batch['target']
            batch_labels = np.concatenate([t['labels'].detach().cpu().numpy().flatten() for t in tgt])
            all_dl_labels = np.concatenate([batch_labels, all_dl_labels])
        n_labels = len(all_dl_labels)

        y = [
            np.sum(all_dl_labels == lbl)*100./n_labels
            for lbl in labels
        ]
        x = np.multiply(labels, off) + i
        plt.bar(x, y, label=f'{dl_lbl}/img={n_images}/ann={n_labels}')

    plt.legend()
    plt.ylabel('Dataset Contribution [%]')
    plt.xticks(np.multiply(labels, off) + len(dataloaders)/2. - 0.5, [str(lbl) for lbl in labels])
    plt.xlabel('Class ID')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
