import numpy as np
from matplotlib import pyplot as plt


def set_seed(seed):
    np.random.seed(seed)


def save_to_npy(embeddings, model, information, tags):
    for tag, d_emb in zip(tags, embeddings):
        np.save(fr"cache/{information}_{model.replace('/', '__')}_{tag}.npy", d_emb)


def load_from_npy(model, information, tags):
    embeddings = []
    for tag in tags:
        embeddings.append(np.load(fr"cache/{information}_{model.replace('/', '__')}_{tag}.npy"))
    return embeddings


def save_to_txt(indices, tags, negative_tags):
    with open(f"cache/__indices_pos_{tags}_neg_{negative_tags}.txt", "w") as f:
        for item in indices:
            f.write(f"{item}\n")


def save_text_file(data, method, tag, model_name: str, metric: str):
    with open(f"cache/results_high_p_{method}_{tag}_{model_name.replace('/', '__')}_{metric}.txt", "w") as f:
        for item, value in data.items():
            f.write(f"{item}\t{value}\n")


def load_text_file(method, tag, model_name: str, metric: str):
    data = {}
    with open(f"cache/{method}_{tag}_{model_name}_{metric}.txt", "r") as f:
        for line in f:
            parts = line.strip().split('\t')
            data[parts[0]] = float(parts[1])
    return data


def plot_pcas(datasets: list, datasets_names):
    colors = plt.cm.viridis(np.linspace(0, 1, len(datasets)))

    # Create the plot
    plt.figure(figsize=(10, 6))

    for data, color in zip(datasets, colors):
        plt.scatter(data[:, 0], data[:, 1], color=color, alpha=0.5)

    plt.xlabel('Principal Component 1', fontsize=17)
    plt.ylabel('Principal Component 2', fontsize=17)
    plt.legend([name for name in datasets_names], fontsize=17)
    plt.grid(True)
    plt.show()


def plot_accuracies(datasets: list, average: bool, model: str, metric: str, radius_eball: float, radius_ecube: float):

    datasets_filenames_eball = [f"cache/eball_{dataset}_{model}_{metric}.txt" for dataset in datasets]
    datasets_filenames_ecube = [f"cache/ecube_{dataset}_{model}_{metric}.txt" for dataset in datasets]
    datasets_filenames_logreg = [f"cache/logreg_{dataset}_{model}_{metric}.txt" for dataset in datasets]
    datasets_filenames_svm = [f"cache/svm_{dataset}_{model}_{metric}.txt" for dataset in datasets]
    datasets_filenames_gmm = [f"cache/gmm_{dataset}_{model}_{metric}.txt" for dataset in datasets]

    data_eball = {}

    for i, file_path in enumerate(datasets_filenames_eball):
        with open(file_path, 'r') as file:
            for line in file:
                # Parse the line
                parts = line.strip().split('\t')
                xy, z = parts[0], float(parts[1])
                x, y = eval(xy)

                if x == float(radius_eball) or average == True:
                    if str(y) not in data_eball:
                        data_eball[str(y)] = []

                    if not average:
                        data_eball[str(y)].append(z)

                    else:
                        if len(data_eball[str(y)]) < i + 1:
                            data_eball[str(y)].append(z)
                        elif z > data_eball[str(y)][i]:
                            data_eball[str(y)][i] = z

    data_ecube = {}

    for i, file_path in enumerate(datasets_filenames_ecube):
        with open(file_path, 'r') as file:
            for line in file:
                # Parse the line
                parts = line.strip().split('\t')
                xy, z = parts[0], float(parts[1])
                x, y = eval(xy)

                if x == float(radius_ecube) or average == True:
                    if str(y) not in data_ecube:
                        data_ecube[str(y)] = []

                    if not average:
                        data_ecube[str(y)].append(z)

                    else:
                        if len(data_ecube[str(y)]) < i + 1:
                            data_ecube[str(y)].append(z)
                        elif z > data_ecube[str(y)][i]:
                            data_ecube[str(y)][i] = z

    data_logreg = {}

    for file_path in datasets_filenames_logreg:
        with open(file_path, 'r') as file:
            for line in file:
                # Parse the line
                parts = line.strip().split('\t')
                y, z = parts[0], float(parts[1])

                if y not in data_logreg:
                    data_logreg[y] = []
                data_logreg[y].append(z)

    data_svm = {}

    for file_path in datasets_filenames_svm:
        with open(file_path, 'r') as file:
            for line in file:
                # Parse the line
                parts = line.strip().split('\t')
                y, z = parts[0], float(parts[1])

                if y not in data_svm:
                    data_svm[y] = []
                data_svm[y].append(z)

    data_gmm = {}

    for file_path in datasets_filenames_gmm:
        with open(file_path, 'r') as file:
            for line in file:
                # Parse the line
                parts = line.strip().split('\t')
                y, z = parts[0], float(parts[1])

                if y not in data_gmm:
                    data_gmm[y] = []
                data_gmm[y].append(z)

    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    plt.figure(figsize=(10, 6))

    if average:
        averages_eball = {y: np.mean(zs) for y, zs in data_eball.items()}
        averages_ecube = {y: np.mean(zs) for y, zs in data_ecube.items()}
        averages_logreg = {y: np.mean(zs) for y, zs in data_logreg.items()}
        averages_svm = {y: np.mean(zs) for y, zs in data_svm.items()}
        averages_gmm = {y: np.mean(zs) for y, zs in data_gmm.items()}
        plt.plot(list(averages_eball.keys())[:9], list(averages_eball.values())[:9], marker='o', label='eball')
        plt.plot(list(averages_ecube.keys())[:9], list(averages_ecube.values())[:9], marker='o', label='ecube')
        plt.plot(list(averages_logreg.keys()), list(averages_logreg.values()), marker='o', label='logreg')
        plt.plot(list(averages_svm.keys()), list(averages_svm.values()), marker='o', label='svm')
        plt.plot(list(averages_gmm.keys()), list(averages_gmm.values()), marker='o', label='gmm')

        plt.title(f'Plot of Average Accuracies for different number of PCAs')
    else:
        averages_eball = {y: zs[0] for y, zs in data_eball.items()}
        averages_ecube = {y: zs[0] for y, zs in data_ecube.items()}
        averages_logreg = {y: zs[0] for y, zs in data_logreg.items()}
        averages_svm = {y: zs[0] for y, zs in data_svm.items()}
        averages_gmm = {y: zs[0] for y, zs in data_gmm.items()}
        plt.plot(list(averages_eball.keys()), list(averages_eball.values()), marker='o', label='eball')
        plt.plot(list(averages_ecube.keys()), list(averages_ecube.values()), marker='o', label='ecube')
        plt.plot(list(averages_logreg.keys()), list(averages_logreg.values()), marker='o', label='logreg')
        plt.plot(list(averages_svm.keys()), list(averages_svm.values()), marker='o', label='svm')
        plt.plot(list(averages_gmm.keys()), list(averages_gmm.values()), marker='o', label='gmm')

        plt.title(f"Plot of Accuracies for different number of PCAs for dataset {datasets[0]}")

    plt.xlabel('# of PCAs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()