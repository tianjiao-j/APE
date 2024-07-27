import matplotlib.pyplot as plt
import yaml
from os import path, listdir, makedirs
import numpy as np


def get_tip_results(log_file, dataset_name):
    with open(log_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if dataset_name in line:
                no_ft = lines[i + 3].split()[1:]
                ft = lines[i + 4].split()[1:]
    no_ft = [float(x) for x in no_ft]
    no_ft.reverse()
    ft = [float(x) for x in ft]
    ft.reverse()
    return no_ft, ft


def get_empirical_tip_results(log_dir, dataset_name):
    for file in listdir(path.join(log_dir, dataset_name)):
        if 'yml' in file:
            filename = file
    if dataset_name == 'imagenet':
        indices = [0, 1, 2, 3, 4]  # [0, 1, 2, 3, 5]
    else:
        indices = [0, 1, 2, 3, 4]
    # fixme: read the best result
    results = yaml.load(open(path.join(log_dir, dataset_name, str(filename)), 'r'), Loader=yaml.FullLoader)
    ft_accs = [results['zs_ft'][i] for i in indices]
    ft_accs_ica = [results['zs_ft_ica'][i] for i in indices]
    accs_ica = [results['no_ft_ica'][i] for i in indices]
    accs = [results['no_ft'][i] for i in indices]

    return accs, accs_ica, ft_accs, ft_accs_ica


# =============================================================================================================
use_empirical_tip = True
#results_dir = 'final_results_a100/extra-cache-ica-fixed-seed'
dataset_names = ['imagenet', 'fgvc', 'oxford_pets', 'stanford_cars', 'eurosat', 'caltech101', 'sun397', 'dtd',
                 'oxford_flowers', 'food101', 'ucf101']
# dataset_names.remove('sun397')
# dataset_names.remove('fgvc')
shots = [1, 2, 4, 8, 16]
outputs_dir = 'outputs/2024_07_27-19_48'

ft_accss = []
ft_accss_ica = []
accss_ica = []
accss = []
zs_ft_accss = []
zs_ft_ica_accss = []
for dataset_name in dataset_names:
    dataset_outputs_dir = path.join(outputs_dir, dataset_name)
    if not path.exists(dataset_outputs_dir):
        continue
    result_exist = False
    for file in listdir(dataset_outputs_dir):
        if 'yaml' in file and dataset_name in file:
            filename = file
            result_exist = True
    if not result_exist:
        continue
    if dataset_name == 'imagenet':
        indices = [0, 1, 2, 3, 4]  # [0, 1, 2, 3, 5]
    else:
        indices = [0, 1, 2, 3, 4]
    # read previous results
    results = yaml.load(open(path.join(dataset_outputs_dir, str(filename)), 'r'), Loader=yaml.FullLoader)
    if use_empirical_tip:
        #if 'ft_ica' in results.keys():
        ft_zs = False
        #ft_accs_ica = [results['ft_ica'][i] for i in indices]
        ft_accs = [results['ft_accs'][i] for i in indices]
        # ft_accss_ica.append(ft_accs_ica)
        ft_accss.append(ft_accs)
        # else:
        #     ft_zs = True
        #     zs_ft_accs = [results['zs_ft'][i] for i in indices]
        #     zs_ft_ica_accs = [results['zs_ft_ica'][i] for i in indices]
        #     zs_ft_accss.append(zs_ft_accs)
        #     zs_ft_ica_accss.append(zs_ft_ica_accs)

        # accs_ica = [results['no_ft_ica'][i] for i in indices]
        accs = [results['accs'][i] for i in indices]
        # accss_ica.append(accs_ica)
        accss.append(accs)
    else:
        no_ft, ft = get_tip_results('exp.log', dataset_name)
        ft_accs = ft
        accs = no_ft
        ft_accss.append(ft_accs)
        accss.append(accs)
        ft_accs_ica = [results['ft_ica'][i] for i in indices]
        accs_ica = [results['no_ft_ica'][i] for i in indices]
        ft_accss_ica.append(ft_accs_ica)
        accss_ica = [accs_ica]

    plt.plot(shots, accs, label='Training-free', marker='o', color='green', linestyle='--')
    #plt.plot(shots, accs_ica, label='Training-free + ICA', marker='o', color='green', linestyle='-')
    # if ft_zs:
    #     plt.plot(shots, zs_ft_accs, label='ZS-FT', marker='^', color='red', linestyle='--')
    #     plt.plot(shots, zs_ft_ica_accs, label='ZS-FT + ICA', marker='^', color='red', linestyle='-')
    # else:
    plt.plot(shots, ft_accs, label='Fine-tuned', marker='s', color='blue', linestyle='--')
        #plt.plot(shots, ft_accs_ica, label='ICA Fine-tuned', marker='s', color='blue', linestyle='-')
    plt.legend(prop={'size': 7})
    plt.title('Average')
    plt.xlabel('Shots')
    plt.ylabel('Accuracy')
    img_path = path.join(outputs_dir, dataset_name, filename.replace('yaml', 'jpg'))
    plt.savefig(img_path, dpi=200)
    plt.close()

    # if dataset_name == 'imagenet':
    #     zs_ft_accs.append([results['ft'][i] for i in indices])
    #     zs_ft_accs_ica.append([results['ft_ica'][i] for i in indices])
    # else:
    #     zs_ft_accs.append([results['zs_ft'][i] for i in indices])
    #     zs_ft_accs_ica.append([results['zs_ft_ica'][i] for i in indices])

accs = np.array(accss).mean(axis=0)
#accs_ica = np.array(accss_ica).mean(axis=0)
# if ft_zs:
#     zs_ft_accs = np.array(zs_ft_accss).mean(axis=0)
#     zs_ft_ica_accs = np.array(zs_ft_ica_accss).mean(axis=0)
# else:
ft_accs = np.array(ft_accss).mean(axis=0)
    #ft_accs_ica = np.array(ft_accss_ica).mean(axis=0)

plt.plot(shots, accs, label='Training-free', marker='o', color='green', linestyle='--')
#plt.plot(shots, accs_ica, label='Training-free + ICA', marker='o', color='green', linestyle='-')
# if ft_zs:
#     plt.plot(shots, zs_ft_accs, label='ZS-FT', marker='^', color='red', linestyle='--')
#     plt.plot(shots, zs_ft_ica_accs, label='ZS-FT + ICA', marker='^', color='red', linestyle='-')
# else:
plt.plot(shots, ft_accs, label='Fine-tuned', marker='s', color='blue', linestyle='--')
   # plt.plot(shots, ft_accs_ica, label='ICA Fine-tuned', marker='s', color='blue', linestyle='-')
plt.legend(prop={'size': 7})
plt.title('Average')
plt.xlabel('Shots')
plt.ylabel('Accuracy')
img_path = path.join(outputs_dir, 'average.jpg')
plt.savefig(img_path, dpi=200)
plt.close()

dict = {}
dict['no_ft'] = accs.tolist()
# if ft_zs:
#     dict['zs_ft'], dict['zs_ft_ica'] = zs_ft_accs.tolist(), zs_ft_ica_accs.tolist()
# else:
dict['ft'] = ft_accs.tolist()

with open(img_path.replace('jpg', 'yml'), 'w') as outfile:
    yaml.dump(dict, outfile, default_flow_style=None)
