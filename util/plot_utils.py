"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path, PurePath


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            print("FIELD:", field)

            if field == 'mAP':
                # --- safe extract mAP ---
                def safe_map(x):
                    if isinstance(x, (list, tuple)) and len(x) > 1:
                        return x[1]   # mAP@[0.5:0.95]
                    return np.nan

                if 'test_coco_eval_bbox' in df.columns:
                    mAP_series = df['test_coco_eval_bbox'].apply(safe_map)

                    # convert numeric + clean
                    mAP_series = pd.to_numeric(mAP_series, errors='coerce')
                    mAP_series = mAP_series.dropna()

                    if len(mAP_series) > 0:
                        mAP_series = mAP_series.ewm(com=ewm_col).mean()
                        axs[j].plot(mAP_series, c=color)
                    else:
                        print("Warning: mAP empty")
                else:
                    print("Warning: test_coco_eval_bbox not found")

            else:
                cols = [f'train_{field}', f'test_{field}']
                valid_cols = [c for c in cols if c in df.columns]

                if len(valid_cols) == 0:
                    print(f"Warning: {field} not found")
                    continue

                df_plot = df[valid_cols].copy()

                # --- fix list values ---
                for c in valid_cols:
                    df_plot[c] = df_plot[c].apply(
                        lambda x: x[0] if isinstance(x, list) else x
                    )

                # --- convert to numeric ---
                df_plot = df_plot.apply(pd.to_numeric, errors='coerce')

                # --- fix warning ---
                df_plot = df_plot.infer_objects(copy=False)

                # --- plot ---
                df_plot.interpolate().ewm(com=ewm_col).mean().plot(
                    y=valid_cols,
                    ax=axs[j],
                    color=[color] * len(valid_cols),
                    style=['-', '--'][:len(valid_cols)]
                )

    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)


def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs
