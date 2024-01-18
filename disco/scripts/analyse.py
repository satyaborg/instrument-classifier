import glob
import pandas as pd
import altair as alt
import disco.config as config
from disco.utils.helpers import load_json


if __name__ == "__main__":
    results = []
    # get all hyperparameters
    for fp in glob.glob(f"{config.EXPORT_PATH}/*/hyperparams.json"):
        hyperparams = load_json(fp)
        results.append(hyperparams)

    all_results = []
    # get all training and evaluation results
    for result in results:
        fp = f"{config.EXPORT_PATH}/{result['runid']}/training.json"
        training = load_json(fp)
        result = {**result, **training}
        result["epochs"] = len(result["loss"])

        fp = f"{config.EXPORT_PATH}/{result['runid']}/evaluation.json"
        evaluation = load_json(fp)
        # add test as a prefix to the keys
        evaluation = {f"test_{k}": v for k, v in evaluation.items()}
        result = {**result, **evaluation}
        all_results.append(result)

    df = pd.DataFrame(all_results)
    # sort by best f1 score
    df.sort_values(by="test_f1_score", inplace=True, ascending=False)
    df.fillna(0.0, inplace=True)
    df.to_csv(f"{config.EXPORT_PATH}/results.csv", index=False)

    # generate plots for the best performing run
    df_plot = pd.DataFrame(
        dict(
            epoch=range(1, len(df.iloc[0].loss) + 1),
            train_loss=df.iloc[0].loss,
            val_loss=df.iloc[0].val_loss,
            train_precision=df.iloc[0].precision,
            val_precision=df.iloc[0].val_precision,
            train_recall=df.iloc[0].recall,
            val_recall=df.iloc[0].val_recall,
            train_f1_score=df.iloc[0].f1_score,
            val_f1_score=df.iloc[0].val_f1_score,
        )
    )
    df_plot = df_plot.melt("epoch")

    loss_plot = (
        alt.Chart(df_plot.loc[df_plot.variable.str.contains("loss")])
        .mark_line(point=False)
        .encode(
            x="epoch",
            y="value",
            color="variable",
            tooltip=["epoch", "value", "variable"],
        )
        .properties(width=800, height=400, title="Loss Curves")
        .interactive()
    )
    loss_plot.save(f"{config.EXPORT_PATH}/loss_plot.html")

    metrics_plot = (
        alt.Chart(df_plot.loc[~df_plot.variable.str.contains("loss")])
        .mark_line(point=False)
        .encode(
            x="epoch",
            y="value",
            color="variable",
            tooltip=["epoch", "value", "variable"],
        )
        .properties(width=800, height=400, title="Metrics")
        .interactive()
    )
    metrics_plot.save(f"{config.EXPORT_PATH}/metrics_plot.html")
    print("==> Done!")
