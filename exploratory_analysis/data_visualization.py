import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as mso

def visualize_missing_values(df):
    """
    Visualize missing values in the dataset.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    mso.bar(df, fontsize=6, color='darkblue', ax=ax)
    plt.show()


def plot_transformed_data(train_df_transformed, transform_cols):
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(transform_cols, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(y=col, data=train_df_transformed)
        plt.title(f"Boxplot of Transformed {col}")

    plt.tight_layout()
    plt.show()


def plot_numeric_distributions(df, num_cols):
    """
    Plot histograms for numeric columns to visualize their distribution.
    """
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
    axes = axes.flat

    for i, col in enumerate(num_cols):
        sns.histplot(df, x=col, stat="count", kde=True, line_kws={"linewidth": 2.0},
                     alpha=0.4, color=(list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"], ax=axes[i])
        sns.rugplot(df, x=col, color=(list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"], ax=axes[i])
        axes[i].set_title(f"{col}", fontsize=10, fontweight="bold", color="darkred")

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, num_cols):
    """
    Plot a correlation matrix for numeric columns.
    """
    corr_matrix = df[num_cols].corr(method="spearman")
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, annot_kws={"fontsize": 6},
                square=True, mask=mask, linewidths=1.0, linecolor="white", ax=ax)
    plt.show()


def plot_boxplots_by_default(df, num_cols):
    """
    Plot box plots for numeric columns against the default status.
    """
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    axes = axes.flat

    for i, col in enumerate(num_cols):
        sns.boxplot(x=df["Default"], y=df[col], ax=axes[i])
        axes[i].set_title(f"{col} by Default", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_important_features(important_features):
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=important_features)
    plt.title("Top 10 Important Features")
    plt.show()
