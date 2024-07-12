import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions_vs_actuals(predictions, actuals, pearson_corr, spearman_corr, r_squared, file_path):
    # Create a jointplot with KDE and scatter plot
    grid = sns.jointplot(x=actuals, y=predictions, kind="scatter", height=8, space=0, ratio=4, alpha=0.6)
    grid = grid.plot_joint(sns.kdeplot, color='r', zorder=0, levels=6)
    
    # Add the identity line
    max_val = max(max(actuals), max(predictions))
    min_val = min(min(actuals), min(predictions))
    grid.ax_joint.plot([min_val, max_val], [min_val, max_val], 'k--')  # black dashed line

    # Set labels and title
    grid.set_axis_labels('Actual Values', 'Predictions')
    grid.fig.suptitle('Predictions vs. Actual Values')

    # Adding Pearson and Spearman correlation, and R-squared on the plot
    plt.text(0.05, 0.95, f'Pearson Correlation: {pearson_corr:.2f}\nSpearman Correlation: {spearman_corr:.2f}\nR-squared: {r_squared:.2f}',
             transform=grid.ax_joint.transAxes, fontsize=12, verticalalignment='top')

    # Save the plot to file
    plt.savefig(file_path)
    plt.close(grid.fig)