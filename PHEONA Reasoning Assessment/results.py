import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tiktoken
from scipy.stats import chi2_contingency

from utils import *

show_plots = True

replace_dict = {
        'NONE': -1,
        'IMV ONLY': 0,
        'NIPPV ONLY': 1,
        'HFNI ONLY': 2,
        'NIPPV TO IMV': 3,
        'HFNI TO IMV': 4,
        'IMV TO NIPPV': 5,
        'IMV TO HFNI': 6
    }

reverse_replace_dict = {v: k for k, v in replace_dict.items()}

hint_replace_dict = {
    -1: 0,
    0: 1,
    1: 2,
    2: 3,
    4: 5,
    5: 6,
    6: -1
}

def check_discrete_distribution_equality(df, col1, col2):
    """
    Check if two discrete distributions are the same using Chi-Square test of independence.
    """
    # Get all unique values across both columns
    all_values = np.union1d(df[col1].unique(), df[col2].unique())
    # Get value counts for each column, reindexed to include all possible values
    counts1 = df[col1].value_counts().reindex(all_values, fill_value=0).values
    counts2 = df[col2].value_counts().reindex(all_values, fill_value=0).values

    # Stack as a contingency table (2 rows: col1, col2)
    observed = np.array([counts1, counts2])
    print(f"Observed counts:\n{observed}")

    # Chi-square test of independence
    chi2_stat, p_value, _, _ = chi2_contingency(observed)

    print(f"Chi-Square Statistic: {chi2_stat}, p-value: {p_value}")
    if p_value < 0.05:
        print("The two distributions are significantly different.")
    else:
        print("The two distributions are not significantly different.")

    return chi2_stat, p_value

def get_context_length(text, model_name="gpt-3.5-turbo"):
    tokenizer = tiktoken.encoding_for_model(model_name)
    tokens = tokenizer.encode(text)
    return len(tokens)

def natural_explanationcorrectness_response_bins_plot(df, filepath):
    model_names_to_display = {
        'mistralsmall24binstruct2501q80': 'Mistral',
        'deepseekr132b': 'DeepSeek',
        'phi414bq80': 'Phi'
    }

    cot_types = ['nocot', 'somecot', 'fullcot']
    cot_type_labels = ['No CoT', 'Some CoT', 'Full CoT']
    response_bin_labels = ['0-500', '501-1000', '1001-1500', '1501+']
    models = df['model'].unique()
    bar_width = 0.45
    n_bins = len(response_bin_labels)
    inner_group_width = bar_width + 0.18
    group_width = n_bins * inner_group_width + 0.6  # width of each outer group (CoT type)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    for i, model in enumerate(models):
        ax = axes[i]
        df_model = df[df['model'] == model]

        # Remove nocot for Mistral and Phi, keep for DeepSeek
        if model_names_to_display.get(model, model) in ['Mistral', 'Phi']:
            cot_types_plot = ['somecot', 'fullcot']
            cot_type_labels_plot = ['Some CoT', 'Full CoT']
            n_cot_plot = 2
        else:
            cot_types_plot = cot_types
            cot_type_labels_plot = cot_type_labels
            n_cot_plot = 3

        x_pos = []
        total_heights = []
        explanation_heights = []
        percent_labels = []

        # Build bar positions and values
        for j, cot in enumerate(cot_types_plot):
            for k, token_bin in enumerate(response_bin_labels):
                x = j * group_width + k * inner_group_width
                x_pos.append(x)
                row = df_model[(df_model['cot_type'] == cot) & (df_model['n_response_tokens'] == token_bin)]
                if not row.empty:
                    total_responses = int(row['total_responses'].values[0])
                    explanation_errors = int(row['sum_explanation_correctness_errors'].values[0])
                else:
                    total_responses = 0
                    explanation_errors = 0
                total_heights.append(total_responses)
                explanation_heights.append(explanation_errors)
                percent = (explanation_errors / total_responses * 100) if total_responses > 0 else 0
                percent_labels.append(f"{percent:.0f}%")

        x_pos = np.array(x_pos)
        total_heights = np.array(total_heights)
        explanation_heights = np.array(explanation_heights)
        other_heights = total_heights - explanation_heights
        other_heights = np.clip(other_heights, 0, None)  # ensure no negative heights

        # Plot stacked bars: grey for "no errors", red for explanation correctness errors
        bar_other = ax.bar(x_pos, other_heights, width=bar_width, label='No Errors', color='#d3d3d3')
        bar_explanation = ax.bar(x_pos, explanation_heights, width=bar_width, bottom=other_heights, label='Explanation Correctness Errors', color='#ac2e44')

        # Add percent labels above each bar (bold, 10pt, not rotated)
        for xpos, total, expl, label in zip(x_pos, total_heights, explanation_heights, percent_labels):
            bar_top = expl + (total - expl)
            if total > 0:
                ax.text(
                    xpos, bar_top + max(total_heights) * 0.01, label,
                    ha='center', va='bottom', fontsize=12, color='#3a414a', fontweight='bold', rotation=90
                )

        # Add gridlines (major: dark grey, minor: light grey)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, which='major', color='#b0b0b0', linewidth=1.2, alpha=0.6)
        ax.yaxis.grid(True, which='minor', color='#e0e0e0', linewidth=0.7, alpha=0.4)
        ax.xaxis.grid(False)
        ax.minorticks_on()

        # Remove all x-ticks and x-tick labels
        ax.set_xticks([])
        ax.set_xticklabels([])

        # Add bin labels above each group
        bin_label_y = -0.01 * max(total_heights) if len(total_heights) > 0 else -1
        for j in range(n_cot_plot):
            group_start = j * group_width
            for k, bin_label in enumerate(response_bin_labels):
                xpos = group_start + k * inner_group_width
                ax.text(
                    xpos, bin_label_y, bin_label,
                    ha='center', va='top', fontsize=14, color='#3a414a', rotation=90, transform=ax.transData
                )

        # Add one label per group of bins (CoT type), centered under each group and further below (fixed axes position)
        for j, cot_label in enumerate(cot_type_labels_plot):
            group_start = j * group_width
            group_end = group_start + (n_bins - 1) * inner_group_width
            group_center = (group_start + group_end) / 2
            ax.text(
                group_center, -0.35, cot_label,
                ha='center', va='top', fontsize=16, color='#3a414a', transform=ax.get_xaxis_transform()
            )

        # Set y-axis to start at 0 and end at 120
        ax.set_ylim(0, 120)

        # Adjust tick params for clarity
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', labelsize=12)
        for spine in ax.spines.values():
            spine.set_color('#3a414a')
        if i == 0:
            ax.set_ylabel('Count', fontsize=16)
        if i == 2:
            ax.legend(fontsize=11, loc='upper right')

        # Add vertical lines to visually separate cot_type groups
        for j in range(1, n_cot_plot):
            xpos = j * group_width - inner_group_width / 2
            ax.axvline(x=xpos, color='#b0b0b0', linestyle='--', linewidth=1)

        # Add model label above each subplot
        ax.set_title(model_names_to_display.get(model, model), fontsize=16)

    # Remove the figure title and expand the graph to fill all available space
    fig.text(0.5, 0.01, 'CoT Type', ha='center', va='center', fontsize=16, color='#3a414a')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(filepath + "natural_explanationcorrectness_response_bins_plot.png", dpi=300, bbox_inches='tight')
    plt.savefig(filepath + "natural_explanationcorrectness_response_bins_plot.svg", bbox_inches='tight')

    if show_plots:
        plt.show()

def natural_response_bins_plot(df, filepath):
    model_names_to_display = {
        'mistralsmall24binstruct2501q80': 'Mistral',
        'deepseekr132b': 'DeepSeek',
        'phi414bq80': 'Phi'
    }

    cot_types = ['nocot', 'somecot', 'fullcot']
    cot_type_labels = ['No CoT', 'Some CoT', 'Full CoT']
    response_bin_labels = ['0-500', '501-1000', '1001-1500', '1501+']
    models = df['model'].unique()
    bar_width = 0.45
    n_bins = len(response_bin_labels)
    inner_group_width = bar_width + 0.18
    group_width = n_bins * inner_group_width + 0.6  # width of each outer group (CoT type)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    for i, model in enumerate(models):
        ax = axes[i]
        df_model = df[df['model'] == model]

        # Remove nocot for Mistral and Phi, keep for DeepSeek
        if model_names_to_display.get(model, model) in ['Mistral', 'Phi']:
            cot_types_plot = ['somecot', 'fullcot']
            cot_type_labels_plot = ['Some CoT', 'Full CoT']
            n_cot_plot = 2
        else:
            cot_types_plot = cot_types
            cot_type_labels_plot = cot_type_labels
            n_cot_plot = 3

        x_pos = []
        total_heights = []
        restoration_heights = []
        shortcut_heights = []
        percent_labels = []

        # Build bar positions and values
        for j, cot in enumerate(cot_types_plot):
            for k, token_bin in enumerate(response_bin_labels):
                x = j * group_width + k * inner_group_width
                x_pos.append(x)
                row = df_model[(df_model['cot_type'] == cot) & (df_model['n_response_tokens'] == token_bin)]
                if not row.empty:
                    total_responses = int(row['total_responses'].values[0])
                    restoration_errors = int(row['sum_restoration_errors'].values[0])
                    shortcut_errors = int(row['sum_unfaithfulshortcut_errors'].values[0])
                else:
                    total_responses = 0
                    restoration_errors = 0
                    shortcut_errors = 0
                total_heights.append(total_responses * 2)
                restoration_heights.append(restoration_errors)
                shortcut_heights.append(shortcut_errors)
                # Calculate percent label
                total_errors = restoration_errors + shortcut_errors
                percent = (total_errors / (total_responses * 2) * 100) if total_responses > 0 else 0
                percent_labels.append(f"{percent:.0f}%")

        x_pos = np.array(x_pos)
        total_heights = np.array(total_heights)
        restoration_heights = np.array(restoration_heights)
        shortcut_heights = np.array(shortcut_heights)
        other_heights = total_heights - (restoration_heights + shortcut_heights)
        other_heights = np.clip(other_heights, 0, None)  # ensure no negative heights

        # Plot stacked bars: grey for "other", blue for shortcut, red for restoration
        bar_other = ax.bar(x_pos, other_heights, width=bar_width, label='No Errors', color='#d3d3d3')
        bar_shortcut = ax.bar(x_pos, shortcut_heights, width=bar_width, bottom=other_heights, label='Unfaithful Shortcut Errors', color='#0e343e')
        bar_restoration = ax.bar(x_pos, restoration_heights, width=bar_width, bottom=other_heights + shortcut_heights, label='Restoration Errors', color='#ac2e44')

        # Add percent labels above each bar (bold, 10pt, not rotated)
        for xpos, total, rest, short, label in zip(x_pos, total_heights, restoration_heights, shortcut_heights, percent_labels):
            bar_top = rest + short + (total - (rest + short))
            if total > 0:
                ax.text(
                    xpos, bar_top + 2, label,
                    ha='center', va='bottom', fontsize=12, color='#3a414a', fontweight='bold', rotation=0
                )

        # Add gridlines (major: dark grey, minor: light grey)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, which='major', color='#b0b0b0', linewidth=1.2, alpha=0.6)
        ax.yaxis.grid(True, which='minor', color='#e0e0e0', linewidth=0.7, alpha=0.4)
        ax.xaxis.grid(False)
        ax.minorticks_on()

        # Remove all x-ticks and x-tick labels
        ax.set_xticks([])
        ax.set_xticklabels([])

        # Move bin labels closer to bars
        bin_label_y = -2
        for j in range(n_cot_plot):
            group_start = j * group_width
            for k, bin_label in enumerate(response_bin_labels):
                xpos = group_start + k * inner_group_width
                ax.text(
                    xpos, bin_label_y, bin_label,
                    ha='center', va='top', fontsize=14, color='#3a414a', rotation=90, transform=ax.transData, clip_on=False
                )

        # Move CoT type group label further down to avoid overlap with bin labels
        for j, cot_label in enumerate(cot_type_labels_plot):
            group_start = j * group_width
            group_end = group_start + (n_bins - 1) * inner_group_width
            group_center = (group_start + group_end) / 2
            ax.text(
                group_center, -0.40, cot_label,
                ha='center', va='top', fontsize=16, color='#3a414a', transform=ax.get_xaxis_transform(), clip_on=False
            )

        # Set y-axis to start at 0 and end at 200
        ax.set_ylim(0, 200)

        # Adjust tick params for clarity
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', labelsize=12)
        for spine in ax.spines.values():
            spine.set_color('#3a414a')
        if i == 0:
            ax.set_ylabel('Count (Max = 2 × Total Responses)', fontsize=16)
        if i == 2:
            ax.legend(fontsize=11, loc='upper right')

        # Add vertical lines to visually separate cot_type groups
        for j in range(1, n_cot_plot):
            xpos = j * group_width - inner_group_width / 2
            ax.axvline(x=xpos, color='#b0b0b0', linestyle='--', linewidth=1)

        # Add model label above each subplot
        ax.set_title(model_names_to_display.get(model, model), fontsize=16)

    # Remove the figure title and expand the graph to fill all available space
    fig.text(0.5, 0.01, 'CoT Type', ha='center', va='center', fontsize=16, color='#3a414a')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(filepath + "natural_response_bins_plot_stacked.png", dpi=300, bbox_inches='tight')
    plt.savefig(filepath + "natural_response_bins_plot_stacked.svg", bbox_inches='tight')

    if show_plots:
        plt.show()

def natural_explanationcorrectness_desc_bins_plot(df, filepath, show_plots=True):
    model_names_to_display = {
        'mistralsmall24binstruct2501q80': 'Mistral',
        'deepseekr132b': 'DeepSeek',
        'phi414bq80': 'Phi'
    }

    cot_types = ['nocot', 'somecot', 'fullcot']
    cot_type_labels = ['No CoT', 'Some CoT', 'Full CoT']
    desc_bin_labels = ['0-100', '101-200', '201-300', '301-400', '401+']
    models = df['model'].unique()
    bar_width = 0.45
    n_bins = len(desc_bin_labels)
    inner_group_width = bar_width + 0.18
    group_width = n_bins * inner_group_width + 0.6
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    for i, model in enumerate(models):
        ax = axes[i]
        df_model = df[df['model'] == model]

        # Remove nocot for Mistral and Phi, keep for DeepSeek
        if model_names_to_display.get(model, model) in ['Mistral', 'Phi']:
            cot_types_plot = ['somecot', 'fullcot']
            cot_type_labels_plot = ['Some CoT', 'Full CoT']
            n_cot_plot = 2
        else:
            cot_types_plot = cot_types
            cot_type_labels_plot = cot_type_labels
            n_cot_plot = 3

        x_pos = []
        total_heights = []
        explanation_heights = []
        other_heights = []
        percent_labels = []

        # Build bar positions and values
        for j, cot in enumerate(cot_types_plot):
            for k, desc_bin in enumerate(desc_bin_labels):
                x = j * group_width + k * inner_group_width
                x_pos.append(x)
                row = df_model[(df_model['cot_type'] == cot) & (df_model['n_description_tokens'] == desc_bin)]
                if not row.empty:
                    total_descriptions = int(row['total_descriptions'].values[0])
                    explanation_errors = int(row['sum_explanation_correctness_errors'].values[0])
                else:
                    total_descriptions = 0
                    explanation_errors = 0
                total_heights.append(total_descriptions)
                explanation_heights.append(explanation_errors)
                other_heights.append(total_descriptions - explanation_errors)
                percent = (explanation_errors / total_descriptions * 100) if total_descriptions > 0 else 0
                percent_labels.append(f"{percent:.0f}%")

        x_pos = np.array(x_pos)
        total_heights = np.array(total_heights)
        explanation_heights = np.array(explanation_heights)
        other_heights = np.array(other_heights)
        other_heights = np.clip(other_heights, 0, None)

        bar_other = ax.bar(x_pos, other_heights, width=bar_width, label='No Errors', color='#d3d3d3')
        bar_explanation = ax.bar(x_pos, explanation_heights, width=bar_width, bottom=other_heights, label='Explanation Correctness Errors', color='#ac2e44')

        # Add bold, rotated percent labels above each bar (10pt)
        for idx, (xpos, total, expl, other, label) in enumerate(zip(x_pos, total_heights, explanation_heights, other_heights, percent_labels)):
            bar_top = expl + other
            if total > 0:
                ax.text(
                    xpos, bar_top + max(total_heights) * 0.01, label,
                    ha='center', va='bottom', fontsize=12, color='#3a414a', fontweight='bold', rotation=90
                )

        ax.set_xticks([])
        ax.set_xticklabels([])

        bin_label_y = -0.01 * max(total_heights) if len(total_heights) > 0 else -1
        for idx, xpos in enumerate(x_pos):
            bin_label = desc_bin_labels[idx % n_bins]
            ax.text(
                xpos, bin_label_y, bin_label,
                ha='center', va='top', fontsize=14, color='#3a414a', rotation=90, transform=ax.transData
            )

        for j, cot_label in enumerate(cot_type_labels_plot):
            group_start = j * group_width
            group_end = group_start + (n_bins - 1) * inner_group_width
            group_center = (group_start + group_end) / 2
            ax.text(
                group_center, -0.25, cot_label,
                ha='center', va='top', fontsize=16, color='#3a414a', transform=ax.get_xaxis_transform()
            )

        ax.set_ylim(0, max(total_heights) * 1.1 if len(total_heights) > 0 else 1)

        ax.set_axisbelow(True)
        ax.yaxis.grid(True, which='major', color='#b0b0b0', linewidth=1.2, alpha=0.6)
        ax.yaxis.grid(True, which='minor', color='#e0e0e0', linewidth=0.7, alpha=0.4)
        ax.xaxis.grid(False)
        ax.minorticks_on()

        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', labelsize=12)
        for spine in ax.spines.values():
            spine.set_color('#3a414a')
        if i == 0:
            ax.set_ylabel('Count', fontsize=16)
            ax.legend(fontsize=11, loc='upper left')

        # Add model label above each subplot
        ax.set_title(model_names_to_display.get(model, model), fontsize=16, color='#3a414a')

        for j in range(1, n_cot_plot):
            xpos = j * group_width - inner_group_width / 2
            ax.axvline(x=xpos, color='#b0b0b0', linestyle='--', linewidth=1)

    # Remove the figure title and expand the graph to fill all available space
    fig.text(0.5, 0.01, 'CoT Type', ha='center', va='center', fontsize=16, color='#3a414a')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(filepath + "natural_explanationcorrectness_desc_bins_plot.png", dpi=300, bbox_inches='tight')
    plt.savefig(filepath + "natural_explanationcorrectness_desc_bins_plot.svg", bbox_inches='tight')

    if show_plots:
        plt.show()

def natural_desc_bins_plot(df, filepath, show_plots=True):
    model_names_to_display = {
        'mistralsmall24binstruct2501q80': 'Mistral',
        'deepseekr132b': 'DeepSeek',
        'phi414bq80': 'Phi'
    }

    cot_types = ['nocot', 'somecot', 'fullcot']
    cot_type_labels = ['No CoT', 'Some CoT', 'Full CoT']
    desc_bin_labels = ['0-100', '101-200', '201-300', '301-400', '401+']
    models = df['model'].unique()
    bar_width = 0.45
    n_bins = len(desc_bin_labels)
    inner_group_width = bar_width + 0.18
    group_width = n_bins * inner_group_width + 0.6
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    for i, model in enumerate(models):
        ax = axes[i]
        df_model = df[df['model'] == model]

        # Remove nocot for Mistral and Phi, keep for DeepSeek
        if model_names_to_display.get(model, model) in ['Mistral', 'Phi']:
            cot_types_plot = ['somecot', 'fullcot']
            cot_type_labels_plot = ['Some CoT', 'Full CoT']
            n_cot_plot = 2
        else:
            cot_types_plot = cot_types
            cot_type_labels_plot = cot_type_labels
            n_cot_plot = 3

        x_pos = []
        total_heights = []
        restoration_heights = []
        shortcut_heights = []
        percent_labels = []

        for j, cot in enumerate(cot_types_plot):
            for k, desc_bin in enumerate(desc_bin_labels):
                x = j * group_width + k * inner_group_width
                x_pos.append(x)
                row = df_model[(df_model['cot_type'] == cot) & (df_model['n_description_tokens'] == desc_bin)]
                if not row.empty:
                    total_descriptions = int(row['total_descriptions'].values[0])
                    restoration_errors = int(row['sum_restoration_errors'].values[0])
                    shortcut_errors = int(row['sum_unfaithfulshortcut_errors'].values[0])
                else:
                    total_descriptions = 0
                    restoration_errors = 0
                    shortcut_errors = 0
                total_heights.append(total_descriptions * 2)
                restoration_heights.append(restoration_errors)
                shortcut_heights.append(shortcut_errors)
                total_errors = restoration_errors + shortcut_errors
                percent = (total_errors / (total_descriptions * 2) * 100) if total_descriptions > 0 else 0
                percent_labels.append(f"{percent:.0f}%")

        x_pos = np.array(x_pos)
        total_heights = np.array(total_heights)
        restoration_heights = np.array(restoration_heights)
        shortcut_heights = np.array(shortcut_heights)
        other_heights = total_heights - (restoration_heights + shortcut_heights)
        other_heights = np.clip(other_heights, 0, None)

        bar_other = ax.bar(x_pos, other_heights, width=bar_width, label='No Errors', color='#d3d3d3')
        bar_shortcut = ax.bar(x_pos, shortcut_heights, width=bar_width, bottom=other_heights, label='Unfaithful Shortcut Errors', color='#0e343e')
        bar_restoration = ax.bar(x_pos, restoration_heights, width=bar_width, bottom=other_heights + shortcut_heights, label='Restoration Errors', color='#ac2e44')

        # Bold, 10pt, rotated percent labels above each bar (no stagger)
        for xpos, total, rest, short, label in zip(x_pos, total_heights, restoration_heights, shortcut_heights, percent_labels):
            bar_top = rest + short + (total - (rest + short))
            if total > 0:
                ax.text(
                    xpos, bar_top + max(total_heights) * 0.01, label,
                    ha='center', va='bottom', fontsize=12, color='#3a414a', fontweight='bold', rotation=90
                )

        ax.set_xticks([])
        ax.set_xticklabels([])

        bin_label_y = -0.01 * max(total_heights) if len(total_heights) > 0 else -1
        for idx, xpos in enumerate(x_pos):
            bin_label = desc_bin_labels[idx % n_bins]
            ax.text(
                xpos, bin_label_y, bin_label,
                ha='center', va='top', fontsize=14, color='#3a414a', rotation=90, transform=ax.transData
            )

        for j, cot_label in enumerate(cot_type_labels_plot):
            group_start = j * group_width
            group_end = group_start + (n_bins - 1) * inner_group_width
            group_center = (group_start + group_end) / 2
            ax.text(
                group_center, -0.25, cot_label,
                ha='center', va='top', fontsize=16, color='#3a414a', transform=ax.get_xaxis_transform()
            )

        ax.set_ylim(0, 50 + (max(total_heights) if len(total_heights) > 0 else 1))

        ax.set_axisbelow(True)
        ax.yaxis.grid(True, which='major', color='#b0b0b0', linewidth=1.2, alpha=0.6)
        ax.yaxis.grid(True, which='minor', color='#e0e0e0', linewidth=0.7, alpha=0.4)
        ax.xaxis.grid(False)
        ax.minorticks_on()

        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', labelsize=12)
        for spine in ax.spines.values():
            spine.set_color('#3a414a')
        if i == 0:
            ax.set_ylabel('Count (Max = 2 × Total Responses)', fontsize=16)
            ax.legend(fontsize=12, loc='upper left')

        # Add model label above each subplot
        ax.set_title(model_names_to_display.get(model, model), fontsize=16, color='#3a414a')

        for j in range(1, n_cot_plot):
            xpos = j * group_width - inner_group_width / 2
            ax.axvline(x=xpos, color='#b0b0b0', linestyle='--', linewidth=1)

    # Remove the figure title and expand the graph to fill all available space
    fig.text(0.5, 0.01, 'CoT Type', ha='center', va='center', fontsize=16, color='#3a414a')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(filepath + "natural_desc_bins_plot_stacked.png", dpi=300, bbox_inches='tight')
    plt.savefig(filepath + "natural_desc_bins_plot_stacked.svg", bbox_inches='tight')

    if show_plots:
        plt.show()

def natural_explanationcorrectness_outcome_plot(df, filepath, show_plots=True):
    model_names_to_display = {
        'mistralsmall24binstruct2501q80': 'Mistral',
        'deepseekr132b': 'DeepSeek',
        'phi414bq80': 'Phi'
    }

    cot_types = ['nocot', 'somecot', 'fullcot']
    cot_type_labels = ['No CoT', 'Some CoT', 'Full CoT']
    models = df['model'].unique()
    outcome_labels = sorted(df['cot_outcome'].unique(), key=lambda x: str(x))
    bar_width = 0.45
    n_outcomes = len(outcome_labels)
    n_cot = len(cot_types)
    inner_group_width = bar_width + 0.18
    group_width = n_outcomes * inner_group_width + 0.6
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    for i, model in enumerate(models):
        ax = axes[i]
        df_model = df[df['model'] == model]

        if model_names_to_display.get(model, model) in ['Mistral', 'Phi']:
            cot_types_plot = ['somecot', 'fullcot']
            cot_type_labels_plot = ['Some CoT', 'Full CoT']
            n_cot_plot = 2
        else:
            cot_types_plot = cot_types
            cot_type_labels_plot = cot_type_labels
            n_cot_plot = 3

        x_pos = []
        total_heights = []
        explanation_heights = []
        other_heights = []
        percent_labels = []

        for j, cot in enumerate(cot_types_plot):
            for k, outcome in enumerate(outcome_labels):
                x = j * group_width + k * inner_group_width
                x_pos.append(x)
                row = df_model[(df_model['cot_type'] == cot) & (df_model['cot_outcome'] == outcome)]
                if not row.empty:
                    total_outcomes = int(row['total_outcomes'].values[0])
                    explanation_errors = int(row['sum_explanation_correctness_errors'].values[0])
                else:
                    total_outcomes = 0
                    explanation_errors = 0
                total_heights.append(total_outcomes)
                explanation_heights.append(explanation_errors)
                other_heights.append(total_outcomes - explanation_errors)
                percent = (explanation_errors / total_outcomes * 100) if total_outcomes > 0 else 0
                percent_labels.append(f"{percent:.0f}%")

        x_pos = np.array(x_pos)
        total_heights = np.array(total_heights)
        explanation_heights = np.array(explanation_heights)
        other_heights = np.array(other_heights)
        other_heights = np.clip(other_heights, 0, None)

        bar_other = ax.bar(x_pos, other_heights, width=bar_width, label='No Errors', color='#d3d3d3')
        bar_explanation = ax.bar(x_pos, explanation_heights, width=bar_width, bottom=other_heights, label='Explanation Correctness Errors', color='#ac2e44')

        # Only add bolded percent labels, staggered, with increased base height, rotated 90 degrees
        for idx, (xpos, total, expl, other, label) in enumerate(zip(x_pos, total_heights, explanation_heights, other_heights, percent_labels)):
            bar_top = expl + other
            if total > 0:
                stagger = (idx % 2) * max(total_heights) * 0.01
                ax.text(
                    xpos, bar_top + max(total_heights) * 0.01 + stagger, label,
                    ha='center', va='bottom', fontsize=12, color='#3a414a', fontweight='bold', rotation=90
                )

        ax.set_xticks([])
        ax.set_xticklabels([])

        bin_label_y = -0.01 * max(total_heights) if len(total_heights) > 0 else -1
        for idx, xpos in enumerate(x_pos):
            outcome_label = outcome_labels[idx % n_outcomes]
            ax.text(
                xpos, bin_label_y, outcome_label,
                ha='center', va='top', fontsize=11, color='#3a414a', rotation=90, transform=ax.transData
            )

        for j, cot_label in enumerate(cot_type_labels_plot):
            group_start = j * group_width
            group_end = group_start + (n_outcomes - 1) * inner_group_width
            group_center = (group_start + group_end) / 2
            ax.text(
                group_center, -0.40, cot_label,
                ha='center', va='top', fontsize=14, color='#3a414a', transform=ax.get_xaxis_transform()
            )

        # Add model label above each subplot
        ax.set_title(model_names_to_display.get(model, model), fontsize=16, color='#3a414a')

        ax.set_ylim(0, 70)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, which='major', color='#b0b0b0', linewidth=1.2, alpha=0.6)
        ax.yaxis.grid(True, which='minor', color='#e0e0e0', linewidth=0.7, alpha=0.4)
        ax.xaxis.grid(False)
        ax.minorticks_on()
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', labelsize=12)
        for spine in ax.spines.values():
            spine.set_color('#3a414a')
        if i == 0:
            ax.set_ylabel('Count', fontsize=16)
            ax.legend(fontsize=12, loc='upper left')

        for j in range(1, n_cot_plot):
            xpos = j * group_width - inner_group_width / 2
            ax.axvline(x=xpos, color='#b0b0b0', linestyle='--', linewidth=1)

    fig.text(0.5, 0.01, 'CoT Type', ha='center', va='center', fontsize=16, color='#3a414a')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(filepath + "natural_explanationcorrectness_outcome_plot.png", dpi=300, bbox_inches='tight')
    plt.savefig(filepath + "natural_explanationcorrectness_outcome_plot.svg", bbox_inches='tight')

    if show_plots:
        plt.show()

def natural_outcome_plot(df, filepath, show_plots=True):
    model_names_to_display = {
        'mistralsmall24binstruct2501q80': 'Mistral',
        'deepseekr132b': 'DeepSeek',
        'phi414bq80': 'Phi'
    }

    cot_types = ['nocot', 'somecot', 'fullcot']
    cot_type_labels = ['No CoT', 'Some CoT', 'Full CoT']
    models = df['model'].unique()
    outcome_labels = sorted(df['cot_outcome'].unique(), key=lambda x: str(x))
    bar_width = 0.45
    n_outcomes = len(outcome_labels)
    n_cot = len(cot_types)
    inner_group_width = bar_width + 0.18
    group_width = n_outcomes * inner_group_width + 0.6  # width of each outer group (CoT type)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    for i, model in enumerate(models):
        ax = axes[i]
        df_model = df[df['model'] == model]

        # Remove nocot for Mistral and Phi, keep for DeepSeek
        if model_names_to_display.get(model, model) in ['Mistral', 'Phi']:
            cot_types_plot = ['somecot', 'fullcot']
            cot_type_labels_plot = ['Some CoT', 'Full CoT']
            n_cot_plot = 2
        else:
            cot_types_plot = cot_types
            cot_type_labels_plot = cot_type_labels
            n_cot_plot = 3

        x_pos = []
        total_heights = []
        restoration_heights = []
        shortcut_heights = []
        percent_labels = []

        # Build bar positions and values
        for j, cot in enumerate(cot_types_plot):
            for k, outcome in enumerate(outcome_labels):
                x = j * group_width + k * inner_group_width
                x_pos.append(x)
                row = df_model[(df_model['cot_type'] == cot) & (df_model['cot_outcome'] == outcome)]
                if not row.empty:
                    total_outcomes = int(row['total_outcomes'].values[0])
                    restoration_errors = int(row['sum_restoration_errors'].values[0])
                    shortcut_errors = int(row['sum_unfaithfulshortcut_errors'].values[0])
                else:
                    total_outcomes = 0
                    restoration_errors = 0
                    shortcut_errors = 0
                total_heights.append(total_outcomes * 2)
                restoration_heights.append(restoration_errors)
                shortcut_heights.append(shortcut_errors)
                # Calculate percent label
                total_errors = restoration_errors + shortcut_errors
                percent = (total_errors / (total_outcomes * 2) * 100) if total_outcomes > 0 else 0
                percent_labels.append(f"{percent:.0f}%")

        x_pos = np.array(x_pos)
        total_heights = np.array(total_heights)
        restoration_heights = np.array(restoration_heights)
        shortcut_heights = np.array(shortcut_heights)
        other_heights = total_heights - (restoration_heights + shortcut_heights)
        other_heights = np.clip(other_heights, 0, None)  # ensure no negative heights

        # Plot stacked bars: grey for "other", blue for shortcut, red for restoration
        bar_other = ax.bar(x_pos, other_heights, width=bar_width, label='No Errors', color='#d3d3d3')
        bar_shortcut = ax.bar(x_pos, shortcut_heights, width=bar_width, bottom=other_heights, label='Unfaithful Shortcut Errors', color='#0e343e')
        bar_restoration = ax.bar(x_pos, restoration_heights, width=bar_width, bottom=other_heights + shortcut_heights, label='Restoration Errors', color='#ac2e44')

        # Add bold percent labels above each bar, rotated 90 degrees
        for idx, (xpos, total, rest, short, label) in enumerate(zip(x_pos, total_heights, restoration_heights, shortcut_heights, percent_labels)):
            bar_top = rest + short + (total - (rest + short))
            if total > 0:
                ax.text(
                    xpos, bar_top + max(total_heights) * 0.01, label,
                    ha='center', va='bottom', fontsize=12, color='#3a414a', fontweight='bold', rotation=0
                )

        # Remove all x-ticks and x-tick labels
        ax.set_xticks([])
        ax.set_xticklabels([])

        # Add outcome labels vertically and directly under each bar
        bin_label_y = -0.01 * max(total_heights) if len(total_heights) > 0 else -1
        for idx, xpos in enumerate(x_pos):
            outcome_label = outcome_labels[idx % n_outcomes]
            ax.text(
                xpos, bin_label_y, outcome_label,
                ha='center', va='top', fontsize=14, color='#3a414a', rotation=90, transform=ax.transData
            )

        # Add one label per group of bins (CoT type), centered under each group and further below (fixed axes position)
        for j, cot_label in enumerate(cot_type_labels_plot):
            group_start = j * group_width
            group_end = group_start + (n_outcomes - 1) * inner_group_width
            group_center = (group_start + group_end) / 2
            ax.text(
                group_center, -0.40, cot_label,
                ha='center', va='top', fontsize=14, color='#3a414a', transform=ax.get_xaxis_transform()
            )

        # Set y-axis to start at 0 and end at 50 + max(total_heights)
        ax.set_ylim(0, 50 + (max(total_heights) if len(total_heights) > 0 else 1))

        # Add gridlines (major: dark grey, minor: light grey)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, which='major', color='#b0b0b0', linewidth=1.2, alpha=0.6)
        ax.yaxis.grid(True, which='minor', color='#e0e0e0', linewidth=0.7, alpha=0.4)
        ax.xaxis.grid(False)
        ax.minorticks_on()

        # Adjust tick params for clarity
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', labelsize=12)
        for spine in ax.spines.values():
            spine.set_color('#3a414a')
        # Add model label above each panel
        ax.set_title(model_names_to_display.get(model, model), fontsize=16, color='#3a414a')
        if i == 0:
            ax.set_ylabel('Count (Max = 2 × Total Responses)', fontsize=16)
            ax.legend(fontsize=12, loc='upper left')

        # Add vertical lines to visually separate cot_type groups
        for j in range(1, n_cot_plot):
            xpos = j * group_width - inner_group_width / 2
            ax.axvline(x=xpos, color='#b0b0b0', linestyle='--', linewidth=1)

    # Remove the figure title and expand the graph to fill all available space
    fig.text(0.5, 0.01, 'CoT Type', ha='center', va='center', fontsize=16, color='#3a414a')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(filepath + "natural_outcome_plot_stacked.png", dpi=300, bbox_inches='tight')
    plt.savefig(filepath + "natural_outcome_plot_stacked.svg", bbox_inches='tight')

    if show_plots:
        plt.show()

def natural_plot(sum_DeepSeek_nocot, sum_mistral_nocot, sum_phi_nocot,
                  sum_DeepSeek_somecot, sum_mistral_somecot, sum_phi_somecot,
                  sum_DeepSeek_fullcot, sum_mistral_fullcot, sum_phi_fullcot,
                  type_of_error, filename):
    models = ['DeepSeek', 'Mistral', 'Phi']
    cot_types = ['Some CoT', 'Full CoT']  # Removed 'No CoT'

    # Only keep Some CoT and Full CoT values (skip No CoT)
    DeepSeek_vals = [
        sum_DeepSeek_somecot,
        sum_DeepSeek_fullcot
    ]
    mistral_vals = [
        sum_mistral_somecot,
        sum_mistral_fullcot
    ]
    phi_vals = [
        sum_phi_somecot,
        sum_phi_fullcot
    ]

    print(f"DeepSeek NOCOT: {sum_DeepSeek_nocot}")

    cot_sums = {
        'DeepSeek': DeepSeek_vals,
        'Mistral': mistral_vals,
        'Phi': phi_vals
    }

    colors = {
        'DeepSeek': '#ac2e44',
        'Mistral': '#0e343e',
        'Phi': '#72ccae'
    }

    x = np.arange(len(cot_types))
    width = 0.25

    _, ax = plt.subplots(layout='constrained')

    hatch_types = ['///', None, '\\\\\\']
    for i, model in enumerate(models):
        offset = (i - 1) * width
        rects = ax.bar(
            x + offset, cot_sums[model], width, label=model,
            color=colors[model], hatch=hatch_types[i], edgecolor='#3a414a'
        )
        ax.bar_label(rects, padding=3, fontsize=16)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which='major', color='#b0b0b0', linewidth=1.2)
    ax.yaxis.grid(True, which='minor', color='#e0e0e0', linewidth=0.7)
    ax.xaxis.grid(False)
    ax.minorticks_on()

    custom_color = '#3a414a'
    ax.set_ylabel('Error Sum', color=custom_color, fontsize=16)
    ax.set_title(f'{type_of_error}', color=custom_color, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(cot_types, color=custom_color, fontsize=16)
    plt.setp(ax.get_yticklabels(), color=custom_color, fontsize=16)
    ax.legend(fontsize=13.5, title_fontsize=13.5)
    ax.set_ylim(0, 40)
    ax.set_yticks(np.arange(0, 41, 5))

    plt.savefig(f"{filename}.png", dpi=600)
    plt.savefig(f"{filename}.svg")
    if show_plots:
        plt.show()

def draw_vertical_pathway(ax, x, y_vals, color='#b0b0b0', alpha=0.25, text_color='#3a414a', diff_bold_color='#ac2e44'):
    y_vals_sorted = sorted(y_vals, key=lambda tup: tup[1], reverse=True)
    if len(y_vals_sorted) >= 2:
        y_poly = [y for _, y, *_ in y_vals_sorted]
        ax.fill_betweenx(y_poly, x-0.13, x+0.13, color=color, alpha=alpha, zorder=1)
        for i in range(len(y_vals_sorted)-1):
            y0 = y_vals_sorted[i][1]
            y1 = y_vals_sorted[i+1][1]
            # Removed the vertical line: ax.plot([x, x], [y0, y1], color=color, alpha=0.7, zorder=2)
            diff = y0 - y1
            ym = (y0 + y1) / 2
    for label, y, c, m, z in y_vals_sorted:
        ax.scatter(x, y, label=label, color=c, marker=m, s=120, zorder=z)

def simulated_plot(df, model_size_dict, filepath, show_plots=True):
    custom_color = '#3a414a'
    diff_bold_color = '#ac2e44'

    model_names_to_display = {
        'mistralsmall24binstruct2501q80': 'Mistral',
        'deepseekr132b': 'DeepSeek',
        'phi414bq80': 'Phi'
    }

    cot_type_order = ['nocot', 'somecot', 'fullcot']
    cot_type_labels = ['No CoT', 'Some CoT', 'Full CoT']
    cot_type_map = {k: i for i, k in enumerate(cot_type_order)}

    models = df['model'].unique()
    fig, axes = plt.subplots(len(models), 3, figsize=(18, 5 * len(models)), sharex='col', constrained_layout=True)
    if len(models) == 1:
        axes = [axes]

    # Legend handles
    legend_elements_random = [
        plt.Line2D([0], [0], marker='h', color='w', label='RandomFewShot + Hint', markerfacecolor='#ac2e44', markersize=14),
        plt.Line2D([0], [0], marker='p', color='w', label='RandomFewShot + NoHint', markerfacecolor='#0e343e', markersize=14),
        plt.Line2D([0], [0], marker='8', color='w', label='Unbiased', markerfacecolor='#72ccae', markersize=14)
    ]
    legend_elements_specific = [
        plt.Line2D([0], [0], marker='h', color='w', label='SpecificFewShot + Hint', markerfacecolor='#ac2e44', markersize=14),
        plt.Line2D([0], [0], marker='p', color='w', label='SpecificFewShot + NoHint', markerfacecolor='#0e343e', markersize=14),
        plt.Line2D([0], [0], marker='8', color='w', label='Unbiased', markerfacecolor='#72ccae', markersize=14)
    ]
    legend_elements_panel3 = [
        plt.Line2D([0], [0], marker='h', color='w', label='Hint Only', markerfacecolor='#ac2e44', markersize=14),
        plt.Line2D([0], [0], marker='8', color='w', label='Unbiased', markerfacecolor='#72ccae', markersize=14)
    ]

    # --- 1. Gather all y-values for all panels ---
    all_yvals = []
    for idx, model in enumerate(models):
        df_model = df[df['model'] == model]
        model_response_size = model_size_dict.get(model, 1000)
        rf_hint = df_model[(df_model['example_type'] == 'randomfewshot') & (df_model['hint_type'] == 'hint')]
        rf_nohint = df_model[(df_model['example_type'] == 'randomfewshot') & (df_model['hint_type'] == 'nohint')]
        unbiased = df_model.drop_duplicates(subset=['cot_type'])
        sf_hint = df_model[(df_model['example_type'] == 'specificfewshot') & (df_model['hint_type'] == 'hint')]
        sf_nohint = df_model[(df_model['example_type'] == 'specificfewshot') & (df_model['hint_type'] == 'nohint')]
        hint_only = df_model[(df_model['example_type'] == 'noexamples') & (df_model['hint_type'] == 'hint')]

        for cot in cot_type_order:
            for v in [
                rf_hint[rf_hint['cot_type'] == cot]['sum_iscorrect'] / model_response_size,
                rf_nohint[rf_nohint['cot_type'] == cot]['sum_iscorrect'] / model_response_size,
                unbiased[unbiased['cot_type'] == cot]['sum_iscorrect_unbiased'] / model_response_size,
                sf_hint[sf_hint['cot_type'] == cot]['sum_iscorrect'] / model_response_size,
                sf_nohint[sf_nohint['cot_type'] == cot]['sum_iscorrect'] / model_response_size,
                unbiased[unbiased['cot_type'] == cot]['sum_iscorrect_unbiased'] / model_response_size,
                unbiased[unbiased['cot_type'] == cot]['sum_iscorrect_unbiased'] / model_response_size,
                hint_only[hint_only['cot_type'] == cot]['sum_iscorrect'] / model_response_size
            ]:
                if not v.empty:
                    all_yvals.append(v.values[0])

    # --- 2. Compute global min and max ---
    if all_yvals:
        global_min = min(all_yvals)
        global_max = max(all_yvals)
    else:
        global_min, global_max = 0, 1

    # --- 3. Plot as before, with gridlines, y-limits, and legends in first row ---
    for idx, model in enumerate(models):
        df_model = df[df['model'] == model]
        model_response_size = model_size_dict.get(model, 1000)

        # PANEL 1: RANDOM FEW SHOT
        ax1 = axes[idx][0] if len(models) > 1 else axes[0]
        rf_hint = df_model[(df_model['example_type'] == 'randomfewshot') & (df_model['hint_type'] == 'hint')]
        rf_nohint = df_model[(df_model['example_type'] == 'randomfewshot') & (df_model['hint_type'] == 'nohint')]
        unbiased = df_model.drop_duplicates(subset=['cot_type'])

        ax1.set_axisbelow(True)
        ax1.yaxis.grid(True, which='major', color='#b0b0b0', linewidth=1.2, alpha=0.6)
        ax1.yaxis.grid(True, which='minor', color='#e0e0e0', linewidth=0.7, alpha=0.4)
        ax1.xaxis.grid(False)
        ax1.minorticks_on()

        for cot in cot_type_order:
            x = cot_type_map[cot]
            yvals = []
            v = rf_hint[rf_hint['cot_type'] == cot]['sum_iscorrect'] / model_response_size
            if not v.empty:
                yvals.append(('RandomFewShot + Hint', v.values[0], '#ac2e44', 'h', 4))
            v = rf_nohint[rf_nohint['cot_type'] == cot]['sum_iscorrect'] / model_response_size
            if not v.empty:
                yvals.append(('RandomFewShot + NoHint', v.values[0], '#0e343e', 'p', 4))
            v = unbiased[unbiased['cot_type'] == cot]['sum_iscorrect_unbiased'] / model_response_size
            if not v.empty:
                yvals.append(('Unbiased', v.values[0], '#72ccae', '8', 5))
            if yvals:
                draw_vertical_pathway(ax1, x, yvals, color='#b0b0b0', alpha=0.25, text_color=custom_color, diff_bold_color=diff_bold_color)
        ax1.set_ylabel('Accuracy', color=custom_color, fontsize=16)
        ax1.tick_params(axis='x', colors=custom_color, labelsize=16)
        ax1.tick_params(axis='y', colors=custom_color, labelsize=16)
        ax1.set_xticks([cot_type_map[c] for c in cot_type_order])
        ax1.set_xticklabels(cot_type_labels, color=custom_color, fontsize=16)
        plt.setp(ax1.get_yticklabels(), color=custom_color, fontsize=16)
        for spine in ax1.spines.values():
            spine.set_color(custom_color)
        ax1.set_ylim(global_min, global_max)
        if idx == 0:
            ax1.legend(handles=legend_elements_random, loc='upper center', fontsize=11.5)

        # PANEL 2: SPECIFIC FEW SHOT
        ax2 = axes[idx][1] if len(models) > 1 else axes[1]
        sf_hint = df_model[(df_model['example_type'] == 'specificfewshot') & (df_model['hint_type'] == 'hint')]
        sf_nohint = df_model[(df_model['example_type'] == 'specificfewshot') & (df_model['hint_type'] == 'nohint')]

        ax2.set_axisbelow(True)
        ax2.yaxis.grid(True, which='major', color='#b0b0b0', linewidth=1.2, alpha=0.6)
        ax2.yaxis.grid(True, which='minor', color='#e0e0e0', linewidth=0.7, alpha=0.4)
        ax2.xaxis.grid(False)
        ax2.minorticks_on()

        for cot in cot_type_order:
            x = cot_type_map[cot]
            yvals = []
            v = sf_hint[sf_hint['cot_type'] == cot]['sum_iscorrect'] / model_response_size
            if not v.empty:
                yvals.append(('SpecificFewShot + Hint', v.values[0], '#ac2e44', 'h', 4))
            v = sf_nohint[sf_nohint['cot_type'] == cot]['sum_iscorrect'] / model_response_size
            if not v.empty:
                yvals.append(('SpecificFewShot + NoHint', v.values[0], '#0e343e', 'p', 4))
            v = unbiased[unbiased['cot_type'] == cot]['sum_iscorrect_unbiased'] / model_response_size
            if not v.empty:
                yvals.append(('Unbiased', v.values[0], '#72ccae', '8', 5))
            if yvals:
                draw_vertical_pathway(ax2, x, yvals, color='#b0b0b0', alpha=0.25, text_color=custom_color, diff_bold_color=diff_bold_color)
        ax2.tick_params(axis='x', colors=custom_color, labelsize=16)
        ax2.tick_params(axis='y', colors=custom_color, labelsize=16)
        ax2.set_xticks([cot_type_map[c] for c in cot_type_order])
        ax2.set_xticklabels(cot_type_labels, color=custom_color, fontsize=16)
        plt.setp(ax2.get_yticklabels(), color=custom_color, fontsize=16)
        for spine in ax2.spines.values():
            spine.set_color(custom_color)
        ax2.set_ylim(global_min, global_max)
        if idx == 0:
            ax2.legend(handles=legend_elements_specific, loc='upper left', fontsize=11.5)

        # PANEL 3: UNBIASED VS HINT-ONLY (no examples)
        ax3 = axes[idx][2] if len(models) > 1 else axes[2]
        hint_only = df_model[(df_model['example_type'] == 'noexamples') & (df_model['hint_type'] == 'hint')]

        ax3.set_axisbelow(True)
        ax3.yaxis.grid(True, which='major', color='#b0b0b0', linewidth=1.2, alpha=0.6)
        ax3.yaxis.grid(True, which='minor', color='#e0e0e0', linewidth=0.7, alpha=0.4)
        ax3.xaxis.grid(False)
        ax3.minorticks_on()

        for cot in cot_type_order:
            x = cot_type_map[cot]
            yvals = []
            v = unbiased[unbiased['cot_type'] == cot]['sum_iscorrect_unbiased'] / model_response_size
            if not v.empty:
                yvals.append(('Unbiased', v.values[0], '#72ccae', '8', 5))
            v = hint_only[hint_only['cot_type'] == cot]['sum_iscorrect'] / model_response_size
            if not v.empty:
                yvals.append(('Hint Only', v.values[0], '#ac2e44', 'h', 4))
            if yvals:
                draw_vertical_pathway(ax3, x, yvals, color='#b0b0b0', alpha=0.25, text_color=custom_color, diff_bold_color=diff_bold_color)
        ax3.tick_params(axis='x', colors=custom_color, labelsize=16)
        ax3.tick_params(axis='y', colors=custom_color, labelsize=16)
        ax3.set_xticks([cot_type_map[c] for c in cot_type_order])
        ax3.set_xticklabels(cot_type_labels, color=custom_color, fontsize=16)
        plt.setp(ax3.get_yticklabels(), color=custom_color, fontsize=16)
        for spine in ax3.spines.values():
            spine.set_color(custom_color)
        ax3.set_ylim(global_min, global_max)
        if idx == 0:
            ax3.legend(handles=legend_elements_panel3, loc='upper left', fontsize=11.5)

        # Model label
        ax_for_label = ax3
        ylim = ax_for_label.get_ylim()
        ymid = (ylim[0] + ylim[1]) / 2
        ax_for_label.text(
            1.04, ymid, model_names_to_display.get(model, ''),
            color=custom_color, fontsize=16,
            va='center', ha='left', rotation=270,
            transform=ax_for_label.get_yaxis_transform()
        )

    # Remove all x-axis labels for all axes
    for row in axes:
        for ax in row:
            ax.set_xlabel('')

    # Tight layout first to pack everything
    plt.tight_layout()
    # Make room for the label below the axes
    plt.subplots_adjust(bottom=0.075)  # Increase if needed

    # Place the label below all subplots and their axis labels
    fig.text(0.5, 0.03, 'CoT Type', ha='center', va='center', fontsize=16, color=custom_color)

    plt.savefig(filepath + "simulated_errors_plot.png", dpi=300)
    plt.savefig(filepath + "simulated_errors_plot.svg")
    if show_plots:
        plt.show()

def plot_panel_discrete_distribution_overlay(simulated_df_dict, models, cot_types, hint_only_responses, savepath=None, show_plots=True):
    """
    Plots a 3x3 panel (model x cot_type) of overlaid discrete distributions for hint-only vs unbiased responses.
    Each subplot overlays the hint-only (red, solid, background) and unbiased (green, transparent, foreground) distributions for a given model and cot_type.
    Bars are touching, p-value is shown, only left-most figure has y-label and y-axis count labels, only right-most has model label.
    Adds a legend below the p-value in the upper right panel.
    Each graph has outcome labels mapped to reverse_replace_dict and a larger centered label for each column that is the CoT type.
    If p < 0.001, show "p < 0.001". If p < 0.05, add an asterisk.
    Only show the x-axis outcome labels and CoT label on the bottommost panel, and move the CoT label further down.
    Only the left-most column has y-axis count labels (not above bars).
    """

    color_unbiased = '#0e343e'  # dark green
    color_hint = '#ac2e44'      # dark red

    model_names_to_display = {
        'mistralsmall24binstruct2501q80': 'Mistral',
        'deepseekr132b': 'DeepSeek',
        'phi414bq80': 'Phi'
    }
    cot_type_labels = ['No CoT', 'Some CoT', 'Full CoT']

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharey='row')
    for i, m in enumerate(models):
        df = simulated_df_dict[m]
        unbiased_cols = {
            'nocot': 'no_cot_parsed_response_shifted',
            'somecot': 'some_cot_parsed_response_shifted',
            'fullcot': 'full_cot_parsed_response_shifted'
        }
        for j, cot in enumerate(cot_types):
            ax = axes[i, j]
            col_unbiased = unbiased_cols[cot]
            col_hint = hint_only_responses[j]
            # Get all unique values across both columns
            all_values = np.union1d(df[col_unbiased].unique(), df[col_hint].unique())
            all_values_sorted = sorted(all_values)
            # Map outcome values to text using reverse_replace_dict
            outcome_labels = [reverse_replace_dict.get(v, str(v)) for v in all_values_sorted]
            counts_hint = df[col_hint].value_counts().reindex(all_values_sorted, fill_value=0)
            counts_unbiased = df[col_unbiased].value_counts().reindex(all_values_sorted, fill_value=0)
            x = np.arange(len(all_values_sorted))
            width = 0.95  # Bars touch each other

            # Plot hint-only (red, solid, background)
            ax.bar(x, counts_hint.values, width=width, label='Hint Only', color=color_hint, alpha=1.0, zorder=1)
            # Plot unbiased (green, transparent, foreground)
            ax.bar(x, counts_unbiased.values, width=width, label='Shifted Unbiased', color=color_unbiased, alpha=0.45, zorder=2)

            ax.set_xticks(x)
            # Only show outcome labels and CoT label on bottommost row
            if i == 2:
                ax.set_xticklabels(outcome_labels, fontsize=13, rotation=90)
                # Move CoT label further down
                ax.text(
                    0.5, -0.75, cot_type_labels[j],
                    ha='center', va='top', fontsize=16, color='#3a414a', transform=ax.transAxes
                )
            else:
                ax.set_xticklabels([''] * len(x))
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, which='major', color='#b0b0b0', linewidth=1.2, alpha=0.6)
            ax.yaxis.grid(True, which='minor', color='#e0e0e0', linewidth=0.7, alpha=0.4)  # Add minor gridlines
            ax.minorticks_on()

            # Only left-most panel gets y-label and y-axis count labels
            if j == 0:
                ax.set_ylabel('Count', fontsize=16, color='#3a414a')
                ax.tick_params(axis='y', labelsize=14, colors='#3a414a')
                for label in ax.get_yticklabels():
                    label.set_visible(True)
            else:
                ax.set_ylabel('')
                for label in ax.get_yticklabels():
                    label.set_visible(False)

            # Chi-square test and p-value annotation (rounded to 3 decimals, asterisk if <0.05, p<0.001 formatting)
            _, p_value = check_discrete_distribution_equality(df, col_unbiased, col_hint)
            if p_value < 0.001:
                p_text = "p < 0.001"
            else:
                p_text = f"p={round(p_value, 3):.3f}"
            if p_value < 0.05:
                p_text += " *"
            ax.text(0.98, 0.92, p_text, ha='right', va='top',
                    transform=ax.transAxes, fontsize=13, color='#3a414a',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            # Only right-most panel gets model label and legend
            if j == 2:
                ax.text(1.08, 0.5, model_names_to_display.get(m, m),
                        color='#3a414a', fontsize=16, va='center', ha='left', rotation=270,
                        transform=ax.transAxes)
                handles = [
                    plt.Line2D([0], [0], color=color_hint, lw=8, label='Hint Only'),
                    plt.Line2D([0], [0], color=color_unbiased, lw=8, alpha=0.45, label='Shifted Unbiased')
                ]
                ax.legend(handles=handles, fontsize=12, loc='upper right', bbox_to_anchor=(1, 0.78))
            else:
                ax.set_title('')

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.savefig(savepath.replace('.png', '.svg'), bbox_inches='tight')
    if show_plots:
        plt.show()

def percent_difference(new, baseline):
    """Return percent difference from baseline to new value."""
    if baseline == 0:
        return 0
    return ((new - baseline) / baseline) * 100

def plot_examples_only_percent_difference(simulated_df_dict, models, savepath=None, show_plots=True):
    """
    Plots 3 panels (one per model), each with 3 groups (CoT types), each group with 2 bars (Random, Specific).
    Bar value is percent difference from unbiased count of '6' outcomes.
    If percent difference > 0, label above bar (rounded 0 decimals); if < 0, label below bar (rounded 0 decimals, same gap).
    Below each group, show "BASELINE: #" where # is the unbiased count for that CoT type, bolded.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    model_names_to_display = {
        'mistralsmall24binstruct2501q80': 'Mistral',
        'deepseekr132b': 'DeepSeek',
        'phi414bq80': 'Phi'
    }

    cot_types = ['nocot', 'somecot', 'fullcot']
    cot_labels = ['No CoT', 'Some CoT', 'Full CoT']
    colors = ['#ac2e44', '#0e343e']

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)  # Increased height
    for i, m in enumerate(models):
        model_df = simulated_df_dict[m]
        # Get unbiased counts
        unbiased_counts = {
            'nocot': (model_df['no_cot_parsed_response'] == 6).sum(),
            'somecot': (model_df['some_cot_parsed_response'] == 6).sum(),
            'fullcot': (model_df['full_cot_parsed_response'] == 6).sum()
        }
        # Get random and specific counts
        random_counts = {
            'nocot': (model_df['nocot_randomfewshot_nohint_parsed_response'] == 6).sum(),
            'somecot': (model_df['somecot_randomfewshot_nohint_parsed_response'] == 6).sum(),
            'fullcot': (model_df['fullcot_randomfewshot_nohint_parsed_response'] == 6).sum()
        }
        specific_counts = {
            'nocot': (model_df['nocot_specificfewshot_nohint_parsed_response'] == 6).sum(),
            'somecot': (model_df['somecot_specificfewshot_nohint_parsed_response'] == 6).sum(),
            'fullcot': (model_df['fullcot_specificfewshot_nohint_parsed_response'] == 6).sum()
        }
        # Calculate percent differences
        percent_diffs = []
        for cot in cot_types:
            pd_random = percent_difference(random_counts[cot], unbiased_counts[cot])
            pd_specific = percent_difference(specific_counts[cot], unbiased_counts[cot])
            percent_diffs.append([pd_random, pd_specific])

        percent_diffs = np.array(percent_diffs)  # shape (3,2)
        ax = axes[i]
        x = np.arange(len(cot_types))
        width = 0.35
        label_gap = 3  # pixels above/below bar

        # Plot bars for each CoT type (bars first)
        rects1 = ax.bar(x - width/2, percent_diffs[:,0], width, label='Random Fewshot', color=colors[0], zorder=2)
        rects2 = ax.bar(x + width/2, percent_diffs[:,1], width, label='Specific Fewshot', color=colors[1], zorder=2)

        # Draw horizontal line after bars
        ax.axhline(0, color='#b0b0b0', linestyle='--', linewidth=1, zorder=1)

        ax.set_xticks(x)
        ax.set_xticklabels(cot_labels, fontsize=14)
        # Only left-most panel gets y-axis label and legend
        if i == 0:
            ax.set_ylabel('Percent Difference from Unbiased', fontsize=15)
            ax.legend(fontsize=13, loc='upper left')
        else:
            ax.set_ylabel('')
        # Clean model names for titles
        ax.set_title(model_names_to_display.get(m, m), fontsize=16)
        ax.yaxis.grid(True, which='major', color='#b0b0b0', linewidth=1.2, alpha=0.6)
        ax.yaxis.grid(True, which='minor', color='#e0e0e0', linewidth=0.7, alpha=0.4)
        ax.minorticks_on()

        # Annotate bars: above if >0, below if <0, rounded to 0 decimals, same gap for both
        for rects in [rects1, rects2]:
            for rect in rects:
                height = rect.get_height()
                label = f'{height:.0f}%'
                if height > 0:
                    ax.annotate(label,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, label_gap), textcoords="offset points",
                        ha='center', va='bottom', fontsize=12)
                elif height < 0:
                    ax.annotate(label,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, -label_gap), textcoords="offset points",
                        ha='center', va='top', fontsize=12)

        # Add baseline text between -50% and -100%
        baseline_y = -75  # Fixed position at -75%
        for idx, cot in enumerate(cot_types):
            ax.text(
                x[idx], baseline_y,
                f"Baseline: {unbiased_counts[cot]}",
                ha='center', va='center', fontsize=14, color='#3a414a', fontweight='bold'
            )
        
        # Set y-limits: -100% at bottom, add padding above max value
        max_value = np.max(percent_diffs)
        upper_limit = max_value + 20 if max_value > 0 else 20  # Add 20% padding above max
        ax.set_ylim(-100, upper_limit)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.savefig(savepath.replace('.png', '.svg'), bbox_inches='tight')
    if show_plots:
        plt.show()

def get_simulated_iscorrect(df):
    df['gt_outcome'] = df['gt_outcome'].astype(int)

    cot_types = ['nocot', 'somecot', 'fullcot']
    example_types = ['randomfewshot', 'specificfewshot']
    hint_types = ['nohint', 'hint']

    iscorrect_colnames = []
    for cot in cot_types:
        for example in example_types:
            for hint in hint_types:
                df[f'{cot}_{example}_{hint}_parsed_response'] = df[f'{cot}_{example}_{hint}_parsed_response'].astype(int)
                colname = f'iscorrect_{cot}_{example}_{hint}'
                df[colname] = np.where(df[f'{cot}_{example}_{hint}_parsed_response'] == df['gt_outcome'], 1, 0)
                iscorrect_colnames.append(colname)

    additional_columns = [
        'no_cot_parsed_response', 'some_cot_parsed_response', 'full_cot_parsed_response',
        'nocot_noexamples_hint_parsed_response', 'somecot_noexamples_hint_parsed_response', 'fullcot_noexamples_hint_parsed_response'
    ]
    df[additional_columns] = df[additional_columns].astype(int)
    for col in additional_columns:
        colname = f'iscorrect_{col}'
        iscorrect_colnames.append(colname)
        df[colname] = np.where(df[col] == df['gt_outcome'], 1, 0)

    return df, iscorrect_colnames

def get_converted_simulated_iscorrect(df, model):
    return_df = pd.DataFrame(columns=['cot_type','example_type', 'hint_type', 'sum_iscorrect'])

    row1 = {'model': model, 'cot_type': 'nocot', 'example_type': 'randomfewshot', 'hint_type': 'nohint', 'sum_iscorrect': df['iscorrect_nocot_randomfewshot_nohint'].sum(),
            'sum_iscorrect_unbiased': df['iscorrect_no_cot_parsed_response'].sum()}
    row2 = {'model': model, 'cot_type': 'nocot', 'example_type': 'randomfewshot', 'hint_type': 'hint', 'sum_iscorrect': df['iscorrect_nocot_randomfewshot_hint'].sum(),
              'sum_iscorrect_unbiased': df['iscorrect_no_cot_parsed_response'].sum()}
    row3 = {'model': model, 'cot_type': 'nocot', 'example_type': 'specificfewshot', 'hint_type': 'nohint', 'sum_iscorrect': df['iscorrect_nocot_specificfewshot_nohint'].sum(),
              'sum_iscorrect_unbiased': df['iscorrect_no_cot_parsed_response'].sum()}
    row4 = {'model': model, 'cot_type': 'nocot', 'example_type': 'specificfewshot', 'hint_type': 'hint', 'sum_iscorrect': df['iscorrect_nocot_specificfewshot_hint'].sum(),
              'sum_iscorrect_unbiased': df['iscorrect_no_cot_parsed_response'].sum()}
    row5 = {'model': model, 'cot_type': 'somecot', 'example_type': 'randomfewshot', 'hint_type': 'nohint', 'sum_iscorrect': df['iscorrect_somecot_randomfewshot_nohint'].sum(),
            'sum_iscorrect_unbiased': df['iscorrect_some_cot_parsed_response'].sum()}
    row6 = {'model': model, 'cot_type': 'somecot', 'example_type': 'randomfewshot', 'hint_type': 'hint', 'sum_iscorrect': df['iscorrect_somecot_randomfewshot_hint'].sum(),
            'sum_iscorrect_unbiased': df['iscorrect_some_cot_parsed_response'].sum()}
    row7 = {'model': model, 'cot_type': 'somecot', 'example_type': 'specificfewshot', 'hint_type': 'nohint', 'sum_iscorrect': df['iscorrect_somecot_specificfewshot_nohint'].sum(),
            'sum_iscorrect_unbiased': df['iscorrect_some_cot_parsed_response'].sum()}
    row8 = {'model': model, 'cot_type': 'somecot', 'example_type': 'specificfewshot', 'hint_type': 'hint', 'sum_iscorrect': df['iscorrect_somecot_specificfewshot_hint'].sum(),
            'sum_iscorrect_unbiased': df['iscorrect_some_cot_parsed_response'].sum()}
    row9 = {'model': model, 'cot_type': 'fullcot', 'example_type': 'randomfewshot', 'hint_type': 'nohint', 'sum_iscorrect': df['iscorrect_fullcot_randomfewshot_nohint'].sum(),
            'sum_iscorrect_unbiased': df['iscorrect_full_cot_parsed_response'].sum()}
    row10 = {'model': model, 'cot_type': 'fullcot', 'example_type': 'randomfewshot', 'hint_type': 'hint', 'sum_iscorrect': df['iscorrect_fullcot_randomfewshot_hint'].sum(),
             'sum_iscorrect_unbiased': df['iscorrect_full_cot_parsed_response'].sum()}
    row11 = {'model': model, 'cot_type': 'fullcot', 'example_type': 'specificfewshot', 'hint_type': 'nohint', 'sum_iscorrect': df['iscorrect_fullcot_specificfewshot_nohint'].sum(),
             'sum_iscorrect_unbiased': df['iscorrect_full_cot_parsed_response'].sum()}
    row12 = {'model': model, 'cot_type': 'fullcot', 'example_type': 'specificfewshot', 'hint_type': 'hint', 'sum_iscorrect': df['iscorrect_fullcot_specificfewshot_hint'].sum(),
             'sum_iscorrect_unbiased': df['iscorrect_full_cot_parsed_response'].sum()}
    row13 = {'model': model, 'cot_type': 'nocot', 'example_type': 'noexamples', 'hint_type': 'hint', 'sum_iscorrect': df['iscorrect_nocot_noexamples_hint_parsed_response'].sum(),
             'sum_iscorrect_unbiased': df['iscorrect_no_cot_parsed_response'].sum()}
    row14 = {'model': model, 'cot_type': 'somecot', 'example_type': 'noexamples', 'hint_type': 'hint', 'sum_iscorrect': df['iscorrect_somecot_noexamples_hint_parsed_response'].sum(),
             'sum_iscorrect_unbiased': df['iscorrect_some_cot_parsed_response'].sum()}
    row15 = {'model': model, 'cot_type': 'fullcot', 'example_type': 'noexamples', 'hint_type': 'hint', 'sum_iscorrect': df['iscorrect_fullcot_noexamples_hint_parsed_response'].sum(),
             'sum_iscorrect_unbiased': df['iscorrect_full_cot_parsed_response'].sum()}
    return_df = pd.concat([return_df, pd.DataFrame([row1, row2, row3, row4, row5, row6, row7, row8, row9, row10, row11, row12, row13, row14, row15])], ignore_index=True)
    return_df['sum_iscorrect'] = return_df['sum_iscorrect'].astype(int)

    return return_df

def get_converted_natural_response_tokens(df, model):
    return_df = pd.DataFrame(columns=['cot_type','total_responses', 'n_response_tokens','sum_restoration_errors'])

    bins = ['0-500', '501-1000', '1001-1500', '1501+']

    for bin in bins:
        row1 = {'model': model, 'cot_type': 'nocot', 'total_responses': df[df['n_tokens_nocot_response_group']==bin].shape[0], 'n_response_tokens': bin, 'sum_restoration_errors': df[df['n_tokens_nocot_response_group'] == bin]['nocot_restoration_error'].sum(), 'sum_unfaithfulshortcut_errors': df[df['n_tokens_nocot_response_group'] == bin]['nocot_unfaithfulshortcut_error'].sum()}
        row2 = {'model': model, 'cot_type': 'somecot', 'total_responses': df[df['n_tokens_somecot_response_group']==bin].shape[0], 'n_response_tokens': bin, 'sum_restoration_errors': df[df['n_tokens_somecot_response_group'] == bin]['somecot_restoration_error'].sum(), 'sum_unfaithfulshortcut_errors': df[df['n_tokens_somecot_response_group'] == bin]['somecot_unfaithfulshortcut_error'].sum()}
        row3 = {'model': model, 'cot_type': 'fullcot', 'total_responses': df[df['n_tokens_fullcot_response_group']==bin].shape[0], 'n_response_tokens': bin, 'sum_restoration_errors': df[df['n_tokens_fullcot_response_group'] == bin]['fullcot_restoration_error'].sum(), 'sum_unfaithfulshortcut_errors': df[df['n_tokens_fullcot_response_group'] == bin]['fullcot_unfaithfulshortcut_error'].sum()}
        return_df = pd.concat([return_df, pd.DataFrame([row1, row2, row3])], ignore_index=True)

    return_df['total_responses'] = return_df['total_responses'].astype(int)
    return_df['sum_restoration_errors'] = return_df['sum_restoration_errors'].astype(int)
    return_df['sum_unfaithfulshortcut_errors'] = return_df['sum_unfaithfulshortcut_errors'].astype(int)
    return_df['n_response_tokens'] = return_df['n_response_tokens'].astype(str)
    return_df['cot_type'] = return_df['cot_type'].astype(str)

    return return_df

def get_converted_explanationcorrectness_response_tokens(df, model):
    return_df = pd.DataFrame(columns=['cot_type','total_responses', 'n_response_tokens','sum_explanation_correctness_errors'])

    bins = ['0-500', '501-1000', '1001-1500', '1501+']

    for bin in bins:
        row1 = {'model': model, 'cot_type': 'nocot', 'total_responses': df[df['n_tokens_nocot_response_group']==bin].shape[0], 'n_response_tokens': bin, 'sum_explanation_correctness_errors': df[df['n_tokens_nocot_response_group'] == bin]['nocot_explanationcorrectness_error'].sum()}
        row2 = {'model': model, 'cot_type': 'somecot', 'total_responses': df[df['n_tokens_somecot_response_group']==bin].shape[0], 'n_response_tokens': bin, 'sum_explanation_correctness_errors': df[df['n_tokens_somecot_response_group'] == bin]['somecot_explanationcorrectness_error'].sum()}
        row3 = {'model': model, 'cot_type': 'fullcot', 'total_responses': df[df['n_tokens_fullcot_response_group']==bin].shape[0], 'n_response_tokens': bin, 'sum_explanation_correctness_errors': df[df['n_tokens_fullcot_response_group'] == bin]['fullcot_explanationcorrectness_error'].sum()}
        return_df = pd.concat([return_df, pd.DataFrame([row1, row2, row3])], ignore_index=True)

    return_df['total_responses'] = return_df['total_responses'].astype(int)
    return_df['sum_explanation_correctness_errors'] = return_df['sum_explanation_correctness_errors'].astype(int)
    return_df['n_response_tokens'] = return_df['n_response_tokens'].astype(str)
    return_df['cot_type'] = return_df['cot_type'].astype(str)

    return return_df

def get_converted_natural_description_tokens(df, model):
    return_df = pd.DataFrame(columns=['cot_type','n_description_tokens','total_descriptions', 'sum_restoration_errors', 'sum_unfaithfulshortcut_errors'])

    bins = ['0-100', '101-200', '201-300', '301-400', '401+']

    for bin in bins:
        row = {'model': model, 'cot_type': 'nocot', 'n_description_tokens': bin, 'total_descriptions': df[df['n_tokens_description_group'] == bin].shape[0], 'sum_restoration_errors': df[df['n_tokens_description_group'] == bin]['nocot_restoration_error'].sum(), 'sum_unfaithfulshortcut_errors': df[df['n_tokens_description_group'] == bin]['nocot_unfaithfulshortcut_error'].sum()}
        return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

        row = {'model': model, 'cot_type': 'somecot', 'n_description_tokens': bin, 'total_descriptions': df[df['n_tokens_description_group'] == bin].shape[0], 'sum_restoration_errors': df[df['n_tokens_description_group'] == bin]['somecot_restoration_error'].sum(), 'sum_unfaithfulshortcut_errors': df[df['n_tokens_description_group'] == bin]['somecot_unfaithfulshortcut_error'].sum()}
        return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

        row = {'model': model, 'cot_type': 'fullcot', 'n_description_tokens': bin, 'total_descriptions': df[df['n_tokens_description_group'] == bin].shape[0], 'sum_restoration_errors': df[df['n_tokens_description_group'] == bin]['fullcot_restoration_error'].sum(), 'sum_unfaithfulshortcut_errors': df[df['n_tokens_description_group'] == bin]['fullcot_unfaithfulshortcut_error'].sum()}
        return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

    return_df['sum_restoration_errors'] = return_df['sum_restoration_errors'].astype(int)
    return_df['sum_unfaithfulshortcut_errors'] = return_df['sum_unfaithfulshortcut_errors'].astype(int)
    return return_df

def get_converted_explanationcorrectness_description_tokens(df, model):
    return_df = pd.DataFrame(columns=['cot_type','n_description_tokens','total_descriptions', 'sum_explanation_correctness_errors'])

    bins = ['0-100', '101-200', '201-300', '301-400', '401+']

    for bin in bins:
        row = {'model': model, 'cot_type': 'nocot', 'n_description_tokens': bin, 'total_descriptions': df[df['n_tokens_description_group'] == bin].shape[0], 'sum_explanation_correctness_errors': df[df['n_tokens_description_group'] == bin]['nocot_explanationcorrectness_error'].sum()}
        return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

        row = {'model': model, 'cot_type': 'somecot', 'n_description_tokens': bin, 'total_descriptions': df[df['n_tokens_description_group'] == bin].shape[0], 'sum_explanation_correctness_errors': df[df['n_tokens_description_group'] == bin]['somecot_explanationcorrectness_error'].sum()}
        return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

        row = {'model': model, 'cot_type': 'fullcot', 'n_description_tokens': bin, 'total_descriptions': df[df['n_tokens_description_group'] == bin].shape[0], 'sum_explanation_correctness_errors': df[df['n_tokens_description_group'] == bin]['fullcot_explanationcorrectness_error'].sum()}
        return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

    return_df['sum_explanation_correctness_errors'] = return_df['sum_explanation_correctness_errors'].astype(int)
    return return_df

def get_converted_natural_outcome(df, model):
    return_df = pd.DataFrame(columns=['cot_type', 'cot_outcome', 'total_outcomes', 'sum_restoration_errors', 'sum_unfaithfulshortcut_errors'])

    outcomes_nocot = df['nocot_outcome'].unique()
    outcomes_somecot = df['somecot_outcome'].unique()
    outcomes_fullcot = df['fullcot_outcome'].unique()

    for outcome in outcomes_nocot:
        row = {'model': model, 'cot_type': 'nocot', 'cot_outcome': outcome, 'total_outcomes': df[df['nocot_outcome'] == outcome].shape[0],'sum_restoration_errors': df[df['nocot_outcome'] == outcome]['nocot_restoration_error'].sum(), 'sum_unfaithfulshortcut_errors': df[df['nocot_outcome'] == outcome]['nocot_unfaithfulshortcut_error'].sum()}
        return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)
    
    for outcome in outcomes_somecot:
        row = {'model': model, 'cot_type': 'somecot', 'cot_outcome': outcome, 'total_outcomes': df[df['somecot_outcome'] == outcome].shape[0], 'sum_restoration_errors': df[df['somecot_outcome'] == outcome]['somecot_restoration_error'].sum(), 'sum_unfaithfulshortcut_errors': df[df['somecot_outcome'] == outcome]['somecot_unfaithfulshortcut_error'].sum()}
        return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

    for outcome in outcomes_fullcot:
        row = {'model': model, 'cot_type': 'fullcot', 'cot_outcome': outcome, 'total_outcomes': df[df['fullcot_outcome'] == outcome].shape[0], 'sum_restoration_errors': df[df['fullcot_outcome'] == outcome]['fullcot_restoration_error'].sum(), 'sum_unfaithfulshortcut_errors': df[df['fullcot_outcome'] == outcome]['fullcot_unfaithfulshortcut_error'].sum()}
        return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

    return_df['sum_restoration_errors'] = return_df['sum_restoration_errors'].astype(int)
    return_df['sum_unfaithfulshortcut_errors'] = return_df['sum_unfaithfulshortcut_errors'].astype(int)

    return_df['cot_outcome'] = return_df['cot_outcome'].astype(int)
    return_df['cot_outcome'] = return_df['cot_outcome'].map(reverse_replace_dict)

    return return_df

def get_converted_explanationcorrectness_outcome(df, model):
    return_df = pd.DataFrame(columns=['cot_type', 'cot_outcome', 'total_outcomes', 'sum_explanation_correctness_errors'])

    print(df.columns)

    df['nocot_outcome'] = df['nocot_response'].apply(lambda x: int(parse_response(x, '1')))
    df['somecot_outcome'] = df['somecot_response'].apply(lambda x: int(parse_response(x, '4')))
    df['fullcot_outcome'] = df['fullcot_response'].apply(lambda x: int(parse_response(x, '8')))

    outcomes_nocot = df['nocot_outcome'].unique()
    outcomes_somecot = df['somecot_outcome'].unique()
    outcomes_fullcot = df['fullcot_outcome'].unique()

    for outcome in outcomes_nocot:
        row = {'model': model, 'cot_type': 'nocot', 'cot_outcome': outcome, 'total_outcomes': df[df['nocot_outcome'] == outcome].shape[0],'sum_explanation_correctness_errors': df[df['nocot_outcome'] == outcome]['nocot_explanationcorrectness_error'].sum()}
        return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)
    
    for outcome in outcomes_somecot:
        row = {'model': model, 'cot_type': 'somecot', 'cot_outcome': outcome, 'total_outcomes': df[df['somecot_outcome'] == outcome].shape[0], 'sum_explanation_correctness_errors': df[df['somecot_outcome'] == outcome]['somecot_explanationcorrectness_error'].sum()}
        return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

    for outcome in outcomes_fullcot:
        row = {'model': model, 'cot_type': 'fullcot', 'cot_outcome': outcome, 'total_outcomes': df[df['fullcot_outcome'] == outcome].shape[0], 'sum_explanation_correctness_errors': df[df['fullcot_outcome'] == outcome]['fullcot_explanationcorrectness_error'].sum()}
        return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

    return_df['sum_explanation_correctness_errors'] = return_df['sum_explanation_correctness_errors'].astype(int)

    return_df['cot_outcome'] = return_df['cot_outcome'].astype(int)
    return_df['cot_outcome'] = return_df['cot_outcome'].map(reverse_replace_dict)

    return return_df

filepath = ''
models = ['mistralsmall24binstruct2501q80', 'deepseekr132b', 'phi414bq80']

natural_explanation_correctness_errors = pd.DataFrame(columns = ['model', 'no_cot_explanationcorrectness', 'some_cot_explanationcorrectness', 'full_cot_explanationcorrectness', 'n_tokens_nocot_response', 'n_tokens_somecot_response', 'n_tokens_fullcot_response', 'n_tokens_description'])
natural_restoration_error_responses = pd.DataFrame(columns = ['model', 'nocot_restoration_error', 'somecot_restoration_error', 'fullcot_restoration_error', 'n_tokens_nocot_response', 'n_tokens_somecot_response', 'n_tokens_fullcot_response', 'n_tokens_description'])
natural_unfaithful_shortcut_errors = pd.DataFrame(columns = ['model', 'nocot_unfaithfulshortcut_error', 'somecot_unfaithfulshortcut_error', 'fullcot_unfaithfulshortcut_error', 'n_tokens_nocot_response', 'n_tokens_somecot_response', 'n_tokens_fullcot_response', 'n_tokens_description'])

natural_grouped_response_tokens = pd.DataFrame()
natural_grouped_explanationcorrectness_response_tokens = pd.DataFrame()
natural_grouped_description_tokens = pd.DataFrame()
natural_grouped_explanationcorrectness_description_tokens = pd.DataFrame()
natural_grouped_outcome = pd.DataFrame()
natural_grouped_explanationcorrectness_outcome = pd.DataFrame()

simulated_errors_df = pd.DataFrame()
model_size_dict = {}

for m in models:
    print('\n\nProcessing model:', m)
    natural_filename = f'natural_results_{m}.xlsx'
    explanationcorrectness_filename = f'explanation_correctness_{m}.csv'
    simulated_filename = f'simulated_results_{m}.csv'
    print(f'Natural filename: {natural_filename}')
    print(f'Simulated filename: {simulated_filename}')

    if (os.path.exists(filepath + natural_filename) and (os.path.exists(filepath + simulated_filename)) and (os.path.exists(filepath + explanationcorrectness_filename))):
        natural_df = pd.read_excel(filepath + natural_filename)
        explanationcorrectness_df = pd.read_csv(filepath + explanationcorrectness_filename)
        explanationcorrectness_df = explanationcorrectness_df.rename(columns={'no_cot_explanationcorrectness': 'nocot_explanationcorrectness_error', 
                                                                              'some_cot_explanationcorrectness': 'somecot_explanationcorrectness_error', 
                                                                              'full_cot_explanationcorrectness': 'fullcot_explanationcorrectness_error',
                                                                              'no_cot_response': 'nocot_response',
                                                                              'some_cot_response': 'somecot_response',
                                                                              'full_cot_response': 'fullcot_response'})
        simulated_df = pd.read_csv(filepath + simulated_filename)
    else:
        print(f'Files not found for model {m}. Skipping...')
        continue

    ## details for each dataframe
    print(f'\nNatural DataFrame shape: {natural_df.shape}')
    print(f'\nExplanation Correctness DataFrame shape: {explanationcorrectness_df.shape}')
    print(f'\nSimulated DataFrame shape: {simulated_df.shape}')

    print('\nNatural DataFrame columns:', natural_df.columns.tolist())
    print('\nExplanation Correctness DataFrame columns:', explanationcorrectness_df.columns.tolist())
    print('\nSimulated DataFrame columns:', simulated_df.columns.tolist())

    print('\nNatural DataFrame head:')
    print(natural_df.head())
    print('\nExplanation Correctness DataFrame head:')
    print(explanationcorrectness_df.head())
    print('\nSimulated DataFrame head:')
    print(simulated_df.head())
   
    ## natural data
    natural_df['model'] = m
    natural_df['n_tokens_nocot_response'] = natural_df['nocot_response'].apply(lambda x: get_context_length(x))
    natural_df['n_tokens_somecot_response'] = natural_df['somecot_response'].apply(lambda x: get_context_length(x))
    natural_df['n_tokens_fullcot_response'] = natural_df['fullcot_response'].apply(lambda x: get_context_length(x))
    natural_df['n_tokens_description'] = natural_df['description'].apply(lambda x: get_context_length(x))

    natural_df['n_tokens_nocot_response_group'] = pd.cut(natural_df['n_tokens_nocot_response'], bins=[0, 500, 1000, 1500, np.inf], labels=['0-500', '501-1000', '1001-1500', '1501+'])
    natural_df['n_tokens_somecot_response_group'] = pd.cut(natural_df['n_tokens_somecot_response'], bins=[0, 500, 1000, 1500, np.inf], labels=['0-500', '501-1000', '1001-1500', '1501+'])
    natural_df['n_tokens_fullcot_response_group'] = pd.cut(natural_df['n_tokens_fullcot_response'], bins=[0, 500, 1000, 1500, np.inf], labels=['0-500', '501-1000', '1001-1500', '1501+'])
    natural_df['n_tokens_description_group'] = pd.cut(natural_df['n_tokens_description'], bins=[0, 100, 200, 300, 400, np.inf], labels=['0-100', '101-200', '201-300', '301-400', '401+'])

    explanationcorrectness_df['model'] = m
    explanationcorrectness_df['n_tokens_nocot_response'] = explanationcorrectness_df['nocot_response'].apply(lambda x: get_context_length(x))
    explanationcorrectness_df['n_tokens_somecot_response'] = explanationcorrectness_df['somecot_response'].apply(lambda x: get_context_length(x))
    explanationcorrectness_df['n_tokens_fullcot_response'] = explanationcorrectness_df['fullcot_response'].apply(lambda x: get_context_length(x))
    explanationcorrectness_df['n_tokens_description'] = explanationcorrectness_df['description'].apply(lambda x: get_context_length(x))

    explanationcorrectness_df['n_tokens_nocot_response_group'] = pd.cut(explanationcorrectness_df['n_tokens_nocot_response'], bins=[0, 500, 1000, 1500, np.inf], labels=['0-500', '501-1000', '1001-1500', '1501+'])
    explanationcorrectness_df['n_tokens_somecot_response_group'] = pd.cut(explanationcorrectness_df['n_tokens_somecot_response'], bins=[0, 500, 1000, 1500, np.inf], labels=['0-500', '501-1000', '1001-1500', '1501+'])
    explanationcorrectness_df['n_tokens_fullcot_response_group'] = pd.cut(explanationcorrectness_df['n_tokens_fullcot_response'], bins=[0, 500, 1000, 1500, np.inf], labels=['0-500', '501-1000', '1001-1500', '1501+'])
    explanationcorrectness_df['n_tokens_description_group'] = pd.cut(explanationcorrectness_df['n_tokens_description'], bins=[0, 100, 200, 300, 400, np.inf], labels=['0-100', '101-200', '201-300', '301-400', '401+'])

    natural_restoration_error_responses = pd.concat([natural_restoration_error_responses, natural_df[['model', 'nocot_restoration_error', 'somecot_restoration_error', 'fullcot_restoration_error', 'n_tokens_nocot_response', 'n_tokens_somecot_response', 'n_tokens_fullcot_response', 'n_tokens_description']]], ignore_index=True)
    natural_unfaithful_shortcut_errors = pd.concat([natural_unfaithful_shortcut_errors, natural_df[['model', 'nocot_unfaithfulshortcut_error', 'somecot_unfaithfulshortcut_error', 'fullcot_unfaithfulshortcut_error', 'n_tokens_nocot_response', 'n_tokens_somecot_response', 'n_tokens_fullcot_response', 'n_tokens_description']]], ignore_index=True)
    natural_explanation_correctness_errors = pd.concat([natural_explanation_correctness_errors, explanationcorrectness_df[['model', 'nocot_explanationcorrectness_error', 'somecot_explanationcorrectness_error', 'fullcot_explanationcorrectness_error', 'n_tokens_nocot_response', 'n_tokens_somecot_response', 'n_tokens_fullcot_response', 'n_tokens_description']]], ignore_index=True)

    converted_natural_response = get_converted_natural_response_tokens(natural_df, m)
    converted_explanationcorrectness_response = get_converted_explanationcorrectness_response_tokens(explanationcorrectness_df, m)
    natural_grouped_response_tokens = pd.concat([natural_grouped_response_tokens, converted_natural_response], ignore_index=True)
    natural_grouped_explanationcorrectness_response_tokens = pd.concat([natural_grouped_explanationcorrectness_response_tokens, converted_explanationcorrectness_response], ignore_index=True)

    converted_natural_descriptions = get_converted_natural_description_tokens(natural_df, m)
    converted_explanationcorrectness_descriptions = get_converted_explanationcorrectness_description_tokens(explanationcorrectness_df, m)
    natural_grouped_description_tokens = pd.concat([natural_grouped_description_tokens, converted_natural_descriptions], ignore_index=True)
    natural_grouped_explanationcorrectness_description_tokens = pd.concat([natural_grouped_explanationcorrectness_description_tokens, converted_explanationcorrectness_descriptions], ignore_index=True)

    converted_natural_outcomes = get_converted_natural_outcome(natural_df, m)
    converted_explanationcorrectness_outcomes = get_converted_explanationcorrectness_outcome(explanationcorrectness_df, m)
    natural_grouped_outcome = pd.concat([natural_grouped_outcome, converted_natural_outcomes], ignore_index=True)
    natural_grouped_explanationcorrectness_outcome = pd.concat([natural_grouped_explanationcorrectness_outcome, converted_explanationcorrectness_outcomes], ignore_index=True)

    ## simulated data
    simulated_df, iscorrect_colnames = get_simulated_iscorrect(simulated_df)
    transformed_simulated_df = get_converted_simulated_iscorrect(simulated_df, m)
    simulated_errors_df = pd.concat([simulated_errors_df, transformed_simulated_df], ignore_index=True)
    model_size_dict[m] = simulated_df.shape[0]

    ### get a separate dataset to check the number of hints - RUN ONCE
    # simulated_df_checkhint = simulated_df.sample(frac=1).reset_index(drop=True)
    # simulated_df_checkhint = simulated_df_checkhint.head(100)[['description', 'gt_outcome', 'nocot_noexamples_hint_response', 'somecot_noexamples_hint_response', 'fullcot_noexamples_hint_response']]
    # simulated_df_checkhint.to_csv(filepath + f'simulated_results_checkhint_{m}.csv', index=False)

# NATURAL ANALYSIS
print('\n\nToken Analysis:')
cols = ['n_tokens_nocot_response', 'n_tokens_somecot_response', 'n_tokens_fullcot_response', 'n_tokens_description']
summary = natural_restoration_error_responses[cols].agg([
    'count',
    'mean',
    'std',
    'min',
    lambda x: x.quantile(0.25),
    'median',
    lambda x: x.quantile(0.75),
    'max'
])
summary.index = ['count', 'mean', 'std', 'min', 'q1', 'median', 'q3', 'max']
print(summary)



print('\n\nGrouped RESPONSE Tokens DataFrame by COT Type (Explanation Correctness):')
print(natural_grouped_explanationcorrectness_response_tokens.shape)
print(natural_grouped_explanationcorrectness_response_tokens.columns.tolist())
print(natural_grouped_explanationcorrectness_response_tokens.head())

print('\n\nGrouped RESPONSE Tokens DataFrame by COT Type (Restoration and Unfaithful Shortcut Errors):')
print(natural_grouped_response_tokens.shape)
print(natural_grouped_response_tokens.columns.tolist())
print(natural_grouped_response_tokens.head())

natural_grouped_explanationcorrectness_response_tokens.to_csv(filepath + 'natural_grouped_explanationcorrectness_response_tokens.csv', index=False)
natural_grouped_response_tokens.to_csv(filepath + 'natural_grouped_response_tokens.csv', index=False)

natural_explanationcorrectness_response_bins_plot(natural_grouped_explanationcorrectness_response_tokens, filepath)
natural_response_bins_plot(natural_grouped_response_tokens, filepath)



print('\n\nGrouped DESCRIPTION Tokens DataFrame by COT Type (Explanation Correctness):')
print(natural_grouped_explanationcorrectness_description_tokens.shape)
print(natural_grouped_explanationcorrectness_description_tokens.columns.tolist())
print(natural_grouped_explanationcorrectness_description_tokens.head())

print('\n\nGrouped DESCRIPTION Tokens DataFrame by COT Type (Restoration and Unfaithful Shortcut Errors):')
print(natural_grouped_description_tokens.shape)
print(natural_grouped_description_tokens.columns.tolist())
print(natural_grouped_description_tokens.head())

natural_grouped_explanationcorrectness_description_tokens.to_csv(filepath + 'natural_grouped_explanationcorrectness_description_tokens.csv', index=False)
natural_grouped_description_tokens.to_csv(filepath + 'natural_grouped_description_tokens.csv', index=False)

natural_explanationcorrectness_desc_bins_plot(natural_grouped_explanationcorrectness_description_tokens, filepath)
natural_desc_bins_plot(natural_grouped_description_tokens, filepath)



print('\n\nGrouped OUTCOMES DataFrame by COT Type (Explanation Correctness):')
print(natural_grouped_explanationcorrectness_outcome.shape)
print(natural_grouped_explanationcorrectness_outcome.columns.tolist())
print(natural_grouped_explanationcorrectness_outcome.head())

print('\n\nGrouped OUTCOMES DataFrame by COT Type (Restoration and Unfaithful Shortcut Errors):')
print(natural_grouped_outcome.shape)
print(natural_grouped_outcome.columns.tolist())
print(natural_grouped_outcome.head())

natural_grouped_explanationcorrectness_outcome.to_csv(filepath + 'natural_grouped_explanationcorrectness_outcome.csv', index=False)
natural_grouped_outcome.to_csv(filepath + 'natural_grouped_outcome.csv', index=False)

natural_explanationcorrectness_outcome_plot(natural_grouped_explanationcorrectness_outcome, filepath)
natural_outcome_plot(natural_grouped_outcome, filepath)



## explanation correctness errors
print('\n\nExplanation Correctness Errors DataFrame:')
print(natural_explanation_correctness_errors.shape)
print(natural_explanation_correctness_errors.columns.tolist())
print(natural_explanation_correctness_errors.head())

sum_deepseek_nocot = natural_explanation_correctness_errors[natural_explanation_correctness_errors['model'] == 'deepseekr132b']['nocot_explanationcorrectness_error'].sum()
sum_mistral_nocot = natural_explanation_correctness_errors[natural_explanation_correctness_errors['model'] == 'mistralsmall24binstruct2501q80']['nocot_explanationcorrectness_error'].sum()
sum_phi_nocot = natural_explanation_correctness_errors[natural_explanation_correctness_errors['model'] == 'phi414bq80']['nocot_explanationcorrectness_error'].sum()

sum_deepseek_somecot = natural_explanation_correctness_errors[natural_explanation_correctness_errors['model'] == 'deepseekr132b']['somecot_explanationcorrectness_error'].sum()
sum_mistral_somecot = natural_explanation_correctness_errors[natural_explanation_correctness_errors['model'] == 'mistralsmall24binstruct2501q80']['somecot_explanationcorrectness_error'].sum()
sum_phi_somecot = natural_explanation_correctness_errors[natural_explanation_correctness_errors['model'] == 'phi414bq80']['somecot_explanationcorrectness_error'].sum()

sum_deepseek_fullcot = natural_explanation_correctness_errors[natural_explanation_correctness_errors['model'] == 'deepseekr132b']['fullcot_explanationcorrectness_error'].sum()
sum_mistral_fullcot = natural_explanation_correctness_errors[natural_explanation_correctness_errors['model'] == 'mistralsmall24binstruct2501q80']['fullcot_explanationcorrectness_error'].sum()
sum_phi_fullcot = natural_explanation_correctness_errors[natural_explanation_correctness_errors['model'] == 'phi414bq80']['fullcot_explanationcorrectness_error'].sum()

natural_plot(sum_deepseek_nocot, sum_mistral_nocot, sum_phi_nocot,
              sum_deepseek_somecot, sum_mistral_somecot, sum_phi_somecot,
              sum_deepseek_fullcot, sum_mistral_fullcot, sum_phi_fullcot,
              'Explanation Correctness Errors',
              filename=f"{filepath}natural_explanation_correctness_errors")

## restoration errors
print('\n\nNatural Restoration Errors DataFrame:')
print(natural_restoration_error_responses.shape)
print(natural_restoration_error_responses.columns.tolist())
print(natural_restoration_error_responses.head())

sum_DeepSeek_nocot = natural_restoration_error_responses[natural_restoration_error_responses['model'] == 'deepseekr132b']['nocot_restoration_error'].sum()
sum_mistral_nocot = natural_restoration_error_responses[natural_restoration_error_responses['model'] == 'mistralsmall24binstruct2501q80']['nocot_restoration_error'].sum()
sum_phi_nocot = natural_restoration_error_responses[natural_restoration_error_responses['model'] == 'phi414bq80']['nocot_restoration_error'].sum()

sum_DeepSeek_somecot = natural_restoration_error_responses[natural_restoration_error_responses['model'] == 'deepseekr132b']['somecot_restoration_error'].sum()
sum_mistral_somecot = natural_restoration_error_responses[natural_restoration_error_responses['model'] == 'mistralsmall24binstruct2501q80']['somecot_restoration_error'].sum()
sum_phi_somecot = natural_restoration_error_responses[natural_restoration_error_responses['model'] == 'phi414bq80']['somecot_restoration_error'].sum()

sum_DeepSeek_fullcot = natural_restoration_error_responses[natural_restoration_error_responses['model'] == 'deepseekr132b']['fullcot_restoration_error'].sum()
sum_mistral_fullcot = natural_restoration_error_responses[natural_restoration_error_responses['model'] == 'mistralsmall24binstruct2501q80']['fullcot_restoration_error'].sum()
sum_phi_fullcot = natural_restoration_error_responses[natural_restoration_error_responses['model'] == 'phi414bq80']['fullcot_restoration_error'].sum()

natural_plot(sum_DeepSeek_nocot, sum_mistral_nocot, sum_phi_nocot,
              sum_DeepSeek_somecot, sum_mistral_somecot, sum_phi_somecot,
              sum_DeepSeek_fullcot, sum_mistral_fullcot, sum_phi_fullcot,
              'Restoration Errors',
              filename=f"{filepath}natural_restoration_errors")

# unfaithful shortcut errors
print('\n\nNatural Unfaithful Shortcut Errors DataFrame:')
print(natural_unfaithful_shortcut_errors.shape)
print(natural_unfaithful_shortcut_errors.columns.tolist())
print(natural_unfaithful_shortcut_errors.head())

sum_DeepSeek_nocot_unfaithful = natural_unfaithful_shortcut_errors[natural_unfaithful_shortcut_errors['model'] == 'deepseekr132b']['nocot_unfaithfulshortcut_error'].sum()
sum_mistral_nocot_unfaithful = natural_unfaithful_shortcut_errors[natural_unfaithful_shortcut_errors['model'] == 'mistralsmall24binstruct2501q80']['nocot_unfaithfulshortcut_error'].sum()
sum_phi_nocot_unfaithful = natural_unfaithful_shortcut_errors[natural_unfaithful_shortcut_errors['model'] == 'phi414bq80']['nocot_unfaithfulshortcut_error'].sum()

sum_DeepSeek_somecot_unfaithful = natural_unfaithful_shortcut_errors[natural_unfaithful_shortcut_errors['model'] == 'deepseekr132b']['somecot_unfaithfulshortcut_error'].sum()
sum_mistral_somecot_unfaithful = natural_unfaithful_shortcut_errors[natural_unfaithful_shortcut_errors['model'] == 'mistralsmall24binstruct2501q80']['somecot_unfaithfulshortcut_error'].sum()
sum_phi_somecot_unfaithful = natural_unfaithful_shortcut_errors[natural_unfaithful_shortcut_errors['model'] == 'phi414bq80']['somecot_unfaithfulshortcut_error'].sum()

sum_DeepSeek_fullcot_unfaithful = natural_unfaithful_shortcut_errors[natural_unfaithful_shortcut_errors['model'] == 'deepseekr132b']['fullcot_unfaithfulshortcut_error'].sum()
sum_mistral_fullcot_unfaithful = natural_unfaithful_shortcut_errors[natural_unfaithful_shortcut_errors['model'] == 'mistralsmall24binstruct2501q80']['fullcot_unfaithfulshortcut_error'].sum()
sum_phi_fullcot_unfaithful = natural_unfaithful_shortcut_errors[natural_unfaithful_shortcut_errors['model'] == 'phi414bq80']['fullcot_unfaithfulshortcut_error'].sum()

natural_plot(sum_DeepSeek_nocot_unfaithful, sum_mistral_nocot_unfaithful, sum_phi_nocot_unfaithful,
              sum_DeepSeek_somecot_unfaithful, sum_mistral_somecot_unfaithful, sum_phi_somecot_unfaithful,
              sum_DeepSeek_fullcot_unfaithful, sum_mistral_fullcot_unfaithful, sum_phi_fullcot_unfaithful,
              'Unfaithful Shortcut Errors',
              filename=f"{filepath}natural_unfaithful_shortcut_errors")

# SIMULATED ANALYSIS
print('\n\nSimulated Errors DataFrame:')
print(simulated_errors_df.shape)
print(simulated_errors_df.columns.tolist())
print(simulated_errors_df.head())

simulated_plot(simulated_errors_df, model_size_dict, filepath)


print('\n\nHint Only Analysis:')

simulated_df_dict = {}

for m in models:
    print(f'\nModel: {m}')
    simulated_filename = f'simulated_results_{m}.csv'
    if os.path.exists(filepath + simulated_filename):
        simulated_df = pd.read_csv(filepath + simulated_filename)
        simulated_df_dict[m] = simulated_df
        simulated_df['no_cot_parsed_response_shifted'] = simulated_df['no_cot_parsed_response'].apply(lambda x: hint_replace_dict.get(x, x))
        simulated_df['some_cot_parsed_response_shifted'] = simulated_df['some_cot_parsed_response'].apply(lambda x: hint_replace_dict.get(x, x))
        simulated_df['full_cot_parsed_response_shifted'] = simulated_df['full_cot_parsed_response'].apply(lambda x: hint_replace_dict.get(x, x))
        hint_only_responses = [
            'nocot_noexamples_hint_parsed_response',
            'somecot_noexamples_hint_parsed_response',
            'fullcot_noexamples_hint_parsed_response'
        ]
    else:
        print(f'Files not found for model {m}. Skipping...')
        continue

cot_types = ['nocot', 'somecot', 'fullcot']
plot_panel_discrete_distribution_overlay(
    simulated_df_dict,
    models,
    cot_types,
    hint_only_responses,
    savepath=filepath + "panel_discrete_distribution_overlay.png",
    show_plots=True
)


print('\n\nExamples-Only Analysis:')
plot_examples_only_percent_difference(
    simulated_df_dict,
    models,
    savepath=filepath + "examples_only_percent_difference.png",
    show_plots=True
)