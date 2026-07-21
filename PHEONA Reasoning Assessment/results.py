import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import tiktoken
from scipy.stats import chi2_contingency, ttest_rel
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score

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

models = ['mistralsmall24binstruct2501q4KM', 'deepseekr132bqwendistillq4KM', 'phi414bq4KM']


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


def calculate_cohen_kappa(df, col1, col2):
    """
    Calculate Cohen's Kappa for inter-annotator agreement between two trials of the same reviewer.
    Assumes df has columns error_type + '_trial1' and error_type + '_trial2'
    Returns: (kappa, lower_ci_95, upper_ci_95)
    """
    
    rater1 = np.array(df[col1].tolist())
    rater2 = np.array(df[col2].tolist())
    
    # Check if both columns contain only a single unique label
    unique_rater1 = np.unique(rater1)
    unique_rater2 = np.unique(rater2)
    
    if len(unique_rater1) == 1 and len(unique_rater2) == 1:
        # Both raters only have one label
        if unique_rater1[0] == unique_rater2[0]:
            # Same single label - perfect agreement
            return 1.0, 1.0, 1.0
        else:
            # Different single labels - perfect disagreement
            return -1.0, -1.0, -1.0
    
    # Calculate kappa
    kappa = cohen_kappa_score(rater1, rater2)
    
    # # Bootstrap confidence intervals
    # n_bootstrap = 10000
    # kappas_bootstrap = []
    # n_samples = len(rater1)
    
    # np.random.seed(42)
    # for _ in range(n_bootstrap):
    #     indices = np.random.choice(n_samples, n_samples, replace=True)
    #     kappa_boot = cohen_kappa_score(rater1[indices], rater2[indices])
    #     kappas_bootstrap.append(kappa_boot)
    
    # # Calculate 95% CI
    # lower_ci = np.percentile(kappas_bootstrap, 2.5)
    # upper_ci = np.percentile(kappas_bootstrap, 97.5)

    lower_ci = 0
    upper_ci = 1
    
    return kappa, lower_ci, upper_ci


def plot_inter_annotator_agreement_heatmap_panel(df, FILEPATH, show_plots=False):
    """
    Create a 3-panel figure for inter-annotator agreement by error type.
    Each panel shows a lower-triangle heatmap with reviewer axes and hierarchical
    model/COT labels so each model has No CoT, Some CoT, and Full CoT in one panel.
    """
    error_groups = [
        ('explanationcorrectness', 'Explanation Correctness Errors'),
        ('unfaithfulshortcut_error', 'Unfaithful Shortcut Errors'),
        ('restoration_error', 'Restoration Errors'),
    ]
    model_order = ['deepseekr132bqwendistillq4KM', 'mistralsmall24binstruct2501q4KM', 'phi414bq4KM']
    model_labels = {
        'deepseekr132bqwendistillq4KM': 'DeepSeek',
        'mistralsmall24binstruct2501q4KM': 'Mistral',
        'phi414bq4KM': 'Phi'
    }
    cot_order = ['nocot', 'somecot', 'fullcot']
    cot_labels = {'nocot': 'No CoT', 'somecot': 'Some CoT', 'fullcot': 'Full CoT'}
    categories = [(model, cot) for model in model_order for cot in cot_order]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True, sharey=True)
    fig.subplots_adjust(bottom=0.28)

    # Kappa buckets and colors (standard interpretation)
    buckets = [(-1.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.0)]
    bucket_labels = ['None', 'Minimal', 'Weak', 'Moderate', 'Strong', 'Almost Perfect']
    # match project colors: blend from light grey to main error color '#ac2e44'
    palette = sns.blend_palette(['#d3d3d3', '#ac2e44'], n_colors=len(buckets))

    def _bucket_color(val):
        if np.isnan(val):
            return '#d3d3d3'
        for b, color in zip(buckets, palette):
            if b[0] <= val <= b[1]:
                return color
        return palette[-1]

    # Precompute global x positions and common y-limits across panels
    n_models = len(model_order)
    n_cots = len(cot_order)
    xs_global = np.arange(n_models * n_cots)
    categories = [(model, cot) for model in model_order for cot in cot_order]

    # compute global y-limits from available CI bounds so negative error bars are included
    all_lows = []
    all_highs = []
    for error_key, _ in error_groups:
        ed = df[df['error_name'].str.contains(error_key, case=False, na=False)].copy()
        for _, r in ed.iterrows():
            if not pd.isna(r.get('lower_95_ci')):
                all_lows.append(float(r['lower_95_ci']))
            if not pd.isna(r.get('upper_95_ci')):
                all_highs.append(float(r['upper_95_ci']))
    if len(all_lows) and len(all_highs):
        global_ymax = max(all_highs)
        margin = max(0.05 * global_ymax, 0.02)
        global_ylim = (0.0, global_ymax + margin)
    else:
        global_ylim = (0.0, 1.0)

    for ax, (error_key, title) in zip(axes, error_groups):
        error_df = df[df['error_name'].str.contains(error_key, case=False, na=False)].copy()

        # build plotting table
        rows = []
        for model in model_order:
            for cot in cot_order:
                match = error_df[(error_df['model'] == model) & (error_df['error_name'].str.contains(cot, case=False, na=False))]
                if not match.empty:
                    k = float(match['cohens_kappa'].values[0])
                    lo = float(match['lower_95_ci'].values[0])
                    hi = float(match['upper_95_ci'].values[0])
                else:
                    k, lo, hi = np.nan, np.nan, np.nan
                rows.append({'model': model, 'cot': cot, 'kappa': k, 'lo': lo, 'hi': hi})

        plot_df = pd.DataFrame(rows)

        # make bars thick and leave a small gap between model groups
        bar_width = 1.0
        group_gap = 0.8

        xs = []
        heights = []
        yerr = [[], []]
        colors = []

        for m_i, model in enumerate(model_order):
            base = m_i * (n_cots + group_gap)
            for c_i, cot in enumerate(cot_order):
                x = base + c_i
                xs.append(x)
                row = plot_df[(plot_df['model'] == model) & (plot_df['cot'] == cot)].iloc[0]
                heights.append(row['kappa'])
                if np.isnan(row['kappa']):
                    yerr[0].append(0)
                    yerr[1].append(0)
                else:
                    yerr[0].append(row['kappa'] - row['lo'])
                    yerr[1].append(row['hi'] - row['kappa'])
                colors.append(_bucket_color(row['kappa']))

        xs = np.array(xs)
        heights = np.array(heights, dtype=float)

        # Add grey borders (no error bars) to match natural_plot appearance
        bars = ax.bar(
            xs, heights, width=bar_width,
            color=colors, edgecolor='#3a414a', align='center'
        )

        # X-axis: CoT labels at the bar positions, model group labels below
        ax.set_xticks(xs)
        ax.set_xticklabels([cot_labels[cot] for _, cot in categories], rotation=90, ha='center', fontsize=20, color='#3a414a')
        ax.tick_params(axis='x', which='both', length=0, pad=4)
        ax.xaxis.set_ticks_position('none')

        # add explicit model labels centered under the middle CoT bar for each model
        for m_i, model in enumerate(model_order):
            group_start = m_i * n_cots
            middle_index = group_start + n_cots // 2
            if middle_index >= len(xs):
                continue
            group_center = xs[middle_index]
            ax.text(
                group_center,
                -0.275,
                model_labels[model],
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='top',
                fontsize=22,
                fontweight='bold',
                color='#3a414a',
                clip_on=False
            )

        # enforce shared y-limits across all panels (includes CI bounds)
        ax.set_ylim(global_ylim)
        ax.set_title(title, fontsize=22, fontweight='bold', color='#3a414a')
        # horizontal gridlines similar to other plots
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, which='major', color='#b0b0b0', linewidth=1.0, alpha=0.6)
        ax.yaxis.grid(True, which='minor', color='#e0e0e0', linewidth=0.7, alpha=0.4)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(plt.NullLocator())

        # ensure x-limits include full bars and leave room for labels
        ax.set_xlim(xs.min() - bar_width, xs.max() + bar_width)

        # collect legend handles once (will add to right-most axis later)
        handles = [plt.Rectangle((0, 0), 1, 1, color=palette[i]) for i in range(len(buckets))]
        legend_handles = handles

    # remove all but left-most y-axis labels
    for i, ax in enumerate(axes):
        if i > 0:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Cohen's Kappa Coefficient", fontsize=22)
            ax.tick_params(labelsize=20)

    # place legend inside the upper-right corner of the right-most subplot
    axes[0].legend(legend_handles, bucket_labels, title='Agreement', loc='upper left', fontsize=17, title_fontsize=17)
    if FILEPATH is not None:
        output_path = os.fspath(FILEPATH)
        base, ext = os.path.splitext(output_path)
        if ext.lower() in {'.png', '.svg'}:
            save_dir = os.path.dirname(output_path) or '.'
            filename = os.path.basename(base)
        else:
            save_dir = output_path
            filename = 'inter_annotator_agreement_panel'

        if os.path.exists(save_dir) and not os.path.isdir(save_dir):
            raise FileExistsError(f'Output directory path exists as a file: {save_dir}')
        os.makedirs(save_dir, exist_ok=True)

        prefix = os.path.join(save_dir, filename)
        plt.savefig(prefix + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(prefix + '.svg', bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)


def natural_plot(df_expl, df_unfaithful, df_restore, filename, error_titles=None, show_plots=show_plots):
    """
    Plot three error types side-by-side. Each subplot shows the three CoT
    values (No/Some/Full) averaged across trials for each model with standard
    deviation error bars. Produces a single legend and a single shared y-axis.

    Args:
        df (pd.DataFrame): trial-level data with columns like
            'nocot_<suffix>', 'somecot_<suffix>', 'fullcot_<suffix>' and a
            `model` column matching the expected model codes.
        error_suffixes (list[str]): list of three suffixes, e.g.
            ['explanationcorrectness_error','restoration_error','unfaithfulshortcut_error']
        filename (str): output filename (without extension) or path prefix.
        error_titles (list[str], optional): display titles for the three panels.
        show_plots (bool): whether to call `plt.show()`.
    """

    # Accept three dataframes: explanation correctness, unfaithful shortcut, restoration
    dfs = [df_expl, df_unfaithful, df_restore]
    # derive suffix for each df by finding a column that starts with 'nocot_'
    error_suffixes = []
    for d in dfs:
        suffix = None
        for col in d.columns:
            if col.startswith('nocot_'):
                suffix = col.replace('nocot_', '')
                break
        if suffix is None:
            # fallback: try somecot_
            for col in d.columns:
                if col.startswith('somecot_'):
                    suffix = col.replace('somecot_', '')
                    break
        if suffix is None:
            raise ValueError('Could not detect error column suffix in one of the provided DataFrames')
        error_suffixes.append(suffix)

    if error_titles is None:
        default_titles = {
            error_suffixes[0]: 'Explanation Correctness Errors',
            error_suffixes[1]: 'Unfaithful Shortcut Errors',
            error_suffixes[2]: 'Restoration Errors'
        }
        error_titles = [default_titles.get(s, s) for s in error_suffixes]

    # expected model identifiers in the data
    model_codes = ['deepseekr132bqwendistillq4KM', 'mistralsmall24binstruct2501q4KM', 'phi414bq4KM']
    display_names = {'deepseekr132bqwendistillq4KM': 'DeepSeek',
                     'mistralsmall24binstruct2501q4KM': 'Mistral',
                     'phi414bq4KM': 'Phi'}

    colors = {'DeepSeek': '#b44357', 'Mistral': '#264851', 'Phi': '#72ccae'}
    cot_keys = ['nocot', 'somecot', 'fullcot']
    cot_labels = ['No CoT', 'Some CoT', 'Full CoT']

    n_panels = len(error_suffixes)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6), constrained_layout=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    # compute statistics for all combinations
    stats = {}
    all_means = []
    all_stds = []
    for suffix, d in zip(error_suffixes, dfs):
        stats[suffix] = {}
        for code in model_codes:
            means = []
            stds = []
            for cot in cot_keys:
                col = f"{cot}_{suffix}"
                if col in d.columns:
                    vals = pd.to_numeric(d.loc[d['model'] == code, col], errors='coerce').dropna().values
                else:
                    vals = np.array([])
                if vals.size > 0:
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals, ddof=0)))
                else:
                    means.append(np.nan)
                    stds.append(0.0)
            stats[suffix][code] = (np.array(means), np.array(stds))
            all_means.extend([m for m in means if not np.isnan(m)])
            all_stds.extend(stds)

    # determine y-limits
    if len(all_means):
        y_max = max(all_means) + max(all_stds)
        y_max *= 1.08
    else:
        y_max = 1.0

    # Plot: each panel is an error type; within panel, grouped bars per CoT with one bar per model
    model_handles = []
    model_labels = []
    hatch_types = ['///', None, '\\\\']
    n_models = len(model_codes)
    x = np.arange(len(cot_keys))
    width = 0.22
    for ax_idx, (ax, suffix, title) in enumerate(zip(axes, error_suffixes, error_titles)):
        for i, code in enumerate(model_codes):
            display = display_names.get(code, code)
            means, stds = stats[suffix][code]
            color = colors.get(display, '#333333')
            offset = (i - (n_models - 1) / 2.0) * width
            rects = ax.bar(x + offset, means, width, yerr=stds, capsize=7,
                           color=color, edgecolor='#3a414a', hatch=hatch_types[i], align='center',
                           error_kw={'linewidth': 3.0, 'capthick': 3.0})
            # collect one handle per model from the first axis for a shared legend
            if ax is axes[0]:
                model_handles.append(rects[0])
                model_labels.append(display)

        ax.set_xticks(x)
        ax.set_xticklabels(cot_labels, fontsize=20, color='#3a414a')
        ax.set_title(title, fontsize=22, fontweight='bold', color='#3a414a')
        # only set x-axis title on the middle plot
        if ax_idx == 1:
            ax.set_xlabel('CoT Type', fontsize=22, color='#3a414a')
        ax.set_ylim(0, max(1.0, y_max))
        # gridlines similar to other plots
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, which='major', color='#b0b0b0', linewidth=1.2, alpha=0.9)
        ax.yaxis.grid(True, which='minor', color='#e0e0e0', linewidth=0.7, alpha=0.6)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(plt.NullLocator())

    # shared y-axis label on left-most plot only
    axes[0].set_ylabel('Average Number of Responses', fontsize=22, color='#3a414a')
    axes[0].tick_params(axis='y', labelsize=20)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    # single legend placed in the top-right of the right-most subplot
    axes[-1].legend(model_handles, model_labels, loc='upper right', ncol=len(model_labels), fontsize=16, title_fontsize=16, title='Model')

    plt.savefig(f"{filename}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{filename}.svg", bbox_inches='tight')
    if show_plots:
        plt.show()


def get_simulated_iscorrect(df):
    df['outcome'] = df['outcome'].astype(int)

    cot_types = ['nocot', 'somecot', 'fullcot']
    example_types = ['randomfewshot', 'specificfewshot']
    hint_types = ['nohint', 'hint']

    iscorrect_colnames = []
    outcome_colnames = []
    for cot in cot_types:
        for example in example_types:
            for hint in hint_types:
                df[f'{cot}_{example}_{hint}_parsed_response'] = df[f'{cot}_{example}_{hint}_parsed_response'].astype(int)
                colname = f'iscorrect_{cot}_{example}_{hint}'
                df[colname] = np.where(df[f'{cot}_{example}_{hint}_parsed_response'] == df['outcome'], 1, 0)
                iscorrect_colnames.append(colname)
                outcome_colnames.append(f'{cot}_{example}_{hint}_parsed_response')

    additional_columns = [
        'nocot_parsed_response', 'somecot_parsed_response', 'fullcot_parsed_response',
        'nocot_noexamples_hint_parsed_response', 'somecot_noexamples_hint_parsed_response', 'fullcot_noexamples_hint_parsed_response'
    ]
    df[additional_columns] = df[additional_columns].astype(int)
    for col in additional_columns:
        outcome_colnames.append(col)
        colname = f'iscorrect_{col}'
        iscorrect_colnames.append(colname)
        df[colname] = np.where(df[col] == df['outcome'], 1, 0)

    return df, iscorrect_colnames, outcome_colnames


def draw_vertical_pathway(ax, x, y_vals, color='#b0b0b0', alpha=0.25, text_color='#3a414a', diff_bold_color='#ac2e44'):
    y_vals_sorted = sorted(y_vals, key=lambda tup: tup[1], reverse=True)
    if len(y_vals_sorted) >= 2:
        y_poly = [y for _, y, *_ in y_vals_sorted]
        ax.fill_betweenx(y_poly, x-0.18, x+0.18, color=color, alpha=alpha, zorder=1)
        for i in range(len(y_vals_sorted)-1):
            y0 = y_vals_sorted[i][1]
            y1 = y_vals_sorted[i+1][1]
            # Removed the vertical line: ax.plot([x, x], [y0, y1], color=color, alpha=0.7, zorder=2)
            diff = y0 - y1
            ym = (y0 + y1) / 2
    for label, y, c, m, z in y_vals_sorted:
        ax.scatter(x, y, label=label, color=c, marker=m, s=750, zorder=z)


def simulated_plot(df, filepath, show_plots=True):
    custom_color = '#3a414a'
    diff_bold_color = '#ac2e44'

    model_names_to_display = {
        'mistralsmall24binstruct2501q4KM': 'Mistral',
        'deepseekr132bqwendistillq4KM': 'DeepSeek',
        'phi414bq4KM': 'Phi'
    }

    cot_type_order = ['nocot', 'somecot', 'fullcot']
    cot_type_labels = ['No CoT', 'Some CoT', 'Full CoT']
    cot_type_map = {k: i for i, k in enumerate(cot_type_order)}

    def get_column_value(df_model, candidates):
        for col in candidates:
            if col in df_model.columns:
                val = df_model[col].iloc[0]
                if pd.notna(val):
                    return float(val)
        return np.nan

    models = df['model'].unique()
    fig, axes = plt.subplots(len(models), 3, figsize=(18, 5 * len(models)), sharex='col', constrained_layout=True)
    if len(models) == 1:
        axes = [axes]

    # Legend handles
    legend_elements_random = [
        plt.Line2D([0], [0], marker='h', color='w', label='RandomFewShot + Hint', markerfacecolor='#ac2e44', markersize=28),
        plt.Line2D([0], [0], marker='p', color='w', label='RandomFewShot + NoHint', markerfacecolor='#0e343e', markersize=28),
        plt.Line2D([0], [0], marker='8', color='w', label='Unbiased', markerfacecolor='#72ccae', markersize=28)
    ]
    legend_elements_specific = [
        plt.Line2D([0], [0], marker='h', color='w', label='SpecificFewShot + Hint', markerfacecolor='#ac2e44', markersize=28),
        plt.Line2D([0], [0], marker='p', color='w', label='SpecificFewShot + NoHint', markerfacecolor='#0e343e', markersize=28),
        plt.Line2D([0], [0], marker='8', color='w', label='Unbiased', markerfacecolor='#72ccae', markersize=28)
    ]
    legend_elements_panel3 = [
        plt.Line2D([0], [0], marker='h', color='w', label='Hint Only', markerfacecolor='#ac2e44', markersize=28),
        plt.Line2D([0], [0], marker='8', color='w', label='Unbiased', markerfacecolor='#72ccae', markersize=28)
    ]

    # --- 1. Gather all y-values for all panels ---
    all_yvals = []
    for model in models:
        df_model = df[df['model'] == model]
        if df_model.empty:
            continue

        for cot in cot_type_order:
            values = [
                get_column_value(df_model, [f'iscorrect_{cot}_randomfewshot_hint', f'iscorrect_{cot}_randomfewshot_hint_parsed_response']),
                get_column_value(df_model, [f'iscorrect_{cot}_randomfewshot_nohint', f'iscorrect_{cot}_randomfewshot_nohint_parsed_response']),
                get_column_value(df_model, [f'iscorrect_{cot}_parsed_response']),
                get_column_value(df_model, [f'iscorrect_{cot}_specificfewshot_hint', f'iscorrect_{cot}_specificfewshot_hint_parsed_response']),
                get_column_value(df_model, [f'iscorrect_{cot}_specificfewshot_nohint', f'iscorrect_{cot}_specificfewshot_nohint_parsed_response']),
                get_column_value(df_model, [f'iscorrect_{cot}_parsed_response']),
                get_column_value(df_model, [f'iscorrect_{cot}_parsed_response']),
                get_column_value(df_model, [f'iscorrect_{cot}_noexamples_hint_parsed_response', f'iscorrect_{cot}_noexamples_hint'])
            ]
            all_yvals.extend([v for v in values if not np.isnan(v)])

    ## uncomment for regular accuracy
    if all_yvals:
        global_min = min(all_yvals) - 0.025
        global_max = max(all_yvals) + 0.05
    else:
        global_min, global_max = 0, 0.48
    global_max = max(0.48, global_max)
    # global_min, global_max = 0.10, 0.225

    # --- 2. Plot using averaged values from df ---
    for idx, model in enumerate(models):
        df_model = df[df['model'] == model]
        if df_model.empty:
            continue

        ax1 = axes[idx][0] if len(models) > 1 else axes[0]
        ax2 = axes[idx][1] if len(models) > 1 else axes[1]
        ax3 = axes[idx][2] if len(models) > 1 else axes[2]

        # PANEL 1: RANDOM FEW SHOT
        ax1.set_axisbelow(True)
        ax1.yaxis.grid(True, which='major', color='#b0b0b0', linewidth=1.2, alpha=0.6)
        ax1.yaxis.grid(True, which='minor', color='#e0e0e0', linewidth=0.7, alpha=0.4)
        ax1.xaxis.grid(False)
        ax1.minorticks_on()

        for cot in cot_type_order:
            x = cot_type_map[cot]
            yvals = []
            hint_val = get_column_value(df_model, [f'iscorrect_{cot}_randomfewshot_hint', f'iscorrect_{cot}_randomfewshot_hint_parsed_response'])
            nohint_val = get_column_value(df_model, [f'iscorrect_{cot}_randomfewshot_nohint', f'iscorrect_{cot}_randomfewshot_nohint_parsed_response'])
            unbiased_val = get_column_value(df_model, [f'iscorrect_{cot}_parsed_response'])
            if not np.isnan(hint_val):
                yvals.append(('RandomFewShot + Hint', hint_val, '#ac2e44', 'h', 4))
            if not np.isnan(nohint_val):
                yvals.append(('RandomFewShot + NoHint', nohint_val, '#0e343e', 'p', 4))
            if not np.isnan(unbiased_val):
                yvals.append(('Unbiased', unbiased_val, '#72ccae', '8', 5))
            if yvals:
                draw_vertical_pathway(ax1, x, yvals, color='#b0b0b0', alpha=0.25, text_color=custom_color, diff_bold_color=diff_bold_color)

        ax1.set_ylabel('Accuracy', color=custom_color, fontsize=22)
        ax1.tick_params(axis='x', colors=custom_color, labelsize=20)
        ax1.tick_params(axis='y', colors=custom_color, labelsize=20)
        ax1.set_xticks([cot_type_map[c] for c in cot_type_order])
        ax1.set_xticklabels(cot_type_labels, color=custom_color, fontsize=22)
        plt.setp(ax1.get_yticklabels(), color=custom_color, fontsize=22)
        for spine in ax1.spines.values():
            spine.set_color(custom_color)
        ax1.set_ylim(global_min, global_max)
        if idx == 0:
            ax1.legend(handles=legend_elements_random, loc='upper left', fontsize=18)

        # PANEL 2: SPECIFIC FEW SHOT
        ax2.set_axisbelow(True)
        ax2.yaxis.grid(True, which='major', color='#b0b0b0', linewidth=1.2, alpha=0.6)
        ax2.yaxis.grid(True, which='minor', color='#e0e0e0', linewidth=0.7, alpha=0.4)
        ax2.xaxis.grid(False)
        ax2.minorticks_on()

        for cot in cot_type_order:
            x = cot_type_map[cot]
            yvals = []
            hint_val = get_column_value(df_model, [f'iscorrect_{cot}_specificfewshot_hint', f'iscorrect_{cot}_specificfewshot_hint_parsed_response'])
            nohint_val = get_column_value(df_model, [f'iscorrect_{cot}_specificfewshot_nohint', f'iscorrect_{cot}_specificfewshot_nohint_parsed_response'])
            unbiased_val = get_column_value(df_model, [f'iscorrect_{cot}_parsed_response'])
            if not np.isnan(hint_val):
                yvals.append(('SpecificFewShot + Hint', hint_val, '#ac2e44', 'h', 4))
            if not np.isnan(nohint_val):
                yvals.append(('SpecificFewShot + NoHint', nohint_val, '#0e343e', 'p', 4))
            if not np.isnan(unbiased_val):
                yvals.append(('Unbiased', unbiased_val, '#72ccae', '8', 5))
            if yvals:
                draw_vertical_pathway(ax2, x, yvals, color='#b0b0b0', alpha=0.25, text_color=custom_color, diff_bold_color=diff_bold_color)

        ax2.tick_params(axis='x', colors=custom_color, labelsize=20)
        ax2.tick_params(axis='y', colors=custom_color, labelsize=20)
        ax2.set_xticks([cot_type_map[c] for c in cot_type_order])
        ax2.set_xticklabels(cot_type_labels, color=custom_color, fontsize=20)
        plt.setp(ax2.get_yticklabels(), color=custom_color, fontsize=20)
        for spine in ax2.spines.values():
            spine.set_color(custom_color)
        ax2.set_ylim(global_min, global_max)
        if idx == 0:
            ax2.legend(handles=legend_elements_specific, loc='upper left', fontsize=18)

        # PANEL 3: UNBIASED VS HINT-ONLY (no examples)
        ax3.set_axisbelow(True)
        ax3.yaxis.grid(True, which='major', color='#b0b0b0', linewidth=1.2, alpha=0.6)
        ax3.yaxis.grid(True, which='minor', color='#e0e0e0', linewidth=0.7, alpha=0.4)
        ax3.xaxis.grid(False)
        ax3.minorticks_on()

        for cot in cot_type_order:
            x = cot_type_map[cot]
            yvals = []
            unbiased_val = get_column_value(df_model, [f'iscorrect_{cot}_parsed_response'])
            hint_only_val = get_column_value(df_model, [f'iscorrect_{cot}_noexamples_hint_parsed_response', f'iscorrect_{cot}_noexamples_hint'])
            if not np.isnan(unbiased_val):
                yvals.append(('Unbiased', unbiased_val, '#72ccae', '8', 5))
            if not np.isnan(hint_only_val):
                yvals.append(('Hint Only', hint_only_val, '#ac2e44', 'h', 4))
            if yvals:
                draw_vertical_pathway(ax3, x, yvals, color='#b0b0b0', alpha=0.25, text_color=custom_color, diff_bold_color=diff_bold_color)

        ax3.tick_params(axis='x', colors=custom_color, labelsize=20)
        ax3.tick_params(axis='y', colors=custom_color, labelsize=20)
        ax3.set_xticks([cot_type_map[c] for c in cot_type_order])
        ax3.set_xticklabels(cot_type_labels, color=custom_color, fontsize=20)
        plt.setp(ax3.get_yticklabels(), color=custom_color, fontsize=20)
        for spine in ax3.spines.values():
            spine.set_color(custom_color)
        ax3.set_ylim(global_min, global_max)
        if idx == 0:
            ax3.legend(handles=legend_elements_panel3, loc='upper left', fontsize=18)

        ax_for_label = ax3
        ylim = ax_for_label.get_ylim()
        ymid = (ylim[0] + ylim[1]) / 2
        ax_for_label.text(
            1.04, ymid, model_names_to_display.get(model, ''),
            color=custom_color, fontsize=22, fontweight='bold',
            va='center', ha='left', rotation=270,
            transform=ax_for_label.get_yaxis_transform()
        )

        # Hide y-axis on middle and right panels for this model row
        ax2.set_ylabel('')
        ax2.tick_params(axis='y', labelleft=False, left=False, right=False)
        ax2.spines['left'].set_visible(True)
        
        ax3.set_ylabel('')
        ax3.tick_params(axis='y', labelleft=False, left=False, right=False)
        ax3.spines['left'].set_visible(True)

    for row in axes:
        for ax in row:
            ax.set_xlabel('')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.075)
    fig.text(0.5, 0.03, 'CoT Type', ha='center', va='center', fontsize=22, color=custom_color)

    plt.savefig(filepath + "simulated_errors_plot.png", dpi=300)
    plt.savefig(filepath + "simulated_errors_plot.svg")
    if show_plots:
        plt.show()


RESULTS_FILEPATH = ''
INDIVIDUAL_FILEPATH = '/individual_reviewers/'
MODELS = ['phi414bq4KM', 'mistralsmall24binstruct2501q4KM', 'deepseekr132bqwendistillq4KM']

SKIP_COHENS_KAPPA = False

if not SKIP_COHENS_KAPPA:
    ## calculate inter-annotator agreement for error annotations (after converting to binary error/no error labels)
    all_rater_data = []

    print('\n-----------------------------------------------\n')
    print('Explanation correctness inter-annotator agreement calculation:\n')
    explanation_correctness_raters = ['SP', 'SY']

    all_explanation_correctness_df = pd.DataFrame()  # Initialize empty DataFrame to store all explanation correctness annotations across models and raters

    for m in MODELS:
        rater_data = pd.DataFrame()  # Initialize empty DataFrame to store rater annotations for this model
        for n_rater, rater_name in enumerate(explanation_correctness_raters):
            explanationcorrectness_filename = f'explanation_correctness_{m}_{rater_name}.xlsx'
            explanationcorrectness_filepath = INDIVIDUAL_FILEPATH + explanationcorrectness_filename
            print(f'Checking for file: {explanationcorrectness_filepath}')
            if os.path.exists(explanationcorrectness_filepath):
                explanationcorrectness_df = pd.read_excel(explanationcorrectness_filepath)
                print(f'\nLoaded file: {explanationcorrectness_filepath} with shape {explanationcorrectness_df.shape}')
                print(f'Columns in the file: {explanationcorrectness_df.columns.tolist()}')
                print(explanationcorrectness_df.head())
                explanationcorrectness_df['model'] = m  # Add model column to the explanation correctness DataFrame
                explanationcorrectness_df['trial'] = explanationcorrectness_df['trial'].astype(int) + 3 * n_rater  # Adjust trial numbers to be unique across raters (assuming each rater has 3 trials
                all_explanation_correctness_df = pd.concat([all_explanation_correctness_df, explanationcorrectness_df], ignore_index=True)  # Append to master DataFrame
                rater_data[f'nocot_explanationcorrectness_{n_rater+1}'] = explanationcorrectness_df['nocot_explanationcorrectness_error'].astype(int)
                rater_data[f'somecot_explanationcorrectness_{n_rater+1}'] = explanationcorrectness_df['somecot_explanationcorrectness_error'].astype(int)
                rater_data[f'fullcot_explanationcorrectness_{n_rater+1}'] = explanationcorrectness_df['fullcot_explanationcorrectness_error'].astype(int)
            else:
                print(f'Explanation correctness file not found for model {m}. Skipping inter-annotator agreement calculation.')

        print(f'\nConstructed rater DataFrame for model {m} with shape {rater_data.shape} and columns: {rater_data.columns.tolist()}')
        print(f'Rater DataFrame head for model {m}:\n{rater_data.head()}')

        for error_type in ['nocot_explanationcorrectness', 'somecot_explanationcorrectness', 'fullcot_explanationcorrectness']:
            print(f'\nCalculating Cohen\'s Kappa for {error_type} in model {m} using columns: {f"{error_type}_1"} and {f"{error_type}_2"}')
            kappa, lower_ci, upper_ci = calculate_cohen_kappa(rater_data, f'{error_type}_1', f'{error_type}_2')
            print(f'Inter-annotator agreement (Cohen\'s Kappa) for {error_type} in model {m}: {kappa:.3f} (95% CI: {lower_ci:.3f} - {upper_ci:.3f})')
            row = {'model': m, 'error_name': error_type, 'cohens_kappa': kappa, 'lower_95_ci': lower_ci, 'upper_95_ci': upper_ci}
            all_rater_data.append(row)


    print('\n-----------------------------------------------\n')
    print('Unfaithful shortcut and restoration error inter-annotator agreement calculation:\n')
    unfaithful_restoration_errors_raters = ['SP', 'DM']

    all_unfaithful_restoration_errors_df = pd.DataFrame()  # Initialize empty DataFrame to store all unfaithful shortcut and restoration error annotations across models and raters

    for m in MODELS:
        rater_data = pd.DataFrame()  # Initialize empty DataFrame to store rater annotations for this model
        for n_rater, rater_name in enumerate(unfaithful_restoration_errors_raters):
            unfaithful_restoration_filename = f'restoration_unfaithful_{m}_{rater_name}.xlsx'
            unfaithful_restoration_filepath = INDIVIDUAL_FILEPATH + unfaithful_restoration_filename
            print(f'Checking for file: {unfaithful_restoration_filepath}')
            if os.path.exists(INDIVIDUAL_FILEPATH + unfaithful_restoration_filename):
                unfaithful_restoration_df = pd.read_excel(unfaithful_restoration_filepath)
                print(f'\nLoaded file: {unfaithful_restoration_filepath} with shape {unfaithful_restoration_df.shape}')
                print(f'Columns in the file: {unfaithful_restoration_df.columns.tolist()}')
                print(unfaithful_restoration_df.head())
                unfaithful_restoration_df['model'] = m  # Add model column to the unfaithful/restoration DataFrame
                unfaithful_restoration_df['trial'] = unfaithful_restoration_df['trial'].astype(int) + 3 * n_rater
                all_unfaithful_restoration_errors_df = pd.concat([all_unfaithful_restoration_errors_df, unfaithful_restoration_df], ignore_index=True)  # Append to master DataFrame
                rater_data[f'nocot_unfaithfulshortcut_error_{n_rater+1}'] = unfaithful_restoration_df['nocot_unfaithfulshortcut_error'].astype(int)
                rater_data[f'somecot_unfaithfulshortcut_error_{n_rater+1}'] = unfaithful_restoration_df['somecot_unfaithfulshortcut_error'].astype(int)
                rater_data[f'fullcot_unfaithfulshortcut_error_{n_rater+1}'] = unfaithful_restoration_df['fullcot_unfaithfulshortcut_error'].astype(int)
                rater_data[f'nocot_restoration_error_{n_rater+1}'] = unfaithful_restoration_df['nocot_restoration_error'].astype(int)
                rater_data[f'somecot_restoration_error_{n_rater+1}'] = unfaithful_restoration_df['somecot_restoration_error'].astype(int)
                rater_data[f'fullcot_restoration_error_{n_rater+1}'] = unfaithful_restoration_df['fullcot_restoration_error'].astype(int)
            else:
                print(f'Unfaithful shortcut error file not found for model {m}. Skipping inter-annotator agreement calculation.')

        print(f'\nConstructed rater DataFrame for model {m} with shape {rater_data.shape} and columns: {rater_data.columns.tolist()}')
        print(f'Rater DataFrame head for model {m}:\n{rater_data.head()}')

        for error_type in ['nocot_unfaithfulshortcut_error', 'somecot_unfaithfulshortcut_error', 'fullcot_unfaithfulshortcut_error', 'nocot_restoration_error', 'somecot_restoration_error', 'fullcot_restoration_error']:
            if f'{error_type}_1' not in rater_data.columns or f'{error_type}_2' not in rater_data.columns:
                print(f'Columns for {error_type} not found in rater data for model {m}. Skipping Cohen\'s Kappa calculation for this error type.')
                continue
            print(f'\nCalculating Cohen\'s Kappa for {error_type} in model {m} using columns: {f"{error_type}_1"} and {f"{error_type}_2"}')
            kappa, lower_ci, upper_ci = calculate_cohen_kappa(rater_data, f'{error_type}_1', f'{error_type}_2')
            print(f'Inter-annotator agreement (Cohen\'s Kappa) for {error_type} in model {m}: {kappa:.3f} (95% CI: {lower_ci:.3f} - {upper_ci:.3f})')
            row = {'model': m, 'error_name': error_type, 'cohens_kappa': kappa, 'lower_95_ci': lower_ci, 'upper_95_ci': upper_ci}
            all_rater_data.append(row)

    all_rater_data = pd.DataFrame(all_rater_data)
    print('\n\nAll inter-annotator agreement results:')
    print(all_rater_data)
    all_rater_data.to_excel(RESULTS_FILEPATH + 'inter_annotator_agreement_results.xlsx', index=False)

    plot_inter_annotator_agreement_heatmap_panel(all_rater_data, RESULTS_FILEPATH + 'inter_annotator_agreement_heatmap_panel.png', show_plots=True)

    all_explanation_correctness_df.to_excel(RESULTS_FILEPATH + 'all_explanation_correctness_annotations.xlsx', index=False)
    all_unfaithful_restoration_errors_df.to_excel(RESULTS_FILEPATH + 'all_unfaithful_restoration_annotations.xlsx', index=False)

else:
    all_explanation_correctness_df = pd.read_excel(RESULTS_FILEPATH + 'all_explanation_correctness_annotations.xlsx')
    all_unfaithful_restoration_errors_df = pd.read_excel(RESULTS_FILEPATH + 'all_unfaithful_restoration_annotations.xlsx')

## Natural Analysis ##
## explanation correctness df details
print('\n-----------------------------------------------\n')
print('Explanation correctness df details:\n')
print(f'Columns in explanation correctness DataFrame: {all_explanation_correctness_df.columns.tolist()}')
print(all_explanation_correctness_df.head())

## unfaithful shortcut and restoration error df details
print('\n-----------------------------------------------\n')
print('Unfaithful shortcut and restoration error df details:\n')
print(f'Columns in unfaithful shortcut and restoration error DataFrame: {all_unfaithful_restoration_errors_df.columns.tolist()}')
print(all_unfaithful_restoration_errors_df.head())

## tokens in responses and descriptions details
all_explanation_correctness_df['n_tokens_nocot_response'] = all_explanation_correctness_df['nocot_response'].apply(lambda x: get_context_length(x))
all_explanation_correctness_df['n_tokens_somecot_response'] = all_explanation_correctness_df['somecot_response'].apply(lambda x: get_context_length(x))
all_explanation_correctness_df['n_tokens_fullcot_response'] = all_explanation_correctness_df['fullcot_response'].apply(lambda x: get_context_length(x))
all_explanation_correctness_df['n_tokens_description'] = all_explanation_correctness_df['description'].apply(lambda x: get_context_length(x))

all_explanation_correctness_df['n_tokens_nocot_response_group'] = pd.cut(all_explanation_correctness_df['n_tokens_nocot_response'], bins=[0, 500, 1000, 1500, np.inf], labels=['0-500', '501-1000', '1001-1500', '1501+'])
all_explanation_correctness_df['n_tokens_somecot_response_group'] = pd.cut(all_explanation_correctness_df['n_tokens_somecot_response'], bins=[0, 500, 1000, 1500, np.inf], labels=['0-500', '501-1000', '1001-1500', '1501+'])
all_explanation_correctness_df['n_tokens_fullcot_response_group'] = pd.cut(all_explanation_correctness_df['n_tokens_fullcot_response'], bins=[0, 500, 1000, 1500, np.inf], labels=['0-500', '501-1000', '1001-1500', '1501+'])
all_explanation_correctness_df['n_tokens_description_group'] = pd.cut(all_explanation_correctness_df['n_tokens_description'], bins=[0, 100, 200, 300, 400, np.inf], labels=['0-100', '101-200', '201-300', '301-400', '401+'])

all_unfaithful_restoration_errors_df['n_tokens_nocot_response'] = all_unfaithful_restoration_errors_df['nocot_response'].apply(lambda x: get_context_length(x))
all_unfaithful_restoration_errors_df['n_tokens_somecot_response'] = all_unfaithful_restoration_errors_df['somecot_response'].apply(lambda x: get_context_length(x))
all_unfaithful_restoration_errors_df['n_tokens_fullcot_response'] = all_unfaithful_restoration_errors_df['fullcot_response'].apply(lambda x: get_context_length(x))
all_unfaithful_restoration_errors_df['n_tokens_description'] = all_unfaithful_restoration_errors_df['description'].apply(lambda x: get_context_length(x))

all_unfaithful_restoration_errors_df['n_tokens_nocot_response_group'] = pd.cut(all_unfaithful_restoration_errors_df['n_tokens_nocot_response'], bins=[0, 500, 1000, 1500, np.inf], labels=['0-500', '501-1000', '1001-1500', '1501+'])
all_unfaithful_restoration_errors_df['n_tokens_somecot_response_group'] = pd.cut(all_unfaithful_restoration_errors_df['n_tokens_somecot_response'], bins=[0, 500, 1000, 1500, np.inf], labels=['0-500', '501-1000', '1001-1500', '1501+'])
all_unfaithful_restoration_errors_df['n_tokens_fullcot_response_group'] = pd.cut(all_unfaithful_restoration_errors_df['n_tokens_fullcot_response'], bins=[0, 500, 1000, 1500, np.inf], labels=['0-500', '501-1000', '1001-1500', '1501+'])
all_unfaithful_restoration_errors_df['n_tokens_description_group'] = pd.cut(all_unfaithful_restoration_errors_df['n_tokens_description'], bins=[0, 100, 200, 300, 400, np.inf], labels=['0-100', '101-200', '201-300', '301-400', '401+'])

print('\n-----------------------------------------------\n')
print('All explanation correctness DataFrame after adding token counts and groups:\n')
print(all_explanation_correctness_df.head())

print('\n-----------------------------------------------\n')
print('All unfaithful shortcut and restoration error DataFrame after adding token counts and groups:\n')
print(all_unfaithful_restoration_errors_df.head())

explanation_correctness_error_by_trial = all_explanation_correctness_df.groupby(['model', 'trial'])[['nocot_explanationcorrectness_error', 'somecot_explanationcorrectness_error', 'fullcot_explanationcorrectness_error']].sum().reset_index()
unfaithful_shortcut_error_by_trial = all_unfaithful_restoration_errors_df.groupby(['model', 'trial'])[['nocot_unfaithfulshortcut_error', 'somecot_unfaithfulshortcut_error', 'fullcot_unfaithfulshortcut_error']].sum().reset_index()
restoration_error_by_trial = all_unfaithful_restoration_errors_df.groupby(['model', 'trial'])[['nocot_restoration_error', 'somecot_restoration_error', 'fullcot_restoration_error']].sum().reset_index()

print('\n\nExplanation Correctness Errors by Trial DataFrame:')
print(explanation_correctness_error_by_trial)
print('\n\nUnfaithful Shortcut Errors by Trial DataFrame:')
print(unfaithful_shortcut_error_by_trial)
print('\n\nRestoration Errors by Trial DataFrame:')
print(restoration_error_by_trial)

natural_plot(explanation_correctness_error_by_trial, unfaithful_shortcut_error_by_trial, restoration_error_by_trial, RESULTS_FILEPATH + 'natural_analysis_panel', error_titles=['Explanation Correctness Errors', 'Unfaithful Shortcut Errors', 'Restoration Errors'], show_plots=True)

## Simulated Analysis ##
all_simulated_errors_df = pd.DataFrame()  # Initialize empty DataFrame to store all simulated errors across models

cols_to_keep = ['trial',
                'description',
                'outcome',
                'hint_gt_outcome',

                'nocot_parsed_response',
                'somecot_parsed_response',
                'fullcot_parsed_response',

                'nocot_noexamples_hint_parsed_response',
                'somecot_noexamples_hint_parsed_response',
                'fullcot_noexamples_hint_parsed_response',

                'nocot_randomfewshot_nohint_parsed_response',
                'somecot_randomfewshot_nohint_parsed_response',
                'fullcot_randomfewshot_nohint_parsed_response',
                
                'nocot_randomfewshot_hint_parsed_response',
                'somecot_randomfewshot_hint_parsed_response',
                'fullcot_randomfewshot_hint_parsed_response',
                
                'nocot_specificfewshot_nohint_parsed_response',
                'somecot_specificfewshot_nohint_parsed_response',
                'fullcot_specificfewshot_nohint_parsed_response',
                
                'nocot_specificfewshot_hint_parsed_response',
                'somecot_specificfewshot_hint_parsed_response',
                'fullcot_specificfewshot_hint_parsed_response']

for m in MODELS:
    simulated_filename = f'simulated_results_{m}.xlsx'
    simulated_filepath = RESULTS_FILEPATH + simulated_filename
    print(f'Checking for file: {simulated_filepath}')
    if os.path.exists(simulated_filepath):
        simulated_df = pd.read_excel(simulated_filepath)
        print(f'\nLoaded file: {simulated_filepath} with shape {simulated_df.shape}')
        print(f'Columns in the file: {simulated_df.columns.tolist()}')
        print(simulated_df.head())
        simulated_df = simulated_df.rename(columns={'no_cot_parsed_response': 'nocot_parsed_response',
                                                    'some_cot_parsed_response': 'somecot_parsed_response',
                                                    'full_cot_parsed_response': 'fullcot_parsed_response'})
        simulated_df['model'] = m
        all_simulated_errors_df = pd.concat([all_simulated_errors_df, simulated_df[cols_to_keep + ['model']]], ignore_index=True)
    else:
        print(f'Simulated iscorrect file not found for model {m}. Skipping simulated analysis for this model.')
        continue

print('\n\nAll simulated errors DataFrame after loading all models:')
print('Shape:', all_simulated_errors_df.shape)
print('Unique models in the DataFrame:', all_simulated_errors_df['model'].unique())
print(f'Columns in all_simulated_errors_df: {all_simulated_errors_df.columns.tolist()}')
print(all_simulated_errors_df.head())

all_iscorrect_df, iscorrect_colnames, outcome_colnames = get_simulated_iscorrect(all_simulated_errors_df)
print('\n\nAll iscorrect DataFrame after calculating iscorrect columns:')
print('Shape:', all_iscorrect_df.shape)
print(f'Columns in all_iscorrect_df: {all_iscorrect_df.columns.tolist()}')
print(all_iscorrect_df.head())

# Group by model and trial to get per-trial statistics
trial_stats = all_iscorrect_df.groupby(['model', 'trial'])[iscorrect_colnames].mean().reset_index()


# Average across trials to get overall mean (preserves plot output)
average_iscorrect_df = trial_stats.groupby('model')[iscorrect_colnames].mean().reset_index()
balanced_iscorrect_df = all_iscorrect_df.groupby('model').apply(lambda x: pd.Series({col: balanced_accuracy_score(x[outcome_colnames[0]], x[col]) for col in iscorrect_colnames})).reset_index()
# Calculate std of trial means (captures trial-to-trial variation with sample std dev)
stddev_iscorrect_df = trial_stats.groupby('model')[iscorrect_colnames].std(ddof=1).reset_index()

# Paired t-tests of model trial means against unbiased parsed-response trial means
alpha = 0.001
baseline_map = {}
for col in iscorrect_colnames:
    parts = col.replace('iscorrect_', '').split('_')
    cot_type = parts[0] if len(parts) >= 1 else None
    if col == f'iscorrect_{cot_type}_parsed_response':
        baseline_map[col] = None
    else:
        baseline_map[col] = f'iscorrect_{cot_type}_parsed_response' if cot_type else None


def fdr_bh(pvals):
    pvals = np.asarray(pvals, dtype=float)
    adjusted = np.full_like(pvals, np.nan)
    valid_mask = ~np.isnan(pvals)
    if valid_mask.sum() == 0:
        return adjusted
    valid_pvals = pvals[valid_mask]
    order = np.argsort(valid_pvals)
    sorted_pvals = valid_pvals[order]
    n = len(sorted_pvals)
    adjusted_sorted = np.empty(n, dtype=float)
    min_adj = 1.0
    for i in range(n - 1, -1, -1):
        adj = sorted_pvals[i] * n / (i + 1)
        min_adj = min(min_adj, adj)
        adjusted_sorted[i] = min_adj
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted[valid_mask] = adjusted_sorted[np.argsort(order)]
    return adjusted


def adjust_fdr_pvalues_df(df):
    pvals = df.drop(columns='model').to_numpy(dtype=float)
    flat = pvals.flatten()
    adjusted_flat = fdr_bh(flat)
    adjusted = adjusted_flat.reshape(pvals.shape)
    adjusted_df = pd.DataFrame(adjusted, columns=df.columns.drop('model'))
    adjusted_df.insert(0, 'model', df['model'].values)
    return adjusted_df

# Build t-test p-values relative to unbiased parsed-response baseline values

t_test_rows = []
for model in trial_stats['model'].unique():
    model_trials = trial_stats[trial_stats['model'] == model].copy()
    t_test_row = {'model': model}
    for col in iscorrect_colnames:
        baseline_col = baseline_map.get(col)
        if baseline_col is None or baseline_col not in model_trials.columns:
            t_test_row[col] = np.nan
            continue

        paired = model_trials[['trial', col, baseline_col]].dropna()
        if paired.shape[0] >= 2:
            _, p_val = ttest_rel(paired[col].astype(float), paired[baseline_col].astype(float), nan_policy='omit')
        else:
            p_val = np.nan
        t_test_row[col] = p_val
    t_test_rows.append(t_test_row)

test_pvalues_df = pd.DataFrame(t_test_rows)
fdr_pvalues_df = adjust_fdr_pvalues_df(test_pvalues_df)

print('\n\nAverage iscorrect values by model and description:')
print(average_iscorrect_df.head())
print('\n\nAverage balanced accuracy of iscorrect values by model and description:')
print(balanced_iscorrect_df.head())
print('\n\nStandard deviation of iscorrect values by model and description:')
print(stddev_iscorrect_df.head())
print('\n\nT-test p-values for model trial means against unbiased parsed-response baselines:')
print(test_pvalues_df.head())
print('\n\nFDR-adjusted p-values for the paired t-tests across all conditions:')
print(fdr_pvalues_df.head())

average_iscorrect_df.to_excel(RESULTS_FILEPATH + 'average_iscorrect_values.xlsx', index=False)
stddev_iscorrect_df.to_excel(RESULTS_FILEPATH + 'stddev_iscorrect_values.xlsx', index=False)

# Create structured long-form Excel file with model, cot_type, experiment, avg, std
structured_data = []

for model in average_iscorrect_df['model'].unique():
    avg_row = average_iscorrect_df[average_iscorrect_df['model'] == model].iloc[0]
    std_row = stddev_iscorrect_df[stddev_iscorrect_df['model'] == model].iloc[0]
    
    for col_name in iscorrect_colnames:
        # Parse column name: iscorrect_<cot>_<experiment>
        # Examples: iscorrect_nocot_randomfewshot_hint, iscorrect_nocot_parsed_response, iscorrect_nocot_noexamples_hint_parsed_response
        parts = col_name.replace('iscorrect_', '').split('_')
        
        cot_type = parts[0]  # nocot, somecot, or fullcot
        
        # Determine experiment type based on remaining parts
        remaining = '_'.join(parts[1:])
        if remaining == 'parsed_response':
            experiment = 'unbiased'
        elif remaining == 'noexamples_hint_parsed_response':
            experiment = 'noexamples_hint'
        else:
            experiment = remaining
        
        avg_val = round(float(avg_row[col_name]), 3)
        std_val = round(float(std_row[col_name]), 3)
        p_val = None
        adjusted_p_val = None
        if not test_pvalues_df.empty:
            p_val_values = test_pvalues_df.loc[test_pvalues_df['model'] == model, col_name].values
            p_val = float(p_val_values[0]) if len(p_val_values) > 0 and not pd.isna(p_val_values[0]) else None
        if not fdr_pvalues_df.empty:
            adj_p_val_values = fdr_pvalues_df.loc[fdr_pvalues_df['model'] == model, col_name].values
            adjusted_p_val = float(adj_p_val_values[0]) if len(adj_p_val_values) > 0 and not pd.isna(adj_p_val_values[0]) else None
        structured_data.append({
            'model': model,
            'cot_type': cot_type,
            'experiment': experiment,
            'avg': avg_val,
            'std': std_val,
            'p_value': round(p_val, 6) if p_val is not None else None,
            'fdr_p_value': round(adjusted_p_val, 6) if adjusted_p_val is not None else None
        })
structured_df = pd.DataFrame(structured_data)
structured_df.to_excel(RESULTS_FILEPATH + 'simulated_results_structured.xlsx', index=False)
print('\n\nStructured results saved to:', RESULTS_FILEPATH + 'simulated_results_structured.xlsx')
print(structured_df.head(20))

simulated_plot(average_iscorrect_df, RESULTS_FILEPATH + 'simulated_analysis_panel', show_plots=True)










## Supplemental Analysis ##
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

def get_converted_unfaithfulrestoration_response_tokens(df, model_list):
    return_df = pd.DataFrame(columns=['cot_type','total_responses', 'n_response_tokens','sum_restoration_errors'])

    bins = ['0-500', '501-1000', '1001-1500', '1501+']

    for model in model_list:
        temp_df = df[df['model'] == model].copy()
        print(f'\nProcessing model: {model} with {temp_df.shape[0]} responses for unfaithful shortcut and restoration error response token analysis.')
        for bin in bins:
            row1 = {'model': model, 'cot_type': 'nocot', 'total_responses': temp_df[temp_df['n_tokens_nocot_response_group'] == bin].shape[0], 'n_response_tokens': bin, 'sum_restoration_errors': temp_df[temp_df['n_tokens_nocot_response_group'] == bin]['nocot_restoration_error'].sum(), 'sum_unfaithfulshortcut_errors': temp_df[temp_df['n_tokens_nocot_response_group'] == bin]['nocot_unfaithfulshortcut_error'].sum()}
            row2 = {'model': model, 'cot_type': 'somecot', 'total_responses': temp_df[temp_df['n_tokens_somecot_response_group'] == bin].shape[0], 'n_response_tokens': bin, 'sum_restoration_errors': temp_df[temp_df['n_tokens_somecot_response_group'] == bin]['somecot_restoration_error'].sum(), 'sum_unfaithfulshortcut_errors': temp_df[temp_df['n_tokens_somecot_response_group'] == bin]['somecot_unfaithfulshortcut_error'].sum()}
            row3 = {'model': model, 'cot_type': 'fullcot', 'total_responses': temp_df[temp_df['n_tokens_fullcot_response_group'] == bin].shape[0], 'n_response_tokens': bin, 'sum_restoration_errors': temp_df[temp_df['n_tokens_fullcot_response_group'] == bin]['fullcot_restoration_error'].sum(), 'sum_unfaithfulshortcut_errors': temp_df[temp_df['n_tokens_fullcot_response_group'] == bin]['fullcot_unfaithfulshortcut_error'].sum()}
            return_df = pd.concat([return_df, pd.DataFrame([row1, row2, row3])], ignore_index=True)

    return_df['total_responses'] = return_df['total_responses'].astype(int)
    return_df['sum_restoration_errors'] = return_df['sum_restoration_errors'].astype(int)
    return_df['sum_unfaithfulshortcut_errors'] = return_df['sum_unfaithfulshortcut_errors'].astype(int)
    return_df['n_response_tokens'] = return_df['n_response_tokens'].astype(str)
    return_df['cot_type'] = return_df['cot_type'].astype(str)

    return return_df

def get_converted_explanationcorrectness_response_tokens(df, model_list):
    return_df = pd.DataFrame(columns=['cot_type','total_responses', 'n_response_tokens','sum_explanation_correctness_errors'])

    bins = ['0-500', '501-1000', '1001-1500', '1501+']

    for model in model_list:
        temp_df = df[df['model'] == model].copy()
        print(f'\nProcessing model: {model} with {temp_df.shape[0]} responses for explanation correctness response token analysis.')
        for bin in bins:
            row1 = {'model': model, 'cot_type': 'nocot', 'total_responses': temp_df[temp_df['n_tokens_nocot_response_group'] == bin].shape[0], 'n_response_tokens': bin, 'sum_explanation_correctness_errors': temp_df[temp_df['n_tokens_nocot_response_group'] == bin]['nocot_explanationcorrectness_error'].sum()}
            row2 = {'model': model, 'cot_type': 'somecot', 'total_responses': temp_df[temp_df['n_tokens_somecot_response_group'] == bin].shape[0], 'n_response_tokens': bin, 'sum_explanation_correctness_errors': temp_df[temp_df['n_tokens_somecot_response_group'] == bin]['somecot_explanationcorrectness_error'].sum()}
            row3 = {'model': model, 'cot_type': 'fullcot', 'total_responses': temp_df[temp_df['n_tokens_fullcot_response_group'] == bin].shape[0], 'n_response_tokens': bin, 'sum_explanation_correctness_errors': temp_df[temp_df['n_tokens_fullcot_response_group'] == bin]['fullcot_explanationcorrectness_error'].sum()}
            return_df = pd.concat([return_df, pd.DataFrame([row1, row2, row3])], ignore_index=True)

    return_df['total_responses'] = return_df['total_responses'].astype(int)
    return_df['sum_explanation_correctness_errors'] = return_df['sum_explanation_correctness_errors'].astype(int)
    return_df['n_response_tokens'] = return_df['n_response_tokens'].astype(str)
    return_df['cot_type'] = return_df['cot_type'].astype(str)

    return return_df

def get_converted_unfaithfulrestoration_description_tokens(df, model_list):
    return_df = pd.DataFrame(columns=['cot_type','n_description_tokens','total_descriptions', 'sum_restoration_errors', 'sum_unfaithfulshortcut_errors'])

    bins = ['0-100', '101-200', '201-300', '301-400', '401+']

    for model in model_list:
        temp_df = df[df['model'] == model].copy()
        print(f'\nProcessing model: {model} with {temp_df.shape[0]} descriptions for unfaithful/restoration error analysis.')
        for bin in bins:
            row = {'model': model, 'cot_type': 'nocot', 'n_description_tokens': bin, 'total_descriptions': temp_df[temp_df['n_tokens_description_group'] == bin].shape[0], 'sum_restoration_errors': temp_df[temp_df['n_tokens_description_group'] == bin]['nocot_restoration_error'].sum(), 'sum_unfaithfulshortcut_errors': temp_df[temp_df['n_tokens_description_group'] == bin]['nocot_unfaithfulshortcut_error'].sum()}
            return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

            row = {'model': model, 'cot_type': 'somecot', 'n_description_tokens': bin, 'total_descriptions': temp_df[temp_df['n_tokens_description_group'] == bin].shape[0], 'sum_restoration_errors': temp_df[temp_df['n_tokens_description_group'] == bin]['somecot_restoration_error'].sum(), 'sum_unfaithfulshortcut_errors': temp_df[temp_df['n_tokens_description_group'] == bin]['somecot_unfaithfulshortcut_error'].sum()}
            return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

            row = {'model': model, 'cot_type': 'fullcot', 'n_description_tokens': bin, 'total_descriptions': temp_df[temp_df['n_tokens_description_group'] == bin].shape[0], 'sum_restoration_errors': temp_df[temp_df['n_tokens_description_group'] == bin]['fullcot_restoration_error'].sum(), 'sum_unfaithfulshortcut_errors': temp_df[temp_df['n_tokens_description_group'] == bin]['fullcot_unfaithfulshortcut_error'].sum()}
            return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

    return_df['sum_restoration_errors'] = return_df['sum_restoration_errors'].astype(int)
    return_df['sum_unfaithfulshortcut_errors'] = return_df['sum_unfaithfulshortcut_errors'].astype(int)
    return return_df

def get_converted_explanationcorrectness_description_tokens(df, model_list):
    return_df = pd.DataFrame(columns=['cot_type','n_description_tokens','total_descriptions', 'sum_explanation_correctness_errors'])

    bins = ['0-100', '101-200', '201-300', '301-400', '401+']

    for model in model_list:
        temp_df = df[df['model'] == model].copy()
        print(f'\nProcessing model: {model} with {temp_df.shape[0]} descriptions for explanation correctness analysis.')
        for bin in bins:
            row = {'model': model, 'cot_type': 'nocot', 'n_description_tokens': bin, 'total_descriptions': temp_df[temp_df['n_tokens_description_group'] == bin].shape[0], 'sum_explanation_correctness_errors': temp_df[temp_df['n_tokens_description_group'] == bin]['nocot_explanationcorrectness_error'].sum()}
            return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

            row = {'model': model, 'cot_type': 'somecot', 'n_description_tokens': bin, 'total_descriptions': temp_df[temp_df['n_tokens_description_group'] == bin].shape[0], 'sum_explanation_correctness_errors': temp_df[temp_df['n_tokens_description_group'] == bin]['somecot_explanationcorrectness_error'].sum()}
            return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

            row = {'model': model, 'cot_type': 'fullcot', 'n_description_tokens': bin, 'total_descriptions': temp_df[temp_df['n_tokens_description_group'] == bin].shape[0], 'sum_explanation_correctness_errors': temp_df[temp_df['n_tokens_description_group'] == bin]['fullcot_explanationcorrectness_error'].sum()}
            return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

    return_df['sum_explanation_correctness_errors'] = return_df['sum_explanation_correctness_errors'].astype(int)
    return return_df

def get_converted_unfaithfulrestoration_outcome(df, model_list):
    return_df = pd.DataFrame(columns=['cot_type', 'cot_outcome', 'total_outcomes', 'sum_restoration_errors', 'sum_unfaithfulshortcut_errors'])

    for model in model_list:
        temp_df = df[df['model'] == model].copy()
        print(f'\nProcessing model: {model} with {temp_df.shape[0]} descriptions for unfaithful/restoration error analysis.')

        outcomes_nocot = temp_df['nocot_outcome'].unique()
        outcomes_somecot = temp_df['somecot_outcome'].unique()
        outcomes_fullcot = temp_df['fullcot_outcome'].unique()

        for outcome in outcomes_nocot:
            row = {'model': model, 'cot_type': 'nocot', 'cot_outcome': outcome, 'total_outcomes': temp_df[temp_df['outcome'] == outcome].shape[0],'sum_restoration_errors': temp_df[temp_df['outcome'] == outcome]['nocot_restoration_error'].sum(), 'sum_unfaithfulshortcut_errors': temp_df[temp_df['outcome'] == outcome]['nocot_unfaithfulshortcut_error'].sum()}
            return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

        for outcome in outcomes_somecot:
            row = {'model': model, 'cot_type': 'somecot', 'cot_outcome': outcome, 'total_outcomes': temp_df[temp_df['outcome'] == outcome].shape[0], 'sum_restoration_errors': temp_df[temp_df['outcome'] == outcome]['somecot_restoration_error'].sum(), 'sum_unfaithfulshortcut_errors': temp_df[temp_df['outcome'] == outcome]['somecot_unfaithfulshortcut_error'].sum()}
            return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

        for outcome in outcomes_fullcot:
            row = {'model': model, 'cot_type': 'fullcot', 'cot_outcome': outcome, 'total_outcomes': temp_df[temp_df['outcome'] == outcome].shape[0], 'sum_restoration_errors': temp_df[temp_df['outcome'] == outcome]['fullcot_restoration_error'].sum(), 'sum_unfaithfulshortcut_errors': temp_df[temp_df['outcome'] == outcome]['fullcot_unfaithfulshortcut_error'].sum()}
            return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

    return_df['sum_restoration_errors'] = return_df['sum_restoration_errors'].astype(int)
    return_df['sum_unfaithfulshortcut_errors'] = return_df['sum_unfaithfulshortcut_errors'].astype(int)

    return_df['cot_outcome'] = return_df['cot_outcome'].astype(int)
    return_df['cot_outcome'] = return_df['cot_outcome'].map(reverse_replace_dict)

    return return_df

def get_converted_explanationcorrectness_outcome(df, model_list):
    return_df = pd.DataFrame(columns=['cot_type', 'cot_outcome', 'total_outcomes', 'sum_explanation_correctness_errors'])

    print(df.columns)

    for model in model_list:
        temp_df = df[df['model'] == model].copy()
        print(f'\nProcessing model: {model} with {temp_df.shape[0]} descriptions for explanation correctness error analysis.')

        outcomes =  [-1, 0, 1, 2, 3, 4, 5, 6]

        for outcome in outcomes:
            row = {'model': model, 'cot_type': 'nocot', 'cot_outcome': outcome, 'total_outcomes': temp_df[temp_df['outcome'] == outcome].shape[0],'sum_explanation_correctness_errors': temp_df[temp_df['outcome'] == outcome]['nocot_explanationcorrectness_error'].sum()}
            return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)
        
        for outcome in outcomes:
            row = {'model': model, 'cot_type': 'somecot', 'cot_outcome': outcome, 'total_outcomes': temp_df[temp_df['outcome'] == outcome].shape[0], 'sum_explanation_correctness_errors': temp_df[temp_df['outcome'] == outcome]['somecot_explanationcorrectness_error'].sum()}
            return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

        for outcome in outcomes:
            row = {'model': model, 'cot_type': 'fullcot', 'cot_outcome': outcome, 'total_outcomes': temp_df[temp_df['outcome'] == outcome].shape[0], 'sum_explanation_correctness_errors': temp_df[temp_df['outcome'] == outcome]['fullcot_explanationcorrectness_error'].sum()}
            return_df = pd.concat([return_df, pd.DataFrame([row])], ignore_index=True)

    return_df['sum_explanation_correctness_errors'] = return_df['sum_explanation_correctness_errors'].astype(int)

    return_df['cot_outcome'] = return_df['cot_outcome'].astype(int)
    return_df['cot_outcome'] = return_df['cot_outcome'].map(reverse_replace_dict)

    return return_df

def natural_explanationcorrectness_response_bins_plot(df, filepath, show_plots=True):
    model_names_to_display = {
        'mistralsmall24binstruct2501q4KM': 'Mistral',
        'deepseekr132bqwendistillq4KM': 'DeepSeek',
        'phi414bq4KM': 'Phi'
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
        ax.set_ylim(0, 600)

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

def natural_response_bins_plot(df, filepath, show_plots=True):
    model_names_to_display = {
        'mistralsmall24binstruct2501q4KM': 'Mistral',
        'deepseekr132bqwendistillq4KM': 'DeepSeek',
        'phi414bq4KM': 'Phi'
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

        # Set y-axis to start at 0 and end at 800
        ax.set_ylim(0, 1200)

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
        'mistralsmall24binstruct2501q4KM': 'Mistral',
        'deepseekr132bqwendistillq4KM': 'DeepSeek',
        'phi414bq4KM': 'Phi'
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

        ax.set_ylim(0, max(total_heights) * 1.35 if len(total_heights) > 0 else 1)

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
        'mistralsmall24binstruct2501q4KM': 'Mistral',
        'deepseekr132bqwendistillq4KM': 'DeepSeek',
        'phi414bq4KM': 'Phi'
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

        ax.set_ylim(0, 350 + (max(total_heights) if len(total_heights) > 0 else 1))

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
        'mistralsmall24binstruct2501q4KM': 'Mistral',
        'deepseekr132bqwendistillq4KM': 'DeepSeek',
        'phi414bq4KM': 'Phi'
    }

    cot_types = ['nocot', 'somecot', 'fullcot']
    cot_type_labels = ['No CoT', 'Some CoT', 'Full CoT']
    models = df['model'].unique()
    outcome_labels = ['NONE', 'IMV ONLY', 'NIPPV ONLY', 'HFNI ONLY', 'NIPPV TO IMV', 'HFNI TO IMV', 'IMV TO NIPPV', 'IMV TO HFNI']
    bar_width = 0.45
    n_outcomes = len(outcome_labels)
    n_cot = len(cot_types)
    inner_group_width = bar_width + 0.18
    group_width = n_outcomes * inner_group_width + 0.6
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    for i, model in enumerate(models):
        ax = axes[i]
        df_model = df[df['model'] == model]

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

        ax.set_ylim(0, 90)
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
        'mistralsmall24binstruct2501q4KM': 'Mistral',
        'deepseekr132bqwendistillq4KM': 'DeepSeek',
        'phi414bq4KM': 'Phi'
    }

    cot_types = ['nocot', 'somecot', 'fullcot']
    cot_type_labels = ['No CoT', 'Some CoT', 'Full CoT']
    models = df['model'].unique()
    outcome_labels = ['NONE', 'IMV ONLY', 'NIPPV ONLY', 'HFNI ONLY', 'NIPPV TO IMV']
    bar_width = 0.45
    n_outcomes = len(outcome_labels)
    n_cot = len(cot_types)
    inner_group_width = bar_width + 0.18
    group_width = n_outcomes * inner_group_width + 0.6  # width of each outer group (CoT type)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    for i, model in enumerate(models):
        ax = axes[i]
        df_model = df[df['model'] == model]

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
        ax.set_ylim(0, 350 + (max(total_heights) if len(total_heights) > 0 else 1))

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
        'mistralsmall24binstruct2501q4KM': 'Mistral',
        'deepseekr132bqwendistillq4KM': 'DeepSeek',
        'phi414bq4KM': 'Phi'
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
                ax.legend(handles=handles, fontsize=11, loc='upper right', bbox_to_anchor=(1, 0.78))
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
        'mistralsmall24binstruct2501q4KM': 'Mistral',
        'deepseekr132bqwendistillq4KM': 'DeepSeek',
        'phi414bq4KM': 'Phi'
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
            print(f"Model: {m}, CoT: {cot}, Random Count: {random_counts[cot]}, Unbiased Count: {unbiased_counts[cot]}, Percent Difference: {pd_random:.2f}%"   )
            pd_specific = percent_difference(specific_counts[cot], unbiased_counts[cot])
            print(f"Model: {m}, CoT: {cot}, Specific Count: {specific_counts[cot]}, Unbiased Count: {unbiased_counts[cot]}, Percent Difference: {pd_specific:.2f}%")
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
        baseline_y = -125  # Fixed position at -75%
        for idx, cot in enumerate(cot_types):
            ax.text(
                x[idx], baseline_y,
                f"Baseline: {unbiased_counts[cot]}",
                ha='center', va='center', fontsize=14, color='#3a414a', fontweight='bold'
            )
        
        # Set y-limits: -100% at bottom, add padding above max valu
        ax.set_ylim(-150, 700)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.savefig(savepath.replace('.png', '.svg'), bbox_inches='tight')
    if show_plots:
        plt.show()


print('\n\nGrouped RESPONSE Tokens DataFrame by COT Type (Explanation Correctness):')
converted_explanationcorrectness_response_tokens_df = get_converted_explanationcorrectness_response_tokens(all_explanation_correctness_df, ['deepseekr132bqwendistillq4KM', 'mistralsmall24binstruct2501q4KM', 'phi414bq4KM'])
print('\n\nConverted Explanation Correctness RESPONSE Tokens DataFrame:')
print(converted_explanationcorrectness_response_tokens_df.shape)
print(converted_explanationcorrectness_response_tokens_df.columns.tolist())
print(converted_explanationcorrectness_response_tokens_df.head())

natural_explanationcorrectness_response_bins_plot(converted_explanationcorrectness_response_tokens_df, filepath=RESULTS_FILEPATH, show_plots=False)


print('\n\nGrouped RESPONSE Tokens DataFrame by COT Type (Restoration and Unfaithful Shortcut Errors):')
converted_response_tokens_df = get_converted_unfaithfulrestoration_response_tokens(all_unfaithful_restoration_errors_df, ['deepseekr132bqwendistillq4KM', 'mistralsmall24binstruct2501q4KM', 'phi414bq4KM'])
print('\n\nConverted RESPONSE Tokens DataFrame:')
print(converted_response_tokens_df.shape)
print(converted_response_tokens_df.columns.tolist())
print(converted_response_tokens_df.head())

natural_response_bins_plot(converted_response_tokens_df, filepath=RESULTS_FILEPATH, show_plots=False)


print('\n\nGrouped DESCRIPTION Tokens DataFrame by COT Type (Explanation Correctness):')
converted_explanationcorrectness_description_tokens_df = get_converted_explanationcorrectness_description_tokens(all_explanation_correctness_df, ['deepseekr132bqwendistillq4KM', 'mistralsmall24binstruct2501q4KM', 'phi414bq4KM'])
print('\n\nConverted Explanation Correctness DESCRIPTION Tokens DataFrame:')
print(converted_explanationcorrectness_description_tokens_df.shape)
print(converted_explanationcorrectness_description_tokens_df.columns.tolist())
print(converted_explanationcorrectness_description_tokens_df.head())

natural_explanationcorrectness_desc_bins_plot(converted_explanationcorrectness_description_tokens_df, filepath=RESULTS_FILEPATH, show_plots=False)


print('\n\nGrouped DESCRIPTION Tokens DataFrame by COT Type (Restoration and Unfaithful Shortcut Errors):')
converted_description_tokens_df = get_converted_unfaithfulrestoration_description_tokens(all_unfaithful_restoration_errors_df, ['deepseekr132bqwendistillq4KM', 'mistralsmall24binstruct2501q4KM', 'phi414bq4KM'])
print('\n\nConverted DESCRIPTION Tokens DataFrame:')
print(converted_description_tokens_df.shape)
print(converted_description_tokens_df.columns.tolist())
print(converted_description_tokens_df.head())

natural_desc_bins_plot(converted_description_tokens_df, filepath=RESULTS_FILEPATH, show_plots=False)


print('\n\nGrouped OUTCOMES DataFrame by COT Type (Explanation Correctness):')
converted_explanationcorrectness_outcome_df = get_converted_explanationcorrectness_outcome(all_explanation_correctness_df, ['deepseekr132bqwendistillq4KM', 'mistralsmall24binstruct2501q4KM', 'phi414bq4KM'])
print('\n\nConverted Explanation Correctness OUTCOMES DataFrame:')
print(converted_explanationcorrectness_outcome_df.shape)
print(converted_explanationcorrectness_outcome_df.columns.tolist())
print(converted_explanationcorrectness_outcome_df.head())

natural_explanationcorrectness_outcome_plot(converted_explanationcorrectness_outcome_df, filepath=RESULTS_FILEPATH, show_plots=False)


print('\n\nGrouped OUTCOMES DataFrame by COT Type (Restoration and Unfaithful Shortcut Errors):')
converted_outcome_df = get_converted_unfaithfulrestoration_outcome(all_unfaithful_restoration_errors_df, ['deepseekr132bqwendistillq4KM', 'mistralsmall24binstruct2501q4KM', 'phi414bq4KM'])
print('\n\nConverted OUTCOMES DataFrame:')
print(converted_outcome_df.shape)
print(converted_outcome_df.columns.tolist())
print(converted_outcome_df.head())

natural_outcome_plot(converted_outcome_df, filepath=RESULTS_FILEPATH, show_plots=False)


print('\n\nHint Only Analysis:')

simulated_df_dict = {}

for m in models:
    print(f'\nModel: {m}')
    simulated_filename = f'simulated_results_{m}.xlsx'
    if os.path.exists(RESULTS_FILEPATH + simulated_filename):
        simulated_df = pd.read_excel(RESULTS_FILEPATH + simulated_filename)
        print(f'Loaded simulated results for model {m} from {simulated_filename}.')
        print(f'Simulated DataFrame shape: {simulated_df.shape}')
        print(f'Simulated DataFrame columns: {simulated_df.columns.tolist()}')
        print(f'Simulated DataFrame head:\n{simulated_df.head()}')
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
    savepath=RESULTS_FILEPATH + "panel_discrete_distribution_overlay.png",
    show_plots=True
)


print('\n\nExamples-Only Analysis:')
plot_examples_only_percent_difference(
    simulated_df_dict,
    models,
    savepath=RESULTS_FILEPATH + "examples_only_percent_difference.png",
    show_plots=True
)