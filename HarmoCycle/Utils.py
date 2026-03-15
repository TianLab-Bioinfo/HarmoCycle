import numpy as np
import anndata
from scipy.fft import fft
import os
import random
import numpy as np
import torch

def calculate_gene_oscillation_properties_v2(adata: anndata.AnnData) -> anndata.AnnData:
    """
    Perform spectral decomposition via Fast Fourier Transform (FFT) to quantify 
    gene-specific oscillatory dynamics.

    This function identifies the dominant harmonic frequency for each gene's expression 
    profile along a provided trajectory (e.g., pseudotime or cell cycle progression). 
    It extracts the peak amplitude and calculates the phase in three formats: 
    radians, degrees, and normalized units. These metrics are instrumental for 
    characterizing cyclic biological processes such as the circadian clock or 
    the mitotic cell cycle.

    Parameters:
    -----------
    adata : anndata.AnnData
        The annotated data matrix of shape (n_cells, n_genes). 
        Note: Observations (.obs) must be pre-sorted according to a temporal 
        or pseudo-temporal axis.
    
    Returns:
    --------
    anndata.AnnData
        A modified copy of the input AnnData object. The following metrics are 
        appended to the .var (feature) metadata:
        - 'dominant_frequency_bin': The discrete frequency index (1-based) 
          corresponding to the maximum spectral power.
        - 'dominant_amplitude': The magnitude of the signal at the dominant 
          frequency, normalized by the number of observations (2/N scaling).
        - 'dominant_phase_rad': Phase angle at the dominant frequency in 
          radians (range: [-π, π]).
        - 'dominant_phase_deg': Phase angle in degrees (range: [0, 360]).
        - 'dominant_phase_norm': Phase angle normalized to the unit interval 
          (range: [0, 1]).
    """
    
    # Create a deep copy to ensure the original object remains immutable
    adata_processed = adata.copy()
    
    # Extract the expression matrix and ensure dense representation for FFT
    X = adata_processed.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    
    # Transpose matrix to (n_genes, n_cells) for vectorization along the cell axis
    X_T = X.T
    n_genes, n_cells = X_T.shape

    if n_cells <= 1:
        raise ValueError("FFT analysis requires more than one observation (cells) to compute frequency components.")
    
    # Apply Fast Fourier Transform across the cellular (temporal) dimension
    fft_result = fft(X_T, axis=1)
    
    # Focus on the positive frequency spectrum, excluding the DC component (index 0)
    # The relevant range extends from the first harmonic to the Nyquist frequency
    positive_freq_range = slice(1, n_cells // 2)
    amplitudes_one_sided = np.abs(fft_result[:, positive_freq_range])
    
    # Identify the local index of the frequency bin with the highest power
    dominant_freq_indices_local = np.argmax(amplitudes_one_sided, axis=1)
    
    # Reconstruct the global frequency index (accounting for the +1 slice offset)
    dominant_freq_indices_global = dominant_freq_indices_local + 1
    
    # Use advanced indexing to extract complex coefficients at dominant frequencies
    gene_dominant_coeffs = fft_result[np.arange(n_genes), dominant_freq_indices_global]
    
    # --- Quantification of Spectral Properties ---
    
    # 1. Amplitude Calculation
    # For non-DC components, the physical amplitude is defined as (2/N) * |coefficient|
    gene_amplitudes = np.abs(gene_dominant_coeffs) * 2 / n_cells
    
    # 2. Phase Calculation
    # Extract the raw phase angle in radians
    gene_phases_rad = np.angle(gene_dominant_coeffs)
    
    # Convert to degrees and map to the circular range [0, 360]
    gene_phases_deg = (np.degrees(gene_phases_rad) + 360) % 360
    
    # Normalize the phase to a [0, 1] interval for comparative analysis
    # Maps [-π, π] to [0, 1]
    gene_phases_norm = (gene_phases_rad + np.pi) / (2 * np.pi)
    
    # --- Metadata Update ---
    adata_processed.var['dominant_frequency_bin'] = dominant_freq_indices_global
    adata_processed.var['dominant_amplitude'] = gene_amplitudes
    adata_processed.var['dominant_phase_rad'] = gene_phases_rad
    adata_processed.var['dominant_phase_deg'] = gene_phases_deg
    adata_processed.var['dominant_phase_norm'] = gene_phases_norm
    
    return adata_processed

def set_seed(seed=3407):
    """Fix all random seeds to ensure reproducibility"""
    
    # 1. PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True  # Use deterministic convolution algorithms
        torch.backends.cudnn.benchmark = False     # Disable automatic optimization
    
    # 2. NumPy
    np.random.seed(seed)
    
    # 3. Python built-in random
    random.seed(seed)
    
    # 4. Environment variable (some libraries depend on this)
    os.environ['PYTHONHASHSEED'] = str(seed)

import pandas as pd
import gseapy as gp
import warnings
from typing import List, Dict

# gseapy may produce warnings about 'outdir', which we can safely ignore
warnings.filterwarnings("ignore", category=UserWarning)

def run_comprehensive_enrichment(gene_list: List[str], organism: str = 'Mouse') -> Dict[str, pd.DataFrame]:
    """
    Perform comprehensive GO (BP, CC, MF) and KEGG pathway enrichment analysis
    for a given gene list.

    Parameters:
    ----------
    gene_list : list
        Input list of gene names. For mouse, gene names are usually capitalized
        with the first letter uppercase and the rest lowercase (e.g., 'Cdk1').
        
    organism : str, optional (default: 'Mouse')
        Organism name. Must be supported by gseapy/enrichr.
        Common options: 'Mouse', 'Human', 'Rat', 'Yeast', 'Fly', 'Fish', 'Worm'.

    Returns:
    -------
    dict
        A dictionary where keys are database names and values are enrichment
        results (pandas DataFrame).
        Example: {'GO_Biological_Process': df1, 'KEGG': df2, ...}
    """
    print(f"Starting enrichment analysis for {len(gene_list)} genes, organism: {organism}")

    # --- 1. Define all gene set libraries to query ---
    # Three main branches of GO
    go_bp_lib = 'GO_Biological_Process_2021'
    go_cc_lib = 'GO_Cellular_Component_2021'
    go_mf_lib = 'GO_Molecular_Function_2021'
    
    # Dynamically search for the latest KEGG library to improve robustness
    available_libraries = gp.get_library_name(organism=organism)
    kegg_library = None
    # Prefer the most recent year
    for year in ['2021', '2019']:
        lib_name = f'KEGG_{year}_{organism}'
        if lib_name in available_libraries:
            kegg_library = lib_name
            break
    
    if kegg_library is None:
        # If no organism-specific KEGG library is found, fall back to human
        print(f"Warning: No KEGG library found for {organism}. Using 'KEGG_2021_Human' as fallback.")
        print("If your organism is not human, KEGG results may be inaccurate.")
        kegg_library = 'KEGG_2021_Human'
        
    gene_sets_to_query = [go_bp_lib, go_cc_lib, go_mf_lib, kegg_library]
    print(f"\nQuerying the following databases:\n{gene_sets_to_query}\n")

    # --- 2. Perform enrichment analysis in one call ---
    try:
        enrichment_results_df = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets_to_query,
            organism=organism,
            outdir=None,  # Set to None so results are returned as DataFrame
            cutoff=0.05   # Report only results with adjusted p-value < 0.05
        ).results
        print("Enrichment analysis completed.")
        
        # --- 3. Split combined results into separate DataFrames ---
        results_dict = {}
        # Group results by 'Gene_set' column
        for db_name, group_df in enrichment_results_df.groupby('Gene_set'):
            # Sort each group by adjusted p-value
            results_dict[db_name] = group_df.sort_values('Adjusted P-value', ascending=True)
            
        # Ensure dictionary contains empty DataFrames for missing libraries
        for lib in gene_sets_to_query:
            if lib not in results_dict:
                results_dict[lib] = pd.DataFrame()
                
        return results_dict

    except Exception as e:
        print(f"Enrichment analysis failed: {e}")
        # If request fails, return empty DataFrames for all libraries
        return {lib: pd.DataFrame() for lib in gene_sets_to_query}

def display_enrichment_results(results_dict: Dict[str, pd.DataFrame], top_n: int = 10):
    """Helper function to neatly print enrichment results."""
    print("\n\n" + "="*60)
    print("                Enrichment Analysis Results")
    print("="*60)
    
    # Define display order and titles
    display_order = {
        'GO_Biological_Process_2021': 'GO Biological Process',
        'GO_Molecular_Function_2021': 'GO Molecular Function',
        'GO_Cellular_Component_2021': 'GO Cellular Component',
    }
    # Add KEGG library dynamically
    kegg_key = next((key for key in results_dict if 'KEGG' in key), None)
    if kegg_key:
        display_order[kegg_key] = 'KEGG Pathways'

    for db_name, title in display_order.items():
        print(f"\n--- {title} (Top {top_n}) ---")
        
        result_df = results_dict.get(db_name)
        
        if result_df is not None and not result_df.empty:
            # Define key columns to display
            display_cols = ['Term', 'Adjusted P-value', 'Odds Ratio', 'Genes']
            # Ensure all columns exist
            display_cols = [col for col in display_cols if col in result_df.columns]
            print(result_df[display_cols].head(top_n).to_string())
        else:
            print(f"No significant enrichment results found for '{db_name}'.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from matplotlib.colors import LinearSegmentedColormap

def plot_go_bp_enrichment(enrichment_results, 
                         organism='Human',
                         top_n=10, 
                         adj_p_cutoff=0.1,
                         color_start="#E3A897", 
                         color_end="#C85B43",
                         title='Top Enriched GO Biological Processes',
                         figsize=(8, 6),
                         save_path=None):
    """
    Plot GO Biological Process enrichment results as a horizontal bar chart.
    Optimized for Adobe Illustrator compatibility (editable fonts and vector output).
    """
    
    # --- Adobe Illustrator compatibility settings ---
    # 1. Set font family to Arial/Helvetica to avoid missing glyphs
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    # 2. Use TrueType fonts (42) so text remains editable in Illustrator
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    # 3. Render math text (e.g., -log10) in regular font instead of italic math font
    plt.rcParams['mathtext.default'] = 'regular'
    # ------------------------------------------------
    
    # Extract GO Biological Process results
    if isinstance(enrichment_results, dict):
        go_bp_data = enrichment_results.get('GO_Biological_Process_2021', pd.DataFrame())
    else:
        go_bp_data = enrichment_results
    
    # Check if data is empty
    if go_bp_data.empty:
        print("Warning: No GO Biological Process data found!")
        return None, None
    
    # Filter significant results by adjusted p-value cutoff
    significant_results = go_bp_data[go_bp_data['Adjusted P-value'] < adj_p_cutoff].copy()
    
    if significant_results.empty:
        print(f"Warning: No significant results found with adjusted p-value < {adj_p_cutoff}")
        return None, None
    
    # Compute -log10(Adjusted P-value) for plotting
    significant_results['-log10(AdjP)'] = -np.log10(significant_results['Adjusted P-value'])
    significant_results = significant_results.sort_values(by='-log10(AdjP)', ascending=True)
    plot_data = significant_results.tail(top_n)
    
    # Clean up GO term names (remove GO IDs and wrap text)
    plot_data['Term'] = plot_data['Term'].str.replace(r'\s\(GO:\d+\)$', '', regex=True)
    plot_data['Term'] = plot_data['Term'].apply(
        lambda x: textwrap.fill(x.capitalize(), width=50, subsequent_indent='  ')
    )
    
    print("--- Data Prepared for Plotting ---")
    print(plot_data[['Term', '-log10(AdjP)', 'Overlap']])
    print("\n")
    
    # Create gradient color map for bars
    cmap = LinearSegmentedColormap.from_list("custom_gradient", [color_start, color_end])
    n_bars = len(plot_data)
    bar_colors = cmap(np.linspace(0, 1, n_bars))
    
    # Define style colors
    text_color = "#333333"
    grid_color = "#D3D3D3"
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bars
    bars = ax.barh(
        plot_data['Term'],
        plot_data['-log10(AdjP)'],
        color=bar_colors,
        height=0.7
    )
    
    # Add overlap labels to each bar
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label = plot_data['Overlap'].iloc[i]
        ax.text(
            width + 0.1,
            bar.get_y() + bar.get_height() / 2,
            label,
            ha='left',
            va='center',
            fontsize=10,
            color=text_color
        )
    
    # Title and axis labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color=text_color)
    ax.set_xlabel('-log$_{10}$(Adjusted P-value)', fontsize=12, fontweight='medium', color=text_color)
    ax.set_ylabel('')
    
    # Remove unnecessary spines and style bottom axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(grid_color)
    
    # Tick and grid styling
    ax.tick_params(axis='y', length=0, pad=10)
    plt.xticks(fontsize=10, color=text_color)
    plt.yticks(fontsize=11, color=text_color)
    ax.grid(axis='x', linestyle='--', color=grid_color, alpha=0.7)
    ax.set_axisbelow(True)
    
    fig.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        # PDF/SVG will produce vector graphics with editable text
        # PNG will produce raster image

        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the figure
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {save_path}")
   
        # Reminder for Illustrator editing
        if not save_path.lower().endswith(('.pdf', '.svg')):
             print("Tip: For Adobe Illustrator editing, save as .pdf or .svg format.")
    
    return fig, ax

import numpy as np
from scipy.stats import pearsonr
import pandas as pd

def calculate_correlation_score(ground_truth_stages: np.ndarray) -> float:
    """
    Calculate the Correlation-Score based on the idea from the reCAT paper.

    This function considers the circular nature of the cell cycle. It tests all
    possible sequence starting points (cut points) and both forward and reverse
    directions, computes the Pearson correlation coefficient (PCC) with the
    ideal linear order, and returns the maximum PCC value.

    Parameters
    ----------
    ground_truth_stages : np.ndarray
        A one-dimensional NumPy array containing the true stage labels of cells
        that have already been ordered according to predicted pseudotime
        (e.g., [0, 0, 1, 1, 2, 2, 0, ...]).

    Returns
    -------
    float
        The highest correlation score computed.
    """
    n_cells = len(ground_truth_stages)
    if n_cells < 2:
        return np.nan

    # Ideal linear order from 0 to n-1
    ideal_order = np.arange(n_cells)
    
    all_pccs = []

    # Iterate over all possible cut points
    for i in range(n_cells):
        # Roll the array to simulate cutting the cycle at different points
        # np.roll(arr, -i) shifts the array left by i positions
        permuted_stages = np.roll(ground_truth_stages, -i)
        
        # 1. Test forward sequence
        pcc_forward, _ = pearsonr(ideal_order, permuted_stages)
        all_pccs.append(pcc_forward)
        
        # 2. Test reverse sequence
        pcc_reverse, _ = pearsonr(ideal_order, permuted_stages[::-1])
        all_pccs.append(pcc_reverse)
        
    # Return the maximum PCC among all calculations
    return max(all_pccs)

def calculate_change_index(ground_truth_stages: np.ndarray) -> float:
    """
    Calculate the Change-Index based on the idea from the reCAT paper.

    This metric measures how smoothly the true stage labels change along the
    predicted cell order. The formula is:
        1 - (sc - 2) / (N - 3)
    where sc is the number of label changes, and N is the total number of cells.

    Parameters
    ----------
    ground_truth_stages : np.ndarray
        A one-dimensional NumPy array containing the true stage labels of cells
        ordered by predicted pseudotime.

    Returns
    -------
    float
        The calculated Change-Index. Returns NaN if the number of cells is <= 3.
    """
    n_cells = len(ground_truth_stages)
    
    # Metric is undefined when N <= 3
    if n_cells <= 3:
        return np.nan
        
    # Count the number of adjacent differences (sc)
    sc = np.sum(np.diff(ground_truth_stages) != 0)
    
    # Apply the formula from the paper
    change_index = 1 - (sc - 2) / (n_cells - 3)
    
    return change_index


import scanpy as sc
import anndata
import pandas as pd
from typing import Dict, List, Any

# Assume these are your custom functions; make sure they are defined and imported
# from your_module import calculate_gene_oscillation_properties, reconstruct_cosine_expression

def create_reference_gene_sets(
    reference_datasets: Dict[str, anndata.AnnData],
    min_frq: float = 1.0,
    max_frq: float = 2.0,
    min_amp: float = 0.01,
    top_n: int = 1000,
    stage_key: str = 'stage',
    stage_mapping: Dict[str, str] = None
) -> Dict[str, Dict[str, List[str]]]:
    """
    Identify and extract marker genes for each cell cycle stage from a series of
    reference datasets.

    This function iterates through each provided AnnData object and performs:
    1. Sort, normalize, and log-transform cells by cell cycle stage.
    2. Compute gene oscillation properties and reconstruct cosine expression patterns.
    3. Use Wilcoxon rank-sum test to identify differentially expressed genes per stage.
    4. Select the top_n genes with the highest scores for each stage.

    Args:
        reference_datasets (Dict[str, anndata.AnnData]):
            A dictionary where keys are dataset names (str) and values are AnnData objects.
            Each AnnData.obs must contain a column representing the cell cycle stage.
        min_frq (float, optional):
            Minimum frequency for cosine reconstruction. Default is 1.0.
        max_frq (float, optional):
            Maximum frequency for cosine reconstruction. Default is 2.0.
        min_amp (float, optional):
            Minimum amplitude for cosine reconstruction. Default is 0.01.
        top_n (int, optional):
            Number of top-ranked genes to select per stage. Default is 1000.
        stage_key (str, optional):
            Column name in AnnData.obs representing cell cycle stage. Default is 'stage'.
        stage_mapping (Dict[str, str], optional):
            Dictionary mapping stage labels in AnnData to standard names.
            Example: {'0': 'G1', '1': 'S', '2': 'G2M'}.
            If None, a default mapping is used.

    Returns:
        Dict[str, Dict[str, List[str]]]:
            A nested dictionary. Outer keys are dataset names, inner dictionaries
            contain lists of marker genes per cell cycle stage.
            Example:
            {
                'dataset1': {'G1': ['geneA', 'geneB'], 'S': [...], 'G2M': [...]},
                'dataset2': {'G1': [...], 'S': [...], 'G2M': [...]}
            }
    """
    # Use default stage mapping if none provided
    if stage_mapping is None:
        stage_mapping = {'0': 'G1', '1': 'S', '2': 'G2M'}

    all_gene_sets = {}

    for name, adata_orig in reference_datasets.items():
        print(f"--- Processing dataset: {name} ---")

        # Check if required column exists
        if stage_key not in adata_orig.obs.columns:
            print(f"Warning: Skipping dataset '{name}' because '.obs' has no column '{stage_key}'.")
            continue

        # 1. Preprocessing
        # Work on a copy to avoid modifying the original data
        adata = adata_orig.copy()
        
        # Sort by stage
        sorted_indices = adata.obs.sort_values(by=stage_key).index
        adata = adata[sorted_indices, :]

        # 2. Core computation
        try:
            adata_adjust = calculate_gene_oscillation_properties_v2(adata)
            cos_adata = reconstruct_cosine_expression(
                adata_adjust,
                min_frequency=min_frq,
                max_frequency=max_frq,
                min_amplitude=min_amp
            )
        except NameError as e:
            print(f"Error: A required function is not defined: {e}")
            print("Please ensure 'calculate_gene_oscillation_properties' and 'reconstruct_cosine_expression' are imported.")
            continue  # Skip this dataset

        # Ensure stage column is categorical for differential analysis
        cos_adata.obs[stage_key] = cos_adata.obs[stage_key].astype(str).astype('category')

        # 3. Differential gene analysis
        sc.tl.rank_genes_groups(
            cos_adata,
            groupby=stage_key,
            method='wilcoxon',
            key_added='rank_genes_stage'
        )

        marker_df = sc.get.rank_genes_groups_df(
            cos_adata,
            key='rank_genes_stage',
            group=None  # Get results for all groups
        )

        # 4. Filter and extract top N genes
        filtered_df = marker_df[(marker_df['scores'] > 0) & (marker_df['pvals_adj'] < 0.05)]
        
        top_genes_df = (
            filtered_df
            .sort_values(['group', 'scores'], ascending=[True, False])
            .groupby('group')
            .head(top_n)
        )

        # 5. Organize and store results
        dataset_gene_set = {}
        for group_id, phase_name in stage_mapping.items():
            genes = top_genes_df[top_genes_df['group'] == group_id]['names'].tolist()
            dataset_gene_set[phase_name] = genes
            print(f"Found {len(genes)} marker genes for phase '{phase_name}'.")

        all_gene_sets[name] = dataset_gene_set

    print("\n--- Processing complete. ---")
    return all_gene_sets


import numpy as np
import scanpy as sc

def reconstruct_cosine_expression(adata, min_frequency=1, max_frequency=None, min_amplitude=0):
    """
    Reconstruct gene expression patterns using only the dominant cosine components.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix containing oscillation properties in .var:
        - dominant_frequency_bin
        - dominant_amplitude
        - dominant_phase_rad
    min_frequency : int, optional (default: 1)
        Minimum frequency bin to include in reconstruction
    max_frequency : int, optional (default: None)
        Maximum frequency bin to include in reconstruction (None for no limit)
    min_amplitude : float, optional (default: 0)
        Minimum amplitude threshold for genes to include
        
    Returns:
    --------
    AnnData
        New AnnData object containing reconstructed cosine expression patterns
    """
    # Check required fields exist
    required_vars = ['dominant_frequency_bin', 'dominant_amplitude', 'dominant_phase_rad']
    for var in required_vars:
        if var not in adata.var.columns:
            raise ValueError(f"Missing required column in adata.var: {var}")
    
    n_cells = adata.shape[0]
    n_genes = adata.shape[1]
    t = np.arange(n_cells)
    
    # Get oscillation properties
    dominant_freqs = adata.var['dominant_frequency_bin'].values
    gene_amplitudes = adata.var['dominant_amplitude'].values
    gene_phases = adata.var['dominant_phase_rad'].values
    
    # Initialize reconstructed matrix
    X_cos = np.zeros((n_genes, n_cells))
    
    # Reconstruct each gene's pattern
    for i in range(n_genes):
        f = dominant_freqs[i]
        A = gene_amplitudes[i]
        phi = gene_phases[i]
        X_cos[i, :] = A * np.cos(2 * np.pi * f * t / n_cells + phi)
    
    # Create new AnnData object
    adata_purecos = sc.AnnData(
        X=X_cos.T,  # Transpose to cells × genes
        obs=adata.obs.copy(),
        var=adata.var.copy()
    )
    
    # Filter genes based on frequency and amplitude criteria
    freq_filter = (adata_purecos.var['dominant_frequency_bin'] >= min_frequency)
    if max_frequency is not None:
        freq_filter = freq_filter & (adata_purecos.var['dominant_frequency_bin'] <= max_frequency)
    amp_filter = (adata_purecos.var['dominant_amplitude'] > min_amplitude)
    
    adata_purecos = adata_purecos[:, freq_filter & amp_filter]
    
    return adata_purecos

import scanpy as sc
import pandas as pd
from anndata import AnnData
import scanpy as sc
import pandas as pd
from anndata import AnnData
from typing import List

def score_multiple_g1_cell_cycle(
    adata: AnnData,
    s_genes_list: List[List[str]],
    g2m_genes_list: List[List[str]],
    g1_genes_list: List[List[str]],
    prefix: str = 'ref'
) -> None:
    """
    Compute cell cycle scores for multiple sets of G1/S/G2M gene lists in an AnnData object.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing single-cell expression data.
    s_genes_list : List[List[str]]
        One or more lists of S-phase genes.
    g2m_genes_list : List[List[str]]
        One or more lists of G2M-phase genes (corresponding to s_genes_list).
    g1_genes_list : List[List[str]]
        Multiple lists of G1-phase genes.
    prefix : str, optional (default: 'ref')
        Prefix for naming the stored score fields.

    Notes
    -----
    All computed scores are added to `adata.obs` with the following naming convention:
    - G1_score_{prefix}{i}
    - S_score_{prefix}{i}
    - G2M_score_{prefix}{i}
    """
    # If only one set of S genes is provided, replicate it to match the number of G1 lists
    if isinstance(s_genes_list[0], str):
        s_genes_list = [s_genes_list] * len(g1_genes_list)
    # If only one set of G2M genes is provided, replicate it to match the number of G1 lists
    if isinstance(g2m_genes_list[0], str):
        g2m_genes_list = [g2m_genes_list] * len(g1_genes_list)

    # Iterate through each set of gene lists
    for i, (s_genes, g2m_genes, g1_genes) in enumerate(zip(s_genes_list, g2m_genes_list, g1_genes_list), 1):
        # Step 1: Score using the provided S/G2M gene sets
        sc.tl.score_genes_cell_cycle(
            adata,
            s_genes=s_genes,
            g2m_genes=g2m_genes
        )
        adata.obs['temp_S_score'] = adata.obs['S_score'].copy()

        # Step 2: Replace S genes with G1 genes and rescore
        sc.tl.score_genes_cell_cycle(
            adata,
            s_genes=g1_genes,
            g2m_genes=g2m_genes
        )

        # Step 3: Save scores for G1, S, and G2M phases
        adata.obs[f'G1_score_{prefix}{i}'] = adata.obs['S_score'].copy()
        adata.obs[f'S_score_{prefix}{i}'] = adata.obs['temp_S_score'].copy()
        adata.obs[f'G2M_score_{prefix}{i}'] = adata.obs['G2M_score'].copy()

        print(f"✅ Completed scoring for set {i} (G1/S/G2M).")

    # Clean up temporary fields
    adata.obs.drop(columns=['S_score', 'G2M_score', 'temp_S_score'], inplace=True, errors='ignore')


    # ===== PyTorch =====
import torch           # Core PyTorch library
import torch.nn as nn  # Neural network layers
from torch.utils.data import Dataset
from torch.autograd import Function  # Custom autograd functions

# Gradient Reversal Layer
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super(GRL, self).__init__()
        self.alpha = alpha
        
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
    
    def set_alpha(self, alpha):
        self.alpha = alpha

class DANN(nn.Module):
    def __init__(self, input_dim, feature_dim=128, num_classes=3):
        super(DANN, self).__init__()
        
        # -------------------------------
        # Feature extractor
        # -------------------------------
        # Maps raw input data into a compact feature representation.
        # Two linear layers with BatchNorm, ReLU activation, and dropout for regularization.
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),       # First hidden layer
            nn.BatchNorm1d(256),             # Normalize activations
            nn.ReLU(True),                   # Non-linear activation
            nn.Dropout(0.3),                 # Dropout to reduce overfitting
            nn.Linear(256, feature_dim),     # Map to feature dimension
            nn.BatchNorm1d(feature_dim),     # Normalize again
            nn.ReLU(True),                   # Activation
        )

        # -------------------------------
        # Label predictor
        # -------------------------------
        # Predicts class labels from extracted features.
        self.label_predictor = nn.Sequential(
            nn.Dropout(0.3),                 # Dropout for robustness
            nn.Linear(feature_dim, num_classes)  # Output logits for classification
        )

        # -------------------------------
        # Domain discriminator
        # -------------------------------
        # Predicts whether features come from source or target domain.
        # Binary classification (source=0, target=1).
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, 64),      # Hidden layer
            nn.BatchNorm1d(64),              # Normalize
            nn.ReLU(True),                   # Activation
            nn.Linear(64, 1)                 # Output single logit for BCEWithLogitsLoss
        )

        # -------------------------------
        # Gradient Reversal Layer (GRL)
        # -------------------------------
        # Reverses gradients during backpropagation to encourage domain-invariant features.
        # Alpha controls the strength of reversal and can be adapted during training.
        self.grl = GRL(alpha=1.0)

    def forward(self, x, mode='source_or_target_features'):
        # Extract features from input
        features = self.feature_extractor(x)

        if mode == 'source_or_target_features':
            # Predict labels
            label_preds = self.label_predictor(features)
            
            # Apply GRL before domain discriminator
            reversed_features = self.grl(features)
            
            # Predict domain (source vs target)
            domain_preds_logits = self.domain_discriminator(reversed_features)
            
            return label_preds, domain_preds_logits

        elif mode == 'target_predict':
            # Only predict labels (used for inference on target domain)
            label_preds = self.label_predictor(features)
            return label_preds

        else:
            # Return raw features only
            return features


class AnnDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = torch.FloatTensor(data)
        if labels is not None:
            self.labels = torch.LongTensor(labels)
        else:
            self.labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx], torch.LongTensor([-1]) # Dummy label for target
            
    
def assign_anchor_cells(adata, cluster_col='GMM_clusters', 
                       score_cols=['G1_corrected', 'S_corrected', 'G2M_score_final'],
                       specificity_threshold=0.7,
                       max_clusters_per_phase=2,
                       if_plot=True):
    """
    Assign anchor cells based on the specificity of cell cycle scores.
    Limits the number of selected clusters per phase to at most 2.

    Parameters:
        adata: AnnData object containing cell observations.
        cluster_col: Column name in adata.obs that stores cluster assignments.
        score_cols: List of columns representing corrected cell cycle scores (G1, S, G2M).
        specificity_threshold: Threshold (0–1) for specificity score. Higher values are stricter.
        max_clusters_per_phase: Maximum number of clusters to select per phase.
        if_plot: Whether to visualize specificity scores as a heatmap.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.special import softmax
    
    # 1. Compute median scores for each cluster
    cluster_medians = adata.obs.groupby(cluster_col)[score_cols].median()
    
    # 2. Compute specificity scores using softmax normalization
    def calc_specificity_scores(df):
        # Convert raw scores into probability distributions using softmax
        prob_matrix = df.apply(lambda x: softmax(x.values), axis=1, result_type='expand')
        prob_matrix.columns = score_cols
        return prob_matrix
    
    specificity_scores = calc_specificity_scores(cluster_medians)
    
    # 3. Identify anchor clusters for each phase
    phase_anchor_clusters = {}
    for phase in score_cols:
        # Sort clusters by specificity score (descending)
        phase_specificity = specificity_scores[phase].sort_values(ascending=False)
        
        # Select clusters that meet threshold, limited to max_clusters_per_phase
        candidate_clusters = phase_specificity[
            phase_specificity >= specificity_threshold
        ].index.tolist()[:max_clusters_per_phase]
        
        # If no cluster meets the threshold, select the top-scoring cluster
        if not candidate_clusters:
            print(f"Warning: No clusters meet specificity threshold {specificity_threshold} for {phase}. "
                  f"Selecting the highest scoring cluster instead.")
            candidate_clusters = [phase_specificity.index[0]]
        
        phase_anchor_clusters[phase] = candidate_clusters
    
    # 4. Mark anchor cells in the original data
    adata.obs['phase_anchor'] = 'None'
    for phase, clusters in phase_anchor_clusters.items():
        phase_name = phase.split('_')[0]  # Extract G1/S/G2M from column name
        cluster_mask = adata.obs[cluster_col].isin(clusters)
        
        # Select top 50% of cells within the cluster as anchors
        phase_score = adata.obs.loc[cluster_mask, phase]
        if len(phase_score) > 0:  # Ensure there are cells to select
            top_cells = phase_score.nlargest(int(len(phase_score) * 1)).index
            adata.obs.loc[top_cells, 'phase_anchor'] = phase_name
    
    if if_plot:
        # 5. Visualize specificity scores as a heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            specificity_scores.T,
            annot=True,
            fmt=".2f",
            cmap='YlOrRd',
            vmin=0,
            vmax=1,
            linewidths=.5
        )
        plt.title(f"Cell Cycle Phase Specificity Scores (Threshold={specificity_threshold})\n"
                  f"Max {max_clusters_per_phase} clusters per phase")
        plt.xlabel("Cluster")
        plt.ylabel("Phase")
        plt.show()
    
    # 6. Print assignment results
    print("\nAnchor cluster assignment results:")
    for phase, clusters in phase_anchor_clusters.items():
        specificity_values = specificity_scores.loc[clusters, phase].values
        print(f"{phase}: {list(zip(clusters, specificity_values))}")
    
    print("\nAnchor cell statistics:")
    print(adata.obs['phase_anchor'].value_counts())
    
    return adata


import numpy as np
import scipy
from scipy.fft import fft, ifft
from anndata import AnnData
from tqdm.auto import tqdm
import torch
import random
import os

def remove_global_top_frequencies(adata: AnnData, n_components: int = 1) -> AnnData:
    removal_adata = adata.copy()
    X = removal_adata.X.toarray() if scipy.sparse.issparse(removal_adata.X) else removal_adata.X.copy()
    n_cells = X.shape[0]
    
    # 1. 计算所有基因的平均频谱幅度
    avg_magnitudes = np.zeros(n_cells // 2 + 1)
    for gene_idx in range(X.shape[1]):
        fft_values = fft(X[:, gene_idx])
        avg_magnitudes += np.abs(fft_values[:n_cells // 2 + 1])
    avg_magnitudes /= X.shape[1]
    
    # 2. 找到整体最强的n个频率（跳过直流分量）
    top_global_indices = np.argsort(avg_magnitudes[1:])[-n_components:] + 1
    
    # 3. 对所有基因统一移除这些频率
    filtered_X = np.zeros_like(X)
    for gene_idx in tqdm(range(X.shape[1]), desc="Global frequency removal"):
        fft_values = fft(X[:, gene_idx])
        fft_values[top_global_indices] = 0
        fft_values[n_cells - top_global_indices] = 0  # 对称分量
        filtered_X[:, gene_idx] = ifft(fft_values).real
    
    removal_adata.X = filtered_X
    return removal_adata

def get_adjust_gene(refer_adata, adata_source, n_gene=200):        
    adata3 = adata_source.copy()
    
    adata3.var.index = adata3.var.index.str.upper()
    
    adata3 = adata3[:, adata3.var_names.intersection(refer_adata.var_names)]
    refer_adata = refer_adata[:, adata3.var_names]
    
    refer_adata = calculate_gene_oscillation_properties_v2(refer_adata)
    select_genes = refer_adata.var.sort_values(by='dominant_amplitude', ascending=False).head(n_gene).index
    
    return select_genes

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, zscore
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity

def calculate_cell_density(data, bandwidth=2):
    """
    Calculate the density value of each cell in PCA space.

    Parameters:
    data: 2D array, shape (n_cells, 2), PCA coordinates
    bandwidth: bandwidth parameter for KDE

    Returns:
    cell_density: array of density values for each cell, shape (n_cells,)
    kde_model: fitted KDE model
    """
    # Use sklearn's KernelDensity
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(data)
    
    # Compute log density for each data point, then convert to density values
    log_density = kde.score_samples(data)
    cell_density = np.exp(log_density)
    
    return cell_density, kde

def find_density_peaks_kde(data, cell_density, min_height=0.1, min_distance=5, n_points=100):
    """
    Find peak points in the density distribution.

    Parameters:
    data: 2D array of PCA coordinates
    cell_density: density values for each cell
    min_height: minimum peak height threshold
    min_distance: minimum distance between peaks
    n_points: resolution of the evaluation grid

    Returns:
    peak_coords: list of peak coordinates (x, y, density)
    density_2d: 2D density matrix
    xx, yy: meshgrid coordinates
    """
    # Create evaluation grid
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range
    
    xx, yy = np.mgrid[x_min:x_max:n_points*1j, y_min:y_max:n_points*1j]
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    
    # Use scipy's gaussian_kde to compute grid density (better for peak detection)
    kde_grid = gaussian_kde(data.T)
    grid_density = kde_grid(grid_points.T)
    density_2d = grid_density.reshape(n_points, n_points)
    
    # Find local maxima
    peaks = []
    peak_coords = []
    
    for i in range(density_2d.shape[0]):
        row_peaks, _ = find_peaks(density_2d[i, :], height=min_height, distance=min_distance)
        for j in row_peaks:
            x_coord = xx[i, 0]
            y_coord = yy[0, j]
            peaks.append((i, j, density_2d[i, j]))
            peak_coords.append((x_coord, y_coord, density_2d[i, j]))
    
    # Remove duplicates and sort by density value
    peak_coords = list(set(peak_coords))
    peak_coords.sort(key=lambda x: x[2], reverse=True)
    
    return peak_coords, density_2d, xx, yy

def visualize_density(data, cell_density, peak_coords, xx, yy, density_2d, save_path=None):
    """
    Visualize density distribution and peaks.

    Parameters:
    data: PCA coordinates
    cell_density: density values for each cell
    peak_coords: list of peak coordinates
    xx, yy: meshgrid coordinates
    density_2d: 2D density matrix
    save_path: optional path to save the figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Scatter plot of cells colored by density
    scatter = ax1.scatter(data[:, 0], data[:, 1], c=cell_density, 
                         s=100, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ax=ax1, label='Cell Density')
    ax1.set_title('Cells Colored by Density')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    
    # 2. KDE contour plot
    density_2d_normalized = zscore(density_2d)
    contour = ax2.contourf(xx, yy, density_2d_normalized, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax2, label='Density')
    
    # Optionally mark density peaks
    # for x, y, density_val in peak_coords[:10]:
    #     ax2.scatter(x, y, c='red', s=100, marker='x', linewidth=2)
    #     ax2.text(x, y, f'{density_val:.2f}', fontsize=8, 
    #             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    ax2.set_title('KDE with Density Peaks')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    
    # 3. Histogram of density values
    ax3.hist(cell_density, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_xlabel('Density Value')
    ax3.set_ylabel('Number of Cells')
    ax3.set_title('Distribution of Cell Density Values')
    ax3.axvline(np.mean(cell_density), color='red', linestyle='--', 
                label=f'Mean: {np.mean(cell_density):.3f}')
    ax3.legend()
    
    plt.tight_layout()
    if save_path is not None:   
        plt.savefig(save_path, dpi=300)
    plt.show()

def main(data, bandwidth=0.5, save_path=None):
    """
    Main function: compute density values for each cell and analyze.

    Returns:
    cell_density: density values for each cell
    peak_coords: coordinates of density peaks
    density_2d: 2D density matrix
    xx, yy: meshgrid coordinates
    """
    # Compute density values
    cell_density, kde_model = calculate_cell_density(data, bandwidth=bandwidth)
    
    # Find density peaks
    peak_coords, density_2d, xx, yy = find_density_peaks_kde(data, cell_density)
    
    # Visualization
    visualize_density(data, cell_density, peak_coords, xx, yy, density_2d, save_path=save_path)
    
    # Print summary statistics
    print(f"Computed density values for {len(cell_density)} cells")
    print(f"Density range: {cell_density.min():.4f} - {cell_density.max():.4f}")
    print(f"Mean density: {cell_density.mean():.4f}")
    print(f"Standard deviation: {cell_density.std():.4f}")
    
    print(f"\nFound {len(peak_coords)} density peaks")
    print("Top 5 density peaks (x, y, density):")
    for i, (x, y, density) in enumerate(peak_coords[:5]):
        print(f"{i+1}. ({x:.3f}, {y:.3f}): {density:.4f}")
    
    return cell_density, peak_coords, density_2d, xx, yy

def gini(x):
    """
    Compute the Gini coefficient for a numeric array.
    """
    x = np.asarray(x, dtype=float).flatten()
    if np.all(x == 0):
        return 0.0
    x_sorted = np.sort(x)
    n = len(x_sorted)
    cum = np.cumsum(x_sorted)
    g = (2.0 * np.sum((np.arange(1, n+1) * x_sorted))) / (n * cum[-1]) - (n + 1) / n
    return g

def merge_genes_groupby(adata, method='sum'):
    """
    Merge duplicate genes using pandas groupby.

    Parameters:
    adata: AnnData object
    method: aggregation method ('sum', 'mean', 'max')

    Returns:
    adata_merged: AnnData object with merged genes
    """
    # Convert expression matrix to DataFrame
    expr_df = pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
        index=adata.obs.index,
        columns=adata.var.index
    )
    
    # Group by gene name and aggregate
    if method == 'sum':
        merged_expr = expr_df.groupby(level=0, axis=1).sum()
    elif method == 'mean':
        merged_expr = expr_df.groupby(level=0, axis=1).mean()
    elif method == 'max':
        merged_expr = expr_df.groupby(level=0, axis=1).max()
    
    # Create new AnnData object
    adata_merged = sc.AnnData(
        X=merged_expr.values,
        obs=adata.obs,
        var=pd.DataFrame(index=merged_expr.columns)
    )
    
    return adata_merged
