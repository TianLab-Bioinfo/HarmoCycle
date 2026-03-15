import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
import scanpy as sc

# =================================================================
# 1. Model Architecture
# =================================================================

class TransformerEncoderBlock(nn.Module):
    """
    Wrapper for a single-layer Transformer Encoder.
    """
    def __init__(self, d_model, dim_feedforward, num_heads=4):
        super().__init__()
        self.encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, x):
        return self.transformer_encoder(x)

class HarmonicTransformerAutoencoder(nn.Module):
    """
    Autoencoder with Transformer bottleneck.
    Forces the latent space to be centered around the origin (0,0)
    to facilitate angular/harmonic analysis.
    """
    def __init__(self, input_dim, latent_dim=2):
        super().__init__()
        
        # --- Encoder ---
        self.encoder_input = nn.Linear(input_dim, 64)
        self.transformer = TransformerEncoderBlock(64, 512, num_heads=4)
        self.encoder_output = nn.Linear(64, latent_dim)

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        # Project to embedding dimension
        x = self.encoder_input(x)
        
        # Reshape for Transformer: (Batch, Sequence_Len=1, Features)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        
        # Map to latent space
        z = self.encoder_output(x)
        
        # --- Center Latent Space ---
        # Subtract batch mean to force data to center at (0,0).
        # This is crucial for 'angle' based sorting.
        z = z - torch.mean(z, dim=0, keepdim=True)
        
        recon = self.decoder(z)
        return z, recon

# =================================================================
# 2. Sorting Strategies (Interface)
# =================================================================

def get_sorting_indices(latent, strategy='angle'):
    """
    Determines the order of cells for the Fourier Transform.
    
    Args:
        latent (Tensor): Latent vectors (Batch, 2).
        strategy (str or callable): 
            - 'angle': Sort by angle relative to origin (Cyclic/Ring topology).
            - 'x_axis': Sort by X-coordinate value (Linear topology).
            - 'pca': Sort by projection onto the 1st Principal Component (Linear).
            - callable: A function that takes `latent` and returns indices.
            
    Returns:
        Tensor: Indices to sort the batch.
    """
    if callable(strategy):
        return strategy(latent)

    if strategy == 'angle':
        # Default for cycles: atan2(y, x)
        # Latent is already centered by the model
        angles = torch.atan2(latent[:, 1], latent[:, 0])
        return torch.argsort(angles)
    
    elif strategy == 'x_axis':
        # Simple linear sort along the first dimension
        return torch.argsort(latent[:, 0])
    
    elif strategy == 'pca':
        # Sort along the direction of maximum variance in the batch
        # Useful if the ring is squashed or linear
        _, _, V = torch.pca_lowrank(latent, q=1)
        # Project data onto the first PC
        proj = torch.matmul(latent, V[:, :1])
        return torch.argsort(proj.squeeze())
        
    else:
        raise ValueError(f"Unknown sorting strategy: {strategy}")

# =================================================================
# 3. Fourier Penalty (Gini Only)
# =================================================================

def fourier_penalty_gini(latent, expression_data, sort_strategy='angle'):
    """
    Calculates the Gini coefficient of the Fourier spectrum magnitude.
    Higher Gini = sparser spectrum = smoother trajectory.
    
    Args:
        latent: Latent representations (Batch, 2)
        expression_data: Original gene expression (Batch, Genes)
        sort_strategy: Method to order the cells before FFT.
    """
    # 1. Get sorting indices based on the chosen strategy
    sorted_idx = get_sorting_indices(latent, strategy=sort_strategy)
    
    # 2. Reorder expression data to form a 'time-series'
    sorted_expr = expression_data[sorted_idx]
    
    # 3. Compute FFT along the batch dimension (dim=0)
    fft_result = torch.fft.fft(sorted_expr, dim=0)
    
    # 4. Compute magnitudes
    x = torch.abs(fft_result) + 1e-8
    
    # 5. Calculate Gini Coefficient (Vectorized)
    n = x.size(0)
    sorted_x, _ = torch.sort(x, dim=0, descending=True)
    cumsum = torch.cumsum(sorted_x, dim=0)
    
    # Lorenz curve calculation
    Lorenz_curve = cumsum / cumsum[-1:] 
    uniform_line = torch.linspace(0, 1, steps=n, device=x.device).unsqueeze(1)
    
    # Gini = Area between uniform line and Lorenz curve
    gini = 1 - 2 * torch.trapz(Lorenz_curve, uniform_line, dim=0)
    
    # Return negative mean Gini (we want to MAXIMIZE Gini, so minimize negative)
    return -torch.mean(gini)

# =================================================================
# 4. Training Logic
# =================================================================

def prepare_data(adata, top_n_genes=None, if_scale=False, device='auto'):
    """Prepare AnnData for PyTorch training."""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Feature selection
    if top_n_genes is not None and top_n_genes > 0:
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=top_n_genes, inplace=True)
        adata = adata[:, adata.var.highly_variable]
    
    # Scaling
    if if_scale:
        sc.pp.scale(adata)
        
    # Convert to dense tensor
    if hasattr(adata.X, "toarray"):
        gene_expression = adata.X.toarray()
    else:
        gene_expression = adata.X
        
    input_dim = gene_expression.shape[1]
    data_tensor = torch.tensor(gene_expression, dtype=torch.float32).to(device)
    
    return data_tensor, input_dim, device

def train_model(model, data_tensor, epochs=100, batch_size=128, 
               sort_strategy='angle', device='cuda', if_verbose=True):
    """
    Main training loop.
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_history = []

    model.train()
    
    for epoch in range(epochs):
        permutation = torch.randperm(data_tensor.size(0))
        epoch_loss = 0

        for i in range(0, data_tensor.size(0), batch_size):
            batch_indices = permutation[i:i+batch_size]
            batch_data = data_tensor[batch_indices]
            
            # Forward pass
            z, recon = model(batch_data)
            
            # Reconstruction Loss (MSE)
            recon_loss = F.mse_loss(recon, batch_data)
            
            # Fourier Penalty (Gini)
            # Uses the flexible sorting interface
            freq_penalty = fourier_penalty_gini(z, batch_data, sort_strategy=sort_strategy)
            
            # Total Loss
            loss = recon_loss + 0.5 * freq_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / (data_tensor.size(0) // batch_size + 1)
        loss_history.append(avg_loss) # Log total loss for stability monitoring
        
        if if_verbose:
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return loss_history

# =================================================================
# 5. Main Execution Pipeline
# =================================================================

def run_pipeline(adata, epochs=500, batch_size=128, 
                top_n_genes=2000, 
                sort_strategy='angle',  # <--- NEW INTERFACE
                device='auto',
                if_scale=False,
                seed=20250618,
                if_verbose=True):
    """
    Run the complete training pipeline.
    
    Parameters:
    - adata: AnnData object.
    - sort_strategy: 'angle', 'x_axis', 'pca', or a custom function.
                     Change this if cells pile up or center is misplaced.
    """
    # Set reproducibility
    def set_seed(s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
            torch.backends.cudnn.deterministic = True 
        np.random.seed(s)
        random.seed(s)
        os.environ['PYTHONHASHSEED'] = str(s)
    
    set_seed(seed)

    # Prepare Data
    data_tensor, input_dim, device = prepare_data(
        adata, 
        top_n_genes if top_n_genes != -1 else None, 
        if_scale=if_scale, 
        device=device
    )
    
    # Initialize Model
    model = HarmonicTransformerAutoencoder(input_dim).to(device)
    
    # Train
    loss_history = train_model(
        model, data_tensor, 
        epochs=epochs, 
        batch_size=batch_size,
        sort_strategy=sort_strategy,
        device=device,
        if_verbose=if_verbose
    )

    # Evaluation & Embedding Extraction
    with torch.no_grad():
        model.eval()
        z, _ = model(data_tensor)
        latent_np = z.cpu().numpy()
        
        # Calculate final pseudo-time/angle for visualization
        # Note: We use the same strategy as training for consistency, 
        # or default to angle for circular plotting.
        if sort_strategy == 'angle' or sort_strategy == 'pca':
            angles = np.arctan2(latent_np[:, 1], latent_np[:, 0])
        else:
            # Fallback for linear strategies
            angles = latent_np[:, 0] 
    
    # Store in AnnData
    adata.obsm['X_latent'] = latent_np
    adata.obs['pseudo_angle'] = angles
    
    return model, loss_history, adata

# =================================================================
# Example Usage: Custom Sorting Function
# =================================================================
# If the default 'angle' causes bunching in the center, you can try
# defining a custom sorter that emphasizes the outer ring or linear projection.

def custom_robust_angle_sort(latent):
    """
    Example custom sorter:
    Ignores points too close to the center when calculating angles 
    (though difficult to implement strictly in batch training without masking).
    """
    # Standard angle
    angles = torch.atan2(latent[:, 1], latent[:, 0])
    return torch.argsort(angles)

# To run:
# model, history, new_adata = run_pipeline(adata, sort_strategy='angle')
# OR
# model, history, new_adata = run_pipeline(adata, sort_strategy='x_axis') # For linear trajectories


import numpy as np
import pandas as pd
import scanpy as sc

def get_periodic_matrix(adata: sc.AnnData, top_k: int = 5, use_highly_variable: bool = True) -> sc.AnnData:
    """
    Identifies the top k dominant frequencies for each gene based on the learned
    cyclic order and reconstructs a smooth, periodic expression matrix.

    This function performs the following steps:
    1. Orders cells and their expression data by `pseudo_angle`.
    2. Applies a Fast Fourier Transform (FFT) to each gene's ordered expression profile.
    3. For each gene, identifies the `top_k` frequencies with the highest magnitude.
    4. Creates a filtered FFT matrix, zeroing out all non-dominant frequencies.
    5. Applies an Inverse FFT (IFFT) to get the smoothed, periodic expression matrix.
    6. Stores the result in `adata.layers['periodic']` and the top frequencies in
       `adata.varm['top_k_frequencies']`.

    Args:
        adata: The AnnData object after running the main pipeline. Must contain
               `adata.obs['pseudo_angle']`.
        top_k: The number of dominant frequencies to keep for each gene.
        use_highly_variable: If True, uses only the genes marked as highly variable
                             (from the `run_pipeline` step). If False, uses all genes.

    Returns:
        The updated AnnData object with the new layer and gene annotations.
    """
    print(f"\n⚡ Generating Periodic Matrix with Top {top_k} Frequencies...")
    
    if 'pseudo_angle' not in adata.obs:
        raise ValueError("`pseudo_angle` not found in adata.obs. Please run the main pipeline first.")

    # --- 1. Get and sort data ---
    print("  - Step 1: Sorting cells by pseudo_angle...")
    if use_highly_variable and 'highly_variable' in adata.var and adata.var['highly_variable'].sum() > 0:
        hvg_mask = adata.var['highly_variable']
        expression_data = adata[:, hvg_mask].X.toarray() if hasattr(adata.X, "toarray") else adata[:, hvg_mask].X
        gene_names = adata.var_names[hvg_mask]
    else:
        expression_data = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        gene_names = adata.var_names

    n_cells, n_genes = expression_data.shape
    sort_indices = np.argsort(adata.obs['pseudo_angle'].values)
    sorted_expression = expression_data[sort_indices]

    # --- 2. Perform FFT ---
    print("  - Step 2: Performing Fast Fourier Transform (FFT)...")
    fft_result = np.fft.fft(sorted_expression, axis=0)
    
    # --- 3. Identify Top K frequencies for each gene ---
    print(f"  - Step 3: Identifying top {top_k} frequencies per gene...")
    fft_magnitudes = np.abs(fft_result)
    
    # We only need to find top frequencies in the first half due to symmetry
    # np.argsort returns indices of smallest to largest, so we take the last k
    # We also handle the case where n_cells // 2 is smaller than top_k
    search_space = min(top_k, n_cells // 2)
    top_k_indices = np.argsort(fft_magnitudes[:n_cells // 2], axis=0)[-search_space:]

    # --- 4. Filter FFT results (Masking) ---
    print("  - Step 4: Filtering non-dominant frequencies...")
    filtered_fft = np.zeros_like(fft_result, dtype=np.complex128)
    
    # Create an index for the columns (genes)
    gene_idx = np.arange(n_genes)
    
    # Use advanced indexing to set the top k positive frequencies.
    # NumPy correctly broadcasts gene_idx to match the shape of top_k_indices.
    filtered_fft[top_k_indices, gene_idx] = fft_result[top_k_indices, gene_idx]
    
    # ========================== CORRECTED CODE BLOCK ==========================
    # To handle the symmetric part correctly without flattening arrays,
    # we calculate all potential symmetric indices and then use a boolean mask.
    
    # Create a mask to identify which of the top frequencies are not the DC component (index 0)
    non_dc_mask = (top_k_indices > 0)
    
    # Get the top k indices that are not the DC component
    non_dc_indices = top_k_indices[non_dc_mask]
    
    # Calculate their symmetric counterparts
    symmetric_indices = n_cells - non_dc_indices
    
    # We need to know which gene each of these indices belongs to.
    # We create a column index array that matches the shape of top_k_indices
    gene_idx_broadcasted = np.broadcast_to(gene_idx, top_k_indices.shape)
    # Then we mask it to get the corresponding column for each non_dc_index
    corresponding_gene_idx = gene_idx_broadcasted[non_dc_mask]
    
    # Use these matched 1D arrays for coordinate indexing
    filtered_fft[symmetric_indices, corresponding_gene_idx] = fft_result[symmetric_indices, corresponding_gene_idx]
    # ==========================================================================

    # --- 5. Perform Inverse FFT ---
    print("  - Step 5: Performing Inverse FFT (IFFT)...")
    periodic_matrix_sorted = np.fft.ifft(filtered_fft, axis=0)
    
    # The result should be real, but take np.real to discard tiny imaginary noise
    periodic_matrix_sorted_real = np.real(periodic_matrix_sorted)
    
    # --- 6. Un-sort the matrix to match original adata cell order ---
    print("  - Step 6: Storing results in AnnData object...")
    unsort_indices = np.argsort(sort_indices)
    periodic_matrix_final = periodic_matrix_sorted_real[unsort_indices]
    
    # Store the results back in the anndata object
    # Create a full-size matrix for the layer
    final_layer = np.zeros(adata.shape, dtype=np.float32)
    if use_highly_variable and 'highly_variable' in adata.var and adata.var['highly_variable'].sum() > 0:
        final_layer[:, hvg_mask] = periodic_matrix_final
    else:
        final_layer = periodic_matrix_final
        
    adata.X = final_layer
    
    # Store the top frequencies for each gene in .varm
    top_freq_df = pd.DataFrame(top_k_indices, columns=gene_names, index=[f'freq_{i+1}' for i in range(search_space)]).T
    # Use .reindex to ensure the DataFrame has all genes, filling non-HVGs with NaN
    adata.varm['top_k_frequencies'] = top_freq_df.reindex(adata.var_names) 
    
    print("✅ Done! Periodic matrix stored in `adata.X`.")
    return adata


import numpy as np
import scanpy as sc
from typing import Optional, List

def calculate_pca_angle(
    adata: sc.AnnData,
    use_rep: str = 'X',
    pca_key_added: str = 'X_pca_circle',
    angle_key_added: str = 'pca_angle',
    map_to_2pi: bool = False,
    plot: bool = True,
    color_by: Optional[List[str]] = None,
    **kwargs
) -> sc.AnnData:
    """
    Computes a circular pseudotime angle based on a 2-component PCA.

    This function performs a PCA on the specified data representation, calculates
    the centroid of the resulting 2D embedding, and then computes the angle of
    each cell relative to this centroid. This is a simple way to establish a
    circular ordering for cyclic processes.

    Args:
        adata:
            The AnnData object to process.
        use_rep:
            The data representation to use for PCA. Can be 'X' or a key from
            `adata.layers`, e.g., 'periodic'.
        pca_key_added:
            The key in `adata.obsm` where the 2D PCA coordinates will be stored.
        angle_key_added:
            The key in `adata.obs` where the calculated angles will be stored.
        map_to_2pi:
            If True, maps the angles from the default [-π, π] range to [0, 2π].
            Defaults to False.
        plot:
            If True, generates a scatter plot of the PCA embedding, colored by
            the new angle and other specified keys.
        color_by:
            A list of keys from `adata.obs` to use for coloring the plot. If None,
            it will default to coloring by the new angle and 'stage' (if available).
        **kwargs:
            Additional keyword arguments passed to `sc.pl.embedding`.

    Returns:
        The modified AnnData object with the new PCA coordinates and angles.
    """
    print(f"🔄 Calculating circular pseudotime from PCA on '{use_rep}'...")

    # --- Step 1: Get data and perform PCA ---
    if use_rep == 'X':
        data_matrix = adata.X
    elif use_rep in adata.layers:
        data_matrix = adata.layers[use_rep]
    else:
        raise ValueError(f"Representation '{use_rep}' not found in adata.X or adata.layers.")

    # Ensure data is dense for PCA
    if hasattr(data_matrix, "toarray"):
        data_matrix = data_matrix.toarray()

    # PCA must have 2 components for angle calculation
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(data_matrix)
    adata.obsm[pca_key_added] = pca_coords
    print(f"   -> Stored 2D PCA results in `adata.obsm['{pca_key_added}']`")


    # --- Step 2 & 3: Compute centroid and shift coordinates ---
    centroid = np.mean(pca_coords, axis=0)
    shifted_coords = pca_coords - centroid

    # --- Step 4: Compute angles ---
    angles = np.arctan2(shifted_coords[:, 1], shifted_coords[:, 0])  # y, x

    if map_to_2pi:
        angles = np.mod(angles, 2 * np.pi)

    # --- Step 5: Save angles to AnnData object ---
    adata.obs[angle_key_added] = angles
    print(f"   -> Saved calculated angles to `adata.obs['{angle_key_added}']`")

    # --- Step 6: Visualization ---
    if plot:
        print("   -> Generating visualization...")
        # Prepare a robust list of keys to color by
        if color_by is None:
            plot_colors = [angle_key_added]
            if 'stage' in adata.obs.columns:
                plot_colors.append('stage')
        else:
            plot_colors = color_by

        sc.pl.embedding(
            adata,
            basis=pca_key_added,
            color=plot_colors,
            title=f'PCA on {use_rep} (colored by angle)',
            cmap='hsv',
            **kwargs
        )

    return adata


import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def run_test(gene_set, adata_source, is_norm=True, is_scale=True):
    """
    Main function: execute the full workflow and return the expression matrix of adjust_adata_sub
    """
    
    # Step 0: Normalize and log-transform if required
    if is_norm:
        sc.pp.normalize_total(adata_source, target_sum=1e4)
        sc.pp.log1p(adata_source)
        
    adata3 = adata_source.copy()
    
    # Step 1: Prepare reference data (convert gene names to uppercase)
    adata3.var.index = adata3.var.index.str.upper()
    
    # Step 2: Select genes of interest and shuffle cells
    select_genes = gene_set
    temp_adata3 = adata3[:, select_genes].copy()
    temp_adata3 = temp_adata3[np.random.permutation(temp_adata3.n_obs), :]
    
    # Step 3: Scale data if required
    if is_scale:
        sc.pp.scale(temp_adata3)    
    
    # Step 4: Run the pipeline (custom function)
    _, _, adata_res1 = run_pipeline(
        temp_adata3,
        epochs=2000,
        batch_size=1024,
        top_n_genes=-1, 
        device='cuda:1'
    )
    
    # Step 5: Sort cells by pseudotime angle
    temp_adata3 = temp_adata3[temp_adata3.obs['pseudo_angle'].sort_values().index, :]
    
    # Step 6: Construct periodic matrix
    periodic_adata = get_periodic_matrix(temp_adata3.copy(), top_k=5, use_highly_variable=False)
    
    # Step 7: Calculate PCA-based angle
    adjust_adata_sub = calculate_pca_angle(
        periodic_adata,
        use_rep='X',
        pca_key_added='X_pca_adjust',
        angle_key_added='pca_adjust_angle',
        plot=True
    )
    
    # Step 8: Sort cells by adjusted PCA angle
    adjust_adata_sub = adjust_adata_sub[adjust_adata_sub.obs.sort_values(by='pca_adjust_angle').index, :]
    
    # Step 9: Return the processed expression matrix
    return adjust_adata_sub