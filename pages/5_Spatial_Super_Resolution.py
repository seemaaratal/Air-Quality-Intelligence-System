from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Spatial Super-Resolution",
    page_icon="🗺️",
    layout="wide"
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PAGE_DIR.parent

SRGAN_DIR = PROJECT_ROOT / "srgan_experiment"
OUTPUT_DIR = SRGAN_DIR / "outputs"

COMPARISON_IMAGE = (
    OUTPUT_DIR / "srgan_comparison.png"
)

RESULT_FILE = (
    OUTPUT_DIR / "final_srgan_evaluation.csv"
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("🗺️ AQI Spatial Super-Resolution")

st.write(
    """
    This module demonstrates an SRGAN-style spatial
    super-resolution approach for enhancing coarse AQI
    surfaces from **8 × 8 resolution to 32 × 32 resolution**.
    """
)

st.info(
    "This SRGAN module is implemented as a separate spatial "
    "enhancement experiment and does not modify the existing "
    "live AQI, LSTM forecasting, hotspot detection or "
    "recommendation modules."
)


# ---------------------------------------------------------
# Workflow
# ---------------------------------------------------------

st.subheader("⚙️ Super-Resolution Workflow")

st.markdown(
    """
    **Low-Resolution AQI Surface (8 × 8)**  
    ↓  
    **Bicubic Upscaling Baseline**  
    ↓  
    **Residual Generator**  
    ↓  
    **Adversarial Training with Discriminator**  
    ↓  
    **Enhanced AQI Surface (32 × 32)**
    """
)


# ---------------------------------------------------------
# Final Evaluation
# ---------------------------------------------------------

st.subheader("📊 Final Model Evaluation")

if RESULT_FILE.exists():

    results = pd.read_csv(RESULT_FILE)

    bicubic_row = results[
        results["Method"] == "Bicubic Baseline"
    ].iloc[0]

    generator_row = results[
        results["Method"] == "Generator V2"
    ].iloc[0]

    srgan_row = results[
        results["Method"] == "SRGAN Generator"
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Bicubic MAE",
            f"{bicubic_row['MAE_AQI']:.3f} AQI"
        )

    with col2:
        st.metric(
            "Generator V2 MAE",
            f"{generator_row['MAE_AQI']:.3f} AQI"
        )

    with col3:
        st.metric(
            "SRGAN MAE",
            f"{srgan_row['MAE_AQI']:.3f} AQI"
        )

    srgan_improvement = (
        (
            bicubic_row["MAE_AQI"]
            - srgan_row["MAE_AQI"]
        )
        / bicubic_row["MAE_AQI"]
    ) * 100

    srgan_vs_generator = (
        (
            generator_row["MAE_AQI"]
            - srgan_row["MAE_AQI"]
        )
        / generator_row["MAE_AQI"]
    ) * 100

    st.success(
        f"SRGAN reduced MAE by "
        f"{srgan_improvement:.2f}% compared with "
        f"bicubic interpolation."
    )

    st.write(
        f"SRGAN also improved MAE by "
        f"**{srgan_vs_generator:.2f}%** compared with "
        f"the non-adversarial Generator V2."
    )

    display_results = results.copy()

    display_results.columns = [
        "Method",
        "MAE (AQI)",
        "MSE",
        "RMSE (AQI)"
    ]

    display_results[
        ["MAE (AQI)", "MSE", "RMSE (AQI)"]
    ] = display_results[
        ["MAE (AQI)", "MSE", "RMSE (AQI)"]
    ].round(3)

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "Final SRGAN evaluation file was not found."
    )


# ---------------------------------------------------------
# Visual Comparison
# ---------------------------------------------------------

st.subheader("🖼️ Spatial Enhancement Comparison")

if COMPARISON_IMAGE.exists():

    st.image(
        str(COMPARISON_IMAGE),
        caption=(
            "Low-resolution AQI surface compared with "
            "bicubic interpolation, SRGAN output and "
            "the target interpolated AQI surface."
        ),
        use_container_width=True
    )

else:

    st.warning(
        "SRGAN comparison image was not found."
    )


# ---------------------------------------------------------
# Model Description
# ---------------------------------------------------------

st.subheader("🧠 SRGAN Architecture")

st.markdown(
    """
    The implemented spatial enhancement module contains:

    - **Generator:** learns spatial corrections over a
      bicubic-upscaled AQI surface.
    - **Residual Learning:** the model predicts corrections
      instead of rebuilding the complete surface from scratch.
    - **Discriminator:** distinguishes generated AQI surfaces
      from target AQI surfaces during adversarial training.
    - **Content Loss:** preserves numerical AQI accuracy.
    - **Adversarial Loss:** encourages the generated spatial
      pattern to resemble the target surface.
    """
)


# ---------------------------------------------------------
# Data Description
# ---------------------------------------------------------

st.subheader("📍 Training Data")

st.write(
    """
    The existing historical AQI dataset contains one fixed
    geographic coordinate for each of 10 representative
    Indian cities.

    Therefore, synchronized city AQI values were converted
    into spatial AQI surfaces using inverse-distance
    interpolation.

    These interpolated 32 × 32 surfaces were downsampled to
    8 × 8 and used to train the spatial super-resolution
    prototype.
    """
)


# ---------------------------------------------------------
# Important Limitation
# ---------------------------------------------------------

st.subheader("⚠️ Important Limitation")

st.warning(
    """
    The high-resolution targets used in this experiment are
    interpolated modelled AQI surfaces.

    They are NOT true street-level high-resolution sensor
    ground-truth maps.

    Therefore, the reported SRGAN performance demonstrates
    reconstruction capability on the experimental spatial
    dataset and should not be interpreted as validated
    street-level AQI accuracy.
    """
)


# ---------------------------------------------------------
# Final Status
# ---------------------------------------------------------

st.subheader("✅ Module Status")

st.success(
    """
    SRGAN-style AQI spatial super-resolution has been
    implemented and experimentally evaluated.

    The best test-set method was the SRGAN Generator.
    """
)