import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Ensure diagrams directory exists
BASE_DIR = Path(__file__).resolve().parent.parent
DIAGRAMS_DIR = BASE_DIR / "diagrams"
DIAGRAMS_DIR.mkdir(exist_ok=True)

# Color Palette (Spotter Corporate / Modern Tech Theme)
C_PRIMARY = "#064A56"     # Deep Teal
C_SECONDARY = "#0284C7"   # Sky Blue
C_ACCENT = "#0D9488"      # Teal Green
C_DARK = "#1E293B"        # Charcoal Slate
C_LIGHT = "#F8FAFC"       # Off White
C_BORDER = "#94A3B8"      # Slate Gray Border
C_WARN = "#D97706"        # Amber Warning
C_SUCCESS = "#16A34A"     # Green
C_PURPLE = "#7C3AED"      # Purple

def create_system_architecture_diagram():
    """Generates UML Component / System Pipeline Architecture diagram."""
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Background canvas
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Title
    ax.text(2, 96, "Spotter Freight ML: End-to-End System & Architecture Pipeline", 
            fontsize=16, fontweight="bold", color=C_PRIMARY)
    ax.text(2, 92.5, "UML Component Dataflow & Modular Service Execution Model", 
            fontsize=10.5, color="#64748B")

    # Helper function for drawing rounded boxes
    def draw_box(x, y, w, h, title, subtitle, items, bg_color="#F8FAFC", border_color=C_BORDER, title_color=C_PRIMARY):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8,rounding_size=1.5",
                                     linewidth=1.5, edgecolor=border_color, facecolor=bg_color)
        ax.add_patch(box)
        ax.text(x + 2, y + h - 3.2, title, fontsize=11, fontweight="bold", color=title_color)
        if subtitle:
            ax.text(x + 2, y + h - 5.8, subtitle, fontsize=8, fontstyle="italic", color="#64748B")
        curr_y = y + h - 9.0
        for item in items:
            ax.text(x + 2, curr_y, f"• {item}", fontsize=8.5, color=C_DARK)
            curr_y -= 3.3

    # Layer 1: Data Ingestion & Sanitization
    draw_box(2, 54, 28, 35, "1. Ingestion & Preprocessing", "src/data_cleaning.py", [
        "train_test.csv (48k loads)",
        "validation.csv (12k loads)",
        "december_chart_inputs.csv (31)",
        "Negative Weight: abs(weight)",
        "Missing Weight: Equip Medians",
        "Missing Market: Daily Mean",
        "72 City Centroid Mapping (1:1)"
    ], bg_color="#F0FDF4", border_color="#86EFAC")

    # Layer 2: Domain Feature Engineering
    draw_box(35, 54, 30, 35, "2. Domain Feature Engine", "src/feature_engineering.py", [
        "Haversine Great-Circle Dist",
        "Route Circuity: Dist / Haversine",
        "Compass Bearing & Lat/Lon Diff",
        "Ton-Miles & Weight-per-Mile",
        "Base Quote: Dist × Quote Signal",
        "Market Adjusted Quote Signal",
        "7-Day Cyclical (dow_sin/cos)"
    ], bg_color="#EFF6FF", border_color="#93C5FD")

    # Layer 3: Dual Validation & Training
    draw_box(70, 54, 28, 35, "3. Dual Validation Engine", "src/train.py", [
        "Protocol 1: Expanding Time CV",
        "  - Win 1: Jan-Apr -> May-Jun",
        "  - Win 2: Jan-Jun -> Jul-Aug",
        "  - Win 3: Jan-Aug -> Sep-Oct",
        "Protocol 2: 5-Fold Bagging",
        "Target: log(posted / base)",
        "Loss: L1 Absolute Error"
    ], bg_color="#F5F3FF", border_color="#C4B5FD")

    # Arrows between top components
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=2, color=C_SECONDARY))
        if label:
            ax.text((x1 + x2)/2, y1 + 1.2, label, fontsize=8, ha="center", color=C_SECONDARY, fontweight="bold")

    draw_arrow(30, 71.5, 35, 71.5, "Sanitized")
    draw_arrow(65, 71.5, 70, 71.5, "30 Features")

    # Layer 4: Model Ensembling (Bottom Left & Middle)
    draw_box(2, 8, 45, 38, "4. Heterogeneous Gradient-Boosted Ensemble", "LightGBM + CatBoost Optimization", [
        "Engine 1: LightGBM Regressor (80% Weight)",
        "  - objective='regression_l1', n_estimators=1000, lr=0.02, num_leaves=35",
        "  - Leaf-wise tree growth with asymmetric split granularity (MAE: $116.16)",
        "Engine 2: CatBoost Regressor (20% Weight)",
        "  - loss_function='MAE', iterations=1000, lr=0.03, depth=6",
        "  - Oblivious symmetric trees with native categorical handling (MAE: $125.63)",
        "Ensemble Blend Formulation:",
        "  - Pred Rate = (Distance × Quote Signal) × exp(0.80·ln(R_lgb) + 0.20·ln(R_cb))",
        "  - Guarantees strictly positive rate (Rate > $0) & relative scale invariance"
    ], bg_color="#FEFCE8", border_color="#FDE047")

    # Layer 5: Inference, Scorer & Reports (Bottom Right)
    draw_box(53, 8, 45, 38, "5. Inference, Verification & Automated Reports", "src/predict.py & score.py", [
        "Inference 1: validation_predictions.csv (12,000 rows)",
        "  - load_id, predicted_rate strictly validated",
        "Inference 2: data/december_chart_inputs.csv (31 rows)",
        "  - Benchmark lane: Lexington -> Fort Wayne (360 mi, Dry Van)",
        "Verification: score.py Execution",
        "  - Produces official chart: scorer_results/candidate_december.png",
        "Automated Technical Reporting (src/report_generator.py):",
        "  - Spotter_ML_Assessment_Report.docx & .pdf with embedded figures",
        "  - LOOM_SCRIPT.md video presentation script with timestamps"
    ], bg_color="#F0FDF4", border_color="#86EFAC")

    # Vertical & Horizontal connection arrows
    # Arrow from Validation (70, 54) down to Model Ensemble (45, 46) - connecting cleanly to top right of Box 4
    ax.annotate("", xy=(44, 46), xytext=(75, 54),
                arrowprops=dict(arrowstyle="->", lw=2, color=C_SECONDARY,
                                connectionstyle="arc3,rad=-0.12"))
    ax.text(54, 49.5, "Optimal Hyperparameters & Weights", fontsize=8.5, fontweight="bold", color=C_SECONDARY)

    # Arrow from Model Ensemble (47, 27) to Inference (53, 27)
    draw_arrow(47, 27, 53, 27, "Deployed Models")

    output_path = DIAGRAMS_DIR / "system_architecture_uml.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {output_path}")

def create_validation_split_diagram():
    """Generates UML Timeline & Architecture of Dual Validation Splits."""
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    # Title
    ax.text(2, 95, "Validation Architecture: Expanding-Window Out-of-Time CV vs Production Bagging",
            fontsize=15, fontweight="bold", color=C_PRIMARY)
    ax.text(2, 91, "Methodological Elimination of Temporal Lookahead Leakage in Freight Market Dynamics",
            fontsize=10, color="#64748B")

    # Timeline Bar Guide
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_x = np.linspace(15, 87, 12)
    for i, m in enumerate(months):
        ax.text(month_x[i] + 2.5, 83.5, m, fontsize=9.5, fontweight="bold", ha="center", color=C_DARK)
        # Vertical grid lines
        ax.plot([month_x[i], month_x[i]], [12, 82], color="#E2E8F0", linestyle="--", lw=1)

    # Window 1
    ax.text(2, 73, "Window 1", fontsize=10, fontweight="bold", color=C_PRIMARY)
    ax.text(2, 69.5, "(Train 1-4, Val 5-6)", fontsize=7.5, color="#64748B")
    box1_tr = patches.Rectangle((15, 68), 24, 6, facecolor=C_SECONDARY, edgecolor=C_PRIMARY, alpha=0.85)
    box1_val = patches.Rectangle((39, 68), 12, 6, facecolor=C_WARN, edgecolor=C_PRIMARY, alpha=0.9)
    ax.add_patch(box1_tr)
    ax.add_patch(box1_val)
    ax.text(27, 70.2, "Train: 19,110 loads", color="white", fontweight="bold", fontsize=8.5, ha="center")
    ax.text(45, 70.2, "Val: 9,696", color="white", fontweight="bold", fontsize=8.5, ha="center")
    ax.text(88, 70.2, "MAE: $96.95 | R²: 0.8346", fontsize=9, fontweight="bold", color=C_PRIMARY)

    # Window 2
    ax.text(2, 58, "Window 2", fontsize=10, fontweight="bold", color=C_PRIMARY)
    ax.text(2, 54.5, "(Train 1-6, Val 7-8)", fontsize=7.5, color="#64748B")
    box2_tr = patches.Rectangle((15, 53), 36, 6, facecolor=C_SECONDARY, edgecolor=C_PRIMARY, alpha=0.85)
    box2_val = patches.Rectangle((51, 53), 12, 6, facecolor=C_WARN, edgecolor=C_PRIMARY, alpha=0.9)
    ax.add_patch(box2_tr)
    ax.add_patch(box2_val)
    ax.text(33, 55.2, "Train: 28,806 loads", color="white", fontweight="bold", fontsize=8.5, ha="center")
    ax.text(57, 55.2, "Val: 9,671", color="white", fontweight="bold", fontsize=8.5, ha="center")
    ax.text(88, 55.2, "MAE: $142.57 | R²: 0.8186", fontsize=9, fontweight="bold", color=C_PRIMARY)

    # Window 3 (Holdout)
    ax.text(2, 43, "Window 3 (Holdout)", fontsize=10, fontweight="bold", color=C_PRIMARY)
    ax.text(2, 39.5, "(Train 1-8, Val 9-10)", fontsize=7.5, color="#64748B")
    box3_tr = patches.Rectangle((15, 38), 48, 6, facecolor=C_SECONDARY, edgecolor=C_PRIMARY, alpha=0.85)
    box3_val = patches.Rectangle((63, 38), 12, 6, facecolor="#DC2626", edgecolor=C_PRIMARY, alpha=0.9)
    ax.add_patch(box3_tr)
    ax.add_patch(box3_val)
    ax.text(39, 40.2, "Train: 38,477 loads (Jan - Aug)", color="white", fontweight="bold", fontsize=8.5, ha="center")
    ax.text(69, 40.2, "Holdout: 9,523", color="white", fontweight="bold", fontsize=8.5, ha="center")
    ax.text(88, 40.2, "MAE: $117.78 | R²: 0.8255", fontsize=9, fontweight="bold", color="#DC2626")

    # Production Deployment Target (Nov-Dec)
    ax.text(2, 28, "Production Target", fontsize=10, fontweight="bold", color=C_PRIMARY)
    ax.text(2, 24.5, "validation.csv", fontsize=7.5, color="#64748B")
    box_all = patches.Rectangle((15, 23), 60, 6, facecolor="#064A56", edgecolor=C_PRIMARY, alpha=0.9)
    box_val = patches.Rectangle((75, 23), 14, 6, facecolor="#16A34A", edgecolor=C_PRIMARY, alpha=0.9)
    ax.add_patch(box_all)
    ax.add_patch(box_val)
    ax.text(45, 25.2, "Production 5-Fold Bagged Ensemble: 48,000 Loads (Jan - Oct)", color="white", fontweight="bold", fontsize=8.5, ha="center")
    ax.text(82, 25.2, "Nov - Dec Test (12,000)", color="white", fontweight="bold", fontsize=8.5, ha="center")
    ax.text(82, 18.8, "OOF CV MAE: $86.11", fontsize=8.5, fontweight="bold", color="#16A34A", ha="center")

    # Summary Callout Box at Bottom
    callout = patches.FancyBboxPatch((15, 3), 80, 13, boxstyle="round,pad=0.5,rounding_size=1",
                                    facecolor="#F0F9FF", edgecolor=C_SECONDARY, linewidth=1.2)
    ax.add_patch(callout)
    ax.text(17, 12.5, "CRITICAL METHODOLOGICAL TAKEAWAYS:", fontsize=9.5, fontweight="bold", color=C_SECONDARY)
    ax.text(17, 8.8, "1. Random K-Fold CV introduces severe temporal lookahead leakage (OOF $86.11 is artificially optimistic).", fontsize=8.5, color=C_DARK)
    ax.text(17, 5.5, "2. Expanding Window Time CV ($119.10 Mean MAE) provides the honest benchmark for forward quarterly deployment.", fontsize=8.5, color=C_DARK)

    output_path = DIAGRAMS_DIR / "validation_strategy_uml.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {output_path}")

def create_target_decision_flow():
    """Generates UML Activity / Decision Matrix for Target & Component Formulation."""
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    # Title
    ax.text(2, 95, "Component & Target Formulation Decision Matrix",
            fontsize=15, fontweight="bold", color=C_PRIMARY)
    ax.text(2, 91, "Empirical Evaluation of Loss Functions, Target Scaling, and Component Architectures",
            fontsize=10, color="#64748B")

    # Table coordinates
    cols = [2, 22, 42, 62, 82]
    rows = [82, 67, 52, 37, 22, 7]

    # Draw Table Header
    headers = ["Component / Choice", "Candidate Approaches", "Holdout MAE ($)", "Key Limitation / Risk", "Final Decision & Rationale"]
    col_widths = [20, 20, 16, 20, 22]

    for j, (h, w) in enumerate(zip(headers, col_widths)):
        rect = patches.Rectangle((cols[j], 83), w, 6, facecolor=C_PRIMARY, edgecolor="white")
        ax.add_patch(rect)
        ax.text(cols[j] + w/2, 85.8, h, fontsize=8.5, fontweight="bold", color="white", ha="center")

    # Row 1: Target Formulation
    r1_data = [
        "Target Variable\nFormulation",
        "A: Direct posted_rate\nB: posted - base (residual)\nC: log(posted / base)",
        "A: $137.23\nB: $117.06\nC: $117.14",
        "Direct target fails on scale.\nResidual risks negative rates.",
        "SELECTED: C (log_ratio)\nEnforces strictly positive rate;\nscale invariant across hauls."
    ]
    # Row 2: Loss Objective
    r2_data = [
        "Loss Function &\nObjective",
        "A: L2 Mean Squared Error\nB: Huber / Pseudo-Huber\nC: L1 Mean Absolute Error",
        "A: $132.40\nB: $122.80\nC: $116.16",
        "L2 heavily distorted by rate spikes.\nHuber hyperparameter sensitive.",
        "SELECTED: C (L1 MAE)\nVastly superior outlier resistance;\ndirectly optimizes competition metric."
    ]
    # Row 3: City Geometry
    r3_data = [
        "Geographic Node\nRepresentation",
        "A: Discrete One-Hot City IDs\nB: Target-Encoded Cities\nC: Continuous GPS Centroids",
        "A: Fails on new nodes\nB: High overfit risk\nC: $95.03 on new nodes",
        "Discrete IDs fail on 8 unseen\nvalidation cities (San Diego, etc.).",
        "SELECTED: C (Continuous GPS)\nHaversine, circuity & bearing\ngeneralize to all new nodes."
    ]
    # Row 4: Calendar Mechanics
    r4_data = [
        "Temporal Seasonality\nEncoding",
        "A: Continuous dayofyear\nB: Month Index (1-12)\nC: 7-Day Cyclical sin/cos",
        "A: Extrapolates badly\nB: High forward drift\nC: $117.78 on Sep-Oct",
        "Tree models cannot extrapolate\nforward dayofyear to Nov-Dec.",
        "SELECTED: C (7-Day Cyclical)\nCaptures weekly logistics cycles\nwithout temporal extrapolation drift."
    ]
    # Row 5: Model Ensemble
    r5_data = [
        "Model Engine &\nEnsemble Blend",
        "A: Single LightGBM\nB: 50% LGBM + 50% CatBoost\nC: 80% LGBM + 20% CatBoost",
        "A: $116.16 (Higher var)\nB: $121.35\nC: $117.78 (RMSE: 637.4)",
        "Single model vulnerable to seed/split\nvariance; 50/50 dilutes accuracy.",
        "SELECTED: C (80/20 Blend)\nCombines LGBM leaf precision with\nCatBoost oblivious tree variance control."
    ]

    all_rows = [r1_data, r2_data, r3_data, r4_data, r5_data]
    y_starts = [67, 51, 35, 19, 3]

    for i, (r, y_top) in enumerate(zip(all_rows, y_starts)):
        bg = "#F8FAFC" if i % 2 == 0 else "#FFFFFF"
        if i == 0:
            bg = "#EFF6FF"
        for j, (val, w) in enumerate(zip(r, col_widths)):
            rect = patches.Rectangle((cols[j], y_top), w, 15, facecolor=bg, edgecolor="#CBD5E1", linewidth=0.8)
            ax.add_patch(rect)
            lines = val.split("\n")
            line_y = y_top + 12
            for line in lines:
                is_bold = "SELECTED" in line or "MAE" in line or "Formulation" in line or "Engine" in line
                c = C_PRIMARY if "SELECTED" in line else (C_DARK if not is_bold else "#0369A1")
                ax.text(cols[j] + 1.2, line_y, line, fontsize=7.2, fontweight="bold" if is_bold else "normal", color=c)
                line_y -= 2.6

    output_path = DIAGRAMS_DIR / "target_decision_flow.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {output_path}")

def create_model_benchmark_comparison_chart():
    """Generates high-resolution comparative bar chart of model benchmarks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    models = [
        "Baseline\nHeuristic",
        "Ridge\nLinear",
        "CatBoost\n(MAE)",
        "XGBoost\n(L1)",
        "LightGBM\n(L1)",
        "Ensemble\n(80/20)",
        "5-Fold CV\n(OOF)"
    ]
    maes = [246.02, 149.15, 125.63, 120.10, 116.16, 117.78, 86.11]
    mapes = [12.11, 8.42, 5.23, 5.08, 4.90, 4.95, 3.90]

    bar_colors = ["#94A3B8", "#64748B", "#F59E0B", "#F97316", "#0284C7", "#064A56", "#16A34A"]

    # Chart 1: Mean Absolute Error (MAE $)
    bars1 = ax1.bar(models, maes, color=bar_colors, width=0.6, edgecolor=C_PRIMARY, linewidth=1.2)
    ax1.set_title("Out-of-Time MAE ($) Comparison (Lower is Better)", fontsize=11.5, fontweight="bold", pad=12, color=C_PRIMARY)
    ax1.set_ylabel("Mean Absolute Error ($)", fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 275)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars1, maes):
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, y + 4, f"${val:.2f}", ha="center", fontsize=8.5, fontweight="bold", color=C_DARK)

    # Reduction callout on chart 1
    ax1.annotate("52.1% Error\nReduction", xy=(5, 120), xytext=(2.2, 200),
                 arrowprops=dict(facecolor="#DC2626", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=8.5, fontweight="bold", color="#DC2626",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEE2E2", edgecolor="#DC2626"))

    # Chart 2: MAPE (%) Comparison
    bars2 = ax2.bar(models, mapes, color=bar_colors, width=0.6, edgecolor=C_PRIMARY, linewidth=1.2)
    ax2.set_title("Mean Absolute Percentage Error (MAPE %) (Lower is Better)", fontsize=11.5, fontweight="bold", pad=12, color=C_PRIMARY)
    ax2.set_ylabel("MAPE (%)", fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 14)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars2, mapes):
        y = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, y + 0.25, f"{val:.2f}%", ha="center", fontsize=8.5, fontweight="bold", color=C_DARK)

    output_path = DIAGRAMS_DIR / "model_benchmark_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {output_path}")

def create_hyperparameter_sensitivity_chart():
    """Generates charts for LightGBM tuning and Ensemble blend weights."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Plot 1: LightGBM num_leaves vs learning rate
    leaves = [35, 45, 55]
    mae_lr02 = [116.16, 117.04, 116.34]
    mae_lr03 = [118.29, 118.02, 116.94]
    mae_lr04 = [118.33, 118.84, 118.85]

    ax1.plot(leaves, mae_lr02, marker="o", lw=2.4, color="#0284C7", label="Learning Rate = 0.02 (Optimal)")
    ax1.plot(leaves, mae_lr03, marker="s", lw=2.0, color="#F59E0B", label="Learning Rate = 0.03")
    ax1.plot(leaves, mae_lr04, marker="^", lw=2.0, color="#EF4444", label="Learning Rate = 0.04")
    ax1.scatter([35], [116.16], s=120, color="#064A56", zorder=5, edgecolor="white", lw=2)
    ax1.annotate("Best: $116.16\n(lr=0.02, leaves=35)", xy=(35, 116.16), xytext=(38, 115.5),
                 arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=1.5),
                 fontsize=8.5, fontweight="bold", color=C_PRIMARY)

    ax1.set_title("LightGBM Hyperparameter Sensitivity (Out-of-Time MAE)", fontsize=11, fontweight="bold", pad=10, color=C_PRIMARY)
    ax1.set_xlabel("Number of Leaves (num_leaves)", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Validation MAE ($)", fontsize=9.5, fontweight="bold")
    ax1.set_xticks([35, 45, 55])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.spines[["top", "right"]].set_visible(False)

    # Plot 2: Ensemble Weight Curve
    lgb_pcts = [0, 20, 40, 50, 60, 70, 75, 80, 85, 90, 100]
    ens_maes = [125.63, 124.20, 122.50, 121.35, 120.20, 118.65, 118.21, 117.78, 117.35, 116.94, 116.16]
    ens_rmse = [639.76, 639.10, 638.60, 638.10, 637.75, 637.65, 637.53, 637.41, 637.30, 637.19, 636.99]

    ax2.plot(lgb_pcts, ens_maes, marker="o", lw=2.4, color=C_PRIMARY, label="Ensemble MAE ($)")
    ax2.scatter([80], [117.78], s=130, color="#16A34A", zorder=5, edgecolor="white", lw=2)
    ax2.annotate("Selected Blend:\n80% LGB + 20% CB\n($117.78 MAE)", xy=(80, 117.78), xytext=(55, 117.0),
                 arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.5),
                 fontsize=8.5, fontweight="bold", color="#16A34A",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#DCFCE7", edgecolor="#16A34A"))

    ax2.set_title("Ensemble Blending Weight Trade-Off Curve", fontsize=11, fontweight="bold", pad=10, color=C_PRIMARY)
    ax2.set_xlabel("LightGBM Weight % (Remainder is CatBoost)", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Ensemble MAE ($)", fontsize=9.5, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8.5, loc="upper right")
    ax2.spines[["top", "right"]].set_visible(False)

    output_path = DIAGRAMS_DIR / "hyperparameter_sensitivity_matrix.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {output_path}")

if __name__ == "__main__":
    create_system_architecture_diagram()
    create_validation_split_diagram()
    create_target_decision_flow()
    create_model_benchmark_comparison_chart()
    create_hyperparameter_sensitivity_chart()
