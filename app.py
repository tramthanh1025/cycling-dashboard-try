from random import choices
from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
from pathlib import Path
import seaborn as sns
import numpy as np
from shinywidgets import output_widget, render_widget
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Load data
base_path = Path(__file__).parent
origin_data = pd.read_parquet(base_path / "light_origin_data.parquet")
prediction_data = pd.read_parquet(base_path / "light_prediction_deviation_data.parquet")


# Global color scheme for consistency across all charts
color_blue  = "#4C78A8"  
color_green = "#72B7B2" 
color_red   = "#b2182b"  
color_orange = "#fdae61"
color_grey  = "#808080" 

color_cat = [
    "#ef8a62", 
    "#fdae61",  
    "#d36c95",  
    "#6a3d9a",
    "#9ab5d0",  
]


# 1. UI DEFINITION
app_ui = ui.page_navbar(
    # INTRO + DATA COLLECTION SITE
    ui.nav_panel("Introduction & Data Collection",
        
    ),


    # EDA SITE
    ui.nav_panel("Exploratory Data Analysis",
        ui.card(
            ui.markdown("""
                This EDA provides a comparative analysis of two primary cycling traffic datasets:
                * **Original Data (`origin_data`):** Ground truth observations.
                * **Prediction Data (`prediction_data`):** The 2025-2026 dataset representing the target period for our analysis and eventual model-generated projections.
            """)
        ),
        
        ui.hr(),
        ui.h2("Section 1: Traffic Overview"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section establishes the baseline behavior of the selected dataset.
        
                    Use these metrics to assess:
                    * **Baseline Overview:** Key metrics covering dataset scale, site coverage, traffic intensity, and directional split.
                    * **Traffic Distribution:** The typical range of cyclist activity.
                    * **Intensity Scaling:** The distribution of positive traffic counts using a logarithmic scale to identify typical vs. peak activity.
                """),
                ui.input_select("s1_dataset", "Choose Dataset:", 
                                choices={
                                    "origin_data": "Original Data", 
                                    "prediction_data": "Prediction Data"
                                }),
            ),
            ui.navset_card_tab(
                ui.nav_panel("Baseline Metrics", 
                    ui.output_ui("value_box_EDA"),
                    ui.markdown("---"),
                    ui.h5("Directional Split"),
                    ui.output_plot("directional_split_plot", height="100px"),
                ),
                ui.nav_panel("Traffic Distribution", 
                    ui.output_plot("traffic_distribution_plot")
                ),
                ui.nav_panel("Intensity Scaling", 
                    ui.output_plot("log_count_distribution_plot")
                )
            )
        ),
        ui.hr(),
        ui.h2("Section 2: Temporal Rhythm"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section explores the time-based patterns and behavioral rhythms of cycling traffic.
                    
                    **What to explore:**
                    * **Temporal Balance:** The distribution of traffic observations across hours, days, and months to check for uniform data coverage.
                    * **Traffic Heatmaps:** Density variations of cycling volume across different times of day, days of the week, and travel directions.
                """),
                ui.input_select("s2_dataset", "Choose Dataset:", 
                                choices={
                                    "origin_data": "Original Data", 
                                    "prediction_data": "Prediction Data"
                                }),
            ),
            ui.navset_card_tab(
                ui.nav_panel("Temporal Balance", 
                    ui.row(
                        ui.column(6, 
                            ui.input_radio_buttons(
                                "s2_time_var", 
                                "Select Time Scale:", 
                                choices={"hour_bin": "Hour bin", "weekday": "Day", "month": "Month"}, 
                                inline=True
                            )
                        ),
                        ui.column(6,
                            ui.input_select(
                                "s2_metric",
                                "Select Metric:",
                                choices={
                                    "n_rows": "Number of observations", 
                                    "cyclists": "Cyclist volume"
                                }
                            )
                        )
                    ),
                    ui.output_plot("temporal_dist_plot")
                ),
                ui.nav_panel("Traffic Heatmaps", 
                    ui.input_select(
                        "s2_heatmap_var", 
                        "Select Heatmap View:", 
                        choices={"weekday": "Hour × Day", "direction": "Hour × Direction"},
                        width="300px"
                    ),
                    ui.output_plot("temporal_heatmap_plot")
                )
            )
        ),
        ui.hr(),
        ui.h2("Section 3: Economics & Environmental Factors (Fuel Price)"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section explores the temporal historical trends of fuel prices (Petrol 95).
                """),
                ui.input_select("s3_dataset", "Choose Dataset:", 
                                choices={
                                    "origin_data": "Original Data", 
                                    "prediction_data": "Prediction Data"
                                }),
            ),
            ui.card( 
                ui.output_plot("fuel_time_series_plot")
            )
        ),
        ui.hr(),
        ui.h2("Section 4: Weather Dimensions (2025-2026)"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section isolates the 2025-2026 prediction dataset to explore how meteorological conditions impact cycling volumes.
                    
                    * **Dual-Axis View:** Compares the average traffic intensity (line) against the raw volume of observations (bars) for each condition.
                    * **Data Volume:** The gray bars indicate how much confidence we have in the average. Low bars mean rare weather events.
                """)
            ),
            ui.card(
                ui.input_select(
                    "s4_weather_var", "Select Weather Variable:", 
                    choices={
                        "temperature_mean": "Temperature", 
                        "precipitation_category": "Precipitation"
                    },
                    width="300px"
                ),
                ui.output_plot("weather_impact_plot")
            )
        ),
        ui.hr(),
        ui.h2("Section 5: Special Events Patterns (2025-2026)"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section explores our first impressions of the data by comparing cycling volumes during special events and holidays against normal days. This is an initial look at raw trends, prior to running the data through our predictive models.
                    
                    * **Baseline:** A "Normal Day" where no events or strikes are occurring.
                    * **Relative Difference:** The raw difference percentage difference in traffic during an event period compared to the baseline.
                    * **Low Confidence:** Events representing less than 1% of the dataset are ghosted/striped, as their averages are highly sensitive to outliers.
                """)
            ),
            ui.card(
                ui.h5("Impact Summary Table"),
                ui.output_data_frame("events_frequency_table"),
                ui.hr(),
                output_widget("events_impact_plot")
            )
        )

    ),

    # MODEL SITE
    ui.nav_panel("Model Construction and Evaluation",
        
        # First model
        ui.h2("Section 1: Modelling Normal Cycling Patterns"),
        ui.br(),
    
        # --- SECTION: MODEL SPECIFICATION ---
        ui.h4("Model Specification"),

        # Full-width column: Timeline Card
        ui.card(
            ui.card_header("Temporal Data Split"),
            output_widget("train_test_timeline_plot")
        ),
        
        # Side-by-side columns: Formula Description & Model Selection
        ui.layout_columns(
            # Left column: Predictors & Formula Card
            ui.card(
                ui.card_header("Model Specification & Formula"),
                ui.markdown("""
                    **Core Decomposition:**
                    
                    > **Observed count** = *Regular structure* + *Irregular deviation* + *Noise*
                    
                    **Response Variable:** * `count_rescaled` (Cyclist volume per 2-hour interval)
                    
                    **Primary Predictors (Regular Structure):**
                    * **Temporal:** `hour_bin`, `month`
                    * **Spatial & Directional:** `site_id`, `direction`
                    * **Calendar-related:** `weekday`, `is_public_holiday`, `is_school_holiday`
                    * **Background Variation:** `fuel_price` *(Accounts for gradual macro-trends in motorised transport costs)*
                    """
                )
            ),
            
            # Right column: Model Selection Card
            ui.card(
                ui.card_header("Model Selection"),
                ui.input_radio_buttons(
                    id="model_choice", 
                    label=ui.HTML("Evaluate Candidate Models:<br><br>"), 
                    choices={
                        "poisson": "Poisson Regression", 
                        "nb": "Negative Binomial (Final Choice)", 
                        "gb": "Histogram Gradient Boosting"
                    }
                ),
                ui.output_ui("dynamic_model_formula")
            )
            
        ),

        ui.hr(),
        
        # --- SECTION: RESULT ---
        ui.h4("Model Results & Performance"),
        
        ui.h6("Zero-Share Capture Comparison", class_="mb-3 text-muted"),
        ui.layout_columns(
            ui.value_box("Observed Zeros in Training Data", "28.3%", style="background-color: #808080; color: white;"),
            ui.value_box("Poisson Predicted Zeros", "8.2%", style="background-color: #b2182b; color: white;"),
            ui.value_box(ui.strong("Negative Binomial Predicted Zeros"), ui.strong("25.0%"), style="background-color: #72B7B2; color: black;"),
        ),
        
        ui.card(
            ui.card_header("Performance Metric Comparison"),
            ui.output_data_frame("model_metrics_table")
        ),

        ui.hr(),

        # DEVIATION CONSTRUCTION
        ui.h2("Section 2: Defining Deviations"),
        
        # Visual Step 1 & Step 2 Flow
        ui.layout_columns(
            ui.card(
                ui.div(
                    ui.h4("Step 1: Model Refitting", class_="text-primary"),
                    ui.p("The final Negative Binomial model is refitted on the entire model-development dataset (May 2022 – April 2025).")
                ),
                class_="text-center p-3 border-top border-primary border-3 shadow-sm"
            ),
            ui.card(
                ui.div(
                    ui.h4("Step 2: Prediction & Comparison", class_="text-primary"),
                    ui.p("For each observation in the prediction period (May 2025 – April 2026), we compute the expected normal count and measure the difference against the observed count.")
                ),
                class_="text-center p-3 border-top border-primary border-3 shadow-sm"
            ),
            col_widths=[6, 6]
        ),
        
        # Visual representation of the Classification Criteria
        ui.card(
            ui.card_header("Step 3: The Classification Criteria", class_="bg-dark text-white fw-bold"),
            ui.markdown("An observation is only classified as a deviation if it meets **all three** of the following thresholds:"),
            
            ui.layout_columns(
                ui.value_box(
                    "Rule 1: Sufficient History", 
                    "≥ 10", 
                    "Historical reference observations must exist", 
                    style="background-color: #808080; color: white;"
                ),
                ui.value_box(
                    "Rule 2: Absolute Threshold", 
                    "> 25", 
                    "Absolute difference between observed and expected counts", 
                    style="background-color: #b2182b; color: white;"
                ),
                ui.value_box(
                    "Rule 3: Relative Threshold", 
                    "> 75%", 
                    "Difference relative to the expected count", 
                    style="background-color: #fdae61; color: black;"
                ),
            ),
            class_="mt-3 shadow-sm"
        ),
        
        # The hand-off note to the other site
        ui.card(
            ui.markdown(
                """
                > 🚦 **Explore the Results:** Applying these criteria resulted in ~11% of traffic being classified as deviations. 
                > *To explore the resulting temporal, weather, and spatial patterns, please navigate to the **Deviations Analysis** site.*
                """
            ),
            class_="bg-light mt-3"
        ),

    ),
    
    # DEVIATIONS SITE
    ui.nav_panel("Deviation Analysis",
        ui.h2("Section 1: Deviations Overview and Sizing"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section provides an overview of detected traffic deviations.

                    A deviation represents a mismatch between:
                    * **Observed cyclist traffic**
                    * **Model-expected cyclist traffic**

                    Deviations are identified only when:
                    * the historical reference group has enough observations
                    * the absolute difference is sufficiently large
                    * the relative difference exceeds the predefined threshold

                    **What to explore:**
                    * Overall frequency of deviations
                    * Balance between higher- and lower-than-expected traffic
                    * Distribution of deviation magnitudes
                """)
            ),
            ui.navset_card_tab(
                ui.nav_panel("Summary Metrics",
                    ui.output_ui("dev_value_boxes")
                ),
                ui.nav_panel("Deviation Size", 
                    ui.output_plot("dev_size_distribution_plot")
                )
            )
        ),
        
        ui.hr(),
        
        ui.h2("Section 2: Temporal Dimensions"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section explores when deviations occur most frequently. It identifies recurring temporal disruption patterns such as rush-hour anomalies or weekend irregularities.

                    The heatmap highlights concentrations of abnormal traffic activity
                    across:
                    * days of the week
                    * hours of the day
                """)
            ),
            ui.navset_card_tab(
            ui.nav_panel(
                "Time Scale Distribution",

                ui.input_select(
                    "dev_time_scale",
                    "Select time scale:",
                    {
                        "month": "Month",
                        "weekday": "Weekday",
                        "hour_bin": "Hour bin"
                    },
                    selected="month",
                    width="300px"
                ),
                ui.br(),
                output_widget("dev_time_scale_plot")
            ),

            ui.nav_panel(
                "Hour × Weekday Heatmap",
                ui.output_plot("dev_temporal_heatmap_plot")
            ),

            ui.nav_panel(
                "Month × Weekday Heatmap",
                ui.output_plot("dev_month_weekday_heatmap_plot")
            )
        )
        ),

        ui.hr(),
        
        ui.h2("Section 3: Weather Effects"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section evaluates whether weather conditions are associated
                    with abnormal cycling behavior.

                    The analysis compares:
                    * the share of higher-than-expected traffic deviations
                    * the share of lower-than-expected traffic deviations
                    * the total number of detected deviations

                    This helps identify weather conditions linked to unusual traffic patterns.
                """)
            ),

            ui.card(

                ui.input_select(
                    "dev_weather_var",
                    "Select Weather Variable:",
                    choices={
                        "temperature_mean": "Temperature",
                        "precipitation_category": "Precipitation"
                    },
                    width="300px"
                ),
                output_widget("dev_weather_impact_plot")
            )
        ),
        
        ui.hr(),
        
        ui.h2("Section 4: Special Events Impact"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section analyzes systematic model deviations occurring during special events and holidays. 
                            
                    * **Deviation Frequency:** Compares how often the model registers significant prediction errors during event periods versus standard, non-event days.
                    * **Deviation Direction:** Breaks down the nature of the model's error—specifically.
                    *Note: Deviation rates for events with very few total observations are highly sensitive.*
                """)
            ),
            ui.navset_card_tab(
                ui.nav_panel(
                    "Event Impact (Volume)",
                    output_widget("dev_special_events_plot")
                ),
                ui.nav_panel(
                    "Deviation Direction",
                    output_widget("dev_event_direction_plot")
                ),
            )
        ),

        ui.hr(),
        
        ui.h2("Section 5: Spatial Patterns"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section investigates where deviations occur across Flanders.

                    **What to explore:**
                    * Geographic clustering of deviations
                    * Sites with the highest abnormal activity
                    * Municipalities contributing most strongly to deviation frequency

                    Larger and darker markers indicate locations with higher deviation intensity.
                """)
            ),
            ui.navset_card_tab(
                ui.nav_panel(
                    "Spatial Maps",
                    ui.input_select(
                        "dev_map_metric",
                        "Select map metric:",
                        choices={
                            "deviation_share": "Total deviation share",
                            "higher_share": "Higher than expected share",
                            "lower_share": "Lower than expected share"
                        },
                        width="300px"
                    ),
                    output_widget("dev_spatial_map_plot")
                ),
                ui.nav_panel(
                    "Top 25 Sites",
                    ui.output_plot("dev_top_sites_plot")
                ),
                ui.nav_panel(
                    "Site Characterization",
                    ui.card(
                        ui.input_select(
                            "map_view",
                            "Map view:",
                            {
                                "direction_profile": "Direction profile",
                                "site_category": "Site category",
                                "main_sensitivity_factor": "Main sensitivity factor",
                            },
                            selected="direction_profile",
                            width="300px"
                        ),
                        output_widget("dev_direction_profile_map")
                    )
                )
            )
        ),
    ),
    title="Cycling Traffic Analysis Dashboard"
)

# 2. SERVER LOGIC
def server(input, output, session):
    # EDA - Section 1
    @reactive.Calc
    def selected_eda_data():
        if input.s1_dataset() == "origin_data":
            return origin_data
        else:
            return prediction_data
    
    @render.ui
    def value_box_EDA():
        data = selected_eda_data()
        data['date'] = pd.to_datetime(data['date'])
        start_date = data['date'].min().strftime('%b %Y')
        end_date = data['date'].max().strftime('%b %Y')
        
        return ui.layout_column_wrap(
            ui.value_box("Records", f"{len(data):,}"),
            ui.value_box("Sites", f"{data['site_id'].nunique()}"),
            ui.value_box("Zero Share", f"{(data['count'] == 0).mean()*100:.1f}%"),
            ui.value_box("Date Range", f"{start_date} - {end_date}"),
            width=1/2,
        )
    
    @render.plot
    def directional_split_plot():
        data = selected_eda_data()
        split = data['direction'].value_counts(normalize=True) * 100
        
        # Default to 0 if a direction is completely missing in a specific dataset
        in_pct = split.get('IN', 0)
        out_pct = split.get('OUT', 0)
        
        fig, ax = plt.subplots(figsize=(8, 1))
        ax.barh([''], [in_pct], color=color_blue, label=f'In ({in_pct:.1f}%)')
        ax.barh([''], [out_pct], left=[in_pct], color=color_green, label=f'Out ({out_pct:.1f}%)')
        
        ax.set_xlim(0, 100)
        ax.axis('off')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
        fig.tight_layout()
        
        return fig
    
    @render.plot
    def traffic_distribution_plot():
        data = selected_eda_data()
        upper_limit = data['count_rescaled'].quantile(0.99)
        filtered_data = data.loc[data['count_rescaled'] <= upper_limit, 'count_rescaled']
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(filtered_data, bins=50, color=color_blue, edgecolor='white', linewidth=0.5)
        
        ax.set_title("Distribution of Rescaled Bicycle Counts (below 99th percentile)", pad=15)
        ax.set_xlabel("2-hour cyclist count")
        ax.set_ylabel("Frequency")
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        return fig
    
    @render.plot
    def log_count_distribution_plot():
        data = selected_eda_data()
        positive_counts = data[data['count_rescaled'] > 0]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(
            data=positive_counts,
            x='count_rescaled',
            bins=30,
            kde=True,
            color=color_green, 
            log_scale=True,
            ax=ax 
        )
        
        ax.set_title("Distribution of Positive Rescaled Bicycle Counts (log scale)", pad=15)
        ax.set_xlabel("Cyclist count (log scale)")
        ax.set_ylabel("Frequency")
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        return fig
    
    # EDA - Section 2
    @reactive.Calc
    def selected_temporal_data():
        if input.s2_dataset() == "origin_data":
            return origin_data
        else:
            return prediction_data

    @render.plot
    def temporal_dist_plot():
        data = selected_temporal_data()
        var_name = input.s2_time_var()
        metric = input.s2_metric()
        
        # 1. Define order
        if var_name == "hour_bin":
            order = list(sorted(data["hour_bin"].dropna().unique()))
            display_name = "Hour bin"
        elif var_name == "weekday":
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            display_name = "Day"
        else: 
            order = list(range(1, 13))
            display_name = "Month"
            
        # 2. Aggregation
        if metric == "n_rows":
            plot_df = data.groupby(var_name).size().reset_index(name="n")
            title = f"Distribution of Observation Frequency by {display_name}"
        else:
            plot_df = data.groupby(var_name)["count_rescaled"].sum().reset_index(name="n")
            title = f"Distribution of Total Cyclist Volume by {display_name}"

        # Reindex for order and handle missing data
        plot_df = plot_df.set_index(var_name).reindex(order).fillna(0).reset_index()
        plot_df["share"] = plot_df["n"] / plot_df["n"].sum()

        # 3. Plotting
        fig, ax = plt.subplots(figsize=(8, 4))
        
        ax.bar(plot_df[var_name].astype(str), plot_df["share"], color=color_blue)
        
        ax.set_title(title, pad=15)
        ax.set_ylabel("Share of Total")
        ax.set_xlabel(display_name)
        
        # Formatting Y-axis to percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    @render.plot
    def temporal_heatmap_plot():
        data = selected_temporal_data()
        view = input.s2_heatmap_var()
        
        # 1. Group by sum of count instead of size
        if view == "weekday":
            heatmap_data = (
                data.groupby(["weekday", "hour_bin"])["count"]
                .sum()
                .unstack(fill_value=0)
            )
            heatmap_data = heatmap_data.reindex(
                index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )
            figsize = (10, 4.8)
            title = "Hour × Day Cyclist Count Heatmap"
        else:
            heatmap_data = (
                data.groupby(["direction", "hour_bin"])["count"]
                .sum()
                .unstack(fill_value=0)
            )
            figsize = (10, 3.6)
            title = "Hour × Direction Cyclist Count Heatmap"
        
        # Common reindexing
        heatmap_data = heatmap_data.reindex(columns=sorted(heatmap_data.columns))
        
        # 2. Build the heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            heatmap_data, 
            cmap="Greens", 
            cbar_kws={"label": "Cyclist count"}, 
            ax=ax
        )
        
        ax.set_title(title, pad=15)
        ax.set_xlabel("Hour bin")
        ax.set_ylabel(view.replace("_", " ").title())
        
        fig.tight_layout()
        return fig
    
    # EDA - Section 3
    @reactive.Calc
    def selected_fuel_data():
        if input.s3_dataset() == "origin_data":
            return origin_data
        else:
            return prediction_data
        
    @render.plot
    def fuel_time_series_plot():
        df = selected_fuel_data()
        
        # 1. Data prep: Get daily level and drop missing values
        fuel_daily = (df[["date", "fuel_price_petrol_95"]].drop_duplicates().dropna().sort_values("date").reset_index(drop=True))
        
        fuel_daily["date"] = pd.to_datetime(fuel_daily["date"])
        
        # 2. Create the time series plot
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.lineplot(data=fuel_daily, x="date", y="fuel_price_petrol_95", color=color_green, ax=ax)
        
        ax.set_title("Fuel Price over Time (Model Development)", pad=15)
        ax.set_xlabel("Date")
        ax.set_ylabel("Fuel price petrol 95")
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        return fig
    
    # EDA - Section 4
    @render.plot
    def weather_impact_plot():
        df = prediction_data.copy()
        var = input.s4_weather_var()
        
        # 1. Prepare data
        if var == "temperature_mean":
            df['plot_var'] = df['temperature_mean'].round()
            x_label = "Temperature (°C)"
            title_text = "Temperature Impact: Cyclist Traffic vs. Data Volume"
        elif var == "wind_speed_mean":
            df['plot_var'] = df['wind_speed_mean'].round()
            x_label = "Wind Speed"
            title_text = "Wind Speed Impact: Cyclist Traffic vs. Data Volume"
        else: 
            df['plot_var'] = df['precipitation_category'].str.replace('_precipitation', '').str.title()
            x_label = "Weather Condition"
            title_text = "Precipitation Impact: Cyclist Traffic vs. Data Volume"
            
        # 2. GroupBy calculation
        summary = df.groupby('plot_var').agg(
            num_obs=('count_rescaled', 'count'),
            avg_traffic=('count_rescaled', 'mean')
        ).reset_index()
        
        # 3. Create dual-axis plot - Changed to landscape (10, 5)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx() 
        
        if var == "precipitation_category":
            order = ['Dry', 'Light', 'Moderate', 'Heavy', 'Snow']
            summary['plot_var'] = pd.Categorical(summary['plot_var'], categories=[p for p in order if p in summary['plot_var'].values], ordered=True)
            summary = summary.sort_values('plot_var')
            x_data = summary['plot_var'].astype(str)
            ax1.bar(x_data, summary['num_obs'], color=color_grey, alpha=0.8)
            ax2.plot(x_data, summary['avg_traffic'], color=color_blue, linewidth=3, marker='o', markersize=8)
        else:
            x_data = pd.to_numeric(summary['plot_var'])
            ax1.bar(x_data, summary['num_obs'], color=color_grey, alpha=0.8, width=0.8)
            ax2.plot(x_data, summary['avg_traffic'], color=color_blue, linewidth=3, marker='o', markersize=8)

        # Labels and Formatting
        ax1.set_xlabel(x_label, fontsize=10)
        ax1.set_ylabel("Number of Observations", color='dimgray', fontsize=10)
        ax2.set_ylabel("Average Cyclist Count", color=color_blue, fontsize=10, fontweight='bold')
        
        # Move the title to the axis object
        ax1.set_title(title_text, pad=20, fontsize=12)
        
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        
        fig.tight_layout()
        return fig
    
    # EDA - Section 5
    @reactive.calc
    def special_events_data():
        df = prediction_data.copy()
        
        event_cols = {
            'is_sport_event': 'Sports Event',
            'is_outdoor_music': 'Outdoor Music',
            'is_indoor_music': 'Indoor Music',
            'is_strike': 'Transport Strike'
        }
        
        valid_cols = {k: v for k, v in event_cols.items() if k in df.columns}
        
        baseline_mask = ~df[list(valid_cols.keys())].any(axis=1)
        normal_obs = baseline_mask.sum()
        baseline_avg = df[baseline_mask]['count_rescaled'].mean()
        total_obs = len(df)
        
        results = [{
            'Condition': 'Normal Day',
            'Total 2hr Intervals': normal_obs,
            '% of Total Year': (normal_obs / total_obs) * 100,
            'Impact (%)': 0.0, 
            'Confidence': 'High'
        }]
        
        for col, name in valid_cols.items():
            obs = df[col].sum()
            avg = df[df[col] == 1]['count_rescaled'].mean() if obs > 0 else 0
            lift = ((avg - baseline_avg) / baseline_avg) * 100 if (baseline_avg > 0 and obs > 0) else 0
            pct_year = (obs / total_obs) * 100
            
            results.append({
                'Condition': name,
                'Total 2hr Intervals': obs,
                '% of Total Year': pct_year,
                'Impact (%)': lift,
                'Confidence': 'Low (<1%)' if pct_year < 1.0 else 'High'
            })
            
        return pd.DataFrame(results)

    @render.data_frame
    def events_frequency_table():
        df = special_events_data()
        table_df = df[['Condition', 'Total 2hr Intervals', '% of Total Year']].copy()
        table_df['% of Total Year'] = table_df['% of Total Year'].map("{:.2f}%".format)
        
        # Sort from most frequent to least frequent
        table_df = table_df.sort_values('Total 2hr Intervals', ascending=False)
        
        return render.DataGrid(table_df, selection_mode="none", width="100%")

    @render_widget
    def events_impact_plot():
        df = special_events_data()
        plot_df = df[df['Condition'] != 'Normal Day'].copy()
        plot_df = plot_df.sort_values('Impact (%)')
        
        # Using global colors!
        colors = [color_red if val < 0 else color_green for val in plot_df['Impact (%)']]
        opacities = [0.4 if conf == 'Low (<1%)' else 1.0 for conf in plot_df['Confidence']]
        patterns = ['/' if conf == 'Low (<1%)' else '' for conf in plot_df['Confidence']]
        
        fig = go.Figure(go.Bar(
            x=plot_df['Impact (%)'],
            y=plot_df['Condition'],
            orientation='h',
            marker=dict(
                color=colors,
                opacity=opacities,
                pattern=dict(shape=patterns),
                line=dict(color='gray', width=[1 if p == '/' else 0 for p in patterns])
            ),
            hovertext=[f"{val:+.2f}%" for val in plot_df['Impact (%)']],
            customdata=plot_df[['Total 2hr Intervals', '% of Total Year']],
            hovertemplate="<b>%{y}</b><br>Impact vs Normal: %{hovertext}<br>Observations: %{customdata[0]} (%{customdata[1]:.2f}% of data)<extra></extra>"
        ))
        
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        
        fig.update_layout(
            title="Relative Difference of Special Events vs. Normal Day",
            xaxis_title="Percentage Change vs. Normal Day (%)",
            yaxis_title="",
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            annotations=[dict(
                x=0.5, y=-0.2, xref='paper', yref='paper',
                text="* Striped bars indicate low statistical confidence (< 1% of total data).",
                showarrow=False, font=dict(color="gray", size=11)
            )]
        )
        return fig

    ###################################################################################

    # Model Site
    # 1. Timeline Visualization
    @render_widget
    def train_test_timeline_plot():
        # Mock data for the timeline - replace dates with your actual cutoffs based on the report
        df_split = pd.DataFrame([
            {"Phase": "Training Set", "Start": "2022-05-01", "End": "2024-04-30"},
            {"Phase": "Validation Set", "Start": "2024-05-01", "End": "2025-04-30"},
            {"Phase": "Prediction Set", "Start": "2025-05-01", "End": "2026-04-30"}
        ])
        
        fig = px.timeline(
            df_split, 
            x_start="Start", 
            x_end="End", 
            y="Phase", 
            color="Phase",
            color_discrete_sequence=[color_blue, color_orange, color_red], 
            height=200
        )
        fig.update_yaxes(autorange="reversed", title="")
        fig.update_layout(
            showlegend=False, 
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    # 2. Dynamic Formula & Methodology Renderer
    @render.ui
    def dynamic_model_formula():
        choice = input.model_choice()
        
        if choice == "nb":
            # Using pure HTML to guarantee a perfect multi-line equation render without LaTeX errors
            return ui.HTML(
                """
                <div style="margin-bottom: 15px;">
                    <strong>Negative Binomial Model (Final Choice):</strong><br>
                    Retained because it successfully handles overdispersion and accurately preserves the distributional structure of the counts (including the ~25% zero-share), which is strictly required for baseline deviation detection.
                </div>
                
                <blockquote style="border-left: 4px solid #ccc; padding-left: 15px; color: #444; font-size: 0.95em; line-height: 1.8; margin-left: 0;">
                    <strong>log(Expected Total Count)</strong> = <br>
                    &nbsp;&nbsp;&beta;<sub>0</sub> + &beta;<sub>1</sub>(Hour) + &beta;<sub>2</sub>(Weekday) + &beta;<sub>3</sub>(Month) <br>
                    &nbsp;&nbsp;+ &beta;<sub>4</sub>(PublicHoliday) + &beta;<sub>5</sub>(SchoolHoliday) + &beta;<sub>6</sub>(Site) <br>
                    &nbsp;&nbsp;+ &beta;<sub>7</sub>(Direction) + &beta;<sub>8</sub>(FuelPrice) <br>
                    &nbsp;&nbsp;+ &beta;<sub>9</sub>(Hour &times; Weekday) + &beta;<sub>10</sub>(Hour &times; Direction) <br>
                    &nbsp;&nbsp;+ log(Number of Periods)
                </blockquote>
                """
            )
        elif choice == "poisson":
            return ui.markdown(
                """
                **Poisson Regression:**
                *Rejected.* It strongly violates the mean-variance assumption (Pearson dispersion ~52.6) and severely underpredicts the share of zero-traffic periods (predicting ~8.2% vs. the observed ~28.3%).
                """
            )
        else:
            return ui.markdown(
                """
                **Histogram Gradient Boosting:**
                *Rejected.* While it yielded lower point-wise error metrics (MAE/RMSE), it systematically underpredicted average counts and strictly failed to capture zero-traffic periods (predicting ~6.1%), making it unsuitable as a regular structure baseline.
                """
            )

    # 3. Model Results Table
    @render.data_frame
    def model_metrics_table():
        # Inserted metrics directly from Table 1 of your methodology report
        metrics_df = pd.DataFrame({
            "Model": ["Negative Binomial", "Histogram Gradient Boosting"],
            "MAE": [16.5161, 12.3838],
            "RMSE": [39.8127, 33.6298],
            "Poisson Deviance": [21.8443, 14.9244],
            "Observed Mean Count": [22.5366, 22.5366],
            "Predicted Mean Count": [22.2970, 18.8975]
        })
        
        return render.DataGrid(metrics_df, selection_mode="none", filters=False)

    ###################################################################################

    # Deviations - Section 1
    @reactive.Calc
    def deviation_data():
        start_date = pd.to_datetime('2025-05-01')

        reference_variables = [
            "site_id",
            "direction",
            "month",
            "weekday",
            "hour_bin",
        ]

        reference_counts = (
            origin_data[origin_data['date'] < start_date]
            .groupby(reference_variables)
            .size()
            .reset_index(name="reference_n")
        )

        df = prediction_data.merge(
            reference_counts,
            on=reference_variables,
            how="left"
        )

        df["reference_n"] = df["reference_n"].fillna(0).astype(int)

        df["difference"] = (
            df["count_rescaled"] - df["expected_count"]
        )

        df["relative_difference"] = (
            df["difference"] / df["expected_count"]
        )

        df["is_deviation"] = (
            (df["reference_n"] >= 10) &
            (df["difference"].abs() > 25) &
            (df["relative_difference"].abs() > 0.75)
        ).astype(int)

        df["deviation_direction"] = "No deviation"

        df.loc[
            (df["is_deviation"] == 1) &
            (df["difference"] > 0),
            "deviation_direction"
        ] = "Higher than expected"

        df.loc[
            (df["is_deviation"] == 1) &
            (df["difference"] < 0),
            "deviation_direction"
        ] = "Lower than expected"

        return df
    
    def summarize_deviations(data, group_variable):
        baseline_share = data["is_deviation"].mean()

        summary = (
            data
            .groupby(group_variable)
            .agg(
                observations=("is_deviation", "size"),
                deviations=("is_deviation", "sum")
            )
            .reset_index()
        )

        summary["deviation_share"] = summary["deviations"] / summary["observations"]
        summary["baseline_deviation_share"] = baseline_share

        return summary

    def summarize_directional_deviations(data, group_variable):
        baseline_higher_share = (
            data["deviation_direction"] == "Higher than expected"
        ).mean()

        baseline_lower_share = (
            data["deviation_direction"] == "Lower than expected"
        ).mean()

        summary = (
            data
            .groupby(group_variable)
            .agg(
                observations=("is_deviation", "size"),
                deviations=("is_deviation", "sum"),
                higher_deviations=(
                    "deviation_direction",
                    lambda x: (x == "Higher than expected").sum()
                ),
                lower_deviations=(
                    "deviation_direction",
                    lambda x: (x == "Lower than expected").sum()
                )
            )
            .reset_index()
        )

        summary["deviation_share"] = summary["deviations"] / summary["observations"]
        summary["higher_deviation_share"] = summary["higher_deviations"] / summary["observations"]
        summary["lower_deviation_share"] = summary["lower_deviations"] / summary["observations"]
        summary["baseline_higher_deviation_share"] = baseline_higher_share
        summary["baseline_lower_deviation_share"] = baseline_lower_share

        return summary
    
    @render.ui
    def dev_value_boxes():
        df = deviation_data()

        total_observations = len(df)
        total_deviations = (df["is_deviation"] == 1).sum()
        deviation_share = total_deviations / total_observations
        
        higher_share = (df["deviation_direction"] == "Higher than expected").mean()
        lower_share = (df["deviation_direction"] == "Lower than expected").mean()
        
        # Calculate Deviation Range
        dev_rows = df[df["is_deviation"] == 1]
        min_dev = dev_rows["difference"].min()
        max_dev = dev_rows["difference"].max()
        range_str = f"{min_dev:,.0f} to {max_dev:,.0f}"

        return ui.layout_column_wrap(
            ui.value_box("Total observations", f"{total_observations:,}"),
            ui.value_box("Detected deviations", f"{total_deviations:,}"),
            ui.value_box("Deviation share", f"{deviation_share:.1%}"),
            ui.value_box("Higher-than-expected share", f"{higher_share:.1%}"),
            ui.value_box("Lower-than-expected share", f"{lower_share:.1%}"),
            ui.value_box("Deviation range", range_str),
            width=1/3
        )
    
    @render.plot
    def dev_size_distribution_plot():

        df = deviation_data()

        deviation_rows = df["is_deviation"] == 1

        deviation_differences = df.loc[
            deviation_rows,
            "difference"
        ]

        lower_limit = deviation_differences.quantile(0.01)
        upper_limit = deviation_differences.quantile(0.99)

        deviation_differences_capped = deviation_differences[
            (deviation_differences >= lower_limit) &
            (deviation_differences <= upper_limit)
        ]

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.hist(
            deviation_differences_capped,
            bins=80,
            color=color_red,
            edgecolor="darkred",
            alpha=0.90
        )

        ax.axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=1
        )

        ax.set_xlabel("Observed count minus expected count")
        ax.set_ylabel("Number of deviations")
        ax.set_title("Distribution of deviation size, restricted to 1st and 99th percentiles")

        plt.tight_layout()

        return fig

    # Deviation - Section 2
    @render_widget
    def dev_time_scale_plot():
        df = deviation_data()
        time_scale = input.dev_time_scale()
        
        # 1. Summarize
        summary = summarize_directional_deviations(df, time_scale)
        
        # 2. Reorder
        if time_scale == "weekday":
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            summary[time_scale] = pd.Categorical(summary[time_scale], categories=weekday_order, ordered=True)
        summary = summary.sort_values(time_scale)

        # 3. Prepare data
        plot_data = summary[[time_scale, "higher_deviation_share", "lower_deviation_share"]].melt(
            id_vars=time_scale,
            value_vars=["higher_deviation_share", "lower_deviation_share"],
            var_name="Direction", 
            value_name="Share"
        )
        plot_data["Direction"] = plot_data["Direction"].replace({
            "higher_deviation_share": "Higher than expected",
            "lower_deviation_share": "Lower than expected"
        })

        # 4. Color Mapping
        direction_colors = {
            "Higher than expected": color_red,
            "Lower than expected": color_orange
        }

        fig = px.bar(
            plot_data,
            x=time_scale,
            y="Share",
            color="Direction",
            color_discrete_map=direction_colors,
            barmode="group",
            title=f"Directional deviation shares by {time_scale}",
            labels={time_scale: time_scale.replace("_", " ").title()}
        )

        # 5. Add baselines with external annotations at the end of the line
        higher_base = summary["baseline_higher_deviation_share"].iloc[0]
        lower_base = summary["baseline_lower_deviation_share"].iloc[0]

        fig.add_hline(y=higher_base, line_dash="dot", line_color=color_red)
        fig.add_hline(y=lower_base, line_dash="dot", line_color=color_orange)
        
        # Annotations positioned at the far right (x=1.0)
        fig.add_annotation(
            xref="paper", x=1.0,  
            y=higher_base, text=f"Baseline = {higher_base:.1%}", 
            showarrow=False, font=dict(color=color_red, size=10), 
            xanchor="left", yshift=5
        )
        fig.add_annotation(
            xref="paper", x=1.0, 
            y=lower_base, text=f"Baseline = {lower_base:.1%}", 
            showarrow=False, font=dict(color=color_orange, size=10), 
            xanchor="left", yshift=5
        )

        fig.update_layout(
            plot_bgcolor="white", 
            paper_bgcolor="white",
            yaxis_tickformat=".1%",
            legend_title_text="Direction"
        )
        
        # Clean up hover box labels
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Direction: %{fullData.name}<br>Share: %{y:.1%}<extra></extra>"
        )
        
        return fig

    @render.plot
    def dev_temporal_heatmap_plot():
        df = deviation_data()
        summary = df.groupby(["weekday", "hour_bin"]).agg(
            observations=("is_deviation", "size"),
            deviations=("is_deviation", "sum")
        ).reset_index()
        summary["deviation_share"] = summary["deviations"] / summary["observations"]

        heatmap_data = summary.pivot(index="weekday", columns="hour_bin", values="deviation_share")
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heatmap_data = heatmap_data.reindex(weekday_order)

        fig, ax = plt.subplots(figsize=(10, 5))
        
        im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd") 

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Deviation share")

        ax.set_xticks(range(len(heatmap_data.columns))); ax.set_xticklabels(heatmap_data.columns)
        ax.set_yticks(range(len(heatmap_data.index))); ax.set_yticklabels(heatmap_data.index)
        
        ax.set_xticks(np.arange(-0.5, len(heatmap_data.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(heatmap_data.index), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_xlabel("Start of 2-hour interval")
        ax.set_ylabel("Weekday")
        ax.set_title("Deviation share by weekday and time of day", pad=15)
        
        plt.tight_layout()
        return fig
    
    @render.plot
    def dev_month_weekday_heatmap_plot():
        df = deviation_data()
        summary = df.groupby(["month", "weekday"]).agg(
            observations=("is_deviation", "size"),
            deviations=("is_deviation", "sum")
        ).reset_index()
        summary["deviation_share"] = summary["deviations"] / summary["observations"]

        heatmap_data = summary.pivot(index="month", columns="weekday", values="deviation_share")
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heatmap_data = heatmap_data.reindex(columns=weekday_order)

        fig, ax = plt.subplots(figsize=(10, 5))
        
        im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Deviation share")

        ax.set_xticks(range(len(heatmap_data.columns))); ax.set_xticklabels(heatmap_data.columns)
        ax.set_yticks(range(len(heatmap_data.index))); ax.set_yticklabels(heatmap_data.index)
        
        ax.set_xticks(np.arange(-0.5, len(heatmap_data.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(heatmap_data.index), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_xlabel("Weekday")
        ax.set_ylabel("Month")
        ax.set_title("Deviation share by month and weekday", pad=15)

        plt.tight_layout()
        return fig
    
    # Deviation - Section 3
    @render_widget
    def dev_weather_impact_plot():
        df = deviation_data().copy()
        var = input.dev_weather_var()

        # 1. Data Prep
        if var == "temperature_mean":
            bins = [-10, 0, 5, 10, 15, 20, 25, 35, 40]
            labels = ["(-10, 0]", "(0, 5]", "(5, 10]", "(10, 15]", "(15, 20]", "(20, 25]", "(25, 35]", "(35, 40]"]
            df["plot_var"] = pd.cut(df["temperature_mean"], bins=bins, labels=labels)
            x_label = "Temperature Bin (°C)"
        else:
            df["plot_var"] = df["precipitation_category"].str.replace("_precipitation", "").str.title()
            x_label = "Precipitation"

        summary = (
            df.groupby("plot_var")
            .agg(
                observations=("is_deviation", "size"),
                higher_share=("deviation_direction", lambda x: (x == "Higher than expected").mean()),
                lower_share=("deviation_direction", lambda x: (x == "Lower than expected").mean())
            )
            .reset_index()
        )
        summary["total_share"] = summary["higher_share"] + summary["lower_share"]

        # 2. Setup Plotly Dual-Axis Figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=summary["plot_var"],
                y=summary["higher_share"],
                name="Traffic Higher Than Expected",
                marker_color=color_red
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Bar(
                x=summary["plot_var"],
                y=summary["lower_share"],
                name="Traffic Lower Than Expected",
                marker_color=color_orange,
                text=[f"{val:.1%}" for val in summary["total_share"]],
                textposition="outside", 
                textfont=dict(size=11, color="black")
            ),
            secondary_y=False,
        )

        # Add Observation Line (Secondary Axis)
        fig.add_trace(
            go.Scatter(
                x=summary["plot_var"],
                y=summary["observations"],
                name="Data Volume (Observations)",
                mode="lines+markers",
                line=dict(color=color_grey, dash="dash", width=2),
                marker=dict(size=8)
            ),
            secondary_y=True,
        )

        # 3. Layout & Styling
        fig.update_layout(
            barmode="stack",
            title=dict(text="Model Error: Deviation Rate and Direction", font=dict(size=18)),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=80, b=20, l=20, r=20),
            hovermode="x unified", 
            hoverlabel=dict(namelength=-1),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )

        # 4. Axis Formatting
        fig.update_xaxes(title_text=x_label, showgrid=False)
        fig.update_yaxes(title_text="Deviation Rate", tickformat=".1%", gridcolor="#e0e0e0", secondary_y=False)
        fig.update_yaxes(title_text="Number of Observations", showgrid=False, secondary_y=True)

        return fig

    # Deviation - Section 4
    def summarize_event_deviations(data):
        event_factors = ["is_strike", "is_outdoor_music", "is_indoor_music", "is_sport_event"]
        event_factor_labels = {
            "is_strike": "Transport strike",
            "is_outdoor_music": "Outdoor music event",
            "is_indoor_music": "Indoor music event",
            "is_sport_event": "Sport event",
        }
        
        summary_list = []
        for factor in event_factors:
            non_event_data = data[data[factor] == 0]
            event_data = data[data[factor] == 1]
            
            summary_list.append({
                "factor_label": event_factor_labels[factor],
                "non_event_deviation_share": non_event_data["is_deviation"].mean(),
                "event_deviation_share": event_data["is_deviation"].mean(),
                "event_higher_share": (event_data["deviation_direction"] == "Higher than expected").mean(),
                "event_lower_share": (event_data["deviation_direction"] == "Lower than expected").mean(),
            })
            
        return pd.DataFrame(summary_list)
    
    @render_widget
    def dev_special_events_plot():
        df = deviation_data()
        summary = summarize_event_deviations(df)
        
        plot_data = summary.melt(
            id_vars=["factor_label"],
            value_vars=["non_event_deviation_share", "event_deviation_share"],
            var_name="period",
            value_name="deviation_share",
        )
        plot_data["period"] = plot_data["period"].map({
            "non_event_deviation_share": "Non-event period",
            "event_deviation_share": "Event period",
        })

        period_colors = {
            "Non-event period": color_grey, 
            "Event period": color_red    
        }

        fig = px.bar(
            plot_data,
            x="factor_label",
            y="deviation_share",
            color="period",
            color_discrete_map=period_colors,
            barmode="group",
            title="Deviation share: Events vs. Non-events",
            labels={"factor_label": "Event factor"}
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=100, b=20, l=20, r=20),
            font=dict(color="#333333", size=13),
            title=dict(x=0.02, font=dict(size=18)),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
            legend_title_text="", 
            yaxis_tickformat=".1%",
            bargap=0.3,
        )

        fig.update_yaxes(gridcolor="#e0e0e0", zerolinecolor="#e0e0e0", title="Deviation share")
        fig.update_xaxes(showgrid=False)

        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Period: %{fullData.name}<br>Share: %{y:.1%}<extra></extra>"
        )

        return fig
    
    @render_widget
    def dev_event_direction_plot():
        df = deviation_data()
        
        event_factors = ["is_strike", "is_outdoor_music", "is_indoor_music", "is_sport_event"]
        event_factor_labels = {
            "is_strike": "Transport strike",
            "is_outdoor_music": "Outdoor music event",
            "is_indoor_music": "Indoor music event",
            "is_sport_event": "Sport event",
        }
        
        # 1. Calculate global baselines 
        baseline_higher = (df["deviation_direction"] == "Higher than expected").mean()
        baseline_lower = (df["deviation_direction"] == "Lower than expected").mean()
        
        # 2. Summarize specific event data
        summary_list = []
        for factor in event_factors:
            event_data = df[df[factor] == 1]
            
            # Prevent division by zero if an event has no observations
            if len(event_data) > 0:
                event_higher = (event_data["deviation_direction"] == "Higher than expected").mean()
                event_lower = (event_data["deviation_direction"] == "Lower than expected").mean()
            else:
                event_higher, event_lower = 0, 0
                
            summary_list.append({
                "factor_label": event_factor_labels[factor],
                "higher_deviation_share": event_higher,
                "lower_deviation_share": event_lower,
                "baseline_higher_deviation_share": baseline_higher,
                "baseline_lower_deviation_share": baseline_lower
            })
            
        summary = pd.DataFrame(summary_list)

        # 3. Prepare data for plotting
        plot_data = summary[["factor_label", "higher_deviation_share", "lower_deviation_share"]].melt(
            id_vars="factor_label",
            value_vars=["higher_deviation_share", "lower_deviation_share"],
            var_name="Direction", 
            value_name="Share"
        )
        plot_data["Direction"] = plot_data["Direction"].replace({
            "higher_deviation_share": "Higher than expected",
            "lower_deviation_share": "Lower than expected"
        })

        # 4. Color Mapping:
        direction_colors = {
            "Higher than expected": color_red,
            "Lower than expected": color_orange
        }

        fig = px.bar(
            plot_data,
            x="factor_label",
            y="Share",
            color="Direction",
            color_discrete_map=direction_colors,
            barmode="group",
            title="Directional deviation shares during events",
            labels={"factor_label": "Event factor"}
        )

        # 5. Add baselines with external annotations
        higher_base = summary["baseline_higher_deviation_share"].iloc[0]
        lower_base = summary["baseline_lower_deviation_share"].iloc[0]

        fig.add_hline(y=higher_base, line_dash="dot", line_color=color_red)
        fig.add_hline(y=lower_base, line_dash="dot", line_color=color_orange)
        
        # Annotations positioned at the far right (x=1.0)
        fig.add_annotation(
            xref="paper", x=1.0,  
            y=higher_base, text=f"Baseline = {higher_base:.1%}", 
            showarrow=False, font=dict(color=color_red, size=10), 
            xanchor="left", yshift=5
        )
        fig.add_annotation(
            xref="paper", x=1.0, 
            y=lower_base, text=f"Baseline = {lower_base:.1%}", 
            showarrow=False, font=dict(color=color_orange, size=10), 
            xanchor="left", yshift=5
        )

        # 6. Formatting matching the time scale plot
        fig.update_layout(
            plot_bgcolor="white", 
            paper_bgcolor="white",
            margin=dict(t=100, b=20, l=20, r=80),
            font=dict(color="#333333", size=13),
            title=dict(x=0.02, font=dict(size=18)),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0, title_text=""),
            yaxis_tickformat=".1%",
            bargap=0.3
        )
        
        fig.update_yaxes(gridcolor="#e0e0e0", zerolinecolor="#e0e0e0", title="Share of event observations")
        fig.update_xaxes(showgrid=False)
        
        # Clean up hover box labels
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Direction: %{fullData.name}<br>Share: %{y:.1%}<extra></extra>"
        )
        
        return fig
    
    # Deviation - Section 5
    @render_widget
    def dev_spatial_map_plot():

        df = deviation_data()

        site_map = (
            df
            .groupby(
                [
                    "site_id",
                    "site_name",
                    "municipality",
                    "latitude",
                    "longitude"
                ],
                dropna=False,
                observed=True
            )
            .agg(
                observations=("is_deviation", "size"),
                deviations=("is_deviation", "sum"),
                higher_deviations=(
                    "deviation_direction",
                    lambda x: (x == "Higher than expected").sum()
                ),
                lower_deviations=(
                    "deviation_direction",
                    lambda x: (x == "Lower than expected").sum()
                )
            )
            .reset_index()
        )

        site_map = site_map.dropna(
            subset=["latitude", "longitude"]
        )

        site_map["deviation_share"] = (
            site_map["deviations"] / site_map["observations"]
        )

        site_map["higher_share"] = (
            site_map["higher_deviations"] / site_map["observations"]
        )

        site_map["lower_share"] = (
            site_map["lower_deviations"] / site_map["observations"]
        )

        metric = input.dev_map_metric()

        metric_settings = {
            "deviation_share": {
                "color": "deviation_share",
                "size": "deviations",
                "title": "Total deviation share by counting site",
                "legend": "Deviation share"
            },
            "higher_share": {
                "color": "higher_share",
                "size": "higher_deviations",
                "title": "Higher-than-expected deviation share by counting site",
                "legend": "Higher share"
            },
            "lower_share": {
                "color": "lower_share",
                "size": "lower_deviations",
                "title": "Lower-than-expected deviation share by counting site",
                "legend": "Lower share"
            }
        }

        selected = metric_settings[metric]

        site_map["site_label"] = (
            site_map["site_id"].astype(str)
            + " - "
            + site_map["site_name"].astype(str).str.replace("_", " ", regex=False).str.title()
            + " ("
            + site_map["municipality"].astype(str).str.title()
            + ")"
        )

        site_map["deviation_share_label"] = (
            (site_map["deviation_share"] * 100).round(1).astype(str) + "%"
        )

        site_map["higher_share_label"] = (
            (site_map["higher_share"] * 100).round(1).astype(str) + "%"
        )

        site_map["lower_share_label"] = (
            (site_map["lower_share"] * 100).round(1).astype(str) + "%"
        )

        fig = px.scatter_mapbox(
            site_map,
            lat="latitude",
            lon="longitude",
            color=selected["color"],
            size=selected["size"],
            color_continuous_scale="YlOrRd",
            size_max=24,
            zoom=7,
            height=700,
            hover_name="site_label",
            custom_data=[
                "observations",
                "deviations",
                "higher_deviations",
                "lower_deviations",
                "deviation_share_label",
                "higher_share_label",
                "lower_share_label",
            ],
            title=selected["title"]
        )

        fig.update_traces(
            hovertemplate=
            "<b>%{hovertext}</b><br><br>"
            + "Observations: %{customdata[0]:,}<br>"
            + "Detected deviations: %{customdata[1]:,}<br>"
            + "Higher-than-expected deviations: %{customdata[2]:,}<br>"
            + "Lower-than-expected deviations: %{customdata[3]:,}<br>"
            + "Deviation share: %{customdata[4]}<br>"
            + "Higher-than-expected share: %{customdata[5]}<br>"
            + "Lower-than-expected share: %{customdata[6]}"
            + "<extra></extra>"
        )

        fig.update_layout(
            mapbox_style="carto-positron",
            paper_bgcolor="#f2f2f2",
            plot_bgcolor="#f2f2f2",
            font=dict(color="#333333"),
            title=dict(x=0.02),
            margin={"r": 0, "t": 55, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                title=selected["legend"]
            )
        )
        return fig
    
    @render.plot
    def dev_top_sites_plot():

        df = deviation_data().copy()

        site_summary = (
            df.groupby(["site_id", "site_name", "municipality"])
            .agg(
                observations=("is_deviation", "size"),
                deviations=("is_deviation", "sum")
            )
            .reset_index()
        )

        site_summary["deviation_share"] = (
            site_summary["deviations"] / site_summary["observations"]
        )

        site_summary["site_label"] = (
            site_summary["site_name"].astype(str)
            .str.replace("_", " ", regex=False)
            .str.title()
            + " ("
            + site_summary["municipality"].astype(str).str.title()
            + ")"
        )

        top_sites = (
            site_summary
            .sort_values("deviations", ascending=False)
            .head(25)
            .sort_values("deviations", ascending=True)
        )

        total_bars = len(top_sites)
        bar_colors = [
            color_grey if i < (total_bars - 5) else color_red 
            for i in range(total_bars)
        ]

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.barh(
            top_sites["site_label"],
            top_sites["deviations"],
            color=bar_colors,
            alpha=0.90
        )

        ax.set_xlabel("Number of detected deviations")
        ax.set_ylabel("")
        ax.set_title("Top 25 counting sites by number of deviations", pad=15)

        plt.tight_layout()

        return fig
    
    @reactive.Calc
    def site_characterisation_data():
        df = deviation_data().copy()

        site_characterisation = (
            df.groupby(["site_id", "site_name", "municipality", "latitude", "longitude"], observed=True)
            .agg(
                observations=("is_deviation", "size"),
                deviations=("is_deviation", "sum"),
                higher_deviations=("deviation_direction", lambda x: (x == "Higher than expected").sum()),
                lower_deviations=("deviation_direction", lambda x: (x == "Lower than expected").sum()),
            )
            .reset_index()
        )

        site_characterisation["deviation_share"] = site_characterisation["deviations"] / site_characterisation["observations"]
        site_characterisation["higher_deviation_share"] = site_characterisation["higher_deviations"] / site_characterisation["observations"]
        site_characterisation["lower_deviation_share"] = site_characterisation["lower_deviations"] / site_characterisation["observations"]

        site_characterisation["direction_profile"] = "Low deviation frequency"
        high = site_characterisation["deviation_share"] >= 0.10

        site_characterisation.loc[
            high & (site_characterisation["higher_deviation_share"] > 1.5 * site_characterisation["lower_deviation_share"]),
            "direction_profile"
        ] = "Mostly higher counts than expected"

        site_characterisation.loc[
            high & (site_characterisation["lower_deviation_share"] > 1.5 * site_characterisation["higher_deviation_share"]),
            "direction_profile"
        ] = "Mostly lower counts than expected"

        site_characterisation.loc[
            high
            & (site_characterisation["higher_deviation_share"] <= 1.5 * site_characterisation["lower_deviation_share"])
            & (site_characterisation["lower_deviation_share"] <= 1.5 * site_characterisation["higher_deviation_share"]),
            "direction_profile"
        ] = "Mixed direction of deviations"

        df["is_cultural_event"] = ((df["is_outdoor_music"] == 1) | (df["is_indoor_music"] == 1)).astype(int)

        df["meaningful_precipitation"] = df["precipitation_category"].isin([
            "moderate_precipitation",
            "heavy_precipitation",
            "snow",
        ]).astype(int)

        cold_threshold = df["temperature_mean"].quantile(0.05)
        warm_threshold = df["temperature_mean"].quantile(0.95)

        df["cold_weather"] = (df["temperature_mean"] <= cold_threshold).astype(int)
        df["warm_weather"] = (df["temperature_mean"] >= warm_threshold).astype(int)

        df["normal_external_conditions"] = (
            (df["is_strike"] == 0)
            & (df["is_cultural_event"] == 0)
            & (df["is_sport_event"] == 0)
            & (df["meaningful_precipitation"] == 0)
            & (df["cold_weather"] == 0)
            & (df["warm_weather"] == 0)
        ).astype(int)

        normal_site_summary = (
            df[df["normal_external_conditions"] == 1]
            .groupby("site_id")
            .agg(
                normal_observations=("is_deviation", "size"),
                normal_deviations=("is_deviation", "sum"),
            )
            .reset_index()
        )

        normal_site_summary["normal_deviation_share"] = (
            normal_site_summary["normal_deviations"] / normal_site_summary["normal_observations"]
        )

        site_characterisation = site_characterisation.merge(
            normal_site_summary,
            on="site_id",
            how="left"
        )

        site_characterisation["normal_deviation_share"] = site_characterisation["normal_deviation_share"].fillna(0)

        factor_columns = [
            "meaningful_precipitation",
            "cold_weather",
            "warm_weather",
            "is_strike",
            "is_cultural_event",
            "is_sport_event",
        ]

        factor_labels = {
            "meaningful_precipitation": "Precipitation",
            "cold_weather": "Cold weather",
            "warm_weather": "Warm weather",
            "is_strike": "Transport strike",
            "is_cultural_event": "Cultural event",
            "is_sport_event": "Sport event",
        }

        factor_summary_list = []

        for factor_column in factor_columns:
            temp = (
                df[df[factor_column] == 1]
                .groupby("site_id")
                .agg(
                    factor_observations=("is_deviation", "size"),
                    factor_deviations=("is_deviation", "sum"),
                )
                .reset_index()
            )

            temp["factor"] = factor_column
            temp["factor_label"] = factor_labels[factor_column]
            temp["factor_deviation_share"] = temp["factor_deviations"] / temp["factor_observations"]

            factor_summary_list.append(temp)

        site_factor_sensitivity = pd.concat(factor_summary_list, ignore_index=True)

        site_factor_sensitivity = site_factor_sensitivity.merge(
            site_characterisation[
                ["site_id", "site_name", "municipality", "deviation_share", "normal_deviation_share"]
            ],
            on="site_id",
            how="left"
        )

        site_factor_sensitivity["lift_from_normal"] = (
            site_factor_sensitivity["factor_deviation_share"]
            - site_factor_sensitivity["normal_deviation_share"]
        )

        site_factor_sensitivity["is_sensitive"] = (
            (site_factor_sensitivity["deviation_share"] >= 0.10)
            & (site_factor_sensitivity["lift_from_normal"] >= 0.05)
        ).astype(int)

        sensitive_only = site_factor_sensitivity[site_factor_sensitivity["is_sensitive"] == 1].copy()

        number_of_sensitivities = (
            sensitive_only.groupby("site_id")
            .size()
            .reset_index(name="number_of_sensitivities")
        )

        sensitive_factor_list = (
            sensitive_only
            .sort_values(["site_id", "lift_from_normal"], ascending=[True, False])
            .groupby("site_id")["factor_label"]
            .apply(lambda x: ", ".join(x))
            .reset_index(name="sensitive_factors")
        )

        main_sensitivity_factor = (
            sensitive_only
            .sort_values(["site_id", "lift_from_normal"], ascending=[True, False])
            .groupby("site_id")
            .first()
            .reset_index()[["site_id", "factor_label", "lift_from_normal"]]
            .rename(columns={
                "factor_label": "main_sensitivity_factor",
                "lift_from_normal": "main_sensitivity_lift",
            })
        )

        site_sensitivity_summary = (
            number_of_sensitivities
            .merge(sensitive_factor_list, on="site_id", how="left")
            .merge(main_sensitivity_factor, on="site_id", how="left")
        )

        site_characterisation = site_characterisation.merge(
            site_sensitivity_summary,
            on="site_id",
            how="left"
        )

        site_characterisation["number_of_sensitivities"] = site_characterisation["number_of_sensitivities"].fillna(0).astype(int)
        site_characterisation["sensitive_factors"] = site_characterisation["sensitive_factors"].fillna("No sensitivities identified")
        site_characterisation["main_sensitivity_factor"] = site_characterisation["main_sensitivity_factor"].fillna("No main sensitivity factor")
        site_characterisation["main_sensitivity_lift"] = site_characterisation["main_sensitivity_lift"].fillna(0)

        site_characterisation["site_category"] = "Stable"

        site_characterisation.loc[
            (site_characterisation["deviation_share"] >= 0.10)
            & (site_characterisation["number_of_sensitivities"] == 0),
            "site_category"
        ] = "Unstable independent of factors"

        site_characterisation.loc[
            (site_characterisation["deviation_share"] >= 0.10)
            & (site_characterisation["number_of_sensitivities"] == 1),
            "site_category"
        ] = "Single-factor-sensitive"

        site_characterisation.loc[
            (site_characterisation["deviation_share"] >= 0.10)
            & (site_characterisation["number_of_sensitivities"] >= 2),
            "site_category"
        ] = "Multiple-factor-sensitive"

        site_characterisation["site_label"] = (
            site_characterisation["site_id"].astype(str)
            + " - "
            + site_characterisation["site_name"].astype(str)
            + " ("
            + site_characterisation["municipality"].astype(str)
            + ")"
        )

        site_characterisation["deviation_share_for_size"] = site_characterisation["deviation_share"] * 100

        return site_characterisation
    
    @render_widget
    def dev_direction_profile_map():
        site_map = site_characterisation_data().dropna(subset=["latitude", "longitude"]).copy()

        if input.map_view() == "direction_profile":
            color_col = "direction_profile"
            legend_title = "Direction profile"
            map_title = "Direction profile of deviations by counting site"

        elif input.map_view() == "site_category":
            color_col = "site_category"
            legend_title = "Site category"
            map_title = "Site category by counting site"

        else:
            color_col = "main_sensitivity_factor"
            legend_title = "Main sensitivity factor"
            map_title = "Main sensitivity factor by counting site"

        site_map["category_label"] = site_map[color_col].astype(str)

        site_map["deviation_share_label"] = (
            (site_map["deviation_share"] * 100).round(1).astype(str) + "%"
        )
        site_map["higher_share_label"] = (
            (site_map["higher_deviation_share"] * 100).round(1).astype(str) + "%"
        )
        site_map["lower_share_label"] = (
            (site_map["lower_deviation_share"] * 100).round(1).astype(str) + "%"
        )

        fig = px.scatter_mapbox(
            site_map,
            lat="latitude",
            lon="longitude",
            color=color_col,
            color_discrete_sequence=color_cat, 
            size="deviation_share_for_size",
            size_max=24,
            zoom=7,
            height=700,
            title=map_title,
            custom_data=[
                "category_label",
                "site_label",
                "deviation_share_label",
                "higher_share_label",
                "lower_share_label",
            ],
        )

        fig.update_traces(
            hovertemplate=
            "<b>%{customdata[0]}</b><br><br>"
            + "Location: %{customdata[1]}<br>"
            + "Deviation share: %{customdata[2]}<br>"
            + "Higher share: %{customdata[3]}<br>"
            + "Lower share: %{customdata[4]}"
            + "<extra></extra>"
        )

        fig.update_layout(
            mapbox_style="carto-positron",
            paper_bgcolor="#f2f2f2",
            plot_bgcolor="#f2f2f2",
            font=dict(color="#333333"),
            title=dict(x=0.02, font=dict(size=18)), 
            legend_title_text=legend_title,
            margin={"r": 0, "t": 55, "l": 0, "b": 0},
        )
        
        return fig

# 3. CREATE APP
app = App(app_ui, server)
