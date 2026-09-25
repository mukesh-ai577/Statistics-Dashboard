import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Statistics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Statistics Dashboard")
st.write(
    "Interactive dashboard for exploring distributions, "
    "hypothesis testing and confidence intervals."
)


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Choose Module",
    [   "📊 Descriptive Statistics",
        "📈 Distributions",
        "🧪 Hypothesis Testing",
        "📐 Confidence Interval"
    ]
)
# ===================================================
# DESCRIPTIVE STATISTICS
# ===================================================

if page == "📊 Descriptive Statistics":

    st.header("📊 Descriptive Statistics")

    st.write(
        "Upload a CSV file and explore basic statistical properties."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_columns) == 0:

            st.warning(
                "No numeric columns found in this dataset."
            )

        else:

            column = st.selectbox(
                "Select Numeric Column",
                numeric_columns
            )

            data = df[column].dropna()

            # -----------------------------
            # STATISTICS
            # -----------------------------

            mean = np.mean(data)
            median = np.median(data)

            mode_result = stats.mode(
                data,
                keepdims=True
            )

            mode = mode_result.mode[0]

            variance = np.var(
                data,
                ddof=1
            )

            std = np.std(
                data,
                ddof=1
            )

            skewness = stats.skew(data)

            kurtosis = stats.kurtosis(data)

            # -----------------------------
            # METRICS
            # -----------------------------

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Mean",
                f"{mean:.3f}"
            )

            col2.metric(
                "Median",
                f"{median:.3f}"
            )

            col3.metric(
                "Mode",
                f"{mode:.3f}"
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Variance",
                f"{variance:.3f}"
            )

            col2.metric(
                "Std Deviation",
                f"{std:.3f}"
            )

            col3.metric(
                "Skewness",
                f"{skewness:.3f}"
            )

            col4.metric(
                "Kurtosis",
                f"{kurtosis:.3f}"
            )

            # -----------------------------
            # SUMMARY
            # -----------------------------

            st.subheader("Statistical Summary")

            summary = pd.DataFrame({
                "Statistic": [
                    "Count",
                    "Mean",
                    "Median",
                    "Mode",
                    "Variance",
                    "Standard Deviation",
                    "Minimum",
                    "Maximum",
                    "Skewness",
                    "Kurtosis"
                ],

                "Value": [
                    len(data),
                    mean,
                    median,
                    mode,
                    variance,
                    std,
                    np.min(data),
                    np.max(data),
                    skewness,
                    kurtosis
                ]
            })

            st.dataframe(
                summary,
                use_container_width=True
            )

            # -----------------------------
            # HISTOGRAM
            # -----------------------------

            st.subheader("Distribution")

            fig, ax = plt.subplots()

            ax.hist(
                data,
                bins=20,
                edgecolor="black"
            )

            ax.set_title(
                f"Distribution of {column}"
            )

            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")

            st.pyplot(fig)

            # -----------------------------
            # BOX PLOT
            # -----------------------------

            st.subheader("Box Plot")

            fig, ax = plt.subplots()

            ax.boxplot(data)

            ax.set_ylabel(column)

            st.pyplot(fig)

# ===================================================
# 1. DISTRIBUTIONS
# ===================================================

if page == "📈 Distributions":

    st.header("📈 Probability Distributions")

    distribution = st.selectbox(
        "Select Distribution",
        [
            "Normal Distribution",
            "Binomial Distribution",
            "Poisson Distribution"
        ]
    )

    # ------------------------------------------------
    # NORMAL DISTRIBUTION
    # ------------------------------------------------

    if distribution == "Normal Distribution":

        st.subheader("Normal Distribution")

        col1, col2 = st.columns(2)

        with col1:
            mean = st.number_input(
                "Mean (μ)",
                value=0.0
            )

        with col2:
            std = st.number_input(
                "Standard Deviation (σ)",
                min_value=0.01,
                value=1.0
            )

        x = np.linspace(
            mean - 4 * std,
            mean + 4 * std,
            500
        )

        y = stats.norm.pdf(
            x,
            mean,
            std
        )

        fig, ax = plt.subplots()

        ax.plot(x, y)

        ax.set_title("Normal Distribution")
        ax.set_xlabel("X")
        ax.set_ylabel("Probability Density")

        st.pyplot(fig)

        st.write("### Probability Calculator")

        x_value = st.number_input(
            "Enter X value",
            value=float(mean)
        )

        probability = stats.norm.cdf(
            x_value,
            mean,
            std
        )

        st.metric(
            "P(X ≤ x)",
            f"{probability:.4f}"
        )


    # ------------------------------------------------
    # BINOMIAL DISTRIBUTION
    # ------------------------------------------------

    elif distribution == "Binomial Distribution":

        st.subheader("Binomial Distribution")

        col1, col2 = st.columns(2)

        with col1:
            n = st.number_input(
                "Number of Trials (n)",
                min_value=1,
                value=10,
                step=1
            )

        with col2:
            p = st.number_input(
                "Probability of Success (p)",
                min_value=0.0,
                max_value=1.0,
                value=0.5
            )

        x = np.arange(0, n + 1)

        y = stats.binom.pmf(
            x,
            n,
            p
        )

        fig, ax = plt.subplots()

        ax.bar(x, y)

        ax.set_title("Binomial Distribution")
        ax.set_xlabel("Number of Successes")
        ax.set_ylabel("Probability")

        st.pyplot(fig)

        k = st.number_input(
            "Enter number of successes",
            min_value=0,
            max_value=int(n),
            value=5
        )

        probability = stats.binom.pmf(
            k,
            n,
            p
        )

        st.metric(
            f"P(X = {k})",
            f"{probability:.4f}"
        )


    # ------------------------------------------------
    # POISSON DISTRIBUTION
    # ------------------------------------------------

    else:

        st.subheader("Poisson Distribution")

        lam = st.number_input(
            "Lambda (λ)",
            min_value=0.1,
            value=5.0
        )

        x = np.arange(
            0,
            max(20, int(lam * 3))
        )

        y = stats.poisson.pmf(
            x,
            lam
        )

        fig, ax = plt.subplots()

        ax.bar(x, y)

        ax.set_title("Poisson Distribution")
        ax.set_xlabel("Number of Events")
        ax.set_ylabel("Probability")

        st.pyplot(fig)

        k = st.number_input(
            "Enter number of events",
            min_value=0,
            value=int(lam)
        )

        probability = stats.poisson.pmf(
            k,
            lam
        )

        st.metric(
            f"P(X = {k})",
            f"{probability:.4f}"
        )


# ===================================================
# 2. HYPOTHESIS TESTING
# ===================================================

elif page == "🧪 Hypothesis Testing":

    st.header("🧪 Hypothesis Testing")

    test = st.selectbox(
        "Select Test",
        [
            "One Sample T-Test",
            "Independent Two Sample T-Test",
            "Chi-Square Test"
        ]
    )

    # ------------------------------------------------
    # ONE SAMPLE T TEST
    # ------------------------------------------------

    if test == "One Sample T-Test":

        st.subheader("One Sample T-Test")

        st.write(
            "Test whether the sample mean is significantly "
            "different from a hypothesized population mean."
        )

        population_mean = st.number_input(
            "Hypothesized Mean (μ₀)",
            value=50.0
        )

        sample_text = st.text_area(
            "Enter sample values separated by commas",
            "52, 48, 51, 53, 49, 50, 54, 47"
        )

        try:

            sample = np.array(
                [
                    float(x.strip())
                    for x in sample_text.split(",")
                ]
            )

            t_stat, p_value = stats.ttest_1samp(
                sample,
                population_mean
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Sample Mean",
                f"{np.mean(sample):.3f}"
            )

            col2.metric(
                "T-Statistic",
                f"{t_stat:.3f}"
            )

            col3.metric(
                "P-Value",
                f"{p_value:.4f}"
            )

            alpha = 0.05

            if p_value < alpha:
                st.error(
                    "Reject H₀: statistically significant difference found."
                )
            else:
                st.success(
                    "Fail to reject H₀: insufficient evidence of a difference."
                )

        except:

            st.warning(
                "Please enter valid numeric values."
            )


    # ------------------------------------------------
    # TWO SAMPLE T TEST
    # ------------------------------------------------

    elif test == "Independent Two Sample T-Test":

        st.subheader(
            "Independent Two Sample T-Test"
        )

        group1_text = st.text_area(
            "Group 1",
            "10, 12, 13, 15, 14"
        )

        group2_text = st.text_area(
            "Group 2",
            "18, 20, 19, 21, 22"
        )

        try:

            group1 = np.array(
                [
                    float(x.strip())
                    for x in group1_text.split(",")
                ]
            )

            group2 = np.array(
                [
                    float(x.strip())
                    for x in group2_text.split(",")
                ]
            )

            t_stat, p_value = stats.ttest_ind(
                group1,
                group2
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "T-Statistic",
                f"{t_stat:.3f}"
            )

            col2.metric(
                "P-Value",
                f"{p_value:.4f}"
            )

            if p_value < 0.05:

                st.error(
                    "Reject H₀: the two group means are significantly different."
                )

            else:

                st.success(
                    "Fail to reject H₀: insufficient evidence of a difference."
                )

        except:

            st.warning(
                "Please enter valid numeric values."
            )


    # ------------------------------------------------
    # CHI SQUARE
    # ------------------------------------------------

    else:

        st.subheader("Chi-Square Test")

        st.write(
            "Test whether two categorical variables are independent."
        )

        observed = np.array(
            [
                [20, 30],
                [30, 20]
            ]
        )

        observed_df = pd.DataFrame(
            observed,
            columns=["Success", "Failure"],
            index=["Group A", "Group B"]
        )

        st.write("Observed Frequencies")

        st.dataframe(
            observed_df
        )

        chi2, p_value, dof, expected = stats.chi2_contingency(
            observed
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Chi-Square",
            f"{chi2:.3f}"
        )

        col2.metric(
            "P-Value",
            f"{p_value:.4f}"
        )

        col3.metric(
            "Degrees of Freedom",
            dof
        )

        if p_value < 0.05:

            st.error(
                "Reject H₀: variables are not independent."
            )

        else:

            st.success(
                "Fail to reject H₀: insufficient evidence of association."
            )


# ===================================================
# 3. CONFIDENCE INTERVAL
# ===================================================

else:

    st.header("📐 Confidence Interval")

    st.write(
        "Calculate a confidence interval for the population mean."
    )

    sample_text = st.text_area(
        "Enter sample values separated by commas",
        "52, 48, 51, 53, 49, 50, 54, 47"
    )

    confidence = st.slider(
        "Confidence Level",
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        step=0.01
    )

    try:

        sample = np.array(
            [
                float(x.strip())
                for x in sample_text.split(",")
            ]
        )

        n = len(sample)

        sample_mean = np.mean(sample)

        sample_std = np.std(
            sample,
            ddof=1
        )

        standard_error = sample_std / np.sqrt(n)

        alpha = 1 - confidence

        t_critical = stats.t.ppf(
            1 - alpha / 2,
            df=n - 1
        )

        margin_error = (
            t_critical *
            standard_error
        )

        lower = sample_mean - margin_error

        upper = sample_mean + margin_error

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Sample Mean",
            f"{sample_mean:.3f}"
        )

        col2.metric(
            "Lower Bound",
            f"{lower:.3f}"
        )

        col3.metric(
            "Upper Bound",
            f"{upper:.3f}"
        )

        st.success(
            f"{confidence * 100:.0f}% Confidence Interval: "
            f"({lower:.3f}, {upper:.3f})"
        )

        # Visualization

        fig, ax = plt.subplots()

        ax.errorbar(
            sample_mean,
            0,
            xerr=margin_error,
            fmt="o",
            capsize=8
        )

        ax.axvline(
            sample_mean,
            linestyle="--"
        )

        ax.set_xlabel("Mean")
        ax.set_yticks([])
        ax.set_title(
            f"{confidence * 100:.0f}% Confidence Interval"
        )

        st.pyplot(fig)

    except:

        st.warning(
            "Please enter valid numeric values."
        )


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.info(
    "Built with Python, NumPy, SciPy, Pandas, "
    "Matplotlib and Streamlit."
)
