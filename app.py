# -*- coding: utf-8 -*-
"""Aplikasi Streamlit: Prediksi Risiko Pembatalan Pesanan E-Commerce.

Jalankan dengan:  streamlit run app.py
"""

from __future__ import annotations

from datetime import time as dt_time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.bundle import ModelBundle, load_bundle
from src.config import (
    APP_SUBTITLE,
    APP_TITLE,
    DAY_NAMES,
    FEATURE_HELP,
    FEATURE_LABELS,
    MONTH_NAMES,
    RISK_COLORS,
    classify_risk,
)
from src.explain import build_recommendations, factor_table
from src.preprocess import build_model_input, describe_row, missing_columns
from src.predict import predict_contributions, predict_dataframe, predict_proba
from src.ui import hero, inject_css, result_card, risk_legend, section

st.set_page_config(
    page_title="Prediksi Pembatalan Pesanan",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

WIDGET_KEYS = [
    "w_qty", "w_weight", "w_discount", "w_ncat", "w_shipcost",
    "w_category", "w_shipping", "w_payment", "w_city", "w_tanggal", "w_jam",
]

TARGET_ALIASES = ("Status Pesanan", "status pesanan", "order_status", "target")

RISK_DOMAIN = ["Rendah", "Sedang", "Tinggi"]
RISK_RANGE = [RISK_COLORS[level] for level in RISK_DOMAIN]


@st.cache_resource(show_spinner="Memuat model ...")
def get_bundle() -> ModelBundle:
    return load_bundle()


# ---------------------------------------------------------------------------
# Helper umum
# ---------------------------------------------------------------------------


def read_csv_auto(source) -> pd.DataFrame:
    """Baca CSV dengan deteksi delimiter dan encoding sederhana."""
    data = pd.read_csv(source, sep=";")
    if data.shape[1] == 1:
        if hasattr(source, "seek"):
            source.seek(0)
        data = pd.read_csv(source, sep=",")
    return data


def option_index(options: list[str], value) -> int:
    text = str(value)
    return options.index(text) if text in options else 0


def example_row(bundle: ModelBundle) -> dict:
    if bundle.examples:
        return dict(bundle.examples[0]["features"])
    row = {}
    for column in bundle.feature_columns:
        if column in bundle.cat_cols:
            row[column] = bundle.cat_defaults.get(column, "")
        else:
            row[column] = bundle.num_defaults.get(column, 0)
    return row


def factor_chart(factors: pd.DataFrame) -> alt.Chart:
    data = factors.copy()
    data["arah_pengaruh"] = data["kontribusi"].apply(
        lambda value: "Menaikkan risiko" if value > 0 else "Menurunkan risiko"
    )
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "porsi:Q",
                title="Kontribusi terhadap prediksi (%)",
                scale=alt.Scale(domain=[0, max(35, data["porsi"].max() * 1.15)]),
            ),
            y=alt.Y(
                "label:N",
                sort=alt.SortField(field="porsi", order="descending"),
                title=None,
            ),
            color=alt.Color(
                "arah_pengaruh:N",
                scale=alt.Scale(
                    domain=["Menaikkan risiko", "Menurunkan risiko"],
                    range=["#dc2626", "#059669"],
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Fitur"),
                alt.Tooltip("porsi:Q", title="Porsi (%)", format=".1f"),
                alt.Tooltip("kontribusi:Q", title="Nilai SHAP", format="+.3f"),
            ],
        )
        .properties(height=max(220, 42 * len(data)))
    )


# ---------------------------------------------------------------------------
# Tab 1: prediksi satu pesanan
# ---------------------------------------------------------------------------


def apply_preset(bundle: ModelBundle, example: dict) -> None:
    """Callback tombol contoh: isi form + hitung prediksi langsung."""
    st.session_state["preset"] = example
    for key in WIDGET_KEYS:
        st.session_state.pop(key, None)

    row = dict(example["features"])
    scaled = build_model_input(pd.DataFrame([row]), bundle)
    st.session_state["result"] = {
        "row": row,
        "prob": float(predict_proba(bundle, scaled)[0]),
        "contrib": predict_contributions(bundle, scaled)[0],
        "timestamp": example.get("timestamp", "-"),
        "actual": example.get("actual"),
    }


def render_examples(bundle: ModelBundle) -> None:
    if not bundle.examples:
        return
    st.markdown("**Coba contoh data nyata dari dataset:**")
    columns = st.columns(len(bundle.examples))
    icons = {"Risiko Rendah": "🟢", "Risiko Sedang": "🟠", "Risiko Tinggi": "🔴"}
    for column, example in zip(columns, bundle.examples):
        with column:
            st.button(
                f"{icons.get(example['label'], '•')} {example['label']}",
                key=f"preset_{example['label']}",
                width="stretch",
                on_click=apply_preset,
                args=(bundle, example),
                help=(
                    f"Contoh pesanan dengan probabilitas batal "
                    f"{example['probability'] * 100:.1f}%."
                ),
            )


def render_form(bundle: ModelBundle) -> None:
    preset = st.session_state.get("preset") or {}
    defaults = bundle.num_defaults

    def preset_number(key: str, fallback: float) -> float:
        value = preset.get(key, fallback)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(fallback)

    with st.form("form_prediksi", border=True):
        st.markdown("##### 🧾 Detail pesanan")
        col1, col2, col3, col4 = st.columns(4)
        qty = col1.number_input(
            "Jumlah produk", min_value=1, max_value=999, step=1,
            value=int(preset_number("total_qty", defaults.get("total_qty", 1))),
            key="w_qty", help=FEATURE_HELP["total_qty"],
        )
        weight = col2.number_input(
            "Berat paket (gram)", min_value=0, max_value=200_000, step=100,
            value=int(preset_number("total_weight_gr", defaults.get("total_weight_gr", 500))),
            key="w_weight", help=FEATURE_HELP["total_weight_gr"],
        )
        discount = col3.number_input(
            "Total diskon (Rp)", min_value=0, max_value=10_000_000, step=1_000,
            value=int(preset_number("Total Diskon", defaults.get("Total Diskon", 0))),
            key="w_discount", help=FEATURE_HELP["Total Diskon"],
        )
        num_categories = col4.number_input(
            "Jumlah kategori produk", min_value=1, max_value=50, step=1,
            value=int(
                preset_number(
                    "num_product_categories",
                    defaults.get("num_product_categories", 1),
                )
            ),
            key="w_ncat", help=FEATURE_HELP["num_product_categories"],
        )

        st.markdown("##### 🛍️ Produk, pengiriman & pembayaran")
        col1, col2, col3, col4 = st.columns(4)
        shipping_cost = col1.number_input(
            "Perkiraan ongkos kirim (Rp)", min_value=0, max_value=1_000_000, step=1_000,
            value=int(
                preset_number(
                    "Perkiraan Ongkos Kirim",
                    defaults.get("Perkiraan Ongkos Kirim", 11000),
                )
            ),
            key="w_shipcost", help=FEATURE_HELP["Perkiraan Ongkos Kirim"],
        )
        category_options = bundle.category_options.get("product_categories", [""])
        category = col2.selectbox(
            "Kategori produk", category_options,
            index=option_index(
                category_options,
                preset.get("product_categories", bundle.cat_defaults.get("product_categories")),
            ),
            key="w_category",
        )
        shipping_options = bundle.category_options.get("Opsi Pengiriman", [""])
        shipping = col3.selectbox(
            "Opsi pengiriman", shipping_options,
            index=option_index(
                shipping_options,
                preset.get("Opsi Pengiriman", bundle.cat_defaults.get("Opsi Pengiriman")),
            ),
            key="w_shipping",
        )
        payment_options = bundle.category_options.get("Metode Pembayaran", [""])
        payment = col4.selectbox(
            "Metode pembayaran", payment_options,
            index=option_index(
                payment_options,
                preset.get("Metode Pembayaran", bundle.cat_defaults.get("Metode Pembayaran")),
            ),
            key="w_payment",
        )

        st.markdown("##### 📍 Lokasi & waktu pesanan")
        col1, col2, col3, col4 = st.columns([2.2, 1.8, 1.4, 1])
        city_options = bundle.category_options.get("Kota/Kabupaten", [""])
        city = col1.selectbox(
            "Kota/Kabupaten", city_options,
            index=option_index(
                city_options,
                preset.get("Kota/Kabupaten", bundle.cat_defaults.get("Kota/Kabupaten")),
            ),
            key="w_city",
        )
        province = bundle.city_province.get(
            city, bundle.cat_defaults.get("Provinsi", "-")
        )
        col2.text_input("Provinsi (otomatis)", value=province, disabled=True)

        default_timestamp = pd.Timestamp("2024-08-15 14:00")
        if preset.get("timestamp"):
            try:
                default_timestamp = pd.Timestamp(preset["timestamp"])
            except (ValueError, TypeError):
                pass
        tanggal = col3.date_input(
            "Tanggal pesanan",
            value=default_timestamp.date(),
            key="w_tanggal",
        )
        jam = col4.time_input(
            "Jam pesanan",
            value=dt_time(default_timestamp.hour, default_timestamp.minute),
            step=900,
            key="w_jam",
        )
        st.caption(
            f"Fitur waktu yang dipakai model: bulan **{MONTH_NAMES[tanggal.month]}**, "
            f"hari **{DAY_NAMES[tanggal.weekday()]}**, jam **{jam.hour:02d}**."
        )

        submitted = st.form_submit_button(
            "🔮 Prediksi risiko pembatalan",
            type="primary",
            width="stretch",
        )

    if submitted:
        row = {
            "total_qty": qty,
            "total_weight_gr": weight,
            "Total Diskon": discount,
            "product_categories": category,
            "num_product_categories": num_categories,
            "Opsi Pengiriman": shipping,
            "Metode Pembayaran": payment,
            "Kota/Kabupaten": city,
            "Provinsi": province,
            "Perkiraan Ongkos Kirim": shipping_cost,
            "order_month": tanggal.month,
            "order_dayofweek": tanggal.weekday(),
            "order_hour": jam.hour,
        }
        try:
            scaled = build_model_input(pd.DataFrame([row]), bundle)
            probability = float(predict_proba(bundle, scaled)[0])
            contribution = predict_contributions(bundle, scaled)[0]
        except Exception as exc:  # noqa: BLE001
            st.error(f"Gagal melakukan prediksi: {exc}")
            return

        st.session_state["result"] = {
            "row": row,
            "prob": probability,
            "contrib": contribution,
            "timestamp": f"{tanggal.strftime('%d %B %Y')} {jam.strftime('%H:%M')}",
            "actual": None,
        }
        st.session_state["preset"] = None


def render_result(bundle: ModelBundle, threshold: float) -> None:
    result = st.session_state.get("result")
    if not result:
        return

    probability = result["prob"]
    risk = classify_risk(probability, threshold)
    section(
        "Hasil prediksi",
        "Angka di bawah adalah estimasi probabilitas pesanan dibatalkan oleh model.",
    )
    result_card(probability, risk, threshold)

    factors = factor_table(bundle, result["contrib"], top_k=6)
    col1, col2 = st.columns([1.35, 1])

    with col1:
        st.markdown("**Faktor paling berpengaruh**")
        st.altair_chart(factor_chart(factors), width="stretch")
        st.caption(
            "Kontribusi dihitung dengan SHAP bawaan XGBoost. Merah = menaikkan "
            "risiko batal, hijau = menurunkan risiko."
        )

    with col2:
        st.markdown("**Rekomendasi tindakan**")
        tips = build_recommendations(result["row"], factors, risk)
        for tip in tips:
            st.markdown(f"- {tip}")

        if result.get("actual") is not None:
            actual = "Batal" if result["actual"] == 1 else "Tidak batal"
            st.info(f"Label aktual pada dataset contoh: **{actual}**.")

    with st.expander("Lihat nilai fitur yang digunakan model"):
        formatted = describe_row(result["row"], bundle)
        detail = pd.DataFrame(
            {"Fitur": list(formatted.keys()), "Nilai": list(formatted.values())}
        )
        st.dataframe(detail, hide_index=True, width="stretch")
        st.caption(
            "Probabilitas dihitung dari 13 fitur di atas setelah melalui imputasi, "
            "label encoding, dan standardisasi."
        )

    with st.expander("Lihat kontribusi seluruh fitur (SHAP)"):
        full = factor_table(bundle, result["contrib"])
        st.dataframe(
            full[["label", "porsi", "arah", "kontribusi"]].rename(
                columns={
                    "label": "Fitur",
                    "porsi": "Porsi (%)",
                    "arah": "Arah pengaruh",
                    "kontribusi": "Nilai SHAP",
                }
            ),
            hide_index=True,
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Tab 2: prediksi batch
# ---------------------------------------------------------------------------


def render_batch_results(bundle: ModelBundle, threshold: float) -> None:
    data: pd.DataFrame | None = st.session_state.get("batch_result")
    if data is None:
        return
    if len(data) == 0:
        st.warning("File tidak berisi baris data yang bisa diprediksi.")
        return

    result = data.copy()
    result["probabilitas_batal"] = result["probabilitas_batal"].astype(float)
    result["tingkat_risiko"] = result["probabilitas_batal"].apply(
        lambda value: classify_risk(value, threshold)
    )
    result["prediksi"] = np.where(
        result["probabilitas_batal"] >= threshold, "Batal", "Tidak batal"
    )

    st.divider()
    section("Ringkasan hasil", f"{len(result):,} pesanan diprediksi.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total pesanan", f"{len(result):,}")
    col2.metric(
        "Rata-rata probabilitas",
        f"{result['probabilitas_batal'].mean() * 100:.1f}%",
    )
    col3.metric(
        "Prediksi batal",
        f"{(result['prediksi'] == 'Batal').sum():,}",
        f"{(result['prediksi'] == 'Batal').mean() * 100:.1f}% dari total",
        delta_color="off",
    )
    high_risk = (result["tingkat_risiko"] == "Tinggi").sum()
    col4.metric("Risiko tinggi", f"{high_risk:,}")

    # Evaluasi bila kolom label tersedia.
    target_column = next((c for c in TARGET_ALIASES if c in result.columns), None)
    if target_column is not None:
        raw_target = result[target_column]
        if pd.api.types.is_numeric_dtype(raw_target):
            actual = raw_target.astype(int)
        else:
            actual = (
                raw_target.astype(str).str.strip().str.lower()
                .isin(["batal", "cancelled", "canceled"])
                .astype(int)
            )
        predicted = (result["prediksi"] == "Batal").astype(int)
        with st.expander("Evaluasi terhadap label aktual", expanded=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Akurasi", f"{accuracy_score(actual, predicted) * 100:.1f}%")
            m2.metric(
                "Presisi",
                f"{precision_score(actual, predicted, zero_division=0) * 100:.1f}%",
            )
            m3.metric(
                "Recall",
                f"{recall_score(actual, predicted, zero_division=0) * 100:.1f}%",
            )
            m4.metric(
                "F1-Score",
                f"{f1_score(actual, predicted, zero_division=0) * 100:.1f}%",
            )
            try:
                m5, m6 = st.columns(2)
                m5.metric(
                    "ROC-AUC",
                    f"{roc_auc_score(actual, result['probabilitas_batal']) * 100:.1f}%",
                )
                m6.metric(
                    "Pembatalan aktual",
                    f"{actual.mean() * 100:.1f}%",
                    f"{int(actual.sum()):,} pesanan",
                    delta_color="off",
                )
            except ValueError:
                pass

            crosstab = pd.crosstab(
                pd.Series(actual.map({0: "Tidak batal", 1: "Batal"}), name="Aktual"),
                pd.Series(predicted.map({0: "Tidak batal", 1: "Batal"}), name="Prediksi"),
            )
            st.markdown("**Confusion matrix**")
            st.dataframe(crosstab, width="stretch")

    # Distribusi probabilitas.
    section("Distribusi probabilitas", "Garis putus-putus menandai ambang keputusan.")
    histogram = (
        alt.Chart(result)
        .mark_bar(opacity=0.9)
        .encode(
            x=alt.X(
                "probabilitas_batal:Q",
                bin=alt.Bin(maxbins=36),
                title="Probabilitas pembatalan",
            ),
            y=alt.Y("count():Q", title="Jumlah pesanan"),
            color=alt.Color(
                "tingkat_risiko:N",
                scale=alt.Scale(domain=RISK_DOMAIN, range=RISK_RANGE),
                title="Tingkat risiko",
            ),
            tooltip=[alt.Tooltip("count():Q", title="Jumlah")],
        )
        .properties(height=260)
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [threshold]}))
        .mark_rule(color="#0f172a", strokeDash=[5, 4], size=1.6)
        .encode(x="x:Q")
    )
    st.altair_chart(histogram + rule, width="stretch")

    section("Tabel hasil", "Gulir untuk melihat seluruh baris, atau unduh sebagai CSV.")
    display_columns = list(result.columns)
    st.dataframe(
        result[display_columns],
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "probabilitas_batal": st.column_config.ProgressColumn(
                "Probabilitas batal",
                format="%.4f",
                min_value=0.0,
                max_value=1.0,
            ),
        },
    )
    st.download_button(
        "⬇️ Unduh hasil prediksi (CSV)",
        result.to_csv(index=False).encode("utf-8-sig"),
        file_name="hasil_prediksi_pembatalan.csv",
        mime="text/csv",
        width="content",
    )


def render_batch_tab(bundle: ModelBundle, threshold: float) -> None:
    section(
        "Prediksi massal dari file CSV",
        "Unggah data pesanan (boleh banyak baris) untuk diprediksi sekaligus.",
    )

    st.markdown(
        "**Kolom yang dibutuhkan:** "
        + ", ".join(f"`{column}`" for column in bundle.feature_columns)
        + ". Kolom tanggal (`Waktu Pesanan Dibuat`) juga bisa dipakai; fitur "
        "bulan/hari/jam akan dihitung otomatis."
    )

    template = pd.DataFrame([example_row(bundle)])
    col1, col2 = st.columns([1, 2])
    col1.download_button(
        "⬇️ Unduh template CSV",
        template.to_csv(index=False).encode("utf-8-sig"),
        file_name="template_prediksi.csv",
        mime="text/csv",
    )
    col2.caption(
        "Template berisi satu baris contoh dengan nama kolom yang benar. "
        "Anda juga bisa memakai file mentah dataset (mis. `all_months_clean.csv`)."
    )

    uploaded = st.file_uploader(
        "Unggah file CSV", type=["csv"], accept_multiple_files=False
    )
    if uploaded is None:
        return

    if st.session_state.get("batch_file") != uploaded.name:
        st.session_state["batch_file"] = uploaded.name
        st.session_state.pop("batch_result", None)

    try:
        raw = read_csv_auto(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"File tidak dapat dibaca sebagai CSV: {exc}")
        return

    missing = missing_columns(raw, bundle)
    if missing:
        st.error(
            "Kolom berikut tidak ditemukan pada file: "
            + ", ".join(f"`{column}`" for column in missing)
            + ". Periksa kembali nama kolomnya (gunakan template bila perlu)."
        )
        return

    st.caption(f"{len(raw):,} baris terbaca.")
    st.dataframe(raw.head(5), hide_index=True, width="stretch")

    if st.button("🚀 Jalankan prediksi batch", type="primary"):
        try:
            with st.spinner("Menghitung prediksi ..."):
                results = predict_dataframe(raw, bundle)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Terjadi kesalahan saat memproses data: {exc}")
            return
        st.session_state["batch_result"] = results

    render_batch_results(bundle, threshold)


# ---------------------------------------------------------------------------
# Tab 3: performa model
# ---------------------------------------------------------------------------


def render_model_tab(bundle: ModelBundle) -> None:
    metrics = bundle.metrics or {}
    section(
        "Performa model",
        f"Hasil evaluasi {bundle.model_name} pada data uji (20% dari dataset).",
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi uji", f"{metrics.get('Test Accuracy (%)', '-')}%")
    col2.metric("F1-Score", f"{metrics.get('F1-Score (%)', '-')}%")
    col3.metric("ROC-AUC", f"{metrics.get('ROC-AUC (%)', '-')}%")
    col4.metric("Gap train–test", f"{metrics.get('Gap (%)', '-')}%")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi latih", f"{metrics.get('Train Accuracy (%)', '-')}%")
    col2.metric("Presisi", f"{metrics.get('Precision (%)', '-')}%")
    col3.metric("Recall", f"{metrics.get('Recall (%)', '-')}%")
    col4.metric("Status fit", metrics.get("Status Fit", "-"))

    if metrics.get("Status Fit") == "Goodfit":
        st.success(
            "Model tergolong **Goodfit**: selisih akurasi latih dan uji hanya "
            f"{metrics.get('Gap (%)', '-')}% sehingga tidak menunjukkan overfitting berarti."
        )
    st.info(
        f"Recall {metrics.get('Recall (%)', '-')}% berarti model menangkap sebagian "
        "pembatalan aktual. Gunakan hasil prediksi sebagai alat bantu prioritas "
        "pemantauan, bukan satu-satunya dasar keputusan."
    )

    st.divider()
    col1, col2 = st.columns([1.4, 1])
    with col1:
        section("Feature importance", "Kontribusi fitur terhadap model (gain).")
        importance = pd.DataFrame(
            {
                "fitur": list(bundle.feature_importance.keys()),
                "gain": list(bundle.feature_importance.values()),
            }
        )
        if importance.empty:
            st.caption("Informasi feature importance tidak tersedia.")
        else:
            importance["label"] = importance["fitur"].map(
                lambda value: FEATURE_LABELS.get(value, value)
            )
            importance["porsi"] = (
                importance["gain"] / importance["gain"].sum() * 100
            )
            chart = (
                alt.Chart(importance)
                .mark_bar(cornerRadiusEnd=4, color="#0f766e")
                .encode(
                    x=alt.X("porsi:Q", title="Kontribusi (%)"),
                    y=alt.Y(
                        "label:N",
                        sort=alt.SortField(field="porsi", order="descending"),
                        title=None,
                    ),
                    tooltip=[
                        alt.Tooltip("label:N", title="Fitur"),
                        alt.Tooltip("porsi:Q", title="Kontribusi (%)", format=".1f"),
                    ],
                )
                .properties(height=380)
            )
            st.altair_chart(chart, width="stretch")

    with col2:
        section("Hyperparameter", "Parameter terbaik hasil grid search.")
        params = bundle.trained_params or {}
        rows = [
            ("Algoritma", bundle.model_name),
            ("Objective", params.get("objective", "binary:logistic")),
            ("Jumlah pohon", params.get("n_estimators", "-")),
            ("Kedalaman maksimum", params.get("max_depth", "-")),
            ("Learning rate", params.get("learning_rate", "-")),
            ("Subsample", params.get("subsample", "-")),
            ("Random state", params.get("random_state", "-")),
        ]
        st.dataframe(
            pd.DataFrame(
                [(name, str(value)) for name, value in rows],
                columns=["Parameter", "Nilai"],
            ),
            hide_index=True,
            width="stretch",
        )

        section("Preprocessing", "Transformasi yang diterapkan sebelum prediksi.")
        st.markdown(
            f"- **Imputasi numerik:** median ({len(bundle.num_cols)} fitur)\n"
            f"- **Imputasi kategorikal:** modus ({len(bundle.cat_cols)} fitur)\n"
            "- **Encoding:** LabelEncoder per kolom kategorikal\n"
            "- **Scaling:** StandardScaler (mean 0, std 1)"
        )
        st.caption(
            "Fitur kategorikal: " + ", ".join(bundle.cat_cols) + "."
        )

    info = bundle.dataset_info
    if info:
        st.divider()
        section("Dataset pelatihan", "Sumber data yang dipakai saat melatih model.")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Jumlah baris", f"{info.get('rows', 0):,}")
        col2.metric(
            "Tingkat pembatalan", f"{info.get('cancel_rate', 0) * 100:.2f}%"
        )
        col3.metric("Pesanan batal", f"{info.get('cancelled', 0):,}")
        col4.metric(
            "Periode data",
            f"{info.get('date_start', '-')}",
            f"s.d. {info.get('date_end', '-')}",
            delta_color="off",
        )


# ---------------------------------------------------------------------------
# Tab 4: panduan
# ---------------------------------------------------------------------------


def render_guide_tab(bundle: ModelBundle) -> None:
    section(
        "Panduan penggunaan",
        "Penjelasan singkat alur aplikasi dan cara menjalankannya.",
    )

    st.markdown(
        f"""
### 1. Cara memakai aplikasi
1. Buka tab **Prediksi**, isi detail pesanan (atau klik salah satu tombol contoh).
2. Tekan **🔮 Prediksi risiko pembatalan**.
3. Anda akan melihat probabilitas batal, tingkat risiko, faktor paling berpengaruh,
   serta rekomendasi tindakan.
4. Untuk banyak pesanan sekaligus, gunakan tab **Prediksi Batch** dan unggah CSV.

### 2. Alur machine learning di balik layar
```
Data pesanan mentah
   │  (jumlah produk, berat, ongkir, kategori, pembayaran, kota, waktu)
   ▼
Imputasi nilai kosong  →  Label encoding  →  StandardScaler
   ▼
Model XGBoost (200 pohon, max_depth 6, learning_rate 0.1)
   ▼
Probabilitas pembatalan  →  ambang keputusan  →  tingkat risiko
```
Model dilatih dengan data seimbang (SMOTE) dan hyperparameter tuning
(5-fold StratifiedKFold, optimasi F1-Score).

### 3. Menjalankan aplikasi
```bash
pip install -r requirements.txt
streamlit run app.py
```
Atau klik dua kali `run_app.bat` (Windows).

### 4. Struktur proyek
```
aplikasi_prediksi_pembatalan/
├── app.py                     # antarmuka Streamlit
├── src/                       # loader model, preprocessing, inferensi, UI
├── models/                    # bundle model siap pakai
├── tools/convert_model.py     # konversi .pth asli → bundle aplikasi
└── requirements.txt
```

### 5. Catatan penting
- Model memakai **{len(bundle.feature_columns)} fitur** yang sama seperti saat pelatihan.
  Nilai kategori yang belum pernah dilihat akan diarahkan ke kategori paling umum.
- Probabilitas bukan kepastian: gunakan sebagai prioritas pemantauan.
- Model tidak memakai data pribadi pembeli; hanya atribut pesanan.
- File `.pth` asli menyimpan snapshot XGBoost format lama sehingga perlu
  dikonversi lebih dulu dengan `python tools/convert_model.py` (sudah dilakukan).

---
**Tugas Besar Machine Learning — CRISP-DM**  
Ganang Ardi Pratama · NPM 202343501804 · Kelas X7L
"""
    )


# ---------------------------------------------------------------------------
# Halaman utama
# ---------------------------------------------------------------------------


def main() -> None:
    inject_css()

    try:
        bundle = get_bundle()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Model gagal dimuat: {exc}")
        st.stop()

    metrics = bundle.metrics or {}
    hero(
        APP_TITLE,
        APP_SUBTITLE,
        [
            f"Model <b>{bundle.model_name}</b>",
            f"Akurasi uji <b>{metrics.get('Test Accuracy (%)', '-')}%</b>",
            f"ROC-AUC <b>{metrics.get('ROC-AUC (%)', '-')}%</b>",
            f"<b>{len(bundle.feature_columns)}</b> fitur",
        ],
    )

    st.session_state.setdefault("preset", None)

    with st.sidebar:
        st.markdown("### ⚙️ Pengaturan prediksi")
        threshold = st.slider(
            "Ambang keputusan",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
            help=(
                "Pesanan diprediksi dibatalkan bila probabilitasnya mencapai "
                "ambang ini. Turunkan ambang untuk menangkap lebih banyak "
                "pembatalan (recall naik, presisi turun)."
            ),
        )
        st.caption(
            f"Prediksi **Batal** bila probabilitas ≥ **{threshold * 100:.0f}%**."
        )

        st.divider()
        st.markdown("### 🎨 Tingkat risiko")
        risk_legend()
        st.caption(
            "Rendah < ½ ambang · Sedang < ambang · Tinggi ≥ ambang. "
            "Ambang saat ini "
            f"{threshold * 100:.0f}%."
        )

        st.divider()
        st.markdown("### 📦 Informasi model")
        st.markdown(
            f"**{bundle.model_name}**  \n"
            f"F1-Score **{metrics.get('F1-Score (%)', '-')}%** · "
            f"Presisi **{metrics.get('Precision (%)', '-')}%** · "
            f"Recall **{metrics.get('Recall (%)', '-')}%**"
        )
        info = bundle.dataset_info
        if info:
            st.caption(
                f"Data: {info.get('rows', 0):,} pesanan "
                f"({info.get('date_start', '-')} – {info.get('date_end', '-')})."
            )
        st.divider()
        st.caption(
            "Tugas Besar Machine Learning · CRISP-DM  \n"
            "Ganang Ardi Pratama — 202343501804"
        )

    tab_predict, tab_batch, tab_model, tab_guide = st.tabs(
        ["🎯 Prediksi", "📁 Prediksi Batch", "📊 Performa Model", "ℹ️ Panduan"]
    )

    with tab_predict:
        section(
            "Prediksi satu pesanan",
            "Isi detail pesanan, atau muat contoh data untuk mencoba lebih cepat.",
        )
        render_examples(bundle)
        render_form(bundle)
        render_result(bundle, threshold)

    with tab_batch:
        render_batch_tab(bundle, threshold)

    with tab_model:
        render_model_tab(bundle)

    with tab_guide:
        render_guide_tab(bundle)

    st.markdown(
        '<p class="small-note" style="text-align:center;margin-top:32px;">'
        "Dibuat untuk Tugas Besar Machine Learning · "
        "Model XGBoost · Dataset E-Commerce Dekorasi Rumah"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
