#!/usr/bin/env python
# coding: utf-8

# Data source
import yfinance as yf

# Data viz
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

sns.set(color_codes=True, rc={'figure.figsize':(12, 4)})
sns.set_palette(sns.color_palette('muted'))

# TDA magic
import gtda.time_series as ts
import gtda.homology as hl
from gtda.plotting import plot_diagram, plot_point_cloud

from plotting import plot_crash_detections
from SWDFT import SWDFT
from HomologicalDerivative import HomologicalDerivative


def fetch_data(ticker, start_year):
    Ticker = yf.Ticker(ticker)
    ticker_df = Ticker.history(period="max")
    price_df = ticker_df['Close']
    price_resampled_df = price_df.resample('24H').ffill()[start_year:]
    return price_resampled_df


def embed(embedding_dimension, embedding_time_delay, swdft, price_values):
    if swdft:
        embedder = SWDFT(
            parameters_type="fixed",
            dimension=embedding_dimension,
            time_delay=embedding_time_delay,
            n_jobs=-1,
        )
    else:
        embedder = ts.SingleTakensEmbedding(
            parameters_type="fixed",
            dimension=embedding_dimension,
            time_delay=embedding_time_delay,
            n_jobs=-1,
        )

    price_embedded = embedder.fit_transform(price_values)
    embedder_time_delay = embedder.time_delay_
    embedder_dimension = embedder.dimension_

    return price_embedded, embedder_time_delay, embedder_dimension


def slide_window(window_size, window_stride, price_embedded):
    sliding_window = ts.SlidingWindow(size=window_size, stride=window_stride)
    price_embedded_windows = sliding_window.fit_transform(price_embedded)

    return price_embedded_windows


def time_index(embedder_dimension, embedder_time_delay, window_size, window_stride, price_values, price_resampled_df):
    window_size_price = window_size + embedder_dimension * embedder_time_delay - 2
    sliding_window_price = ts.SlidingWindow(size=window_size_price, stride=window_stride)
    window_indices = sliding_window_price.slice_windows(price_values)
    indices = [win[1] - 1 for win in window_indices[1:]]
    time_idx = price_resampled_df.iloc[indices].index

    return time_idx


def detect_crash(
        ticker, start_year,
        embedding_dimension, embedding_time_delay, swdft,
        window_size, window_stride,
        Persistence, metric_params):
    price_resampled_df = fetch_data(ticker, start_year)
    price_values = price_resampled_df.values

    price_embedded, embedder_time_delay, embedder_dimension = embed(embedding_dimension, embedding_time_delay, swdft, price_values)

    price_embedded_windows = slide_window(window_size, window_stride, price_embedded)

    time_idx = time_index(embedder_dimension, embedder_time_delay, window_size, window_stride, price_values, price_resampled_df)
    resampled_price = price_resampled_df.loc[time_idx]

    homology_dimensions = (0, 1)
    VR = Persistence(homology_dimensions=homology_dimensions, n_jobs=-1)
    diagrams = VR.fit_transform(price_embedded_windows)

    # Landscape distance between diagrams obtained from two successive windows as ``Homological derivatives''
    landscape_hom_der = HomologicalDerivative(
        metric="landscape", metric_params=metric_params, order=2, n_jobs=-1
    )
    landscape_succ_dists = landscape_hom_der.fit_transform(diagrams)

    # $l^p$ norm of the Betti curves between diagrams obtained from two successive windows as ``Homological derivatives''
    metric_params = {"p": 2, "n_bins": 1000}
    bettiHomDer = HomologicalDerivative(
        metric='betti', metric_params=metric_params, order=2, n_jobs=-1
    )
    betti_succ_dists = bettiHomDer.fit_transform(diagrams)

    return time_idx, resampled_price, landscape_succ_dists, betti_succ_dists


# Parameters
embedding_dimension = 3
embedding_time_delay = 2
swdft = False
window_size = 31
window_stride = 4
Persistence = hl.VietorisRipsPersistence
distance = 'landscape'
metric_params = {"p": 2, "n_layers": 10, "n_bins": 1000}
threshold = 0.3


time_idx, resampled_price, landscape_succ_dists, betti_succ_dists = detect_crash(
        '^GSPC',
        '1980',
        embedding_dimension,
        embedding_time_delay,
        swdft,
        window_size,
        window_stride,
        Persistence,
        metric_params
)
if distance == 'landscape':
    plot_crash_detections(
        start_date="2000-01-01",
        end_date="2100-01-01",
        threshold=threshold,
        distances=landscape_succ_dists,
        time_index_derivs=time_idx,
        price_resampled_derivs=resampled_price,
        metric_name='landscape'
    )
elif distance == 'betti':
    plot_crash_detections(
        start_date="2000-01-01",
        end_date="2100-01-01",
        threshold=threshold,
        distances=betti_succ_dists,
        time_index_derivs=time_idx,
        price_resampled_derivs=resampled_price,
        metric_name='betti'
    )
else:
    raise ValueError(f"Unknown distance type {distance}")
