import streamlit as st
import ee
import folium
import streamlit.components.v1 as components
from google.oauth2 import service_account
from datetime import date

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Estimasi PM2.5 Bulan Agustus 2026 Pulau Belitung ",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Estimasi PM2.5 – Belitung")
st.caption(
    "CAMS PM2.5 + Sentinel-5P + MODIS + SRTM | "
    "Random Forest spatial downscaling prototype"
)

# ============================================================
# EARTH ENGINE
# ============================================================

@st.cache_resource
def initialize_earth_engine():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/cloud-platform"
        ]
    )

    ee.Initialize(
        credentials=credentials,
        project="wasfiq12"
    )

    return True


try:
    initialize_earth_engine()
except Exception as e:
    st.error("Earth Engine gagal diinisialisasi.")
    st.code(str(e))
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Pengaturan")

start_date = st.sidebar.date_input(
    "Tanggal mulai",
    value=date(2026, 8, 1)
)

end_date = st.sidebar.date_input(
    "Tanggal akhir",
    value=date(2026, 8, 31)
)

scale = st.sidebar.selectbox(
    "Resolusi pemrosesan",
    [1000, 2000, 5000],
    index=0,
    format_func=lambda x: f"{x/1000:g} km"
)

if start_date >= end_date:
    st.sidebar.error("Tanggal akhir harus lebih besar dari tanggal mulai.")
    st.stop()

START_DATE = start_date.isoformat()
# Earth Engine filterDate memakai end-exclusive.
# Tambahkan satu hari supaya tanggal akhir ikut masuk.
END_DATE_EXCLUSIVE = (
    end_date.fromordinal(end_date.toordinal() + 1)
).isoformat()

# ============================================================
# AOI - SESUAI IPYNB
# ============================================================

aoi = ee.FeatureCollection(
    "projects/wasfiq12/assets/Belitung"
)

if aoi.size().getInfo() == 0:
    st.error("Asset AOI Belitung tidak ditemukan atau tidak dapat diakses.")
    st.stop()

# ============================================================
# CAMS PM2.5
# ============================================================

cams = (
    ee.ImageCollection("ECMWF/CAMS/NRT")
    .filterDate(START_DATE, END_DATE_EXCLUSIVE)
    .filterBounds(aoi)
    .select("particulate_matter_d_less_than_25_um_surface")
)

cams_count = cams.size().getInfo()

if cams_count == 0:
    st.error("Tidak ada image CAMS pada periode yang dipilih.")
    st.stop()


def cams_to_ugm3(img):
    return (
        img
        .multiply(1e9)
        .rename("CAMS_PM25")
        .copyProperties(img, img.propertyNames())
    )


cams_pm25 = cams.map(cams_to_ugm3)

cams_mean = (
    cams_pm25
    .mean()
    .rename("CAMS_PM25")
    .clip(aoi)
)

# ============================================================
# SENTINEL-5P NO2
# ============================================================

no2 = (
    ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
    .filterDate(START_DATE, END_DATE_EXCLUSIVE)
    .filterBounds(aoi)
    .select("tropospheric_NO2_column_number_density")
    .mean()
    .rename("NO2")
    .clip(aoi)
)

# ============================================================
# SENTINEL-5P CO
# ============================================================

co = (
    ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CO")
    .filterDate(START_DATE, END_DATE_EXCLUSIVE)
    .filterBounds(aoi)
    .select("CO_column_number_density")
    .mean()
    .rename("CO")
    .clip(aoi)
)

# ============================================================
# SENTINEL-5P SO2
# ============================================================

so2 = (
    ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_SO2")
    .filterDate(START_DATE, END_DATE_EXCLUSIVE)
    .filterBounds(aoi)
    .select("SO2_column_number_density")
    .mean()
    .rename("SO2")
    .clip(aoi)
)

# ============================================================
# MODIS AOD
# ============================================================

aod = (
    ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
    .filterDate(START_DATE, END_DATE_EXCLUSIVE)
    .filterBounds(aoi)
    .select("Optical_Depth_047")
    .mean()
    .multiply(0.001)
    .rename("AOD")
    .clip(aoi)
)

# ============================================================
# MODIS NDVI
# ============================================================

ndvi = (
    ee.ImageCollection("MODIS/061/MOD13Q1")
    .filterDate(START_DATE, END_DATE_EXCLUSIVE)
    .filterBounds(aoi)
    .select("NDVI")
    .mean()
    .multiply(0.0001)
    .rename("NDVI")
    .clip(aoi)
)

# ============================================================
# SRTM ELEVATION
# ============================================================

elevation = (
    ee.Image("USGS/SRTMGL1_003")
    .select("elevation")
    .rename("Elevation")
    .clip(aoi)
)

# ============================================================
# PREDICTORS - SESUAI IPYNB
# ============================================================

predictors = (
    cams_mean
    .addBands(no2)
    .addBands(co)
    .addBands(so2)
    .addBands(aod)
    .addBands(ndvi)
    .addBands(elevation)
)

# ============================================================
# RANDOM FOREST - SESUAI IPYNB
# ============================================================

training = predictors.sample(
    region=aoi.geometry(),
    scale=1000,
    numPixels=10000,
    seed=42,
    geometries=False
)

rf = ee.Classifier.smileRandomForest(
    numberOfTrees=200,
    seed=42
).setOutputMode("REGRESSION")

rf_model = rf.train(
    features=training,
    classProperty="CAMS_PM25",
    inputProperties=[
        "NO2",
        "CO",
        "SO2",
        "AOD",
        "NDVI",
        "Elevation"
    ]
)

# ============================================================
# DOWNSCALED PM2.5 - SESUAI IPYNB
# ============================================================

downscaled_pm25 = (
    predictors
    .select([
        "NO2",
        "CO",
        "SO2",
        "AOD",
        "NDVI",
        "Elevation"
    ])
    .classify(rf_model)
    .rename("PM25_DOWNSCALED")
    .clip(aoi)
)

# ============================================================
# STATISTIK PM2.5
# ============================================================

stats = downscaled_pm25.reduceRegion(
    reducer=(
        ee.Reducer.minMax()
        .combine(
            reducer2=ee.Reducer.mean(),
            sharedInputs=True
        )
        .combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True
        )
    ),
    geometry=aoi.geometry(),
    scale=1000,
    maxPixels=1e13,
    bestEffort=True
).getInfo()

pm_min = stats.get("PM25_DOWNSCALED_min")
pm_max = stats.get("PM25_DOWNSCALED_max")
pm_mean = stats.get("PM25_DOWNSCALED_mean")
pm_std = stats.get("PM25_DOWNSCALED_stdDev")

if None in [pm_min, pm_max, pm_mean, pm_std]:
    st.error("Statistik PM2.5 tidak berhasil dihitung.")
    st.stop()

data_range = pm_max - pm_min

if data_range == 0:
    data_range = 1

vis_min = max(0, pm_min - data_range * 0.05)
vis_max = pm_max + data_range * 0.05

# ============================================================
# CAMS STATISTIK
# ============================================================

cams_stats = cams_mean.reduceRegion(
    reducer=(
        ee.Reducer.minMax()
        .combine(
            reducer2=ee.Reducer.mean(),
            sharedInputs=True
        )
    ),
    geometry=aoi.geometry(),
    scale=45000,
    maxPixels=1e13,
    bestEffort=True
).getInfo()

cams_min = cams_stats.get("CAMS_PM25_min")
cams_max = cams_stats.get("CAMS_PM25_max")
cams_mean_value = cams_stats.get("CAMS_PM25_mean")

# ============================================================
# MAP CENTER
# ============================================================

center = aoi.geometry().centroid().coordinates().getInfo()
center_lon = center[0]
center_lat = center[1]

# ============================================================
# FOLIUM - TANPA OSM
# ============================================================

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=9,
    tiles=None,
    control_scale=True
)

# Google Satellite
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attr="Google Satellite",
    name="Google Satellite",
    overlay=False,
    control=True
).add_to(m)

# Google Hybrid
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    attr="Google Hybrid",
    name="Google Hybrid",
    overlay=False,
    control=True
).add_to(m)

# ============================================================
# PALETTE
# ============================================================

pm25_palette = [
    "08306B",
    "2171B5",
    "41AB5D",
    "A1D76A",
    "FFFFBF",
    "FEE08B",
    "FDAE61",
    "F46D43",
    "D73027",
    "A50026"
]

# ============================================================
# CAMS LAYER
# ============================================================

if cams_min is not None and cams_max is not None:
    cams_range = cams_max - cams_min

    if cams_range == 0:
        cams_range = 1

    cams_vis = {
        "min": max(0, cams_min - cams_range * 0.05),
        "max": cams_max + cams_range * 0.05,
        "palette": pm25_palette
    }

    cams_mapid = cams_mean.getMapId(cams_vis)

    folium.TileLayer(
        tiles=cams_mapid["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name="CAMS PM2.5",
        overlay=True,
        control=True,
        opacity=0.60
    ).add_to(m)

# ============================================================
# DOWNSCALED LAYER
# ============================================================

downscaled_vis = {
    "min": vis_min,
    "max": vis_max,
    "palette": pm25_palette
}

downscaled_mapid = downscaled_pm25.getMapId(downscaled_vis)

folium.TileLayer(
    tiles=downscaled_mapid["tile_fetcher"].url_format,
    attr="Google Earth Engine",
    name="PM2.5 Downscaled",
    overlay=True,
    control=True,
    opacity=0.80
).add_to(m)

# ============================================================
# AOI
# ============================================================

folium.GeoJson(
    aoi.geometry().getInfo(),
    name="Belitung",
    style_function=lambda feature: {
        "fillColor": "none",
        "color": "black",
        "weight": 2,
        "fillOpacity": 0
    }
).add_to(m)

# ============================================================
# LEGEND - RANGE SESUAI DATA
# ============================================================

legend_html = f"""
<div style="
position: fixed;
bottom: 30px;
left: 30px;
width: 285px;
z-index: 9999;
font-size: 13px;
background-color: white;
border: 2px solid grey;
border-radius: 8px;
padding: 12px;
box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
">

<b style="font-size:15px;">
Estimasi PM2.5 Belitung
</b>

<br>

<span style="font-size:12px;">
{start_date.strftime('%d %b %Y')} – {end_date.strftime('%d %b %Y')}
</span>

<br><br>

<div style="
height:20px;
background: linear-gradient(
to right,
#08306B,
#2171B5,
#41AB5D,
#A1D76A,
#FFFFBF,
#FEE08B,
#FDAE61,
#F46D43,
#D73027,
#A50026
);
border:1px solid #555;
"></div>

<div style="
display:flex;
justify-content:space-between;
font-size:11px;
margin-top:3px;
">

<span>{vis_min:.1f}</span>
<span>{pm_mean:.1f}</span>
<span>{vis_max:.1f}</span>

</div>

<div style="margin-top:8px;">

<b>Min:</b> {pm_min:.2f} µg/m³<br>
<b>Max:</b> {pm_max:.2f} µg/m³<br>
<b>Mean:</b> {pm_mean:.2f} µg/m³<br>
<b>Std:</b> {pm_std:.2f} µg/m³

</div>

</div>
"""

m.get_root().html.add_child(
    folium.Element(legend_html)
)

folium.LayerControl(
    collapsed=False
).add_to(m)

# ============================================================
# DASHBOARD
# ============================================================

st.subheader("📊 Statistik")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "CAMS Mean",
    f"{cams_mean_value:.2f} µg/m³"
    if cams_mean_value is not None else "N/A"
)

c2.metric(
    "Downscaled Mean",
    f"{pm_mean:.2f} µg/m³"
)

c3.metric(
    "Downscaled Min",
    f"{pm_min:.2f} µg/m³"
)

c4.metric(
    "Downscaled Max",
    f"{pm_max:.2f} µg/m³"
)

st.caption(
    f"CAMS images: {cams_count} | "
    f"Resolusi pemrosesan: {scale/1000:g} km"
)

st.subheader("🗺️ Peta")

components.html(
    m.get_root().render(),
    height=720,
    scrolling=False
)

with st.expander("📋 Statistik lengkap"):
    st.write({
        "CAMS minimum (µg/m³)": cams_min,
        "CAMS maximum (µg/m³)": cams_max,
        "CAMS mean (µg/m³)": cams_mean_value,
        "Downscaled minimum (µg/m³)": pm_min,
        "Downscaled maximum (µg/m³)": pm_max,
        "Downscaled mean (µg/m³)": pm_mean,
        "Downscaled std dev (µg/m³)": pm_std
    })

with st.expander("ℹ️ Metodologi"):
    st.markdown(
        """2026 | Wasfi Qordowi
        """
    )
