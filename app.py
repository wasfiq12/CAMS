import streamlit as st
import ee
import folium
import streamlit.components.v1 as components
from google.oauth2 import service_account

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Estimasi PM2.5 Bulan Agustus 2026 Pulau Belitung",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Estimasi PM2.5 – Belitung")

# ============================================================
# KONFIGURASI TETAP
# ============================================================

PROJECT_ID = "wasfiq12"
AOI_ASSET = "projects/wasfiq12/assets/Belitung"

START_DATE = "2026-08-01"
END_DATE_EXCLUSIVE = "2026-09-01"

PROCESSING_SCALE = 1000
RF_TREES = 200
RF_SEED = 42

# ============================================================
# EARTH ENGINE
# ============================================================

@st.cache_resource(show_spinner=False)
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
        project=PROJECT_ID
    )

    return True


try:
    initialize_earth_engine()
except Exception as e:
    st.error("Earth Engine gagal diinisialisasi.")
    st.code(str(e))
    st.stop()


# ============================================================
# FUNGSI UTAMA - DI-CACHE
#
# Seluruh proses berat Earth Engine berada di dalam satu cache.
# Setelah proses pertama selesai, rerun Streamlit tidak akan
# mengulang CAMS/S5P/MODIS/RF/reduceRegion/getMapId.
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def build_pm25_dashboard():

    # --------------------------------------------------------
    # AOI
    # --------------------------------------------------------

    aoi = ee.FeatureCollection(AOI_ASSET)

    # Tidak perlu aoi.size().getInfo().
    # Asset langsung digunakan sehingga satu request EE dihilangkan.

    # --------------------------------------------------------
    # CAMS PM2.5
    # --------------------------------------------------------

    cams = (
        ee.ImageCollection("ECMWF/CAMS/NRT")
        .filterDate(START_DATE, END_DATE_EXCLUSIVE)
        .filterBounds(aoi)
        .select("particulate_matter_d_less_than_25_um_surface")
    )

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

    # --------------------------------------------------------
    # SENTINEL-5P NO2
    # --------------------------------------------------------

    no2 = (
        ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
        .filterDate(START_DATE, END_DATE_EXCLUSIVE)
        .filterBounds(aoi)
        .select("tropospheric_NO2_column_number_density")
        .mean()
        .rename("NO2")
        .clip(aoi)
    )

    # --------------------------------------------------------
    # SENTINEL-5P CO
    # --------------------------------------------------------

    co = (
        ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CO")
        .filterDate(START_DATE, END_DATE_EXCLUSIVE)
        .filterBounds(aoi)
        .select("CO_column_number_density")
        .mean()
        .rename("CO")
        .clip(aoi)
    )

    # --------------------------------------------------------
    # SENTINEL-5P SO2
    # --------------------------------------------------------

    so2 = (
        ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_SO2")
        .filterDate(START_DATE, END_DATE_EXCLUSIVE)
        .filterBounds(aoi)
        .select("SO2_column_number_density")
        .mean()
        .rename("SO2")
        .clip(aoi)
    )

    # --------------------------------------------------------
    # MODIS AOD
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MODIS NDVI
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SRTM
    # --------------------------------------------------------

    elevation = (
        ee.Image("USGS/SRTMGL1_003")
        .select("elevation")
        .rename("Elevation")
        .clip(aoi)
    )

    # --------------------------------------------------------
    # PREDICTORS
    # --------------------------------------------------------

    predictors = (
        cams_mean
        .addBands(no2)
        .addBands(co)
        .addBands(so2)
        .addBands(aod)
        .addBands(ndvi)
        .addBands(elevation)
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    training = predictors.sample(
        region=aoi.geometry(),
        scale=PROCESSING_SCALE,
        numPixels=10000,
        seed=RF_SEED,
        geometries=False
    )

    rf = ee.Classifier.smileRandomForest(
        numberOfTrees=RF_TREES,
        seed=RF_SEED
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

    # --------------------------------------------------------
    # DOWNSCALED PM2.5
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RENTANG VISUALISASI
    # Min/Max tetap dihitung hanya untuk skala warna peta.
    # Tidak ditampilkan sebagai statistik dashboard.
    # --------------------------------------------------------

    stats = downscaled_pm25.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=aoi.geometry(),
        scale=PROCESSING_SCALE,
        maxPixels=1e13,
        bestEffort=True
    ).getInfo()

    pm_min = stats.get("PM25_DOWNSCALED_min")
    pm_max = stats.get("PM25_DOWNSCALED_max")

    if pm_min is None or pm_max is None:
        raise RuntimeError("Rentang PM2.5 tidak berhasil dihitung.")

    data_range = pm_max - pm_min

    if data_range == 0:
        data_range = 1

    vis_min = max(0, pm_min - data_range * 0.05)
    vis_max = pm_max + data_range * 0.05

    # --------------------------------------------------------
    # CENTER
    # Hindari getInfo centroid: pusat Belitung dibuat tetap.
    # --------------------------------------------------------

    center_lat = -2.8700
    center_lon = 107.9200

    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=None,
        control_scale=True
    )

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Hybrid",
        name="Google Hybrid",
        overlay=False,
        control=True
    ).add_to(m)

    palette = [
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

    # --------------------------------------------------------
    # DOWNSCALED MAP
    # --------------------------------------------------------

    downscaled_vis = {
        "min": vis_min,
        "max": vis_max,
        "palette": palette
    }

    downscaled_mapid = downscaled_pm25.getMapId(downscaled_vis)

    folium.TileLayer(
        tiles=downscaled_mapid["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name="PM2.5 Resolusi 1 km",
        overlay=True,
        control=True,
        opacity=0.80
    ).add_to(m)

    # --------------------------------------------------------
    # AOI
    #
    # AOI GeoJSON dibuat hanya sekali di cache.
    # --------------------------------------------------------

    aoi_geojson = aoi.geometry().getInfo()

    folium.GeoJson(
        aoi_geojson,
        name="Belitung",
        style_function=lambda feature: {
            "fillColor": "none",
            "color": "black",
            "weight": 2,
            "fillOpacity": 0
        }
    ).add_to(m)

    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

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
    PM2.5 Downscaled 1 km
    </b>

    <br>

    <span style="font-size:12px;">
    Rata-rata 01–31 Agustus 2026
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
    <span>{vis_max:.1f} µg/m³</span>

    </div>

    </div>
    """

    m.get_root().html.add_child(
        folium.Element(legend_html)
    )

    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    # --------------------------------------------------------
    # RENDER MAP SEKALI
    # --------------------------------------------------------

    map_html = m.get_root().render()

    return map_html


# ============================================================
# TAMPILKAN HASIL
# ============================================================

try:
    map_html = build_pm25_dashboard()

except Exception as e:
    st.error("Gagal memproses peta PM2.5.")
    st.code(str(e))
    st.stop()


# ============================================================
# PETA
# ============================================================

components.html(
    map_html,
    height=720,
    scrolling=False
)
