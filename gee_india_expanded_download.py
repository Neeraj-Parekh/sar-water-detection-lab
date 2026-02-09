#!/usr/bin/env python3
"""
GEE India Chips Download Script - Expanded Coverage
====================================================
Downloads chips from all regions of India including:
- Urban areas (Mumbai, Delhi, Bangalore, Chennai, Kolkata)
- Coastal regions (Kerala backwaters, Sundarbans, Gujarat coast)
- Mountain regions (Himalayan lakes, Kashmir)
- Agricultural areas (Punjab, Haryana, UP)
- Arid regions (Rajasthan, Gujarat Rann)

Run in Google Earth Engine Code Editor or use earthengine-api.

Author: AI Assistant
Date: 2026-01-24
"""

# ============================================================
# CONFIGURATION
# ============================================================

CHIP_SIZE = 512  # pixels
RESOLUTION = 10  # meters (Sentinel-1)
BUFFER_M = CHIP_SIZE * RESOLUTION / 2  # Buffer in meters

# Date ranges
SAR_START = "2023-01-01"
SAR_END = "2023-12-31"

# Export settings
EXPORT_FOLDER = "india_chips_expanded"
CRS = "EPSG:4326"

# ============================================================
# CHIP LOCATIONS - EXPANDED COVERAGE
# ============================================================

CHIP_LOCATIONS = {
    # ==================== URBAN AREAS ====================
    "urban": [
        {"name": "mumbai_harbor", "coords": [72.85, 18.92], "type": "urban_coastal"},
        {"name": "mumbai_thane_creek", "coords": [72.99, 19.18], "type": "urban_creek"},
        {"name": "delhi_yamuna_1", "coords": [77.23, 28.65], "type": "urban_river"},
        {"name": "delhi_yamuna_2", "coords": [77.28, 28.58], "type": "urban_river"},
        {"name": "bangalore_lakes_1", "coords": [77.60, 12.97], "type": "urban_lake"},
        {"name": "bangalore_lakes_2", "coords": [77.58, 13.02], "type": "urban_lake"},
        {"name": "chennai_adyar", "coords": [80.26, 13.00], "type": "urban_estuary"},
        {"name": "chennai_cooum", "coords": [80.28, 13.08], "type": "urban_river"},
        {"name": "kolkata_hooghly_1", "coords": [88.32, 22.55], "type": "urban_river"},
        {"name": "kolkata_hooghly_2", "coords": [88.28, 22.62], "type": "urban_river"},
        {
            "name": "hyderabad_hussain_sagar",
            "coords": [78.47, 17.42],
            "type": "urban_lake",
        },
        {
            "name": "hyderabad_osmansagar",
            "coords": [78.32, 17.37],
            "type": "urban_reservoir",
        },
        {
            "name": "ahmedabad_sabarmati",
            "coords": [72.58, 23.03],
            "type": "urban_river",
        },
        {"name": "pune_mula_mutha", "coords": [73.87, 18.52], "type": "urban_river"},
        {"name": "jaipur_mansagar", "coords": [75.85, 26.95], "type": "urban_lake"},
    ],
    # ==================== COASTAL REGIONS ====================
    "coastal": [
        {"name": "kerala_vembanad_1", "coords": [76.35, 9.60], "type": "backwater"},
        {"name": "kerala_vembanad_2", "coords": [76.30, 9.55], "type": "backwater"},
        {"name": "kerala_alleppey", "coords": [76.32, 9.50], "type": "canal"},
        {"name": "kerala_kuttanad", "coords": [76.43, 9.27], "type": "below_sea_level"},
        {"name": "kerala_ashtamudi", "coords": [76.58, 8.98], "type": "estuary"},
        {"name": "goa_mandovi", "coords": [73.95, 15.50], "type": "estuary"},
        {"name": "goa_zuari", "coords": [73.90, 15.40], "type": "estuary"},
        {"name": "sundarbans_1", "coords": [88.85, 21.95], "type": "mangrove"},
        {"name": "sundarbans_2", "coords": [88.90, 21.90], "type": "mangrove"},
        {"name": "sundarbans_3", "coords": [88.80, 22.00], "type": "tidal"},
        {"name": "bhitarkanika_1", "coords": [86.85, 20.70], "type": "mangrove"},
        {"name": "bhitarkanika_2", "coords": [86.80, 20.75], "type": "delta"},
        {"name": "pulicat_lake", "coords": [80.23, 13.60], "type": "lagoon"},
        {"name": "chilika_coast", "coords": [85.50, 19.65], "type": "lagoon"},
        {"name": "pichavaram", "coords": [79.78, 11.43], "type": "mangrove"},
        {"name": "coringa", "coords": [82.25, 16.75], "type": "delta"},
        {"name": "gujarat_gulf_khambhat", "coords": [72.25, 21.50], "type": "gulf"},
        {"name": "gujarat_gulf_kutch", "coords": [69.50, 22.50], "type": "gulf"},
    ],
    # ==================== MOUNTAIN/HIGH ELEVATION ====================
    "mountain": [
        {
            "name": "pangong_tso_1",
            "coords": [78.70, 33.75],
            "type": "high_altitude_lake",
        },
        {
            "name": "pangong_tso_2",
            "coords": [78.85, 33.80],
            "type": "high_altitude_lake",
        },
        {"name": "tso_moriri", "coords": [78.30, 32.90], "type": "high_altitude_lake"},
        {"name": "dal_lake_1", "coords": [74.87, 34.10], "type": "mountain_lake"},
        {"name": "dal_lake_2", "coords": [74.85, 34.12], "type": "mountain_lake"},
        {"name": "wular_lake_1", "coords": [74.55, 34.35], "type": "mountain_lake"},
        {"name": "wular_lake_2", "coords": [74.58, 34.32], "type": "mountain_lake"},
        {"name": "nainital", "coords": [79.46, 29.38], "type": "hill_lake"},
        {"name": "bhimtal", "coords": [79.55, 29.35], "type": "hill_lake"},
        {"name": "loktak_1", "coords": [93.78, 24.55], "type": "floating_island"},
        {"name": "loktak_2", "coords": [93.82, 24.50], "type": "floating_island"},
    ],
    # ==================== LARGE RIVERS ====================
    "rivers": [
        {"name": "ganga_varanasi_1", "coords": [83.00, 25.30], "type": "wide_river"},
        {"name": "ganga_varanasi_2", "coords": [82.95, 25.32], "type": "wide_river"},
        {"name": "ganga_patna", "coords": [85.15, 25.60], "type": "wide_river"},
        {
            "name": "brahmaputra_guwahati_1",
            "coords": [91.75, 26.18],
            "type": "wide_river",
        },
        {
            "name": "brahmaputra_guwahati_2",
            "coords": [91.80, 26.15],
            "type": "wide_river",
        },
        {
            "name": "brahmaputra_majuli_1",
            "coords": [94.20, 26.95],
            "type": "river_island",
        },
        {
            "name": "brahmaputra_majuli_2",
            "coords": [94.15, 26.90],
            "type": "river_island",
        },
        {
            "name": "godavari_rajahmundry",
            "coords": [81.78, 17.00],
            "type": "delta_river",
        },
        {"name": "krishna_vijayawada", "coords": [80.62, 16.50], "type": "wide_river"},
        {"name": "narmada_bharuch", "coords": [73.00, 21.70], "type": "wide_river"},
        {"name": "mahanadi_cuttack", "coords": [85.88, 20.47], "type": "wide_river"},
        {
            "name": "cauvery_srirangapatna",
            "coords": [76.68, 12.42],
            "type": "river_island",
        },
        {"name": "chambal_1", "coords": [78.25, 26.50], "type": "ravine_river"},
        {"name": "chambal_2", "coords": [78.30, 26.45], "type": "ravine_river"},
    ],
    # ==================== RESERVOIRS & DAMS ====================
    "reservoirs": [
        {
            "name": "nagarjuna_sagar_1",
            "coords": [79.30, 16.57],
            "type": "large_reservoir",
        },
        {
            "name": "nagarjuna_sagar_2",
            "coords": [79.35, 16.55],
            "type": "large_reservoir",
        },
        {"name": "hirakud_1", "coords": [83.87, 21.52], "type": "large_reservoir"},
        {"name": "hirakud_2", "coords": [83.82, 21.55], "type": "large_reservoir"},
        {"name": "bhakra_1", "coords": [76.43, 31.42], "type": "mountain_reservoir"},
        {"name": "bhakra_2", "coords": [76.45, 31.40], "type": "mountain_reservoir"},
        {"name": "tungabhadra", "coords": [76.33, 15.27], "type": "reservoir"},
        {"name": "krishna_raja_sagar", "coords": [76.57, 12.42], "type": "reservoir"},
        {"name": "stanley_reservoir", "coords": [77.78, 11.85], "type": "reservoir"},
        {"name": "ujjani", "coords": [75.12, 18.05], "type": "reservoir"},
        {"name": "pench", "coords": [79.20, 21.65], "type": "forest_reservoir"},
    ],
    # ==================== WETLANDS & FLOODPLAINS ====================
    "wetlands": [
        {"name": "keoladeo_1", "coords": [77.52, 27.17], "type": "bird_sanctuary"},
        {"name": "keoladeo_2", "coords": [77.54, 27.15], "type": "bird_sanctuary"},
        {"name": "kolleru_1", "coords": [81.20, 16.60], "type": "freshwater_lake"},
        {"name": "kolleru_2", "coords": [81.25, 16.58], "type": "freshwater_lake"},
        {"name": "harike_1", "coords": [74.95, 31.17], "type": "confluence_wetland"},
        {"name": "harike_2", "coords": [74.98, 31.15], "type": "confluence_wetland"},
        {"name": "deepor_beel", "coords": [91.65, 26.13], "type": "ramsar"},
        {
            "name": "east_kolkata_wetlands",
            "coords": [88.42, 22.55],
            "type": "urban_wetland",
        },
        {"name": "point_calimere", "coords": [79.85, 10.30], "type": "coastal_wetland"},
        {"name": "chilika_wetland_1", "coords": [85.35, 19.70], "type": "lagoon"},
        {"name": "chilika_wetland_2", "coords": [85.40, 19.75], "type": "lagoon"},
        {"name": "vembanad_kol", "coords": [76.38, 9.58], "type": "paddy_wetland"},
    ],
    # ==================== ARID/SEMI-ARID ====================
    "arid": [
        {"name": "rann_kutch_1", "coords": [69.85, 23.85], "type": "salt_marsh"},
        {"name": "rann_kutch_2", "coords": [70.00, 23.80], "type": "salt_marsh"},
        {"name": "rann_kutch_3", "coords": [69.70, 24.00], "type": "salt_marsh"},
        {"name": "sambhar_lake_1", "coords": [75.00, 26.92], "type": "salt_lake"},
        {"name": "sambhar_lake_2", "coords": [74.95, 26.95], "type": "salt_lake"},
        {"name": "thar_seasonal_1", "coords": [71.00, 27.00], "type": "ephemeral"},
        {"name": "thar_seasonal_2", "coords": [71.50, 26.50], "type": "ephemeral"},
        {"name": "deccan_tank_1", "coords": [77.50, 15.50], "type": "irrigation_tank"},
        {"name": "deccan_tank_2", "coords": [77.00, 16.00], "type": "irrigation_tank"},
        {"name": "nalsarovar", "coords": [72.02, 22.78], "type": "seasonal_wetland"},
    ],
    # ==================== FLOOD PRONE AREAS ====================
    "flood_prone": [
        {"name": "kosi_flood_1", "coords": [87.02, 25.98], "type": "flood_zone"},
        {"name": "kosi_flood_2", "coords": [86.95, 26.02], "type": "flood_zone"},
        {"name": "gandak_flood", "coords": [84.95, 26.10], "type": "flood_zone"},
        {"name": "assam_flood_1", "coords": [91.70, 26.10], "type": "floodplain"},
        {"name": "assam_flood_2", "coords": [91.65, 26.05], "type": "floodplain"},
        {"name": "bihar_flood_1", "coords": [85.50, 25.80], "type": "floodplain"},
        {"name": "bihar_flood_2", "coords": [85.55, 25.75], "type": "floodplain"},
        {"name": "west_bengal_flood", "coords": [88.40, 24.00], "type": "floodplain"},
    ],
}

# ============================================================
# JAVASCRIPT CODE FOR GEE CODE EDITOR
# ============================================================

GEE_JAVASCRIPT = """
// ============================================================
// India Expanded Chips - GEE Export Script
// ============================================================
// Paste this in GEE Code Editor: https://code.earthengine.google.com/

// Configuration
var CHIP_SIZE = 512;
var RESOLUTION = 10;
var START_DATE = '2023-01-01';
var END_DATE = '2023-12-31';
var EXPORT_FOLDER = 'india_chips_expanded';

// ============================================================
// CHIP LOCATIONS (copy from Python config above)
// ============================================================

var locations = [
  // URBAN
  {name: 'mumbai_harbor', lon: 72.85, lat: 18.92, type: 'urban'},
  {name: 'delhi_yamuna', lon: 77.23, lat: 28.65, type: 'urban'},
  {name: 'bangalore_lakes', lon: 77.60, lat: 12.97, type: 'urban'},
  {name: 'chennai_adyar', lon: 80.26, lat: 13.00, type: 'urban'},
  {name: 'kolkata_hooghly', lon: 88.32, lat: 22.55, type: 'urban'},
  {name: 'hyderabad_hussain_sagar', lon: 78.47, lat: 17.42, type: 'urban'},
  
  // COASTAL
  {name: 'kerala_vembanad', lon: 76.35, lat: 9.60, type: 'coastal'},
  {name: 'kerala_alleppey', lon: 76.32, lat: 9.50, type: 'coastal'},
  {name: 'sundarbans', lon: 88.85, lat: 21.95, type: 'coastal'},
  {name: 'goa_mandovi', lon: 73.95, lat: 15.50, type: 'coastal'},
  {name: 'pulicat_lake', lon: 80.23, lat: 13.60, type: 'coastal'},
  
  // MOUNTAIN
  {name: 'pangong_tso', lon: 78.70, lat: 33.75, type: 'mountain'},
  {name: 'dal_lake', lon: 74.87, lat: 34.10, type: 'mountain'},
  {name: 'wular_lake', lon: 74.55, lat: 34.35, type: 'mountain'},
  {name: 'loktak', lon: 93.78, lat: 24.55, type: 'mountain'},
  
  // RIVERS
  {name: 'ganga_varanasi', lon: 83.00, lat: 25.30, type: 'river'},
  {name: 'brahmaputra_guwahati', lon: 91.75, lat: 26.18, type: 'river'},
  {name: 'godavari', lon: 81.78, lat: 17.00, type: 'river'},
  {name: 'narmada', lon: 73.00, lat: 21.70, type: 'river'},
  
  // RESERVOIRS
  {name: 'nagarjuna_sagar', lon: 79.30, lat: 16.57, type: 'reservoir'},
  {name: 'hirakud', lon: 83.87, lat: 21.52, type: 'reservoir'},
  {name: 'bhakra', lon: 76.43, lat: 31.42, type: 'reservoir'},
  
  // WETLANDS
  {name: 'keoladeo', lon: 77.52, lat: 27.17, type: 'wetland'},
  {name: 'chilika', lon: 85.35, lat: 19.70, type: 'wetland'},
  {name: 'kolleru', lon: 81.20, lat: 16.60, type: 'wetland'},
  
  // ARID
  {name: 'rann_kutch', lon: 69.85, lat: 23.85, type: 'arid'},
  {name: 'sambhar_lake', lon: 75.00, lat: 26.92, type: 'arid'},
  
  // FLOOD PRONE
  {name: 'kosi_flood', lon: 87.02, lat: 25.98, type: 'flood'},
  {name: 'assam_flood', lon: 91.70, lat: 26.10, type: 'flood'},
];

// ============================================================
// FUNCTIONS
// ============================================================

function getChipBounds(lon, lat) {
  var buffer = CHIP_SIZE * RESOLUTION / 2;
  var point = ee.Geometry.Point([lon, lat]);
  return point.buffer(buffer).bounds();
}

function getSentinel1Composite(bounds) {
  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(bounds)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .select(['VV', 'VH']);
  
  return s1.median().clip(bounds);
}

function getDEM(bounds) {
  var srtm = ee.Image('USGS/SRTMGL1_003').select('elevation');
  var slope = ee.Terrain.slope(srtm);
  
  return srtm.addBands(slope).clip(bounds);
}

function getHAND(bounds) {
  // HAND from MERIT Hydro
  var hand = ee.Image('MERIT/Hydro/v1_0_1')
    .select('hnd')
    .rename('HAND');
  
  return hand.clip(bounds);
}

function getTWI(bounds) {
  var srtm = ee.Image('USGS/SRTMGL1_003');
  var slope = ee.Terrain.slope(srtm);
  
  // Flow accumulation proxy
  var flowAcc = ee.Image('WWF/HydroSHEDS/15ACC');
  
  // TWI = ln(a / tan(b))
  var slopeRad = slope.multiply(Math.PI).divide(180);
  var tanSlope = slopeRad.tan().max(0.01);  // Avoid division by zero
  var twi = flowAcc.log().divide(tanSlope).rename('TWI');
  
  return twi.clip(bounds);
}

function getMNDWI(bounds) {
  // Use Sentinel-2 for MNDWI
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(bounds)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .select(['B3', 'B11']);  // Green, SWIR
  
  var composite = s2.median();
  var mndwi = composite.normalizedDifference(['B3', 'B11']).rename('MNDWI');
  
  return mndwi.clip(bounds);
}

function getJRCWater(bounds) {
  var jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    .select('occurrence')
    .divide(100)
    .rename('JRC_water');
  
  return jrc.clip(bounds);
}

function exportChip(location, index) {
  var bounds = getChipBounds(location.lon, location.lat);
  
  // Get all layers
  var s1 = getSentinel1Composite(bounds);
  var dem = getDEM(bounds);
  var hand = getHAND(bounds);
  var twi = getTWI(bounds);
  var mndwi = getMNDWI(bounds);
  var jrc = getJRCWater(bounds);
  
  // Stack all bands
  var stack = s1
    .addBands(mndwi)
    .addBands(dem.select('elevation').rename('DEM'))
    .addBands(hand)
    .addBands(dem.select('slope').rename('SLOPE'))
    .addBands(twi)
    .addBands(jrc);
  
  // Export
  var filename = 'chip_' + String(index + 1).padStart(3, '0') + '_' + location.name;
  
  Export.image.toDrive({
    image: stack,
    description: filename,
    folder: EXPORT_FOLDER,
    fileNamePrefix: filename,
    region: bounds,
    scale: RESOLUTION,
    crs: 'EPSG:4326',
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF'
  });
  
  print('Queued export: ' + filename);
}

// ============================================================
// MAIN EXECUTION
// ============================================================

print('India Expanded Chips Export');
print('Total locations: ' + locations.length);

// Export all chips
for (var i = 0; i < locations.length; i++) {
  exportChip(locations[i], i);
}

print('All exports queued. Go to Tasks tab to run them.');

// Visualize one sample
var sampleBounds = getChipBounds(locations[0].lon, locations[0].lat);
var sampleS1 = getSentinel1Composite(sampleBounds);
Map.centerObject(sampleBounds, 12);
Map.addLayer(sampleS1.select('VV'), {min: -25, max: 0}, 'VV');
Map.addLayer(sampleS1.select('VH'), {min: -30, max: -5}, 'VH');
"""

# ============================================================
# PYTHON SCRIPT TO GENERATE FULL GEE SCRIPT
# ============================================================


def generate_gee_script():
    """Generate complete GEE JavaScript code with all locations."""

    # Flatten all locations
    all_locations = []
    chip_id = 1

    for category, locs in CHIP_LOCATIONS.items():
        for loc in locs:
            all_locations.append(
                {
                    "id": chip_id,
                    "name": f"{loc['name']}",
                    "lon": loc["coords"][0],
                    "lat": loc["coords"][1],
                    "category": category,
                    "type": loc["type"],
                }
            )
            chip_id += 1

    print(f"Total chip locations: {len(all_locations)}")
    print("\nBreakdown by category:")
    for cat in CHIP_LOCATIONS:
        print(f"  {cat}: {len(CHIP_LOCATIONS[cat])} chips")

    # Generate JavaScript array
    js_array = "var locations = [\n"
    for loc in all_locations:
        js_array += f"  {{id: {loc['id']}, name: '{loc['name']}', lon: {loc['lon']}, lat: {loc['lat']}, category: '{loc['category']}', type: '{loc['type']}'}},\n"
    js_array += "];\n"

    return js_array, all_locations


def main():
    """Generate and save GEE export script."""
    js_array, locations = generate_gee_script()

    # Save to file
    with open(
        "/media/neeraj-parekh/Data1/sar soil system/chips/gui/gee_india_expanded.js",
        "w",
    ) as f:
        f.write("// Auto-generated GEE script for India expanded coverage\n")
        f.write(f"// Total chips: {len(locations)}\n")
        f.write(f"// Generated: 2026-01-24\n\n")
        f.write(js_array)
        f.write("\n")
        f.write(GEE_JAVASCRIPT)

    print(f"\nSaved GEE script to: gee_india_expanded.js")
    print(f"Run in GEE Code Editor: https://code.earthengine.google.com/")

    # Also save locations as JSON
    import json

    with open(
        "/media/neeraj-parekh/Data1/sar soil system/chips/gui/india_chip_locations.json",
        "w",
    ) as f:
        json.dump(
            {
                "total_chips": len(locations),
                "categories": list(CHIP_LOCATIONS.keys()),
                "locations": locations,
            },
            f,
            indent=2,
        )

    print(f"Saved locations JSON to: india_chip_locations.json")


if __name__ == "__main__":
    main()
