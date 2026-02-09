/*
 * JRC Global Surface Water Labels Export Script
 * =============================================
 * 
 * Exports SAR chips with HIGH-QUALITY JRC water labels instead of weak MNDWI.
 * JRC has <1% false detection rate (validated with 40,000 points).
 * 
 * Usage: Run in Google Earth Engine Code Editor
 * 
 * Author: SAR Water Detection Lab
 * Date: 2026-01-19
 */

// =============================================================================
// Configuration
// =============================================================================

var EXPORT = {
    folder: 'SAR_JRC_Chips',
    scale: 10,  // Sentinel-1 resolution
    crs: 'EPSG:4326',
    maxPixels: 1e13
};

// Chip locations (same as original)
var CHIPS = [
    // Large Lakes
    { id: 'chip_001', type: 'large_lake', center: [85.45, 19.85], name: 'Chilika' },
    { id: 'chip_002', type: 'large_lake', center: [74.45, 34.10], name: 'Wular' },
    { id: 'chip_003', type: 'large_lake', center: [93.85, 24.55], name: 'Loktak' },

    // Wide Rivers
    { id: 'chip_010', type: 'wide_river', center: [91.80, 26.20], name: 'Brahmaputra' },
    { id: 'chip_011', type: 'wide_river', center: [83.00, 25.35], name: 'Ganga' },
    { id: 'chip_012', type: 'wide_river', center: [80.90, 16.50], name: 'Godavari' },

    // Narrow Rivers
    { id: 'chip_020', type: 'narrow_river', center: [77.10, 28.60], name: 'Yamuna' },
    { id: 'chip_021', type: 'narrow_river', center: [79.50, 23.20], name: 'Narmada' },
    { id: 'chip_022', type: 'narrow_river', center: [77.60, 12.30], name: 'Cauvery' },
];

// =============================================================================
// Data Sources
// =============================================================================

// Sentinel-1 SAR
var S1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .select(['VV', 'VH']);

// JRC Global Surface Water (HIGH QUALITY - <1% false detection)
var JRC = ee.Image('JRC/GSW1_4/GlobalSurfaceWater');

// SRTM DEM
var DEM = ee.Image('USGS/SRTMGL1_003').select('elevation');

// WWF HydroSHEDS for HAND approximation
var HYDRO = ee.Image('WWF/HydroSHEDS/03CONDEM');

// =============================================================================
// Functions
// =============================================================================

function getJRCWaterMask(geometry) {
    /**
     * Get high-quality water mask from JRC GSW.
     * 
     * JRC provides:
     * - max_extent: Water detected at least once 1984-2021
     * - occurrence: Percentage of time water was present
     * - seasonality: Number of months with water
     */

    var maxExtent = JRC.select('max_extent');
    var occurrence = JRC.select('occurrence');
    var seasonality = JRC.select('seasonality');

    // High-confidence water: occurred >50% of valid observations
    var permanentWater = occurrence.gt(50);

    // Seasonal water: present 5+ months/year
    var seasonalWater = seasonality.gte(5);

    // Combined water mask
    var waterMask = permanentWater.or(seasonalWater)
        .clip(geometry)
        .rename('JRC_Water');

    return waterMask;
}

function computeTerrainFeatures(geometry) {
    /**
     * Compute terrain-derived features.
     */
    var elevation = DEM.clip(geometry);
    var slope = ee.Terrain.slope(DEM).clip(geometry);

    // TWI approximation
    var flowAccum = HYDRO.clip(geometry);
    var slopeRad = slope.multiply(Math.PI).divide(180);
    var twi = flowAccum.log().divide(slopeRad.tan().add(0.001));

    // HAND approximation (using flow accumulation)
    // Note: True HAND requires hydrological analysis
    var hand = elevation.subtract(flowAccum.multiply(0.01)).max(0);

    return {
        elevation: elevation.rename('DEM'),
        slope: slope.rename('Slope'),
        twi: twi.rename('TWI'),
        hand: hand.rename('HAND')
    };
}

function exportChip(chip) {
    /**
     * Export a single chip with all features and JRC labels.
     */
    var center = ee.Geometry.Point(chip.center);
    var region = center.buffer(2560).bounds();  // ~5km x 5km

    // Get SAR data (median of recent year)
    var sar = S1
        .filterBounds(region)
        .filterDate('2023-01-01', '2023-12-31')
        .median()
        .clip(region);

    // Get terrain features
    var terrain = computeTerrainFeatures(region);

    // Get JRC water label (HIGH QUALITY)
    var jrcWater = getJRCWaterMask(region);

    // Stack all bands
    var stack = sar
        .addBands(terrain.elevation)
        .addBands(terrain.slope)
        .addBands(terrain.hand)
        .addBands(terrain.twi)
        .addBands(jrcWater);

    // Add metadata
    stack = stack.set({
        'chip_id': chip.id,
        'chip_type': chip.type,
        'chip_name': chip.name,
        'label_source': 'JRC_GSW',
        'label_quality': 'HIGH'
    });

    // Export
    Export.image.toDrive({
        image: stack,
        description: chip.id + '_JRC_8band',
        folder: EXPORT.folder,
        fileNamePrefix: chip.id + '_' + chip.type + '_JRC',
        region: region,
        scale: EXPORT.scale,
        crs: EXPORT.crs,
        maxPixels: EXPORT.maxPixels
    });

    print('Queued:', chip.id, chip.name);
}

// =============================================================================
// Main Execution
// =============================================================================

print('=== JRC Global Surface Water Export ===');
print('Total chips:', CHIPS.length);
print('Label source: JRC GSW (<1% false detection)');
print('');

// Export all chips
CHIPS.forEach(exportChip);

print('');
print('All exports queued. Check Tasks tab to run.');
print('Expected bands: VV, VH, DEM, Slope, HAND, TWI, JRC_Water');
