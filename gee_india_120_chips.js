/*
 * ENHANCED INDIA-FOCUSED SAR WATER CHIP EXPORT
 * ==============================================
 * 
 * 120 Total Chips (90 existing + 30 new):
 *   - Coastal/Mangroves (10)
 *   - Backwaters/Canals (10)
 *   - High-Elevation/Floods (10)
 * 
 * With JRC Global Surface Water labels (< 1% error rate)
 * 
 * Author: SAR Water Detection Lab
 * Date: 2026-01-23
 */

// ============================================================
// CHIP DEFINITIONS - 120 CHIPS ACROSS 9 WATER TYPES
// ============================================================

var chipDefinitions = [
    // LARGE LAKES (15 chips) - EXISTING
    { id: 'chip_001', type: 'large_lakes', center: [85.32, 19.70], name: 'Chilika' },
    { id: 'chip_002', type: 'large_lakes', center: [85.25, 19.65], name: 'Chilika' },
    { id: 'chip_003', type: 'large_lakes', center: [85.40, 19.75], name: 'Chilika' },
    { id: 'chip_004', type: 'large_lakes', center: [85.28, 19.80], name: 'Chilika' },
    { id: 'chip_005', type: 'large_lakes', center: [85.45, 19.68], name: 'Chilika' },
    { id: 'chip_006', type: 'large_lakes', center: [74.60, 34.40], name: 'Wular' },
    { id: 'chip_007', type: 'large_lakes', center: [74.55, 34.35], name: 'Wular' },
    { id: 'chip_008', type: 'large_lakes', center: [74.65, 34.45], name: 'Wular' },
    { id: 'chip_009', type: 'large_lakes', center: [74.58, 34.43], name: 'Wular' },
    { id: 'chip_010', type: 'large_lakes', center: [74.63, 34.38], name: 'Wular' },
    { id: 'chip_011', type: 'large_lakes', center: [93.85, 24.55], name: 'Loktak' },
    { id: 'chip_012', type: 'large_lakes', center: [93.80, 24.50], name: 'Loktak' },
    { id: 'chip_013', type: 'large_lakes', center: [93.90, 24.60], name: 'Loktak' },
    { id: 'chip_014', type: 'large_lakes', center: [93.83, 24.58], name: 'Loktak' },
    { id: 'chip_015', type: 'large_lakes', center: [93.88, 24.53], name: 'Loktak' },

    // WIDE RIVERS (15 chips) - EXISTING
    { id: 'chip_016', type: 'rivers_wide', center: [91.90, 26.25], name: 'Brahmaputra' },
    { id: 'chip_017', type: 'rivers_wide', center: [91.85, 26.20], name: 'Brahmaputra' },
    { id: 'chip_018', type: 'rivers_wide', center: [91.95, 26.30], name: 'Brahmaputra' },
    { id: 'chip_019', type: 'rivers_wide', center: [92.00, 26.28], name: 'Brahmaputra' },
    { id: 'chip_020', type: 'rivers_wide', center: [91.88, 26.23], name: 'Brahmaputra' },
    { id: 'chip_021', type: 'rivers_wide', center: [83.00, 25.35], name: 'Ganga' },
    { id: 'chip_022', type: 'rivers_wide', center: [82.95, 25.30], name: 'Ganga' },
    { id: 'chip_023', type: 'rivers_wide', center: [83.05, 25.40], name: 'Ganga' },
    { id: 'chip_024', type: 'rivers_wide', center: [83.03, 25.33], name: 'Ganga' },
    { id: 'chip_025', type: 'rivers_wide', center: [82.98, 25.38], name: 'Ganga' },
    { id: 'chip_026', type: 'rivers_wide', center: [81.80, 16.75], name: 'Godavari' },
    { id: 'chip_027', type: 'rivers_wide', center: [81.75, 16.70], name: 'Godavari' },
    { id: 'chip_028', type: 'rivers_wide', center: [81.85, 16.80], name: 'Godavari' },
    { id: 'chip_029', type: 'rivers_wide', center: [81.78, 16.78], name: 'Godavari' },
    { id: 'chip_030', type: 'rivers_wide', center: [81.83, 16.73], name: 'Godavari' },

    // NARROW RIVERS (15 chips) - EXISTING
    { id: 'chip_031', type: 'rivers_narrow', center: [77.25, 28.65], name: 'Yamuna' },
    { id: 'chip_032', type: 'rivers_narrow', center: [77.20, 28.60], name: 'Yamuna' },
    { id: 'chip_033', type: 'rivers_narrow', center: [77.30, 28.70], name: 'Yamuna' },
    { id: 'chip_034', type: 'rivers_narrow', center: [77.23, 28.68], name: 'Yamuna' },
    { id: 'chip_035', type: 'rivers_narrow', center: [77.28, 28.63], name: 'Yamuna' },
    { id: 'chip_036', type: 'rivers_narrow', center: [76.75, 22.75], name: 'Narmada' },
    { id: 'chip_037', type: 'rivers_narrow', center: [76.70, 22.70], name: 'Narmada' },
    { id: 'chip_038', type: 'rivers_narrow', center: [76.80, 22.80], name: 'Narmada' },
    { id: 'chip_039', type: 'rivers_narrow', center: [76.73, 22.78], name: 'Narmada' },
    { id: 'chip_040', type: 'rivers_narrow', center: [76.78, 22.73], name: 'Narmada' },
    { id: 'chip_041', type: 'rivers_narrow', center: [78.70, 10.90], name: 'Cauvery' },
    { id: 'chip_042', type: 'rivers_narrow', center: [78.65, 10.85], name: 'Cauvery' },
    { id: 'chip_043', type: 'rivers_narrow', center: [78.75, 10.95], name: 'Cauvery' },
    { id: 'chip_044', type: 'rivers_narrow', center: [78.68, 10.93], name: 'Cauvery' },
    { id: 'chip_045', type: 'rivers_narrow', center: [78.73, 10.88], name: 'Cauvery' },

    // WETLANDS (15 chips) - EXISTING
    { id: 'chip_046', type: 'wetlands', center: [77.52, 27.15], name: 'Keoladeo' },
    { id: 'chip_047', type: 'wetlands', center: [77.50, 27.13], name: 'Keoladeo' },
    { id: 'chip_048', type: 'wetlands', center: [77.54, 27.17], name: 'Keoladeo' },
    { id: 'chip_049', type: 'wetlands', center: [77.51, 27.16], name: 'Keoladeo' },
    { id: 'chip_050', type: 'wetlands', center: [77.53, 27.14], name: 'Keoladeo' },
    { id: 'chip_051', type: 'wetlands', center: [81.30, 16.60], name: 'Kolleru' },
    { id: 'chip_052', type: 'wetlands', center: [81.25, 16.55], name: 'Kolleru' },
    { id: 'chip_053', type: 'wetlands', center: [81.35, 16.65], name: 'Kolleru' },
    { id: 'chip_054', type: 'wetlands', center: [81.28, 16.63], name: 'Kolleru' },
    { id: 'chip_055', type: 'wetlands', center: [81.33, 16.58], name: 'Kolleru' },
    { id: 'chip_056', type: 'wetlands', center: [75.05, 31.17], name: 'Harike' },
    { id: 'chip_057', type: 'wetlands', center: [75.00, 31.12], name: 'Harike' },
    { id: 'chip_058', type: 'wetlands', center: [75.10, 31.22], name: 'Harike' },
    { id: 'chip_059', type: 'wetlands', center: [75.03, 31.20], name: 'Harike' },
    { id: 'chip_060', type: 'wetlands', center: [75.08, 31.15], name: 'Harike' },

    // RESERVOIRS (15 chips) - EXISTING
    { id: 'chip_061', type: 'reservoirs', center: [79.40, 16.60], name: 'Nagarjuna' },
    { id: 'chip_062', type: 'reservoirs', center: [79.35, 16.55], name: 'Nagarjuna' },
    { id: 'chip_063', type: 'reservoirs', center: [79.45, 16.65], name: 'Nagarjuna' },
    { id: 'chip_064', type: 'reservoirs', center: [79.38, 16.63], name: 'Nagarjuna' },
    { id: 'chip_065', type: 'reservoirs', center: [79.43, 16.58], name: 'Nagarjuna' },
    { id: 'chip_066', type: 'reservoirs', center: [83.90, 21.55], name: 'Hirakud' },
    {
        id: 'chip_067', type: 'reservoirs", center: [83.85, 21.50], name: 'Hirakud'},
  { id: 'chip_068', type: 'reservoirs', center: [83.95, 21.60], name: 'Hirakud' },
  { id: 'chip_069', type: 'reservoirs', center: [83.88, 21.58], name: 'Hirakud' },
    { id: 'chip_070', type: 'reservoirs', center: [83.93, 21.53], name: 'Hirakud' },
    { id: 'chip_071', type: 'reservoirs', center: [76.50, 31.45], name: 'Bhakra' },
    { id: 'chip_072', type: 'reservoirs', center: [76.45, 31.40], name: 'Bhakra' },
    { id: 'chip_073', type: 'reservoirs', center: [76.55, 31.50], name: 'Bhakra' },
    { id: 'chip_074', type: 'reservoirs', center: [76.48, 31.48], name: 'Bhakra' },
    { id: 'chip_075', type: 'reservoirs', center: [76.53, 31.43], name: 'Bhakra' },

    // SPARSE/ARID (15 chips) - EXISTING
    { id: 'chip_076', type: 'sparse_arid', center: [69.90, 23.87], name: 'Rann' },
    { id: 'chip_077', type: 'sparse_arid', center: [69.85, 23.82], name: 'Rann' },
    { id: 'chip_078', type: 'sparse_arid', center: [69.95, 23.92], name: 'Rann' },
    { id: 'chip_079', type: 'sparse_arid', center: [69.88, 23.90], name: 'Rann' },
    { id: 'chip_080', type: 'sparse_arid', center: [69.93, 23.85], name: 'Rann' },
    { id: 'chip_081', type: 'sparse_arid', center: [71.25, 26.45], name: 'Thar' },
    { id: 'chip_082', type: 'sparse_arid', center: [71.20, 26.40], name: 'Thar' },
    { id: 'chip_083', type: 'sparse_arid', center: [71.30, 26.50], name: 'Thar' },
    { id: 'chip_084', type: 'sparse_arid', center: [71.23, 26.48], name: 'Thar' },
    { id: 'chip_085', type: 'sparse_arid', center: [71.28, 26.43], name: 'Thar' },
    { id: 'chip_086', type: 'sparse_arid', center: [77.50, 16.80], name: 'Deccan' },
    { id: 'chip_087', type: 'sparse_arid', center: [77.45, 16.75], name: 'Deccan' },
    { id: 'chip_088', type: 'sparse_arid', center: [77.55, 16.85], name: 'Deccan' },
    { id: 'chip_089', type: 'sparse_arid', center: [77.48, 16.83], name: 'Deccan' },
    { id: 'chip_090', type: 'sparse_arid', center: [77.53, 16.78], name: 'Deccan' },

    // ⭐ COASTAL/MANGROVES (10 chips) - NEW
    { id: 'chip_091', type: 'coastal_mangroves', center: [88.85, 21.95], name: 'Sundarbans' },
    { id: 'chip_092', type: 'coastal_mangroves', center: [88.90, 21.90], name: 'Sundarbans' },
    { id: 'chip_093', type: 'coastal_mangroves', center: [86.85, 20.70], name: 'Bhitarkanika' },
    { id: 'chip_094', type: 'coastal_mangroves', center: [86.80, 20.75], name: 'Bhitarkanika' },
    { id: 'chip_095', type: 'coastal_mangroves', center: [80.23, 13.60], name: 'Pulicat' },
    { id: 'chip_096', type: 'coastal_mangroves', center: [80.20, 13.55], name: 'Pulicat' },
    { id: 'chip_097', type: 'coastal_mangroves', center: [85.50, 19.65], name: 'Chilika_Coastal' },
    { id: 'chip_098', type: 'coastal_mangroves', center: [79.78, 11.43], name: 'Pichavaram' },
    { id: 'chip_099', type: 'coastal_mangroves', center: [82.25, 16.75], name: 'Coringa' },
    { id: 'chip_100', type: 'coastal_mangroves', center: [88.80, 22.00], name: 'Sundarbans_Tidal' },

    // ⭐ BACKWATERS/CANALS (10 chips) - NEW
    { id: 'chip_101', type: 'backwaters', center: [76.35, 9.60], name: 'Vembanad' },
    { id: 'chip_102', type: 'backwaters', center: [76.30, 9.65], name: 'Vembanad' },
    { id: 'chip_103', type: 'backwaters', center: [76.32, 9.50], name: 'Alleppey' },
    { id: 'chip_104', type: 'backwaters', center: [76.43, 9.27], name: 'Kuttanad' },
    { id: 'chip_105', type: 'backwaters', center: [76.58, 8.98], name: 'Ashtamudi' },
    { id: 'chip_106', type: 'backwaters', center: [73.95, 15.50], name: 'Mandovi' },
    { id: 'chip_107', type: 'backwaters', center: [73.90, 15.40], name: 'Zuari' },
    { id: 'chip_108', type: 'backwaters', center: [76.25, 9.98], name: 'Cochin' },
    { id: 'chip_109', type: 'backwaters', center: [76.58, 8.88], name: 'Kollam' },
    { id: 'chip_110', type: 'backwaters', center: [76.22, 10.12], name: 'Periyar' },

    // ⭐ HIGH-ELEVATION/FLOODS (10 chips) - NEW
    { id: 'chip_111', type: 'high_elevation', center: [78.70, 33.75], name: 'Pangong' },
    { id: 'chip_112', type: 'high_elevation', center: [78.30, 32.90], name: 'TsoMoriri' },
    { id: 'chip_113', type: 'high_elevation', center: [91.70, 26.10], name: 'Brahmaputra_Flood' },
    { id: 'chip_114', type: 'high_elevation', center: [91.65, 26.05], name: 'Brahmaputra_Paddies' },
    { id: 'chip_115', type: 'high_elevation', center: [94.20, 26.95], name: 'Majuli' },
    { id: 'chip_116', type: 'high_elevation', center: [87.02, 25.98], name: 'Koshi' },
    { id: 'chip_117', type: 'high_elevation', center: [84.95, 26.10], name: 'Gandak' },
    { id: 'chip_118', type: 'high_elevation', center: [78.25, 26.50], name: 'Chambal' },
    { id: 'chip_119', type: 'high_elevation', center: [79.20, 21.65], name: 'Pench' },
    { id: 'chip_120', type: 'high_elevation', center: [94.90, 27.48], name: 'Dibrugarh' }
];

var config = {
    chipSize: 512,
    scale: 10,
    startDate: '2023-01-01',
    endDate: '2023-12-31'
};

// ============================================================
// DATA PROCESSING (SAME AS ORIGINAL + JRC)
// ============================================================

function getS1Composite(geometry, start, end) {
    var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(geometry)
        .filterDate(start, end)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select(['VV', 'VH']);

    return s1.map(function (img) {
        return img.focal_median(30, 'circle', 'meters');
    }).median();
}

function getJRCWaterMask(geometry) {
    var jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater');
    var occurrence = jrc.select('occurrence').clip(geometry);
    var seasonality = jrc.select('seasonality').clip(geometry);

    var permanentWater = occurrence.gt(50);
    var seasonalWater = seasonality.gte(5);
    var waterMask = permanentWater.or(seasonalWater).unmask(0).rename('JRC_Water');

    return waterMask;
}

function computeDEMBands(geometry) {
    var srtm = ee.Image('USGS/SRTMGL1_003').clip(geometry);
    var elevation = srtm.select('elevation');
    var slope = ee.Terrain.slope(srtm);

    var flowAcc = ee.Image('WWF/HydroSHEDS/15ACC').clip(geometry);
    var windowSize = 30;
    var localMin = elevation.reduceNeighborhood({
        reducer: ee.Reducer.min(),
        kernel: ee.Kernel.circle(windowSize, 'pixels')
    });
    var hand = elevation.subtract(localMin).max(0);

    var slopeRadians = slope.multiply(Math.PI).divide(180);
    var tanSlope = slopeRadians.tan().max(0.001);
    var twi = flowAcc.log().divide(tanSlope).clamp(-10, 30);

    return ee.Image.cat([
        elevation.rename('DEM'),
        hand.rename('HAND'),
        slope.rename('SLOPE'),
        twi.rename('TWI')
    ]).float();
}

// ============================================================
// EXPORT FUNCTION
// ============================================================

function exportChipBatch(startIdx, endIdx) {
    var batch = chipDefinitions.slice(startIdx, endIdx);

    print('Processing chips ' + startIdx + ' to ' + (endIdx - 1));

    for (var i = 0; i < batch.length; i++) {
        var chip = batch[i];

        var halfSize = (config.chipSize * config.scale / 2) / 111320;
        var geometry = ee.Geometry.Rectangle([
            chip.center[0] - halfSize,
            chip.center[1] - halfSize,
            chip.center[0] + halfSize,
            chip.center[1] + halfSize
        ]);

        var s1 = getS1Composite(geometry, config.startDate, config.endDate);
        var jrcWater = getJRCWaterMask(geometry);
        var demBands = computeDEMBands(geometry);

        var fullStack = s1.addBands(demBands).addBands(jrcWater).float();

        // Export 7-band stack + JRC label
        Export.image.toDrive({
            image: fullStack,
            description: chip.id + '_' + chip.type + '_features',
            folder: 'SAR_India_120_Chips',
            fileNamePrefix: chip.id + '_' + chip.type + '_7band_jrc_f32',
            region: geometry,
            scale: config.scale,
            crs: 'EPSG:4326',
            maxPixels: 1e13,
            fileFormat: 'GeoTIFF'
        });
    }
}

// ============================================================
// MAIN EXECUTION
// ============================================================

print('INDIA-FOCUSED SAR WATER CHIPS - 120 TOTAL');
print('Chips: 90 existing + 30 new (coastal/backwaters/floods)');
print('Labels: JRC Global Surface Water (< 1% error)');
print('Bands: VV, VH, DEM, HAND, SLOPE, TWI, JRC_Water');
print('');

// BATCH 1: Chips 1-30
exportChipBatch(0, 30);

// BATCH 2: Uncomment for chips 31-60
// exportChipBatch(30, 60);

// BATCH 3: Uncomment for chips 61-90
// exportChipBatch(60, 90);

// BATCH 4: Uncomment for NEW chips 91-120 ⭐
// exportChipBatch(90, 120);

print('✅ Batch queued! Go to Tasks tab and RUN');
