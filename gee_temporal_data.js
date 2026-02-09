/*
 * Google Earth Engine Script: SAR Temporal Data for Water Detection
 * ==================================================================
 * 
 * Purpose: Download multi-temporal Sentinel-1 data for wet soil discrimination.
 * 
 * Key insight from our research:
 * - 46% of false positives are wet soil (VH: -22 to -14 dB)
 * - Wet soil dries over ~1 week (VH increases 3-5 dB)
 * - Permanent water is stable or varies with wind
 * 
 * This script creates:
 * 1. Monthly composites for seasonal analysis
 * 2. Temporal statistics (mean, std, min, max)
 * 3. Change detection metrics
 * 
 * Author: SAR Water Detection Lab
 * Date: 2026-01-25
 */

// =============================================================================
// CONFIGURATION
// =============================================================================

// Define your area of interest (example: Brahmaputra region, India)
var aoi = ee.Geometry.Rectangle([89.5, 25.5, 92.5, 27.5]);

// Time range
var startDate = '2024-01-01';
var endDate = '2024-12-31';

// Export settings
var exportScale = 10;  // meters
var exportFolder = 'SAR_Water_Detection';

// =============================================================================
// FUNCTIONS
// =============================================================================

/**
 * Convert to dB scale
 */
function toDb(image) {
  return ee.Image(10).multiply(image.log10()).copyProperties(image, ['system:time_start']);
}

/**
 * Add date properties
 */
function addDate(image) {
  var date = ee.Date(image.get('system:time_start'));
  return image
    .set('year', date.get('year'))
    .set('month', date.get('month'))
    .set('day', date.get('day'))
    .set('doy', date.getRelative('day', 'year'));
}

/**
 * Compute texture (coefficient of variation)
 */
function addTexture(image) {
  var kernel = ee.Kernel.square({radius: 3, units: 'pixels'});
  var vhMean = image.select('VH').reduceNeighborhood({
    reducer: ee.Reducer.mean(),
    kernel: kernel
  }).rename('VH_mean_local');
  
  var vhStd = image.select('VH').reduceNeighborhood({
    reducer: ee.Reducer.stdDev(),
    kernel: kernel
  }).rename('VH_std_local');
  
  var texture = vhStd.divide(vhMean.abs().add(0.0001)).rename('VH_texture');
  
  return image.addBands([vhMean, vhStd, texture]);
}

/**
 * Compute VV/VH ratio
 */
function addRatio(image) {
  var ratio = image.select('VV').subtract(image.select('VH')).rename('VV_VH_ratio');
  return image.addBands(ratio);
}

// =============================================================================
// LOAD AND PROCESS SENTINEL-1 DATA
// =============================================================================

// Load Sentinel-1 GRD collection
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))  // Consistent geometry
  .select(['VV', 'VH']);

print('Total Sentinel-1 images:', s1.size());

// Convert to dB and add properties
var s1Db = s1.map(toDb).map(addDate).map(addRatio).map(addTexture);

// =============================================================================
// TEMPORAL COMPOSITES
// =============================================================================

// Monthly composites
var months = ee.List.sequence(1, 12);
var years = ee.List([2024]);

var monthlyComposites = ee.ImageCollection.fromImages(
  years.map(function(year) {
    return months.map(function(month) {
      var filtered = s1Db
        .filter(ee.Filter.calendarRange(year, year, 'year'))
        .filter(ee.Filter.calendarRange(month, month, 'month'));
      
      var composite = filtered.median()
        .set('year', year)
        .set('month', month)
        .set('system:time_start', ee.Date.fromYMD(year, month, 1).millis());
      
      return composite;
    });
  }).flatten()
);

print('Monthly composites:', monthlyComposites.size());

// =============================================================================
// TEMPORAL STATISTICS
// =============================================================================

// Full year statistics
var vhMean = s1Db.select('VH').mean().rename('VH_mean');
var vhStd = s1Db.select('VH').reduce(ee.Reducer.stdDev()).rename('VH_std');
var vhMin = s1Db.select('VH').min().rename('VH_min');
var vhMax = s1Db.select('VH').max().rename('VH_max');
var vhRange = vhMax.subtract(vhMin).rename('VH_range');

var vvMean = s1Db.select('VV').mean().rename('VV_mean');
var vvStd = s1Db.select('VV').reduce(ee.Reducer.stdDev()).rename('VV_std');

// Temporal stability (low std = stable = permanent water)
var temporalStability = ee.Image(1).divide(vhStd.add(0.1)).rename('temporal_stability');

// Combine all temporal stats
var temporalStats = vhMean
  .addBands(vhStd)
  .addBands(vhMin)
  .addBands(vhMax)
  .addBands(vhRange)
  .addBands(vvMean)
  .addBands(vvStd)
  .addBands(temporalStability);

// =============================================================================
// SEASONAL ANALYSIS
// =============================================================================

// Define seasons for India
var preMonsoon = s1Db.filter(ee.Filter.calendarRange(3, 5, 'month'));  // Mar-May
var monsoon = s1Db.filter(ee.Filter.calendarRange(6, 9, 'month'));     // Jun-Sep
var postMonsoon = s1Db.filter(ee.Filter.calendarRange(10, 11, 'month')); // Oct-Nov
var winter = s1Db.filter(ee.Filter.calendarRange(12, 2, 'month'));     // Dec-Feb

var preMonsoonVh = preMonsoon.select('VH').median().rename('VH_preMonsoon');
var monsoonVh = monsoon.select('VH').median().rename('VH_monsoon');
var postMonsoonVh = postMonsoon.select('VH').median().rename('VH_postMonsoon');
var winterVh = winter.select('VH').median().rename('VH_winter');

// Monsoon change (flooding indicator)
var monsoonChange = monsoonVh.subtract(preMonsoonVh).rename('VH_monsoon_change');

// Seasonal composite
var seasonalStats = preMonsoonVh
  .addBands(monsoonVh)
  .addBands(postMonsoonVh)
  .addBands(winterVh)
  .addBands(monsoonChange);

// =============================================================================
// WET SOIL VS WATER DISCRIMINATION
// =============================================================================

/*
 * Key discrimination metrics:
 * 1. Temporal stability: Water is stable, wet soil dries
 * 2. Minimum VH: Permanent water has consistently low VH
 * 3. Range: Wet soil has high range (dries out)
 */

// Permanent water indicator
// Low VH_min + low std = permanent water
var permanentWaterScore = vhMin.multiply(-1).subtract(20)  // Higher when VH_min is lower
  .add(temporalStability.multiply(2))
  .rename('permanent_water_score');

// Wet soil indicator  
// Moderate VH_min + high range = wet soil
var wetSoilScore = vhRange.subtract(3)  // Higher range = more likely wet soil
  .subtract(temporalStability)  // Less stable = more likely wet soil
  .rename('wet_soil_score');

// Combined discrimination layer
var waterVsWetSoil = permanentWaterScore.subtract(wetSoilScore).rename('water_vs_wetsoil');

// =============================================================================
// VISUALIZATION
// =============================================================================

// Center map
Map.centerObject(aoi, 9);

// Add layers
Map.addLayer(s1Db.select('VH').median().clip(aoi), 
  {min: -25, max: -10, palette: ['blue', 'white', 'green']}, 
  'VH Median');

Map.addLayer(temporalStats.select('VH_std').clip(aoi),
  {min: 0, max: 5, palette: ['blue', 'yellow', 'red']},
  'VH Temporal Std');

Map.addLayer(waterVsWetSoil.clip(aoi),
  {min: -5, max: 5, palette: ['brown', 'white', 'blue']},
  'Water vs Wet Soil');

// =============================================================================
// EXPORT FUNCTIONS
// =============================================================================

// Export temporal statistics
Export.image.toDrive({
  image: temporalStats.clip(aoi).toFloat(),
  description: 'SAR_Temporal_Stats',
  folder: exportFolder,
  fileNamePrefix: 'sar_temporal_stats',
  region: aoi,
  scale: exportScale,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// Export seasonal statistics
Export.image.toDrive({
  image: seasonalStats.clip(aoi).toFloat(),
  description: 'SAR_Seasonal_Stats',
  folder: exportFolder,
  fileNamePrefix: 'sar_seasonal_stats',
  region: aoi,
  scale: exportScale,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// Export water discrimination layer
Export.image.toDrive({
  image: waterVsWetSoil.clip(aoi).toFloat(),
  description: 'SAR_Water_Discrimination',
  folder: exportFolder,
  fileNamePrefix: 'sar_water_discrimination',
  region: aoi,
  scale: exportScale,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// =============================================================================
// EXPORT MONTHLY COMPOSITES (for detailed temporal analysis)
// =============================================================================

// Function to export a single monthly composite
var exportMonthly = function(image, month) {
  var monthStr = ee.String(ee.Number(month).format('%02d'));
  Export.image.toDrive({
    image: image.select(['VH', 'VV', 'VV_VH_ratio']).clip(aoi).toFloat(),
    description: ee.String('SAR_Monthly_2024_').cat(monthStr).getInfo(),
    folder: exportFolder,
    fileNamePrefix: 'sar_monthly_2024_' + month,
    region: aoi,
    scale: exportScale,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
};

// Export first 4 months as example (uncomment to export all)
// exportMonthly(monthlyComposites.filter(ee.Filter.eq('month', 1)).first(), '01');
// exportMonthly(monthlyComposites.filter(ee.Filter.eq('month', 2)).first(), '02');
// exportMonthly(monthlyComposites.filter(ee.Filter.eq('month', 3)).first(), '03');
// exportMonthly(monthlyComposites.filter(ee.Filter.eq('month', 4)).first(), '04');

// =============================================================================
// PRINT SUMMARY
// =============================================================================

print('=== SAR TEMPORAL DATA EXPORT ===');
print('AOI:', aoi);
print('Date range:', startDate, 'to', endDate);
print('Export scale:', exportScale, 'meters');
print('');
print('Exports ready:');
print('1. SAR_Temporal_Stats - Mean, Std, Min, Max, Range, Stability');
print('2. SAR_Seasonal_Stats - Pre-monsoon, Monsoon, Post-monsoon, Winter');
print('3. SAR_Water_Discrimination - Water vs Wet Soil score');
print('');
print('Click "Run" in the Tasks tab to start exports.');
