// load the generated CONUS %ISA map for visualization
// The GEE App link is: https://gers.users.earthengine.app/view/conus-isa

// var paletteISP = ['#000000',
//     '#d1d1d1', '#d2cccc', '#d2cccc', '#d2c5c5', '#d2c5c5', '#d2c1c1', '#d2c1c1', '#d2c1c1', '#d2b8b8', '#d2b8b8',
//     '#d2b8b8', '#d2b8b8', '#d2b5b5', '#d2acac', '#d2acac', '#d2acac', '#d2acac', '#d2aaaa', '#d2aaaa', '#dba8a8',
//     '#db9f9f', '#db9f9f', '#db9f9f', '#db9e9e', '#db9b9b', '#db9b9b', '#dd9999', '#dd9292', '#dd9292', '#dd9292',
//     '#df9191', '#df8585', '#df8585', '#df8585', '#df8585', '#df8282', '#df8282', '#df8282', '#df7979', '#df7979',
//     '#df7777', '#df7777', '#df7575', '#df7575', '#df6c6c', '#df6c6c', '#df6c6c', '#df6c6c', '#e86b6b', '#e86868',
//     '#e86666', '#e85f5f', '#e85f5f', '#ea5f5f', '#ea5b5b', '#ea5b5b', '#ea5252', '#ea5252', '#ea5252', '#ea5151',
//     '#eb5151', '#eb4f4f', '#eb4646', '#eb4646', '#eb4646', '#eb4444', '#eb4444', '#eb3939', '#eb3939', '#eb3939',
//     '#eb3939', '#eb3939', '#eb3838', '#eb3333', '#eb3333', '#eb3333', '#eb2c2c', '#eb2b2b', '#eb2828', '#eb82eb',
//     '#ea79eb', '#df77eb', '#df6ceb', '#dd6ceb', '#d268eb', '#d25feb', '#d15eeb', '#cc52eb', '#c552eb', '#c54feb',
//     '#c146eb', '#b844eb', '#b839eb', '#b539eb', '#ac38eb', '#ac33eb', '#aa2beb', '#9f28eb', '#9f1feb', '#9e1feb'
// ];

var paletteISP = [
    '#000004', '#010106', '#02020b', '#03030f', '#050416', '#06051a', '#090720', '#0b0924', '#0e0b2b', '#120d31',
    '#140e36', '#180f3d', '#1a1042', '#1e1149', '#21114e', '#251255', '#29115a', '#2d1161', '#331067', '#36106b',
    '#3b0f70', '#3f0f72', '#440f76', '#471078', '#4c117a', '#51127c', '#54137d', '#59157e', '#5c167f', '#601880',
    '#641a80', '#681c81', '#6b1d81', '#701f81', '#752181', '#782281', '#7c2382', '#802582', '#842681', '#882781',
    '#8c2981', '#902a81', '#942c80', '#992d80', '#9c2e7f', '#a1307e', '#a5317e', '#aa337d', '#ad347c', '#b2357b',
    '#b73779', '#ba3878', '#bf3a77', '#c23b75', '#c73d73', '#ca3e72', '#cf4070', '#d2426f', '#d6456c', '#db476a',
    '#de4968', '#e24d66', '#e44f64', '#e85362', '#ea5661', '#ed5a5f', '#ef5d5e', '#f2625d', '#f4675c', '#f56b5c',
    '#f7705c', '#f8745c', '#f9795d', '#fa7d5e', '#fb835f', '#fc8961', '#fc8c63', '#fd9266', '#fd9668', '#fd9b6b',
    '#fe9f6d', '#fea571', '#fea973', '#feae77', '#feb47b', '#feb77e', '#febd82', '#fec185', '#fec68a', '#feca8d',
    '#fecf92', '#fed395', '#fed89a', '#fddea0', '#fde2a3', '#fde7a9', '#fdebac', '#fcf0b2', '#fcf4b6', '#fcf9bb',
    '#fcfdbf'
];
var visParaISP = {min: 0, max: 100, palette: paletteISP};


// var paletteNumChange = ['#000000','#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
// var visParaNumChange = {min: 0, max: 5, palette: paletteNumChange};
var paletteNumChange = ['#000000','#440154', '#21918c', '#fde725']
var visParaNumChange = {min: 0, max: 3, palette: paletteNumChange};


var paletteCombinedISChange =  ['#000000',
    '#6ca966', '#b3afa4', '#fc8a6a', '#fc8464', '#fb7d5d', '#fb7757', '#fb7151', '#fb6b4b', '#f96346', '#f75c41',
    '#f6553c', '#f44f39', '#f24734', '#f0402f', '#ed392b', '#e83429', '#e22e27', '#dc2924', '#d72322', '#d11e1f',
    '#cb181d', '#c5171c', '#bf151b', '#b91419', '#b31218', '#ad1117', '#a81016', '#9f0e14', '#960b13', '#8c0912',
    '#820711', '#79040f', '#6f020e', '#67000d', '#b6b6d8', '#b1b1d5', '#adabd2', '#a8a6cf', '#a3a0cb', '#9e9bc8',
    '#9a96c6', '#9591c4', '#908dc2', '#8d89c0', '#8885be', '#8380bb', '#7f7bb9', '#7b74b5', '#786db2', '#7566ae',
    '#715faa', '#6e58a7', '#6a51a3', '#674ba0', '#63449d', '#603e9a', '#5c3797', '#593093', '#552a90', '#52238d',
    '#4f1d8b', '#4c1788', '#481185', '#450b82', '#42057f', '#3f007d', '#bd784c', '#b9754b', '#b47249', '#b17047',
    '#ad6d46', '#a96b44', '#a46842', '#a16641', '#9d633f', '#98603d', '#945e3c', '#915b3a', '#8c5838', '#885637',
    '#845435', '#805134', '#7c4e32', '#784c30', '#74492f', '#6f462d', '#6b442b', '#68422a', '#633e28', '#5f3c26',
    '#5b3a25', '#583723', '#533421', '#4f3220', '#4b301e', '#462d1c', '#432a1b', '#3f2819', '#94c4df', '#8cc0dd',
    '#84bcdb', '#7cb7da', '#74b3d8', '#6caed6', '#65aad4', '#5fa6d1', '#58a1cf', '#539ecd', '#4d99ca', '#4695c8',
    '#4090c5', '#3b8bc2', '#3686c0', '#3181bd', '#2c7cba', '#2676b8', '#2171b5', '#1d6cb1', '#1967ad', '#1562a9',
    '#125da6', '#0e58a2', '#0a539e', '#084e98', '#084990', '#084488', '#083e81', '#083979', '#083471', '#08306b',
    '#fed778', '#fed36f', '#fece65', '#feca5d', '#fec754', '#fec24d', '#febd49', '#feb744', '#feb23f', '#fead3a',
    '#fea634', '#fea030', '#fe9b2b', '#fd9627', '#fa9125', '#f88b22', '#f6861f', '#f4811d', '#f17b1a', '#ee7617',
    '#ec7014', '#e86c12', '#e46710', '#e0630d', '#dc5e0b', '#d75908', '#d35406', '#cf5004', '#cb4b02', '#c44802',
    '#be4503', '#b84203'
];
var visParaCombinedISChange = {min: 0, max: 162, palette: paletteCombinedISChange};


var CONUSISP = {
    'Google Earth Image': getCONUSISPMap('2022'), // 'Google Earth Image

    // 'CONUS-ISP 1985': getCONUSISPMap('1985'),
    'CONUS-ISP 1988': getCONUSISPMap('1988').visualize(visParaISP),
    'CONUS-ISP 1990': getCONUSISPMap('1990').visualize(visParaISP),
    'CONUS-ISP 1995': getCONUSISPMap('1995').visualize(visParaISP),
    'CONUS-ISP 2000': getCONUSISPMap('2000').visualize(visParaISP),
    'CONUS-ISP 2005': getCONUSISPMap('2005').visualize(visParaISP),
    'CONUS-ISP 2010': getCONUSISPMap('2010').visualize(visParaISP),
    'CONUS-ISP 2015': getCONUSISPMap('2015').visualize(visParaISP),
    'CONUS-ISP 2020': getCONUSISPMap('2020').visualize(visParaISP),

    'CONUS Accumulated IS Change': getCONUSCombinedAccumulateISChangeMap('without_sm_sample').visualize(visParaCombinedISChange),
    'CONUS Number of IS changes': getCONUSAccumulateISChangeMap('num_changes_without_sm').visualize(visParaNumChange),

    // 'AnnualNLCD-ISP 1985': getAnnualNLCDISPMap(1985).visualize(visParaISP),
    'AnnualNLCD-ISP 1988': getAnnualNLCDISPMap(1988).visualize(visParaISP),
    'AnnualNLCD-ISP 1990': getAnnualNLCDISPMap(1990).visualize(visParaISP),
    // 'AnnualNLCD-ISP 1995': getAnnualNLCDISPMap(1995).visualize(visParaISP),
    'AnnualNLCD-ISP 2000': getAnnualNLCDISPMap(2000).visualize(visParaISP),
    // 'AnnualNLCD-ISP 2005': getAnnualNLCDISPMap(2005).visualize(visParaISP),
    'AnnualNLCD-ISP 2010': getAnnualNLCDISPMap(2010).visualize(visParaISP),
    // 'AnnualNLCD-ISP 2015': getAnnualNLCDISPMap(2015).visualize(visParaISP),
    'AnnualNLCD-ISP 2020': getAnnualNLCDISPMap(2020).visualize(visParaISP),
    // 'AnnualNLCD-ISP 2022': getAnnualNLCDISPMap(2022).visualize(visParaISP),
};


var leftMap = ui.Map();
leftMap.setControlVisibility(false);
var leftTitle = ui.Label('Left Map Title', {fontWeight: 'bold', fontSize: '24px',position: 'top-left'});
leftMap.add(leftTitle);

addISPSelector(leftMap, 8, 'top-left', leftTitle);   // set the default display year as 2022

var rightMap = ui.Map();
rightMap.setControlVisibility(false);
var rightTitle = ui.Label('Right Map Title', {fontWeight: 'bold', fontSize: '24px',position: 'top-right'});
rightMap.add(rightTitle);

addISPSelector(rightMap, 0, 'top-right', rightTitle);


var splitPanelMap = ui.Panel(ui.SplitPanel({
  firstPanel: leftMap,
  secondPanel: rightMap,
  wipe: true,
  style: {stretch: 'both'}
}));

var toolPanel = ui.Panel({
    style: {width: '400px'}
});

var splitPanel = ui.SplitPanel(splitPanelMap, toolPanel);

ui.root.clear();
ui.root.add(splitPanel);

// Set the SplitPanel as the only thing in the UI root.

// Link the two maps.
var linker = ui.Map.Linker([leftMap, rightMap]);

leftMap.setCenter(-95, 38, 5); //default location
leftMap.setOptions('SATELLITE'); //Set satellite as the base layer
rightMap.setCenter(-95, 38, 5);
rightMap.setOptions('SATELLITE');

// add right-side panels
// var verticalFlow = ui.Panel.Layout.flow('vertical');


var title = ui.Panel([
    ui.Label({
        value: 'CONUS Percent Impervious Surface Area (%ISA) Map',
        style: {fontSize: '20px', color: 'Green', fontWeight: 'bold'}
    }),
    ui.Label({
        value: "A 30-m CONUS %ISA map (1988-2020) produced by the GERS Lab (beta version), with selected USGS Annual NLCD %ISA maps used for comparison.",
        style: {fontSize: '14px', color: 'Black', fontWeight: 'normal'}
    }),
]);

toolPanel.add(title);


var createLegendISP = function () {

    // Create the legend image for IS percentage
    var gradient = ee.Image.pixelLonLat()
        .select('latitude').int()
        .visualize({min: 0, max: 100, palette: paletteISP});

    var thumbnail = ui.Thumbnail({
        image: gradient,
        params: {bbox: '0,0,10,100', dimensions: '20x160', format: 'png'},
        style: {
            stretch: 'vertical',
            margin: '2px 2px 4px 10px',
            height: '160px',
            width: '20px'
        },
    });

    var labelStyle = {margin: '0px 0px 14px 2px', textAlign: 'center', fontSize: '14px'}

    // Create legend labels for IS percentage
    var legendLabels = ui.Panel({
        widgets: [
            ui.Label('100', labelStyle),
            ui.Label('80', labelStyle),
            ui.Label('60', labelStyle),
            ui.Label('40', labelStyle),
            ui.Label('20', labelStyle),
            ui.Label('0', labelStyle),
        ],
        layout: ui.Panel.Layout.flow('vertical'),
        style: {stretch: 'horizontal'}
    });

    // Define the legend panel for IS percentage
    var legendPanelISP = ui.Panel([thumbnail, legendLabels],
        ui.Panel.Layout.flow('horizontal'), {
            margin: '0px 0px',
        })

    return legendPanelISP;
}


// Define the legend title for IS percentage
var legendISPTitle = ui.Label(
    'Percent Impervious Surface Area (%ISA)',
    {fontWeight: 'bold', fontSize: '14px',});

// add the legend title to the tool panel
toolPanel.add(legendISPTitle);

// Add the legend panel to the tool panel
var legendPanelISP = createLegendISP();
toolPanel.add(legendPanelISP);


function makeColorbar(cmap, title, max_tick) {

    // set position of panel
    var legend_panel = ui.Panel({
        style: {
            padding: '4px 15px'
        }
    });

    // Create legend title
    var legendTitle = ui.Label({
        value: title,
        style: {
            // fontWeight: 'bold',
            fontSize: '14px',
            margin: '0 0 4px 0',
            padding: '0'
        }
    });
    legend_panel.add(legendTitle)

    // var lat = ee.Image.pixelLonLat().select('latitude');
    // var gradient = lat.multiply((cmap.max - cmap.min) / 100.0).add(cmap.min);
    // var legendImage = gradient.visualize(cmap);

    var gradient = ee.Image.pixelLonLat().select('latitude').visualize(cmap);

    var thumb = ui.Thumbnail({
        image: gradient,
        params: {
            bbox: '0,0,8,50',
            dimensions: '20x130',
            format: 'png'
        },
        style: {
            position: 'bottom-center',
            stretch: 'vertical',
        }
    });

    var tick_labels = ui.Panel({
        widgets: [
            ui.Label(max_tick),
            ui.Label({
                style: {
                    stretch: 'vertical'
                }
            }),
            ui.Label('1989')
        ],
        layout: ui.Panel.Layout.flow('vertical'),
        style: {
            stretch: 'vertical',
            maxHeight: '130',
            padding: '0px 0px 0px 0px'
        }
    });
    var colorbar_panel = ui.Panel({
        widgets: [thumb, tick_labels],
        layout: ui.Panel.Layout.flow('horizontal')
    });
    return legend_panel.add(colorbar_panel);
}


var visParaExpansion = {
    min: 1.0,
    max: 48.0,
    palette: ['#fc8a6a', '#fc8464', '#fb7d5d', '#fb7757', '#fb7151', '#fb6b4b', '#f96346', '#f75c41', '#f6553c', '#f44f39',
        '#f24734', '#f0402f', '#ed392b', '#e83429', '#e22e27', '#dc2924', '#d72322', '#d11e1f', '#cb181d', '#c5171c',
        '#bf151b', '#b91419', '#b31218', '#ad1117', '#a81016', '#9f0e14', '#960b13', '#8c0912', '#820711', '#79040f',
        '#6f020e', '#67000d'],
};


var visParaIntensification = {
    min: 1.0,
    max: 48.0,
    palette: ['#b6b6d8', '#b1b1d5', '#adabd2', '#a8a6cf', '#a3a0cb', '#9e9bc8', '#9a96c6', '#9591c4', '#908dc2', '#8d89c0',
        '#8885be', '#8380bb', '#7f7bb9', '#7b74b5', '#786db2', '#7566ae', '#715faa', '#6e58a7', '#6a51a3', '#674ba0',
        '#63449d', '#603e9a', '#5c3797', '#593093', '#552a90', '#52238d', '#4f1d8b', '#4c1788', '#481185', '#450b82',
        '#42057f', '#3f007d'],
};

var visParaDecline = {
    min: 1.0,
    max: 48.0,
    palette: ['#bd784c', '#b9754b', '#b47249', '#b17047', '#ad6d46', '#a96b44', '#a46842', '#a16641', '#9d633f', '#98603d',
    '#945e3c', '#915b3a', '#8c5838', '#885637', '#845435', '#805134', '#7c4e32', '#784c30', '#74492f', '#6f462d',
    '#6b442b', '#68422a', '#633e28', '#5f3c26', '#5b3a25', '#583723', '#533421', '#4f3220', '#4b301e', '#462d1c',
    '#432a1b', '#3f2819'],
};

var visParaReversal = {
    min: 1.0,
    max: 48.0,
    palette: ['#94c4df', '#8cc0dd', '#84bcdb', '#7cb7da', '#74b3d8', '#6caed6', '#65aad4', '#5fa6d1', '#58a1cf', '#539ecd',
    '#4d99ca', '#4695c8', '#4090c5', '#3b8bc2', '#3686c0', '#3181bd', '#2c7cba', '#2676b8', '#2171b5', '#1d6cb1',
    '#1967ad', '#1562a9', '#125da6', '#0e58a2', '#0a539e', '#084e98', '#084990', '#084488', '#083e81', '#083979',
    '#083471', '#08306b'],
};


var cbarExpansion = makeColorbar(visParaExpansion, 'IS Expansion', '2020')
var cbarIntensification = makeColorbar(visParaIntensification, 'IS Intensification', '2020')
var cbarDecline = makeColorbar(visParaDecline, 'IS Decline', '2020')
var cbarReversal = makeColorbar(visParaReversal, 'IS Reversal', '2020')


var legends_panel = ui.Panel({
    widgets: [cbarExpansion, cbarIntensification, cbarDecline, cbarReversal],
    layout: ui.Panel.Layout.flow('horizontal'),
});


var createLegendCombinedAccumulateISChange = function () {

    var legend = ui.Panel();

    // Create color blocks for legend
    var makeRow = function (year, color) {
        var colorBox = ui.Label({
            style: {
                backgroundColor: color,
                padding: '8px',
                margin: '0px 0px 0px 0px'
            }
        });

        var description = ui.Label({
            value: year.toString(),
            style: {margin: '0px 0px 4px 6px', fontSize: '14px'}
        });

        return ui.Panel({
            widgets: [colorBox, description],
            layout: ui.Panel.Layout.Flow('horizontal'),
            style: {margin: '0px 0px 0px 8px'}
        });
    };

    /*
    1 - Stable natural
    2 - Stable IS
     */

    // Add color blocks to the legend
    var legendLabels = ['Stable natural', 'Stable IS',];
    var legendColors = ['#6ca966', '#b3afa4', ];

    for (var i = 0; i < legendLabels.length; i++) {
        legend.add(makeRow(legendLabels[i], legendColors[i]));
    }

    return legend;
}


// Define the legend title for IS percentage
var legendCombinedISChangeTitle = ui.Label(
    'Accumulated Recent Impervious Surface Change',
    {fontWeight: 'bold', fontSize: '14px',});

// add the legend title to the tool panel
toolPanel.add(legendCombinedISChangeTitle);

// Add the legend panel to the tool panel
var legendPanelCombinedAccumulateISChange = createLegendCombinedAccumulateISChange();
toolPanel.add(legendPanelCombinedAccumulateISChange);

toolPanel.add(legends_panel);


// Create the legend for IS change times
var createLegendISChangeTimes = function () {

    var legend = ui.Panel();

    // Create color blocks for legend
    var makeRow = function (year, color) {
        var colorBox = ui.Label({
            style: {
                backgroundColor: color,
                padding: '8px',
                margin: '0px 0px 0px 0px'
            }
        });

        var description = ui.Label({
            value: year.toString(),
            style: {margin: '0px 0px 4px 6px', fontSize: '14px'}
        });

        return ui.Panel({
            widgets: [colorBox, description],
            layout: ui.Panel.Layout.Flow('horizontal'),
            style: {margin: '0px 0px 0px 8px'}
        });
    };

    // Add color blocks to the legend
    // var legendLabels = ['1', '2', '3', '4', '>=5'];
    // var legendColors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'];
    var legendLabels = ['1', '2', '>=3'];
    var legendColors = ['#440154', '#21918c', '#fde725'];

    for (var i = 0; i < legendLabels.length; i++) {
        legend.add(makeRow(legendLabels[i], legendColors[i]));
    }

    return legend;
};


// Define the legend title for IS change Times
var legendISChangeTimesTitle = ui.Label(
    'Number of Impervious surface changes',
    {fontWeight: 'bold', fontSize: '14px',});

toolPanel.add(legendISChangeTimesTitle);

var legendNumISChange = createLegendISChangeTimes();
toolPanel.add(legendNumISChange);



// add the source code panel
var GEEISATimeSeriesHeader = ui.Label('GEE App for time series viewing', {fontSize: '12px', fontWeight: 'bold'});
var GEEISATimeSeriesText = ui.Label(
    'GEE App: CONUS Subpixel %ISA Time Series View',
    {fontSize: '12px'});
GEEISATimeSeriesText.setUrl('https://faluhong.users.earthengine.app/view/conus-isp-time-series');
var GEEISATimeSeriesPanel = ui.Panel([GEEISATimeSeriesHeader, GEEISATimeSeriesText],
    'flow', {fontWeight: 'bold', fontSize: '12px', width: '300px'});
toolPanel.add(GEEISATimeSeriesPanel);

// add the source code panel
var sourceCodeHeader = ui.Label('Source code', {fontSize: '12px', fontWeight: 'bold'});
var sourceCodeText = ui.Label(
    'GitHub: CONUS Subpixel %ISA mapping',
    {fontSize: '12px'});
sourceCodeText.setUrl('https://github.com/faluhong/subpixel-isa-us');
var sourceCodePanel = ui.Panel([sourceCodeHeader, sourceCodeText], 'flow', {fontWeight: 'bold', fontSize: '12px', width: '300px'});
toolPanel.add(sourceCodePanel);


// Adds a layer selection widget to the given year, to allow users to display the land cover on the given year in the associated map
function addISPSelector(mapToChange, defaultValue, position, title) {
  var label = ui.Label('ISP', {fontWeight: 'bold', fontSize: '12px'});

  function updateMap(selection) {

      // print((ee.String(selection).contains('CONUS-ISP')));
      if (ee.String(selection).compareTo('Google Earth Image').getInfo()===0){
          // Display the default Google Earth Image
          mapToChange.layers().set(0);
          title.setValue('Google Earth Image');
          }
      else{
          var ispSelection = CONUSISP[selection];
          mapToChange.layers().set(0, ui.Map.Layer(ispSelection));
          title.setValue(selection);
      }
  }

  var select = ui.Select({items: Object.keys(CONUSISP), onChange: updateMap});
  select.setValue(Object.keys(CONUSISP)[defaultValue], true);

  var controlPanel = ui.Panel({widgets: [label, select], style: {position: position}});
  mapToChange.add(controlPanel);
}


function getCONUSISPMap(year){
    // var fileNameCONUSISP = 'projects/ee-faluhong/assets/conus_isp/conus_isp_filter_'+ year;
    var fileNameCONUSISP = 'projects/ee-faluhong/assets/conus_isp_binary_is_ndvi015/conus_isp_post_processing_binary_is_ndvi015_sm_' + year;

    var conusISP = ee.Image(fileNameCONUSISP);
    conusISP = conusISP.updateMask(conusISP.neq(255));  // mask out the no data value 255

    return conusISP;
}


function getCONUSAccumulateISChangeMap(accumulate_flag){
    // get the CONUS accumulate IS change map
    // parameter accumulate_flag: string, 'recent_with_sm', 'first_with_sm', 'recent_without_sm', 'first_without_sm',
    //                                    'first_year_with_sm', 'recent_year_with_sm', 'first_year_without_sm', 'recent_year_without_sm'
    //                                    'num_changes_with_sm' 'num_changes_without_sm'

    var fileNameCONUSAccumulateISChange = ('projects/ee-faluhong/assets/conus_isp_binary_is_ndvi015/' +
        'conus_accumulate_is_change_1988_2020_post_processing_binary_is_ndvi015_sm_' + accumulate_flag);

    var conusAccumulateISChangeMap = ee.Image(fileNameCONUSAccumulateISChange);
    conusAccumulateISChangeMap = conusAccumulateISChangeMap.updateMask(conusAccumulateISChangeMap.neq(255));  // mask out the no data value 255

    return conusAccumulateISChangeMap;
}


function getCONUSCombinedAccumulateISChangeMap(combined_accumulate_flag){
    // get the CONUS accumulate IS change map
    // parameter accumulate_flag: string, 'with_sm', 'without_sm'

    var fileNameCONUSCombinedAccumulateISChange = ('projects/ee-faluhong/assets/conus_isp_binary_is_ndvi015/' +
        'conus_combine_binary_is_ndvi015_sm_recent_is_change_type_year_' + combined_accumulate_flag);

    var conusCombinedAccumulateISChangeMap = ee.Image(fileNameCONUSCombinedAccumulateISChange);
    conusCombinedAccumulateISChangeMap = conusCombinedAccumulateISChangeMap.updateMask(conusCombinedAccumulateISChangeMap.neq(0));  // mask out the no data value 0
    conusCombinedAccumulateISChangeMap = conusCombinedAccumulateISChangeMap.updateMask(conusCombinedAccumulateISChangeMap.neq(255));  // mask out the no data value 255

    return conusCombinedAccumulateISChangeMap;
}


function getAnnualNLCDISPMap(year){
    // parameter year: int, the year of the annual NLCD data

    var annualNLCDISP = ee.ImageCollection("projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/FRACTIONAL_IMPERVIOUS_SURFACE");
    annualNLCDISP = annualNLCDISP.filter(ee.Filter.eq('year', year));
    annualNLCDISP = annualNLCDISP.first();
    annualNLCDISP = annualNLCDISP.updateMask(annualNLCDISP.neq(250));

    return annualNLCDISP;
}





